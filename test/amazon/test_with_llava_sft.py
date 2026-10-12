import torch
import argparse
from multiprocess import set_start_method
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from PIL import ImageOps
from torch.nn.functional import softmax
from torch.cuda.amp import autocast
import os
import csv
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, choices=['Baby_Products', 'Video_Games', 'Musical_Instruments', 'Sports_and_Outdoors'],
                    help='Amazon dataset category.')
parser.add_argument('--rag', action='store_true',
                    help='Load RAG test dataset and RAG-trained model.')
parser.add_argument('--variant', default='', choices=[
    '',
    'rag',
    'rag-no_mm',
    'rag-no_bfs',
    'rag-no_path_hist',
    'rag-no_cf_text',
    'baseline',
], help='Named ablation variant suffix to evaluate.')
parser.add_argument('--k', type=int, default=None,
                    help='Retrieval-K sensitivity: evaluate Amazon-{dataset}-test-rag-k{K} with the matching per-K model (K=5 maps to -rag).')
args = parser.parse_args()
DATASET = args.dataset
if args.k is not None:
    _suffix = "-rag" if args.k == 5 else f"-rag-k{args.k}"
else:
    _suffix = f"-{args.variant}" if args.variant else ("-rag" if args.rag else "")

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

HF_CACHE_DIR = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
BASE_SAVE_DIR = os.environ.get('MODEL_SAVE_DIR', os.path.join(os.path.dirname(HF_CACHE_DIR), 'LLaVA'))

EPOCH = 4
LORA_R = 16
base_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
peft_model_id = os.path.join(BASE_SAVE_DIR, f"llava-v1.6-mistral-7b-hf-lora-{DATASET}{_suffix}-e{EPOCH}-r{LORA_R}")
config = PeftConfig.from_pretrained(peft_model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(base_model_id,
                                                          cache_dir=HF_CACHE_DIR,
                                                        #   attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.float16,
                                                          )

processor = LlavaNextProcessor.from_pretrained(base_model_id, cache_dir=HF_CACHE_DIR)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = PeftModel.from_pretrained(model, peft_model_id).eval()
model.tie_weights()
print(f"PEFT model loaded")

dataset = load_from_disk(f"Amazon-{DATASET}-test{_suffix}")
print(dataset)

processor.tokenizer.add_tokens(
    ["<|image|>", "<pad>"], special_tokens=True
)

Yes_id, No_id = processor.tokenizer.convert_tokens_to_ids('Yes'), processor.tokenizer.convert_tokens_to_ids('No')
yes_id, no_id = processor.tokenizer.convert_tokens_to_ids('yes'), processor.tokenizer.convert_tokens_to_ids('no')

def gpu_computation(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)
    yes_logits_batch, no_logits_batch = [], []

    batch_images = batch['image']

    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)

    padded_images = []
    for img in batch_images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height
            padding = (
                delta_width // 2, delta_height // 2,
                delta_width - (delta_width // 2), delta_height - (delta_height // 2))
            new_img = ImageOps.expand(img.convert('RGB'), border=padding, fill='black')
            padded_images.append(new_img)

    batch['image'] = padded_images

    batch_size = len(batch['image'])
    model_inputs = processor(text=batch['prompt'], images=batch['image'], return_tensors="pt", padding=True).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        outputs = model.generate(**model_inputs, max_new_tokens=30, return_dict_in_generate=True, output_scores=True)

    scores = outputs['scores'][0]
    yes_probs = scores[:, Yes_id]
    no_probs = scores[:, No_id]

    sequences = outputs['sequences']
    pred_token = sequences[:, -3:]
    batch['output'] = processor.batch_decode(pred_token, skip_special_tokens=True)
    print(batch['output'])

    for i in range(batch_size):
        yes_logits_batch.append(yes_probs[i].item())
        no_logits_batch.append(no_probs[i].item())

    return {"yes_logits": yes_logits_batch, "no_logits": no_logits_batch}


def recall_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    retrieved_positives = np.sum(sorted_labels[:, :k], axis=1)
    total_positives = np.ones_like(retrieved_positives)
    recall_scores = retrieved_positives / total_positives
    return np.mean(recall_scores)


def mrr_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    reciprocal_ranks = np.zeros(y_true.shape[0])

    for i, labels in enumerate(sorted_labels[:, :k]):
        first_pos = np.where(labels == 1)[0]
        if first_pos.size > 0:
            reciprocal_ranks[i] = 1 / (first_pos[0] + 1)

    return np.mean(reciprocal_ranks)


def ndcg_at_k(y_true, y_prob, k):
    def dcg_at_k(scores, k):
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum((2 ** scores - 1) / discounts, axis=1)
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_scores = np.take_along_axis(y_true, sorted_indices, axis=1)[:, :k]
    dcg_scores = dcg_at_k(sorted_scores, k)
    ideal_sorted_scores = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg_scores = dcg_at_k(ideal_sorted_scores, k)
    epsilon = 1e-10
    ndcg_scores = dcg_scores / (idcg_scores + epsilon)
    return np.mean(ndcg_scores)

if __name__ == "__main__":
    set_start_method("spawn")
    torch.cuda.empty_cache()
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=6,
        with_rank=True,
        num_proc=torch.cuda.device_count(),
        # num_proc=6
    )
    updated_dataset = updated_dataset.sort("user")
    yes_logits = torch.tensor(updated_dataset['yes_logits'])
    no_logits = torch.tensor(updated_dataset['no_logits'])
    labels = np.array(updated_dataset['label'])
    yes_prob = torch.stack([no_logits, yes_logits], dim=1)
    yes_probs = F.softmax(yes_prob, dim=1)[:, 1].cpu().numpy()

    yes_probs = yes_probs.reshape(-1, 21)
    labels = labels.reshape(-1, 21)

    y_preds = np.argmax(yes_probs, axis=1)

    users = updated_dataset['user']
    unique_users = [users[i * 21] for i in range(len(users) // 21)]
    os.makedirs("predictions", exist_ok=True)
    with open(f"predictions/user_predictions_sft_{DATASET}{_suffix}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "pred_scores", "true_labels", "top1_pred_idx"])
        for i, user in enumerate(unique_users):
            writer.writerow([
                user,
                yes_probs[i].tolist(),
                labels[i].tolist(),
                int(y_preds[i]),
            ])
    print(f"Per-user predictions saved to predictions/user_predictions_sft_{DATASET}{_suffix}.csv")

    metrics = {
        "AUC": roc_auc_score(labels.flatten(), yes_probs.flatten()),
        "Recall@3": recall_at_k(labels, yes_probs, 3),
        "Recall@5": recall_at_k(labels, yes_probs, 5),
        "Recall@10": recall_at_k(labels, yes_probs, 10),
        "MRR@3": mrr_at_k(labels, yes_probs, 3),
        "MRR@5": mrr_at_k(labels, yes_probs, 5),
        "MRR@10": mrr_at_k(labels, yes_probs, 10),
        "NDCG@3": ndcg_at_k(labels, yes_probs, 3),
        "NDCG@5": ndcg_at_k(labels, yes_probs, 5),
        "NDCG@10": ndcg_at_k(labels, yes_probs, 10),
    }
    for name, value in metrics.items():
        print(f"{name}: {value}")
    with open(f"predictions/metrics_sft_{DATASET}{_suffix}.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")
    print(f"Metrics saved to predictions/metrics_sft_{DATASET}{_suffix}.txt")
