import os
import argparse
import numpy as np
import torch
import faiss
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

# Run from MLLM-MSR/ root.
# Usage:
#   python path_retriever/clip_embedder.py \
#       --dataset microlens \
#       --image_dir data/microlens/MicroLens-50k_covers \
#       --title_file data/microlens/processed/MicroLens-50k_titles.csv \
#       --title_format csv
#
#   python path_retriever/clip_embedder.py \
#       --dataset Baby_Products \
#       --image_dir data/amazon/Amazon_images/Baby_Products \
#       --title_file data/amazon/processed/Baby_Products_item_desc.tsv \
#       --title_format tsv

parser = argparse.ArgumentParser(description='Build CLIP embeddings + FAISS index for items')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset name (microlens | Baby_Products | Video_Games)')
parser.add_argument('--image_dir', type=str, required=True,
                    help='Directory containing item cover images (files named by item ID)')
parser.add_argument('--title_file', type=str, required=True,
                    help='Path to item titles file')
parser.add_argument('--title_format', type=str, required=True, choices=['csv', 'tsv'],
                    help='csv for MicroLens (no header, cols: item,title); tsv for Amazon (cols: item_id,summary)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: path_retriever/saved_explanations/clip_{dataset})')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14')
args = parser.parse_args()

output_dir = args.output_dir or f'path_retriever/saved_explanations/clip_{args.dataset}'
os.makedirs(output_dir, exist_ok=True)

# Load titles
print(f"Loading titles from {args.title_file}")
if args.title_format == 'csv':
    title_df = pd.read_csv(args.title_file, header=None, names=['item', 'title'])
    title_df['item'] = title_df['item'].astype(str)
    title_df['title'] = title_df['title'].fillna('')
else:
    title_df = pd.read_csv(args.title_file, sep='\t', header=None, usecols=[0, 2], names=['item', 'summary'])
    title_df = title_df.rename(columns={'item_id': 'item', 'summary': 'title'})
    title_df['item'] = title_df['item'].astype(str)
    title_df['title'] = title_df['title'].fillna('')
title_map = title_df.set_index('item')['title'].to_dict()
print(f"  {len(title_map)} item titles loaded")

# Enumerate images
print(f"Enumerating images from {args.image_dir}")
image_dir = Path(args.image_dir)
image_files = [(p, p.stem) for p in image_dir.glob('*') if p.is_file()]
print(f"  {len(image_files)} image files found")

# Load CLIP
print(f"Loading CLIP model: {args.model_name}")
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
model = CLIPModel.from_pretrained(args.model_name).to(device)
model.eval()
processor = CLIPProcessor.from_pretrained(args.model_name)
print(f"  Using device: {device}")

# Compute embeddings
all_embeddings = []
all_item_ids = []
skipped = 0

for batch_start in tqdm(range(0, len(image_files), args.batch_size), desc="Embedding items"):
    batch = image_files[batch_start: batch_start + args.batch_size]
    images, texts, item_ids = [], [], []

    for img_path, stem in batch:
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            skipped += 1
            continue
        images.append(img)
        texts.append(title_map.get(stem, ''))
        item_ids.append(int(stem))

    if not images:
        continue

    with torch.no_grad():
        inputs = processor(text=texts, images=images, return_tensors='pt',
                           padding=True, truncation=True, max_length=77).to(device)
        img_feats = model.get_image_features(pixel_values=inputs['pixel_values'])
        txt_feats = model.get_text_features(input_ids=inputs['input_ids'],
                                            attention_mask=inputs['attention_mask'])

    if not isinstance(img_feats, torch.Tensor):
        img_feats = img_feats.pooler_output
    if not isinstance(txt_feats, torch.Tensor):
        txt_feats = txt_feats.pooler_output
    img_feats = img_feats.cpu().float().numpy()
    txt_feats = txt_feats.cpu().float().numpy()

    fused = (img_feats + txt_feats) / 2.0
    # L2-normalise each row so IndexFlatIP == cosine similarity
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    fused = fused / norms

    all_embeddings.append(fused)
    all_item_ids.extend(item_ids)

if skipped:
    print(f"[Warning] Skipped {skipped} items due to image load errors")

embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
item_ids_arr = np.array(all_item_ids, dtype=np.int64)
print(f"\nEmbedding matrix: {embeddings.shape}")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

# Save
emb_path = os.path.join(output_dir, 'embeddings.npy')
ids_path = os.path.join(output_dir, 'item_ids.npy')
idx_path = os.path.join(output_dir, 'faiss.index')

np.save(emb_path, embeddings)
np.save(ids_path, item_ids_arr)
faiss.write_index(index, idx_path)

print(f"\nSaved:")
print(f"  {emb_path}")
print(f"  {ids_path}")
print(f"  {idx_path}")
