from datasets import Dataset, Image, DatasetDict
from pathlib import Path
import pandas as pd
import argparse
import pickle
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, choices=['Baby_Products', 'Video_Games', 'Musical_Instruments', 'Sports_and_Outdoors'],
                    help='Amazon dataset category.')
parser.add_argument('--no_bfs', action='store_true', help='Exclude BFS co-occurrence paths.')
parser.add_argument('--no_mm', action='store_true', help='Exclude MM similarity-based paths.')
parser.add_argument('--history_mode', choices=['path', 'baseline'], default='path',
                    help='Use path-guided full-history selection or baseline top-k history.')
parser.add_argument('--hide_cf_signals', action='store_true',
                    help='Use retrieved paths for history selection but omit CF signal text from prompts.')
parser.add_argument('--k', type=int, default=5,
                    help='Top-K for CF signal lines and selected history items (sensitivity sweep).')
args = parser.parse_args()
DATASET = args.dataset

def _variant_suffix(no_bfs, no_mm, history_mode, hide_cf_signals):
    if history_mode == 'baseline' and no_bfs and no_mm:
        return "-baseline"
    if history_mode == 'baseline':
        return "-rag-no_path_hist"
    if hide_cf_signals:
        return "-rag-no_cf_text"
    if no_bfs:
        return "-rag-no_bfs"
    if no_mm:
        return "-rag-no_mm"
    return "-rag"

VARIANT = _variant_suffix(args.no_bfs, args.no_mm, args.history_mode, args.hide_cf_signals)
if args.k != 5:
    assert VARIANT == "-rag", "--k sweep is only supported for the full RAG variant"
    VARIANT = f"-rag-k{args.k}"


def get_image_df(folder_path):
    folder_path = Path(folder_path)
    full_paths, file_names = [], []
    for file_path in folder_path.glob('*'):
        if file_path.is_file():
            full_paths.append(str(file_path.absolute()))
            file_names.append(file_path.stem)
    return pd.DataFrame({"image": full_paths, "item": file_names})


train_pair_file_path = f"../../data/amazon/Split/{DATASET}/train_pairs.csv"
df_train = pd.read_csv(train_pair_file_path)
df_train['item'] = df_train['item'].astype(str)
df_train['user'] = df_train['user'].astype(str)

val_pair_file_path = f"../../data/amazon/Split/{DATASET}/val_pairs.csv"
df_val = pd.read_csv(val_pair_file_path)
df_val['item'] = df_val['item'].astype(str)
df_val['user'] = df_val['user'].astype(str)


baseline_hist_df = pd.read_csv(f"../../data/amazon/processed/{DATASET}_user_items_negs.tsv",
                               sep='\t', names=['user', 'item_list', 'neg_item_list'])
baseline_hist_df['baseline_history'] = baseline_hist_df['item_list'].apply(
    lambda x: [i.strip() for i in str(x).strip().split(',')[:-1]]
)
baseline_hist_df['user'] = baseline_hist_df['user'].astype(str)
baseline_hist_df = baseline_hist_df[['user', 'baseline_history']]

if args.history_mode == 'baseline':
    user_hist_df = baseline_hist_df.rename(columns={'baseline_history': 'history'})
else:
    user_hist_df = pd.read_csv(f"../../data/amazon/processed/{DATASET}_user_full_history.tsv",
                               sep='\t', names=['user', 'item_list'])
    user_hist_df['history'] = user_hist_df['item_list'].apply(
        lambda x: [i.strip() for i in str(x).strip().split(',') if i.strip()]
    )
user_hist_df['user'] = user_hist_df['user'].astype(str)
user_hist_df = user_hist_df[['user', 'history']]


item_desc_path = f"../../data/amazon/processed/{DATASET}_item_desc.tsv"
item_desc_df = pd.read_csv(item_desc_path, sep='\t', header=None, usecols=[0, 2], names=['item_id', 'summary'])
item_desc_df = item_desc_df.rename(columns={'item_id': 'item', 'summary': 'title'})
item_desc_df['item'] = item_desc_df['item'].astype(str)
# item_desc_df['title'] = item_desc_df['title'].fillna('')

item_title_map = item_desc_df.set_index('item')['title'].to_dict()


image_df = get_image_df(f"../../data/amazon/Amazon_images/{DATASET}")
image_df['item'] = image_df['item'].astype(str)


df_train = pd.merge(df_train, image_df, on="item")
df_train = pd.merge(df_train, item_desc_df, on="item")
df_train = pd.merge(df_train, user_hist_df, on="user")
df_train = pd.merge(df_train, baseline_hist_df, on="user")

df_val = pd.merge(df_val, image_df, on="item")
df_val = pd.merge(df_val, item_desc_df, on="item")
df_val = pd.merge(df_val, user_hist_df, on="user")
df_val = pd.merge(df_val, baseline_hist_df, on="user")


def _is_valid_path(path):
    for i in range(len(path) - 1):
        if path[i][2] != path[i + 1][1]:
            return False
    return len(path) > 0


def _filter_paths(paths):
    seen = set()
    result = []
    for path in paths:
        if not _is_valid_path(path):
            continue
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _load_path_lookup(split_tag):
    lookup = {}
    sources = [
        (f"../../path_retriever/saved_explanations/{DATASET}/bfs_amazon_{DATASET}_{split_tag}_pred_edge_to_paths", "BFS"),
        (f"../../path_retriever/saved_explanations/{DATASET}/mm_amazon_{DATASET}_{split_tag}_pred_edge_to_paths", "MM"),
    ]
    if args.no_bfs:
        sources = [(f, l) for f, l in sources if l != "BFS"]
    if args.no_mm:
        sources = [(f, l) for f, l in sources if l != "MM"]
    for paths_file, label in sources:
        if os.path.exists(paths_file):
            with open(paths_file, "rb") as f:
                raw = pickle.load(f)
            count = 0
            for (src, tgt), paths in raw.items():
                key = (src[1], tgt[1])
                filtered = _filter_paths(paths)
                if filtered:
                    lookup.setdefault(key, []).extend(filtered)
                    count += 1
            print(f"Loaded {count} {label} path entries from {paths_file}")
        else:
            print(f"[Info] {label} paths not found for {split_tag}, skipping.")
    return lookup


_trn_path_lookup = _load_path_lookup("trn")
_val_path_lookup = _load_path_lookup("val")


def _short_title(item_id):
    title = item_title_map.get(str(item_id), "Unknown")
    for sep in [',', '|', ' – ', '(']:
        idx = title.find(sep)
        if idx != -1:
            title = title[:idx]
    return ' '.join(title.split()[:7])


def _select_top_bridge_items(paths, target_item_id, n=5):
    bridge_item_to_users = {}
    bridge_item_is_bfs = {}
    bridge_item_to_hist = {}

    for path in paths:
        if len(path) == 3:  # BFS: user→bridge_item→bridge_user→target
            bridge_item_id = path[0][2]
            bridge_user_id = path[1][2]
            bridge_item_is_bfs[bridge_item_id] = True
            bridge_item_to_hist.setdefault(bridge_item_id, None)
        else:               # MM (4-hop): user→hist_item→bridge_item→bridge_user→target
            bridge_item_id = path[1][2]
            bridge_user_id = path[2][2]
            if bridge_item_id not in bridge_item_is_bfs:
                bridge_item_is_bfs[bridge_item_id] = False
                bridge_item_to_hist[bridge_item_id] = path[0][2]  # history_item
        bridge_item_to_users.setdefault(bridge_item_id, set()).add(bridge_user_id)

    min_bfs = round(n * 0.6)
    ranked = sorted(bridge_item_to_users.items(), key=lambda x: len(x[1]), reverse=True)
    bfs_items = [(bid, u) for bid, u in ranked if bridge_item_is_bfs.get(bid, False)]
    rest_items = [(bid, u) for bid, u in ranked if not bridge_item_is_bfs.get(bid, False)]

    guaranteed = bfs_items[:min_bfs]
    remaining_bfs = bfs_items[min_bfs:]
    remaining_pool = sorted(remaining_bfs + rest_items, key=lambda x: len(x[1]), reverse=True)
    top = guaranteed + remaining_pool[:n - len(guaranteed)]

    target_title = _short_title(target_item_id)
    lines = []
    for bridge_item_id, users in top:
        count = len(users)
        user_str = f"{count} other user{'s' if count > 1 else ''}"
        if bridge_item_is_bfs.get(bridge_item_id, True):
            lines.append(
                f"the user bought '{_short_title(bridge_item_id)}'"
                f" → also bought by {user_str}"
                f" → who also bought '{target_title}'"
            )
        else:
            hist = bridge_item_to_hist.get(bridge_item_id)
            if bridge_item_id == target_item_id:
                lines.append(
                    f"the user bought '{_short_title(hist)}'"
                    f" → visually similar to '{target_title}'"
                )
            else:
                lines.append(
                    f"the user bought '{_short_title(hist)}'"
                    f" → visually similar to '{_short_title(bridge_item_id)}'"
                    f" → also bought by {user_str}"
                    f" → who also bought '{target_title}'"
                )
    return lines


def _select_relevant_history(full_history, baseline_history, paths, target_item_id, k=10):
    target_item = str(target_item_id)
    history = [str(item) for item in full_history if str(item) != target_item]
    history_pos = {item: idx for idx, item in enumerate(history)}
    history_set = set(history)
    baseline_items = [str(item) for item in baseline_history if str(item) != target_item]

    bfs_hist_scores = {}
    mm_hist_scores = {}
    for path in paths:
        if len(path) == 3:  # BFS: user→bridge_item→bridge_user→target
            hist_item = str(path[0][2])
            bridge_user = str(path[1][2])
            if hist_item in history_set:
                bfs_hist_scores.setdefault(hist_item, set()).add(bridge_user)
        else:               # MM (4-hop): user→hist_item→sim_item→bridge_user→target
            hist_item = str(path[0][2])
            bridge_user = str(path[2][2])
            if hist_item in history_set and hist_item not in bfs_hist_scores:
                mm_hist_scores.setdefault(hist_item, set()).add(bridge_user)

    min_bfs = round(k * 0.6)
    bfs_ranked = sorted(bfs_hist_scores, key=lambda x: len(bfs_hist_scores[x]), reverse=True)
    mm_ranked = sorted(mm_hist_scores, key=lambda x: len(mm_hist_scores[x]), reverse=True)

    guaranteed = bfs_ranked[:min_bfs]
    selected = set(guaranteed)
    result = list(guaranteed)

    overflow_and_mm = sorted(
        [(it, len(bfs_hist_scores[it])) for it in bfs_ranked[min_bfs:]] +
        [(it, len(mm_hist_scores[it])) for it in mm_ranked if it not in selected],
        key=lambda x: x[1], reverse=True
    )
    for it, _ in overflow_and_mm:
        if len(result) >= k:
            break
        if it not in selected:
            result.append(it)
            selected.add(it)

    for it in reversed(baseline_items):
        if len(result) >= k:
            break
        if it not in selected:
            result.append(it)
            selected.add(it)

    result.sort(key=lambda x: history_pos.get(x, len(history)))
    return result[:k]


def build_prompt(row, path_lookup, max_title_len=350):
    target_user_id = int(row['user'])
    paths = path_lookup.get((target_user_id, int(row['item'])), [])
    if args.history_mode == 'baseline':
        display_history = [str(item) for item in row['history'] if str(item) != str(row['item'])]
    else:
        display_history = _select_relevant_history(row['history'], row['baseline_history'], paths, row['item'], k=args.k)

    history_lines = "\n".join(
        f"{i + 1}. {item_title_map.get(item, 'Unknown')[:max_title_len]}"
        for i, item in enumerate(display_history)
    )

    path_section = ""
    if paths and not args.hide_cf_signals:
        path_lines = "\n".join(f"- {line}" for line in _select_top_bridge_items(paths, int(row['item']), n=args.k))
        bfs_count = sum(1 for p in paths if len(p) == 3)
        mm_count = sum(1 for p in paths if len(p) == 4)
        path_section = (
            f"\nCollaborative filtering signals ({bfs_count} co-occurrence, {mm_count} similarity-based):\n{path_lines}\n"
        )

    prompt = (
        f"The user's interaction history:\n{history_lines}\n"
        f"{path_section}\n"
        f"Please predict whether this user would interact with the product at the next opportunity. "
        f"The product's title is '{row['title'][:max_title_len]}', and the given image is this product's image. "
        "Please only response 'yes' or 'no' based on your judgement, do not include any other content "
        "including words, space, and punctuations in your response."
    )
    return prompt


df_train['prompt'] = df_train.apply(lambda r: build_prompt(r, _trn_path_lookup), axis=1)
df_train['ground_truth'] = df_train.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
trn_covered = df_train.apply(lambda r: (int(r['user']), int(r['item'])) in _trn_path_lookup, axis=1).sum()
print(f"Train CF coverage: {trn_covered}/{len(df_train)} ({100*trn_covered/len(df_train):.1f}%)")
trn_pos = df_train[df_train['label'] == 1]
trn_covered = trn_pos.apply(lambda r: (int(r['user']), int(r['item'])) in _trn_path_lookup, axis=1).sum()
print(f"Train CF coverage (positive samples): {trn_covered}/{len(trn_pos)} ({100*trn_covered/len(trn_pos):.1f}%)")
trn_neg = df_train[df_train['label'] == 0]
trn_neg_covered = trn_neg.apply(lambda r: (int(r['user']), int(r['item'])) in _trn_path_lookup, axis=1).sum()
print(f"Train CF coverage (negative samples): {trn_neg_covered}/{len(trn_neg)} ({100*trn_neg_covered/len(trn_neg):.1f}%)")
df_train = df_train[['prompt', 'image', 'ground_truth']]

df_val['prompt'] = df_val.apply(lambda r: build_prompt(r, _val_path_lookup), axis=1)
df_val['ground_truth'] = df_val.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
val_covered = df_val.apply(lambda r: (int(r['user']), int(r['item'])) in _val_path_lookup, axis=1).sum()
print(f"Val CF coverage: {val_covered}/{len(df_val)} ({100*val_covered/len(df_val):.1f}%)")
val_pos = df_val[df_val['label'] == 1]
val_covered = val_pos.apply(lambda r: (int(r['user']), int(r['item'])) in _val_path_lookup, axis=1).sum()
print(f"Val CF coverage (positive samples): {val_covered}/{len(val_pos)} ({100*val_covered/len(val_pos):.1f}%)")
val_neg = df_val[df_val['label'] == 0]
val_neg_covered = val_neg.apply(lambda r: (int(r['user']), int(r['item'])) in _val_path_lookup, axis=1).sum()
print(f"Val CF coverage (negative samples): {val_neg_covered}/{len(val_neg)} ({100*val_neg_covered/len(val_neg):.1f}%)")
df_val = df_val[['prompt', 'image', 'ground_truth']]

train_dataset = Dataset.from_pandas(df_train)
train_dataset = train_dataset.cast_column("image", Image())
train_dataset = train_dataset.select(range(min(25000, len(train_dataset))))
train_dataset = train_dataset.shuffle(seed=2024)

val_dataset = Dataset.from_pandas(df_val)
val_dataset = val_dataset.cast_column("image", Image())
val_dataset = val_dataset.select(range(min(1000, len(val_dataset))))

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
print(dataset)
dataset.save_to_disk(f"Amazon-{DATASET}-training-recurrent{VARIANT}")
