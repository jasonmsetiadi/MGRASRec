import os
import pickle
import argparse
from collections import defaultdict

import dgl
import numpy as np
import faiss
import pandas as pd
from tqdm.auto import tqdm

# Run from path_retriever/microlens/
# Requires: dgl_extractor.py run first, and clip_embedder.py run for microlens.
# Usage:
#   python multimodal_retriever.py --split trn
#   python multimodal_retriever.py --split val
#   python multimodal_retriever.py --split tst

parser = argparse.ArgumentParser(description='4-hop multimodal path retrieval for MicroLens')
parser.add_argument('--split', type=str, required=True, choices=['trn', 'val', 'tst'])
parser.add_argument('--graph_dir', type=str, default='data')
parser.add_argument('--pairs_dir', type=str, default='../../data/microlens/Split')
parser.add_argument('--saved_explanation_dir', type=str, default='../saved_explanations/microlens')
parser.add_argument('--clip_dir', type=str, default='../saved_explanations/clip_microlens',
                    help='Directory containing embeddings.npy, item_ids.npy, faiss.index')
parser.add_argument('--top_k', type=int, default=300,
                    help='Max CLIP nearest neighbours to evaluate per history item. '
                         'min_similarity does most of the quality filtering; this cap '
                         'mainly guards against near-duplicate-cluster blowups (some '
                         'items have 1000+ neighbours clearing the similarity threshold).')
parser.add_argument('--min_similarity', type=float, default=None,
                    help='Minimum cosine similarity for a neighbour to qualify as a bridge item '
                         '(default: 0.62 for microlens)')
parser.add_argument('--max_bridge_users', type=int, default=100,
                    help='Cap on bridge users sampled per target item')
args = parser.parse_args()

_DATASET_MIN_SIMILARITY = {'microlens': 0.731}
if args.min_similarity is None:
    args.min_similarity = _DATASET_MIN_SIMILARITY['microlens']
    print(f"Using dataset default min_similarity={args.min_similarity} for microlens")

SPLIT_CSV = {'trn': 'train_pairs.csv', 'val': 'val_pairs.csv', 'tst': 'test_pairs.csv'}

graph_path = os.path.join(args.graph_dir, f'{args.split}_graph.bin')
pairs_path = os.path.join(args.pairs_dir, SPLIT_CSV[args.split])
out_file = os.path.join(
    args.saved_explanation_dir,
    f'mm_microlens_{args.split}_pred_edge_to_paths'
)

# Load graph
print(f"Loading graph from {graph_path}")
g = dgl.load_graphs(graph_path)[0][0]
print(f"  Users: {g.num_nodes('user')}, Items: {g.num_nodes('item')}")

print("Building adjacency lists...")
user_to_items: dict[int, set] = defaultdict(set)
src, dst = g.edges(etype='have_watched')
for u, i in zip(src.tolist(), dst.tolist()):
    user_to_items[u].add(i)

item_to_users: dict[int, set] = defaultdict(set)
src, dst = g.edges(etype='watched_by')
for i, u in zip(src.tolist(), dst.tolist()):
    item_to_users[i].add(u)

print(f"  {len(user_to_items)} users with watch history")
print(f"  {len(item_to_users)} items with watchers")

# Load CLIP artefacts
print(f"Loading CLIP artefacts from {args.clip_dir}")
embeddings = np.load(os.path.join(args.clip_dir, 'embeddings.npy')).astype(np.float32)
item_ids_arr = np.load(os.path.join(args.clip_dir, 'item_ids.npy'))
faiss_index = faiss.read_index(os.path.join(args.clip_dir, 'faiss.index'))

item_id_to_idx: dict[int, int] = {int(iid): idx for idx, iid in enumerate(item_ids_arr)}
print(f"  {len(item_id_to_idx)} items in CLIP index")

# Load pairs, restricted to the same first-N-users subset that
# dataset_create_rag.py / multi_col_dataset_rag.py actually consume
# (train: first 12500 users, val: first 500, test: first 250 — by
# row-appearance order in the full pairs.csv, not the OOV-filtered _mllm.csv).
N_USERS = {'trn': 12500, 'val': 500, 'tst': 250}
pairs_df = pd.read_csv(pairs_path)
pairs_df['user'] = pairs_df['user'].astype(int)
pairs_df['item'] = pairs_df['item'].astype(int)
first_users = pairs_df['user'].drop_duplicates().head(N_USERS[args.split])
pairs_df = pairs_df[pairs_df['user'].isin(set(first_users))]
all_pairs = list(zip(pairs_df['user'], pairs_df['item']))
print(f"\nLoaded {len(all_pairs)} pairs from {pairs_path} "
      f"(restricted to first {N_USERS[args.split]} users)")
print(f"Settings: top_k={args.top_k}, min_similarity={args.min_similarity}, "
      f"max_bridge_users={args.max_bridge_users}\n")


def mm_paths(user_id: int, item_id: int) -> list:
    """
    Find 4-hop multimodal CF paths:
      user → history_item --similar_to--> bridge_item → bridge_user → target_item
    """
    user_items = user_to_items.get(user_id, set())
    bridge_users = list(item_to_users.get(item_id, set()))
    if len(bridge_users) > args.max_bridge_users:
        bridge_users = bridge_users[:args.max_bridge_users]

    valid_hist = [i for i in user_items if i in item_id_to_idx]
    if not valid_hist or not bridge_users:
        return []

    query_embs = embeddings[[item_id_to_idx[i] for i in valid_hist]]
    scores_arr, neighbor_idxs = faiss_index.search(query_embs, args.top_k + 1)

    # sim_item → history_item mapping (exclude user's own items, apply similarity floor)
    sim_to_hist: dict[int, int] = {}
    for h_pos, hist_item in enumerate(valid_hist):
        for nb, score in zip(neighbor_idxs[h_pos], scores_arr[h_pos]):
            if score < args.min_similarity:
                continue
            sim_id = int(item_ids_arr[nb])
            if sim_id != hist_item and sim_id not in user_items:
                sim_to_hist.setdefault(sim_id, hist_item)

    if not sim_to_hist:
        return []

    paths = []
    for bridge_user in bridge_users:
        if bridge_user == user_id:
            continue
        bu_items = user_to_items.get(bridge_user, set())
        for sim_item, hist_item in sim_to_hist.items():
            if sim_item in bu_items:
                paths.append([
                    (('user', 'have_watched', 'item'), user_id,     hist_item),
                    (('item', 'visually_similar_to', 'item'), hist_item, sim_item),
                    (('item', 'watched_by', 'user'),  sim_item,  bridge_user),
                    (('user', 'have_watched', 'item'), bridge_user,  item_id),
                ])
    return paths


# Run retrieval over all pairs
pred_edge_to_paths: dict = {}
found = 0
for user_id, item_id in tqdm(all_pairs):
    paths = mm_paths(user_id, item_id)
    if paths:
        key = (('user', user_id), ('item', item_id))
        pred_edge_to_paths[key] = paths
        found += 1

print(f"\nMM retrieval found paths for {found} / {len(all_pairs)} pairs "
      f"({found / len(all_pairs) * 100:.1f}%)")

os.makedirs(args.saved_explanation_dir, exist_ok=True)
with open(out_file, 'wb') as f:
    pickle.dump(pred_edge_to_paths, f)
print(f"Saved to {out_file}")

# Path-count distribution
total = len(all_pairs)
path_counts = {k: len(v) for k, v in pred_edge_to_paths.items()}
zero  = total - len(path_counts)
one   = sum(1 for v in path_counts.values() if v == 1)
two   = sum(1 for v in path_counts.values() if v == 2)
three = sum(1 for v in path_counts.values() if v == 3)
four  = sum(1 for v in path_counts.values() if v == 4)
five  = sum(1 for v in path_counts.values() if v == 5)
more  = sum(1 for v in path_counts.values() if v > 5)

print(f"\n{'Category':<15} {'Count':>8} {'%':>7}")
print("-" * 32)
print(f"{'Total':<15} {total:>8}")
print(f"{'0 paths':<15} {zero:>8} {zero / total * 100:>6.1f}%")
print(f"{'1 path':<15} {one:>8} {one / total * 100:>6.1f}%")
print(f"{'2 paths':<15} {two:>8} {two / total * 100:>6.1f}%")
print(f"{'3 paths':<15} {three:>8} {three / total * 100:>6.1f}%")
print(f"{'4 paths':<15} {four:>8} {four / total * 100:>6.1f}%")
print(f"{'5 paths':<15} {five:>8} {five / total * 100:>6.1f}%")
print(f"{'> 5 paths':<15} {more:>8} {more / total * 100:>6.1f}%")
