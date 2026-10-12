import os
import pickle
import argparse
from collections import defaultdict

import dgl
import pandas as pd
from tqdm.auto import tqdm

# Run from path_retriever/microlens/
# Usage:
#   python bfs.py --split trn
#   python bfs.py --split val
#   python bfs.py --split tst
#
# Requires: dgl_extractor.py to have been run first for the given split.

parser = argparse.ArgumentParser(description='BFS-based path retrieval for MicroLens user-item pairs')
parser.add_argument('--split', type=str, required=True, choices=['trn', 'val', 'tst'])
parser.add_argument('--graph_dir', type=str, default='data',
                    help='Directory containing {split}_graph.bin files')
parser.add_argument('--pairs_dir', type=str, default='../../data/microlens/Split',
                    help='Directory containing {split}_pairs.csv files')
parser.add_argument('--saved_explanation_dir', type=str, default='../saved_explanations/microlens',
                    help='Directory to write path output files')
parser.add_argument('--max_bridge_users', type=int, default=100,
                    help='Cap on bridge users sampled per target item (for popular items)')
args = parser.parse_args()

SPLIT_CSV = {'trn': 'train_pairs.csv', 'val': 'val_pairs.csv', 'tst': 'test_pairs.csv'}

graph_path = os.path.join(args.graph_dir, f'{args.split}_graph.bin')
pairs_path = os.path.join(args.pairs_dir, SPLIT_CSV[args.split])
out_file = os.path.join(
    args.saved_explanation_dir,
    f'bfs_microlens_{args.split}_pred_edge_to_paths'
)

print(f"Loading graph from {graph_path}")
g = dgl.load_graphs(graph_path)[0][0]
print(f"  Users: {g.num_nodes('user')}, Items: {g.num_nodes('item')}")

# Build adjacency dicts from have_watched / watched_by edges
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


def bfs_paths(user_id: int, item_id: int) -> list:
    """
    Find 3-hop CF paths: user → bridge_item → bridge_user → target_item.
    Path format: list of (canonical_etype, src_id, tgt_id).
    """
    paths = []
    user_items = user_to_items.get(user_id, set())
    if not user_items:
        return paths

    bridge_users = list(item_to_users.get(item_id, set()))
    if len(bridge_users) > args.max_bridge_users:
        bridge_users = bridge_users[:args.max_bridge_users]

    for bridge_user in bridge_users:
        if bridge_user == user_id:
            continue
        shared_items = user_items & user_to_items.get(bridge_user, set())
        for bridge_item in shared_items:
            paths.append([
                (('user', 'have_watched', 'item'), user_id, bridge_item),
                (('item', 'watched_by', 'user'), bridge_item, bridge_user),
                (('user', 'have_watched', 'item'), bridge_user, item_id),
            ])
    return paths


# Run BFS over all pairs
pred_edge_to_paths: dict = {}
found = 0
for user_id, item_id in tqdm(all_pairs):
    paths = bfs_paths(user_id, item_id)
    if paths:
        key = (('user', user_id), ('item', item_id))
        pred_edge_to_paths[key] = paths
        found += 1

print(f"\nBFS found paths for {found} / {len(all_pairs)} pairs ({found / len(all_pairs) * 100:.1f}%)")

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
