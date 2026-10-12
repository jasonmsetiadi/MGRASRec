import os
import dgl
import argparse
import pandas as pd
from tqdm import tqdm

# Run from path_retriever/microlens/
# Usage:
#   python dgl_extractor.py --split trn
#   python dgl_extractor.py --split val
#   python dgl_extractor.py --split tst

parser = argparse.ArgumentParser(description='Build DGL graph for MicroLens')
parser.add_argument('--split', type=str, required=True, choices=['trn', 'val', 'tst'])
parser.add_argument('--full_pairs_path', type=str, default='../../data/microlens/processed/MicroLens-50k_pairs.csv',
                    help='Path to full MicroLens-50k_pairs.csv (used to build rich graph edges)')
parser.add_argument('--pairs_dir', type=str, default='../../data/microlens/Split',
                    help='Directory containing {split}_pairs.csv files')
parser.add_argument('--output_dir', type=str, default='data',
                    help='Directory to save {split}_graph.bin')
args = parser.parse_args()

SPLIT_CSV = {'trn': 'train_pairs.csv', 'val': 'val_pairs.csv', 'tst': 'test_pairs.csv'}

os.makedirs(args.output_dir, exist_ok=True)

print(f"Processing MicroLens {args.split} split...")

# Load full interaction history (all 359k interactions, not the top-6 truncated version)
full_df = pd.read_csv(args.full_pairs_path)
full_df['user'] = full_df['user'].astype(int)
full_df['item'] = full_df['item'].astype(int)

# Load prediction-target pairs for this split
pairs_path = os.path.join(args.pairs_dir, SPLIT_CSV[args.split])
pairs_df = pd.read_csv(pairs_path)
pairs_df['user'] = pairs_df['user'].astype(int)
pairs_df['item'] = pairs_df['item'].astype(int)
likes_pairs = list(zip(pairs_df.loc[pairs_df['label'] == 1, 'user'],
                       pairs_df.loc[pairs_df['label'] == 1, 'item']))
likes_set = set(likes_pairs)

# Build have_watched edges from full history, excluding likes (prediction-target) edges
have_watched_pairs = []
for row in tqdm(full_df.itertuples(), total=len(full_df), desc="Building edges"):
    user, item = row.user, row.item
    if (user, item) not in likes_set:
        have_watched_pairs.append((user, item))

all_users = set(u for u, _ in have_watched_pairs) | set(u for u, _ in likes_pairs)
all_items = set(i for _, i in have_watched_pairs) | set(i for _, i in likes_pairs)
num_users = max(all_users) + 1
num_items = max(all_items) + 1

print(f"  Users: {num_users}, Items: {num_items}")
print(f"  have_watched edges: {len(have_watched_pairs)}, likes edges: {len(likes_pairs)}")

g = dgl.heterograph({
    ('user', 'have_watched', 'item'): ([], []),
    ('item', 'watched_by', 'user'): ([], []),
    ('user', 'likes', 'item'): ([], []),
}, num_nodes_dict={'user': num_users, 'item': num_items})

if have_watched_pairs:
    w_src, w_dst = zip(*have_watched_pairs)
    g.add_edges(list(w_src), list(w_dst), etype='have_watched')
    g.add_edges(list(w_dst), list(w_src), etype='watched_by')

if likes_pairs:
    l_src, l_dst = zip(*likes_pairs)
    g.add_edges(list(l_src), list(l_dst), etype='likes')

print(f"  have_watched edges in graph: {g.num_edges('have_watched')}")
print(f"  watched_by edges in graph:   {g.num_edges('watched_by')}")
print(f"  likes edges in graph:        {g.num_edges('likes')}")

out_path = os.path.join(args.output_dir, f'{args.split}_graph.bin')
dgl.save_graphs(out_path, [g])
print(f"Graph saved to {out_path}")
