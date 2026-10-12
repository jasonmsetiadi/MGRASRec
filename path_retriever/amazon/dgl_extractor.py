import os
import dgl
import argparse
import pandas as pd
from tqdm import tqdm

# Run from path_retriever/amazon/
# Usage:
#   python dgl_extractor.py --category Baby_Products --split trn
#   python dgl_extractor.py --category Baby_Products --split val
#   python dgl_extractor.py --category Baby_Products --split tst
#   python dgl_extractor.py --category Video_Games --split tst

parser = argparse.ArgumentParser(description='Build DGL graph for Amazon')
parser.add_argument('--category', type=str, required=True, choices=['Baby_Products', 'Video_Games'])
parser.add_argument('--split', type=str, required=True, choices=['trn', 'val', 'tst'])
parser.add_argument('--data_dir', type=str, default='../../data/amazon',
                    help='Path to Amazon processed data root')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Directory to save {split}_graph.bin (default: data/{category})')
args = parser.parse_args()

output_dir = args.output_dir or f'data/{args.category}'

SPLIT_CSV = {'trn': 'train', 'val': 'val', 'tst': 'test'}

os.makedirs(output_dir, exist_ok=True)

print(f"Processing Amazon {args.category} {args.split} split...")

# Load all historical interactions from full pairs file (not truncated user_items_negs.tsv)
hist_path = f'{args.data_dir}/processed/{args.category}_u_i_pairs.tsv'
hist_df = pd.read_csv(hist_path, sep='\t')
hist_df['user_id'] = hist_df['user_id'].astype(int)
hist_df['item_id'] = hist_df['item_id'].astype(int)

# Load prediction-target pairs for this split
pairs_path = f'{args.data_dir}/Split/{args.category}/{SPLIT_CSV[args.split]}_pairs.csv'
pairs_df = pd.read_csv(pairs_path)
pairs_df['user'] = pairs_df['user'].astype(int)
pairs_df['item'] = pairs_df['item'].astype(int)
likes_pairs = list(zip(pairs_df.loc[pairs_df['label'] == 1, 'user'],
                       pairs_df.loc[pairs_df['label'] == 1, 'item']))
likes_set = set(likes_pairs)

# Build have_bought edges from full history, excluding likes (prediction-target) edges
have_bought_pairs = []
for row in tqdm(hist_df.itertuples(), total=len(hist_df), desc="Building edges"):
    user, item = row.user_id, row.item_id
    if (user, item) not in likes_set:
        have_bought_pairs.append((user, item))

all_users = set(u for u, _ in have_bought_pairs) | set(u for u, _ in likes_pairs)
all_items = set(i for _, i in have_bought_pairs) | set(i for _, i in likes_pairs)
num_users = max(all_users) + 1
num_items = max(all_items) + 1

print(f"  Users: {num_users}, Items: {num_items}")
print(f"  have_bought edges: {len(have_bought_pairs)}, likes edges: {len(likes_pairs)}")

g = dgl.heterograph({
    ('user', 'have_bought', 'item'): ([], []),
    ('item', 'bought_by', 'user'): ([], []),
    ('user', 'likes', 'item'): ([], []),
}, num_nodes_dict={'user': num_users, 'item': num_items})

if have_bought_pairs:
    b_src, b_dst = zip(*have_bought_pairs)
    g.add_edges(list(b_src), list(b_dst), etype='have_bought')
    g.add_edges(list(b_dst), list(b_src), etype='bought_by')

if likes_pairs:
    l_src, l_dst = zip(*likes_pairs)
    g.add_edges(list(l_src), list(l_dst), etype='likes')

print(f"  have_bought edges in graph: {g.num_edges('have_bought')}")
print(f"  bought_by edges in graph:   {g.num_edges('bought_by')}")
print(f"  likes edges in graph:       {g.num_edges('likes')}")

out_path = os.path.join(output_dir, f'{args.split}_graph.bin')
dgl.save_graphs(out_path, [g])
print(f"Graph saved to {out_path}")

# Also save a copy to PaGE-Link datasets
pagelink_out = f'../PaGE-Link/datasets/amazon_{args.category}_{args.split}.bin'
dgl.save_graphs(pagelink_out, [g])
print(f"PaGE-Link copy saved to {pagelink_out}")
