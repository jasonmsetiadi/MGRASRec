"""Generate train/val/test pair CSVs for an Amazon dataset.

Reads the pre-split user_items_negs_{train,val,test}.csv files produced by
process_data.py and outputs three CSVs with columns: user, item, label.

Negative sampling ratios:
  train: 1 negative per positive
  val:   1 negative per positive
  test:  20 negatives per positive (all available)

Usage:
    python split.py                          # defaults to Baby_Products
    python split.py Video_Games
"""

import sys
import os
import numpy as np
import pandas as pd

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'Baby_Products'
IN_DIR = './processed'
OUT_DIR = f'./Split/{DATASET}'

NEG_TRAIN = 1
NEG_VAL = 1
NEG_TEST = 20

RNG = np.random.default_rng(seed=42)


def load_split(kind):
    path = os.path.join(IN_DIR, f'{DATASET}_user_items_negs_{kind}.csv')
    df = pd.read_csv(path, sep='\t', header=None, names=['user_id', 'pos', 'neg'])
    df['pos'] = df['pos'].apply(lambda x: str(x).strip().split(','))
    df['neg'] = df['neg'].apply(lambda x: str(x).strip().split(','))
    return df


def make_pairs(df, n_neg):
    rows = []
    for _, row in df.iterrows():
        pos_item = row['pos'][-1].strip()
        rows.append([row['user_id'], pos_item, 1])
        neg_pool = row['neg']
        chosen = RNG.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False)
        for item in chosen:
            rows.append([row['user_id'], item.strip(), 0])
    return pd.DataFrame(rows, columns=['user', 'item', 'label'])


os.makedirs(OUT_DIR, exist_ok=True)

for kind, n_neg in [('train', NEG_TRAIN), ('val', NEG_VAL), ('test', NEG_TEST)]:
    df = load_split(kind)
    pairs = make_pairs(df, n_neg)
    out_path = os.path.join(OUT_DIR, f'{kind}_pairs.csv')
    pairs.to_csv(out_path, index=False)
    print(f"Saved {len(pairs)} rows to {out_path}")
