import pandas as pd
import numpy as np

train = pd.read_csv('playground-series-s6e9/train.csv')
train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)

triplet_stats = train.groupby(['Annual_Income_USD', 'Subsidy_Available', 'Range_Anxiety_Level'])['target'].agg(['count', 'mean', 'std'])
print(f"Total unique triplets in train: {len(triplet_stats):,}")
multi = triplet_stats[triplet_stats['count'] > 1]
pure = triplet_stats[(triplet_stats['std'] == 0) & (triplet_stats['count'] > 1)]
print(f"Multi-sample triplets (>1 row): {len(multi):,}")
print(f"100% PURE (std == 0.0) multi-sample triplets: {len(pure):,} ({len(pure)/len(multi)*100:.2f}%)")

pure_0 = triplet_stats[(triplet_stats['mean'] == 0.0) & (triplet_stats['count'] >= 5)]
pure_1 = triplet_stats[(triplet_stats['mean'] == 1.0) & (triplet_stats['count'] >= 5)]
print(f"Triplets with count >= 5 and 100% Zero: {len(pure_0):,}")
print(f"Triplets with count >= 5 and 100% One:  {len(pure_1):,}")

# What percentage of rows fall into 100% deterministic (mean=0 or mean=1) with count >= 5?
rows_pure_0 = pure_0['count'].sum()
rows_pure_1 = pure_1['count'].sum()
print(f"Total training rows in >=5 sample pure triplets: {rows_pure_0 + rows_pure_1:,} ({((rows_pure_0 + rows_pure_1)/len(train)*100):.2f}%)")
