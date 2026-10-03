import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit
import time

print("Loading data...", flush=True)
t0 = time.time()
train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
train['y'] = y

# Create clean triplet strings
train['triplet'] = train['Annual_Income_USD'].astype(str) + '__' + train['Subsidy_Available'].astype(str) + '__' + train['Range_Anxiety_Level'].astype(str)
train['pair'] = train['Annual_Income_USD'].astype(str) + '__' + train['Subsidy_Available'].astype(str)
train['inc_str'] = train['Annual_Income_USD'].astype(str)

print(f"Data prepared in {time.time()-t0:.1f}s", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

global_prior = y.mean()

for m in [5.0, 10.0, 20.0, 50.0]:
    oof_triplet = np.zeros(len(train), dtype=np.float64)
    oof_pair = np.zeros(len(train), dtype=np.float64)
    oof_inc = np.zeros(len(train), dtype=np.float64)

    for tr_idx, va_idx in skf.split(train, y):
        tr_df = train.iloc[tr_idx]
        va_df = train.iloc[va_idx]
        
        # 1. Triplet TE
        grp_triplet = tr_df.groupby('triplet')['y'].agg(['count', 'mean'])
        s_trip = ((grp_triplet['count'] * grp_triplet['mean'] + m * global_prior) / (grp_triplet['count'] + m)).to_dict()
        oof_triplet[va_idx] = va_df['triplet'].map(s_trip).fillna(global_prior).values
        
        # 2. Pair TE
        grp_pair = tr_df.groupby('pair')['y'].agg(['count', 'mean'])
        s_pair = ((grp_pair['count'] * grp_pair['mean'] + m * global_prior) / (grp_pair['count'] + m)).to_dict()
        oof_pair[va_idx] = va_df['pair'].map(s_pair).fillna(global_prior).values
        
        # 3. Exact Income TE
        grp_inc = tr_df.groupby('inc_str')['y'].agg(['count', 'mean'])
        s_inc = ((grp_inc['count'] * grp_inc['mean'] + m * global_prior) / (grp_inc['count'] + m)).to_dict()
        oof_inc[va_idx] = va_df['inc_str'].map(s_inc).fillna(global_prior).values

    print(f"Smoothing m={m:4.1f}:")
    print(f"  Exact Income TE Standalone AUC:        {roc_auc_score(y, oof_inc):.6f}")
    print(f"  Income+Subsidy Pair TE Standalone AUC: {roc_auc_score(y, oof_pair):.6f}")
    print(f"  Triplet (Inc+Sub+Anx) TE Standalone:   {roc_auc_score(y, oof_triplet):.6f}")
