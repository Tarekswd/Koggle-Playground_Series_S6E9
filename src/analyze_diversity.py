import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

y = (pd.read_csv('playground-series-s6e9/train.csv')['Will_Buy_EV'] == 'Yes').astype(int).values

files = [
    'oof_predictions.npy',
    'ladder_lgb_oof.npy',
    'optuna_lgb_oof.npy',
    'xgb79_oof.npy',
    'cb79_oof.npy',
    'xgb_oof.npy',
    'catboost_oof.npy',
    'stacking_logit_pinnacle_oof.npy',
    'triplet_10fold_lgb_oof.npy'
]

main_pred = np.load('stacking_logit_pinnacle_oof.npy')

print("=" * 75)
print(f"{'Model File':32s} | {'CV AUC':8s} | {'Spearman vs Main':16s} | {'Valid Diverse':13s}")
print("=" * 75)
for f in files:
    try:
        p = np.load(f)
        auc = roc_auc_score(y, p)
        corr, _ = spearmanr(main_pred, p)
        valid = "YES (< 0.99)" if (corr < 0.99 and auc >= 0.940) else "NO"
        if f == 'stacking_logit_pinnacle_oof.npy':
            valid = "MAIN (Anchor)"
        print(f"{f:32s} | {auc:.6f} | {corr:16.6f} | {valid:13s}")
    except Exception as e:
        print(f"{f:32s} | ERROR: {e}")
print("=" * 75)
