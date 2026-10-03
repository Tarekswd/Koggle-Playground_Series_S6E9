"""
3-Way Frontier Ensemble: Generator-Aware Ridge + LGB Phase 1 + CatBoost Digit Ladder
Finds optimal blend weights on OOF and outputs calibrated final submissions.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from statistics import NormalDist
import os

print("=" * 70)
print("Evaluating 3-Way Frontier Ensemble (Ridge + LGB-P1 + CatBoost)")
print("=" * 70)

# Check files
files = [
    'generator_aware_ridge_oof.npy', 'generator_aware_ridge_test.npy',
    'lgb_phase1_oof.npy', 'lgb_phase1_test.npy',
    'catboost_digit_ladder_oof.npy', 'catboost_digit_ladder_test.npy'
]

missing = [f for f in files if not os.path.exists(f)]
if missing:
    print(f"Waiting for files to finish: {missing}")
    exit(1)

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

ridge_oof = np.load('generator_aware_ridge_oof.npy')
ridge_test = np.load('generator_aware_ridge_test.npy')

lgb_oof = np.load('lgb_phase1_oof.npy')
lgb_test = np.load('lgb_phase1_test.npy')

cb_oof = np.load('catboost_digit_ladder_oof.npy')
cb_test = np.load('catboost_digit_ladder_test.npy')

print(f"Individual Model OOF AUCs:")
print(f"  1. Generator-Aware Ridge: {roc_auc_score(y, ridge_oof):.7f}")
print(f"  2. LGB Phase 1 Extended:  {roc_auc_score(y, lgb_oof):.7f}")
print(f"  3. CatBoost Digit-Ladder: {roc_auc_score(y, cb_oof):.7f}")

# Rank transform
r_ridge_oof = rankdata(ridge_oof) / len(ridge_oof)
r_ridge_test = rankdata(ridge_test) / len(ridge_test)

r_lgb_oof = rankdata(lgb_oof) / len(lgb_oof)
r_lgb_test = rankdata(lgb_test) / len(lgb_test)

r_cb_oof = rankdata(cb_oof) / len(cb_oof)
r_cb_test = rankdata(cb_test) / len(cb_test)

# Diversity check: Spearman correlation between rank predictions
print("\nPrediction Spearman Rank Correlations:")
print(f"  Ridge vs LGB:    {np.corrcoef(r_ridge_oof, r_lgb_oof)[0, 1]:.5f}")
print(f"  Ridge vs CB:     {np.corrcoef(r_ridge_oof, r_cb_oof)[0, 1]:.5f}")
print(f"  LGB vs CB:       {np.corrcoef(r_lgb_oof, r_cb_oof)[0, 1]:.5f}")

# Grid search for optimal 3-way rank weights
best_auc = 0.0
best_weights = None

print("\nRunning fine grid search for optimal rank blend weights...")
step = 0.02
for w_ridge in np.arange(0.60, 0.92, step):
    for w_lgb in np.arange(0.04, 0.35, step):
        w_cb = 1.0 - w_ridge - w_lgb
        if w_cb < 0.02 or w_cb > 0.35:
            continue
        blend_oof = w_ridge * r_ridge_oof + w_lgb * r_lgb_oof + w_cb * r_cb_oof
        score = roc_auc_score(y, blend_oof)
        if score > best_auc:
            best_auc = score
            best_weights = (w_ridge, w_lgb, w_cb)

w_r, w_l, w_c = best_weights
print(f"\nOptimal 3-Way Rank Weights:")
print(f"  Ridge:    {w_r:.3f}")
print(f"  LGB-P1:   {w_l:.3f}")
print(f"  CatBoost: {w_c:.3f}")
print(f"  --> Blended OOF AUC: {best_auc:.7f}")
print(f"  --> Lift over best single model (Ridge): +{best_auc - roc_auc_score(y, ridge_oof):.7f}")

# Also test Gaussian CDF (probit) blend
nd = NormalDist()
def probit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.array([nd.inv_cdf(x) for x in p])

def inv_probit(z):
    return np.array([nd.cdf(x) for x in z])

z_ridge_oof = probit(r_ridge_oof)
z_lgb_oof = probit(r_lgb_oof)
z_cb_oof = probit(r_cb_oof)

z_blend_oof = w_r * z_ridge_oof + w_l * z_lgb_oof + w_c * z_cb_oof
z_auc = roc_auc_score(y, z_blend_oof)
print(f"Probit-space Blended OOF AUC: {z_auc:.7f}")

# Generate test predictions
test_rank_blend = w_r * r_ridge_test + w_l * r_lgb_test + w_c * r_cb_test
sub_rank = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_rank_blend})
sub_rank.to_csv('submissions/submission_3way_optimal_rank.csv', index=False)
print("Saved: submissions/submission_3way_optimal_rank.csv")

# Also generate a balanced frontier version (e.g. 75/15/10 and 70/15/15)
for wr, wl, wc, name in [
    (0.80, 0.10, 0.10, 'submission_3way_80_10_10.csv'),
    (0.75, 0.15, 0.10, 'submission_3way_75_15_10.csv'),
    (0.70, 0.15, 0.15, 'submission_3way_70_15_15.csv'),
]:
    blend = wr * r_ridge_test + wl * r_lgb_test + wc * r_cb_test
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': blend})
    sub.to_csv(f'submissions/{name}', index=False)
    oof_score = roc_auc_score(y, wr * r_ridge_oof + wl * r_lgb_oof + wc * r_cb_oof)
    print(f"Saved: submissions/{name} (OOF AUC: {oof_score:.7f})")

print("=" * 70)
print("All 3-way frontier submissions ready!")
print("=" * 70)
