import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.special import logit, expit

print("=== OPTIMIZING ADVANCED ENSEMBLE & RANK BLEND ===")

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

# 1. Base components
oof_lgb = np.load('oof_predictions.npy')
oof_xgb = np.load('xgb_oof.npy')

# Compute Chris Deotte Buy Score for train
inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values
score_raw = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

print(f"LGB OOF AUC: {roc_auc_score(y, oof_lgb):.6f}")
print(f"XGB OOF AUC: {roc_auc_score(y, oof_xgb):.6f}")
print(f"Formula OOF AUC: {roc_auc_score(y, score_raw):.6f}")

# Convert to normalized ranks (0 to 1)
rank_lgb = rankdata(oof_lgb) / len(y)
rank_xgb = rankdata(oof_xgb) / len(y)
rank_form = rankdata(score_raw) / len(y)

print("\n--- 1. Testing Linear Probability Blends ---")
best_p_auc = 0.0
best_p_weights = None
for w_lgb in np.linspace(0.70, 0.98, 15):
    for w_xgb in np.linspace(0.0, 1.0 - w_lgb, 10):
        w_form = 1.0 - w_lgb - w_xgb
        if w_form < 0:
            continue
        p = w_lgb * oof_lgb + w_xgb * oof_xgb + w_form * (rank_form)
        auc = roc_auc_score(y, p)
        if auc > best_p_auc:
            best_p_auc = auc
            best_p_weights = (w_lgb, w_xgb, w_form)
print(f"Best Prob Blend: LGB={best_p_weights[0]:.3f}, XGB={best_p_weights[1]:.3f}, Form={best_p_weights[2]:.3f} -> AUC: {best_p_auc:.6f}")

print("\n--- 2. Testing Rank Averaging Blends ---")
best_r_auc = 0.0
best_r_weights = None
for w_lgb in np.linspace(0.70, 0.98, 15):
    for w_xgb in np.linspace(0.0, 1.0 - w_lgb, 10):
        w_form = 1.0 - w_lgb - w_xgb
        if w_form < 0:
            continue
        r = w_lgb * rank_lgb + w_xgb * rank_xgb + w_form * rank_form
        auc = roc_auc_score(y, r)
        if auc > best_r_auc:
            best_r_auc = auc
            best_r_weights = (w_lgb, w_xgb, w_form)
print(f"Best Rank Blend: LGB={best_r_weights[0]:.3f}, XGB={best_r_weights[1]:.3f}, Form={best_r_weights[2]:.3f} -> AUC: {best_r_auc:.6f}")

print("\n--- 3. Testing Power-Rank (Power-Gmean) Blends ---")
best_pow_auc = 0.0
best_pow_params = None
eps = 1e-7
for p in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    for w_xgb in [0.05, 0.10, 0.15, 0.20]:
        for w_form in [0.0, 0.02, 0.05]:
            w_lgb = 1.0 - w_xgb - w_form
            # Power weighted sum
            pw = w_lgb * (rank_lgb ** p) + w_xgb * (rank_xgb ** p) + w_form * (rank_form ** p)
            auc = roc_auc_score(y, pw)
            if auc > best_pow_auc:
                best_pow_auc = auc
                best_pow_params = (p, w_lgb, w_xgb, w_form)
print(f"Best Power-Rank: power={best_pow_params[0]}, LGB={best_pow_params[1]:.3f}, XGB={best_pow_params[2]:.3f}, Form={best_pow_params[3]:.3f} -> AUC: {best_pow_auc:.6f}")

print("\n--- 4. Testing Exact Clamping on Best Blend ---")
# Take the best predictions
w_lgb, w_xgb, w_form = best_p_weights
pred = w_lgb * oof_lgb + w_xgb * oof_xgb + w_form * rank_form
base_auc = roc_auc_score(y, pred)
print(f"Unclamped AUC: {base_auc:.6f}")

# Millionaire Cliff: income >= 170537
cliff_mask = (train['Annual_Income_USD'].values >= 170537)
print(f"Cliff rows: {cliff_mask.sum()}, positive rate: {y[cliff_mask].mean():.5f}")

# Zero floor: score < 0.70260
floor_mask = (score_raw < 0.70260)
print(f"Floor rows: {floor_mask.sum()}, positive rate: {y[floor_mask].mean():.5f}")

# Ceiling: score > 7.03543
ceil_mask = (score_raw > 7.03543)
print(f"Ceil rows: {ceil_mask.sum()}, positive rate: {y[ceil_mask].mean():.5f}")

pred_clamped = pred.copy()
pred_clamped[cliff_mask] = 1.0
pred_clamped[ceil_mask] = 1.0
pred_clamped[floor_mask] = 0.0

clamped_auc = roc_auc_score(y, pred_clamped)
print(f"Clamped AUC: {clamped_auc:.6f} (Delta: {clamped_auc - base_auc:+.6f})")
