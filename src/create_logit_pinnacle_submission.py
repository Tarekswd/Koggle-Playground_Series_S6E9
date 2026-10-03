"""
=============================================================================
BAYESIAN LOG-ODDS (LOGIT) META-BLEND & SUBMISSION GENERATOR
=============================================================================
Combines:
1. LightGBM 75-Feature (Dual Target Encoded)
2. Quantization Ladder LightGBM (CTGAN Rungs 100/500/1000/5000)
3. XGBoost 5-Fold Hist
4. CatBoost 5-Fold Symmetric
in Logit (Log-Odds) space, followed by Sigmoid projection and Deterministic Clamping.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import time

print("=" * 70, flush=True)
print("BAYESIAN LOGIT META-BLEND PIPELINE", flush=True)
print("=" * 70, flush=True)

BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)

print(f"Train samples: {N_TRAIN:,} | Test samples: {N_TEST:,}", flush=True)

# 1. Load All OOF and Test Predictions
p_lgb    = np.load('oof_predictions.npy')
t_lgb    = np.load('test_predictions.npy')
p_ladder = np.load('ladder_lgb_oof.npy')
t_ladder = np.load('ladder_lgb_test_preds.npy')
p_xgb    = np.load('xgb_oof.npy')
t_xgb    = np.load('xgb_test_preds.npy')
p_cb     = np.load('catboost_oof.npy')
t_cb     = np.load('catboost_test_preds.npy')

print("\nIndividual Base Model OOF AUCs:", flush=True)
print(f"  LightGBM 75-Feat:       {roc_auc_score(y, p_lgb):.6f}", flush=True)
print(f"  Quantization Ladder:    {roc_auc_score(y, p_ladder):.6f}", flush=True)
print(f"  CatBoost 5-Fold:        {roc_auc_score(y, p_cb):.6f}", flush=True)
print(f"  XGBoost 5-Fold:         {roc_auc_score(y, p_xgb):.6f}", flush=True)

# 2. Conversion to Logit Space
def to_odds(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

z_lgb    = to_odds(p_lgb)
z_ladder = to_odds(p_ladder)
z_xgb    = to_odds(p_xgb)
z_cb     = to_odds(p_cb)

zt_lgb    = to_odds(t_lgb)
zt_ladder = to_odds(t_ladder)
zt_xgb    = to_odds(t_xgb)
zt_cb     = to_odds(t_cb)

# 3. Deterministic Clamping
def clamp(preds, df):
    i = df['Annual_Income_USD'].values / 100000.0
    e = df['Environmental_Concern_Level'].values.astype(float)
    s = (df['Subsidy_Available'] == 'Yes').astype(float).values
    m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    sc = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h
    out = preds.copy()
    out[df['Annual_Income_USD'].values >= 170537] = 1.0
    out[sc < 0.70260] = 0.0
    out[sc > 7.03543] = 1.0
    return out

# 4. Systematic Fine-Grained Search for Optimal Logit Weights
print("\n[OPTIMIZATION] Searching optimal Bayesian Logit fusion weights...", flush=True)
t0 = time.time()
best_auc = 0.0
best_w = None

for w_lgb in np.linspace(0.60, 0.80, 9):
    for w_lad in np.linspace(0.15, 0.35, 9):
        for w_x in [0.0, 0.02, 0.04]:
            w_c = 1.0 - w_lgb - w_lad - w_x
            if w_c < -1e-5: continue
            w_c = max(0.0, w_c)
            z_blend = w_lgb * z_lgb + w_lad * z_ladder + w_x * z_xgb + w_c * z_cb
            p_blend = expit(z_blend)
            p_clamped = clamp(p_blend, train)
            score = roc_auc_score(y, p_clamped)
            if score > best_auc:
                best_auc = score
                best_w = (w_lgb, w_lad, w_x, w_c)

print(f"Search complete in {time.time()-t0:.1f}s!", flush=True)
print(f"Optimal Logit Weights:", flush=True)
print(f"  LightGBM 75-Feat:    {best_w[0]:.3f}", flush=True)
print(f"  Quantization Ladder: {best_w[1]:.3f}", flush=True)
print(f"  XGBoost:             {best_w[2]:.3f}", flush=True)
print(f"  CatBoost:            {best_w[3]:.3f}", flush=True)

print("\n" + "=" * 70, flush=True)
print(f"*** FINAL PINNACLE CLAMPED LOGIT OOF AUC: {best_auc:.6f} ***", flush=True)
print(f"    Previous Pinnacle (linear rank):       0.946010", flush=True)
print(f"    Net Gain:                             +{best_auc - 0.946010:.6f}", flush=True)
print("=" * 70, flush=True)

# 5. Build Final Test Predictions
z_test_final = (best_w[0] * zt_lgb +
                best_w[1] * zt_ladder +
                best_w[2] * zt_xgb +
                best_w[3] * zt_cb)
p_test_final = expit(z_test_final)
p_test_clamped = clamp(p_test_final, test)

# Save arrays
np.save('bayesian_logit_pinnacle_oof.npy', clamp(expit(best_w[0]*z_lgb + best_w[1]*z_ladder + best_w[2]*z_xgb + best_w[3]*z_cb), train))
np.save('bayesian_logit_pinnacle_test.npy', p_test_clamped)

# Save submission CSV
out_path = 'submissions/submission_bayesian_logit_pinnacle.csv'
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': p_test_clamped})
sub.to_csv(out_path, index=False)
print(f"\nSaved submission to: {out_path}", flush=True)

# Sanity checks
assert len(sub) == 286571, f"Expected 286571 rows, got {len(sub)}"
assert not sub['Will_Buy_EV'].isnull().any(), "Found NaNs in submission!"
assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all(), "Values outside [0, 1]!"
print("Verification PASSED: 286,571 rows, zero NaNs, valid probabilities.", flush=True)
