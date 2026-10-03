"""
=============================================================================
TRI-BRIDGED BAYESIAN LOGIT PINNACLE META-ENSEMBLE (OOF AUC: 0.946052)
=============================================================================
Combines:
1. 70% LightGBM 75-Feature Baseline (Dual Target Encoded, Digit Decomposed)
2. 15% Quantization Ladder LightGBM (CTGAN Rungs 100/500/1000/5000)
3. 15% Optuna 144-Feature Bayesian LightGBM (Expanded Manifold Encoding)
in Bayesian Log-Odds (Logit) Evidence Fusion space + Deterministic Clamping.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import time

print("=" * 70, flush=True)
print("TRI-BRIDGED BAYESIAN LOGIT PINNACLE GENERATOR", flush=True)
print("=" * 70, flush=True)

BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)

print(f"Loaded Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

# 1. Load the 3 Complementary Models
p_lgb    = np.load('oof_predictions.npy')
t_lgb    = np.load('test_predictions.npy')
p_ladder = np.load('ladder_lgb_oof.npy')
t_ladder = np.load('ladder_lgb_test_preds.npy')
p_optuna = np.load('optuna_lgb_oof.npy')
t_optuna = np.load('optuna_lgb_test.npy')

print("\nModel Component Standalone AUCs:", flush=True)
print(f"  [1] LightGBM 75-Feat Baseline:     {roc_auc_score(y, p_lgb):.6f}", flush=True)
print(f"  [2] Quantization Ladder LightGBM:  {roc_auc_score(y, p_ladder):.6f}", flush=True)
print(f"  [3] Optuna 144-Feat LightGBM:      {roc_auc_score(y, p_optuna):.6f}", flush=True)

# 2. Convert to Log-Odds Space
def to_odds(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

z_lgb    = to_odds(p_lgb)
z_ladder = to_odds(p_ladder)
z_optuna = to_odds(p_optuna)

zt_lgb    = to_odds(t_lgb)
zt_ladder = to_odds(t_ladder)
zt_optuna = to_odds(t_optuna)

# 3. Deterministic Boundary Clamping
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

# 4. Tri-Bridged Logit Fusion
w_lgb, w_ladder, w_optuna = 0.70, 0.15, 0.15
print(f"\nFusion Weights: {w_lgb:.2f}*LGB75 + {w_ladder:.2f}*Ladder + {w_optuna:.2f}*Optuna", flush=True)

z_oof_final = w_lgb * z_lgb + w_ladder * z_ladder + w_optuna * z_optuna
p_oof_final = expit(z_oof_final)
p_oof_clamped = clamp(p_oof_final, train)
final_auc = roc_auc_score(y, p_oof_clamped)

print("\n" + "=" * 70, flush=True)
print(f"*** TRI-BRIDGED PINNACLE OOF AUC: {final_auc:.6f} ***", flush=True)
print(f"    Previous Pinnacle (linear rank): 0.946010  (LB: 0.94608)", flush=True)
print(f"    Net Gain:                       +{final_auc - 0.946010:.6f}", flush=True)
print("=" * 70, flush=True)

# 5. Build Test Predictions
z_test_final = w_lgb * zt_lgb + w_ladder * zt_ladder + w_optuna * zt_optuna
p_test_final = expit(z_test_final)
p_test_clamped = clamp(p_test_final, test)

# Save numpy predictions
np.save('tribridged_logit_pinnacle_oof.npy', p_oof_clamped)
np.save('tribridged_logit_pinnacle_test.npy', p_test_clamped)

# Save submission CSV
out_path = 'submissions/submission_tribridged_logit_pinnacle.csv'
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': p_test_clamped})
sub.to_csv(out_path, index=False)
print(f"\nSaved submission to: {out_path}", flush=True)

# Validation checks
assert len(sub) == 286571, f"Expected 286571 rows, got {len(sub)}"
assert not sub['Will_Buy_EV'].isnull().any(), "Found NaNs!"
assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all(), "Probabilities outside [0, 1]!"
print(f"Clamped zeros: {(sub['Will_Buy_EV'] == 0.0).sum():,}")
print(f"Clamped ones:  {(sub['Will_Buy_EV'] == 1.0).sum():,}")
print("Verification PASSED: File is ready for immediate Kaggle submission.", flush=True)
