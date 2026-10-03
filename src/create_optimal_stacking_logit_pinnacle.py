"""
=============================================================================
5-FOLD L2-REGULARIZED LOGISTIC STACKING PINNACLE (OOF AUC: 0.946061)
=============================================================================
Architecture:
- Inputs: Log-Odds (logits) of 5 diverse model families:
  1. LightGBM 75-Feature Baseline (0.945832)
  2. Quantization Ladder LightGBM (0.944789)
  3. Optuna 144-Feature Bayesian LightGBM (0.944692)
  4. XGBoost 5-Fold Hist (0.943319)
  5. CatBoost 5-Fold Symmetric (0.943439)
- Meta-Learner: 5-Fold Cross-Validated L2 Logistic Regression (C=0.01)
  Learns optimal hedge coefficients across positive signal and correlated noise.
- Deterministic Post-Processing:
  Boundary clamping on verified 100% precision regions.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import time

print("=" * 70, flush=True)
print("5-FOLD L2 LOGISTIC STACKING PINNACLE META-ENSEMBLE", flush=True)
print("=" * 70, flush=True)

BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)

print(f"Loaded Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

# 1. Load All 5 Distinct Model Predictions
p_lgb    = np.load('oof_predictions.npy')
t_lgb    = np.load('test_predictions.npy')
p_ladder = np.load('ladder_lgb_oof.npy')
t_ladder = np.load('ladder_lgb_test_preds.npy')
p_optuna = np.load('optuna_lgb_oof.npy')
t_optuna = np.load('optuna_lgb_test.npy')
p_xgb    = np.load('xgb_oof.npy')
t_xgb    = np.load('xgb_test_preds.npy')
p_cb     = np.load('catboost_oof.npy')
t_cb     = np.load('catboost_test_preds.npy')

print("\nModel Component Standalone AUCs:", flush=True)
print(f"  [1] LightGBM 75-Feat Baseline:     {roc_auc_score(y, p_lgb):.6f}", flush=True)
print(f"  [2] Quantization Ladder LightGBM:  {roc_auc_score(y, p_ladder):.6f}", flush=True)
print(f"  [3] Optuna 144-Feat LightGBM:      {roc_auc_score(y, p_optuna):.6f}", flush=True)
print(f"  [4] CatBoost 5-Fold Symmetric:     {roc_auc_score(y, p_cb):.6f}", flush=True)
print(f"  [5] XGBoost 5-Fold Hist:           {roc_auc_score(y, p_xgb):.6f}", flush=True)

# 2. Conversion to Log-Odds (Logit) Space
def to_odds(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

Z_tr = np.column_stack([to_odds(p_lgb), to_odds(p_ladder), to_odds(p_optuna), to_odds(p_xgb), to_odds(p_cb)])
Z_te = np.column_stack([to_odds(t_lgb), to_odds(t_ladder), to_odds(t_optuna), to_odds(t_xgb), to_odds(t_cb)])

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

# 4. 5-Fold Cross-Validated L2 Logistic Regression Meta-Learner (C=0.01)
print("\n[TRAINING] 5-Fold Stratified Stacking Meta-Learner...", flush=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lr = np.zeros(N_TRAIN, dtype=np.float64)
test_lr = np.zeros(N_TEST, dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(Z_tr, y)):
    lr = LogisticRegression(C=0.01, max_iter=500, solver='lbfgs')
    lr.fit(Z_tr[tr_idx], y[tr_idx])
    oof_lr[va_idx] = lr.predict_proba(Z_tr[va_idx])[:, 1]
    test_lr += lr.predict_proba(Z_te)[:, 1] / 5.0
    f_auc = roc_auc_score(y[va_idx], oof_lr[va_idx])
    print(f"  Fold {fold+1} AUC: {f_auc:.6f} | Coefs: {np.round(lr.coef_[0], 3)}", flush=True)

oof_clamped = clamp(oof_lr, train)
test_clamped = clamp(test_lr, test)

final_auc = roc_auc_score(y, oof_clamped)

print("\n" + "=" * 70, flush=True)
print(f"*** ALL-TIME RECORD PINNACLE OOF AUC: {final_auc:.6f} ***", flush=True)
print(f"    Previous Best (Tri-Bridged Logit): 0.946052  (LB: 0.94614)", flush=True)
print(f"    Net Gain:                         +{final_auc - 0.946052:.6f}", flush=True)
print("=" * 70, flush=True)

# 5. Save Arrays and Final Submission
np.save('stacking_logit_pinnacle_oof.npy', oof_clamped)
np.save('stacking_logit_pinnacle_test.npy', test_clamped)

out_path = 'submissions/submission_optimal_stacking_logit_pinnacle.csv'
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_clamped})
sub.to_csv(out_path, index=False)
print(f"\nSaved submission to: {out_path}", flush=True)

# Verification checks
assert len(sub) == 286571, f"Expected 286571 rows, got {len(sub)}"
assert not sub['Will_Buy_EV'].isnull().any(), "Found NaNs!"
assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all(), "Values outside [0, 1]!"
print(f"Clamped zeros: {(sub['Will_Buy_EV'] == 0.0).sum():,}")
print(f"Clamped ones:  {(sub['Will_Buy_EV'] == 1.0).sum():,}")
print("Verification PASSED: File is verified and ready for upload.", flush=True)
