"""
=============================================================================
ULTIMATE STACKER — Mathematical Strategy to Break 0.950 ROC-AUC
=============================================================================

MATHEMATICAL APPROACH:
1. Level-2 Stacked Generalization (Wolpert 1992)
   - Use 4 OOF base models as meta-features
   - Train LightGBM meta-learner in OOF manner (no leakage)
   - AUC-optimal calibration via Platt Scaling

2. Polynomial Feature Expansion (Degree-2 Interactions)
   - Hadamard products of (buy_score, env, sub, med_anx, high_anx, income)
   - Captures synergistic effects: e.g., high_income x high_env = near-certain buyer

3. Isotonic Regression Calibration
   - Monotone mapping of raw ensemble output to calibrated probabilities
   - Preserves rank order, removes systematic bias

4. RBF Gaussian Kernel Expansion of Buy Score
   - Embed buy_score into 8 Gaussian kernels centered at quantiles
   - Gives the meta-learner direct visibility into nonlinear score regions

5. Grid-searched optimal blend weights on OOF
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import rankdata
import time

print("=" * 70, flush=True)
print("ULTIMATE STACKER — Breaking 0.950 ROC-AUC", flush=True)
print("=" * 70, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data & base OOF arrays
# ─────────────────────────────────────────────────────────────────────────────
BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)

print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

oof_lgb  = np.load('oof_predictions.npy')
oof_lad  = np.load('ladder_lgb_oof.npy')
oof_xgb  = np.load('xgb_oof.npy')
oof_cb   = np.load('catboost_oof.npy')

test_lgb = np.load('test_predictions.npy')
test_lad = np.load('ladder_lgb_test_preds.npy')
test_xgb = np.load('xgb_test_preds.npy')
test_cb  = np.load('catboost_test_preds.npy')

print(f"Individual OOF AUCs:", flush=True)
print(f"  LGB-75:   {roc_auc_score(y, oof_lgb):.6f}")
print(f"  Ladder:   {roc_auc_score(y, oof_lad):.6f}")
print(f"  XGBoost:  {roc_auc_score(y, oof_xgb):.6f}")
print(f"  CatBoost: {roc_auc_score(y, oof_cb):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute polynomial + RBF formula features
# ─────────────────────────────────────────────────────────────────────────────
def compute_formula_features(df):
    inc   = df['Annual_Income_USD'].values / 100000.0
    env   = df['Environmental_Concern_Level'].values.astype(float)
    sub   = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med   = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high  = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    cars  = df['Number_of_Cars_Owned'].values.astype(float)
    age   = df['Age'].values.astype(float)
    home  = df['Charging_Stations_Near_Home'].values.astype(float)
    work  = df['Charging_Stations_Near_Work'].values.astype(float)
    commute = df['Daily_Commute_km'].values.astype(float)

    buy_score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med - 3.0 * high

    base = np.column_stack([inc, env, sub, med, high, cars, age, home, work, commute, buy_score])

    # Degree-2 polynomial on primary formula vars
    poly_input = np.column_stack([inc, env, sub, med, high, buy_score])
    pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_feats = pf.fit_transform(poly_input)

    # RBF Gaussian kernel expansion of buy_score (8 centers)
    rbf_centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    rbf_bw = 0.75
    rbf_feats = np.exp(-0.5 * ((buy_score[:, None] - rbf_centers[None, :]) / rbf_bw) ** 2)

    # Cliff / floor / ceiling flags
    cliff_flag = (df['Annual_Income_USD'].values >= 170537).astype(float)
    floor_flag = (buy_score < 0.70260).astype(float)
    ceil_flag  = (buy_score > 7.03543).astype(float)

    # Ratio features
    charger_ratio = (home + work) / (1.0 + cars)
    inc_env = inc * env
    inc_sub = inc * sub
    env_sub = env * sub

    extra = np.column_stack([charger_ratio, inc_env, inc_sub, env_sub,
                             cliff_flag, floor_flag, ceil_flag])

    return np.hstack([base, poly_feats, rbf_feats, extra])

print("\nBuilding polynomial + RBF feature matrix...", flush=True)
F_train = compute_formula_features(train)
F_test  = compute_formula_features(test)
print(f"Formula feature matrix: train={F_train.shape}, test={F_test.shape}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build meta-feature matrix
# ─────────────────────────────────────────────────────────────────────────────
def rank_norm(arr, n):
    return rankdata(arr) / n

oof_geomean = np.power(np.clip(oof_lgb * oof_lad * oof_xgb * oof_cb, 1e-9, None), 0.25)
test_geomean = np.power(np.clip(test_lgb * test_lad * test_xgb * test_cb, 1e-9, None), 0.25)

oof_harmean = 4.0 / (1.0/(oof_lgb+1e-9) + 1.0/(oof_lad+1e-9) + 1.0/(oof_xgb+1e-9) + 1.0/(oof_cb+1e-9))
test_harmean = 4.0 / (1.0/(test_lgb+1e-9) + 1.0/(test_lad+1e-9) + 1.0/(test_xgb+1e-9) + 1.0/(test_cb+1e-9))

oof_prev_best = 0.70*rank_norm(oof_lgb,N_TRAIN) + 0.25*rank_norm(oof_lad,N_TRAIN) + 0.03*rank_norm(oof_xgb,N_TRAIN) + 0.02*rank_norm(oof_cb,N_TRAIN)
test_prev_best = 0.70*rank_norm(test_lgb,N_TEST) + 0.25*rank_norm(test_lad,N_TEST) + 0.03*rank_norm(test_xgb,N_TEST) + 0.02*rank_norm(test_cb,N_TEST)

oof_meta = np.column_stack([
    oof_lgb, oof_lad, oof_xgb, oof_cb,
    rank_norm(oof_lgb, N_TRAIN), rank_norm(oof_lad, N_TRAIN),
    rank_norm(oof_xgb, N_TRAIN), rank_norm(oof_cb, N_TRAIN),
    oof_geomean, oof_harmean, oof_prev_best,
])
test_meta = np.column_stack([
    test_lgb, test_lad, test_xgb, test_cb,
    rank_norm(test_lgb, N_TEST), rank_norm(test_lad, N_TEST),
    rank_norm(test_xgb, N_TEST), rank_norm(test_cb, N_TEST),
    test_geomean, test_harmean, test_prev_best,
])

X_train_stk = np.hstack([oof_meta, F_train])
X_test_stk  = np.hstack([test_meta, F_test])
print(f"Stacked feature matrix: train={X_train_stk.shape}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Level-2 LightGBM Meta-Learner (OOF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STAGE 1] LightGBM Meta-Learner (5-fold OOF)...", flush=True)
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params_meta = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 5,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'random_state': 123,
    'n_jobs': -1,
    'verbose': -1,
}

oof_lgb_stk  = np.zeros(N_TRAIN, dtype=np.float64)
test_lgb_stk = np.zeros(N_TEST,  dtype=np.float64)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(SKF.split(X_train_stk, y)):
    X_tr, y_tr = X_train_stk[tr_idx], y[tr_idx]
    X_va, y_va = X_train_stk[va_idx], y[va_idx]
    tr_ds = lgb.Dataset(X_tr, label=y_tr)
    va_ds = lgb.Dataset(X_va, label=y_va, reference=tr_ds)
    cb_list = [lgb.early_stopping(stopping_rounds=60, verbose=False),
               lgb.log_evaluation(period=500)]
    m = lgb.train(params_meta, tr_ds, num_boost_round=2000,
                  valid_sets=[va_ds], callbacks=cb_list)
    oof_lgb_stk[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
    fold_auc = roc_auc_score(y_va, oof_lgb_stk[va_idx])
    print(f"  Meta-LGB Fold {fold+1}: AUC={fold_auc:.6f}  (best_iter={m.best_iteration})", flush=True)
    test_lgb_stk += m.predict(X_test_stk, num_iteration=m.best_iteration) / 5.0

meta_lgb_auc = roc_auc_score(y, oof_lgb_stk)
print(f"\n*** Meta-LGB OOF AUC: {meta_lgb_auc:.6f} ({time.time()-t0:.1f}s) ***\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Logistic Regression Platt-Scaling Meta-Learner (OOF)
# ─────────────────────────────────────────────────────────────────────────────
print("[STAGE 2] Logistic Regression Platt-Scaling (5-fold OOF)...", flush=True)
oof_lr_stk  = np.zeros(N_TRAIN, dtype=np.float64)
test_lr_stk = np.zeros(N_TEST,  dtype=np.float64)
scaler = StandardScaler()

for fold, (tr_idx, va_idx) in enumerate(SKF.split(X_train_stk, y)):
    X_tr = scaler.fit_transform(X_train_stk[tr_idx])
    X_va = scaler.transform(X_train_stk[va_idx])
    X_te = scaler.transform(X_test_stk)
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_tr, y[tr_idx])
    oof_lr_stk[va_idx] = lr.predict_proba(X_va)[:, 1]
    test_lr_stk += lr.predict_proba(X_te)[:, 1] / 5.0

meta_lr_auc = roc_auc_score(y, oof_lr_stk)
print(f"*** Logistic Regression OOF AUC: {meta_lr_auc:.6f} ***\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Isotonic Regression Calibration
# ─────────────────────────────────────────────────────────────────────────────
print("[STAGE 3] Isotonic Regression Calibration...", flush=True)
oof_iso  = np.zeros(N_TRAIN, dtype=np.float64)
test_iso = np.zeros(N_TEST,  dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(SKF.split(oof_lgb_stk.reshape(-1, 1), y)):
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(oof_lgb_stk[tr_idx], y[tr_idx])
    oof_iso[va_idx] = iso.predict(oof_lgb_stk[va_idx])
    test_iso += iso.predict(test_lgb_stk) / 5.0

meta_iso_auc = roc_auc_score(y, oof_iso)
print(f"*** Isotonic-Calibrated OOF AUC: {meta_iso_auc:.6f} ***\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Grid-search optimal final blend weights
# ─────────────────────────────────────────────────────────────────────────────
print("[STAGE 4] Grid-searching optimal blend weights...", flush=True)
best_auc = 0.0
best_w = (1.0, 0.0, 0.0)
for w1 in np.arange(0.5, 1.01, 0.05):
    for w2 in np.arange(0.0, 0.51, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(0.0, w3)
        blend = w1 * oof_lgb_stk + w2 * oof_lr_stk + w3 * oof_iso
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)

w1, w2, w3 = best_w
print(f"  Best weights: LGB-Meta={w1:.2f}  LR-Platt={w2:.2f}  Isotonic={w3:.2f}")
print(f"  Unclamped OOF AUC: {best_auc:.6f}", flush=True)

oof_final   = w1 * oof_lgb_stk + w2 * oof_lr_stk + w3 * oof_iso
test_final  = w1 * test_lgb_stk + w2 * test_lr_stk + w3 * test_iso

# ─────────────────────────────────────────────────────────────────────────────
# 8. Deterministic Hard Clamping
# ─────────────────────────────────────────────────────────────────────────────
def apply_clamping(preds, df):
    inc_v = df['Annual_Income_USD'].values / 100000.0
    env_v = df['Environmental_Concern_Level'].values.astype(float)
    sub_v = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med_v = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    hi_v  = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    score = 1.2*inc_v + 0.6*env_v + 2.0*sub_v - 1.0*med_v - 3.0*hi_v
    out = preds.copy()
    cliff = df['Annual_Income_USD'].values >= 170537
    floor = score < 0.70260
    ceil  = score > 7.03543
    out[cliff] = 1.0
    out[floor] = 0.0
    out[ceil]  = 1.0
    return out, int(cliff.sum()), int(floor.sum()), int(ceil.sum())

oof_clamped, nc, nf, nh = apply_clamping(oof_final, train)
final_auc = roc_auc_score(y, oof_clamped)
print(f"\n  Clamping: Cliff={nc}, Floor={nf}, Ceiling={nh}", flush=True)
print(f"\n{'='*70}", flush=True)
print(f"  *** ULTIMATE STACKER FINAL OOF ROC-AUC: {final_auc:.6f} ***", flush=True)
print(f"{'='*70}\n", flush=True)

test_clamped, tc, tf, th = apply_clamping(test_final, test)
print(f"Test clamping: Cliff={tc}, Floor={tf}, Ceiling={th}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Save & submit
# ─────────────────────────────────────────────────────────────────────────────
np.save('stacker_oof.npy', oof_clamped)
np.save('stacker_test.npy', test_clamped)

sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_clamped})
out_path = 'submissions/submission_ultimate_stacker.csv'
sub.to_csv(out_path, index=False)
print(f"Saved {out_path}", flush=True)
assert len(sub) == 286571
assert not sub['Will_Buy_EV'].isnull().any()
print("Verification PASSED — 286571 rows, no nulls.")

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(f"  Base LGB-75:          {roc_auc_score(y, oof_lgb):.6f}")
print(f"  Ladder LGB:           {roc_auc_score(y, oof_lad):.6f}")
print(f"  XGBoost:              {roc_auc_score(y, oof_xgb):.6f}")
print(f"  CatBoost:             {roc_auc_score(y, oof_cb):.6f}")
print(f"  Previous Pinnacle:    0.946010")
print(f"  Meta-LGB Stacker:     {meta_lgb_auc:.6f}")
print(f"  LR Platt-Scaling:     {meta_lr_auc:.6f}")
print(f"  Isotonic-Calibrated:  {meta_iso_auc:.6f}")
print(f"  ULTIMATE FINAL:       {final_auc:.6f}  <-- NEW RECORD?")
print("="*70)
