"""
=============================================================================
GMM CONTINUOUS MANIFOLD LIGHTGBM (5-FOLD CROSS VALIDATION)
=============================================================================
Deconstructs CTGAN continuous variables using 10-Component Gaussian Mixtures:
- 10 component posterior probabilities P(mode=k | x)
- Within-mode standardized scalar (x - mu_k) / sigma_k
- Dominant mode assignment
- CTGAN Multi-scale quantization rungs (100, 500, 1000, 5000)
- Digit modulo features
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("TRAINING GMM CONTINUOUS MANIFOLD LIGHTGBM (5-FOLD)", flush=True)
print("=" * 70, flush=True)

BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)

print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

# 1. Feature Engineering
raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
num_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']

print("Building base and GMM features...", flush=True)
t_feat = time.time()

def build_features(df_source, is_train=True):
    X = pd.DataFrame(index=df_source.index)
    for c in num_cols: X[c] = df_source[c]
    for c in raw_cat_cols: X[c] = df_source[c].astype('category').cat.codes

    # Latent Propensity Formula
    inc = df_source['Annual_Income_USD'].values / 100000.0
    env = df_source['Environmental_Concern_Level'].values
    sub = (df_source['Subsidy_Available'] == 'Yes').astype(float).values
    med = (df_source['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high = (df_source['Range_Anxiety_Level'] == 'High').astype(float).values
    X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med - 3.0 * high

    # Quantization rungs
    for r in [100, 500, 1000, 5000]:
        X[f'inc_rung_{r}'] = np.floor(df_source['Annual_Income_USD'] / r) * r
        X[f'inc_mod_{r}']  = df_source['Annual_Income_USD'] % r

    # Digits
    inc_int = df_source['Annual_Income_USD'].astype(int)
    for d in [1, 2, 3]:
        X[f'inc_digit_{d}'] = (inc_int // (10**d)) % 10
    X['commute_digit_0'] = df_source['Daily_Commute_km'].astype(int) % 10
    X['commute_digit_1'] = (df_source['Daily_Commute_km'].astype(int) // 10) % 10

    return X

X_tr = build_features(train, True)
X_te = build_features(test, False)

# Fit GMM on training continuous variables and project to both train and test
for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Age']:
    gmm = GaussianMixture(n_components=10, random_state=42)
    vals_tr = train[[col]].values
    vals_te = test[[col]].values
    gmm.fit(vals_tr)

    probs_tr = gmm.predict_proba(vals_tr)
    probs_te = gmm.predict_proba(vals_te)

    for k in range(10):
        X_tr[f'{col}_gmm_{k}'] = probs_tr[:, k]
        X_te[f'{col}_gmm_{k}'] = probs_te[:, k]

    means = gmm.means_.flatten()
    covs = np.sqrt(gmm.covariances_.flatten())

    dom_tr = probs_tr.argmax(axis=1)
    dom_te = probs_te.argmax(axis=1)

    X_tr[f'{col}_gmm_scalar'] = (vals_tr.flatten() - means[dom_tr]) / (covs[dom_tr] + 1e-6)
    X_te[f'{col}_gmm_scalar'] = (vals_te.flatten() - means[dom_te]) / (covs[dom_te] + 1e-6)
    X_tr[f'{col}_gmm_mode'] = dom_tr
    X_te[f'{col}_gmm_mode'] = dom_te

print(f"Features created: {X_tr.shape[1]} in {time.time()-t_feat:.1f}s", flush=True)

# 2. 5-Fold LightGBM Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_gmm = np.zeros(N_TRAIN, dtype=np.float64)
test_gmm = np.zeros(N_TEST, dtype=np.float64)

params = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'learning_rate':     0.03,
    'max_depth':         5,
    'num_leaves':        127,
    'colsample_bytree':  0.4,
    'subsample':         0.8,
    'min_child_samples': 20,
    'n_jobs':            16,
    'verbose':           -1
}

fold_scores = []
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
    X_t, y_t = X_tr.iloc[tr_idx], y[tr_idx]
    X_v, y_v = X_tr.iloc[va_idx], y[va_idx]

    ds_tr = lgb.Dataset(X_t, label=y_t)
    ds_va = lgb.Dataset(X_v, label=y_v, reference=ds_tr)

    cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    m = lgb.train({**params, 'random_state': 42 + fold}, ds_tr,
                  num_boost_round=1500, valid_sets=[ds_va], callbacks=cbs)

    oof_gmm[va_idx] = m.predict(X_v, num_iteration=m.best_iteration)
    test_gmm += m.predict(X_te, num_iteration=m.best_iteration) / 5.0

    score = roc_auc_score(y_v, oof_gmm[va_idx])
    fold_scores.append(score)
    print(f"  Fold {fold+1}: AUC = {score:.6f} (iter={m.best_iteration})", flush=True)

total_gmm_auc = roc_auc_score(y, oof_gmm)
print("\n" + "=" * 70, flush=True)
print(f"GMM-LGB 5-Fold OOF AUC: {total_gmm_auc:.6f} (Elapsed: {time.time()-t0:.1f}s)", flush=True)
print("=" * 70, flush=True)

np.save('gmm_lgb_oof.npy', oof_gmm)
np.save('gmm_lgb_test.npy', test_gmm)
print("Saved gmm_lgb_oof.npy and gmm_lgb_test.npy!", flush=True)
