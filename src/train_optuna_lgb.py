"""
=============================================================================
FAST OPTUNA BAYESIAN HPO — LightGBM with 144-Feature Expanded Set
=============================================================================
Changes vs slow version:
- gbdt ONLY (no dart — dart has no early stopping, takes forever)
- 500 max rounds per trial with 40-round early stopping
- 20 trials (enough for TPE convergence)
- Parallel n_jobs=-1 tree fitting
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("FAST OPTUNA HPO — 20 trials, gbdt, 500 rounds max", flush=True)
print("=" * 70, flush=True)

BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)
print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Build 144-feature expanded set
# ─────────────────────────────────────────────────────────────────────────────
def build_base_features(df):
    X = pd.DataFrame(index=df.index)
    inc_raw = df['Annual_Income_USD'].values
    inc   = inc_raw / 100000.0
    env   = df['Environmental_Concern_Level'].values.astype(float)
    sub   = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med   = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high  = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    no_anx = (df['Range_Anxiety_Level'] == 'None').astype(float).values
    cars  = df['Number_of_Cars_Owned'].values.astype(float)
    age   = df['Age'].values.astype(float)
    home  = df['Charging_Stations_Near_Home'].values.astype(float)
    work  = df['Charging_Stations_Near_Work'].values.astype(float)
    commute = df['Daily_Commute_km'].values.astype(float)

    buy_score = 1.2*inc + 0.6*env + 2.0*sub - 1.0*med - 3.0*high

    # Raw numerics
    X['Annual_Income_USD'] = inc_raw
    X['Age'] = age
    X['Daily_Commute_km'] = commute
    X['Number_of_Cars_Owned'] = cars
    X['Charging_Stations_Near_Home'] = home
    X['Charging_Stations_Near_Work'] = work
    X['Environmental_Concern_Level'] = env

    # Categorical codes
    for c in ['Gender','City_Type','Current_Car_Type','Home_Charging_Possible',
              'Subsidy_Available','Range_Anxiety_Level']:
        X[c] = df[c].astype('category').cat.codes

    # Buy score and polynomial
    X['buy_score']      = buy_score
    X['buy_score_2']    = buy_score ** 2
    X['buy_score_3']    = buy_score ** 3
    X['buy_score_sqrt'] = np.sqrt(np.clip(buy_score, 0, None))

    # Income transforms
    X['inc_normalized'] = inc
    X['inc_log']  = np.log1p(inc_raw)
    X['inc_sqrt'] = np.sqrt(inc_raw)
    X['inc_sq']   = inc ** 2

    # Boundary flags
    X['cliff_flag']   = (inc_raw >= 170537).astype(np.int8)
    X['floor_flag']   = (buy_score < 0.70260).astype(np.int8)
    X['ceiling_flag'] = (buy_score > 7.03543).astype(np.int8)

    # Interaction features
    X['inc_env']     = inc * env
    X['inc_sub']     = inc * sub
    X['env_sub']     = env * sub
    X['inc_no_anx']  = inc * no_anx
    X['env_no_anx']  = env * no_anx
    X['inc_env_sub'] = inc * env * sub
    X['charger_sum'] = home + work
    X['charger_ratio']      = (home + work) / (cars + 1.0)
    X['income_per_commute'] = inc_raw / (commute + 1.0)
    X['env_per_car']        = env / (cars + 1.0)

    # Digit features
    for d in [1, 2, 3, 4, 5]:
        X[f'inc_digit_{d}'] = (df['Annual_Income_USD'] // (10**d)) % 10
    X['inc_rem100']  = df['Annual_Income_USD'] % 100
    X['inc_rem500']  = df['Annual_Income_USD'] % 500
    X['inc_rem1000'] = df['Annual_Income_USD'] % 1000
    X['commute_int']     = commute.astype(int)
    X['commute_digit_0'] = commute.astype(int) % 10
    X['commute_digit_1'] = (commute.astype(int) // 10) % 10
    X['commute_dec_1']   = (commute * 10).astype(int) % 10

    # Quantization ladder rungs
    X['inc_q100']  = df['Annual_Income_USD'] // 100
    X['inc_q500']  = df['Annual_Income_USD'] // 500
    X['inc_q1000'] = df['Annual_Income_USD'] // 1000
    X['inc_q5000'] = df['Annual_Income_USD'] // 5000
    X['commute_q1'] = commute.astype(int)
    X['commute_q5'] = (commute // 5).astype(int)
    X['age_q5']  = (age // 5).astype(int)
    X['age_q10'] = (age // 10).astype(int)

    # Frequency encodings
    for c in ['inc_q100','inc_q1000','commute_q1','Annual_Income_USD','Daily_Commute_km','Age']:
        freq = X[c].value_counts(normalize=True)
        X[f'{c}_freq'] = X[c].map(freq).values

    return X

print("Building base features...", flush=True)
X_base_tr = build_base_features(train)
X_base_te = build_base_features(test)
print(f"Base feature matrix: {X_base_tr.shape}", flush=True)

# OOF Target Encodings: 14 cols × 6 smoothing values
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()

te_cols = ['Gender','City_Type','Current_Car_Type','Home_Charging_Possible',
           'Subsidy_Available','Range_Anxiety_Level',
           'inc_q100','inc_q500','inc_q1000','inc_q5000',
           'commute_q1','commute_q5','age_q5','age_q10']
smoothing_vals = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

oof_te_tr = pd.DataFrame(index=train.index)
te_te     = pd.DataFrame(index=test.index)
for c in te_cols:
    for s in smoothing_vals:
        col = f'{c}_te_s{int(s)}'
        oof_te_tr[col] = 0.0
        te_te[col]     = 0.0

print(f"Computing {len(te_cols)*len(smoothing_vals)} OOF target encodings...", flush=True)
for fold, (tr_idx, va_idx) in enumerate(SKF.split(train, y)):
    y_tr = y[tr_idx]
    for c in te_cols:
        tr_s = X_base_tr.iloc[tr_idx][c]
        va_s = X_base_tr.iloc[va_idx][c]
        te_s = X_base_te[c]
        stats = pd.DataFrame({'key': tr_s, 'y': y_tr}).groupby('key')['y'].agg(['count','mean'])
        for s in smoothing_vals:
            col = f'{c}_te_s{int(s)}'
            sm = (stats['count']*stats['mean'] + s*global_mean) / (stats['count'] + s)
            oof_te_tr.loc[va_idx, col] = va_s.map(sm).fillna(global_mean).values
            te_te[col] += te_s.map(sm).fillna(global_mean).values / 5.0
    print(f"  TE fold {fold+1}/5 done", flush=True)

X_train_full = pd.concat([X_base_tr, oof_te_tr], axis=1)
X_test_full  = pd.concat([X_base_te, te_te], axis=1)
print(f"Final expanded feature matrix: {X_train_full.shape}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fast Optuna: gbdt only, 500 rounds, early_stopping=40
# ─────────────────────────────────────────────────────────────────────────────
print("\n[OPTUNA] Starting fast Bayesian HPO — 20 trials...", flush=True)
TRIAL_START = time.time()

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42,
    }

    fold_aucs = []
    for tr_idx, va_idx in SKF.split(X_train_full, y):
        X_tr, y_tr = X_train_full.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train_full.iloc[va_idx], y[va_idx]
        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
        cbs = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
        m = lgb.train(params, ds_tr, num_boost_round=500,
                      valid_sets=[ds_va], callbacks=cbs)
        preds = m.predict(X_va, num_iteration=m.best_iteration)
        fold_aucs.append(roc_auc_score(y_va, preds))

    mean_auc = float(np.mean(fold_aucs))
    tn = trial.number
    elapsed = time.time() - TRIAL_START
    print(f"  Trial {tn+1:02d}: AUC={mean_auc:.6f}  elapsed={elapsed:.0f}s", flush=True)
    return mean_auc

sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=20, show_progress_bar=False)

print(f"\n[OPTUNA] Done! Best OOF AUC = {study.best_value:.6f}", flush=True)
print(f"[OPTUNA] Best params:", flush=True)
for k, v in study.best_params.items():
    print(f"  {k}: {v}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Final retrain with best params — more rounds (1500) and tighter early stop
# ─────────────────────────────────────────────────────────────────────────────
print("\n[FINAL] Retraining with best params — 1500 rounds, early_stop=80...", flush=True)

best_p = dict(study.best_params)
best_p.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'subsample_freq': 1,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
})

oof_optuna  = np.zeros(N_TRAIN, dtype=np.float64)
test_optuna = np.zeros(N_TEST,  dtype=np.float64)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(SKF.split(X_train_full, y)):
    X_tr, y_tr = X_train_full.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X_train_full.iloc[va_idx], y[va_idx]
    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
    cbs = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)]
    m = lgb.train(best_p, ds_tr, num_boost_round=1500,
                  valid_sets=[ds_va], callbacks=cbs)
    oof_optuna[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
    fold_auc = roc_auc_score(y_va, oof_optuna[va_idx])
    print(f"  Final Fold {fold+1}: {fold_auc:.6f}  best_iter={m.best_iteration}", flush=True)
    test_optuna += m.predict(X_test_full, num_iteration=m.best_iteration) / 5.0

raw_auc = roc_auc_score(y, oof_optuna)
print(f"\n*** OPTUNA LGB OOF AUC (raw): {raw_auc:.6f} in {time.time()-t0:.1f}s ***", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Clamping
# ─────────────────────────────────────────────────────────────────────────────
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

oof_clamped  = clamp(oof_optuna, train)
test_clamped = clamp(test_optuna, test)
final_auc = roc_auc_score(y, oof_clamped)

print(f"\n{'='*70}", flush=True)
print(f"  *** OPTUNA FINAL OOF AUC (clamped): {final_auc:.6f} ***", flush=True)
print(f"{'='*70}\n", flush=True)

np.save('optuna_lgb_oof.npy', oof_clamped)
np.save('optuna_lgb_test.npy', test_clamped)

sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_clamped})
out_path = 'submissions/submission_optuna_lgb.csv'
sub.to_csv(out_path, index=False)
print(f"Saved {out_path}", flush=True)
assert len(sub) == 286571
assert not sub['Will_Buy_EV'].isnull().any()
print("Verification PASSED — 286571 rows, no nulls.", flush=True)

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"  Previous best (Pinnacle): 0.946010")
print(f"  Optuna best trial AUC:    {study.best_value:.6f}")
print(f"  OPTUNA FINAL (clamped):   {final_auc:.6f}  <-- NEW RECORD?")
print("="*70)
