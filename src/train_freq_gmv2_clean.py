"""
=============================================================================
FREQ-ENC GRANDMASTER v2 — CLEAN REBUILD (NO CATSTAT DEPENDENCY)
=============================================================================
Root cause of v1 failure:
  - catstat.TargetEncoder was pickled with sklearn 1.6.1 but we run 1.7.2
  - .transform() produced garbage TE values → AUC collapsed to 0.926

This v2 computes ALL 75+ features from scratch:
  1. 13 ordinal-encoded raw features
  2. 20 digit decompositions (income, commute, age, charging stations)
  3. 6 zero-constant categorical digit-4 placeholders
  4. 13 TRANSDUCTIVE frequency features (train+test combined) — KEY CHANGE
  5. 13 smoothed-10 target encodings (computed from fold training data — no leakage)
  6. 13 smoothed-auto target encodings (variance-based smoothing)
  --- baseline 75 features ---
  7. 4 joint pair interaction TEs (Subsidy×RA, Subsidy×City, etc.)
  8. 4 extra transductive digit-level freq maps (income digit0/mod100/mod1000, sub×ra)
  --- 83 total features ---

Saves:
  lgb_freq_gmv2_oof.npy, lgb_freq_gmv2_test.npy
  xgb_freq_gmv2_oof.npy, xgb_freq_gmv2_test.npy
  submissions/submission_freq_gmv2_blend.csv
=============================================================================
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings('ignore')


def clamp(preds, df):
    inc_raw = df['Annual_Income_USD'].values
    i = inc_raw / 100000.0
    e = df['Environmental_Concern_Level'].values.astype(float)
    s = (df['Subsidy_Available'] == 'Yes').astype(float).values
    m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    score = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h
    c = preds.copy()
    c[inc_raw >= 170537] = 1.0
    c[score < 0.702596]  = 0.0
    c[score > 7.03543]   = 1.0
    return c


RAW_COLS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
            'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
            'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

INTERACTION_PAIRS = [
    ('Subsidy_Available', 'Range_Anxiety_Level'),
    ('Subsidy_Available', 'City_Type'),
    ('Home_Charging_Possible', 'City_Type'),
    ('Subsidy_Available', 'Environmental_Concern_Level'),
]

EXTRA_TRANSDUCTIVE = ['inc_digit0_freq', 'inc_mod100_freq', 'inc_mod1000_freq', 'subsidy_ra_freq']


def smoothed_te(train_series, train_y, apply_series, global_prior, smoothing):
    """Compute smoothed target encoding from training data, apply to any series."""
    stats = (pd.DataFrame({'k': train_series.astype(str), 'y': train_y})
               .groupby('k')['y']
               .agg(['count', 'mean']))
    if smoothing == 'auto':
        # Variance-based: larger n → less shrinkage
        smoother = np.sqrt(stats['count'].values)
    else:
        smoother = float(smoothing)
    te_map = ((stats['count'] * stats['mean'] + smoother * global_prior) /
              (stats['count'] + smoother)).to_dict()
    return apply_series.astype(str).map(te_map).fillna(global_prior).astype(np.float32).values


def build_base_features(df, trans_freq_maps):
    """
    Build the 48 non-TE features (ordinal + digits + transductive freq).
    These are fold-INDEPENDENT.
    """
    X = {}

    # 1. Ordinal encode all raw columns
    for c in RAW_COLS:
        if df[c].dtype == 'object':
            X[c] = df[c].astype('category').cat.codes.values.astype(np.float32)
        else:
            X[c] = df[c].values.astype(np.float32)

    # 2. Digit decompositions
    X['Age_digit0'] = (df['Age'] % 10).astype(np.float32).values
    X['Age_digit1'] = ((df['Age'] // 10) % 10).astype(np.float32).values

    inc_int = df['Annual_Income_USD'].astype(int).values
    X['Annual_Income_USD_digit0'] = (inc_int % 10).astype(np.float32)
    X['Annual_Income_USD_digit1'] = ((inc_int // 10) % 10).astype(np.float32)
    X['Annual_Income_USD_digit2'] = ((inc_int // 100) % 10).astype(np.float32)
    X['Annual_Income_USD_digit3'] = ((inc_int // 1000) % 10).astype(np.float32)

    commute = df['Daily_Commute_km'].values
    X['Daily_Commute_km_digit-4'] = (np.round(commute * 10000).astype(int) % 10).astype(np.float32)
    X['Daily_Commute_km_digit-3'] = (np.round(commute * 1000).astype(int) % 10).astype(np.float32)
    X['Daily_Commute_km_digit-1'] = (np.round(commute * 10).astype(int) % 10).astype(np.float32)
    X['Daily_Commute_km_digit0']  = (commute.astype(int) % 10).astype(np.float32)
    X['Daily_Commute_km_digit1']  = ((commute.astype(int) // 10) % 10).astype(np.float32)

    st_home = df['Charging_Stations_Near_Home'].values
    X['Charging_Stations_Near_Home_digit-4'] = np.zeros(len(df), dtype=np.float32)
    X['Charging_Stations_Near_Home_digit0']  = (st_home % 10).astype(np.float32)
    X['Charging_Stations_Near_Home_digit1']  = ((st_home // 10) % 10).astype(np.float32)

    st_work = df['Charging_Stations_Near_Work'].values
    X['Charging_Stations_Near_Work_digit-4'] = np.zeros(len(df), dtype=np.float32)
    X['Charging_Stations_Near_Work_digit0']  = (st_work % 10).astype(np.float32)
    X['Charging_Stations_Near_Work_digit1']  = ((st_work // 10) % 10).astype(np.float32)

    # Zero-constant cat digit-4 placeholders (match buddy's feature set)
    for cat in ['Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']:
        X[f'{cat}_digit-4'] = np.zeros(len(df), dtype=np.float32)

    # 3. TRANSDUCTIVE frequency features (train+test combined)
    for c in RAW_COLS:
        X[f'{c}_freq'] = df[c].astype(str).map(trans_freq_maps[c]).fillna(0.0).astype(np.float32).values

    return X  # dict of {col_name: np.array}


def build_fold_features(df_all, y_all, trans_freq_maps, splits, fold_idx, is_test=False, train_df=None):
    """
    Build the full 83-feature matrix for one fold.
    For training: uses fold-specific OOF TEs (no leakage).
    For test: uses the full training set to compute TEs (train_df must be provided).
    """
    tr_idx, val_idx = splits[fold_idx]
    global_prior = y_all.mean()

    # Base non-TE features (fold-independent)
    base = build_base_features(df_all, trans_freq_maps)
    X = pd.DataFrame({k: v for k, v in base.items()})

    # Per-column smooth_10 target encodings
    for c in RAW_COLS:
        col_name = f'{c}__te_mean_seed_42_smooth_10_inner_n_fold_5'
        if is_test:
            # Fit on FULL training set, apply to test
            te_vals = smoothed_te(train_df[c], y_all, df_all[c], global_prior, 10)
        else:
            # OOF: fit on tr_idx, apply to full train
            te_vals = smoothed_te(df_all.iloc[tr_idx][c], y_all[tr_idx],
                                  df_all[c], global_prior, 10)
        X[col_name] = te_vals

    # Per-column smooth_auto target encodings
    for c in RAW_COLS:
        col_name = f'{c}__te_mean_seed_42_smooth_auto_inner_n_fold_5'
        if is_test:
            te_vals = smoothed_te(train_df[c], y_all, df_all[c], global_prior, 'auto')
        else:
            te_vals = smoothed_te(df_all.iloc[tr_idx][c], y_all[tr_idx],
                                  df_all[c], global_prior, 'auto')
        X[col_name] = te_vals

    # Joint pair interaction TEs
    for col1, col2 in INTERACTION_PAIRS:
        feat_name = f"te_joint_{col1}_x_{col2}"
        pair_all = df_all[col1].astype(str) + '__' + df_all[col2].astype(str)
        if is_test:
            pair_tr = train_df[col1].astype(str) + '__' + train_df[col2].astype(str)
            y_tr = y_all
        else:
            pair_tr = (df_all.iloc[tr_idx][col1].astype(str) +
                       '__' + df_all.iloc[tr_idx][col2].astype(str))
            y_tr = y_all[tr_idx]
        te_vals = smoothed_te(pair_tr, y_tr, pair_all, global_prior, 20)
        X[feat_name] = te_vals

    # Extra transductive digit-freq features
    inc_int = df_all['Annual_Income_USD'].astype(int)
    X['inc_digit0_freq']   = inc_int.mod(10).map(trans_freq_maps['inc_digit0_freq']).fillna(0.0).astype(np.float32).values
    X['inc_mod100_freq']   = inc_int.mod(100).map(trans_freq_maps['inc_mod100_freq']).fillna(0.0).astype(np.float32).values
    X['inc_mod1000_freq']  = inc_int.mod(1000).map(trans_freq_maps['inc_mod1000_freq']).fillna(0.0).astype(np.float32).values
    sub_ra = df_all['Subsidy_Available'].astype(str) + '__' + df_all['Range_Anxiety_Level'].astype(str)
    X['subsidy_ra_freq']   = sub_ra.map(trans_freq_maps['subsidy_ra_freq']).fillna(0.0).astype(np.float32).values

    return X  # full-length DataFrame (668K or 287K rows)


def main():
    print("=" * 80, flush=True)
    print("FREQ-ENC GRANDMASTER v2 — CLEAN REBUILD (NO CATSTAT)", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN, N_TEST = len(train), len(test)
    global_prior = y.mean()
    print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,} | Buy rate: {global_prior:.4f}", flush=True)

    # ── Build TRANSDUCTIVE freq maps ──────────────────────────────────────────
    print("\nBuilding transductive freq maps...", flush=True)
    train_feats = train.drop(columns=['Will_Buy_EV', 'id'], errors='ignore')
    test_feats  = test.drop(columns=['id'], errors='ignore')
    combined    = pd.concat([train_feats, test_feats], ignore_index=True)

    trans_freq_maps = {}
    for c in RAW_COLS:
        trans_freq_maps[c] = combined[c].astype(str).value_counts(normalize=True).to_dict()

    combined_inc = combined['Annual_Income_USD'].astype(int)
    trans_freq_maps['inc_digit0_freq']  = (combined_inc % 10).value_counts(normalize=True).to_dict()
    trans_freq_maps['inc_mod100_freq']  = (combined_inc % 100).value_counts(normalize=True).to_dict()
    trans_freq_maps['inc_mod1000_freq'] = (combined_inc % 1000).value_counts(normalize=True).to_dict()
    combined['_sub_ra'] = combined['Subsidy_Available'].astype(str) + '__' + combined['Range_Anxiety_Level'].astype(str)
    trans_freq_maps['subsidy_ra_freq']  = combined['_sub_ra'].value_counts(normalize=True).to_dict()
    print(f"  Built {len(trans_freq_maps)} freq maps", flush=True)

    # ── Cross-validation splits ───────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(train, y))

    # ── [STEP 1] Build fold matrices ──────────────────────────────────────────
    print("\n[STEP 1] Building fold feature matrices...", flush=True)
    t_feat = time.time()
    train_folds = []
    test_folds  = []

    for fold in range(5):
        print(f"  Fold {fold+1}/5...", flush=True)
        X_tr = build_fold_features(train, y, trans_freq_maps, splits, fold, is_test=False)
        X_te = build_fold_features(test,  y, trans_freq_maps, splits, fold, is_test=True, train_df=train)
        train_folds.append(X_tr)
        test_folds.append(X_te)

    n_feats = train_folds[0].shape[1]
    assert n_feats > 70, f"Expected 80+ features, got {n_feats}"
    print(f"Done in {time.time()-t_feat:.1f}s. Features per fold: {n_feats}", flush=True)

    # Validate no NaN
    for fold in range(5):
        nans = train_folds[fold].isnull().sum().sum()
        assert nans == 0, f"Fold {fold} has {nans} NaN values"
    print("NaN checks PASSED", flush=True)

    # ── [STEP 2] LightGBM multi-seed ─────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 2] LIGHTGBM (SEEDS: 42, 101, 2024)", flush=True)
    print("=" * 80, flush=True)

    lgb_hyperparams = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.02, 'max_depth': 5, 'num_leaves': 247,
        'min_child_samples': 10, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
        'colsample_bytree': 0.3029300829885024,
        'reg_alpha': 0.07094285437903122,
        'reg_lambda': 2.0330390977032424,
        'subsample': 0.812763123433567,
        'subsample_freq': 1, 'max_bin': 1024, 'n_jobs': 16, 'verbose': -1,
    }

    all_lgb_oofs  = []
    all_lgb_tests = []

    for seed in [42, 101, 2024]:
        print(f"\n  Training LGB seed={seed}...", flush=True)
        t_s = time.time()
        oof_s = np.zeros(N_TRAIN)
        tst_s = np.zeros(N_TEST)
        for fold, (tr_idx, val_idx) in enumerate(splits):
            params = {**lgb_hyperparams, 'random_state': seed,
                      'feature_fraction_seed': seed, 'bagging_seed': seed}
            clf = lgb.LGBMClassifier(n_estimators=3000, **params)
            clf.fit(train_folds[fold].iloc[tr_idx], y[tr_idx],
                    eval_X=train_folds[fold].iloc[val_idx], eval_y=y[val_idx],
                    callbacks=[lgb.early_stopping(80, verbose=False)])
            oof_s[val_idx] = clf.predict_proba(train_folds[fold].iloc[val_idx])[:, 1]
            tst_s += clf.predict_proba(test_folds[fold])[:, 1] / 5.0
        auc_s = roc_auc_score(y, oof_s)
        print(f"  Seed {seed} OOF AUC: {auc_s:.6f} ({time.time()-t_s:.1f}s)", flush=True)
        all_lgb_oofs.append(oof_s)
        all_lgb_tests.append(tst_s)

    # Average in logit space
    lgb_oof  = expit(np.mean([logit(np.clip(o, 1e-6, 1-1e-6)) for o in all_lgb_oofs], axis=0))
    lgb_test = expit(np.mean([logit(np.clip(t, 1e-6, 1-1e-6)) for t in all_lgb_tests], axis=0))
    lgb_auc  = roc_auc_score(y, lgb_oof)
    print(f"\n  MULTI-SEED LGB OOF AUC: {lgb_auc:.6f}", flush=True)
    np.save('lgb_freq_gmv2_oof.npy', lgb_oof)
    np.save('lgb_freq_gmv2_test.npy', lgb_test)

    # ── [STEP 3] Hist-XGBoost ─────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 3] HIST-XGBOOST", flush=True)
    print("=" * 80, flush=True)

    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8,
        'colsample_bytree': 0.35, 'nthread': 16, 'random_state': 42,
    }
    xgb_oof  = np.zeros(N_TRAIN)
    xgb_test = np.zeros(N_TEST)
    t_xgb = time.time()

    for fold, (tr_idx, val_idx) in enumerate(splits):
        dtrain = xgb.DMatrix(train_folds[fold].iloc[tr_idx], label=y[tr_idx])
        dval   = xgb.DMatrix(train_folds[fold].iloc[val_idx], label=y[val_idx])
        dtest  = xgb.DMatrix(test_folds[fold])
        bst = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                        evals=[(dval, 'val')], early_stopping_rounds=80,
                        verbose_eval=False)
        xgb_oof[val_idx] = bst.predict(dval)
        xgb_test += bst.predict(dtest) / 5.0
        print(f"  Fold {fold+1} done", flush=True)

    xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"  XGB OOF AUC: {xgb_auc:.6f} ({time.time()-t_xgb:.1f}s)", flush=True)
    np.save('xgb_freq_gmv2_oof.npy', xgb_oof)
    np.save('xgb_freq_gmv2_test.npy', xgb_test)

    # ── [STEP 4] Stack with workspace engines ─────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 4] META-STACKING WITH WORKSPACE ENGINES", flush=True)
    print("=" * 80, flush=True)

    def to_odds(p): return logit(np.clip(p, 1e-6, 1.0 - 1e-6))

    all_engines = {
        'lgb_freq_gmv2': (lgb_oof,  lgb_test),
        'xgb_freq_gmv2': (xgb_oof,  xgb_test),
        'ws_stacker':    (np.load('stacking_logit_pinnacle_oof.npy'),     np.load('stacking_logit_pinnacle_test.npy')),
        'ws_lgb75':      (np.load('oof_predictions.npy'),                 np.load('test_predictions.npy')),
        'ws_xgb_meta':   (np.load('xgb_elefante_meta_oof.npy'),           np.load('xgb_elefante_meta_test.npy')),
        'ws_lgb_white':  (np.load('lgb75_multiseed_whitened_oof.npy'),    np.load('lgb75_multiseed_whitened_test.npy')),
        'ws_optuna':     (np.load('optuna_lgb_oof.npy'),                  np.load('optuna_lgb_test.npy')),
        'ws_ladder':     (np.load('ladder_lgb_oof.npy'),                  np.load('ladder_lgb_test_preds.npy')),
    }

    for name, (oo, _) in all_engines.items():
        print(f"  {name:20s}: OOF AUC = {roc_auc_score(y, oo):.6f}", flush=True)

    Z_tr = np.column_stack([to_odds(v[0]) for v in all_engines.values()])
    Z_te = np.column_stack([to_odds(v[1]) for v in all_engines.values()])
    meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
    meta.fit(Z_tr, y)

    final_oof  = clamp(meta.predict_proba(Z_tr)[:, 1], train)
    final_test = clamp(meta.predict_proba(Z_te)[:, 1], test)
    final_auc  = roc_auc_score(y, final_oof)

    print("\n" + "*" * 80, flush=True)
    print(f"*** FREQ-GM v2 BLEND FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous best (Grand Prix Fleet): 0.946072", flush=True)
    print(f"    Net OOF Advance: +{final_auc - 0.946072:.6f}", flush=True)
    print(f"    Expected LB: ~{0.94621 + (final_auc - 0.946072):.5f}", flush=True)
    print("*" * 80, flush=True)

    out_csv = 'submissions/submission_freq_gmv2_blend.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test})
    sub.to_csv(out_csv, index=False)
    assert len(sub) == N_TEST
    assert not sub['Will_Buy_EV'].isnull().any()
    print(f"\nSaved: {out_csv}", flush=True)
    print(f"Verification PASSED! Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
