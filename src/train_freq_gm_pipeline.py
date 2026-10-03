"""
=============================================================================
FREQ-ENC GRANDMASTER: TRANSDUCTIVE 75-FEATURE PIPELINE
=============================================================================
The KEY INSIGHT from Phase 1 diagnosis:
  - Our fleet engines (27 features) got +0.0005 from freq-enc, topping at 0.9443
  - The workspace stacker is already at 0.9461 — 0.0018 higher
  - Adding freq-enc to SMALL fleet wasn't enough to bridge the gap

The SOLUTION: Add transductive freq-enc to the FULL 75-FEATURE grandmaster
pipeline. The existing `train_grandmaster_75_ensemble.py` uses freq maps from
lgb_ev_model.joblib which are TRAIN-ONLY. We replace these with:
  combined = pd.concat([train, test]) → TRUE TRANSDUCTIVE freq maps

Expected individual engine AUC: 0.9460+ (vs current 0.9458)
Expected blend AUC after stacking with workspace engines: 0.9463+

Saves:
  lgb_freq_gm_oof.npy / lgb_freq_gm_test.npy   (LGB result)
  xgb_freq_gm_oof.npy / xgb_freq_gm_test.npy   (XGB result)
  submissions/submission_freq_gm_blend.csv
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


def main():
    print("=" * 80, flush=True)
    print("FREQ-ENC GRANDMASTER: TRANSDUCTIVE 75-FEATURE PIPELINE", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN, N_TEST = len(train), len(test)
    print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

    # ── Load buddy pipeline for its 75-feature scaffold ──────────────────────
    d = joblib.load('lgb_ev_model.joblib')
    cols_reference = d['feature_columns_per_fold'][0]
    raw_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    print(f"Buddy feature columns: {len(cols_reference)}", flush=True)

    # ── Build TRANSDUCTIVE frequency maps (train + test combined) ─────────────
    print("\nBuilding TRANSDUCTIVE frequency maps (train+test)...", flush=True)
    train_feats = train.drop(columns=['Will_Buy_EV', 'id'], errors='ignore')
    test_feats  = test.drop(columns=['id'], errors='ignore')
    combined    = pd.concat([train_feats, test_feats], ignore_index=True)

    transductive_freq_maps = {}
    for c in raw_cols:
        transductive_freq_maps[c] = combined[c].astype(str).value_counts(normalize=True).to_dict()

    # Extra digit-level transductive freq maps
    combined_inc = combined['Annual_Income_USD'].astype(int)
    transductive_freq_maps['inc_digit0']  = (combined_inc % 10).value_counts(normalize=True).to_dict()
    transductive_freq_maps['inc_mod100']  = (combined_inc % 100).value_counts(normalize=True).to_dict()
    transductive_freq_maps['inc_mod1000'] = (combined_inc % 1000).value_counts(normalize=True).to_dict()

    # Cross-feature freq map targeting the error hotspot
    combined['_sub_ra'] = combined['Subsidy_Available'].astype(str) + '__' + combined['Range_Anxiety_Level'].astype(str)
    transductive_freq_maps['sub_x_ra'] = combined['_sub_ra'].value_counts(normalize=True).to_dict()

    print(f"  Built {len(transductive_freq_maps)} transductive freq maps", flush=True)

    # ── Joint interaction pairs ───────────────────────────────────────────────
    interaction_pairs = [
        ('Subsidy_Available', 'Range_Anxiety_Level'),
        ('Subsidy_Available', 'City_Type'),
        ('Home_Charging_Possible', 'City_Type'),
        ('Subsidy_Available', 'Environmental_Concern_Level')
    ]
    global_prior = y.mean()
    SMOOTHING = 20.0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(train, y))

    # ── Feature builder with TRANSDUCTIVE freq maps ───────────────────────────
    def build_features_transductive(df, fold_idx, tr_idx_for_te):
        """
        Builds the same 75 buddy features but replaces train-only freq maps
        with TRANSDUCTIVE (train+test) freq maps.
        """
        X = pd.DataFrame(index=df.index)

        for c in raw_cols:
            if df[c].dtype == 'object':
                X[c] = df[c].astype('category').cat.codes
            else:
                X[c] = df[c]

        # Digit decompositions (same as buddy)
        X['Age_digit0'] = df['Age'] % 10
        X['Age_digit1'] = (df['Age'] // 10) % 10

        inc_int = df['Annual_Income_USD'].astype(int)
        X['Annual_Income_USD_digit0'] = inc_int % 10
        X['Annual_Income_USD_digit1'] = (inc_int // 10) % 10
        X['Annual_Income_USD_digit2'] = (inc_int // 100) % 10
        X['Annual_Income_USD_digit3'] = (inc_int // 1000) % 10

        commute_float = df['Daily_Commute_km']
        X['Daily_Commute_km_digit-4'] = (np.round(commute_float * 10000).astype(int)) % 10
        X['Daily_Commute_km_digit-3'] = (np.round(commute_float * 1000).astype(int)) % 10
        X['Daily_Commute_km_digit-1'] = (np.round(commute_float * 10).astype(int)) % 10
        X['Daily_Commute_km_digit0']  = (commute_float.astype(int)) % 10
        X['Daily_Commute_km_digit1']  = (commute_float.astype(int) // 10) % 10

        st_home = df['Charging_Stations_Near_Home']
        X['Charging_Stations_Near_Home_digit-4'] = 0
        X['Charging_Stations_Near_Home_digit0']  = st_home % 10
        X['Charging_Stations_Near_Home_digit1']  = (st_home // 10) % 10

        st_work = df['Charging_Stations_Near_Work']
        X['Charging_Stations_Near_Work_digit-4'] = 0
        X['Charging_Stations_Near_Work_digit0']  = st_work % 10
        X['Charging_Stations_Near_Work_digit1']  = (st_work // 10) % 10

        for cat in ['Gender', 'City_Type', 'Current_Car_Type',
                    'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']:
            X[f'{cat}_digit-4'] = 0

        # ── KEY CHANGE: Use TRANSDUCTIVE freq maps instead of buddy's train-only ──
        for c in raw_cols:
            X[f'{c}_freq'] = df[c].astype(str).map(transductive_freq_maps[c]).fillna(0.0).values

        # Fold-specific target encodings from buddy (same as before — uses fold tr_idx)
        encoders = d['encoders_per_fold'][fold_idx]
        te10   = encoders[0].transform(df[raw_cols])
        te_auto = encoders[1].transform(df[raw_cols])
        for col_name in encoders[0].get_feature_names_out():
            orig_c = col_name.split('__')[0]
            X[f"{orig_c}__te_mean_seed_42_smooth_10_inner_n_fold_5"] = te10[col_name].values
        for col_name in encoders[1].get_feature_names_out():
            orig_c = col_name.split('__')[0]
            X[f"{orig_c}__te_mean_seed_42_smooth_auto_inner_n_fold_5"] = te_auto[col_name].values

        # ── Extra transductive features not in buddy ───────────────────────────
        inc_raw = df['Annual_Income_USD'].astype(int)
        X['inc_digit0_freq']  = inc_raw.mod(10).map(transductive_freq_maps['inc_digit0']).fillna(0.0).values
        X['inc_mod100_freq']  = inc_raw.mod(100).map(transductive_freq_maps['inc_mod100']).fillna(0.0).values
        X['inc_mod1000_freq'] = inc_raw.mod(1000).map(transductive_freq_maps['inc_mod1000']).fillna(0.0).values
        sub_ra = df['Subsidy_Available'].astype(str) + '__' + df['Range_Anxiety_Level'].astype(str)
        X['subsidy_rangeanx_freq'] = sub_ra.map(transductive_freq_maps['sub_x_ra']).fillna(0.0).values

        # Restrict to buddy reference cols + our new 4 extra cols
        existing = [c for c in cols_reference if c in X.columns]
        extra_cols = ['inc_digit0_freq', 'inc_mod100_freq', 'inc_mod1000_freq', 'subsidy_rangeanx_freq']
        return X[existing + extra_cols]

    # ── Build fold matrices ───────────────────────────────────────────────────
    print("\n[STEP 1] Building fold feature matrices with transductive freq...", flush=True)
    t_feat = time.time()
    train_folds = []
    test_folds  = []

    for fold in range(5):
        print(f"  Fold {fold+1}/5...", flush=True)
        tr_idx, val_idx = splits[fold]
        X_tr = build_features_transductive(train, fold, tr_idx)
        X_te = build_features_transductive(test,  fold, tr_idx)

        # Joint interaction TEs (fold-specific, no leakage)
        for col1, col2 in interaction_pairs:
            feat_name = f"te_joint_{col1}_x_{col2}"
            pair_tr   = train.iloc[tr_idx][col1].astype(str) + '__' + train.iloc[tr_idx][col2].astype(str)
            pair_all  = train[col1].astype(str) + '__' + train[col2].astype(str)
            pair_te   = test[col1].astype(str) + '__' + test[col2].astype(str)
            stats     = pd.DataFrame({'pair': pair_tr, 'target': y[tr_idx]}).groupby('pair').agg(['count', 'mean'])['target']
            smap      = ((stats['count'] * stats['mean'] + SMOOTHING * global_prior) / (stats['count'] + SMOOTHING)).to_dict()
            X_tr[feat_name] = pair_all.map(smap).fillna(global_prior).astype(np.float32).values
            X_te[feat_name] = pair_te.map(smap).fillna(global_prior).astype(np.float32).values

        train_folds.append(X_tr)
        test_folds.append(X_te)

    n_feats = train_folds[0].shape[1]
    print(f"Done in {time.time()-t_feat:.1f}s. Features per fold: {n_feats}", flush=True)

    # ── [STEP 2] LightGBM with transductive freq ──────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 2] LIGHTGBM (MULTI-SEED: 42, 101, 2024)", flush=True)
    print("=" * 80, flush=True)

    lgb_hyperparams = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.02, 'max_depth': 5, 'num_leaves': 247,
        'min_child_samples': 10, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
        'colsample_bytree': 0.3029300829885024, 'reg_alpha': 0.07094285437903122,
        'reg_lambda': 2.0330390977032424, 'subsample': 0.812763123433567,
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
                    eval_set=[(train_folds[fold].iloc[val_idx], y[val_idx])],
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
    print(f"\n  MULTI-SEED LGB (3 seeds) OOF AUC: {lgb_auc:.6f}", flush=True)
    np.save('lgb_freq_gm_oof.npy', lgb_oof)
    np.save('lgb_freq_gm_test.npy', lgb_test)

    # ── [STEP 3] Hist-XGBoost with transductive freq ──────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 3] HIST-XGBOOST", flush=True)
    print("=" * 80, flush=True)

    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8,
        'colsample_bytree': 0.35, 'nthread': 16, 'random_state': 42
    }
    xgb_oof  = np.zeros(N_TRAIN)
    xgb_test = np.zeros(N_TEST)
    t_xgb = time.time()
    for fold, (tr_idx, val_idx) in enumerate(splits):
        dtrain = xgb.DMatrix(train_folds[fold].iloc[tr_idx], label=y[tr_idx])
        dval   = xgb.DMatrix(train_folds[fold].iloc[val_idx], label=y[val_idx])
        dtest  = xgb.DMatrix(test_folds[fold])
        bst = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                        evals=[(dval, 'val')], early_stopping_rounds=80, verbose_eval=False)
        xgb_oof[val_idx] = bst.predict(dval)
        xgb_test += bst.predict(dtest) / 5.0
        print(f"  Fold {fold+1} done", flush=True)
    xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"  XGB OOF AUC: {xgb_auc:.6f} ({time.time()-t_xgb:.1f}s)", flush=True)
    np.save('xgb_freq_gm_oof.npy', xgb_oof)
    np.save('xgb_freq_gm_test.npy', xgb_test)

    # ── [STEP 4] Stack all models ─────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 4] META-STACKING: Freq-GM + Workspace Engines", flush=True)
    print("=" * 80, flush=True)

    all_engines = {
        'lgb_freq_gm':  (lgb_oof, lgb_test),
        'xgb_freq_gm':  (xgb_oof, xgb_test),
        'ws_stacker':   (np.load('stacking_logit_pinnacle_oof.npy'), np.load('stacking_logit_pinnacle_test.npy')),
        'ws_lgb75':     (np.load('oof_predictions.npy'),             np.load('test_predictions.npy')),
        'ws_xgb_meta':  (np.load('xgb_elefante_meta_oof.npy'),       np.load('xgb_elefante_meta_test.npy')),
        'ws_lgb_white': (np.load('lgb75_multiseed_whitened_oof.npy'),'lgb75_multiseed_whitened_test.npy'),
        'ws_optuna':    (np.load('optuna_lgb_oof.npy'),              np.load('optuna_lgb_test.npy')),
        'ws_ladder':    (np.load('ladder_lgb_oof.npy'),              np.load('ladder_lgb_test_preds.npy')),
    }
    # Fix: load numpy arrays for string paths
    for k in list(all_engines.keys()):
        oo, tt = all_engines[k]
        if isinstance(tt, str):
            all_engines[k] = (oo, np.load(tt))

    def to_odds(p): return logit(np.clip(p, 1e-6, 1.0-1e-6))

    for name, (oo, _) in all_engines.items():
        print(f"  {name:25s}: {roc_auc_score(y, oo):.6f}", flush=True)

    Z_tr = np.column_stack([to_odds(all_engines[k][0]) for k in all_engines])
    Z_te = np.column_stack([to_odds(all_engines[k][1]) for k in all_engines])

    meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
    meta.fit(Z_tr, y)
    final_oof  = clamp(meta.predict_proba(Z_tr)[:, 1], train)
    final_test = clamp(meta.predict_proba(Z_te)[:, 1], test)
    final_auc  = roc_auc_score(y, final_oof)

    print("\n" + "*" * 80, flush=True)
    print(f"*** FREQ-GM BLEND FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous best (Grand Prix Fleet): 0.946072", flush=True)
    print(f"    Net OOF Advance:                  +{final_auc - 0.946072:.6f}", flush=True)
    print(f"    Expected LB:                      ~{0.94621 + (final_auc - 0.946072):.5f}", flush=True)
    print("*" * 80, flush=True)

    out_csv = 'submissions/submission_freq_gm_blend.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test})
    sub.to_csv(out_csv, index=False)
    assert len(sub) == N_TEST
    assert not sub['Will_Buy_EV'].isnull().any()
    print(f"\nSaved: {out_csv}", flush=True)
    print(f"Verification PASSED! Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
