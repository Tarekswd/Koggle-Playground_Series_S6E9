"""
=============================================================================
EXACT 75-FEATURE LIGHTGBM MULTI-SEED VARIANCE WHITENING
=============================================================================
Uses the exact 75-feature representation, fold splits, and hyperparameter tokens
from lgb_ev_model.joblib (which scored 0.945832 OOF alone, Fold 3 at 0.946804).

Trains 2 additional distinct seeds (Seed 101, Seed 777) on the identical 5 folds:
- Same out-of-fold target encoding transformations
- Varying random_state, feature_fraction_seed, and bagging_seed
- Cancels Monte Carlo tree-subsample variance like 1/sqrt(K)
- Averages all seeds in log-odds space
=============================================================================
"""

import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit
import time
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70, flush=True)
    print("EXACT 75-FEATURE LIGHTGBM MULTI-SEED VARIANCE WHITENING", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)

    print(f"Train: {N_TRAIN:,} rows | Test: {N_TEST:,} rows", flush=True)

    # 1. Load baseline specification
    d = joblib.load('lgb_ev_model.joblib')
    freq_maps = d['frequency_maps']
    cols_f0 = d['feature_columns_per_fold'][0]
    base_params = d['params_lgb']

    print(f"Loaded lgb_ev_model.joblib: {len(cols_f0)} features per fold", flush=True)

    raw_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

    def build_75_features(df, fold_idx):
        X = pd.DataFrame(index=df.index)

        # 1. Base 13 features
        for c in raw_cols:
            if df[c].dtype == 'object':
                X[c] = df[c].astype('category').cat.codes
            else:
                X[c] = df[c]

        # 2. Digit features
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

        # 3. Frequency features
        for c in raw_cols:
            X[f'{c}_freq'] = df[c].astype(str).map(freq_maps[c]).fillna(0.0).values

        # 4. Target encodings
        encoders = d['encoders_per_fold'][fold_idx]
        te10 = encoders[0].transform(df[raw_cols])
        for orig_c, col_name in zip(raw_cols, encoders[0].get_feature_names_out()):
            target_col = f"{orig_c}__te_mean_seed_42_smooth_10_inner_n_fold_5"
            X[target_col] = te10[col_name].values

        te_auto = encoders[1].transform(df[raw_cols])
        for orig_c, col_name in zip(raw_cols, encoders[1].get_feature_names_out()):
            target_col = f"{orig_c}__te_mean_seed_42_smooth_auto_inner_n_fold_5"
            X[target_col] = te_auto[col_name].values

        # Align exactly with model columns
        cols = d['feature_columns_per_fold'][fold_idx]
        return X[cols]

    print("Pre-building 5 fold feature representations for Train & Test...", flush=True)
    t_feat = time.time()
    train_feat_folds = [build_75_features(train, f) for f in range(5)]
    test_feat_folds  = [build_75_features(test, f) for f in range(5)]
    print(f"Features built in {time.time()-t_feat:.1f}s!", flush=True)

    # Load existing baseline seed 60 (from joblib)
    oof_seed_orig = np.load('oof_predictions.npy')
    test_seed_orig = np.load('test_predictions.npy')
    print(f"Seed 60 (Original) OOF AUC: {roc_auc_score(y, oof_seed_orig):.6f}", flush=True)

    seeds = [101, 777]
    all_oof_seeds = [oof_seed_orig]
    all_test_seeds = [test_seed_orig]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(train, y))

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---", flush=True)
        t_seed = time.time()
        oof_seed = np.zeros(N_TRAIN, dtype=np.float64)
        test_seed = np.zeros(N_TEST, dtype=np.float64)

        for fold, (tr_idx, val_idx) in enumerate(fold_splits):
            t_f = time.time()
            X_tr = train_feat_folds[fold].iloc[tr_idx]
            y_tr = y[tr_idx]
            X_val = train_feat_folds[fold].iloc[val_idx]
            y_val = y[val_idx]
            X_te = test_feat_folds[fold]

            cur_params = dict(base_params)
            cur_params.update({
                'random_state': seed + fold,
                'feature_fraction_seed': seed + fold * 7,
                'bagging_seed': seed + fold * 13,
                'n_jobs': 16,
                'verbose': -1
            })

            clf = lgb.LGBMClassifier(**cur_params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

            val_preds = clf.predict_proba(X_val)[:, 1]
            oof_seed[val_idx] = val_preds
            test_seed += clf.predict_proba(X_te)[:, 1] / 5.0

            fold_auc = roc_auc_score(y_val, val_preds)
            print(f"  Fold {fold+1} AUC: {fold_auc:.6f} (best_iter={clf.best_iteration_}, time={time.time()-t_f:.1f}s)", flush=True)

        seed_auc = roc_auc_score(y, oof_seed)
        print(f"Seed {seed} Overall OOF AUC: {seed_auc:.6f} ({time.time()-t_seed:.1f}s)", flush=True)
        all_oof_seeds.append(oof_seed)
        all_test_seeds.append(test_seed)

    # 3. Variance Whitening in Log-Odds Space
    def to_odds(p):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        return logit(p_c)

    z_oof_whitened = np.mean([to_odds(p) for p in all_oof_seeds], axis=0)
    z_test_whitened = np.mean([to_odds(p) for p in all_test_seeds], axis=0)

    p_oof_whitened = expit(z_oof_whitened)
    p_test_whitened = expit(z_test_whitened)

    whitened_auc = roc_auc_score(y, p_oof_whitened)
    print("\n" + "=" * 70, flush=True)
    print(f"*** 3-SEED VARIANCE-WHITENED LIGHTGBM OOF AUC: {whitened_auc:.6f} ***", flush=True)
    print(f"    Single Seed Baseline AUC:                   {roc_auc_score(y, oof_seed_orig):.6f}", flush=True)
    print(f"    Net Variance Reduction Gain:                +{whitened_auc - roc_auc_score(y, oof_seed_orig):.6f}", flush=True)
    print("=" * 70 + "\n", flush=True)

    np.save('lgb75_multiseed_whitened_oof.npy', p_oof_whitened)
    np.save('lgb75_multiseed_whitened_test.npy', p_test_whitened)
    print("Saved lgb75_multiseed_whitened_oof.npy and lgb75_multiseed_whitened_test.npy!", flush=True)
    print(f"Total script execution time: {time.time()-t_start:.1f}s", flush=True)

if __name__ == '__main__':
    main()
