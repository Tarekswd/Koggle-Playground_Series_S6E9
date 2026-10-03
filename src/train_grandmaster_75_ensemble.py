"""
=============================================================================
GRANDMASTER 75-FEATURE MULTI-MODEL ENSEMBLE PIPELINE
=============================================================================
1. Pre-builds 5-fold feature matrices with:
   - Exact 75 buddy features (digits, frequency, smooth_10, smooth_auto)
   - 4 High-leverage joint interaction target encodings:
     * Subsidy_Available x Range_Anxiety_Level
     * Subsidy_Available x City_Type
     * Home_Charging_Possible x City_Type
     * Subsidy_Available x Environmental_Concern_Level
2. Trains Multi-Seed LightGBM (Seeds 101, 2024) to bag with Seed 42
3. Trains Full 5-Fold Hist-XGBoost on 75+ features
4. Trains Full 5-Fold CatBoost on 75+ features
5. Stacks all diverse models in Logit (Log-Odds) space with L2 Regularization
6. Enforces exact deterministic boundary manifolds
7. Generates submissions/submission_grandmaster_75_pinnacle.csv
=============================================================================
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

def main():
    print("=" * 80, flush=True)
    print("STARTING GRANDMASTER 75-FEATURE MULTI-MODEL ENSEMBLE PIPELINE", flush=True)
    print("=" * 80, flush=True)

    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)
    print(f"Loaded Train: {N_TRAIN:,} rows | Test: {N_TEST:,} rows", flush=True)

    # 1. Load buddy reference specification
    d = joblib.load('lgb_ev_model.joblib')
    freq_maps = d['frequency_maps']
    cols_reference = d['feature_columns_per_fold'][0]
    raw_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

    print(f"Base buddy feature count: {len(cols_reference)}", flush=True)

    # 2. Joint interaction definitions
    interaction_pairs = [
        ('Subsidy_Available', 'Range_Anxiety_Level'),
        ('Subsidy_Available', 'City_Type'),
        ('Home_Charging_Possible', 'City_Type'),
        ('Subsidy_Available', 'Environmental_Concern_Level')
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(train, y))

    def build_base_75(df, fold_idx):
        X = pd.DataFrame(index=df.index)
        for c in raw_cols:
            if df[c].dtype == 'object':
                X[c] = df[c].astype('category').cat.codes
            else:
                X[c] = df[c]

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

        for c in raw_cols:
            fmap = freq_maps[c]
            X[f'{c}_freq'] = df[c].astype(str).map(fmap).fillna(0.0).values

        encoders = d['encoders_per_fold'][fold_idx]
        te10   = encoders[0].transform(df[raw_cols])
        te_auto = encoders[1].transform(df[raw_cols])
        for col_name in encoders[0].get_feature_names_out():
            orig_c = col_name.split('__')[0]
            target_col = f"{orig_c}__te_mean_seed_42_smooth_10_inner_n_fold_5"
            X[target_col] = te10[col_name].values
        for col_name in encoders[1].get_feature_names_out():
            orig_c = col_name.split('__')[0]
            target_col = f"{orig_c}__te_mean_seed_42_smooth_auto_inner_n_fold_5"
            X[target_col] = te_auto[col_name].values

        return X[cols_reference]

    print("\n[STEP 1] Generating fold-specific feature matrices with interaction target encodings...", flush=True)
    t_feat0 = time.time()
    
    train_fold_matrices = []
    test_fold_matrices = []

    global_prior = y.mean()
    SMOOTHING = 20.0

    for fold in range(5):
        print(f"  Building features for Fold {fold+1}/5...", flush=True)
        tr_idx, val_idx = splits[fold]
        
        X_tr_base = build_base_75(train, fold)
        X_te_base = build_base_75(test, fold)

        # Compute Out-of-Fold smoothed target encodings for joint interaction pairs
        for col1, col2 in interaction_pairs:
            feat_name = f"te_joint_{col1}_x_{col2}"
            pair_train_tr = train.iloc[tr_idx][col1].astype(str) + '__' + train.iloc[tr_idx][col2].astype(str)
            pair_train_all = train[col1].astype(str) + '__' + train[col2].astype(str)
            pair_test_all = test[col1].astype(str) + '__' + test[col2].astype(str)

            # Target stats on training fold only
            stats = pd.DataFrame({'pair': pair_train_tr, 'target': y[tr_idx]}).groupby('pair').agg(['count', 'mean'])['target']
            smooth_map = ((stats['count'] * stats['mean'] + SMOOTHING * global_prior) / (stats['count'] + SMOOTHING)).to_dict()

            X_tr_base[feat_name] = pair_train_all.map(smooth_map).fillna(global_prior).astype(np.float32).values
            X_te_base[feat_name] = pair_test_all.map(smooth_map).fillna(global_prior).astype(np.float32).values

        train_fold_matrices.append(X_tr_base)
        test_fold_matrices.append(X_te_base)

    print(f"All 5 fold feature representations built in {time.time()-t_feat0:.1f}s!", flush=True)
    print(f"Total features per fold: {train_fold_matrices[0].shape[1]} (75 base + 4 joint TEs)", flush=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # [STEP 2] MULTI-SEED LIGHTGBM BAGGING (Seeds 101, 2024 + Seed 42 baseline)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 2] MULTI-SEED LIGHTGBM (SEEDS 101 & 2024)", flush=True)
    print("=" * 80, flush=True)

    lgb_base_oof = np.load('oof_predictions.npy')
    lgb_base_test = np.load('test_predictions.npy')
    print(f"Seed 42 Baseline LightGBM OOF AUC: {roc_auc_score(y, lgb_base_oof):.6f}", flush=True)

    lgb_seeds = [101, 2024]
    all_lgb_oofs = [lgb_base_oof]
    all_lgb_tests = [lgb_base_test]

    lgb_hyperparams = {
        'objective':         'binary',
        'metric':            'auc',
        'boosting_type':     'gbdt',
        'learning_rate':     0.02,
        'max_depth':         5,
        'num_leaves':        247,
        'min_child_samples': 10,
        'min_child_weight':  0.001,
        'min_split_gain':    0.0,
        'colsample_bytree':  0.3029300829885024,
        'reg_alpha':         0.07094285437903122,
        'reg_lambda':        2.0330390977032424,
        'subsample':         0.812763123433567,
        'subsample_freq':    1,
        'max_bin':           1024,
        'n_jobs':            16,
        'verbose':           -1,
    }

    for seed in lgb_seeds:
        print(f"\n--- Training LightGBM Seed {seed} ---", flush=True)
        t_seed = time.time()
        oof_s = np.zeros(N_TRAIN, dtype=np.float64)
        test_s = np.zeros(N_TEST, dtype=np.float64)

        for fold in range(5):
            tr_idx, val_idx = splits[fold]
            X_tr = train_fold_matrices[fold].iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = train_fold_matrices[fold].iloc[val_idx]
            y_va = y[val_idx]

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

            cbs = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=-1)]
            
            cur_p = {**lgb_hyperparams, 'random_state': seed + fold}
            m = lgb.train(cur_p, ds_tr, num_boost_round=3000, valid_sets=[ds_va], callbacks=cbs)

            oof_s[val_idx] = m.predict(X_va, num_iteration=m.best_iteration)
            test_s += m.predict(test_fold_matrices[fold], num_iteration=m.best_iteration) / 5.0
            print(f"  Fold {fold+1} AUC: {roc_auc_score(y_va, oof_s[val_idx]):.6f} (iter {m.best_iteration})", flush=True)

        s_auc = roc_auc_score(y, oof_s)
        print(f"LightGBM Seed {seed} Full OOF AUC: {s_auc:.6f} in {time.time()-t_seed:.1f}s", flush=True)
        all_lgb_oofs.append(oof_s)
        all_lgb_tests.append(test_s)
        np.save(f'lgb_seed_{seed}_oof.npy', oof_s)
        np.save(f'lgb_seed_{seed}_test.npy', test_s)

    # Multi-seed log-odds average
    lgb_ms_oof_odds = np.mean([logit(np.clip(p, 1e-6, 1.0-1e-6)) for p in all_lgb_oofs], axis=0)
    lgb_ms_test_odds = np.mean([logit(np.clip(p, 1e-6, 1.0-1e-6)) for p in all_lgb_tests], axis=0)
    lgb_ms_oof = expit(lgb_ms_oof_odds)
    lgb_ms_test = expit(lgb_ms_test_odds)
    print(f"\n>>> Multi-Seed Bagged LightGBM (3 Seeds) OOF AUC: {roc_auc_score(y, lgb_ms_oof):.6f} <<<", flush=True)
    np.save('lgb_multiseed_3seeds_oof.npy', lgb_ms_oof)
    np.save('lgb_multiseed_3seeds_test.npy', lgb_ms_test)

    # ─────────────────────────────────────────────────────────────────────────────
    # [STEP 3] TRAIN FULL 5-FOLD XGBOOST ON 79 FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 3] TRAINING 5-FOLD XGBOOST (HIST) ON FULL 79 FEATURES", flush=True)
    print("=" * 80, flush=True)

    xgb_oof = np.zeros(N_TRAIN, dtype=np.float64)
    xgb_test = np.zeros(N_TEST, dtype=np.float64)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'learning_rate': 0.04,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'max_bin': 1024,
        'nthread': 16,
        'random_state': 42
    }

    t_xgb0 = time.time()
    for fold in range(5):
        t_f = time.time()
        tr_idx, val_idx = splits[fold]
        X_tr = train_fold_matrices[fold].iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = train_fold_matrices[fold].iloc[val_idx]
        y_va = y[val_idx]
        X_te = test_fold_matrices[fold]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_va, label=y_va)
        dtest  = xgb.DMatrix(X_te)

        cur_params = {**xgb_params, 'random_state': 42 + fold}
        bst = xgb.train(cur_params, dtrain, num_boost_round=1200, evals=[(dval, 'val')],
                        early_stopping_rounds=40, verbose_eval=200)

        xgb_oof[val_idx] = bst.predict(dval)
        xgb_test += bst.predict(dtest) / 5.0

        f_auc = roc_auc_score(y_va, xgb_oof[val_idx])
        print(f"  XGBoost Fold {fold+1} AUC: {f_auc:.6f} in {time.time()-t_f:.1f}s (iter {bst.best_iteration})", flush=True)

    total_xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"\n>>> Full 5-Fold XGBoost 79-Feat OOF AUC: {total_xgb_auc:.6f} in {time.time()-t_xgb0:.1f}s <<<", flush=True)
    np.save('xgb79_oof.npy', xgb_oof)
    np.save('xgb79_test.npy', xgb_test)

    # ─────────────────────────────────────────────────────────────────────────────
    # [STEP 4] TRAIN FULL 5-FOLD CATBOOST ON 79 FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 4] TRAINING 5-FOLD CATBOOST ON FULL 79 FEATURES", flush=True)
    print("=" * 80, flush=True)

    cb_oof = np.zeros(N_TRAIN, dtype=np.float64)
    cb_test = np.zeros(N_TEST, dtype=np.float64)

    t_cb0 = time.time()
    for fold in range(5):
        t_f = time.time()
        tr_idx, val_idx = splits[fold]
        X_tr = train_fold_matrices[fold].iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = train_fold_matrices[fold].iloc[val_idx]
        y_va = y[val_idx]
        X_te = test_fold_matrices[fold]

        cb = CatBoostClassifier(
            iterations=700,
            learning_rate=0.06,
            depth=6,
            eval_metric='AUC',
            random_seed=42 + fold,
            thread_count=-1,
            verbose=200
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40, verbose=200)

        cb_oof[val_idx] = cb.predict_proba(X_va)[:, 1]
        cb_test += cb.predict_proba(X_te)[:, 1] / 5.0

        f_auc = roc_auc_score(y_va, cb_oof[val_idx])
        print(f"  CatBoost Fold {fold+1} AUC: {f_auc:.6f} in {time.time()-t_f:.1f}s", flush=True)

    total_cb_auc = roc_auc_score(y, cb_oof)
    print(f"\n>>> Full 5-Fold CatBoost 79-Feat OOF AUC: {total_cb_auc:.6f} in {time.time()-t_cb0:.1f}s <<<", flush=True)
    np.save('cb79_oof.npy', cb_oof)
    np.save('cb79_test.npy', cb_test)

    # ─────────────────────────────────────────────────────────────────────────────
    # [STEP 5] L2 REGULARIZED LOGISTIC STACKER IN LOG-ODDS SPACE
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 5] GRANDMASTER L2 LOG-ODDS STACKING ACROSS ALL MODEL FAMILIES", flush=True)
    print("=" * 80, flush=True)

    ladder_oof  = np.load('ladder_lgb_oof.npy')
    ladder_test = np.load('ladder_lgb_test_preds.npy')
    optuna_oof  = np.load('optuna_lgb_oof.npy')
    optuna_test = np.load('optuna_lgb_test.npy')

    # Convert all candidate model predictions to log-odds
    def to_odds(p):
        return logit(np.clip(p, 1e-6, 1.0 - 1e-6))

    Z_train = np.column_stack([
        to_odds(lgb_ms_oof),     # 1. Multi-Seed Bagged LightGBM 79
        to_odds(ladder_oof),     # 2. Ladder LightGBM
        to_odds(optuna_oof),     # 3. Optuna 144-feature LightGBM
        to_odds(xgb_oof),        # 4. Strong XGBoost 79
        to_odds(cb_oof),         # 5. Strong CatBoost 79
    ])

    Z_test = np.column_stack([
        to_odds(lgb_ms_test),
        to_odds(ladder_test),
        to_odds(optuna_test),
        to_odds(xgb_test),
        to_odds(cb_test),
    ])

    feature_names = ['LGB_MultiSeed', 'LGB_Ladder', 'LGB_Optuna', 'XGB_79', 'CB_79']

    # 5-fold out-of-fold stacking meta-learner
    stack_oof = np.zeros(N_TRAIN, dtype=np.float64)
    stack_test = np.zeros(N_TEST, dtype=np.float64)
    meta_coefs = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        meta_clf = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=1000)
        meta_clf.fit(Z_train[tr_idx], y[tr_idx])
        stack_oof[val_idx] = meta_clf.predict_proba(Z_train[val_idx])[:, 1]
        stack_test += meta_clf.predict_proba(Z_test)[:, 1] / 5.0
        meta_coefs.append(meta_clf.coef_[0])
        print(f"  Stack Fold {fold+1} OOF AUC: {roc_auc_score(y[val_idx], stack_oof[val_idx]):.6f}", flush=True)

    avg_coef = np.mean(meta_coefs, axis=0)
    print("\nLearned Meta-Stacker Weights on Log-Odds:", flush=True)
    for name, c in zip(feature_names, avg_coef):
        print(f"  {name:15s}: {c:+.4f}", flush=True)

    unclamped_auc = roc_auc_score(y, stack_oof)
    print(f"\nUnclamped Grandmaster Stacker OOF AUC: {unclamped_auc:.6f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # [STEP 6] DETERMINISTIC MANIFOLD CLAMPING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("[STEP 6] ENFORCING DETERMINISTIC INVARIANT MANIFOLDS", flush=True)
    print("=" * 80, flush=True)

    def apply_clamping(preds, df):
        inc = df['Annual_Income_USD'].values / 100000.0
        env = df['Environmental_Concern_Level'].values.astype(float)
        sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
        med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
        high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
        s_score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

        clamped = preds.copy()
        clamped[df['Annual_Income_USD'].values >= 170537] = 1.0
        clamped[s_score < 0.70260] = 0.0
        clamped[s_score > 7.03543] = 1.0
        return clamped

    final_oof_clamped = apply_clamping(stack_oof, train)
    final_test_clamped = apply_clamping(stack_test, test)

    grandmaster_auc = roc_auc_score(y, final_oof_clamped)
    print(f"\n****************************************************************", flush=True)
    print(f"*** FINAL GRANDMASTER PINNACLE CLAMPED OOF ROC-AUC: {grandmaster_auc:.6f} ***", flush=True)
    print(f"****************************************************************\n", flush=True)

    np.save('grandmaster_pinnacle_oof.npy', final_oof_clamped)
    np.save('grandmaster_pinnacle_test.npy', final_test_clamped)

    out_csv = 'submissions/submission_grandmaster_75_pinnacle.csv'
    sub_df = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test_clamped})
    sub_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)

    assert len(sub_df) == 286571
    assert not sub_df['Will_Buy_EV'].isnull().any()
    assert (sub_df['Will_Buy_EV'] >= 0.0).all() and (sub_df['Will_Buy_EV'] <= 1.0).all()
    print("ALL VERIFICATIONS PASSED: Submission is ready for Kaggle submission!", flush=True)

if __name__ == '__main__':
    main()
