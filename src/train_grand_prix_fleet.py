"""
=============================================================================
GRAND PRIX FLEET ENGINE: SCALING LAW ENSEMBLE (32-48 DIVERSE ENGINES)
=============================================================================
Implements Strategy 1: The "48-Engine" Multi-Model Ensemble Scaling Law.
Generates diverse GBDT engines (LightGBM, Hist-XGBoost, CatBoost) varying:
  - Architecture: LightGBM (leaf-wise), Hist-XGBoost (depth-wise), CatBoost (symmetric)
  - Depth: max_depth in [4, 5, 6]
  - Column Subsampling: colsample_bytree in [0.25, 0.35, 0.50]
  - Priors: with and without Chris Deotte's analytical base_margin
  - Random Seeds: 111, 222, 333, 444, 555, 666, 777, 888, 999, ...

Pre-computes feature matrices ONCE in memory for ultra-fast training.
Saves individual predictions into fleet/ and tracks the variance-reduction curve.
=============================================================================
"""

import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs('fleet', exist_ok=True)

def main():
    print("=" * 75, flush=True)
    print("STARTING GRAND PRIX FLEET ENGINE GENERATOR", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)

    print(f"Dataset: Train={N_TRAIN:,}, Test={N_TEST:,}", flush=True)

    # 1. Base Feature Matrix + CTGAN Digit Decomposition
    print("Constructing in-memory feature matrices...", flush=True)
    def build_features(df):
        X = pd.DataFrame(index=df.index)
        raw_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                    'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                    'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                    'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

        for c in raw_cols:
            if df[c].dtype == 'object':
                X[c] = df[c].astype('category').cat.codes
            else:
                X[c] = df[c]

        inc_raw = df['Annual_Income_USD'].values.astype(int)
        X['inc_digit0'] = inc_raw % 10
        X['inc_digit1'] = (inc_raw // 10) % 10
        X['inc_digit2'] = (inc_raw // 100) % 10
        X['inc_digit3'] = (inc_raw // 1000) % 10
        X['inc_digit4'] = (inc_raw // 10000) % 10
        X['inc_mod100'] = inc_raw % 100
        X['inc_mod1000'] = inc_raw % 1000
        X['inc_mod5000'] = inc_raw % 5000

        # CTGAN Structural Flags
        X['is_30k'] = (inc_raw == 30000).astype(int)
        X['in_dead_zone'] = ((inc_raw >= 38000) & (inc_raw <= 42000)).astype(int)
        X['is_cliff'] = (inc_raw >= 170537).astype(int)

        # Interactions
        home_chg = (df['Home_Charging_Possible'] == 'Yes').astype(int)
        sub_yes  = (df['Subsidy_Available'] == 'Yes').astype(int)
        X['home_station_dep'] = df['Charging_Stations_Near_Home'] * (1 - home_chg)
        X['commute_inc_ratio'] = df['Daily_Commute_km'] / (df['Annual_Income_USD'] / 1000.0 + 1)
        X['env_inc_mult'] = df['Environmental_Concern_Level'] * (df['Annual_Income_USD'] / 10000.0)
        X['env_sub_mult'] = df['Environmental_Concern_Level'] * sub_yes
        return X

    X_train = build_features(train)
    X_test  = build_features(test)
    print(f"Features ready: {X_train.shape[1]} features (built in {time.time()-t0:.1f}s)", flush=True)

    # 2. Chris Deotte Base Margin
    def compute_base_margin(df):
        inc = df['Annual_Income_USD'].values / 100000.0
        env = df['Environmental_Concern_Level'].values.astype(float)
        sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
        med = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
        high = (df['Range_Anxiety_Level'] == 'High').astype(float).values
        score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med - 3.0 * high
        return 2.1740 * (score - 5.5) - 0.2437

    bm_train = compute_base_margin(train)
    bm_test  = compute_base_margin(test)

    # 3. 5-Fold Stratified Split (Seed 42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_train, y))

    # Engine Specification Grid
    engines = [
        # --- LightGBM Engines ---
        {'id': 'lgb_d4_c25_s111', 'type': 'lgb', 'depth': 4, 'leaves': 16, 'colsample': 0.25, 'subsample': 0.8, 'lr': 0.04, 'seed': 111},
        {'id': 'lgb_d5_c30_s222', 'type': 'lgb', 'depth': 5, 'leaves': 31, 'colsample': 0.30, 'subsample': 0.8, 'lr': 0.035, 'seed': 222},
        {'id': 'lgb_d5_c40_s333', 'type': 'lgb', 'depth': 5, 'leaves': 31, 'colsample': 0.40, 'subsample': 0.85, 'lr': 0.035, 'seed': 333},
        {'id': 'lgb_d6_c30_s444', 'type': 'lgb', 'depth': 6, 'leaves': 45, 'colsample': 0.30, 'subsample': 0.8, 'lr': 0.03, 'seed': 444},
        {'id': 'lgb_d6_c45_s555', 'type': 'lgb', 'depth': 6, 'leaves': 45, 'colsample': 0.45, 'subsample': 0.8, 'lr': 0.03, 'seed': 555},
        {'id': 'lgb_d5_c35_s666', 'type': 'lgb', 'depth': 5, 'leaves': 31, 'colsample': 0.35, 'subsample': 0.75, 'lr': 0.04, 'seed': 666},

        # --- Hist-XGBoost Engines (Standard & Base-Margin) ---
        {'id': 'xgb_d4_c40_s101', 'type': 'xgb', 'depth': 4, 'colsample': 0.40, 'subsample': 0.8, 'lr': 0.04, 'seed': 101, 'bm': False},
        {'id': 'xgb_d5_c35_s202', 'type': 'xgb', 'depth': 5, 'colsample': 0.35, 'subsample': 0.8, 'lr': 0.035, 'seed': 202, 'bm': False},
        {'id': 'xgb_d5_c50_s303', 'type': 'xgb', 'depth': 5, 'colsample': 0.50, 'subsample': 0.8, 'lr': 0.035, 'seed': 303, 'bm': True},
        {'id': 'xgb_d6_c40_s404', 'type': 'xgb', 'depth': 6, 'colsample': 0.40, 'subsample': 0.8, 'lr': 0.03, 'seed': 404, 'bm': True},
        {'id': 'xgb_d6_c55_s505', 'type': 'xgb', 'depth': 6, 'colsample': 0.55, 'subsample': 0.8, 'lr': 0.03, 'seed': 505, 'bm': False},

        # --- CatBoost Engines (Symmetric Trees) ---
        {'id': 'cb_d5_l5_s701',   'type': 'cb',  'depth': 5, 'l2': 5.0,  'lr': 0.05, 'seed': 701},
        {'id': 'cb_d6_l10_s702',  'type': 'cb',  'depth': 6, 'l2': 10.0, 'lr': 0.04, 'seed': 702},
        {'id': 'cb_d5_l3_s703',   'type': 'cb',  'depth': 5, 'l2': 3.0,  'lr': 0.05, 'seed': 703},
    ]

    print(f"\nTotal Engines to Train in this Run: {len(engines)}", flush=True)

    for idx, eng in enumerate(engines):
        eng_id = eng['id']
        oof_path  = f'fleet/{eng_id}_oof.npy'
        test_path = f'fleet/{eng_id}_test.npy'

        if os.path.exists(oof_path) and os.path.exists(test_path):
            existing_oof = np.load(oof_path)
            print(f"[{idx+1}/{len(engines)}] {eng_id:22s} ALREADY CACHED -> AUC: {roc_auc_score(y, existing_oof):.6f}", flush=True)
            continue

        print(f"\n[{idx+1}/{len(engines)}] Training Engine: {eng_id} ({eng['type'].upper()})...", flush=True)
        t_eng = time.time()
        oof_preds = np.zeros(N_TRAIN, dtype=np.float64)
        test_preds = np.zeros(N_TEST, dtype=np.float64)

        # A. LightGBM Engine
        if eng['type'] == 'lgb':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': eng['lr'],
                'max_depth': eng['depth'],
                'num_leaves': eng['leaves'],
                'colsample_bytree': eng['colsample'],
                'subsample': eng['subsample'],
                'n_estimators': 2000,
                'n_jobs': 16,
                'verbose': -1,
                'random_state': eng['seed']
            }
            for fold, (tr_idx, val_idx) in enumerate(splits):
                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_train.iloc[tr_idx], y[tr_idx],
                        eval_set=[(X_train.iloc[val_idx], y[val_idx])],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
                oof_preds[val_idx] = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
                test_preds += clf.predict_proba(X_test)[:, 1] / 5.0

        # B. Hist-XGBoost Engine
        elif eng['type'] == 'xgb':
            use_bm = eng.get('bm', False)
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'learning_rate': eng['lr'],
                'max_depth': eng['depth'],
                'subsample': eng['subsample'],
                'colsample_bytree': eng['colsample'],
                'nthread': 16,
                'random_state': eng['seed']
            }
            dtest = xgb.DMatrix(X_test, base_margin=bm_test if use_bm else None)

            for fold, (tr_idx, val_idx) in enumerate(splits):
                bm_tr = bm_train[tr_idx] if use_bm else None
                bm_va = bm_train[val_idx] if use_bm else None
                dtrain = xgb.DMatrix(X_train.iloc[tr_idx], label=y[tr_idx], base_margin=bm_tr)
                dval   = xgb.DMatrix(X_train.iloc[val_idx], label=y[val_idx], base_margin=bm_va)
                bst = xgb.train(params, dtrain, num_boost_round=1500, evals=[(dval, 'val')],
                                early_stopping_rounds=50, verbose_eval=False)
                oof_preds[val_idx] = bst.predict(dval)
                test_preds += bst.predict(dtest) / 5.0

        # C. CatBoost Engine
        elif eng['type'] == 'cb':
            for fold, (tr_idx, val_idx) in enumerate(splits):
                cb = CatBoostClassifier(
                    iterations=1200,
                    learning_rate=eng['lr'],
                    depth=eng['depth'],
                    l2_leaf_reg=eng['l2'],
                    eval_metric='AUC',
                    random_seed=eng['seed'] + fold,
                    thread_count=16,
                    verbose=False
                )
                cb.fit(X_train.iloc[tr_idx], y[tr_idx],
                       eval_set=(X_train.iloc[val_idx], y[val_idx]),
                       early_stopping_rounds=50, verbose=False)
                oof_preds[val_idx] = cb.predict_proba(X_train.iloc[val_idx])[:, 1]
                test_preds += cb.predict_proba(X_test)[:, 1] / 5.0

        auc_eng = roc_auc_score(y, oof_preds)
        print(f"  --> Engine {eng_id} OOF AUC: {auc_eng:.6f} ({time.time()-t_eng:.1f}s)", flush=True)

        np.save(oof_path, oof_preds)
        np.save(test_path, test_preds)

    print("\n" + "=" * 75, flush=True)
    print("ALL FLEET ENGINES TRAINED AND PERSISTED SUCCESSFULLY!", flush=True)
    print(f"Total time elapsed: {time.time()-t0:.1f}s", flush=True)
    print("=" * 75, flush=True)

if __name__ == '__main__':
    main()
