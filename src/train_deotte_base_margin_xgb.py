"""
=============================================================================
CHRIS DEOTTE BASE-MARGIN HIST-XGBOOST (5-FOLD CV)
=============================================================================
Incorporates the exact data-generating macro utility formula as the base_margin:
  Score = 1.2*(income/1e5) + 0.6*env + 2.0*sub - 1.0*anxiety_med - 3.0*anxiety_high
  base_margin = 2.1740 * (Score - 5.5) - 0.2437

Features:
  - 13 base raw features
  - CTGAN digit decomposition (% 10, // 10, % 100, % 1000, % 5000, // 10000)
  - CTGAN structural flags (is_30k, in_dead_zone, is_cliff)
  - Charging infrastructure interaction: Charging_Stations_Near_Home * (1 - Home_Charging)
  - Environmental x Income, Environmental x Subsidy interactions
  - Commute vs Income ratio
=============================================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import os

def main():
    print("=" * 70, flush=True)
    print("TRAINING CHRIS DEOTTE BASE-MARGIN HIST-XGBOOST (5-FOLD CV)", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)

    print(f"Loaded train ({N_TRAIN:,} rows) and test ({N_TEST:,} rows)", flush=True)

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

        # CTGAN structural flags
        X['is_30k'] = (inc_raw == 30000).astype(int)
        X['in_dead_zone'] = ((inc_raw >= 38000) & (inc_raw <= 42000)).astype(int)
        X['is_cliff'] = (inc_raw >= 170537).astype(int)

        # Interactions
        home_chg = (df['Home_Charging_Possible'] == 'Yes').astype(int)
        sub_yes = (df['Subsidy_Available'] == 'Yes').astype(int)
        X['home_station_dep'] = df['Charging_Stations_Near_Home'] * (1 - home_chg)
        X['commute_inc_ratio'] = df['Daily_Commute_km'] / (df['Annual_Income_USD'] / 1000.0 + 1)
        X['env_inc_mult'] = df['Environmental_Concern_Level'] * (df['Annual_Income_USD'] / 10000.0)
        X['env_sub_mult'] = df['Environmental_Concern_Level'] * sub_yes
        return X

    print("Building engineered feature matrices...", flush=True)
    X_train = build_features(train)
    X_test  = build_features(test)
    print(f"Total features: {X_train.shape[1]}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(N_TRAIN, dtype=np.float64)
    test_preds = np.zeros(N_TEST, dtype=np.float64)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'nthread': 16,
        'random_state': 42
    }

    dtest = xgb.DMatrix(X_test, base_margin=bm_test)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        t_fold = time.time()
        X_tr, y_tr, bm_tr = X_train.iloc[tr_idx], y[tr_idx], bm_train[tr_idx]
        X_val, y_val, bm_val = X_train.iloc[val_idx], y[val_idx], bm_train[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, base_margin=bm_tr)
        dval   = xgb.DMatrix(X_val, label=y_val, base_margin=bm_val)

        evallist = [(dval, 'val')]
        bst = xgb.train(params, dtrain, num_boost_round=1500, evals=evallist,
                        early_stopping_rounds=50, verbose_eval=False)

        val_pred = bst.predict(dval)
        oof_preds[val_idx] = val_pred
        test_preds += bst.predict(dtest) / 5.0

        f_auc = roc_auc_score(y_val, val_pred)
        print(f"Fold {fold+1} AUC: {f_auc:.6f} (best_iter={bst.best_iteration}, time={time.time()-t_fold:.1f}s)", flush=True)

    total_auc = roc_auc_score(y, oof_preds)
    print(f"\n*** CHRIS DEOTTE BASE-MARGIN HIST-XGBOOST OOF AUC: {total_auc:.6f} ***", flush=True)

    np.save('xgb_base_margin_oof.npy', oof_preds)
    np.save('xgb_base_margin_test.npy', test_preds)
    print("Saved xgb_base_margin_oof.npy and xgb_base_margin_test.npy!", flush=True)
    print(f"Total time elapsed: {time.time()-t_start:.1f}s", flush=True)

if __name__ == '__main__':
    main()
