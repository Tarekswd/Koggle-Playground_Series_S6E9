"""
=============================================================================
EXACT BUDDY ARCHITECTURE MULTI-SEED ENSEMBLE (SEEDS 101, 202 + SEED 42)
=============================================================================
Uses the EXACT 75-feature pipeline and hyperparameters from lgb_ev_model.joblib:
- max_depth: 5
- colsample_bytree: 0.302930
- learning_rate: 0.02
- num_leaves: 247
- max_bin: 1024
- min_child_samples: 10
- reg_alpha: 0.070943
- reg_lambda: 2.033039
- subsample: 0.812763
=============================================================================
"""

import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from scipy.stats import rankdata
import time

def main():
    print("=" * 70, flush=True)
    print("EXACT BUDDY ARCHITECTURE MULTI-SEED ENSEMBLE", flush=True)
    print("=" * 70, flush=True)

    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)
    print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,}", flush=True)

    # 1. Load buddy model specification
    d = joblib.load('lgb_ev_model.joblib')
    freq_maps = d['frequency_maps']
    cols_reference = d['feature_columns_per_fold'][0]
    params_buddy = d['params_lgb']
    print(f"Buddy feature count: {len(cols_reference)}", flush=True)
    print(f"Buddy parameters: max_depth={params_buddy.get('max_depth')}, lr={params_buddy.get('learning_rate')}, colsample={params_buddy.get('colsample_bytree'):.4f}, max_bin={params_buddy.get('max_bin')}", flush=True)

    # 2. Exact 75-feature generator
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

        # 2. Digit features (exact buddy model spec)
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

        # 3. Frequency features (from buddy model freq_maps)
        for c in raw_cols:
            fmap = freq_maps[c]
            X[f'{c}_freq'] = df[c].astype(str).map(fmap).fillna(0.0).values

        # 4. Target encodings (fold-specific, from buddy model)
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

        # Align to exact buddy column order
        return X[cols_reference]

    # 3. Load baseline Seed 42 buddy predictions
    oof_seed42  = np.load('oof_predictions.npy')
    test_seed42 = np.load('test_predictions.npy')
    print(f"Seed 42 Baseline OOF AUC: {roc_auc_score(y, oof_seed42):.6f}", flush=True)

    # 4. Train 2 additional seeds with EXACT params: seeds 101, 202
    NEW_SEEDS = [101, 202]
    all_oof_seeds  = [oof_seed42]
    all_test_seeds = [test_seed42]

    lgb_params = {
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

    # Pre-build feature sets for each buddy fold (0..4) to save computation time!
    print("Pre-building 5 fold feature representations...", flush=True)
    t_feat = time.time()
    train_features_per_fold = [build_75_features(train, f) for f in range(5)]
    test_features_per_fold  = [build_75_features(test, f) for f in range(5)]
    print(f"Features ready in {time.time()-t_feat:.1f}s!", flush=True)

    for seed in NEW_SEEDS:
        print(f"\n--- Training Seed {seed} ---", flush=True)
        t0 = time.time()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        oof_s  = np.zeros(N_TRAIN, dtype=np.float64)
        test_s = np.zeros(N_TEST,  dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
            buddy_fold = fold % 5
            X_tr_fold = train_features_per_fold[buddy_fold]
            X_te_fold = test_features_per_fold[buddy_fold]

            X_tr = X_tr_fold.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_tr_fold.iloc[va_idx]
            y_va = y[va_idx]

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

            cbs = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=-1)]
            
            cur_params = {**lgb_params, 'random_state': seed + fold}
            m = lgb.train(cur_params, ds_tr, num_boost_round=3000,
                          valid_sets=[ds_va], callbacks=cbs)

            oof_s[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
            test_s += m.predict(X_te_fold, num_iteration=m.best_iteration) / 5.0

            f_auc = roc_auc_score(y_va, oof_s[va_idx])
            print(f"  Fold {fold+1}: {f_auc:.6f} (iter={m.best_iteration})", flush=True)

        seed_auc = roc_auc_score(y, oof_s)
        print(f"  Seed {seed} OOF AUC: {seed_auc:.6f} ({time.time()-t0:.1f}s)", flush=True)
        all_oof_seeds.append(oof_s)
        all_test_seeds.append(test_s)
        np.save(f'lgb75_exact_seed_{seed}_oof.npy', oof_s)
        np.save(f'lgb75_exact_seed_{seed}_test.npy', test_s)

    # 5. Multi-seed Ensembling
    print("\n" + "=" * 70, flush=True)
    print("MULTI-SEED ENSEMBLE RESULTS (EXACT ARCHITECTURE)", flush=True)
    print("=" * 70, flush=True)

    def to_odds(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

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

    # Logit Average of LGBM seeds
    lgb_ms_oof_odds  = np.mean([to_odds(p) for p in all_oof_seeds], axis=0)
    lgb_ms_test_odds = np.mean([to_odds(p) for p in all_test_seeds], axis=0)
    lgb_ms_oof_logit = expit(lgb_ms_oof_odds)
    lgb_ms_test_logit = expit(lgb_ms_test_odds)
    print(f"Logit Multi-Seed LGBM OOF AUC: {roc_auc_score(y, lgb_ms_oof_logit):.6f}", flush=True)

    np.save('lgb75_exact_multiseed_oof.npy', lgb_ms_oof_logit)
    np.save('lgb75_exact_multiseed_test.npy', lgb_ms_test_logit)

    # 6. Build the Ultimate Pinnacle Meta-Blend in Logit Space:
    oof_ladder  = np.load('ladder_lgb_oof.npy')
    test_ladder = np.load('ladder_lgb_test_preds.npy')
    oof_xgb     = np.load('xgb_oof.npy')
    test_xgb    = np.load('xgb_test_preds.npy')
    oof_cb      = np.load('catboost_oof.npy')
    test_cb     = np.load('catboost_test_preds.npy')

    z_lgb    = to_odds(lgb_ms_oof_logit)
    z_ladder = to_odds(oof_ladder)
    z_xgb    = to_odds(oof_xgb)
    z_cb     = to_odds(oof_cb)

    z_test_lgb    = to_odds(lgb_ms_test_logit)
    z_test_ladder = to_odds(test_ladder)
    z_test_xgb    = to_odds(test_xgb)
    z_test_cb     = to_odds(test_cb)

    print("\n[OPTIMIZE] Tuning Logit Meta-Blend weights...", flush=True)
    best_auc = 0.0
    best_weights = None

    for w_lgb in [0.60, 0.65, 0.70, 0.75]:
        for w_lad in [0.20, 0.25, 0.30, 0.35]:
            for w_x in [0.0, 0.02, 0.04]:
                w_c = 1.0 - w_lgb - w_lad - w_x
                if w_c < 0: continue
                z_blend = w_lgb * z_lgb + w_lad * z_ladder + w_x * z_xgb + w_c * z_cb
                p_blend = expit(z_blend)
                p_clamped = clamp(p_blend, train)
                score = roc_auc_score(y, p_clamped)
                if score > best_auc:
                    best_auc = score
                    best_weights = (w_lgb, w_lad, w_x, w_c)

    print(f"Optimal Weights: LGB={best_weights[0]:.2f}, Ladder={best_weights[1]:.2f}, XGB={best_weights[2]:.2f}, CB={best_weights[3]:.2f}", flush=True)
    print(f"\n{'='*70}", flush=True)
    print(f"*** ULTIMATE PINNACLE CLAMPED OOF AUC: {best_auc:.6f} ***", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Test predictions
    z_test_final = (best_weights[0] * z_test_lgb +
                    best_weights[1] * z_test_ladder +
                    best_weights[2] * z_test_xgb +
                    best_weights[3] * z_test_cb)
    p_test_final = expit(z_test_final)
    p_test_clamped = clamp(p_test_final, test)

    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': p_test_clamped})
    out_path = 'submissions/submission_exact_buddy_multiseed_pinnacle.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved submission: {out_path}", flush=True)
    assert len(sub) == 286571
    assert not sub['Will_Buy_EV'].isnull().any()
    print("Verification PASSED.", flush=True)

if __name__ == '__main__':
    main()
