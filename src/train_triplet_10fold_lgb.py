"""
=============================================================================
10-FOLD TRIPLET-TARGET-ENCODED LIGHTGBM & DUAL-PARADIGM MICRO-BLEND
=============================================================================
1. Exploits the 98.42% exact (Income, Subsidy, Range_Anxiety) test overlap.
2. Injects fold-safe Laplace-smoothed empirical target rates at multiple radii:
   - Triplet TE (Income x Subsidy x Anxiety) with m=5.0 and m=20.0
   - Pair TE (Income x Subsidy) with m=10.0
   - Exact Income TE with m=10.0
   - Triplet frequency count
3. Resolves Simpson's Paradox:
   - Public charging dependency = Charging_Stations_Near_Home * (1 - Home_Charging)
   - Stations ratio and home/work interactions
4. 10-Fold Stratified Cross-Validation (90% train per fold for ultra-sharp posteriors)
5. Micro-Blends with the noise-hedged optimal stacker (0.94619 LB) in log-odds space.
6. Enforces exact invariant boundary manifolds.
7. Generates submissions/submission_grandmaster_triplet_microblend_pinnacle.csv
=============================================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

def main():
    print("=" * 80, flush=True)
    print("STARTING 10-FOLD TRIPLET-TARGET-ENCODED LIGHTGBM & MICRO-BLEND", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)
    print(f"Loaded Train: {N_TRAIN:,} rows | Test: {N_TEST:,} rows", flush=True)

    # 1. Feature Engineering Base
    print("\n[STEP 1] Engineering Base and Simpson's Paradox Features...", flush=True)
    
    def build_core_features(df):
        X = pd.DataFrame(index=df.index)
        
        # Raw Numerics
        for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                  'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                  'Environmental_Concern_Level']:
            X[c] = df[c].values

        # Categoricals as category dtype
        cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
                    'Subsidy_Available', 'Range_Anxiety_Level']
        for c in cat_cols:
            X[c] = df[c].astype('category')

        # Domain signals & ground-truth recipe
        inc = df['Annual_Income_USD'].values / 100000.0
        env = df['Environmental_Concern_Level'].values.astype(float)
        sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
        med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
        high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
        low_anx = (df['Range_Anxiety_Level'] == 'Low').astype(float).values
        home_chg = (df['Home_Charging_Possible'] == 'Yes').astype(float).values

        X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx + 1.0 * low_anx + 0.2 * home_chg
        X['buy_score_sq'] = X['buy_score'] ** 2

        # Simpson's Paradox Infrastructure Features
        X['public_station_dependency'] = df['Charging_Stations_Near_Home'].values * (1.0 - home_chg)
        X['stations_total'] = df['Charging_Stations_Near_Home'].values + df['Charging_Stations_Near_Work'].values
        X['stations_diff'] = df['Charging_Stations_Near_Home'].values - df['Charging_Stations_Near_Work'].values
        X['income_per_commute'] = df['Annual_Income_USD'].values / (df['Daily_Commute_km'].values + 1.0)
        X['commute_per_station'] = df['Daily_Commute_km'].values / (X['stations_total'] + 1.0)

        # Digit decompositions
        inc_int = df['Annual_Income_USD'].astype(int).values
        for d in [1, 2, 3, 4]:
            X[f'inc_digit_{d}'] = (inc_int // (10**d)) % 10

        commute_float = df['Daily_Commute_km'].values
        X['commute_digit_1']  = (commute_float.astype(int) // 10) % 10
        X['commute_digit_0']  = commute_float.astype(int) % 10
        X['commute_dec_1']    = (np.round(commute_float * 10).astype(int)) % 10
        X['commute_dec_3']    = (np.round(commute_float * 1000).astype(int)) % 10
        X['commute_dec_4']    = (np.round(commute_float * 10000).astype(int)) % 10

        # Frequency features
        for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Age']:
            vc = df[col].value_counts(normalize=True)
            X[f'{col}_freq'] = df[col].map(vc).values.astype(np.float32)

        return X

    X_train_base = build_core_features(train)
    X_test_base  = build_core_features(test)

    # 2. String keys for exact empirical matching
    train['triplet_key'] = train['Annual_Income_USD'].astype(str) + '__' + train['Subsidy_Available'].astype(str) + '__' + train['Range_Anxiety_Level'].astype(str)
    test['triplet_key']  = test['Annual_Income_USD'].astype(str) + '__' + test['Subsidy_Available'].astype(str) + '__' + test['Range_Anxiety_Level'].astype(str)

    train['pair_key'] = train['Annual_Income_USD'].astype(str) + '__' + train['Subsidy_Available'].astype(str)
    test['pair_key']  = test['Annual_Income_USD'].astype(str) + '__' + test['Subsidy_Available'].astype(str)

    train['inc_key'] = train['Annual_Income_USD'].astype(str)
    test['inc_key']  = test['Annual_Income_USD'].astype(str)

    # 3. 10-Fold Stratified Cross Validation
    N_SPLITS = 10
    print(f"\n[STEP 2] Setting up {N_SPLITS}-Fold Cross-Validation...", flush=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits = list(skf.split(train, y))

    global_prior = y.mean()

    lgb_params = {
        'objective':         'binary',
        'metric':            'auc',
        'boosting_type':     'gbdt',
        'learning_rate':     0.02,
        'max_depth':         5,
        'num_leaves':        127,
        'min_child_samples': 20,
        'colsample_bytree':  0.40,
        'subsample':         0.80,
        'subsample_freq':    1,
        'max_bin':           1024,
        'reg_alpha':         0.1,
        'reg_lambda':        2.0,
        'n_jobs':            16,
        'verbose':           -1,
        'random_state':      42
    }

    oof_triplet_lgb = np.zeros(N_TRAIN, dtype=np.float64)
    test_triplet_lgb = np.zeros(N_TEST, dtype=np.float64)

    print(f"\n[STEP 3] Training {N_SPLITS}-Fold LightGBM with Fold-Safe Empirical Posteriors...", flush=True)

    train['target'] = y

    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_f0 = time.time()
        tr_df = train.iloc[tr_idx]
        val_df = train.iloc[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # Fold-Safe Target Encodings (Fast Vectorized GroupBy)
        # 1. Triplet TE (m=5.0 and m=20.0)
        grp_trip = tr_df.groupby('triplet_key')['target'].agg(['count', 'mean'])
        map_trip_5 = ((grp_trip['count'] * grp_trip['mean'] + 5.0 * global_prior) / (grp_trip['count'] + 5.0)).to_dict()
        map_trip_20 = ((grp_trip['count'] * grp_trip['mean'] + 20.0 * global_prior) / (grp_trip['count'] + 20.0)).to_dict()
        count_trip = grp_trip['count'].to_dict()

        # 2. Pair TE (m=10.0)
        grp_pair = tr_df.groupby('pair_key')['target'].agg(['count', 'mean'])
        map_pair_10 = ((grp_pair['count'] * grp_pair['mean'] + 10.0 * global_prior) / (grp_pair['count'] + 10.0)).to_dict()

        # 3. Exact Income TE (m=10.0)
        grp_inc = tr_df.groupby('inc_key')['target'].agg(['count', 'mean'])
        map_inc_10 = ((grp_inc['count'] * grp_inc['mean'] + 10.0 * global_prior) / (grp_inc['count'] + 10.0)).to_dict()

        # Build feature views for this fold
        X_tr = X_train_base.iloc[tr_idx].copy()
        X_va = X_train_base.iloc[val_idx].copy()
        X_te = X_test_base.copy()

        # Map empirical posteriors
        X_tr['te_triplet_m5']  = tr_df['triplet_key'].map(map_trip_5).fillna(global_prior).astype(np.float32)
        X_va['te_triplet_m5']  = val_df['triplet_key'].map(map_trip_5).fillna(global_prior).astype(np.float32)
        X_te['te_triplet_m5']  = test['triplet_key'].map(map_trip_5).fillna(global_prior).astype(np.float32)

        X_tr['te_triplet_m20'] = tr_df['triplet_key'].map(map_trip_20).fillna(global_prior).astype(np.float32)
        X_va['te_triplet_m20'] = val_df['triplet_key'].map(map_trip_20).fillna(global_prior).astype(np.float32)
        X_te['te_triplet_m20'] = test['triplet_key'].map(map_trip_20).fillna(global_prior).astype(np.float32)

        X_tr['te_pair_m10']    = tr_df['pair_key'].map(map_pair_10).fillna(global_prior).astype(np.float32)
        X_va['te_pair_m10']    = val_df['pair_key'].map(map_pair_10).fillna(global_prior).astype(np.float32)
        X_te['te_pair_m10']    = test['pair_key'].map(map_pair_10).fillna(global_prior).astype(np.float32)

        X_tr['te_income_m10']  = tr_df['inc_key'].map(map_inc_10).fillna(global_prior).astype(np.float32)
        X_va['te_income_m10']  = val_df['inc_key'].map(map_inc_10).fillna(global_prior).astype(np.float32)
        X_te['te_income_m10']  = test['inc_key'].map(map_inc_10).fillna(global_prior).astype(np.float32)

        X_tr['triplet_count']  = tr_df['triplet_key'].map(count_trip).fillna(0).astype(np.float32)
        X_va['triplet_count']  = val_df['triplet_key'].map(count_trip).fillna(0).astype(np.float32)
        X_te['triplet_count']  = test['triplet_key'].map(count_trip).fillna(0).astype(np.float32)

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_val, reference=ds_tr)

        cbs = [lgb.early_stopping(stopping_rounds=50, verbose=False),
               lgb.log_evaluation(period=-1)]

        cur_params = {**lgb_params, 'random_state': 42 + fold}
        bst = lgb.train(cur_params, ds_tr, num_boost_round=3000, valid_sets=[ds_va], callbacks=cbs)

        oof_triplet_lgb[val_idx] = bst.predict(X_va, num_iteration=bst.best_iteration)
        test_triplet_lgb += bst.predict(X_te, num_iteration=bst.best_iteration) / float(N_SPLITS)

        f_auc = roc_auc_score(y_val, oof_triplet_lgb[val_idx])
        print(f"  Fold {fold+1:2d}/{N_SPLITS} AUC: {f_auc:.6f} in {time.time()-t_f0:.1f}s (iter {bst.best_iteration})", flush=True)

    triplet_lgb_auc = roc_auc_score(y, oof_triplet_lgb)
    print(f"\n>>> 10-Fold Triplet LightGBM OOF ROC-AUC: {triplet_lgb_auc:.6f} <<<", flush=True)
    np.save('triplet_10fold_lgb_oof.npy', oof_triplet_lgb)
    np.save('triplet_10fold_lgb_test.npy', test_triplet_lgb)

    # 4. Deterministic Clamping
    def clamp(preds, df):
        inc = df['Annual_Income_USD'].values / 100000.0
        env = df['Environmental_Concern_Level'].values.astype(float)
        sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
        med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
        high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
        low_anx = (df['Range_Anxiety_Level'] == 'Low').astype(float).values
        home_chg = (df['Home_Charging_Possible'] == 'Yes').astype(float).values
        sc = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx + 1.0 * low_anx + 0.2 * home_chg

        out = preds.copy()
        out[df['Annual_Income_USD'].values >= 170537] = 1.0
        out[sc < 0.70260] = 0.0
        out[sc > 7.03543] = 1.0
        return out

    # 5. Dual-Paradigm Micro-Blend
    print("\n" + "=" * 80, flush=True)
    print("[STEP 4] DUAL-PARADIGM MICRO-BLEND (NOISE-HEDGE STACKER + TRIPLET LGB)", flush=True)
    print("=" * 80, flush=True)

    oof_opt_stacker  = np.load('stacking_logit_pinnacle_oof.npy')
    test_opt_stacker = np.load('stacking_logit_pinnacle_test.npy')
    print(f"Optimal Stacker (0.94619 LB) OOF AUC: {roc_auc_score(y, oof_opt_stacker):.6f}", flush=True)

    def to_odds(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return logit(p)

    z_stack_oof = to_odds(oof_opt_stacker)
    z_trip_oof  = to_odds(oof_triplet_lgb)

    z_stack_test = to_odds(test_opt_stacker)
    z_trip_test  = to_odds(test_triplet_lgb)

    best_auc = 0.0
    best_w = 0.5

    for w in np.linspace(0.0, 1.0, 101):
        z_blend = w * z_stack_oof + (1.0 - w) * z_trip_oof
        p_clamped = clamp(expit(z_blend), train)
        score = roc_auc_score(y, p_clamped)
        if score > best_auc:
            best_auc = score
            best_w = w

    print(f"\nOptimal Micro-Blend Weight: {best_w:.2f} * Stacker + {1.0 - best_w:.2f} * Triplet-LGB", flush=True)
    print(f"****************************************************************", flush=True)
    print(f"*** FINAL GRANDMASTER MICRO-BLEND CLAMPED OOF ROC-AUC: {best_auc:.6f} ***", flush=True)
    print(f"****************************************************************\n", flush=True)

    # Generate Final Submission
    z_final_test = best_w * z_stack_test + (1.0 - best_w) * z_trip_test
    p_final_test = clamp(expit(z_final_test), test)

    np.save('grandmaster_triplet_microblend_oof.npy', clamp(expit(best_w * z_stack_oof + (1.0 - best_w) * z_trip_oof), train))
    np.save('grandmaster_triplet_microblend_test.npy', p_final_test)

    out_csv = 'submissions/submission_grandmaster_triplet_microblend_pinnacle.csv'
    sub_df = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': p_final_test})
    sub_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)

    assert len(sub_df) == 286571
    assert not sub_df['Will_Buy_EV'].isnull().any()
    assert (sub_df['Will_Buy_EV'] >= 0.0).all() and (sub_df['Will_Buy_EV'] <= 1.0).all()
    print("ALL SANITY VERIFICATIONS PASSED: File is verified and ready for upload!", flush=True)
    print(f"Total pipeline time: {time.time()-t_start:.1f}s", flush=True)

if __name__ == '__main__':
    main()
