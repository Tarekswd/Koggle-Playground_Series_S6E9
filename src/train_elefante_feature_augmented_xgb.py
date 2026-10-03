"""
=============================================================================
FEATURE-AUGMENTED XGBOOST ENGINE & CARUANA DIVERSITY ENSEMBLE
(Inspired by Paul Bryan Elefante, Rank 2 Solution)
=============================================================================
1. Builds a feature-augmented representation feeding Level-0 model predictions
   (LightGBM 75, Ladder LGB, CatBoost) and inter-model disagreement deltas
   as continuous features into a high-capacity Hist-XGBoost model.
2. Trains across 5 Stratified folds on 16 CPU threads.
3. Evaluates Spearman correlation with existing anchor models.
4. Optimizes Caruana Hill-Climbing / Logit Meta-Blend across the diverse pool
   (incorporating models with Spearman correlation < 0.99).
5. Applies verified deterministic boundary clamping.
6. Generates submissions/submission_paul_elefante_rank2_grandmaster.csv
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')

def main():
    print("=" * 80, flush=True)
    print("STARTING FEATURE-AUGMENTED XGBOOST PIPELINE (RANK 2 ARCHITECTURE)", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)
    print(f"Loaded Train: {N_TRAIN:,} rows | Test: {N_TEST:,} rows", flush=True)

    # 1. Load Level-0 Basic Model Predictions
    print("\n[STEP 1] Loading Level-0 Basic Model Predictions...", flush=True)
    p_lgb    = np.load('oof_predictions.npy')
    t_lgb    = np.load('test_predictions.npy')
    p_ladder = np.load('ladder_lgb_oof.npy')
    t_ladder = np.load('ladder_lgb_test_preds.npy')
    p_cb     = np.load('catboost_oof.npy')
    t_cb     = np.load('catboost_test_preds.npy')
    p_cb79   = np.load('cb79_oof.npy')
    t_cb79   = np.load('cb79_test.npy')

    def to_odds(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return logit(p)

    def build_augmented_features(df, p_l, p_lad, p_c, p_c79):
        X = pd.DataFrame(index=df.index)

        # Meta-Features: Inner-fold predictions in log-odds space
        z_l   = to_odds(p_l)
        z_lad = to_odds(p_lad)
        z_c   = to_odds(p_c)
        z_c79 = to_odds(p_c79)

        X['meta_z_lgb']    = z_l
        X['meta_z_ladder'] = z_lad
        X['meta_z_cb']     = z_c
        X['meta_z_cb79']   = z_c79

        # Inter-model disagreement & uncertainty features
        X['meta_diff_lgb_cb']   = z_l - z_c
        X['meta_diff_lgb_lad']  = z_l - z_lad
        X['meta_diff_cb_cb79']  = z_c - z_c79
        X['meta_model_std']     = np.std([z_l, z_lad, z_c, z_c79], axis=0)

        # Raw Numerics
        for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                  'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                  'Environmental_Concern_Level']:
            X[c] = df[c].values

        # Categoricals (enabled for Hist-XGBoost)
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

        # Simpson's Paradox Infrastructure features
        X['public_station_dependency'] = df['Charging_Stations_Near_Home'].values * (1.0 - home_chg)
        X['stations_total'] = df['Charging_Stations_Near_Home'].values + df['Charging_Stations_Near_Work'].values
        X['income_per_commute'] = df['Annual_Income_USD'].values / (df['Daily_Commute_km'].values + 1.0)

        # Digit decompositions
        inc_int = df['Annual_Income_USD'].astype(int).values
        for d in [1, 2, 3]:
            X[f'inc_digit_{d}'] = (inc_int // (10**d)) % 10

        commute_float = df['Daily_Commute_km'].values
        X['commute_dec_1'] = (np.round(commute_float * 10).astype(int)) % 10
        X['commute_dec_3'] = (np.round(commute_float * 1000).astype(int)) % 10

        return X

    print("Building Feature-Augmented Train and Test Matrices...", flush=True)
    X_train_aug = build_augmented_features(train, p_lgb, p_ladder, p_cb, p_cb79)
    X_test_aug  = build_augmented_features(test,  t_lgb, t_ladder, t_cb, t_cb79)
    print(f"Matrix ready: {X_train_aug.shape[1]} features (including {8} Level-0 prediction & disagreement features)", flush=True)

    # 2. 5-Fold Stratified Cross-Validation
    print("\n[STEP 2] Training 5-Fold Feature-Augmented Hist-XGBoost...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_train_aug, y))

    oof_xgb_elefante  = np.zeros(N_TRAIN, dtype=np.float64)
    test_xgb_elefante = np.zeros(N_TEST,  dtype=np.float64)

    xgb_params = {
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'tree_method':      'hist',
        'learning_rate':    0.03,
        'max_depth':        5,
        'colsample_bytree': 0.50,
        'subsample':        0.80,
        'max_bin':          1024,
        'nthread':          16,
        'random_state':     42
    }

    dtest = xgb.DMatrix(X_test_aug, enable_categorical=True)

    t_xgb_start = time.time()
    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_f0 = time.time()
        X_tr = X_train_aug.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train_aug.iloc[val_idx]
        y_val = y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval   = xgb.DMatrix(X_va, label=y_val, enable_categorical=True)

        cur_params = {**xgb_params, 'random_state': 42 + fold}
        bst = xgb.train(cur_params, dtrain, num_boost_round=1500, evals=[(dval, 'val')],
                        early_stopping_rounds=40, verbose_eval=False)

        oof_xgb_elefante[val_idx] = bst.predict(dval)
        test_xgb_elefante += bst.predict(dtest) / 5.0

        f_auc = roc_auc_score(y_val, oof_xgb_elefante[val_idx])
        print(f"  Fold {fold+1}/5 AUC: {f_auc:.6f} in {time.time()-t_f0:.1f}s (best iter {bst.best_iteration})", flush=True)

    total_elefante_auc = roc_auc_score(y, oof_xgb_elefante)
    print(f"\n>>> Full 5-Fold Feature-Augmented XGBoost OOF ROC-AUC: {total_elefante_auc:.6f} in {time.time()-t_xgb_start:.1f}s <<<", flush=True)

    np.save('xgb_elefante_meta_oof.npy', oof_xgb_elefante)
    np.save('xgb_elefante_meta_test.npy', test_xgb_elefante)

    # 3. Check Diversity vs Anchor Model
    anchor_oof = np.load('stacking_logit_pinnacle_oof.npy')
    anchor_test = np.load('stacking_logit_pinnacle_test.npy')
    corr_elefante, _ = spearmanr(anchor_oof, oof_xgb_elefante)
    print(f"\nSpearman Correlation (Feature-Augmented XGB vs Anchor Stacker): {corr_elefante:.6f}", flush=True)

    # 4. Deterministic Clamping Definition
    inc_tr = train['Annual_Income_USD'].values / 100000.0
    env_tr = train['Environmental_Concern_Level'].values.astype(float)
    sub_tr = (train['Subsidy_Available'] == 'Yes').astype(float).values
    med_tr = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high_tr = (train['Range_Anxiety_Level'] == 'High').astype(float).values
    score5_tr = 1.2 * inc_tr + 0.6 * env_tr + 2.0 * sub_tr - 1.0 * med_tr - 3.0 * high_tr

    inc_te = test['Annual_Income_USD'].values / 100000.0
    env_te = test['Environmental_Concern_Level'].values.astype(float)
    sub_te = (test['Subsidy_Available'] == 'Yes').astype(float).values
    med_te = (test['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high_te = (test['Range_Anxiety_Level'] == 'High').astype(float).values
    score5_te = 1.2 * inc_te + 0.6 * env_te + 2.0 * sub_te - 1.0 * med_te - 3.0 * high_te

    def clamp(preds, is_train=True):
        out = preds.copy()
        if is_train:
            out[train['Annual_Income_USD'].values >= 170537] = 1.0
            out[score5_tr < 0.702596] = 0.0
            out[score5_tr > 7.03543] = 1.0
        else:
            out[test['Annual_Income_USD'].values >= 170537] = 1.0
            out[score5_te < 0.702596] = 0.0
            out[score5_te > 7.03543] = 1.0
        return out

    # 5. Dual-Paradigm Logit Micro-Blend Optimization
    print("\n" + "=" * 80, flush=True)
    print("[STEP 3] DIVERSE GRANDMASTER ENSEMBLE (ANCHOR STACKER + ELEFANTE XGB + CATBOOST)", flush=True)
    print("=" * 80, flush=True)

    z_anchor_oof = to_odds(anchor_oof)
    z_xgb_oof    = to_odds(oof_xgb_elefante)
    z_cb79_oof   = to_odds(p_cb79)

    z_anchor_test = to_odds(anchor_test)
    z_xgb_test    = to_odds(test_xgb_elefante)
    z_cb79_test   = to_odds(t_cb79)

    best_auc = 0.0
    best_weights = None

    for w_anchor in np.linspace(0.40, 0.85, 10):
        for w_xgb in np.linspace(0.10, 0.50, 9):
            w_cb = 1.0 - w_anchor - w_xgb
            if w_cb < 0.0 or w_cb > 0.25:
                continue
            z_blend = w_anchor * z_anchor_oof + w_xgb * z_xgb_oof + w_cb * z_cb79_oof
            score = roc_auc_score(y, clamp(expit(z_blend), is_train=True))
            if score > best_auc:
                best_auc = score
                best_weights = (w_anchor, w_xgb, w_cb)

    print(f"Optimal Blend Weights:", flush=True)
    print(f"  Anchor Stacker (0.94619 LB)  : {best_weights[0]*100:.1f}%", flush=True)
    print(f"  Feature-Augmented XGBoost    : {best_weights[1]*100:.1f}%", flush=True)
    print(f"  Diverse CatBoost 79 (rho<0.99): {best_weights[2]*100:.1f}%", flush=True)

    print("\n" + "*" * 75, flush=True)
    print(f"*** FINAL GRANDMASTER ELEFANTE ENSEMBLE OOF ROC-AUC: {best_auc:.6f} ***", flush=True)
    print(f"    Previous Best Record:                           0.946061  (LB: 0.94619)", flush=True)
    print(f"    Net CV Advance:                                +{best_auc - 0.946061:.6f}", flush=True)
    print("*" * 75 + "\n", flush=True)

    # 6. Generate Test Submission
    z_final_test = best_weights[0] * z_anchor_test + best_weights[1] * z_xgb_test + best_weights[2] * z_cb79_test
    p_final_test = clamp(expit(z_final_test), is_train=False)

    np.save('elefante_grandmaster_oof.npy', clamp(expit(best_weights[0] * z_anchor_oof + best_weights[1] * z_xgb_oof + best_weights[2] * z_cb79_oof), is_train=True))
    np.save('elefante_grandmaster_test.npy', p_final_test)

    out_csv = 'submissions/submission_paul_elefante_rank2_grandmaster.csv'
    sub_df = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': p_final_test})
    sub_df.to_csv(out_csv, index=False)
    print(f"Saved submission to: {out_csv}", flush=True)

    assert len(sub_df) == 286571
    assert not sub_df['Will_Buy_EV'].isnull().any()
    assert (sub_df['Will_Buy_EV'] >= 0.0).all() and (sub_df['Will_Buy_EV'] <= 1.0).all()
    print("ALL VERIFICATIONS PASSED: File is verified and ready for upload!", flush=True)
    print(f"Total execution time: {time.time()-t_start:.1f}s", flush=True)

if __name__ == '__main__':
    main()
