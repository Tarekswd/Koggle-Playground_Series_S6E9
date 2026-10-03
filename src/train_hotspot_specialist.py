"""
=============================================================================
HOTSPOT SPECIALIST: MIXTURE-OF-EXPERTS FOR ERROR HOTSPOT
=============================================================================
DIAGNOSIS: 85.48% of all ranking errors come from the cell:
    Subsidy_Available == 'Yes' AND Range_Anxiety_Level == 'Low'
    → Global stacker AUC drops to 0.906350 in this region

STRATEGY: Mixture-of-Experts
    1. Train a HIGH-QUALITY specialist model ONLY on hotspot samples
       (Subsidy=Yes & Range_Anxiety=Low training rows)
    2. For hotspot TEST samples: blend specialist + main model with alpha=0.6 specialist
    3. For non-hotspot samples: use main model unchanged

WHY THIS WORKS:
    - The global model must fit ALL cells simultaneously → underfits hotspot
    - A specialist trained ONLY on hotspot can focus entirely on:
      Income, Environmental Concern, Home Charging, Age within this cell
    - We have ~107K hotspot train samples → sufficient for a strong model
    - OOF is computed within the hotspot fold only to avoid leakage

FEATURES FOR SPECIALIST:
    - All 13 raw columns (Subsidy & Range_Anxiety constant → low importance)
    - Transductive freq encoding (train+test combined) for ALL cats
    - CTGAN digit decompositions of Income
    - Additional interactions: Income×Env, Income×HomeCharging, Age×Income
    - Cross-column freq maps for Income ranges
    - CatBoost native categorical handling

EXPECTED GAIN:
    If hotspot AUC improves from 0.906 → 0.930:
    Δglobal_AUC ≈ 0.1667 × (0.930 - 0.906) × 0.85 ≈ +0.0034
    Combined with calibration: expected +0.0020-0.0030 LB

Saves:
    specialist_oof.npy         (hotspot-region OOF predictions, full train length)
    specialist_test.npy        (full test predictions)
    submissions/submission_hotspot_specialist_blend.csv
=============================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
import warnings
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


def build_specialist_features(df, combined):
    """
    Richer feature set tuned for the hotspot cell (Subsidy=Yes, RA=Low).
    Within this cell, Subsidy and Range_Anxiety are constant so the model
    must rely on Income, Env, HomeCharging, Age, City, CarType etc.
    """
    X = pd.DataFrame(index=df.index)

    raw_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

    # Ordinal encoding
    for c in raw_cols:
        if df[c].dtype == 'object':
            X[c] = df[c].astype('category').cat.codes
        else:
            X[c] = df[c]

    inc = df['Annual_Income_USD'].values.astype(int)
    # Digit decompositions of income (CTGAN structural signal)
    for shift in [1, 10, 100, 1000, 10000]:
        X[f'inc_d{shift}'] = (inc // shift) % 10
    X['inc_mod100']   = inc % 100
    X['inc_mod1000']  = inc % 1000
    X['inc_mod5000']  = inc % 5000
    X['inc_mod10000'] = inc % 10000

    # CTGAN structural flags
    X['is_30k']       = (inc == 30000).astype(int)
    X['in_dead_zone'] = ((inc >= 38000) & (inc <= 42000)).astype(int)
    X['is_cliff']     = (inc >= 170537).astype(int)
    X['inc_log']      = np.log1p(inc)
    X['inc_sqrt']     = np.sqrt(inc)

    # Key interactions within hotspot
    home_chg  = (df['Home_Charging_Possible'] == 'Yes').astype(float)
    env       = df['Environmental_Concern_Level'].values.astype(float)
    age       = df['Age'].values.astype(float)
    inc_norm  = inc / 100000.0

    X['env_x_inc']        = env * inc_norm
    X['env_x_home']       = env * home_chg
    X['inc_x_home']       = inc_norm * home_chg
    X['age_x_inc']        = age * inc_norm
    X['age_x_env']        = age * env
    X['home_station_dep'] = df['Charging_Stations_Near_Home'] * (1 - home_chg)
    X['commute_inc']      = df['Daily_Commute_km'] / (inc_norm + 0.01)
    X['env_x_stations']   = env * df['Charging_Stations_Near_Work']
    X['inc_x_ncars']      = inc_norm * df['Number_of_Cars_Owned']

    # Income buckets (fine-grained within hotspot)
    X['inc_bucket_10k']   = (inc // 10000).clip(0, 30)
    X['inc_bucket_5k']    = (inc // 5000).clip(0, 60)
    X['inc_bucket_20k']   = (inc // 20000).clip(0, 15)

    # ── TRANSDUCTIVE frequency encoding ──────────────────────────────────────
    cat_cols = ['Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    for c in cat_cols:
        freq = combined[c].value_counts(normalize=True).to_dict()
        X[f'{c}_freq'] = df[c].map(freq).fillna(0.0)

    # Digit-level freq (CTGAN income structural spikes)
    combined_inc = combined['Annual_Income_USD'].values.astype(int)
    digit_maps = {
        'inc_d1_freq':    (combined_inc // 1) % 10,
        'inc_d10_freq':   (combined_inc // 10) % 10,
        'inc_mod100_freq': combined_inc % 100,
        'inc_mod1000_freq': combined_inc % 1000,
    }
    local_digits = {
        'inc_d1_freq':    (inc // 1) % 10,
        'inc_d10_freq':   (inc // 10) % 10,
        'inc_mod100_freq': inc % 100,
        'inc_mod1000_freq': inc % 1000,
    }
    for feat, comb_vals in digit_maps.items():
        freq = pd.Series(comb_vals).value_counts(normalize=True).to_dict()
        X[feat] = pd.Series(local_digits[feat]).map(freq).fillna(0.0).values

    # Age freq
    age_freq = combined['Age'].value_counts(normalize=True).to_dict()
    X['Age_freq'] = df['Age'].map(age_freq).fillna(0.0)

    # Env freq
    env_freq = combined['Environmental_Concern_Level'].value_counts(normalize=True).to_dict()
    X['env_freq'] = df['Environmental_Concern_Level'].map(env_freq).fillna(0.0)

    # Cross-feature hotspot freq (Home × City)
    combined['_home_city'] = combined['Home_Charging_Possible'].astype(str) + '__' + combined['City_Type'].astype(str)
    df_cross = df['Home_Charging_Possible'].astype(str) + '__' + df['City_Type'].astype(str)
    cross_freq = combined['_home_city'].value_counts(normalize=True).to_dict()
    X['home_city_freq'] = df_cross.map(cross_freq).fillna(0.0)

    # Income × Env × HomeCharging triple interaction
    X['triple_inc_env_home'] = inc_norm * env * home_chg

    return X


def main():
    print("=" * 80, flush=True)
    print("HOTSPOT SPECIALIST: MIXTURE-OF-EXPERTS FOR ERROR HOTSPOT", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN, N_TEST = len(train), len(test)

    # ── Identify hotspot region ───────────────────────────────────────────────
    hotspot_mask_train = (
        (train['Subsidy_Available'] == 'Yes') &
        (train['Range_Anxiety_Level'] == 'Low')
    ).values
    hotspot_mask_test = (
        (test['Subsidy_Available'] == 'Yes') &
        (test['Range_Anxiety_Level'] == 'Low')
    ).values

    n_hotspot_train = hotspot_mask_train.sum()
    n_hotspot_test  = hotspot_mask_test.sum()
    hotspot_rate    = hotspot_mask_train.mean()

    print(f"Hotspot: Subsidy=Yes & Range_Anxiety=Low", flush=True)
    print(f"  Train hotspot samples: {n_hotspot_train:,} ({hotspot_rate:.1%} of train)", flush=True)
    print(f"  Test  hotspot samples: {n_hotspot_test:,} ({hotspot_mask_test.mean():.1%} of test)", flush=True)

    # ── Build transductive freq maps ──────────────────────────────────────────
    train_feats = train.drop(columns=['Will_Buy_EV', 'id'], errors='ignore')
    test_feats  = test.drop(columns=['id'], errors='ignore')
    combined    = pd.concat([train_feats, test_feats], ignore_index=True)
    print(f"\nCombined for freq maps: {len(combined):,} rows", flush=True)

    # ── Build full feature matrices ───────────────────────────────────────────
    print("Building feature matrices...", flush=True)
    X_train_full = build_specialist_features(train, combined)
    X_test_full  = build_specialist_features(test, combined)
    n_feats = X_train_full.shape[1]
    print(f"Features: {n_feats}", flush=True)
    assert X_train_full.isnull().sum().sum() == 0
    assert X_test_full.isnull().sum().sum() == 0
    print("NaN check: PASSED", flush=True)

    # ── Hotspot subsets ───────────────────────────────────────────────────────
    X_hot_train = X_train_full[hotspot_mask_train]
    y_hot       = y[hotspot_mask_train]
    X_hot_test  = X_test_full[hotspot_mask_test]
    print(f"\nHotspot training set: {X_hot_train.shape}", flush=True)

    # ── 5-Fold CV on hotspot only ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hot_splits = list(skf.split(X_hot_train, y_hot))

    # ── SPECIALIST A: LightGBM on hotspot ────────────────────────────────────
    print("\n[A] Training LightGBM Hotspot Specialist...", flush=True)
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.02,
        'max_depth': 6, 'num_leaves': 63, 'colsample_bytree': 0.5,
        'subsample': 0.8, 'reg_alpha': 0.05, 'reg_lambda': 1.0,
        'min_child_samples': 20, 'n_jobs': 16, 'verbose': -1,
    }
    lgb_hot_oof  = np.zeros(n_hotspot_train)
    lgb_hot_test = np.zeros(n_hotspot_test)

    for seed in [42, 101, 2024]:
        seed_oof  = np.zeros(n_hotspot_train)
        seed_test = np.zeros(n_hotspot_test)
        for fold, (tr_idx, val_idx) in enumerate(hot_splits):
            clf = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **lgb_params)
            clf.fit(X_hot_train.iloc[tr_idx], y_hot[tr_idx],
                    eval_set=[(X_hot_train.iloc[val_idx], y_hot[val_idx])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
            seed_oof[val_idx] = clf.predict_proba(X_hot_train.iloc[val_idx])[:, 1]
            seed_test += clf.predict_proba(X_hot_test)[:, 1] / 5.0
        lgb_hot_oof  += logit(np.clip(seed_oof,  1e-6, 1-1e-6)) / 3
        lgb_hot_test += logit(np.clip(seed_test, 1e-6, 1-1e-6)) / 3

    lgb_hot_oof  = expit(lgb_hot_oof)
    lgb_hot_test = expit(lgb_hot_test)
    lgb_hot_auc  = roc_auc_score(y_hot, lgb_hot_oof)
    print(f"  LGB Specialist Hotspot OOF AUC: {lgb_hot_auc:.6f}", flush=True)

    # ── SPECIALIST B: CatBoost on hotspot ─────────────────────────────────────
    print("\n[B] Training CatBoost Hotspot Specialist...", flush=True)
    cb_hot_oof  = np.zeros(n_hotspot_train)
    cb_hot_test = np.zeros(n_hotspot_test)

    for fold, (tr_idx, val_idx) in enumerate(hot_splits):
        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=7,
            l2_leaf_reg=5.0, eval_metric='AUC',
            random_seed=42 + fold, thread_count=16, verbose=False
        )
        cb.fit(X_hot_train.iloc[tr_idx], y_hot[tr_idx],
               eval_set=(X_hot_train.iloc[val_idx], y_hot[val_idx]),
               early_stopping_rounds=100, verbose=False)
        cb_hot_oof[val_idx] = cb.predict_proba(X_hot_train.iloc[val_idx])[:, 1]
        cb_hot_test += cb.predict_proba(X_hot_test)[:, 1] / 5.0

    cb_hot_auc = roc_auc_score(y_hot, cb_hot_oof)
    print(f"  CB Specialist Hotspot OOF AUC: {cb_hot_auc:.6f}", flush=True)

    # ── Blend specialists ─────────────────────────────────────────────────────
    def to_odds(p): return logit(np.clip(p, 1e-6, 1-1e-6))

    Z_hot = np.column_stack([to_odds(lgb_hot_oof), to_odds(cb_hot_oof)])
    meta_hot = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000)
    meta_hot.fit(Z_hot, y_hot)
    specialist_oof_hot  = meta_hot.predict_proba(Z_hot)[:, 1]
    Z_hot_te = np.column_stack([to_odds(lgb_hot_test), to_odds(cb_hot_test)])
    specialist_test_hot = meta_hot.predict_proba(Z_hot_te)[:, 1]

    blend_hot_auc = roc_auc_score(y_hot, specialist_oof_hot)
    print(f"\n  SPECIALIST BLEND Hotspot AUC: {blend_hot_auc:.6f}", flush=True)
    print(f"  Baseline stacker hotspot AUC was: 0.906350", flush=True)
    print(f"  Hotspot AUC Improvement: +{blend_hot_auc - 0.906350:.6f}", flush=True)

    # ── Reconstruct full-length OOF and Test arrays ───────────────────────────
    specialist_oof_full  = np.full(N_TRAIN, np.nan)
    specialist_test_full = np.full(N_TEST,  np.nan)
    specialist_oof_full[hotspot_mask_train]  = specialist_oof_hot
    specialist_test_full[hotspot_mask_test]  = specialist_test_hot

    # For non-hotspot: fill with the main stacker predictions
    main_oof  = np.load('stacking_logit_pinnacle_oof.npy')
    main_test = np.load('stacking_logit_pinnacle_test.npy')
    specialist_oof_full[~hotspot_mask_train]  = main_oof[~hotspot_mask_train]
    specialist_test_full[~hotspot_mask_test]  = main_test[~hotspot_mask_test]

    np.save('specialist_oof.npy',  specialist_oof_full)
    np.save('specialist_test.npy', specialist_test_full)

    full_specialist_auc = roc_auc_score(y, specialist_oof_full)
    print(f"\n  Full-train specialist OOF AUC: {full_specialist_auc:.6f}", flush=True)

    # ── Mixture-of-Experts Final Blend ────────────────────────────────────────
    print("\n[C] Building Mixture-of-Experts Final Blend...", flush=True)

    # Load all workspace engines
    all_engines = {
        'specialist': (specialist_oof_full, specialist_test_full),
        'stacker':    (np.load('stacking_logit_pinnacle_oof.npy'),      np.load('stacking_logit_pinnacle_test.npy')),
        'lgb75':      (np.load('oof_predictions.npy'),                  np.load('test_predictions.npy')),
        'xgb_meta':   (np.load('xgb_elefante_meta_oof.npy'),            np.load('xgb_elefante_meta_test.npy')),
        'lgb_white':  (np.load('lgb75_multiseed_whitened_oof.npy'),      np.load('lgb75_multiseed_whitened_test.npy')),
        'optuna':     (np.load('optuna_lgb_oof.npy'),                   np.load('optuna_lgb_test.npy')),
        'ladder':     (np.load('ladder_lgb_oof.npy'),                   np.load('ladder_lgb_test_preds.npy')),
    }

    for name, (oo, _) in all_engines.items():
        print(f"  {name:15s}: {roc_auc_score(y, oo):.6f}", flush=True)

    Z_tr = np.column_stack([to_odds(v[0]) for v in all_engines.values()])
    Z_te = np.column_stack([to_odds(v[1]) for v in all_engines.values()])

    meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
    meta.fit(Z_tr, y)
    final_oof  = clamp(meta.predict_proba(Z_tr)[:, 1], train)
    final_test = clamp(meta.predict_proba(Z_te)[:, 1], test)
    final_auc  = roc_auc_score(y, final_oof)

    print("\n" + "*" * 80, flush=True)
    print(f"*** HOTSPOT SPECIALIST BLEND FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous best (Grand Prix Fleet):      0.946072", flush=True)
    print(f"    Net OOF Advance:                       +{final_auc - 0.946072:.6f}", flush=True)
    print(f"    Expected LB:                           ~{0.94621 + (final_auc - 0.946072):.5f}", flush=True)
    print("*" * 80, flush=True)

    # ── Strategy 2: Direct hotspot override blend ─────────────────────────────
    # In the hotspot region, linearly blend specialist (alpha) + main stacker (1-alpha)
    print("\n[D] Testing direct hotspot override at various alpha values...", flush=True)
    best_alpha, best_auc_override = 0.0, 0.0
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        override_oof = main_oof.copy()
        override_oof[hotspot_mask_train] = (
            alpha * specialist_oof_hot + (1 - alpha) * main_oof[hotspot_mask_train]
        )
        override_oof = clamp(override_oof, train)
        auc = roc_auc_score(y, override_oof)
        print(f"  alpha={alpha:.1f}: OOF AUC = {auc:.6f}", flush=True)
        if auc > best_auc_override:
            best_auc_override = auc
            best_alpha = alpha

    print(f"\n  Best alpha: {best_alpha} → OOF AUC: {best_auc_override:.6f}", flush=True)

    # Generate best override submission
    override_test = main_test.copy()
    override_test[hotspot_mask_test] = (
        best_alpha * specialist_test_hot + (1 - best_alpha) * main_test[hotspot_mask_test]
    )
    override_test = clamp(override_test, test)

    # Pick best between L2 blend and override
    chosen_oof, chosen_test, chosen_name = (
        (final_oof, final_test, 'L2_blend')
        if final_auc >= best_auc_override
        else (override_oof, override_test, f'override_a{best_alpha}')
    )
    chosen_auc = max(final_auc, best_auc_override)

    out_csv = 'submissions/submission_hotspot_specialist_blend.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': chosen_test})
    sub.to_csv(out_csv, index=False)
    assert len(sub) == N_TEST
    assert not sub['Will_Buy_EV'].isnull().any()

    print(f"\n*** CHOSEN METHOD: {chosen_name} | OOF AUC: {chosen_auc:.6f} ***", flush=True)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Verification PASSED! Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
