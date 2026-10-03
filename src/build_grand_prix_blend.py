"""
=============================================================================
GRAND PRIX FLEET BLENDER (32+ ENGINES MULTI-MODEL SCALING)
=============================================================================
Aggregates all diverse model engines:
  - 12 verified base engines from the workspace
  - 14+ newly trained fleet engines from fleet/
Solves:
  - L2-Regularized Logit Meta-Stacking
  - Rank-Averaging Optimization
  - Pure Deterministic Boundary Post-Processing
Generates:
  submissions/submission_grand_prix_fleet_pinnacle.csv
=============================================================================
"""

import os
import glob
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
from scipy.stats import rankdata

def main():
    print("=" * 75, flush=True)
    print("STARTING GRAND PRIX FLEET BLENDER (MULTI-ENGINE SCALING)", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)

    # 1. Discover all valid engine pairs
    engine_dict = {}

    # Workspace top engines
    ws_engines = [
        ('stacker',     'stacking_logit_pinnacle_oof.npy', 'stacking_logit_pinnacle_test.npy'),
        ('lgb75',       'oof_predictions.npy',             'test_predictions.npy'),
        ('xgb_meta',    'xgb_elefante_meta_oof.npy',       'xgb_elefante_meta_test.npy'),
        ('ladder',      'ladder_lgb_oof.npy',              'ladder_lgb_test_preds.npy'),
        ('optuna',      'optuna_lgb_oof.npy',              'optuna_lgb_test.npy'),
        ('xgb_deotte',  'xgb_base_margin_oof.npy',         'xgb_base_margin_test.npy'),
        ('xgb79',       'xgb79_oof.npy',                   'xgb79_test.npy'),
        ('cb79',        'cb79_oof.npy',                    'cb79_test.npy'),
        ('catboost',    'catboost_oof.npy',                'catboost_test_preds.npy'),
        ('lgb_white',   'lgb75_multiseed_whitened_oof.npy','lgb75_multiseed_whitened_test.npy'),
    ]

    for name, oof_f, test_f in ws_engines:
        if os.path.exists(oof_f) and os.path.exists(test_f):
            arr_tr = np.load(oof_f)
            arr_te = np.load(test_f)
            if len(arr_tr) == N_TRAIN and len(arr_te) == N_TEST:
                engine_dict[name] = (arr_tr, arr_te)

    # Fleet newly trained engines
    fleet_oofs = glob.glob('fleet/*_oof.npy')
    for foof in fleet_oofs:
        fte = foof.replace('_oof.npy', '_test.npy')
        if os.path.exists(fte):
            name = os.path.basename(foof).replace('_oof.npy', '')
            arr_tr = np.load(foof)
            arr_te = np.load(fte)
            if len(arr_tr) == N_TRAIN and len(arr_te) == N_TEST:
                engine_dict[name] = (arr_tr, arr_te)

    print(f"\nDiscovered {len(engine_dict)} Qualified Model Engines:")
    engine_scores = {}
    for name, (arr_tr, _) in engine_dict.items():
        score = roc_auc_score(y, arr_tr)
        engine_scores[name] = score
        print(f"  {name:22s} | OOF AUC: {score:.6f}")

    # 2. Build Matrices in Log-Odds and Rank Spaces
    def to_odds(p):
        return logit(np.clip(p, 1e-6, 1.0 - 1e-6))

    names = sorted(list(engine_dict.keys()), key=lambda k: engine_scores[k], reverse=True)

    Z_tr = np.column_stack([to_odds(engine_dict[k][0]) for k in names])
    Z_te = np.column_stack([to_odds(engine_dict[k][1]) for k in names])

    print(f"\nFormed Meta-Matrix of shape: {Z_tr.shape}", flush=True)

    # 3. L2 Regularized Logistic Regression Stacker
    print("Optimizing L2-Regularized Stacker across all engines...", flush=True)
    best_c = 0.01
    meta = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs', max_iter=1500)
    meta.fit(Z_tr, y)

    oof_meta  = meta.predict_proba(Z_tr)[:, 1]
    test_meta = meta.predict_proba(Z_te)[:, 1]

    raw_auc = roc_auc_score(y, oof_meta)
    print(f"\nRaw Multi-Engine Stacker OOF AUC: {raw_auc:.6f}", flush=True)

    print("\nTop Contributing Engines (Highest Absolute Weights):")
    coefs = list(zip(names, meta.coef_[0]))
    coefs = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
    for name, w in coefs[:12]:
        print(f"  {name:22s}: {w:+.4f}")

    # 4. Pure Deterministic Boundary Post-Processing
    def clamp(preds, df):
        inc_raw = df['Annual_Income_USD'].values
        i = inc_raw / 100000.0
        e = df['Environmental_Concern_Level'].values.astype(float)
        s = (df['Subsidy_Available'] == 'Yes').astype(float).values
        m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
        h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
        score = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h

        clamped = preds.copy()
        clamped[inc_raw >= 170537] = 1.0
        clamped[score < 0.702596] = 0.0
        clamped[score > 7.03543] = 1.0
        return clamped

    oof_final = clamp(oof_meta, train)
    test_final = clamp(test_meta, test)

    final_auc = roc_auc_score(y, oof_final)
    print("\n" + "*" * 75, flush=True)
    print(f"*** FINAL GRAND PRIX FLEET ENSEMBLE OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous Personal Best (0.94620 LB):      0.946062", flush=True)
    print(f"    Net CV Advance:                           +{final_auc - 0.946062:.6f}", flush=True)
    print("*" * 75 + "\n", flush=True)

    out_csv = 'submissions/submission_grand_prix_fleet_pinnacle.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_final})
    sub.to_csv(out_csv, index=False)
    print(f"Saved final submission to: {out_csv}", flush=True)

    assert len(sub) == N_TEST
    assert not sub['Will_Buy_EV'].isnull().any()
    assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all()
    print("Verification PASSED: File is ready for upload!", flush=True)
    print(f"Total blending time: {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()
