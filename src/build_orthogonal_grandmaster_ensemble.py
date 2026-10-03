"""
=============================================================================
ORTHOGONAL GRANDMASTER 8-ENGINE STACKING & DETERMINISTIC POST-PROCESSING
=============================================================================
Combines the full suite of diverse, low-correlation models:
1. stacker      (Previous L2 Logistic Pinnacle, verified 0.94619 Public LB)
2. lgb75        (Original 75-feature LightGBM baseline: 0.945832)
3. xgb_deotte   (Chris Deotte Base-Margin Hist-XGBoost: 0.944126)
4. cb79         (CatBoost symmetric oblivious trees, rho = 0.9875: 0.943445)
5. ladder       (Quantization ladder LightGBM: 0.944789)
6. xgb79        (79-feature Hist-XGBoost: 0.944155)
7. optuna       (144-feature Optuna LightGBM: 0.944692)
8. xgb_meta     (Feature-Augmented Hist-XGBoost: 0.945926)

Methodology:
- Log-Odds Projection: z = log(p / (1 - p))
- L2-Regularized Logistic Stacking (C = 0.01) with noise cancellation
- Pure Deterministic Boundary Post-Processing:
    * Annual_Income_USD >= 170,537 -> 1.0 (100% pure positive in train)
    * S5(X) < 0.702596 -> 0.0 (5,595 train rows, exactly 0 positives)
    * S5(X) > 7.03543 -> 1.0 (230 train rows, exactly 230 positives)
- Generates: submissions/submission_orthogonal_grandmaster_pinnacle.csv
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
import os
import time

def main():
    print("=" * 70, flush=True)
    print("BUILDING ORTHOGONAL GRANDMASTER 8-ENGINE ENSEMBLE", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

    def to_odds(p):
        return logit(np.clip(p, 1e-6, 1.0 - 1e-6))

    # All 8 verified engines
    model_paths = {
        'stacker':    ('stacking_logit_pinnacle_oof.npy', 'stacking_logit_pinnacle_test.npy'),
        'lgb75':      ('oof_predictions.npy',             'test_predictions.npy'),
        'xgb_deotte': ('xgb_base_margin_oof.npy',         'xgb_base_margin_test.npy'),
        'cb79':       ('cb79_oof.npy',                    'cb79_test.npy'),
        'ladder':     ('ladder_lgb_oof.npy',              'ladder_lgb_test_preds.npy'),
        'xgb79':      ('xgb79_oof.npy',                   'xgb79_test.npy'),
        'optuna':     ('optuna_lgb_oof.npy',              'optuna_lgb_test.npy'),
        'xgb_meta':   ('xgb_elefante_meta_oof.npy',       'xgb_elefante_meta_test.npy'),
    }

    # Verify all artifacts exist
    missing = []
    for name, (oof_p, test_p) in model_paths.items():
        if not os.path.exists(oof_p) or not os.path.exists(test_p):
            missing.append((name, oof_p, test_p))

    if missing:
        print("\n[!] Missing prerequisite artifacts:")
        for name, oof_p, test_p in missing:
            print(f"  - {name}: missing {oof_p} or {test_p}")
        return

    # Load arrays
    Z_train = []
    Z_test = []
    names = list(model_paths.keys())

    print("\nLoaded 8 Model Engines:")
    for name in names:
        oof_p, test_p = model_paths[name]
        p_oof = np.load(oof_p)
        p_te  = np.load(test_p)
        print(f"  {name:15s} | OOF AUC: {roc_auc_score(y, p_oof):.6f}")
        Z_train.append(to_odds(p_oof))
        Z_test.append(to_odds(p_te))

    Z_tr = np.column_stack(Z_train)
    Z_te = np.column_stack(Z_test)

    # L2 Logistic Stacker
    print("\nFitting L2-Regularized Logistic Stacker in Log-Odds Space (C=0.01)...", flush=True)
    meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=1000)
    meta.fit(Z_tr, y)

    print("\nLearned Model Weights:")
    for name, coef in zip(names, meta.coef_[0]):
        print(f"  {name:15s}: {coef:+.4f}")
    print(f"  Intercept      : {meta.intercept_[0]:+.4f}")

    oof_meta = meta.predict_proba(Z_tr)[:, 1]
    test_meta = meta.predict_proba(Z_te)[:, 1]

    raw_stack_auc = roc_auc_score(y, oof_meta)
    print(f"\nRaw Stacker OOF AUC: {raw_stack_auc:.6f}", flush=True)

    # Deterministic Boundary Clamping
    def clamp(preds, df):
        inc_raw = df['Annual_Income_USD'].values
        i = inc_raw / 100000.0
        e = df['Environmental_Concern_Level'].values.astype(float)
        s = (df['Subsidy_Available'] == 'Yes').astype(float).values
        m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
        h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
        score = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h

        clamped = preds.copy()
        # 100% pure thresholds identified in EDA
        clamped[inc_raw >= 170537] = 1.0
        clamped[score < 0.702596] = 0.0
        clamped[score > 7.03543] = 1.0
        return clamped

    oof_clamped = clamp(oof_meta, train)
    test_clamped = clamp(test_meta, test)

    final_auc = roc_auc_score(y, oof_clamped)
    print("\n" + "*" * 75, flush=True)
    print(f"*** FINAL ORTHOGONAL GRANDMASTER ENSEMBLE OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous Best Record (0.94619 LB):               0.946061", flush=True)
    print(f"    Net CV Advance:                                  +{final_auc - 0.946061:.6f}", flush=True)
    print("*" * 75 + "\n", flush=True)

    out_csv = 'submissions/submission_orthogonal_grandmaster_pinnacle.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_clamped})
    sub.to_csv(out_csv, index=False)
    print(f"Saved submission to: {out_csv}", flush=True)

    assert len(sub) == 286571
    assert not sub['Will_Buy_EV'].isnull().any()
    assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all()
    print("Verification PASSED: File is verified and ready for Kaggle submission!", flush=True)
    print(f"Total execution time: {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()
