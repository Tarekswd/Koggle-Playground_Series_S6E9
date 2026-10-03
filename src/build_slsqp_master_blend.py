"""
=============================================================================
SLSQP MASTER BLEND: OPTIMAL AUC-MAXIMIZING ENSEMBLE
=============================================================================
Phase 2 of the Master Plan: Combines ALL available engines using SLSQP
(Sequential Least Squares Quadratic Programming) to directly maximize AUC.

Inputs (up to 38 engines):
  - fleet/         Original 14 fleet engines (no freq enc)
  - fleet_freq/    New 14 freq-enc fleet engines (Phase 1)
  - Workspace top engines (stacker, lgb75, xgb_meta, etc.)

Algorithm:
  - Minimize  -AUC(weighted_blend)
  - Subject to: sum(weights) = 1, all weights >= 0
  - Uses scipy.optimize.minimize with method='SLSQP'

Post-processing:
  - Same deterministic boundary clamping as build_grand_prix_blend.py
  - Income >= 170537 → 1.0 (invariant rule)
  - S5 score < 0.702596 → 0.0
  - S5 score > 7.03543  → 1.0

Outputs:
  - submissions/submission_freq_fleet_slsqp.csv  (best submission)
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
from scipy.optimize import minimize


def clamp(preds, df):
    """Deterministic boundary post-processing (invariant rules)."""
    inc_raw = df['Annual_Income_USD'].values
    i = inc_raw / 100000.0
    e = df['Environmental_Concern_Level'].values.astype(float)
    s = (df['Subsidy_Available'] == 'Yes').astype(float).values
    m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    score = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h

    clamped = preds.copy()
    clamped[inc_raw >= 170537] = 1.0
    clamped[score < 0.702596]  = 0.0
    clamped[score > 7.03543]   = 1.0
    return clamped


def neg_auc_prob(weights, Z_oof, y):
    """Objective: negative AUC of probability-space weighted blend."""
    blend = Z_oof @ weights
    return -roc_auc_score(y, blend)


def main():
    print("=" * 75, flush=True)
    print("STARTING SLSQP MASTER BLEND (Phase 2)", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN = len(train)
    N_TEST  = len(test)

    # ── 1. Collect all engine predictions ────────────────────────────────────
    engine_dict = {}

    # Workspace top engines (same as build_grand_prix_blend.py)
    ws_engines = [
        ('stacker',    'stacking_logit_pinnacle_oof.npy', 'stacking_logit_pinnacle_test.npy'),
        ('lgb75',      'oof_predictions.npy',             'test_predictions.npy'),
        ('xgb_meta',   'xgb_elefante_meta_oof.npy',       'xgb_elefante_meta_test.npy'),
        ('ladder',     'ladder_lgb_oof.npy',              'ladder_lgb_test_preds.npy'),
        ('optuna',     'optuna_lgb_oof.npy',              'optuna_lgb_test.npy'),
        ('xgb_deotte', 'xgb_base_margin_oof.npy',         'xgb_base_margin_test.npy'),
        ('xgb79',      'xgb79_oof.npy',                   'xgb79_test.npy'),
        ('cb79',       'cb79_oof.npy',                    'cb79_test.npy'),
        ('catboost',   'catboost_oof.npy',                'catboost_test_preds.npy'),
        ('lgb_white',  'lgb75_multiseed_whitened_oof.npy','lgb75_multiseed_whitened_test.npy'),
    ]
    for name, oof_f, test_f in ws_engines:
        if os.path.exists(oof_f) and os.path.exists(test_f):
            arr_tr = np.load(oof_f)
            arr_te = np.load(test_f)
            if len(arr_tr) == N_TRAIN and len(arr_te) == N_TEST:
                engine_dict[f'ws_{name}'] = (arr_tr, arr_te)

    # Original fleet (no freq enc)
    for foof in glob.glob('fleet/*_oof.npy'):
        fte = foof.replace('_oof.npy', '_test.npy')
        if os.path.exists(fte):
            name = 'fl_' + os.path.basename(foof).replace('_oof.npy', '')
            arr_tr = np.load(foof)
            arr_te = np.load(fte)
            if len(arr_tr) == N_TRAIN and len(arr_te) == N_TEST:
                engine_dict[name] = (arr_tr, arr_te)

    # Freq-enc fleet (Phase 1 output)
    freq_fleet_count = 0
    for foof in glob.glob('fleet_freq/*_oof.npy'):
        fte = foof.replace('_oof.npy', '_test.npy')
        if os.path.exists(fte):
            name = 'fq_' + os.path.basename(foof).replace('_oof.npy', '')
            arr_tr = np.load(foof)
            arr_te = np.load(fte)
            if len(arr_tr) == N_TRAIN and len(arr_te) == N_TEST:
                engine_dict[name] = (arr_tr, arr_te)
                freq_fleet_count += 1

    print(f"\nDiscovered {len(engine_dict)} Total Engine Predictions:", flush=True)
    print(f"  -- Workspace engines:        {len(ws_engines)}", flush=True)
    print(f"  -- Original fleet engines:   {len([k for k in engine_dict if k.startswith('fl_')])}", flush=True)
    print(f"  -- Freq-enc fleet engines:   {freq_fleet_count}", flush=True)

    engine_scores = {}
    sorted_names = []
    for name, (arr_tr, _) in engine_dict.items():
        score = roc_auc_score(y, arr_tr)
        engine_scores[name] = score

    sorted_names = sorted(engine_dict.keys(), key=lambda k: engine_scores[k], reverse=True)
    print(f"\nTop 10 engines by OOF AUC:", flush=True)
    for name in sorted_names[:10]:
        print(f"  {name:30s} | OOF AUC: {engine_scores[name]:.6f}", flush=True)

    # ── 2. Build probability matrix (NOT logit — SLSQP works in prob space) ──
    Z_tr = np.column_stack([engine_dict[k][0] for k in sorted_names])
    Z_te = np.column_stack([engine_dict[k][1] for k in sorted_names])
    n_eng = len(sorted_names)
    print(f"\nMeta-matrix shape: {Z_tr.shape}", flush=True)

    # ── 3a. SLSQP AUC-Direct Optimization ────────────────────────────────────
    print("\nRunning SLSQP AUC-direct optimization...", flush=True)
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds      = [(0.0, 1.0)] * n_eng
    x0          = np.ones(n_eng) / n_eng

    result = minimize(
        neg_auc_prob,
        x0,
        args=(Z_tr, y),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-10}
    )

    slsqp_weights = result.x
    slsqp_oof  = Z_tr @ slsqp_weights
    slsqp_test = Z_te @ slsqp_weights
    slsqp_auc  = roc_auc_score(y, slsqp_oof)
    print(f"SLSQP OOF AUC (pre-clamp):  {slsqp_auc:.6f}", flush=True)
    print(f"SLSQP Convergence: {result.message}", flush=True)

    # Show top contributing engines
    print("\nTop 15 engines by SLSQP weight:", flush=True)
    weight_pairs = sorted(zip(sorted_names, slsqp_weights), key=lambda x: x[1], reverse=True)
    for name, w in weight_pairs[:15]:
        print(f"  {name:30s}: {w:.4f}  (OOF: {engine_scores[name]:.6f})")

    # ── 3b. L2 Logit Stacker (baseline comparison) ────────────────────────────
    print("\nRunning L2 logit stacker (baseline comparison)...", flush=True)
    def to_odds(p):
        return logit(np.clip(p, 1e-6, 1.0 - 1e-6))
    Z_lo_tr = np.column_stack([to_odds(engine_dict[k][0]) for k in sorted_names])
    Z_lo_te = np.column_stack([to_odds(engine_dict[k][1]) for k in sorted_names])
    meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
    meta.fit(Z_lo_tr, y)
    logit_oof  = meta.predict_proba(Z_lo_tr)[:, 1]
    logit_test = meta.predict_proba(Z_lo_te)[:, 1]
    logit_auc  = roc_auc_score(y, logit_oof)
    print(f"L2 Logit OOF AUC (pre-clamp): {logit_auc:.6f}", flush=True)

    # ── 3c. Rank Average (diversity baseline) ────────────────────────────────
    print("\nComputing rank average...", flush=True)
    from scipy.stats import rankdata
    rank_oof  = np.zeros(N_TRAIN)
    rank_test = np.zeros(N_TEST)
    for name in sorted_names:
        rank_oof  += rankdata(engine_dict[name][0]) / len(sorted_names)
        rank_test += rankdata(engine_dict[name][1]) / len(sorted_names)
    rank_oof  /= N_TRAIN
    rank_test /= N_TEST
    rank_auc = roc_auc_score(y, rank_oof)
    print(f"Rank Average OOF AUC (pre-clamp): {rank_auc:.6f}", flush=True)

    # ── 4. Pick best pre-clamp method ────────────────────────────────────────
    candidates = [
        ('SLSQP',        slsqp_auc,  slsqp_oof,  slsqp_test),
        ('L2_Logit',     logit_auc,  logit_oof,  logit_test),
        ('RankAvg',      rank_auc,   rank_oof,   rank_test),
    ]
    best_name, best_auc, best_oof, best_test = max(candidates, key=lambda x: x[1])
    print(f"\nBest pre-clamp method: {best_name} (AUC={best_auc:.6f})", flush=True)

    # ── 5. Apply deterministic clamping ──────────────────────────────────────
    final_oof  = clamp(best_oof, train)
    final_test = clamp(best_test, test)
    final_auc  = roc_auc_score(y, final_oof)

    print("\n" + "*" * 75, flush=True)
    print(f"*** SLSQP MASTER BLEND FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Grand Prix Fleet OOF (previous):   0.946072", flush=True)
    print(f"    Net OOF Advance:                   +{final_auc - 0.946072:+.6f}", flush=True)
    print(f"    Expected LB (if CV is honest):     ~{0.94621 + (final_auc - 0.946072):.5f}", flush=True)
    print("*" * 75 + "\n", flush=True)

    # ── 6. Save submission ────────────────────────────────────────────────────
    out_csv = 'submissions/submission_freq_fleet_slsqp.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test})
    sub.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)

    assert len(sub) == N_TEST
    assert not sub['Will_Buy_EV'].isnull().any()
    assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all()
    print(f"Verification PASSED: File ready for upload!", flush=True)

    # Also save a breakdown comparison
    print("\n--- BLENDING STRATEGY COMPARISON (pre-clamp) ---", flush=True)
    for name, auc, _, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {auc:.6f}", flush=True)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
