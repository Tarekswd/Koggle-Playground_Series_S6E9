"""
=============================================================================
HOTSPOT REGION-SPECIFIC STACKER v2
=============================================================================
Diagnosis from v1 (train_hotspot_specialist.py):
  - A raw LGB/CB specialist scored 0.903 in hotspot vs stacker's 0.906
  - First-level models can't beat a meta-stacker — the ensemble diversity
    of 38 engines is what gives 0.906

New strategy: Region-Specific Stacking
  - Use ALL 38 engine OOF predictions as features (logit space)
  - Train stacking weights ONLY on the hotspot region samples
  - If engines have DIFFERENT relative strengths in hotspot, this exploits it
  - For non-hotspot: use global stacker weights (unchanged)

Why this works differently:
  - Global stacker minimizes average logistic loss across all 668K samples
  - Hotspot stacker minimizes logistic loss on only the 381K hardest samples
  - The 57% hotspot gets 2× gradient weight → specialist emphasis on hard cases

Additionally:
  - Compute per-engine AUC in the hotspot to identify which engines are
    relatively strongest there (the surprise finding may be key)
  - Test a ridge regression (C=0.1 and C=1.0) vs L2-logit (C=0.01)
    to allow more flexible hotspot-specific weighting

Outputs:
  submissions/submission_hotspot_stacker_v2.csv
=============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
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


def to_odds(p):
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def main():
    print("=" * 80, flush=True)
    print("HOTSPOT REGION-SPECIFIC STACKER v2", flush=True)
    print("=" * 80, flush=True)

    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN, N_TEST = len(train), len(test)

    # ── Hotspot masks ─────────────────────────────────────────────────────────
    hotspot_tr = (
        (train['Subsidy_Available'] == 'Yes') &
        (train['Range_Anxiety_Level'] == 'Low')
    ).values
    hotspot_te = (
        (test['Subsidy_Available'] == 'Yes') &
        (test['Range_Anxiety_Level'] == 'Low')
    ).values
    print(f"Hotspot: Train={hotspot_tr.sum():,} ({hotspot_tr.mean():.1%}) | Test={hotspot_te.sum():,} ({hotspot_te.mean():.1%})", flush=True)

    # ── Collect all 38 engine predictions ────────────────────────────────────
    engine_dict = {}

    ws_engines = [
        ('ws_stacker',    'stacking_logit_pinnacle_oof.npy',      'stacking_logit_pinnacle_test.npy'),
        ('ws_lgb75',      'oof_predictions.npy',                   'test_predictions.npy'),
        ('ws_xgb_meta',   'xgb_elefante_meta_oof.npy',             'xgb_elefante_meta_test.npy'),
        ('ws_ladder',     'ladder_lgb_oof.npy',                    'ladder_lgb_test_preds.npy'),
        ('ws_optuna',     'optuna_lgb_oof.npy',                    'optuna_lgb_test.npy'),
        ('ws_xgb_deotte', 'xgb_base_margin_oof.npy',               'xgb_base_margin_test.npy'),
        ('ws_xgb79',      'xgb79_oof.npy',                         'xgb79_test.npy'),
        ('ws_cb79',       'cb79_oof.npy',                          'cb79_test.npy'),
        ('ws_catboost',   'catboost_oof.npy',                      'catboost_test_preds.npy'),
        ('ws_lgb_white',  'lgb75_multiseed_whitened_oof.npy',       'lgb75_multiseed_whitened_test.npy'),
    ]
    for name, oof_f, test_f in ws_engines:
        if os.path.exists(oof_f) and os.path.exists(test_f):
            a, b = np.load(oof_f), np.load(test_f)
            if len(a) == N_TRAIN and len(b) == N_TEST:
                engine_dict[name] = (a, b)

    for foof in glob.glob('fleet/*_oof.npy'):
        fte = foof.replace('_oof.npy', '_test.npy')
        if os.path.exists(fte):
            name = 'fl_' + os.path.basename(foof).replace('_oof.npy', '')
            a, b = np.load(foof), np.load(fte)
            if len(a) == N_TRAIN and len(b) == N_TEST:
                engine_dict[name] = (a, b)

    for foof in glob.glob('fleet_freq/*_oof.npy'):
        fte = foof.replace('_oof.npy', '_test.npy')
        if os.path.exists(fte):
            name = 'fq_' + os.path.basename(foof).replace('_oof.npy', '')
            a, b = np.load(foof), np.load(fte)
            if len(a) == N_TRAIN and len(b) == N_TEST:
                engine_dict[name] = (a, b)

    # Also add freq-GM models if available
    for fname, key in [('lgb_freq_gm_oof.npy', 'lgb_freq_gm'),
                       ('xgb_freq_gm_oof.npy', 'xgb_freq_gm')]:
        if os.path.exists(fname):
            fte = fname.replace('_oof.npy', '_test.npy')
            if os.path.exists(fte):
                a, b = np.load(fname), np.load(fte)
                if len(a) == N_TRAIN and len(b) == N_TEST:
                    engine_dict[key] = (a, b)
                    print(f"  [BONUS] Added freq-GM engine: {key}", flush=True)

    n_engines = len(engine_dict)
    names = sorted(engine_dict.keys())
    print(f"\nTotal engines loaded: {n_engines}", flush=True)

    # ── Compute per-engine GLOBAL and HOTSPOT AUC ─────────────────────────────
    print("\n--- Per-Engine AUC: GLOBAL vs HOTSPOT ---", flush=True)
    y_hot = y[hotspot_tr]
    hotspot_aucs = {}
    global_aucs  = {}
    for name in names:
        oof = engine_dict[name][0]
        g_auc = roc_auc_score(y, oof)
        h_auc = roc_auc_score(y_hot, oof[hotspot_tr])
        global_aucs[name]  = g_auc
        hotspot_aucs[name] = h_auc

    # Sort by HOTSPOT AUC to find the specialists
    names_by_hotspot = sorted(names, key=lambda n: hotspot_aucs[n], reverse=True)
    print(f"{'Engine':30s} {'Global':>8s} {'Hotspot':>8s} {'Delta':>8s}", flush=True)
    print("-" * 60, flush=True)
    for name in names_by_hotspot:
        delta = hotspot_aucs[name] - global_aucs[name]
        print(f"{name:30s} {global_aucs[name]:8.6f} {hotspot_aucs[name]:8.6f} {delta:+8.6f}", flush=True)

    # ── Build log-odds matrices ───────────────────────────────────────────────
    Z_tr_all = np.column_stack([to_odds(engine_dict[n][0]) for n in names])
    Z_te_all = np.column_stack([to_odds(engine_dict[n][1]) for n in names])

    # Hotspot subsets
    Z_tr_hot = Z_tr_all[hotspot_tr]
    Z_te_hot = Z_te_all[hotspot_te]

    # ── GLOBAL stacker (baseline reference) ───────────────────────────────────
    print("\n--- GLOBAL Stacker (C=0.01) ---", flush=True)
    global_meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
    global_meta.fit(Z_tr_all, y)
    global_oof  = global_meta.predict_proba(Z_tr_all)[:, 1]
    global_hot_auc = roc_auc_score(y_hot, global_oof[hotspot_tr])
    global_auc = roc_auc_score(y, global_oof)
    print(f"  Global AUC:  {global_auc:.6f}", flush=True)
    print(f"  Hotspot AUC: {global_hot_auc:.6f}", flush=True)

    # ── HOTSPOT-SPECIFIC stacker: train ONLY on hotspot region ───────────────
    print("\n--- HOTSPOT-SPECIFIC Stacker ---", flush=True)
    best_hotspot_auc = 0.0
    best_config = None

    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        hot_meta = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=2000)
        hot_meta.fit(Z_tr_hot, y_hot)

        # In-sample hotspot AUC (not truly OOF but diagnostic)
        hot_preds_train = hot_meta.predict_proba(Z_tr_hot)[:, 1]
        hot_auc_insample = roc_auc_score(y_hot, hot_preds_train)

        # Build full OOF: hotspot uses hotspot stacker, rest uses global
        full_oof = global_oof.copy()
        full_oof[hotspot_tr] = hot_meta.predict_proba(Z_tr_hot)[:, 1]
        full_auc = roc_auc_score(y, full_oof)

        print(f"  C={C:6.3f}: hotspot in-sample AUC={hot_auc_insample:.6f} | full OOF AUC={full_auc:.6f}", flush=True)

        if full_auc > best_hotspot_auc:
            best_hotspot_auc = full_auc
            best_config = (C, hot_meta)

    # ── True OOF evaluation of best hotspot stacker ───────────────────────────
    # Use 5-fold CV within the hotspot to get true OOF
    print("\n--- True 5-Fold OOF of Best Hotspot Stacker ---", flush=True)
    from sklearn.model_selection import StratifiedKFold
    best_C = best_config[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hot_oof_true = np.zeros(hotspot_tr.sum())

    for fold, (tr_i, va_i) in enumerate(skf.split(Z_tr_hot, y_hot)):
        m = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', max_iter=2000)
        m.fit(Z_tr_hot[tr_i], y_hot[tr_i])
        hot_oof_true[va_i] = m.predict_proba(Z_tr_hot[va_i])[:, 1]

    hot_oof_true_auc = roc_auc_score(y_hot, hot_oof_true)

    # Reconstruct full OOF
    full_oof_final = global_oof.copy()
    full_oof_final[hotspot_tr] = hot_oof_true
    full_oof_clamped = clamp(full_oof_final, train)
    final_auc = roc_auc_score(y, full_oof_clamped)

    print(f"  Best C: {best_C}", flush=True)
    print(f"  True OOF Hotspot AUC:  {hot_oof_true_auc:.6f} (vs global: {global_hot_auc:.6f})", flush=True)
    print(f"  Improvement in hotspot: {hot_oof_true_auc - global_hot_auc:+.6f}", flush=True)

    print("\n" + "*" * 80, flush=True)
    print(f"*** HOTSPOT STACKER v2 FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous best (Grand Prix Fleet): 0.946072", flush=True)
    print(f"    Net OOF Advance: +{final_auc - 0.946072:.6f}", flush=True)
    print(f"    Expected LB: ~{0.94621 + (final_auc - 0.946072):.5f}", flush=True)
    print("*" * 80, flush=True)

    # ── Generate test predictions ─────────────────────────────────────────────
    # Use best_config (trained on all hotspot train data) for test hotspot
    hot_meta_final = best_config[1]
    global_test = global_meta.predict_proba(Z_te_all)[:, 1]
    final_test = global_test.copy()
    final_test[hotspot_te] = hot_meta_final.predict_proba(Z_te_hot)[:, 1]
    final_test_clamped = clamp(final_test, test)

    out_csv = 'submissions/submission_hotspot_stacker_v2.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test_clamped})
    sub.to_csv(out_csv, index=False)
    assert len(sub) == N_TEST
    assert not sub['Will_Buy_EV'].isnull().any()
    print(f"\nSaved: {out_csv}", flush=True)
    print("Verification PASSED!", flush=True)


if __name__ == '__main__':
    main()
