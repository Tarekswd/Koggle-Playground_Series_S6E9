"""
=============================================================================
CARUANA HILL-CLIMBING ENSEMBLE SELECTOR (PAUL BRYAN ELEFANTE / RANK 2 METHOD)
=============================================================================
Greedily optimizes ROC-AUC by iteratively selecting models from the diverse pool
with replacement, guided by low Spearman correlation (< 0.99).
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
from scipy.special import logit, expit

print("=" * 75)
print("RUNNING CARUANA HILL-CLIMBING ENSEMBLE SELECTOR")
print("=" * 75)

train = pd.read_csv('playground-series-s6e9/train.csv')
test  = pd.read_csv('playground-series-s6e9/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

# 1. Load candidate models
model_files = [
    ('Main_Stacker_0.94619', 'stacking_logit_pinnacle_oof.npy', 'stacking_logit_pinnacle_test.npy'),
    ('LGB_Baseline_75',     'oof_predictions.npy',             'test_predictions.npy'),
    ('LGB_Ladder_75',       'ladder_lgb_oof.npy',              'ladder_lgb_test_preds.npy'),
    ('LGB_Optuna_144',      'optuna_lgb_oof.npy',              'optuna_lgb_test.npy'),
    ('XGB_Hist_79',         'xgb79_oof.npy',                   'xgb79_test.npy'),
    ('CB_Symmetric_79',     'cb79_oof.npy',                    'cb79_test.npy'),
    ('XGB_Baseline_14',     'xgb_oof.npy',                     'xgb_test_preds.npy'),
    ('CB_Baseline_14',      'catboost_oof.npy',                'catboost_test_preds.npy'),
    ('Tribridged_0.94614',  'tribridged_logit_pinnacle_oof.npy','tribridged_logit_pinnacle_test.npy'),
]

oofs = {}
tests = {}
main_name = 'Main_Stacker_0.94619'

print("\n--- Model Pool Characteristics ---")
for name, oof_f, test_f in model_files:
    p_oof = np.load(oof_f)
    p_test = np.load(test_f)
    oofs[name] = p_oof
    tests[name] = p_test
    auc = roc_auc_score(y, p_oof)
    corr, _ = spearmanr(oofs[main_name], p_oof)
    print(f"  {name:25s} | AUC: {auc:.6f} | Spearman vs Main: {corr:.6f}")

# Deterministic Clamping function
inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values.astype(float)
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high = (train['Range_Anxiety_Level'] == 'High').astype(float).values
score5_tr = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med - 3.0 * high

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

# Convert models to log-odds space for linear combination
def to_odds(p):
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))

z_oofs = {name: to_odds(p) for name, p in oofs.items()}
z_tests = {name: to_odds(p) for name, p in tests.items()}

# 2. Caruana Hill-Climbing Algorithm
print("\n--- Starting Caruana Hill-Climbing Optimization ---")
N_ITER = 60
ensemble_names = [main_name]
current_z_oof = z_oofs[main_name].copy()
current_auc = roc_auc_score(y, clamp(expit(current_z_oof), is_train=True))
print(f"Iteration 0: Best Model = {main_name} | Clamped AUC = {current_auc:.6f}")

for it in range(1, N_ITER + 1):
    best_candidate = None
    best_candidate_auc = current_auc
    best_new_z = None

    for name in oofs.keys():
        # Trial adding this candidate to current ensemble
        cand_z = (current_z_oof * len(ensemble_names) + z_oofs[name]) / (len(ensemble_names) + 1)
        cand_pred = clamp(expit(cand_z), is_train=True)
        cand_auc = roc_auc_score(y, cand_pred)
        if cand_auc > best_candidate_auc:
            best_candidate_auc = cand_auc
            best_candidate = name
            best_new_z = cand_z

    if best_candidate is not None:
        ensemble_names.append(best_candidate)
        current_z_oof = best_new_z
        current_auc = best_candidate_auc
        print(f"Iteration {it:2d}: Added {best_candidate:22s} | New Clamped AUC: {current_auc:.6f}")
    else:
        print(f"Iteration {it:2d}: Converged (no further improvement)")
        break

print("\n" + "=" * 75)
print(f"FINAL HILL-CLIMBING CLAMPED OOF ROC-AUC: {current_auc:.6f}")
print("=" * 75)

# Count model weights
counts = pd.Series(ensemble_names).value_counts()
print("\nSelected Ensemble Weights:")
for name, cnt in counts.items():
    pct = cnt / len(ensemble_names) * 100
    print(f"  {name:25s}: {cnt:3d} parts ({pct:5.1f}%)")

# 3. Build Test Predictions
current_z_test = np.zeros(len(test))
for name in ensemble_names:
    current_z_test += z_tests[name]
current_z_test /= len(ensemble_names)

final_test_pred = clamp(expit(current_z_test), is_train=False)

out_csv = 'submissions/submission_paul_elefante_hill_climbing_pinnacle.csv'
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test_pred})
sub.to_csv(out_csv, index=False)
print(f"\nSaved submission to: {out_csv}")
assert len(sub) == 286571
assert not sub['Will_Buy_EV'].isnull().any()
assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all()
print("Verification PASSED: File is ready for upload!")
