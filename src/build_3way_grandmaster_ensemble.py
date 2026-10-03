import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("=== 3-WAY GRANDMASTER META-ENSEMBLE (LIGHTGBM + XGBOOST + CATBOOST) ===", flush=True)

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

# Load OOF predictions
print("Loading OOF predictions...", flush=True)
oof_lgb = np.load('oof_predictions.npy')
oof_xgb = np.load('xgb_oof.npy')
oof_cb = np.load('catboost_oof.npy')

print(f"1. LightGBM (75-fe) OOF AUC: {roc_auc_score(y_train, oof_lgb):.6f}", flush=True)
print(f"2. XGBoost  (29-fe) OOF AUC: {roc_auc_score(y_train, oof_xgb):.6f}", flush=True)
print(f"3. CatBoost (29-fe) OOF AUC: {roc_auc_score(y_train, oof_cb):.6f}", flush=True)

# Compute normalized ranks
r_lgb = rankdata(oof_lgb) / len(y_train)
r_xgb = rankdata(oof_xgb) / len(y_train)
r_cb = rankdata(oof_cb) / len(y_train)

# Search optimal weights
print("\nSearching optimal rank-blending weights...", flush=True)
best_auc = 0.0
best_weights = None

for w_lgb in np.linspace(0.60, 0.90, 31):
    for w_xgb in np.linspace(0.05, 0.35, 31):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < 0.0:
            continue
        blend = w_lgb * r_lgb + w_xgb * r_xgb + w_cb * r_cb
        auc = roc_auc_score(y_train, blend)
        if auc > best_auc:
            best_auc = auc
            best_weights = (w_lgb, w_xgb, w_cb)

w1, w2, w3 = best_weights
print(f"\n*** Optimal Weights: LightGBM = {w1:.3f}, XGBoost = {w2:.3f}, CatBoost = {w3:.3f} ***", flush=True)
print(f"*** 3-Way Ensembled OOF ROC-AUC: {best_auc:.6f} (vs LGBM 0.945832, +{best_auc - 0.945832:.6f}) ***", flush=True)

# Load Test predictions
print("\nLoading Test predictions...", flush=True)
test_lgb = np.load('test_predictions.npy')
test_xgb = np.load('xgb_test_preds.npy')
test_cb = np.load('catboost_test_preds.npy')

test_r_lgb = rankdata(test_lgb) / len(test_lgb)
test_r_xgb = rankdata(test_xgb) / len(test_xgb)
test_r_cb = rankdata(test_cb) / len(test_cb)

final_test_blend = w1 * test_r_lgb + w2 * test_r_xgb + w3 * test_r_cb

# Calculate Chris Deotte Formula for deterministic test boundaries
inc = test['Annual_Income_USD'].values / 100000.0
env = test['Environmental_Concern_Level'].values
sub = (test['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (test['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (test['Range_Anxiety_Level'] == 'High').astype(float).values
score_test = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

# Apply hard deterministic boundary clamping
cliff_mask = (test['Annual_Income_USD'].values >= 170537)
floor_mask = (score_test < 0.70260)
ceil_mask = (score_test > 7.03543)

print(f"Applying Millionaire Cliff to {cliff_mask.sum()} test rows -> 1.0", flush=True)
print(f"Applying Zero Floor to {floor_mask.sum()} test rows -> 0.0", flush=True)
print(f"Applying High Ceiling to {ceil_mask.sum()} test rows -> 1.0", flush=True)

final_test_blend[cliff_mask] = 1.0
final_test_blend[floor_mask] = 0.0
final_test_blend[ceil_mask] = 1.0

# Build final submission dataframe
sub_final = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': final_test_blend
})

out_path = 'submissions/submission_3way_grandmaster_ensemble.csv'
sub_final.to_csv(out_path, index=False)
print(f"\nSaved final submission to {out_path}", flush=True)

# Verification checks
assert len(sub_final) == 286571, "Row count mismatch!"
assert not sub_final['Will_Buy_EV'].isnull().any(), "Found NaNs!"
assert (sub_final['Will_Buy_EV'] >= 0.0).all() and (sub_final['Will_Buy_EV'] <= 1.0).all(), "Values out of bounds!"
print("Verification checks ALL PASSED! 3-Way Grandmaster Ensemble is READY FOR FIRST PLACE!", flush=True)
