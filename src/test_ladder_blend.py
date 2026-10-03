import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("=== TESTING QUANTIZATION LADDER INTEGRATION INTO META-ENSEMBLE ===", flush=True)

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

# Load all 4 OOF predictions
oof_lgb_75 = np.load('oof_predictions.npy')
oof_ladder = np.load('ladder_lgb_oof.npy')
oof_xgb = np.load('xgb_oof.npy')
oof_cb = np.load('catboost_oof.npy')

print(f"LGB 75-fe OOF AUC:            {roc_auc_score(y_train, oof_lgb_75):.6f}")
print(f"Quantization Ladder OOF AUC:  {roc_auc_score(y_train, oof_ladder):.6f}")
print(f"XGBoost 5-Fold OOF AUC:       {roc_auc_score(y_train, oof_xgb):.6f}")
print(f"CatBoost 5-Fold OOF AUC:      {roc_auc_score(y_train, oof_cb):.6f}")

r_lgb = rankdata(oof_lgb_75) / len(y_train)
r_ladder = rankdata(oof_ladder) / len(y_train)
r_xgb = rankdata(oof_xgb) / len(y_train)
r_cb = rankdata(oof_cb) / len(y_train)

# Evaluate blend of LGB 75-fe + Ladder LGB
print("\n--- Testing 2-Way LightGBM (75-fe + Ladder) Blend ---")
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    b = (1 - w) * r_lgb + w * r_ladder
    auc = roc_auc_score(y_train, b)
    print(f"Weight Ladder={w:.1f}, Base={1-w:.1f} -> AUC: {auc:.6f}")

# Multi-model grid optimization
best_auc = 0.0
best_weights = None

for w_lgb in [0.65, 0.70, 0.75, 0.80]:
    for w_ladder in [0.05, 0.10, 0.15, 0.20]:
        for w_xgb in [0.03, 0.05, 0.08]:
            w_cb = 1.0 - w_lgb - w_ladder - w_xgb
            if w_cb < 0:
                continue
            b = w_lgb * r_lgb + w_ladder * r_ladder + w_xgb * r_xgb + w_cb * r_cb
            auc = roc_auc_score(y_train, b)
            if auc > best_auc:
                best_auc = auc
                best_weights = (w_lgb, w_ladder, w_xgb, w_cb)

w_lgb, w_ladder, w_xgb, w_cb = best_weights
print(f"\n*** Optimal Multi-Model Weights: LGB_75={w_lgb:.3f}, Ladder={w_ladder:.3f}, XGB={w_xgb:.3f}, CB={w_cb:.3f} ***")
print(f"*** Best Meta-Ensemble OOF ROC-AUC: {best_auc:.6f} ***")

# Apply deterministic boundaries
test_lgb_75 = np.load('test_predictions.npy')
test_ladder = np.load('ladder_lgb_test_preds.npy')
test_xgb = np.load('xgb_test_preds.npy')
test_cb = np.load('catboost_test_preds.npy')

r_t_lgb = rankdata(test_lgb_75) / len(test_lgb_75)
r_t_ladder = rankdata(test_ladder) / len(test_ladder)
r_t_xgb = rankdata(test_xgb) / len(test_xgb)
r_t_cb = rankdata(test_cb) / len(test_cb)

final_test = (
    w_lgb * r_t_lgb +
    w_ladder * r_t_ladder +
    w_xgb * r_t_xgb +
    w_cb * r_t_cb
)

# Chris Deotte deterministic recipe
inc = test['Annual_Income_USD'].values / 100000.0
env = test['Environmental_Concern_Level'].values
sub = (test['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (test['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (test['Range_Anxiety_Level'] == 'High').astype(float).values
score_test = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

cliff_mask = (test['Annual_Income_USD'].values >= 170537)
floor_mask = (score_test < 0.70260)
ceil_mask = (score_test > 7.03543)

final_test[cliff_mask] = 1.0
final_test[floor_mask] = 0.0
final_test[ceil_mask] = 1.0

sub_ladder = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': final_test
})

out_path = 'submissions/submission_quantization_ladder_grandmaster_ensemble.csv'
sub_ladder.to_csv(out_path, index=False)
print(f"Saved final hyper-correct submission to {out_path}!")

assert len(sub_ladder) == 286571
assert not sub_ladder['Will_Buy_EV'].isnull().any()
print("Verification PASSED!")
