import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("=== 4-WAY META-ENSEMBLE (LIGHTGBM + XGBOOST + CATBOOST + PSEUDO-XGB) ===", flush=True)

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

oof_lgb = np.load('oof_predictions.npy')
oof_xgb = np.load('xgb_oof.npy')
oof_cb = np.load('catboost_oof.npy')

# Load pseudo test predictions
sub_pseudo = pd.read_csv('submissions/submission_target_0.952_pseudo_ensemble.csv')
test_pseudo_probs = sub_pseudo['Will_Buy_EV'].values

# Load individual test predictions
test_lgb = np.load('test_predictions.npy')
test_xgb = np.load('xgb_test_preds.npy')
test_cb = np.load('catboost_test_preds.npy')

r_test_lgb = rankdata(test_lgb) / len(test_lgb)
r_test_xgb = rankdata(test_xgb) / len(test_xgb)
r_test_cb = rankdata(test_cb) / len(test_cb)
r_test_pseudo = rankdata(test_pseudo_probs) / len(test_pseudo_probs)

# Blend: 70% LightGBM + 15% Pseudo-Ensemble + 8% CatBoost + 7% XGBoost
mega_test_blend = (
    0.70 * r_test_lgb +
    0.15 * r_test_pseudo +
    0.08 * r_test_cb +
    0.07 * r_test_xgb
)

# Apply deterministic boundaries
inc = test['Annual_Income_USD'].values / 100000.0
env = test['Environmental_Concern_Level'].values
sub = (test['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (test['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (test['Range_Anxiety_Level'] == 'High').astype(float).values
score_test = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

cliff_mask = (test['Annual_Income_USD'].values >= 170537)
floor_mask = (score_test < 0.70260)
ceil_mask = (score_test > 7.03543)

mega_test_blend[cliff_mask] = 1.0
mega_test_blend[floor_mask] = 0.0
mega_test_blend[ceil_mask] = 1.0

sub_mega = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': mega_test_blend
})

out_mega = 'submissions/submission_mega_4way_grandmaster_ensemble.csv'
sub_mega.to_csv(out_mega, index=False)
print(f"Saved {out_mega} successfully!", flush=True)

# Verification
assert len(sub_mega) == 286571
assert not sub_mega['Will_Buy_EV'].isnull().any()
print("Verification PASSED: submission_mega_4way_grandmaster_ensemble.csv is ready!")
