import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("=== CREATING ULTIMATE QUANTIZATION PINNACLE SUBMISSION ===")

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

oof_lgb_75 = np.load('oof_predictions.npy')
oof_ladder = np.load('ladder_lgb_oof.npy')
oof_xgb = np.load('xgb_oof.npy')
oof_cb = np.load('catboost_oof.npy')

r_lgb = rankdata(oof_lgb_75) / len(y_train)
r_ladder = rankdata(oof_ladder) / len(y_train)
r_xgb = rankdata(oof_xgb) / len(y_train)
r_cb = rankdata(oof_cb) / len(y_train)

# Direct optimal weights
w_base, w_lad, w_x, w_c = 0.70, 0.25, 0.03, 0.02
print(f"Optimal Weights: Base={w_base:.3f}, Ladder={w_lad:.3f}, XGB={w_x:.3f}, CB={w_c:.3f}")

oof_unclamped = w_base * r_lgb + w_lad * r_ladder + w_x * r_xgb + w_c * r_cb
print(f"Unclamped OOF AUC: {roc_auc_score(y_train, oof_unclamped):.6f}")

# Clamping evaluation on train
inc_tr = train['Annual_Income_USD'].values / 100000.0
env_tr = train['Environmental_Concern_Level'].values
sub_tr = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_tr = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_tr = (train['Range_Anxiety_Level'] == 'High').astype(float).values
score_tr = 1.2 * inc_tr + 0.6 * env_tr + 2.0 * sub_tr - 1.0 * med_tr - 3.0 * high_tr

cliff_tr = (train['Annual_Income_USD'].values >= 170537)
floor_tr = (score_tr < 0.70260)
ceil_tr = (score_tr > 7.03543)

oof_pinnacle = oof_unclamped.copy()
oof_pinnacle[cliff_tr] = 1.0
oof_pinnacle[floor_tr] = 0.0
oof_pinnacle[ceil_tr] = 1.0

pinnacle_auc = roc_auc_score(y_train, oof_pinnacle)
print(f"\n*** FINAL PINNACLE OOF ROC-AUC: {pinnacle_auc:.6f} ***")

# Generate test predictions
test_lgb_75 = np.load('test_predictions.npy')
test_ladder = np.load('ladder_lgb_test_preds.npy')
test_xgb = np.load('xgb_test_preds.npy')
test_cb = np.load('catboost_test_preds.npy')

r_t_lgb = rankdata(test_lgb_75) / len(test_lgb_75)
r_t_lad = rankdata(test_ladder) / len(test_ladder)
r_t_xgb = rankdata(test_xgb) / len(test_xgb)
r_t_cb = rankdata(test_cb) / len(test_cb)

test_pinnacle = w_base * r_t_lgb + w_lad * r_t_lad + w_x * r_t_xgb + w_c * r_t_cb

inc_te = test['Annual_Income_USD'].values / 100000.0
env_te = test['Environmental_Concern_Level'].values
sub_te = (test['Subsidy_Available'] == 'Yes').astype(float).values
med_te = (test['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_te = (test['Range_Anxiety_Level'] == 'High').astype(float).values
score_te = 1.2 * inc_te + 0.6 * env_te + 2.0 * sub_te - 1.0 * med_te - 3.0 * high_te

cliff_te = (test['Annual_Income_USD'].values >= 170537)
floor_te = (score_te < 0.70260)
ceil_te = (score_te > 7.03543)

print(f"Applying Millionaire Cliff: {cliff_te.sum()} rows -> 1.0")
print(f"Applying Zero Floor:       {floor_te.sum()} rows -> 0.0")
print(f"Applying High Ceiling:     {ceil_te.sum()} rows -> 1.0")

test_pinnacle[cliff_te] = 1.0
test_pinnacle[floor_te] = 0.0
test_pinnacle[ceil_te] = 1.0

sub_pin = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': test_pinnacle
})

out_file = 'submissions/submission_ultimate_quantization_pinnacle.csv'
sub_pin.to_csv(out_file, index=False)
print(f"Saved {out_file} successfully!")

assert len(sub_pin) == 286571
assert not sub_pin['Will_Buy_EV'].isnull().any()
print("Verification 100% PASSED!")
