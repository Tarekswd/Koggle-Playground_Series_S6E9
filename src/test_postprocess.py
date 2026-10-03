import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

d = joblib.load('lgb_ev_model.joblib')
print("Model keys:", list(d.keys()))
oof = np.load('oof_predictions.npy')
test_preds = np.load('test_predictions.npy')

train = pd.read_csv('playground-series-s6e9/train.csv', usecols=['id', 'Will_Buy_EV', 'Annual_Income_USD'])
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

baseline_auc = roc_auc_score(y, oof)
print(f"Base OOF AUC: {baseline_auc:.6f}")

# Test Millionaire Cliff post-processing on OOF:
# If Annual_Income_USD >= 170537, in train all 393 are 1!
# What were the predictions for these 393 rows?
cliff_idx = train['Annual_Income_USD'] >= 170537
print(f"Cliff count in train: {cliff_idx.sum()}")
print(f"LGB OOF min prob for cliff rows: {oof[cliff_idx].min():.5f}, max: {oof[cliff_idx].max():.5f}")

# What if we set cliff rows to 1.0 (or 1.0 + epsilon)?
oof_cliff = oof.copy()
oof_cliff[cliff_idx] = 1.0
auc_cliff = roc_auc_score(y, oof_cliff)
print(f"OOF AUC after setting cliff rows to 1.0: {auc_cliff:.6f} (Delta: {auc_cliff - baseline_auc:+.6f})")

# Test low score post-processing:
inc = train['Annual_Income_USD'].values / 100000.0
env = pd.read_csv('playground-series-s6e9/train.csv', usecols=['Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level'])
sub = (env['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (env['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (env['Range_Anxiety_Level'] == 'High').astype(float).values
score = 1.2 * inc + 0.6 * env['Environmental_Concern_Level'].values + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

zero_idx = score < 0.70260
print(f"Zero score rows in train: {zero_idx.sum()}")
print(f"LGB OOF min prob for zero rows: {oof[zero_idx].min():.5f}, max: {oof[zero_idx].max():.5f}")

oof_both = oof_cliff.copy()
oof_both[zero_idx] = 0.0
auc_both = roc_auc_score(y, oof_both)
print(f"OOF AUC after setting zero rows to 0.0: {auc_both:.6f} (Delta: {auc_both - baseline_auc:+.6f})")
