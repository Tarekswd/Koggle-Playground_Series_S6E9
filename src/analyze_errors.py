import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

print("Loading data for error analysis...", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
oof = np.load('oof_predictions.npy')

# Compute ranking error:
# For y=1: error is (1 - oof)
# For y=0: error is oof
errors = np.where(y == 1, 1.0 - oof, oof)

print(f"Mean error: {errors.mean():.4f}, median error: {np.median(errors):.4f}")

# Look at false negatives: y == 1, but oof < 0.20
fn_mask = (y == 1) & (oof < 0.20)
print(f"Severe False Negatives (y=1, oof < 0.20): {fn_mask.sum()} ({fn_mask.sum() / (y == 1).sum():.2%})")

# Look at false positives: y == 0, but oof > 0.80
fp_mask = (y == 0) & (oof > 0.80)
print(f"Severe False Positives (y=0, oof > 0.80): {fp_mask.sum()} ({fp_mask.sum() / (y == 0).sum():.2%})")

print("\nProfile of Severe False Negatives:")
print(train[fn_mask][['Annual_Income_USD', 'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level', 'Daily_Commute_km', 'Home_Charging_Possible']].describe())

print("\nProfile of Severe False Positives:")
print(train[fp_mask][['Annual_Income_USD', 'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level', 'Daily_Commute_km', 'Home_Charging_Possible']].describe())
