import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import roc_auc_score

print("Loading train targets...", flush=True)
train_df = pd.read_csv('playground-series-s6e9/train.csv', usecols=['id', 'Will_Buy_EV'])
y_true = (train_df['Will_Buy_EV'] == 'Yes').astype(int).values
print(f"Target count: {len(y_true)}, positive rate: {y_true.mean():.5f}", flush=True)

oof = np.load('oof_predictions.npy')
print(f"oof_predictions.npy shape: {oof.shape}", flush=True)
print(f"oof min: {oof.min():.5f}, max: {oof.max():.5f}, mean: {oof.mean():.5f}", flush=True)

auc = roc_auc_score(y_true, oof)
print(f"\n=======================================================", flush=True)
print(f"*** BASELINE oof_predictions.npy ROC-AUC: {auc:.6f} ***", flush=True)
print(f"=======================================================\n", flush=True)

test_preds = np.load('test_predictions.npy')
print(f"test_predictions.npy shape: {test_preds.shape}", flush=True)

print("\nInspecting lgb_ev_model.joblib...", flush=True)
lgb_model = joblib.load('lgb_ev_model.joblib')
print(f"Model object type: {type(lgb_model)}", flush=True)
if hasattr(lgb_model, 'feature_name_'):
    print(f"Feature count: {len(lgb_model.feature_name_)}")
    print(f"Features: {lgb_model.feature_name_}")
if hasattr(lgb_model, 'feature_importances_'):
    fi = pd.Series(lgb_model.feature_importances_, index=lgb_model.feature_name_).sort_values(ascending=False)
    print("\nFeature importances top 15:")
    print(fi.head(15))

print("\nInspecting heart_disease_xgb_model.json...", flush=True)
with open('heart_disease_xgb_model.json', 'r') as f:
    h_data = json.load(f)
learner = h_data.get('learner', {})
feat_names = learner.get('feature_names', [])
print(f"Heart disease features: {feat_names}", flush=True)
