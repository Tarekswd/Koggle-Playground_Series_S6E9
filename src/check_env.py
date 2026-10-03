import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import roc_auc_score

print(f"Python: {sys.version}", flush=True)

for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'xgboost', 'catboost', 'torch', 'joblib']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'available')
        print(f"  {pkg}: {v}", flush=True)
    except Exception as e:
        print(f"  {pkg}: NOT AVAILABLE ({e})", flush=True)

try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"Torch check error: {e}", flush=True)

print("\n--- DATA INSPECTION ---", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
sub = pd.read_csv('playground-series-s6e9/sample_submission.csv')

print(f"Train shape: {train.shape}", flush=True)
print(f"Test shape: {test.shape}", flush=True)
print(f"Sub shape: {sub.shape}", flush=True)
print(f"Train columns:\n{list(train.columns)}", flush=True)
print(f"Train dtypes:\n{train.dtypes}", flush=True)
print(f"Train missing values count: {train.isnull().sum().sum()}", flush=True)
print(f"Test missing values count: {test.isnull().sum().sum()}", flush=True)

print("\nTarget summary:", flush=True)
print(train['Will_Buy_EV'].value_counts(dropna=False), flush=True)
print(f"Target rate: {train['Will_Buy_EV'].mean():.5f}", flush=True)

print("\n--- UNZIPPED ARTIFACTS ---", flush=True)
if os.path.exists('oof_predictions.npy'):
    oof = np.load('oof_predictions.npy')
    print(f"oof_predictions.npy shape: {oof.shape}, min: {oof.min():.5f}, max: {oof.max():.5f}, mean: {oof.mean():.5f}", flush=True)
    if len(oof) == len(train):
        score = roc_auc_score(train['Will_Buy_EV'], oof)
        print(f"*** BASELINE oof_predictions.npy AUC: {score:.6f} ***", flush=True)

if os.path.exists('test_predictions.npy'):
    tpreds = np.load('test_predictions.npy')
    print(f"test_predictions.npy shape: {tpreds.shape}, min: {tpreds.min():.5f}, max: {tpreds.max():.5f}", flush=True)

if os.path.exists('lgb_ev_model.joblib'):
    print("Loading lgb_ev_model.joblib...", flush=True)
    lgb_obj = joblib.load('lgb_ev_model.joblib')
    print(f"lgb_ev_model.joblib type: {type(lgb_obj)}", flush=True)
    if hasattr(lgb_obj, 'feature_name_'):
        print(f"LGB feature names: {lgb_obj.feature_name_}", flush=True)
    elif hasattr(lgb_obj, 'feature_names_in_'):
        print(f"LGB feature_names_in_: {lgb_obj.feature_names_in_}", flush=True)
    elif isinstance(lgb_obj, dict):
        print(f"Keys: {list(lgb_obj.keys())}", flush=True)

if os.path.exists('heart_disease_xgb_model.json'):
    with open('heart_disease_xgb_model.json', 'r') as f:
        xgb_info = json.load(f)
    print(f"heart_disease_xgb_model.json keys: {list(xgb_info.keys())[:5]}", flush=True)
    learner = xgb_info.get('learner', {})
    features = learner.get('feature_names', [])
    print(f"heart_disease feature names: {features}", flush=True)
