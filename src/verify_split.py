import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("Loading model and data...", flush=True)
d = joblib.load('lgb_ev_model.joblib')
m0 = d['models'][0]
cols0 = d['feature_columns_per_fold'][0]
freq_maps = d['frequency_maps']
encs0 = d['encoders_per_fold'][0]

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
oof = np.load('oof_predictions.npy')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train, y))

# Let's inspect how the original model transformed val_idx
print(f"OOF AUC on val_idx: {roc_auc_score(y[val_idx], oof[val_idx]):.6f}")
