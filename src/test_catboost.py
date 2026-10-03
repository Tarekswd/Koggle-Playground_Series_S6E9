import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

print("Loading data...", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

features = [
    'Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
    'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
    'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
    'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level'
]

X = train[features].copy()
cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
for col in cat_cols:
    X[col] = X[col].astype(str)

# Add digit features
for d in [1, 2, 3, 4]:
    X[f'inc_digit_{d}'] = (train['Annual_Income_USD'] // (10**d)) % 10
X[f'commute_digit_1'] = (train['Daily_Commute_km'].astype(int) // 10) % 10
X[f'commute_digit_0'] = train['Daily_Commute_km'].astype(int) % 10

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X, y))

cb = CatBoostClassifier(
    iterations=600,
    learning_rate=0.08,
    depth=6,
    eval_metric='AUC',
    cat_features=cat_cols,
    random_seed=42,
    thread_count=-1,
    verbose=100
)

t0 = time.time()
cb.fit(
    X.iloc[train_idx], y[train_idx],
    eval_set=(X.iloc[val_idx], y[val_idx]),
    early_stopping_rounds=40
)
t1 = time.time()
val_preds = cb.predict_proba(X.iloc[val_idx])[:, 1]
score = roc_auc_score(y[val_idx], val_preds)
print(f"CatBoost Fold 0 AUC: {score:.6f} in {t1 - t0:.1f}s", flush=True)

# Also test correlation with unzipped LGB oof on val_idx
oof_lgb = np.load('oof_predictions.npy')
print(f"LGB Fold 0 AUC: {roc_auc_score(y[val_idx], oof_lgb[val_idx]):.6f}")

from scipy.stats import spearmanr
corr, _ = spearmanr(val_preds, oof_lgb[val_idx])
print(f"Spearman rank correlation between CatBoost and LGB: {corr:.5f}")

# Blend CatBoost + LGB
from scipy.stats import rankdata
r_cb = rankdata(val_preds) / len(val_preds)
r_lgb = rankdata(oof_lgb[val_idx]) / len(val_idx)

for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    b = w * r_cb + (1 - w) * r_lgb
    print(f"Blend Weight CB {w:.1f} + LGB {1-w:.1f} -> AUC: {roc_auc_score(y[val_idx], b):.6f}")
