import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit

print("Loading data...", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values

score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx
base_margin = 2.1746 * score - 12.2046

# Basic preprocessing
features = [
    'Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
    'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
    'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
    'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level'
]

X = train[features].copy()
cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
for col in cat_cols:
    X[col] = X[col].astype('category')

# Add score as an explicit feature too
X['buy_score'] = score
X['cliff_170537'] = (train['Annual_Income_USD'] >= 170537).astype(int)

# Quick digit decomposition on income, commute, age
for d in [1, 2, 3, 4]:
    X[f'inc_digit_{d}'] = (train['Annual_Income_USD'] // (10**d)) % 10
X[f'commute_digit_1'] = (train['Daily_Commute_km'].astype(int) // 10) % 10
X[f'commute_digit_0'] = train['Daily_Commute_km'].astype(int) % 10

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_no_margin = np.zeros(len(train))
oof_preds_with_margin = np.zeros(len(train))

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'n_estimators': 1500,
    'verbose': -1,
    'n_jobs': -1
}

# Run on fold 0 first for fast validation
train_idx, val_idx = next(skf.split(X, y))
print(f"Fold 0: train len={len(train_idx)}, val len={len(val_idx)}", flush=True)

# 1. No margin
dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
dval = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], reference=dtrain)

model1 = lgb.train(
    params,
    dtrain,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
preds1 = model1.predict(X.iloc[val_idx])
auc1 = roc_auc_score(y[val_idx], preds1)
print(f"Fold 0 AUC WITHOUT init_score: {auc1:.6f}", flush=True)

# 2. With init_score (base margin)
dtrain_margin = lgb.Dataset(X.iloc[train_idx], label=y[train_idx], init_score=base_margin[train_idx])
dval_margin = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], reference=dtrain_margin, init_score=base_margin[val_idx])

model2 = lgb.train(
    params,
    dtrain_margin,
    valid_sets=[dval_margin],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
# predict returns raw margin adjustments when init_score is used, or probability?
# In LightGBM, predict(raw_score=True) returns raw margin, expit(base_margin + raw_score) gives probability!
raw_preds = model2.predict(X.iloc[val_idx], raw_score=True)
preds2 = expit(base_margin[val_idx] + raw_preds)
auc2 = roc_auc_score(y[val_idx], preds2)
print(f"Fold 0 AUC WITH init_score (base margin): {auc2:.6f}", flush=True)
print(f"Delta: {auc2 - auc1:+.6f}", flush=True)
