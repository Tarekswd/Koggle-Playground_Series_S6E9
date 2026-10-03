import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, rankdata
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
    X[col] = X[col].astype('category')

# Add secret recipe buy_score
inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values
X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

# Add digit decomposition
for d in [1, 2, 3, 4]:
    X[f'inc_digit_{d}'] = (train['Annual_Income_USD'] // (10**d)) % 10
X['commute_digit_1'] = (train['Daily_Commute_km'].astype(int) // 10) % 10
X['commute_digit_0'] = train['Daily_Commute_km'].astype(int) % 10

# Add key interactions
X['income_per_commute'] = train['Annual_Income_USD'] / (train['Daily_Commute_km'] + 1.0)
X['stations_total'] = train['Charging_Stations_Near_Home'] + train['Charging_Stations_Near_Work']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X, y))

dtrain = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx], enable_categorical=True)
dval = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx], enable_categorical=True)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'nthread': -1
}

t0 = time.time()
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1200,
    evals=[(dval, 'val')],
    early_stopping_rounds=40,
    verbose_eval=100
)
t1 = time.time()

preds_xgb = bst.predict(dval)
auc_xgb = roc_auc_score(y[val_idx], preds_xgb)
print(f"\nXGBoost Fold 0 AUC: {auc_xgb:.6f} (training time: {t1 - t0:.1f}s)", flush=True)

# Blend with unzipped LGB OOF
oof_lgb = np.load('oof_predictions.npy')
auc_lgb = roc_auc_score(y[val_idx], oof_lgb[val_idx])
print(f"LGB Fold 0 AUC: {auc_lgb:.6f}", flush=True)

corr, _ = spearmanr(preds_xgb, oof_lgb[val_idx])
print(f"Spearman rank correlation: {corr:.5f}", flush=True)

r_xgb = rankdata(preds_xgb) / len(preds_xgb)
r_lgb = rankdata(oof_lgb[val_idx]) / len(val_idx)

best_auc = 0.0
best_w = 0.0
for w in np.linspace(0.0, 1.0, 21):
    blend = w * r_xgb + (1 - w) * r_lgb
    score = roc_auc_score(y[val_idx], blend)
    if score > best_auc:
        best_auc = score
        best_w = w
    print(f"Weight XGB {w:.2f} + LGB {1-w:.2f} -> AUC: {score:.6f}")

print(f"\n*** BEST BLEND AUC: {best_auc:.6f} with XGB weight {best_w:.2f} (Gain over LGB: {best_auc - auc_lgb:+.6f}) ***")
