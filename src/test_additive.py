import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import time

print("Loading data for Additive Model Test...", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

# Build feature set
X = pd.DataFrame(index=train.index)
for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
          'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
    X[c] = train[c]

for c in raw_cat_cols:
    X[c] = train[c].astype('category')

# Add secret recipe buy score
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
X['age_digit_1'] = (train['Age'] // 10) % 10
X['age_digit_0'] = train['Age'] % 10
X['commute_dec_1'] = (train['Daily_Commute_km'] * 10).astype(int) % 10

# Constraints: list of column names
interaction_constraints = [[c] for c in X.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X, y))

dtrain = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx], enable_categorical=True)
dval = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx], enable_categorical=True)

# 2. Additive XGBoost (forbidding column combinations via interaction_constraints)
print("\n--- Training Additive XGBoost (interaction_constraints: [[col]]) ---", flush=True)
params_add = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_bin': 1024,
    'learning_rate': 0.05,
    'max_depth': 8,
    'interaction_constraints': interaction_constraints,
    'subsample': 0.8,
    'random_state': 42,
    'nthread': -1
}
t0 = time.time()
bst_add = xgb.train(params_add, dtrain, num_boost_round=3000, evals=[(dval, 'val')],
                    early_stopping_rounds=60, verbose_eval=300)
preds_add = bst_add.predict(dval)
auc_add = roc_auc_score(y[val_idx], preds_add)
print(f"\n*** Additive XGBoost Fold 0 AUC: {auc_add:.6f} in {time.time() - t0:.1f}s ***", flush=True)

# Blend with unzipped LGB
oof_lgb = np.load('oof_predictions.npy')
preds_lgb = oof_lgb[val_idx]
auc_lgb = roc_auc_score(y[val_idx], preds_lgb)
print(f"LGB Baseline Fold 0 AUC: {auc_lgb:.6f}", flush=True)

r_add = rankdata(preds_add) / len(preds_add)
r_lgb = rankdata(preds_lgb) / len(preds_lgb)

print("\n--- Evaluating Additive + LGB Blend ---")
best_b_auc = 0.0
best_w = 0.0
for w in np.linspace(0.0, 1.0, 21):
    b = w * r_add + (1 - w) * r_lgb
    score = roc_auc_score(y[val_idx], b)
    if score > best_b_auc:
        best_b_auc = score
        best_w = w
    print(f"Weight Additive {w:.2f} + LGB {1-w:.2f} -> AUC: {score:.6f}")

print(f"\n*** BEST BLEND AUC: {best_b_auc:.6f} with Additive Weight {best_w:.2f} (Gain: {best_b_auc - auc_lgb:+.6f}) ***")
