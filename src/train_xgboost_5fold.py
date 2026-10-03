import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("Loading data for 5-fold XGBoost...", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

def build_features(df):
    X = pd.DataFrame(index=df.index)
    
    # Raw features
    for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
              'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
        X[c] = df[c]
        
    for c in raw_cat_cols:
        X[c] = df[c].astype('category')
        
    # Chris Deotte Secret Recipe Buy Score
    inc = df['Annual_Income_USD'].values / 100000.0
    env = df['Environmental_Concern_Level'].values
    sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx
    
    # Deterministic Boundary Flags
    X['cliff_170537'] = (df['Annual_Income_USD'] >= 170537).astype(np.int8)
    X['zero_region'] = (X['buy_score'] < 0.70260).astype(np.int8)
    X['high_region'] = (X['buy_score'] > 7.03543).astype(np.int8)
    
    # Key ratios & interactions
    X['income_per_commute'] = df['Annual_Income_USD'] / (df['Daily_Commute_km'] + 1.0)
    X['stations_total'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    X['env_x_sub'] = env * sub
    X['stations_diff'] = df['Charging_Stations_Near_Home'] - df['Charging_Stations_Near_Work']
    
    # Digit decomposition
    for d in [1, 2, 3, 4]:
        X[f'inc_digit_{d}'] = (df['Annual_Income_USD'] // (10**d)) % 10
    X['commute_digit_1'] = (df['Daily_Commute_km'].astype(int) // 10) % 10
    X['commute_digit_0'] = df['Daily_Commute_km'].astype(int) % 10
    X['age_digit_1'] = (df['Age'] // 10) % 10
    X['age_digit_0'] = df['Age'] % 10
    
    return X

print("Engineering features...", flush=True)
X_train = build_features(train)
X_test = build_features(test)

print(f"Features count: {X_train.shape[1]}", flush=True)

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train), dtype=np.float32)
test_preds_xgb = np.zeros(len(test), dtype=np.float32)
fold_scores = []

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.04,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'nthread': -1
}

dtest = xgb.DMatrix(X_test, enable_categorical=True)

start_time = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"\n--- Training XGBoost Fold {fold + 1}/5 ---", flush=True)
    f_t0 = time.time()
    
    dtrain = xgb.DMatrix(X_train.iloc[train_idx], label=y[train_idx], enable_categorical=True)
    dval = xgb.DMatrix(X_train.iloc[val_idx], label=y[val_idx], enable_categorical=True)
    
    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=150
    )
    
    val_preds = bst.predict(dval)
    oof_xgb[val_idx] = val_preds
    fold_auc = roc_auc_score(y[val_idx], val_preds)
    fold_scores.append(fold_auc)
    print(f"Fold {fold + 1} Best AUC: {fold_auc:.6f} in {time.time() - f_t0:.1f}s", flush=True)
    
    test_preds_xgb += bst.predict(dtest) / 5.0

total_xgb_auc = roc_auc_score(y, oof_xgb)
print(f"\n=======================================================", flush=True)
print(f"*** XGBoost 5-Fold OOF ROC-AUC: {total_xgb_auc:.6f} ***", flush=True)
print(f"Fold scores: {[round(s, 6) for s in fold_scores]}")
print(f"Total training time: {time.time() - start_time:.1f}s", flush=True)
print(f"=======================================================\n", flush=True)

np.save('xgb_oof.npy', oof_xgb)
np.save('xgb_test_preds.npy', test_preds_xgb)

# Now optimize ensemble with unzipped LGB OOF
oof_lgb = np.load('oof_predictions.npy')
test_lgb = np.load('test_predictions.npy')
auc_lgb = roc_auc_score(y, oof_lgb)
print(f"LGB Baseline OOF AUC: {auc_lgb:.6f}", flush=True)

# Rank transform both
r_lgb_oof = rankdata(oof_lgb) / len(oof_lgb)
r_xgb_oof = rankdata(oof_xgb) / len(oof_xgb)

r_lgb_test = rankdata(test_lgb) / len(test_lgb)
r_xgb_test = rankdata(test_preds_xgb) / len(test_preds_xgb)

best_auc = 0.0
best_w = 0.0
for w in np.linspace(0.0, 1.0, 41):
    blend_oof = w * r_xgb_oof + (1.0 - w) * r_lgb_oof
    score = roc_auc_score(y, blend_oof)
    if score > best_auc:
        best_auc = score
        best_w = w

print(f"\n*** OPTIMAL BLEND OOF ROC-AUC: {best_auc:.6f} ***")
print(f"Optimal weights: {best_w:.2f} XGB + {1.0 - best_w:.2f} LGB")
print(f"Gain over standalone LGB: {best_auc - auc_lgb:+.6f}")

# Generate ensemble test predictions
ensemble_test = best_w * r_xgb_test + (1.0 - best_w) * r_lgb_test

# Apply deterministic post-processing boundaries
cliff_test = test['Annual_Income_USD'] >= 170537
zero_test = X_test['buy_score'] < 0.70260
high_test = X_test['buy_score'] > 7.03543

ensemble_test[cliff_test] = 1.0
ensemble_test[zero_test] = 0.0
ensemble_test[high_test] = 1.0

sub_ensemble = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': ensemble_test
})
sub_ensemble.to_csv('submissions/submission_best_ensemble_lgb_xgb.csv', index=False)
print("Saved submissions/submission_best_ensemble_lgb_xgb.csv", flush=True)

# Verify submission
assert len(sub_ensemble) == 286571
assert not sub_ensemble['Will_Buy_EV'].isnull().any()
print("Verification PASSED: submission_best_ensemble_lgb_xgb.csv is ready for leaderboard submission!")
