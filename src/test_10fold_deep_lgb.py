import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

print("=== TESTING 10-FOLD DEEP LIGHTGBM CONFIGURATION ===")

train = pd.read_csv('playground-series-s6e9/train.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

# Build unified feature set
def build_deep_features(df):
    X = pd.DataFrame(index=df.index)
    for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
              'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
        X[c] = df[c]
    for c in raw_cat_cols:
        X[c] = df[c].astype('category').cat.codes
        
    inc = df['Annual_Income_USD'].values / 100000.0
    env = df['Environmental_Concern_Level'].values
    sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx
    
    X['cliff_170537'] = (df['Annual_Income_USD'] >= 170537).astype(np.int8)
    X['income_per_commute'] = df['Annual_Income_USD'] / (df['Daily_Commute_km'] + 1.0)
    X['stations_total'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    
    # Digits
    for d in [1, 2, 3, 4]:
        X[f'inc_digit_{d}'] = (df['Annual_Income_USD'] // (10**d)) % 10
    X['commute_digit_1'] = (df['Daily_Commute_km'].astype(int) // 10) % 10
    X['commute_digit_0'] = df['Daily_Commute_km'].astype(int) % 10
    X['commute_dec_1'] = (df['Daily_Commute_km'] * 10).astype(int) % 10
    
    # Quantization rungs
    X['inc_q100'] = df['Annual_Income_USD'] // 100
    X['inc_q500'] = df['Annual_Income_USD'] // 500
    X['inc_q1000'] = df['Annual_Income_USD'] // 1000
    X['inc_q5000'] = df['Annual_Income_USD'] // 5000
    X['inc_rem100'] = df['Annual_Income_USD'] % 100
    X['inc_rem1000'] = df['Annual_Income_USD'] % 1000
    
    X['commute_q1'] = df['Daily_Commute_km'].astype(int)
    X['commute_q5'] = (df['Daily_Commute_km'] // 5).astype(int)
    X['commute_rem1'] = df['Daily_Commute_km'] - df['Daily_Commute_km'].astype(int)
    
    # Frequencies
    for c in ['inc_q100', 'inc_q1000', 'commute_q1', 'Annual_Income_USD', 'Daily_Commute_km', 'Age']:
        freq = X[c].value_counts(normalize=True)
        X[f'{c}_freq'] = X[c].map(freq).values
        
    return X

print("Building features...")
X_train = build_deep_features(train)

# 10-Fold Split
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
tr_idx, va_idx = next(skf.split(X_train, y_train))
print(f"Training on {len(tr_idx)} rows (90%), Validating on {len(va_idx)} rows (10%)...")

# Fold Target Encodings
global_mean = y_train[tr_idx].mean()
for c in ['inc_q100', 'inc_q500', 'inc_q1000', 'commute_q1', 'Annual_Income_USD']:
    stats = pd.DataFrame({'k': X_train.iloc[tr_idx][c], 'y': y_train[tr_idx]}).groupby('k')['y'].agg(['count', 'mean'])
    for s in [10.0, 50.0]:
        smoothed = (stats['count'] * stats['mean'] + s * global_mean) / (stats['count'] + s)
        X_train[f'{c}_te_s{int(s)}'] = X_train[c].map(smoothed).fillna(global_mean).values

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 255,
    'max_depth': 6,
    'min_child_samples': 15,
    'subsample': 0.85,
    'colsample_bytree': 0.35,
    'reg_alpha': 0.08,
    'reg_lambda': 2.0,
    'max_bin': 1024,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

trn_data = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx])
val_data = lgb.Dataset(X_train.iloc[va_idx], label=y_train[va_idx], reference=trn_data)

callbacks = [lgb.early_stopping(stopping_rounds=60, verbose=False), lgb.log_evaluation(period=200)]

t0 = time.time()
model = lgb.train(
    params,
    trn_data,
    num_boost_round=2500,
    valid_sets=[val_data],
    callbacks=callbacks
)

preds_va = model.predict(X_train.iloc[va_idx], num_iteration=model.best_iteration)
auc_val = roc_auc_score(y_train[va_idx], preds_va)

print(f"\n*** 10-Fold Fold 0 Validation ROC-AUC: {auc_val:.6f} in {time.time() - t0:.1f}s ***")
