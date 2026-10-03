import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import time

print("=== STARTING QUANTIZATION LADDER 5-FOLD LIGHTGBM PIPELINE ===", flush=True)

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

def build_quantization_base_features(df):
    X = pd.DataFrame(index=df.index)
    
    # 1. Base numeric
    for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
              'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
        X[c] = df[c]
        
    for c in raw_cat_cols:
        X[c] = df[c].astype('category').cat.codes
        
    # 2. Chris Deotte Buying Recipe
    inc = df['Annual_Income_USD'].values / 100000.0
    env = df['Environmental_Concern_Level'].values
    sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx
    
    # 3. Boundaries & ratios
    X['cliff_170537'] = (df['Annual_Income_USD'] >= 170537).astype(np.int8)
    X['income_per_commute'] = df['Annual_Income_USD'] / (df['Daily_Commute_km'] + 1.0)
    X['stations_total'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    
    # 4. Digit features
    for d in [1, 2, 3, 4]:
        X[f'inc_digit_{d}'] = (df['Annual_Income_USD'] // (10**d)) % 10
    X['commute_digit_1'] = (df['Daily_Commute_km'].astype(int) // 10) % 10
    X['commute_digit_0'] = df['Daily_Commute_km'].astype(int) % 10
    X['commute_dec_1'] = (df['Daily_Commute_km'] * 10).astype(int) % 10
    
    # 5. QUANTIZATION LADDER RUNGS (CTGAN Mode Keys)
    # Income Ladder: 100, 500, 1000, 5000
    X['inc_q100'] = df['Annual_Income_USD'] // 100
    X['inc_q500'] = df['Annual_Income_USD'] // 500
    X['inc_q1000'] = df['Annual_Income_USD'] // 1000
    X['inc_q5000'] = df['Annual_Income_USD'] // 5000
    
    # Remainders (offset from quantization mode)
    X['inc_rem100'] = df['Annual_Income_USD'] % 100
    X['inc_rem1000'] = df['Annual_Income_USD'] % 1000
    
    # Commute Ladder: 1km, 5km, 10km
    X['commute_q1'] = df['Daily_Commute_km'].astype(int)
    X['commute_q5'] = (df['Daily_Commute_km'] // 5).astype(int)
    X['commute_rem1'] = df['Daily_Commute_km'] - df['Daily_Commute_km'].astype(int)
    
    # Age Ladder: 5 years, 10 years
    X['age_q5'] = (df['Age'] // 5).astype(int)
    X['age_q10'] = (df['Age'] // 10).astype(int)
    
    # 6. Frequency encodings of keys
    for c in ['inc_q100', 'inc_q1000', 'commute_q1', 'Annual_Income_USD', 'Daily_Commute_km', 'Age']:
        freq = X[c].value_counts(normalize=True)
        X[f'{c}_freq'] = X[c].map(freq).values
        
    return X

print("Building base features with Quantization Ladder...", flush=True)
X_train_base = build_quantization_base_features(train)
X_test_base = build_quantization_base_features(test)

print(f"Base feature matrix shape: {X_train_base.shape}", flush=True)

# 7. Strictly Leaked-Free Out-Of-Fold Smoothed Target Encoding on Ladder Keys
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ladder_te_cols = ['inc_q100', 'inc_q500', 'inc_q1000', 'commute_q1', 'commute_q5', 'age_q5']
smoothing_factors = [10.0, 50.0]

oof_te_train = pd.DataFrame(index=train.index)
te_test = pd.DataFrame(index=test.index)

for c in ladder_te_cols:
    for s in smoothing_factors:
        col_name = f'{c}_te_s{int(s)}'
        oof_te_train[col_name] = 0.0
        te_test[col_name] = 0.0

global_mean = y_train.mean()

print("Calculating strict out-of-fold smoothed target encodings on ladder rungs...", flush=True)
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train)):
    y_tr = y_train[tr_idx]
    
    for c in ladder_te_cols:
        tr_series = X_train_base.iloc[tr_idx][c]
        va_series = X_train_base.iloc[va_idx][c]
        te_series = X_test_base[c]
        
        # Calculate stats on training fold ONLY
        stats = pd.DataFrame({'key': tr_series, 'y': y_tr}).groupby('key')['y'].agg(['count', 'mean'])
        
        for s in smoothing_factors:
            col_name = f'{c}_te_s{int(s)}'
            # Smoothed mean formula: (count * mean + s * global_mean) / (count + s)
            smoothed_map = (stats['count'] * stats['mean'] + s * global_mean) / (stats['count'] + s)
            
            oof_te_train.loc[va_idx, col_name] = va_series.map(smoothed_map).fillna(global_mean).values
            te_test[col_name] += te_series.map(smoothed_map).fillna(global_mean).values / 5.0

# Combine base features and ladder target encodings
X_train_full = pd.concat([X_train_base, oof_te_train], axis=1)
X_test_full = pd.concat([X_test_base, te_test], axis=1)

print(f"Total engineered features with Quantization Ladder: {X_train_full.shape[1]} columns", flush=True)

# 8. Train 5-Fold LightGBM
params_lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

oof_preds_ladder = np.zeros(len(train), dtype=np.float32)
test_preds_ladder = np.zeros(len(test), dtype=np.float32)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train)):
    X_tr, y_tr = X_train_full.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train_full.iloc[va_idx], y_train[va_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=200)]
    
    model = lgb.train(
        params_lgb,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=callbacks
    )
    
    oof_preds_ladder[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(y_va, oof_preds_ladder[va_idx])
    print(f"Quantization Ladder Fold {fold+1} Validation AUC: {fold_auc:.6f}", flush=True)
    
    test_preds_ladder += model.predict(X_test_full, num_iteration=model.best_iteration) / 5.0

total_ladder_auc = roc_auc_score(y_train, oof_preds_ladder)
print(f"\n*** Quantization Ladder 5-Fold LightGBM OOF ROC-AUC: {total_ladder_auc:.6f} in {time.time() - t0:.1f}s ***", flush=True)

np.save('ladder_lgb_oof.npy', oof_preds_ladder)
np.save('ladder_lgb_test_preds.npy', test_preds_ladder)
print("Saved ladder_lgb_oof.npy and ladder_lgb_test_preds.npy successfully!", flush=True)
