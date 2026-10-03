import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import time

print("=== STARTING 5-FOLD CATBOOST MODEL FOR 3-WAY ENSEMBLE ===", flush=True)

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y_train = (train['Will_Buy_EV'] == 'Yes').astype(int).values

raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

def build_catboost_features(df):
    X = pd.DataFrame(index=df.index)
    # Raw numeric
    for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
              'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
        X[c] = df[c]
        
    # Ordinal numeric for categoricals (avoids slow string permutations)
    for c in raw_cat_cols:
        X[c] = df[c].astype('category').cat.codes
        
    # Chris Deotte Buying Recipe
    inc = df['Annual_Income_USD'].values / 100000.0
    env = df['Environmental_Concern_Level'].values
    sub = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high_anx = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    X['buy_score'] = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx
    
    # Boundary and interaction features
    X['cliff_170537'] = (df['Annual_Income_USD'] >= 170537).astype(np.int8)
    X['income_per_commute'] = df['Annual_Income_USD'] / (df['Daily_Commute_km'] + 1.0)
    X['stations_total'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    X['home_chg_num'] = (df['Home_Charging_Possible'] == 'Yes').astype(float)
    X['sub_num'] = sub
    
    # Digit features
    for d in [1, 2, 3, 4]:
        X[f'inc_digit_{d}'] = (df['Annual_Income_USD'] // (10**d)) % 10
    X['commute_digit_1'] = (df['Daily_Commute_km'].astype(int) // 10) % 10
    X['commute_digit_0'] = df['Daily_Commute_km'].astype(int) % 10
    X['commute_dec_1'] = (df['Daily_Commute_km'] * 10).astype(int) % 10
    
    # Frequency encoding
    for c in ['Annual_Income_USD', 'Daily_Commute_km', 'Age']:
        freq = df[c].value_counts(normalize=True)
        X[f'{c}_freq'] = df[c].map(freq).values
        
    return X

print("Engineering features for train and test...", flush=True)
X_train = build_catboost_features(train)
X_test = build_catboost_features(test)

print(f"Feature matrix shape: {X_train.shape}", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cb = np.zeros(len(train), dtype=np.float32)
test_preds_cb = np.zeros(len(test), dtype=np.float32)

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.07,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        thread_count=-1,
        verbose=200
    )
    
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40, verbose=200)
    
    oof_cb[val_idx] = cb.predict_proba(X_va)[:, 1]
    fold_auc = roc_auc_score(y_va, oof_cb[val_idx])
    print(f"CatBoost Fold {fold+1} Validation AUC: {fold_auc:.6f}", flush=True)
    
    test_preds_cb += cb.predict_proba(X_test)[:, 1] / 5.0

total_cb_auc = roc_auc_score(y_train, oof_cb)
print(f"\n*** 5-Fold CatBoost OOF ROC-AUC: {total_cb_auc:.6f} in {time.time() - t0:.1f}s ***", flush=True)

np.save('catboost_oof.npy', oof_cb)
np.save('catboost_test_preds.npy', test_preds_cb)
print("Saved catboost_oof.npy and catboost_test_preds.npy successfully!", flush=True)
