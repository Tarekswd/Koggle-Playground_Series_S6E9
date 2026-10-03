import joblib
import pandas as pd
import numpy as np

print("Testing reproduction of the 75-feature pipeline...")

d = joblib.load('lgb_ev_model.joblib')
train = pd.read_csv('playground-series-s6e9/train.csv', nrows=1000)
oof = np.load('oof_predictions.npy')[:1000]

freq_maps = d['frequency_maps']
enc_f0 = d['encoders_per_fold'][0]
cols_f0 = d['feature_columns_per_fold'][0]
mod_f0 = d['models'][0]

print(f"Target columns to reproduce: {len(cols_f0)}")

def build_75_features(df, fold_idx):
    X = pd.DataFrame(index=df.index)
    
    # 1. Base 13 features
    raw_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    
    for c in raw_cols:
        if df[c].dtype == 'object':
            X[c] = df[c].astype('category').cat.codes
        else:
            X[c] = df[c]
            
    # 2. Digit features
    # Age digits: 0, 1
    X['Age_digit0'] = df['Age'] % 10
    X['Age_digit1'] = (df['Age'] // 10) % 10
    
    # Income digits: 0, 1, 2, 3
    inc_int = df['Annual_Income_USD'].astype(int)
    X['Annual_Income_USD_digit0'] = inc_int % 10
    X['Annual_Income_USD_digit1'] = (inc_int // 10) % 10
    X['Annual_Income_USD_digit2'] = (inc_int // 100) % 10
    X['Annual_Income_USD_digit3'] = (inc_int // 1000) % 10
    
    # Daily commute: digit-4, -3, -1, 0, 1
    # Note: commute has up to 4 decimal places in synthetic generation!
    commute_float = df['Daily_Commute_km']
    X['Daily_Commute_km_digit-4'] = (np.round(commute_float * 10000).astype(int)) % 10
    X['Daily_Commute_km_digit-3'] = (np.round(commute_float * 1000).astype(int)) % 10
    X['Daily_Commute_km_digit-1'] = (np.round(commute_float * 10).astype(int)) % 10
    X['Daily_Commute_km_digit0'] = (commute_float.astype(int)) % 10
    X['Daily_Commute_km_digit1'] = (commute_float.astype(int) // 10) % 10
    
    # Stations Near Home: digit-4, 0, 1
    st_home = df['Charging_Stations_Near_Home']
    X['Charging_Stations_Near_Home_digit-4'] = 0  # mostly 0
    X['Charging_Stations_Near_Home_digit0'] = st_home % 10
    X['Charging_Stations_Near_Home_digit1'] = (st_home // 10) % 10
    
    # Stations Near Work: digit-4, 0, 1
    st_work = df['Charging_Stations_Near_Work']
    X['Charging_Stations_Near_Work_digit-4'] = 0
    X['Charging_Stations_Near_Work_digit0'] = st_work % 10
    X['Charging_Stations_Near_Work_digit1'] = (st_work // 10) % 10
    
    # Categorical digit-4 (flags / indicator representations)
    for cat in ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']:
        X[f'{cat}_digit-4'] = 0
        
    # 3. Frequency features
    for c in raw_cols:
        fmap = freq_maps[c]
        # keys in fmap are strings
        X[f'{c}_freq'] = df[c].astype(str).map(fmap).fillna(0.0).values
        
    # 4. Target encodings
    encoders = d['encoders_per_fold'][fold_idx]
    # Encoder 0: smooth_10
    te10 = encoders[0].transform(df[raw_cols])
    for orig_c, col_name in zip(raw_cols, encoders[0].get_feature_names_out()):
        target_col = f"{orig_c}__te_mean_seed_42_smooth_10_inner_n_fold_5"
        X[target_col] = te10[col_name].values
        
    # Encoder 1: smooth_auto
    te_auto = encoders[1].transform(df[raw_cols])
    for orig_c, col_name in zip(raw_cols, encoders[1].get_feature_names_out()):
        target_col = f"{orig_c}__te_mean_seed_42_smooth_auto_inner_n_fold_5"
        X[target_col] = te_auto[col_name].values
        
    # Ensure exact column alignment
    X = X[cols_f0]
    return X

X_sample = build_75_features(train, 0)
print(f"Sample shape: {X_sample.shape}")
print(f"Columns exactly match: {list(X_sample.columns) == cols_f0}")

# Test model predictions on sample
preds_sample = mod_f0.predict_proba(X_sample)[:, 1]
print(f"First 10 sample predictions: {preds_sample[:10]}")
print(f"First 10 OOF predictions:    {oof[:10]}")
print(f"Correlation with OOF: {np.corrcoef(preds_sample, oof)[0, 1]:.6f}")
