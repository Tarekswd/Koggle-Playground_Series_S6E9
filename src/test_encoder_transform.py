import joblib
import pandas as pd
import numpy as np

d = joblib.load('lgb_ev_model.joblib')
freq_maps = d['frequency_maps']
encoders_f0 = d['encoders_per_fold'][0]
cols_f0 = d['feature_columns_per_fold'][0]

print("Cols in fold 0:", len(cols_f0))

# Load a small sample of train
train = pd.read_csv('playground-series-s6e9/train.csv', nrows=100)

# Check how frequency columns are created:
for col, fmap in freq_maps.items():
    s = train[col].map(fmap)
    print(f"Mapped {col}_freq: nulls = {s.isnull().sum()}")

# Check how encoders transform
try:
    print("Testing encoder 0 transform on sample...")
    res0 = encoders_f0[0].transform(train)
    print(f"Encoder 0 transform success, output shape: {res0.shape}")
    res1 = encoders_f0[1].transform(train)
    print(f"Encoder 1 transform success, output shape: {res1.shape}")
except Exception as e:
    print(f"Encoder transform error: {e}")
