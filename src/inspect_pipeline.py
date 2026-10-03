import joblib
import pandas as pd
import numpy as np

d = joblib.load('lgb_ev_model.joblib')
print("Keys:", d.keys())
print("frequency_maps keys:", list(d['frequency_maps'].keys()))
for k in list(d['frequency_maps'].keys())[:3]:
    print(f"frequency_maps[{k}] sample items:", list(d['frequency_maps'][k].items())[:5])

print("\nencoders_per_fold length:", len(d['encoders_per_fold']))
print("encoders_per_fold[0]:", type(d['encoders_per_fold'][0]))
if isinstance(d['encoders_per_fold'][0], list):
    print("Length of encoders list in fold 0:", len(d['encoders_per_fold'][0]))
    for enc in d['encoders_per_fold'][0][:5]:
        print("Encoder type:", type(enc), getattr(enc, '__dict__', None))
elif isinstance(d['encoders_per_fold'][0], dict):
    print("Keys in fold 0 encoders:", list(d['encoders_per_fold'][0].keys()))

print("\nFeature columns per fold length:", len(d['feature_columns_per_fold']))
print("Feature columns [0] count:", len(d['feature_columns_per_fold'][0]))
print("Features [0]:", d['feature_columns_per_fold'][0])
