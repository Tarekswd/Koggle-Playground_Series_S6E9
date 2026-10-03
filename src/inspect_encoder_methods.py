import joblib
import pandas as pd

d = joblib.load('lgb_ev_model.joblib')
encoders_f0 = d['encoders_per_fold'][0]
print(f"encoders_f0 type: {type(encoders_f0)}, len: {len(encoders_f0)}")
for i, enc in enumerate(encoders_f0):
    print(f"Encoder {i}: {type(enc)}, methods: {[m for m in dir(enc) if not m.startswith('_')][:8]}")
