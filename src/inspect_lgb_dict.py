import joblib

d = joblib.load('lgb_ev_model.joblib')
print("Keys in lgb_ev_model.joblib:", list(d.keys()))
for k, v in d.items():
    print(f"Key: {k}, Type: {type(v)}")
    if isinstance(v, (list, tuple)):
        print(f"  Length: {len(v)}, first elem type: {type(v[0]) if len(v)>0 else None}")
    elif isinstance(v, dict):
        print(f"  Subkeys: {list(v.keys())[:5]}")
