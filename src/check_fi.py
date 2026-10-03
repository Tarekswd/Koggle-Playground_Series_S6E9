import joblib
import pandas as pd

d = joblib.load('lgb_ev_model.joblib')
m0 = d['models'][0]
cols = d['feature_columns_per_fold'][0]
fi = pd.Series(m0.feature_importances_, index=cols).sort_values(ascending=False)
print("Top 25 feature importances of LightGBM fold 0:")
print(fi.head(25))
