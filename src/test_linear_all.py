import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
num_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ]
)

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])

pipe.fit(train, y)
preds = pipe.predict_proba(train)[:, 1]
auc_all_linear = roc_auc_score(y, preds)
print(f"*** ALL FEATURES LOGISTIC REGRESSION ROC-AUC: {auc_all_linear:.6f} ***")

clf = pipe.named_steps['clf']
feature_names = num_cols + list(pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols))
coef_df = pd.DataFrame({'feature': feature_names, 'coef': clf.coef_[0]}).sort_values(by='coef', key=abs, ascending=False)
print("\nTop coefficients:")
print(coef_df.to_string(index=False))
