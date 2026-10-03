import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

# Let's inspect all candidate features normalized to sensible scales:
# Base variables:
inc = train['Annual_Income_USD'] / 100000.0
env = train['Environmental_Concern_Level']
sub = (train['Subsidy_Available'] == 'Yes').astype(float)
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float)
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float)
home_chg = (train['Home_Charging_Possible'] == 'Yes').astype(float)
commute = train['Daily_Commute_km'] / 100.0
stations_home = train['Charging_Stations_Near_Home'] / 10.0
stations_work = train['Charging_Stations_Near_Work'] / 10.0
cars = train['Number_of_Cars_Owned']
age = train['Age'] / 100.0

# Build candidate matrix
X_candidates = pd.DataFrame({
    'inc': inc,
    'env': env,
    'sub': sub,
    'med_anx': med_anx,
    'high_anx': high_anx,
    'home_chg': home_chg,
    'commute': commute,
    'stations_home': stations_home,
    'stations_work': stations_work,
    'cars': cars,
    'age': age,
    # One hot city and car type
    'city_suburban': (train['City_Type'] == 'Suburban').astype(float),
    'city_rural': (train['City_Type'] == 'Rural').astype(float),
    'car_truck': (train['Current_Car_Type'] == 'Truck').astype(float),
    'car_suv': (train['Current_Car_Type'] == 'SUV').astype(float),
    'car_hatch': (train['Current_Car_Type'] == 'Hatchback').astype(float),
})

# Add interactions
X_candidates['inc_x_sub'] = inc * sub
X_candidates['home_chg_x_stations'] = home_chg * stations_home
X_candidates['commute_x_high_anx'] = commute * high_anx
X_candidates['inc_x_high_anx'] = inc * high_anx
X_candidates['env_x_sub'] = env * sub

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_candidates, y)
preds = clf.predict_proba(X_candidates)[:, 1]
auc = roc_auc_score(y, preds)
print(f"Logistic Regression on extended formula candidates AUC: {auc:.6f}")

coef_series = pd.Series(clf.coef_[0], index=X_candidates.columns).sort_values(ascending=False)
print("\nFitted coefficients:")
print(coef_series)
