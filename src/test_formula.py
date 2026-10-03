import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

print("Annual_Income_USD stats:")
print(train['Annual_Income_USD'].describe())

print("\nEnvironmental_Concern_Level values:")
print(train['Environmental_Concern_Level'].value_counts(dropna=False).sort_index())

print("\nSubsidy_Available values:")
print(train['Subsidy_Available'].value_counts(dropna=False))

print("\nRange_Anxiety_Level values:")
print(train['Range_Anxiety_Level'].value_counts(dropna=False))

# Let's map them to formula features:
# income: Annual_Income_USD / 100,000
# concern for planet: Environmental_Concern_Level
# subsidy: 1 if Subsidy_Available == 'Yes' else 0
# medium range worry: 1 if Range_Anxiety_Level == 'Medium' else 0
# high range worry: 1 if Range_Anxiety_Level == 'High' else 0

inc = train['Annual_Income_USD'] / 100000.0
env = train['Environmental_Concern_Level']
sub = (train['Subsidy_Available'] == 'Yes').astype(float)
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float)
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float)

buy_score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

auc_formula = roc_auc_score(y, buy_score)
print(f"\n*** FORMULA RAW SCORE ROC-AUC: {auc_formula:.6f} ***")

# Let's fit a logistic regression on these exact features and see the optimal weights!
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_formula = np.column_stack([inc, env, sub, med_anx, high_anx])
clf = LogisticRegression(max_iter=1000)
clf.fit(X_formula, y)
lr_preds = clf.predict_proba(X_formula)[:, 1]
auc_lr = roc_auc_score(y, lr_preds)
print(f"*** LOGISTIC REGRESSION ON FORMULA FEATURES ROC-AUC: {auc_lr:.6f} ***")
print(f"Learned coefficients: {clf.coef_[0]}")
print(f"Learned intercept: {clf.intercept_[0]}")
print("Relative to inc coef (clf.coef_[0] / clf.coef_[0][0] * 1.2):")
print((clf.coef_[0] / clf.coef_[0][0]) * 1.2)
