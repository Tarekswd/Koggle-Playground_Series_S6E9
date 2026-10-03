import pandas as pd
import numpy as np

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')

y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

print("Income max:", train['Annual_Income_USD'].max(), test['Annual_Income_USD'].max())

over_thresh_train = train[train['Annual_Income_USD'] >= 170537]
print(f"Train rows with income >= 170537: {len(over_thresh_train)}")
print("Target values for these rows:")
print(over_thresh_train['Will_Buy_EV'].value_counts())

over_thresh_test = test[test['Annual_Income_USD'] >= 170537]
print(f"Test rows with income >= 170537: {len(over_thresh_test)}")

# Let's check low income threshold too!
print("\nChecking lowest income thresholds:")
for t in [30000, 32000, 35000, 38000, 40000, 42000, 45000]:
    sub = train[train['Annual_Income_USD'] <= t]
    pos = (sub['Will_Buy_EV'] == 'Yes').sum()
    print(f"Income <= {t}: count={len(sub)}, positive={pos}, rate={pos/len(sub):.5f}")

# Let's check the Chris Deotte Buy Score on train:
inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
subsidy = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values

score = 1.2 * inc + 0.6 * env + 2.0 * subsidy - 1.0 * med_anx - 3.0 * high_anx

print(f"\nScore summary:")
print(f"Min score: {score.min():.4f}, Max score: {score.max():.4f}")

# Check accuracy / deterministic behavior around thresholds of score
print("\nScore bins vs positive rate:")
score_bins = pd.cut(score, bins=np.linspace(score.min(), score.max(), 25))
print(pd.Series(y).groupby(score_bins, observed=False).agg(['count', 'mean']))

# Also check if score >= 5.5:
over_55 = (score >= 5.5)
print(f"\nScore >= 5.5: count={over_55.sum()}, positive rate={y[over_55].mean():.5f}")
print(f"Score < 5.5: count={(~over_55).sum()}, positive rate={y[~over_55].mean():.5f}")
