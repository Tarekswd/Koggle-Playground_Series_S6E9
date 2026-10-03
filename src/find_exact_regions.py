import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train = pd.read_csv('playground-series-s6e9/train.csv')
test = pd.read_csv('playground-series-s6e9/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values

score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

print("Looking for deterministic subsets:")
# Check min score where y == 1
min_score_pos = score[y == 1].min()
max_score_neg = score[y == 0].max()
print(f"Min score for y=1: {min_score_pos:.5f}")
print(f"Max score for y=0: {max_score_neg:.5f}")

# How many rows are below min_score_pos?
zeros = (score < min_score_pos).sum()
print(f"Rows with score < {min_score_pos:.5f}: {zeros} (100% are y=0!)")

# How many rows are above max_score_neg?
ones = (score > max_score_neg).sum()
print(f"Rows with score > {max_score_neg:.5f}: {ones} (100% are y=1!)")

# Check if there are exact duplicates between train and test!
feature_cols = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
train_dup = train.duplicated(subset=feature_cols).sum()
print(f"\nTrain duplicate feature rows: {train_dup}")

# Check train/test overlap
combined = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
total_dup = combined.duplicated(subset=feature_cols).sum()
print(f"Total duplicates across train+test: {total_dup}")

# Check test rows that appear in train
train_tuples = set(tuple(x) for x in train[feature_cols].values)
test_tuples = [tuple(x) for x in test[feature_cols].values]
in_train = sum(1 for t in test_tuples if t in train_tuples)
print(f"Test rows exactly appearing in train: {in_train} / {len(test)}")
