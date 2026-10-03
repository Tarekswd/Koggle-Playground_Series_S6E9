import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

oof = np.load('oof_predictions.npy')
train = pd.read_csv('playground-series-s6e9/train.csv', usecols=['Will_Buy_EV'])
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

base_auc = roc_auc_score(y, oof)
print(f"Base OOF AUC: {base_auc:.6f}")

# Look at probability calibration and power transforms for rank-preserving ROC-AUC
# ROC-AUC is purely rank-based. Monotonic transformations don't change single model AUC,
# BUT in blending, non-linear calibration or rank transformations drastically improve blend AUC!

# Let's check how many samples have extreme probabilities
print(f"OOF < 0.01: {(oof < 0.01).sum()}, positive rate: {y[oof < 0.01].mean():.5f}")
print(f"OOF < 0.001: {(oof < 0.001).sum()}, positive rate: {y[oof < 0.001].mean():.5f}")
print(f"OOF > 0.99: {(oof > 0.99).sum()}, positive rate: {y[oof > 0.99].mean():.5f}")
print(f"OOF > 0.999: {(oof > 0.999).sum()}, positive rate: {y[oof > 0.999].mean():.5f}")
