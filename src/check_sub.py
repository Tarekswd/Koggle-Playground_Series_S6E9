import pandas as pd
import numpy as np

sub = pd.read_csv('playground-series-s6e9/sample_submission.csv')
print("Sample submission head:")
print(sub.head())
print("Sample submission shape:", sub.shape)

test = pd.read_csv('playground-series-s6e9/test.csv', usecols=['id'])
print(f"IDs match test exactly: {(sub['id'] == test['id']).all()}")

tpreds = np.load('test_predictions.npy')
print(f"test_predictions shape: {tpreds.shape}")
print(f"test_predictions min: {tpreds.min():.5f}, max: {tpreds.max():.5f}, mean: {tpreds.mean():.5f}")
print(f"NaN count in test_predictions: {np.isnan(tpreds).sum()}")
