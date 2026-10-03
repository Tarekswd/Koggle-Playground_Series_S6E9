import numpy as np
import pandas as pd

oof = np.load('oof_predictions.npy')
train = pd.read_csv('playground-series-s6e9/train.csv', usecols=['Will_Buy_EV'])
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

for t in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]:
    mask = oof < t
    pos = y[mask].sum()
    total = mask.sum()
    print(f"OOF < {t:.4f}: count={total}, positive={pos}, rate={pos/total:.6f}, false_zeros={pos}")

# Find max threshold with exactly 0 positives
sorted_indices = np.argsort(oof)
first_pos_idx = np.where(y[sorted_indices] == 1)[0][0]
zero_error_thresh = oof[sorted_indices[first_pos_idx]]
print(f"\nZero error lower threshold: {zero_error_thresh:.7f}, count: {first_pos_idx}")
