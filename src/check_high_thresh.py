import numpy as np
import pandas as pd

oof = np.load('oof_predictions.npy')
train = pd.read_csv('playground-series-s6e9/train.csv', usecols=['Will_Buy_EV'])
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

for t in [0.95, 0.96, 0.97, 0.98, 0.985, 0.99]:
    mask = oof > t
    pos = y[mask].sum()
    total = mask.sum()
    print(f"OOF > {t:.3f}: count={total}, positive={pos}, rate={pos/total:.5f}, errors={total - pos}")
