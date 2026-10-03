import pandas as pd
import numpy as np

print("=== INSPECTING QUANTIZATION LADDER RUNGS ===")

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

inc = train['Annual_Income_USD'].values
commute = train['Daily_Commute_km'].values
age = train['Age'].values

print(f"Total rows: {len(train)}")
print(f"Income raw unique: {len(np.unique(inc))}")

for q in [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]:
    binned = inc // q
    n_bins = len(np.unique(binned))
    print(f"Income Ladder q={q:5d}: {n_bins:5d} unique bins")

print(f"\nCommute raw unique: {len(np.unique(commute))}")
for q in [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]:
    binned = np.floor(commute / q).astype(int)
    n_bins = len(np.unique(binned))
    print(f"Commute Ladder q={q:4.1f}: {n_bins:5d} unique bins")

# Check if there are exact zero-variance or high-bias bins
for q in [1000, 5000]:
    df_temp = pd.DataFrame({'bin': inc // q, 'y': y})
    rates = df_temp.groupby('bin')['y'].agg(['count', 'mean'])
    print(f"\nIncome q={q} rate stats (min, 25%, median, 75%, max):")
    print(rates['mean'].describe().round(4))
