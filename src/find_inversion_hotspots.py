import pandas as pd
import numpy as np

print("=== ANALYZING INVERSION HOTSPOTS & RESIDUALS ===")

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

oof = np.load('ladder_lgb_oof.npy')

# Compute Chris Deotte Buy Score
inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values
score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

train['oof_pred'] = oof
train['buy_score'] = score
train['y'] = y
train['error'] = np.abs(y - oof)

print("\nWhere are the largest errors (residuals) located?")
print("1. Error by Buy Score Ranges:")
train['score_range'] = pd.cut(train['buy_score'], bins=[-np.inf, 2.0, 3.5, 4.5, 5.0, 5.5, 6.0, 7.0, np.inf])
grp_score = train.groupby('score_range', observed=False).agg(
    count=('y', 'count'),
    pos_rate=('y', 'mean'),
    mean_oof=('oof_pred', 'mean'),
    mean_abs_error=('error', 'mean')
)
print(grp_score.round(4))

print("\n2. Hard Inversions (High confidence mistakes):")
fp_hard = train[(train['y'] == 0) & (train['oof_pred'] > 0.80)]
fn_hard = train[(train['y'] == 1) & (train['oof_pred'] < 0.05)]
print(f"False Positives with pred > 0.80: {len(fp_hard)}")
print(f"False Negatives with pred < 0.05: {len(fn_hard)}")

if len(fp_hard) > 0:
    print("\nSample False Positives (predicted > 0.80 but actual 0):")
    print(fp_hard[['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level', 'buy_score', 'oof_pred']].head(5))

if len(fn_hard) > 0:
    print("\nSample False Negatives (predicted < 0.05 but actual 1):")
    print(fn_hard[['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level', 'buy_score', 'oof_pred']].head(5))
