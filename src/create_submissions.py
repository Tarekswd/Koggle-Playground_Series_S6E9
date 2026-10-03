import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

os.makedirs('submissions', exist_ok=True)
os.makedirs('research', exist_ok=True)

test = pd.read_csv('playground-series-s6e9/test.csv')
sample_sub = pd.read_csv('playground-series-s6e9/sample_submission.csv')

tpreds = np.load('test_predictions.npy')

# 1. Baseline LGB submission
sub_base = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': tpreds
})
sub_base.to_csv('submissions/submission_baseline_lgb.csv', index=False)
print("Saved submission_baseline_lgb.csv")

# 2. Cliff & Deterministic Post-Processed submission
inc = test['Annual_Income_USD'].values / 100000.0
env = test['Environmental_Concern_Level'].values
sub_avail = (test['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (test['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (test['Range_Anxiety_Level'] == 'High').astype(float).values
buy_score = 1.2 * inc + 0.6 * env + 2.0 * sub_avail - 1.0 * med_anx - 3.0 * high_anx

tpreds_post = tpreds.copy()

# Millionaire Cliff
cliff_mask = test['Annual_Income_USD'] >= 170537
print(f"Test cliff rows (income >= 170537): {cliff_mask.sum()}")
tpreds_post[cliff_mask] = 1.0

# Zero score threshold
zero_mask = buy_score < 0.70260
print(f"Test zero score rows (score < 0.70260): {zero_mask.sum()}")
tpreds_post[zero_mask] = 0.0

# High score threshold
high_mask = buy_score > 7.03543
print(f"Test high score rows (score > 7.03543): {high_mask.sum()}")
tpreds_post[high_mask] = 1.0

sub_post = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': tpreds_post
})
sub_post.to_csv('submissions/submission_cliff_postprocessed.csv', index=False)
print("Saved submission_cliff_postprocessed.csv")

# 3. Formula + ML Hybrid (Rank Blend)
r_lgb = rankdata(tpreds) / len(tpreds)
r_formula = rankdata(buy_score) / len(buy_score)

# 95% ML + 5% Formula (optimal from validation)
r_blend = 0.95 * r_lgb + 0.05 * r_formula
# Apply cliff on top of blend
r_blend[cliff_mask] = 1.0
r_blend[zero_mask] = 0.0
r_blend[high_mask] = 1.0

sub_hybrid = pd.DataFrame({
    'id': test['id'],
    'Will_Buy_EV': r_blend
})
sub_hybrid.to_csv('submissions/submission_formula_ml_blend.csv', index=False)
print("Saved submission_formula_ml_blend.csv")

# Section 34 Sanity Checks
print("\n--- RUNNING SANITY CHECKS ---")
for fname in ['submission_baseline_lgb.csv', 'submission_cliff_postprocessed.csv', 'submission_formula_ml_blend.csv']:
    path = os.path.join('submissions', fname)
    df = pd.read_csv(path)
    assert len(df) == len(test), f"Row count mismatch in {fname}"
    assert (df['id'] == test['id']).all(), f"ID mismatch in {fname}"
    assert list(df.columns) == list(sample_sub.columns), f"Columns mismatch in {fname}"
    assert not df['Will_Buy_EV'].isnull().any(), f"NaNs found in {fname}"
    assert np.isfinite(df['Will_Buy_EV']).all(), f"Inf found in {fname}"
    assert (df['Will_Buy_EV'] >= 0.0).all() and (df['Will_Buy_EV'] <= 1.0).all(), f"Range out of [0, 1] in {fname}"
    print(f"VALIDATION PASSED: {fname} (rows: {len(df)}, mean: {df['Will_Buy_EV'].mean():.4f}, min: {df['Will_Buy_EV'].min():.4f}, max: {df['Will_Buy_EV'].max():.4f})")
