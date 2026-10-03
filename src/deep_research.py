import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from scipy.special import expit

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values

score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

print("--- 1. ANALYZING THE TARGET FORMULA & NOISE DISTRIBUTION ---")
# If target = (score + wobble >= threshold)
# Let's inspect the transition curve: P(y=1 | score)
df_score = pd.DataFrame({'score': score, 'y': y})
df_score['bin'] = pd.cut(df_score['score'], bins=100)
agg = df_score.groupby('bin', observed=False).agg(
    mean_score=('score', 'mean'),
    count=('y', 'count'),
    pos_rate=('y', 'mean')
).dropna()

print(agg[(agg['pos_rate'] > 0.0) & (agg['pos_rate'] < 1.0)].head(15))
print(agg[(agg['pos_rate'] > 0.0) & (agg['pos_rate'] < 1.0)].tail(15))

# Let's test Probit vs Logistic vs Linear Fit on score
from scipy.optimize import minimize

# Fit probit: P = norm.cdf((score - mu) / sigma)
def probit_loss(params):
    mu, sigma = params
    z = (score - mu) / max(sigma, 1e-4)
    p = np.clip(norm.cdf(z), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

res_probit = minimize(probit_loss, [5.5, 1.0], method='Nelder-Mead')
mu_p, sig_p = res_probit.x
p_probit = norm.cdf((score - mu_p) / sig_p)
print(f"\nProbit fit: mu = {mu_p:.4f}, sigma = {sig_p:.4f}")
print(f"Probit LogLoss: {res_probit.fun:.5f}, AUC: {roc_auc_score(y, p_probit):.6f}")

# Fit Logit: P = expit((score - mu) / s)
def logit_loss(params):
    mu, s = params
    z = (score - mu) / max(s, 1e-4)
    p = np.clip(expit(z), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

res_logit = minimize(logit_loss, [5.5, 0.5], method='Nelder-Mead')
mu_l, s_l = res_logit.x
p_logit = expit((score - mu_l) / s_l)
print(f"Logit fit: mu = {mu_l:.4f}, s = {s_l:.4f}")
print(f"Logit LogLoss: {res_logit.fun:.5f}, AUC: {roc_auc_score(y, p_logit):.6f}")

# Check ID correlation or structure
print("\n--- 2. ID STRUCTURE ANALYSIS ---")
print("Target rate across 10 ID deciles:")
train['id_decile'] = pd.qcut(train['id'], q=10)
print(train.groupby('id_decile', observed=False)['Will_Buy_EV'].apply(lambda s: (s == 'Yes').mean()))
