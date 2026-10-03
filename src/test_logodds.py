import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values

score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx

# Fit 1D logistic regression
lr1d = LogisticRegression()
lr1d.fit(score.reshape(-1, 1), y)
w = lr1d.coef_[0][0]
b = lr1d.intercept_[0]
print(f"Fit on Chris score: log_odds = {w:.4f} * score + {b:.4f}")
print(f"Zero point (where log_odds=0, P=0.5): score = {-b/w:.4f}")

# Notice {-b/w} is approximately 5.5!
probs_1d = expit(w * score + b)
print(f"1D Formula AUC: {roc_auc_score(y, probs_1d):.6f}")
print(f"1D Formula LogLoss: {log_loss(y, probs_1d):.5f}")

# Now let's check what the existing unzipped OOF model predictions look like:
oof_lgb = np.load('oof_predictions.npy')
print(f"OOF LGB AUC: {roc_auc_score(y, oof_lgb):.6f}")
print(f"OOF LGB LogLoss: {log_loss(y, oof_lgb):.5f}")

# What if we blend them in logit space?
eps = 1e-6
clipped_lgb = np.clip(oof_lgb, eps, 1 - eps)
logit_lgb = logit(clipped_lgb)
base_log_odds = w * score + b

for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
    blended_logits = (1 - alpha) * base_log_odds + alpha * logit_lgb
    blended_probs = expit(blended_logits)
    print(f"Alpha {alpha:.2f} (LGB) + {1-alpha:.2f} (Base) -> AUC: {roc_auc_score(y, blended_probs):.6f}")
