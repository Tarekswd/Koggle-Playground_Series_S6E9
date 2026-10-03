import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit

print("Loading datasets and predictions...", flush=True)
train = pd.read_csv('playground-series-s6e9/train.csv')
test  = pd.read_csv('playground-series-s6e9/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

p_lgb    = np.load('oof_predictions.npy')
t_lgb    = np.load('test_predictions.npy')

p_ladder = np.load('ladder_lgb_oof.npy')
t_ladder = np.load('ladder_lgb_test_preds.npy')

p_optuna = np.load('optuna_lgb_oof.npy')
t_optuna = np.load('optuna_lgb_test.npy')

p_xgb79  = np.load('xgb79_oof.npy')
t_xgb79  = np.load('xgb79_test.npy')

p_cb79   = np.load('cb79_oof.npy')
t_cb79   = np.load('cb79_test.npy')

print("\n--- Verified Standalone Model AUCs ---", flush=True)
print(f"  [1] LightGBM Baseline 75:  {roc_auc_score(y, p_lgb):.6f}", flush=True)
print(f"  [2] LightGBM Ladder:       {roc_auc_score(y, p_ladder):.6f}", flush=True)
print(f"  [3] LightGBM Optuna 144:   {roc_auc_score(y, p_optuna):.6f}", flush=True)
print(f"  [4] XGBoost 79 (Hist):     {roc_auc_score(y, p_xgb79):.6f}", flush=True)
print(f"  [5] CatBoost 79:           {roc_auc_score(y, p_cb79):.6f}", flush=True)

def to_odds(p):
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))

def clamp(preds, df):
    i = df['Annual_Income_USD'].values / 100000.0
    e = df['Environmental_Concern_Level'].values.astype(float)
    s = (df['Subsidy_Available'] == 'Yes').astype(float).values
    m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    sc = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h
    out = preds.copy()
    out[df['Annual_Income_USD'].values >= 170537] = 1.0
    out[sc < 0.70260] = 0.0
    out[sc > 7.03543] = 1.0
    return out

Z_tr = np.column_stack([to_odds(p_lgb), to_odds(p_ladder), to_odds(p_optuna), to_odds(p_xgb79), to_odds(p_cb79)])
Z_te = np.column_stack([to_odds(t_lgb), to_odds(t_ladder), to_odds(t_optuna), to_odds(t_xgb79), to_odds(t_cb79)])

feature_names = ['LGB_Base', 'LGB_Ladder', 'LGB_Optuna', 'XGB_79', 'CB_79']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_auc = 0.0
best_C = None
best_oof = None
best_test = None

for C in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    oof_lr = np.zeros(len(train))
    test_lr = np.zeros(len(test))
    for tr_idx, va_idx in skf.split(Z_tr, y):
        lr = LogisticRegression(C=C, max_iter=500, solver='lbfgs')
        lr.fit(Z_tr[tr_idx], y[tr_idx])
        oof_lr[va_idx] = lr.predict_proba(Z_tr[va_idx])[:, 1]
        test_lr += lr.predict_proba(Z_te)[:, 1] / 5.0
    clamped = clamp(oof_lr, train)
    score = roc_auc_score(y, clamped)
    print(f"C={C:6.3f} -> Clamped OOF ROC-AUC: {score:.6f}", flush=True)
    if score > best_auc:
        best_auc = score
        best_C = C
        best_oof = clamped
        best_test = clamp(test_lr, test)

print(f"\n>>> BEST REGULARIZED STACKER (C={best_C}) OOF AUC: {best_auc:.6f} <<<", flush=True)

# Also let's inspect the weights with best C
lr_full = LogisticRegression(C=best_C, max_iter=500, solver='lbfgs')
lr_full.fit(Z_tr, y)
print("Learned Full-Dataset Coefficients on Log-Odds:", flush=True)
for name, c in zip(feature_names, lr_full.coef_[0]):
    print(f"  {name:12s}: {c:+.4f}", flush=True)
print(f"  Intercept   : {lr_full.intercept_[0]:+.4f}", flush=True)

# Save test predictions
sub_path = 'submissions/submission_upgraded_grandmaster_stacking_pinnacle.csv'
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': best_test})
sub.to_csv(sub_path, index=False)
print(f"\nSaved submission to: {sub_path}", flush=True)
assert len(sub) == 286571
assert not sub['Will_Buy_EV'].isnull().any()
print("Verification PASSED!")
