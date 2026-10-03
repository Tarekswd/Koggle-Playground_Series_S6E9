import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

train = pd.read_csv('playground-series-s6e9/train.csv', usecols=['Will_Buy_EV', 'Annual_Income_USD', 'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level'])
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values

oof_lgb = np.load('oof_predictions.npy')
print(f"OOF LGB AUC: {roc_auc_score(y, oof_lgb):.6f}")

inc = train['Annual_Income_USD'].values / 100000.0
env = train['Environmental_Concern_Level'].values
sub = (train['Subsidy_Available'] == 'Yes').astype(float).values
med_anx = (train['Range_Anxiety_Level'] == 'Medium').astype(float).values
high_anx = (train['Range_Anxiety_Level'] == 'High').astype(float).values

buy_score = 1.2 * inc + 0.6 * env + 2.0 * sub - 1.0 * med_anx - 3.0 * high_anx
print(f"Formula AUC: {roc_auc_score(y, buy_score):.6f}")

# Rank blend of LGB + Formula
r_lgb = rankdata(oof_lgb) / len(oof_lgb)
r_formula = rankdata(buy_score) / len(buy_score)

for w in np.linspace(0.0, 1.0, 21):
    blend = w * r_lgb + (1 - w) * r_formula
    auc = roc_auc_score(y, blend)
    print(f"Weight LGB {w:.2f} + Formula {1-w:.2f} -> AUC: {auc:.6f}")
