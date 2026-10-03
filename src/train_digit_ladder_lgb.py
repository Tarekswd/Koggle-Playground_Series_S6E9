"""
Train Feature Ladder LightGBM with Digit Decomposition and Multi-Scale In-Fold Target Encoding.
Based on the verified 0.94550 single-model ladder notebook.
Uses all 16 CPU threads for fast parallel execution.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

import gc
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("S6E9: Training Digit Ladder LightGBM (16 CPU Threads)")
print("=" * 70)

t0_total = time.time()

train = pd.read_csv("playground-series-s6e9/train.csv")
test = pd.read_csv("playground-series-s6e9/test.csv")
sample_sub = pd.read_csv("playground-series-s6e9/sample_submission.csv")

y = (train["Will_Buy_EV"] == "Yes").astype(int).values
n_train = len(train)
n_test = len(test)

cats = [c for c in test.columns if test[c].dtype == "object"]
nums = [c for c in test.columns if c not in cats + ["id"]]
print(f"Data loaded: Train={train.shape}, Test={test.shape}, Positive rate={y.mean():.4f}")

full = pd.concat([train.drop(columns=["Will_Buy_EV"]), test], ignore_index=True).drop(columns=["id"])

def base_features(df):
    d = df.copy()
    subsidy = (d["Subsidy_Available"] == "Yes").astype(int)
    d["subsidy_x_concern"] = subsidy * d["Environmental_Concern_Level"]
    d["anxiety_ord"] = d["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    d["total_charging"] = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["log_income"] = np.log1p(d["Annual_Income_USD"])
    d["income_is_floor"] = (d["Annual_Income_USD"] == 30000).astype(int)
    d["latent_buy_score"] = 1.2 * (d["Annual_Income_USD"] / 1e5) + 0.6 * d["Environmental_Concern_Level"] + 2.0 * subsidy - d["anxiety_ord"]
    for c in cats:
        d[c] = d[c].astype("category")
    return d

def digit_features(df):
    out = {}
    for c in nums:
        xi = np.round(df[c].values * 1e4).astype(np.int64)
        for j in range(8):
            digit = ((xi // 10 ** j) % 10).astype(np.int8)
            if digit.min() != digit.max():
                out[f"{c}_d{j - 4}"] = digit
    return pd.DataFrame(out)

def freq_features(df):
    return pd.DataFrame({f"{c}_freq": df[c].map(df[c].value_counts(normalize=True)).astype("float32")
                         for c in nums + cats})

print("Extracting feature representations...")
t_f0 = time.time()
F0 = base_features(full)
DG = digit_features(full)
FQ = freq_features(full)
X_all = pd.concat([F0, DG, FQ], axis=1)
print(f"Features ready: {X_all.shape[1]} columns in {time.time() - t_f0:.1f}s")

te_cols = nums + cats

def add_target_encoding(A, B, C, y_tr):
    A, B, C = A.copy(), B.copy(), C.copy()
    for sm in ["auto", 10, 100]:
        enc = TargetEncoder(smooth=sm, cv=5, random_state=SEED, target_type="binary")
        ea = enc.fit_transform(A[te_cols].astype(str), y_tr)
        eb = enc.transform(B[te_cols].astype(str))
        ec = enc.transform(C[te_cols].astype(str))
        for i, c in enumerate(te_cols):
            A[f"{c}_te{sm}"] = ea[:, i].astype("float32")
            B[f"{c}_te{sm}"] = eb[:, i].astype("float32")
            C[f"{c}_te{sm}"] = ec[:, i].astype("float32")
    return A, B, C

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.08,
    num_leaves=63,
    min_child_samples=100,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    cat_smooth=20,
    num_threads=16,
    verbose=-1,
    seed=SEED
)

Xtr = X_all.iloc[:n_train].reset_index(drop=True)
Xte = X_all.iloc[n_train:].reset_index(drop=True)

oof = np.zeros(n_train, dtype=np.float64)
test_pred = np.zeros(n_test, dtype=np.float64)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("\nStarting 5-Fold Stratified Training with In-Fold Target Encoding...")

for fold, (tr, va) in enumerate(cv.split(Xtr, y)):
    t_fold = time.time()
    A, B, C = Xtr.iloc[tr], Xtr.iloc[va], Xte
    A, B, C = add_target_encoding(A, B, C, y[tr])
    
    trn_data = lgb.Dataset(A, label=y[tr])
    val_data = lgb.Dataset(B, label=y[va], reference=trn_data)
    
    model = lgb.train(
        params,
        trn_data,
        num_boost_round=2500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(60, verbose=False)]
    )
    
    val_pred = model.predict(B, num_iteration=model.best_iteration)
    oof[va] = val_pred
    test_pred += model.predict(C, num_iteration=model.best_iteration) / 5.0
    
    fold_auc = roc_auc_score(y[va], val_pred)
    print(f"Fold {fold + 1}/5 | Best Iter: {model.best_iteration:4d} | AUC: {fold_auc:.6f} | Time: {time.time() - t_fold:.1f}s")
    
    del trn_data, val_data, model, A, B, C
    gc.collect()

total_auc = roc_auc_score(y, oof)
print("=" * 70)
print(f"TOTAL OOF AUC: {total_auc:.6f} in {time.time() - t0_total:.1f}s")
print("=" * 70)

# Save predictions
np.save("digit_ladder_lgb_oof.npy", oof)
np.save("digit_ladder_lgb_test.npy", test_pred)

sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": test_pred})
sub.to_csv("submissions/submission_digit_ladder_lgb.csv", index=False)
print("Saved predictions to digit_ladder_lgb_oof.npy, digit_ladder_lgb_test.npy, and submissions/submission_digit_ladder_lgb.csv")
