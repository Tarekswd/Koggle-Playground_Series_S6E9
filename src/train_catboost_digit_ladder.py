"""
CatBoost Digit-Ladder Model — Phase 2 of the 0.95 roadmap.
Features: Extended formula + 16-channel digit decomposition + freq encoding + multi-scale in-fold TE.
Uses all available CPU cores. OOF AUC target: 0.947+
"""
from __future__ import annotations
import gc, time, os, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("S6E9: CatBoost Digit-Ladder (Phase 2 — 0.95 Roadmap)")
print("=" * 70)
t0_total = time.time()

train = pd.read_csv("playground-series-s6e9/train.csv")
test  = pd.read_csv("playground-series-s6e9/test.csv")
y     = (train["Will_Buy_EV"] == "Yes").astype(int).values
n_train, n_test = len(train), len(test)
print(f"Train: {train.shape}, Test: {test.shape}, Buy rate: {y.mean():.4f}")

cats = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
nums = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]

full = pd.concat([train.drop(columns=["Will_Buy_EV"]), test],
                  ignore_index=True).drop(columns=["id"])


def extended_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: extended generator formula + secondary feature interactions."""
    d = df.copy()
    subsidy = (d["Subsidy_Available"] == "Yes").astype(int)
    home_chg = (d["Home_Charging_Possible"] == "Yes").astype(int)
    anx_ord  = d["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)

    # Core formula (known generator signal)
    d["anxiety_ord"]      = anx_ord
    d["subsidy_num"]      = subsidy
    d["home_chg_num"]     = home_chg

    # Extended formula: all 4 formula variables + home charging + commute sigmoid
    inc_norm = d["Annual_Income_USD"] / 1e5
    com_norm = d["Daily_Commute_km"] / 83.0   # 83 km = zero-buyer cliff
    d["latent_buy_score"] = (1.2 * inc_norm
                             + 0.6 * d["Environmental_Concern_Level"]
                             + 2.0 * subsidy
                             - anx_ord
                             + 0.5 * home_chg       # Phase 1: home charging bonus
                             - 0.4 * com_norm)       # Phase 1: commute penalty (linear approx)

    # Interaction features
    d["subsidy_x_concern"]   = subsidy * d["Environmental_Concern_Level"]
    d["subsidy_x_home"]      = subsidy * home_chg
    d["concern_x_home"]      = d["Environmental_Concern_Level"] * home_chg
    d["income_x_concern"]    = inc_norm * d["Environmental_Concern_Level"]
    d["total_charging"]      = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["log_income"]          = np.log1p(d["Annual_Income_USD"])
    d["income_is_floor"]     = (d["Annual_Income_USD"] == 30000).astype(int)
    d["income_per_commute"]  = d["Annual_Income_USD"] / (d["Daily_Commute_km"] + 1.0)
    d["commute_cliff"]       = (d["Daily_Commute_km"] >= 83).astype(int)
    d["income_cliff"]        = (d["Annual_Income_USD"] >= 170537).astype(int)
    d["income_zero_zone"]    = ((d["Annual_Income_USD"] >= 31004) &
                                (d["Annual_Income_USD"] <= 41970)).astype(int)

    # Cat columns: keep as-is for CatBoost native handling
    for c in cats:
        d[c] = d[c].astype(str)
    return d


def digit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in nums:
        xi = np.round(df[c].values * 1e4).astype(np.int64)
        for j in range(8):
            digit = ((xi // 10 ** j) % 10).astype(np.int8)
            if digit.min() != digit.max():
                out[f"{c}_d{j - 4}"] = digit
    return pd.DataFrame(out, index=df.index)


def freq_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{c}_freq": df[c].map(df[c].value_counts(normalize=True)).astype("float32")
         for c in nums + cats},
        index=df.index
    )


print("Extracting features...")
t_f = time.time()
F0 = extended_base_features(full)
DG = digit_features(full)
FQ = freq_features(full)
X_all = pd.concat([F0, DG, FQ], axis=1)
print(f"Total features: {X_all.shape[1]} in {time.time() - t_f:.1f}s")

# Identify cat column indices for CatBoost
cat_feature_names = [c for c in cats if c in X_all.columns]
cat_feature_indices = [X_all.columns.get_loc(c) for c in cat_feature_names]

te_cols = nums + cats  # columns for target encoding


def add_target_encoding(A: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame,
                         y_tr: np.ndarray) -> tuple:
    A, B, C = A.copy(), B.copy(), C.copy()
    for sm in ["auto", 10, 100]:
        enc = TargetEncoder(smooth=sm, cv=5, random_state=SEED, target_type="binary")
        ea  = enc.fit_transform(A[te_cols].astype(str), y_tr)
        eb  = enc.transform(B[te_cols].astype(str))
        ec  = enc.transform(C[te_cols].astype(str))
        for i, c in enumerate(te_cols):
            A[f"{c}_te{sm}"] = ea[:, i].astype("float32")
            B[f"{c}_te{sm}"] = eb[:, i].astype("float32")
            C[f"{c}_te{sm}"] = ec[:, i].astype("float32")
    return A, B, C


params = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5.0,
    min_data_in_leaf=30,
    eval_metric="AUC",
    random_seed=SEED,
    thread_count=-1,   # all CPUs
    verbose=False,
    use_best_model=True,
    early_stopping_rounds=80,
)

Xtr = X_all.iloc[:n_train].reset_index(drop=True)
Xte = X_all.iloc[n_train:].reset_index(drop=True)

oof       = np.zeros(n_train, dtype=np.float64)
test_pred = np.zeros(n_test,  dtype=np.float64)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("\n5-Fold Stratified Training with In-Fold Target Encoding...")

for fold, (tr, va) in enumerate(cv.split(Xtr, y)):
    t_fold = time.time()
    A, B, C = Xtr.iloc[tr], Xtr.iloc[va], Xte
    A, B, C = add_target_encoding(A, B, C, y[tr])

    # After TE, cat columns have been string-encoded; recompute cat indices
    fold_cat_names = [c for c in cat_feature_names if c in A.columns]
    fold_cat_idx   = [A.columns.get_loc(c) for c in fold_cat_names]

    train_pool = Pool(A, label=y[tr],  cat_features=fold_cat_idx)
    val_pool   = Pool(B, label=y[va],  cat_features=fold_cat_idx)
    test_pool  = Pool(C, cat_features=fold_cat_idx)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)

    val_pred   = model.predict_proba(val_pool)[:, 1]
    oof[va]    = val_pred
    test_pred += model.predict_proba(test_pool)[:, 1] / 5.0

    fold_auc = roc_auc_score(y[va], val_pred)
    print(f"Fold {fold + 1}/5 | Best Iter: {model.best_iteration_:4d} | "
          f"AUC: {fold_auc:.6f} | Time: {time.time() - t_fold:.1f}s")

    del train_pool, val_pool, test_pool, model, A, B, C
    gc.collect()

total_auc = roc_auc_score(y, oof)
print("=" * 70)
print(f"CATBOOST OOF AUC: {total_auc:.6f}  ({time.time() - t0_total:.1f}s total)")
print("=" * 70)

np.save("catboost_digit_ladder_oof.npy",  oof)
np.save("catboost_digit_ladder_test.npy", test_pred)

sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": test_pred})
sub.to_csv("submissions/submission_catboost_digit_ladder.csv", index=False)
print("Saved: catboost_digit_ladder_oof.npy, catboost_digit_ladder_test.npy, "
      "submissions/submission_catboost_digit_ladder.csv")
