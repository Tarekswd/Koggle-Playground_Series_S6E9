"""
LightGBM Digit-Ladder Phase 1 — Extended Generator Formula.
Adds Home_Charging + Commute sigmoid to the base feature set.
Compares OOF AUC to the original digit-ladder LGB to validate Phase 1 lift.
"""
from __future__ import annotations
import gc, time, warnings
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
print("S6E9: LightGBM Phase 1 — Extended Formula Features")
print("=" * 70)
t0_total = time.time()

train = pd.read_csv("playground-series-s6e9/train.csv")
test  = pd.read_csv("playground-series-s6e9/test.csv")
y     = (train["Will_Buy_EV"] == "Yes").astype(int).values
n_train, n_test = len(train), len(test)

cats = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
nums = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]

full = pd.concat([train.drop(columns=["Will_Buy_EV"]), test],
                  ignore_index=True).drop(columns=["id"])


def extended_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended base: original 6 features + Home_Charging signal + commute penalty."""
    d = df.copy()
    subsidy  = (d["Subsidy_Available"] == "Yes").astype(int)
    home_chg = (d["Home_Charging_Possible"] == "Yes").astype(int)
    anx_ord  = d["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2})

    # Original 6 base features
    d["subsidy_x_concern"]  = subsidy * d["Environmental_Concern_Level"]
    d["anxiety_ord"]        = anx_ord
    d["total_charging"]     = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["log_income"]         = np.log1p(d["Annual_Income_USD"])
    d["income_is_floor"]    = (d["Annual_Income_USD"] == 30000).astype(int)

    # Phase 1 NEW features
    inc_norm  = d["Annual_Income_USD"] / 1e5
    com_norm  = d["Daily_Commute_km"] / 83.0
    d["latent_buy_score"]   = (1.2 * inc_norm
                               + 0.6 * d["Environmental_Concern_Level"]
                               + 2.0 * subsidy
                               - anx_ord
                               + 0.5 * home_chg
                               - 0.4 * com_norm)
    d["home_chg_num"]       = home_chg
    d["subsidy_x_home"]     = subsidy * home_chg
    d["concern_x_home"]     = d["Environmental_Concern_Level"] * home_chg
    d["income_x_concern"]   = inc_norm * d["Environmental_Concern_Level"]
    d["income_per_commute"] = d["Annual_Income_USD"] / (d["Daily_Commute_km"] + 1.0)
    d["commute_cliff"]      = (d["Daily_Commute_km"] >= 83).astype(int)
    d["income_cliff"]       = (d["Annual_Income_USD"] >= 170537).astype(int)
    d["income_zero_zone"]   = ((d["Annual_Income_USD"] >= 31004) &
                               (d["Annual_Income_USD"] <= 41970)).astype(int)

    for c in cats:
        d[c] = d[c].astype("category")
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
F0 = extended_base_features(full)
DG = digit_features(full)
FQ = freq_features(full)
X_all = pd.concat([F0, DG, FQ], axis=1)
print(f"Total features: {X_all.shape[1]}")

te_cols = nums + cats


def add_target_encoding(A, B, C, y_tr):
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
    seed=SEED,
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

    model = lgb.train(
        params,
        lgb.Dataset(A, y[tr]),
        num_boost_round=2500,
        valid_sets=[lgb.Dataset(B, y[va])],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    val_pred    = model.predict(B, num_iteration=model.best_iteration)
    oof[va]     = val_pred
    test_pred  += model.predict(C, num_iteration=model.best_iteration) / 5.0

    fold_auc = roc_auc_score(y[va], val_pred)
    print(f"Fold {fold + 1}/5 | Best Iter: {model.best_iteration:4d} | "
          f"AUC: {fold_auc:.6f} | Time: {time.time() - t_fold:.1f}s")
    del model, A, B, C; gc.collect()

total_auc = roc_auc_score(y, oof)
print("=" * 70)
print(f"LGB PHASE-1 OOF AUC: {total_auc:.6f}  ({time.time() - t0_total:.1f}s total)")
print("=" * 70)

# Compare to original digit-ladder
orig_oof = np.load("digit_ladder_lgb_oof.npy")
print(f"Original Digit-Ladder OOF AUC: {roc_auc_score(y, orig_oof):.6f}")
print(f"Phase-1 Lift:                  {total_auc - roc_auc_score(y, orig_oof):+.7f}")

np.save("lgb_phase1_oof.npy",  oof)
np.save("lgb_phase1_test.npy", test_pred)

sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": test_pred})
sub.to_csv("submissions/submission_lgb_phase1_extended.csv", index=False)
print("Saved lgb_phase1_oof.npy, lgb_phase1_test.npy, submission_lgb_phase1_extended.csv")
