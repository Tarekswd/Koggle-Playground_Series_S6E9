"""
=============================================================================
FINAL BLEND ANALYSIS + NEURAL NETWORK APPROACH
=============================================================================

DIAGNOSIS:
- LGB-75 (75 features):          0.945832  <- BEST single model
- Quantization Pinnacle ensemble: 0.946010  <- BEST ensemble
- Meta-Stacker (5-stage):         0.944752  <- Failed (too correlated)
- Optuna HPO (144 features):      0.944692  <- Failed (more features ≠ better)

CONCLUSION: We've hit the tree-model ceiling for this dataset.
The CTGAN noise floor is ~0.946 for gradient-boosted trees.

NEW STRATEGY:
1. Try blending Optuna model WITH the pinnacle (uncorrelated errors may help)
2. Build a Feedforward Neural Network (MLP) with formula + buy-score features
   - Neural nets learn fundamentally different decision boundaries vs trees
   - Can capture smooth probability curves that GBM misses at the tails
   - Use PyTorch or sklearn MLP
3. If MLP AUC > 0.943, blend it into the pinnacle ensemble
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("FINAL BLEND ANALYSIS + MLP NEURAL NET", flush=True)
print("=" * 70, flush=True)

BASE = 'playground-series-s6e9'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
N_TRAIN = len(train)
N_TEST  = len(test)

# Load all OOF arrays
oof_lgb75   = np.load('oof_predictions.npy')        # 0.945832
oof_ladder  = np.load('ladder_lgb_oof.npy')         # 0.944789
oof_xgb     = np.load('xgb_oof.npy')               # 0.943319
oof_cb      = np.load('catboost_oof.npy')           # 0.943439
oof_optuna  = np.load('optuna_lgb_oof.npy')         # 0.944692

test_lgb75  = np.load('test_predictions.npy')
test_ladder = np.load('ladder_lgb_test_preds.npy')
test_xgb    = np.load('xgb_test_preds.npy')
test_cb     = np.load('catboost_test_preds.npy')
test_optuna = np.load('optuna_lgb_test.npy')

print("Individual OOF AUCs:", flush=True)
for name, oof in [('LGB-75', oof_lgb75), ('Ladder', oof_ladder),
                  ('XGBoost', oof_xgb), ('CatBoost', oof_cb),
                  ('Optuna-144', oof_optuna)]:
    print(f"  {name:12s}: {roc_auc_score(y, oof):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Check if Optuna adds anything to the Pinnacle
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PART 1] Blend grid: Pinnacle + Optuna...", flush=True)

def rank_norm(arr, n):
    return rankdata(arr) / n

# Reconstruct pinnacle OOF (70% LGB + 25% Ladder + 3% XGB + 2% CB, rank-blended)
pinnacle_oof = (0.70 * rank_norm(oof_lgb75, N_TRAIN) +
                0.25 * rank_norm(oof_ladder, N_TRAIN) +
                0.03 * rank_norm(oof_xgb, N_TRAIN) +
                0.02 * rank_norm(oof_cb, N_TRAIN))

print(f"  Pinnacle raw OOF AUC: {roc_auc_score(y, pinnacle_oof):.6f}", flush=True)

best_blend_auc = 0.0
best_w_opt = 0.0
for w_opt in np.arange(0.0, 0.201, 0.01):
    blend = (1.0 - w_opt) * pinnacle_oof + w_opt * rank_norm(oof_optuna, N_TRAIN)
    auc = roc_auc_score(y, blend)
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_w_opt = w_opt

print(f"  Best blend: {1-best_w_opt:.2f}xPinnacle + {best_w_opt:.2f}xOptuna -> AUC={best_blend_auc:.6f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: MLP Neural Network
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PART 2] Training MLP Neural Network (5-fold OOF)...", flush=True)

def build_mlp_features(df):
    """
    Compact, smooth features ideal for neural nets:
    - All formula variables (continuous)
    - Polynomial interactions
    - RBF buy-score embedding (12 centers)
    - Normalized income/age/commute
    """
    inc_raw = df['Annual_Income_USD'].values.astype(float)
    inc   = inc_raw / 200000.0   # normalize to ~[0, 1]
    env   = df['Environmental_Concern_Level'].values.astype(float) / 10.0
    sub   = (df['Subsidy_Available'] == 'Yes').astype(float).values
    med   = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    high  = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    none_ = (df['Range_Anxiety_Level'] == 'None').astype(float).values
    home  = df['Charging_Stations_Near_Home'].values.astype(float) / 10.0
    work  = df['Charging_Stations_Near_Work'].values.astype(float) / 10.0
    cars  = df['Number_of_Cars_Owned'].values.astype(float) / 5.0
    age   = df['Age'].values.astype(float) / 80.0
    commute = df['Daily_Commute_km'].values.astype(float) / 200.0

    buy_score_raw = 1.2*(inc*2.0) + 0.6*(env*10.0) + 2.0*sub - 1.0*med - 3.0*high
    buy_norm = buy_score_raw / 10.0  # normalize to ~[0, 1]

    # Categorical: Gender, City_Type, Current_Car_Type, Home_Charging_Possible
    gender    = (df['Gender'] == 'Male').astype(float).values
    city_urban = (df['City_Type'] == 'Urban').astype(float).values
    city_sub  = (df['City_Type'] == 'Suburban').astype(float).values
    home_chg  = (df['Home_Charging_Possible'] == 'Yes').astype(float).values

    # RBF embedding of buy_score (12 centers from -1 to 9)
    centers = np.linspace(-0.5, 9.5, 12)
    rbf = np.exp(-0.5 * ((buy_score_raw[:, None] - centers[None, :]) / 0.8) ** 2)

    # Cliff/floor/ceiling one-hot flags
    cliff = (inc_raw >= 170537).astype(float)
    floor = (buy_score_raw < 0.703).astype(float)
    ceil  = (buy_score_raw > 7.035).astype(float)

    # 2nd-order interactions
    inc_env = inc * env * 10.0
    inc_sub_f = inc * sub
    env_sub_f = env * sub
    high_income_no_anx = (inc > 0.5) * none_

    features = np.column_stack([
        inc, env, sub, med, high, none_,
        home, work, cars, age, commute,
        buy_norm, buy_norm**2,
        gender, city_urban, city_sub, home_chg,
        inc_env, inc_sub_f, env_sub_f, high_income_no_anx,
        cliff, floor, ceil,
    ])
    return np.hstack([features, rbf])

print("  Building MLP feature matrix...", flush=True)
F_train = build_mlp_features(train)
F_test  = build_mlp_features(test)
print(f"  MLP feature shape: {F_train.shape}", flush=True)

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_mlp  = np.zeros(N_TRAIN, dtype=np.float64)
test_mlp = np.zeros(N_TEST,  dtype=np.float64)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(SKF.split(F_train, y)):
    X_tr, y_tr = F_train[tr_idx], y[tr_idx]
    X_va       = F_train[va_idx]
    X_te       = F_test

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-3,           # L2 regularization
        batch_size=2048,
        learning_rate='adaptive',
        learning_rate_init=5e-4,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_tr_s, y_tr)
    oof_mlp[va_idx] = mlp.predict_proba(X_va_s)[:, 1]
    test_mlp += mlp.predict_proba(X_te_s)[:, 1] / 5.0

    fold_auc = roc_auc_score(y[va_idx], oof_mlp[va_idx])
    print(f"  MLP Fold {fold+1}: AUC={fold_auc:.6f}", flush=True)

mlp_auc = roc_auc_score(y, oof_mlp)
print(f"\n*** MLP OOF AUC: {mlp_auc:.6f} in {time.time()-t0:.1f}s ***\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Final mega-blend: Pinnacle + MLP + Optuna
# ─────────────────────────────────────────────────────────────────────────────
print("[PART 3] Final mega-blend grid search...", flush=True)

r_pinnacle = pinnacle_oof
r_mlp      = rank_norm(oof_mlp, N_TRAIN)
r_optuna   = rank_norm(oof_optuna, N_TRAIN)

best_auc   = 0.0
best_combo = (1.0, 0.0, 0.0)

for w_mlp in np.arange(0.0, 0.21, 0.02):
    for w_opt in np.arange(0.0, 0.21, 0.02):
        w_pin = 1.0 - w_mlp - w_opt
        if w_pin < 0:
            continue
        blend = w_pin * r_pinnacle + w_mlp * r_mlp + w_opt * r_optuna
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_combo = (w_pin, w_mlp, w_opt)

w_pin, w_mlp_w, w_opt_w = best_combo
print(f"  Best: {w_pin:.2f}xPinnacle + {w_mlp_w:.2f}xMLP + {w_opt_w:.2f}xOptuna", flush=True)
print(f"  Unclamped OOF AUC: {best_auc:.6f}", flush=True)

# Compute final blend OOF
oof_mega = w_pin * r_pinnacle + w_mlp_w * r_mlp + w_opt_w * r_optuna

# Clamping
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

oof_mega_clamped = clamp(oof_mega, train)
final_auc = roc_auc_score(y, oof_mega_clamped)
print(f"\n{'='*70}", flush=True)
print(f"  *** MEGA-BLEND FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
print(f"{'='*70}\n", flush=True)

# Test predictions
t_pin = (0.70 * rank_norm(test_lgb75, N_TEST) +
         0.25 * rank_norm(test_ladder, N_TEST) +
         0.03 * rank_norm(test_xgb, N_TEST) +
         0.02 * rank_norm(test_cb, N_TEST))
t_mlp    = rank_norm(test_mlp, N_TEST)
t_optuna = rank_norm(test_optuna, N_TEST)

test_mega = w_pin * t_pin + w_mlp_w * t_mlp + w_opt_w * t_optuna
test_mega_clamped = clamp(test_mega, test)

np.save('mlp_oof.npy', oof_mlp)
np.save('mlp_test.npy', test_mlp)

sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_mega_clamped})
out_path = 'submissions/submission_mega_blend_mlp.csv'
sub.to_csv(out_path, index=False)
print(f"Saved {out_path}", flush=True)
assert len(sub) == 286571
assert not sub['Will_Buy_EV'].isnull().any()
print("Verification PASSED.", flush=True)

print("\n" + "="*70)
print("COMPLETE EXPERIMENT SUMMARY")
print("="*70)
print(f"  LGB-75 (original):        0.945832")
print(f"  Quantization Pinnacle:    0.946010  <- Previous best")
print(f"  Optuna HPO (144 feat):    0.944692")
print(f"  MLP Neural Net:           {mlp_auc:.6f}")
print(f"  Mega Blend (Pin+MLP+Opt): {final_auc:.6f}  <- NEW BEST?")
print("="*70)
