from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

import gc
import json
import platform
import time
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(16)
torch.set_num_interop_threads(4)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

DATA_DIR = Path('playground-series-s6e9')
EXP_NAME = MODEL_NAME = 'GENERATOR_AWARE_RIDGE'
OOF_DIR, TEST_PRED_DIR = Path('oof'), Path('test_preds')
SUBMISSION_DIR = Path('submissions')
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)
TEST_PRED_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS, INNER_FOLDS, RANDOM_STATE = 10, 5, 42
L2 = 10.0
LBFGS_ITERS = 200
OUT_TEMPERATURE, OUT_OFFSET = 1.25, -1.0
TARGET, ID_COL = 'Will_Buy_EV', 'id'
FLOAT_COLS = ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']
INT_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']
CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
COLS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
        'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
        'Environmental_Concern_Level'] + CAT_COLS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Starting {EXP_NAME} | device={DEVICE} | threads={torch.get_num_threads()}')

# 1. Load Data
t0 = time.time()
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
ids = sample[ID_COL].to_numpy()
test = test.set_index(ID_COL).reindex(ids).reset_index()

y = (train[TARGET] == 'Yes').astype(np.int8).to_numpy()
ntr, nte = len(train), len(test)
X = pd.concat([train[COLS], test[COLS]], ignore_index=True)
N = len(X)
SPLITS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(ntr), y))
print(f'Data loaded: train={ntr}, test={nte} in {time.time()-t0:.1f}s')

# 2. Tokenizer & Keys
t0 = time.time()
TOK = AutoTokenizer.from_pretrained('gpt2')
inc_str = X.Annual_Income_USD.astype(np.int64).astype(str).to_numpy()
u_inc, inv_inc = np.unique(inc_str, return_inverse=True)
tok_ids = [TOK(' ' + s).input_ids for s in u_inc]

def fz(values):
    return pd.factorize(values, sort=False)[0].astype(np.int64)

def qbin(values, q):
    ranks = pd.Series(values).rank(method='first')
    return fz(pd.qcut(ranks, q, labels=False, duplicates='drop').to_numpy())

K = {}
K['L1'] = fz(np.array([t[0] for t in tok_ids], dtype=np.int64)[inv_inc])
K['L2'] = fz(np.array([t[0] * 60000 + (t[1] if len(t) > 1 else -1) for t in tok_ids], dtype=np.int64)[inv_inc])
K['IV'] = fz(inc_str)
K['LAST'] = fz(np.array([len(t) * 60000 + t[-1] for t in tok_ids], dtype=np.int64)[inv_inc])
cm = X.Daily_Commute_km.to_numpy(np.float64)
K['CV'] = fz(cm)
K['CI'] = fz(np.floor(cm))
for c in COLS:
    if c not in ('Annual_Income_USD', 'Daily_Commute_km'):
        K[c] = fz(X[c].to_numpy())
K['AGEDEC'] = fz((X.Age // 10).to_numpy())
K['CMTBIN'] = qbin(cm, 10)
K['SHB'] = fz(np.minimum(X.Charging_Stations_Near_Home, 6).to_numpy())
K['SWB'] = fz(np.minimum(X.Charging_Stations_Near_Work, 6).to_numpy())
K['INCQ'] = qbin(X.Annual_Income_USD.to_numpy(), 5)
K['INC20'] = qbin(X.Annual_Income_USD.to_numpy(), 20)
K['INC_MOD100'] = fz((X.Annual_Income_USD.astype(np.int64) % 100).to_numpy())
K['INC_DIV1000'] = fz((X.Annual_Income_USD.astype(np.int64) // 1000).to_numpy())
K['CELL'] = fz(((K['Subsidy_Available'] * 5 + K['Environmental_Concern_Level']) * 3 + K['Range_Anxiety_Level']) * 2 + K['Home_Charging_Possible'])

def cross(a, b):
    return fz(K[a] * (int(K[b].max()) + 1) + K[b])

FLAT = {}
for lv in ['L1', 'L2', 'LAST', 'IV', 'CV', 'CI', 'INC_MOD100', 'INC_DIV1000', 'Age']:
    for a in (5, 50):
        FLAT[f'r_{lv}_a{a}'] = (K[lv], a)
for lv in ['L1', 'L2', 'LAST']:
    for c in ['City_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Environmental_Concern_Level',
              'Range_Anxiety_Level', 'Gender', 'Current_Car_Type', 'Number_of_Cars_Owned',
              'AGEDEC', 'CMTBIN', 'SHB', 'SWB']:
        FLAT[f'tokx_{lv}_{c}'] = (cross(lv, c), 20)
for c in ['City_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level', 'INCQ']:
    FLAT[f'cmtx_CI_{c}'] = (cross('CI', c), 20)
for c in ['Home_Charging_Possible', 'City_Type', 'Range_Anxiety_Level']:
    FLAT[f'cmtx_CV_{c}'] = (cross('CV', c), 20)
for name, key, alpha in [
    ('cell', K['CELL'], 5), ('cell_x_L1', cross('CELL', 'L1'), 20),
    ('cell_x_CI', cross('CELL', 'CI'), 20), ('cell_x_INCQ', cross('CELL', 'INCQ'), 20)]:
    FLAT[name] = (key, alpha)

CATS = ['Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level', 'Home_Charging_Possible',
        'City_Type', 'Current_Car_Type', 'Number_of_Cars_Owned', 'Gender', 'AGEDEC', 'SHB', 'SWB']
for c in CATS:
    for suffix, other in [('INC20', 'INC20'), ('CMT10', 'CMTBIN')]:
        FLAT[f'xc_{c}_{suffix}'] = (cross(c, other), 20)
for i, c1 in enumerate(CATS):
    for c2 in CATS[i + 1:]:
        FLAT[f'xp_{c1}_{c2}'] = (cross(c1, c2), 20)
K['CELLI'] = cross('CELL', 'INC20')
for name, key in [('x3_CELL_INC20', K['CELLI']), ('x3_CELL_CMT10', cross('CELL', 'CMTBIN'))]:
    FLAT[name] = (key, 20)
for c in ['City_Type', 'Current_Car_Type', 'Number_of_Cars_Owned', 'AGEDEC']:
    FLAT[f'x4_CELLI_{c}'] = (cross('CELLI', c), 20)

CHAINS = {f'chain_S{S}': ([K['L1'], K['L2'], K['IV']], S) for S in (5, 20, 80)}
CHAINS.update({f'chainC_S{S}': ([K['CI'], K['CV']], S) for S in (5, 20)})
LABEL_NAMES = list(FLAT) + [f'{n}_lv{i}' for n, (ks, _) in CHAINS.items() for i in range(len(ks))]

# 3. Label-free features
LF = []
for nm in ['IV', 'L2', 'L1', 'CV', 'LAST']:
    LF.append(np.log1p(np.bincount(K[nm])[K[nm]]))
inc_f = X.Annual_Income_USD.to_numpy(np.float64)
env = X.Environmental_Concern_Level.to_numpy(np.float64)
subs = (X.Subsidy_Available == 'Yes').to_numpy()
anx_formula = X.Range_Anxiety_Level.map({'Low': 0, 'Medium': 1, 'High': 3}).to_numpy(np.float64)
LF.append(1.2 * inc_f / 1e5 + 0.6 * env + 2 * subs - anx_formula)
for c in COLS[:7]:
    v = X[c].to_numpy(np.float64)
    LF.append(v)
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        LF.append(np.maximum(v - np.quantile(v, q), 0.0))
OH = pd.get_dummies(X[CAT_COLS + ['Environmental_Concern_Level', 'Number_of_Cars_Owned']].astype(str), dtype=np.float32)
LF = np.column_stack(LF + [OH.to_numpy()]).astype(np.float32)

MIXV = {
    'env': env, 'sub': subs.astype(float),
    'anx': X.Range_Anxiety_Level.map({'Low': 0, 'Medium': 1, 'High': 2}).to_numpy(np.float64),
    'home': (X.Home_Charging_Possible == 'Yes').to_numpy(float),
    'urban': (X.City_Type == 'Urban').to_numpy(float), 'rural': (X.City_Type == 'Rural').to_numpy(float),
    'suv': (X.Current_Car_Type == 'SUV').to_numpy(float), 'truck': (X.Current_Car_Type == 'Truck').to_numpy(float),
    'sedan': (X.Current_Car_Type == 'Sedan').to_numpy(float), 'male': (X.Gender == 'Male').to_numpy(float),
    'cars': X.Number_of_Cars_Owned.to_numpy(float), 'age': X.Age.to_numpy(float),
    'sth': X.Charging_Stations_Near_Home.to_numpy(float), 'stw': X.Charging_Stations_Near_Work.to_numpy(float),
    'cmt': cm, 'inc': inc_f / 1e4,
}
ivcv = fz(np.char.add(np.char.add(inc_str.astype(str), '_'), X.Daily_Commute_km.astype(str).to_numpy()))
mix_keys = {'IV': K['IV'], 'L2': K['L2'], 'L1': K['L1'], 'CV': K['CV'], 'IVCV': ivcv}
MIX = []
for kn, key in mix_keys.items():
    counts = np.bincount(key).astype(np.float64)
    n = counts[key]
    MIX.append(np.log(np.maximum(n, 1.0)))
    for vn, v in MIXV.items():
        if (kn in ('IV', 'L2', 'L1') and vn == 'inc') or (kn == 'CV' and vn == 'cmt') or (kn == 'IVCV' and vn in ('inc', 'cmt')):
            continue
        global_mean = float(v.mean())
        sums = np.bincount(key, weights=v, minlength=len(counts))[key]
        MIX.append((sums - v + 5 * global_mean) / (n - 1 + 5) - global_mean)
MIX = np.column_stack(MIX).astype(np.float32)
LF = np.concatenate([LF, MIX], axis=1)
del MIX, MIXV, mix_keys
gc.collect()

P = len(LABEL_NAMES) + LF.shape[1]
print(f'Feature engineering complete: Total P={P} in {time.time()-t0:.1f}s')

# 4. Encoding functions
def _stats(key, fit):
    size = int(key.max()) + 1
    return (np.bincount(key[fit], weights=y[fit], minlength=size),
            np.bincount(key[fit], minlength=size))

def _logit_rate(rate):
    rate = np.clip(rate, 1e-6, 1 - 1e-6)
    return np.log(rate / (1 - rate)).astype(np.float32)

def encode(fit, app, out, rows):
    prior = float(y[fit].mean())
    j = 0
    for _, (key, alpha) in FLAT.items():
        s, c = _stats(key, fit)
        ka = key[app]
        out[rows, j] = _logit_rate((s[ka] + alpha * prior) / (c[ka] + alpha))
        j += 1
    for _, (keys, strength) in CHAINS.items():
        post = np.full(len(app), prior, np.float64)
        for key in keys:
            s, c = _stats(key, fit)
            ka = key[app]
            post = (s[ka] + strength * post) / (c[ka] + strength)
            out[rows, j] = _logit_rate(post)
            j += 1

def fit_glm(Z, yy, l2=L2, iters=LBFGS_ITERS):
    w = torch.zeros(Z.shape[1], device=Z.device, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, device=Z.device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        logits = Z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yy, reduction='sum')
        loss = loss + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()

def to_probability(eta):
    z = np.clip(OUT_TEMPERATURE * eta + OUT_OFFSET, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

# 5. Training loop
oof = np.zeros(ntr, dtype=np.float64)
test_fold_preds = []
n_label = len(LABEL_NAMES)

print(f'\nStarting {N_FOLDS}-Fold Training...')
run_start = time.time()
for k, (fit_rows, valid_rows) in enumerate(SPLITS):
    f_t0 = time.time()
    test_rows = np.arange(ntr, N)
    Mfit = np.empty((len(fit_rows), P), np.float32)
    Mvalid = np.empty((len(valid_rows), P), np.float32)
    Mtest = np.empty((nte, P), np.float32)

    encode(fit_rows, valid_rows, Mvalid, slice(None))
    encode(fit_rows, test_rows, Mtest, slice(None))

    position = np.full(ntr, -1, np.int64)
    position[fit_rows] = np.arange(len(fit_rows))
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for inner_fit_rel, inner_app_rel in inner.split(np.zeros(len(fit_rows)), y[fit_rows]):
        inner_fit = fit_rows[inner_fit_rel]
        inner_app = fit_rows[inner_app_rel]
        encode(inner_fit, inner_app, Mfit, position[inner_app])

    Mfit[:, n_label:] = LF[fit_rows]
    Mvalid[:, n_label:] = LF[valid_rows]
    Mtest[:, n_label:] = LF[ntr:]

    mu = Mfit.mean(axis=0, dtype=np.float64)
    sd = Mfit.std(axis=0, dtype=np.float64) + 1e-6

    # Standardize in-place to preserve RAM
    Mfit -= mu.astype(np.float32)
    Mfit /= sd.astype(np.float32)
    Zfit = torch.from_numpy(Mfit.astype(np.float64)).to(DEVICE)
    del Mfit
    yy = torch.from_numpy(y[fit_rows].astype(np.float64)).to(DEVICE)

    w, b = fit_glm(Zfit, yy)
    del Zfit, yy
    gc.collect()

    Mvalid -= mu.astype(np.float32)
    Mvalid /= sd.astype(np.float32)
    Zvalid = torch.from_numpy(Mvalid.astype(np.float64)).to(DEVICE)
    del Mvalid

    Mtest -= mu.astype(np.float32)
    Mtest /= sd.astype(np.float32)
    Ztest = torch.from_numpy(Mtest.astype(np.float64)).to(DEVICE)
    del Mtest

    eta_valid = (Zvalid @ w + b).cpu().numpy()
    eta_test = (Ztest @ w + b).cpu().numpy()
    del Zvalid, Ztest, w, b
    gc.collect()

    val_pred = to_probability(eta_valid)
    t_pred = to_probability(eta_test)
    oof[valid_rows] = val_pred
    test_fold_preds.append(t_pred)

    fold_auc = roc_auc_score(y[valid_rows], val_pred)
    print(f'FOLD {k+1:02d}/{N_FOLDS} | AUC = {fold_auc:.6f} | time = {time.time()-f_t0:.1f}s')

oof_auc = roc_auc_score(y, oof)
test_pred = np.mean(test_fold_preds, axis=0)
print(f'\n======================================================')
print(f'POOLED OOF AUC = {oof_auc:.6f} | Total Time: {(time.time()-run_start)/60:.1f}m')
print(f'======================================================')

# Save arrays
np.save('generator_aware_ridge_oof.npy', oof)
np.save('generator_aware_ridge_test.npy', test_pred)

# Save submission
ridge_sub = pd.DataFrame({ID_COL: ids, TARGET: test_pred})
ridge_sub.to_csv('ridge_submission.csv', index=False)
ridge_sub.to_csv(SUBMISSION_DIR / 'submission_generator_aware_ridge.csv', index=False)
print('Saved ridge_submission.csv')

# 6. Apply Full 0.94677 Ridge Tower Blend with 4 Stable Rules
print('\nComputing Final 0.94677 Ridge Tower Blend...')
def average_rank(values):
    return pd.Series(values).rank(method='average').to_numpy(np.float64)

def probit_rank(values):
    probabilities = average_rank(values) / (len(values) + 1.0)
    normal = NormalDist()
    return np.fromiter((normal.inv_cdf(float(p)) for p in probabilities), dtype=np.float64, count=len(probabilities))

# Sources for the blend
# 1. Generator-aware ridge
ridge_preds = test_pred

# 2. EV Grand Prix later snapshot (our tribridged logit pinnacle or freq fleet)
gp_later_file = 'submissions/submission_tribridged_logit_pinnacle.csv'
gp_later = pd.read_csv(gp_later_file).set_index(ID_COL)[TARGET].reindex(ids).to_numpy(np.float64)

# 3. EV Grand Prix earlier snapshot (grand prix fleet pinnacle)
gp_earlier_file = 'submissions/submission_grand_prix_fleet_pinnacle.csv'
if not Path(gp_earlier_file).exists():
    gp_earlier_file = 'submissions/submission_freq_fleet_slsqp.csv'
gp_earlier = pd.read_csv(gp_earlier_file).set_index(ID_COL)[TARGET].reindex(ids).to_numpy(np.float64)

# Frozen weights from 0.94677 solution:
SOURCES = {
    'Ridge': (ridge_preds, 0.444707),
    'GP_later': (gp_later, 0.331541),
    'GP_earlier': (gp_earlier, 0.223752),
}

tower_score = np.zeros(len(ids), dtype=np.float64)
for name, (preds, weight) in SOURCES.items():
    tower_score += weight * probit_rank(preds)

# Four Stable Ordering Rules
income = test['Annual_Income_USD'].to_numpy(np.float64)
commute = test['Daily_Commute_km'].to_numpy(np.float64)
no_subsidy = test['Subsidy_Available'].astype(str).to_numpy() == 'No'
concern_1 = test['Environmental_Concern_Level'].to_numpy() == 1
anxious = np.isin(test['Range_Anxiety_Level'].astype(str).to_numpy(), ['Medium', 'High'])

rules = [
    ('Income >= 170537 -> top', income >= 170537, +1),
    ('31004 <= income <= 41970 -> bottom', (income >= 31004) & (income <= 41970), -1),
    ('Commute >= 83 km -> bottom', commute >= 83, -1),
    ('Income 30000 + no subsidy + concern/anxiety -> bottom', (income == 30000) & no_subsidy & (concern_1 | anxious), -1),
]

ordered = average_rank(tower_score) / len(tower_score)
for label, mask, direction in rules:
    ordered[mask] += 2.0 * direction
final_pred = average_rank(ordered) / len(ordered)

final_sub = pd.DataFrame({ID_COL: ids, TARGET: final_pred})
final_path = SUBMISSION_DIR / 'submission_0.94677_ridge_tower_4rules.csv'
final_sub.to_csv(final_path, index=False)
print(f'Successfully saved final blend to: {final_path}')
print('Done!')
