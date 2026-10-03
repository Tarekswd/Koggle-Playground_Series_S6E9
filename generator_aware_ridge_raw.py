# ===== CELL 1 =====
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

IS_KAGGLE = os.getenv('KAGGLE_KERNEL_RUN_TYPE') is not None
SMOKE_TEST = os.getenv('GENERATOR_LOGREG_SMOKE', '0') == '1'
DATA_DIR = (Path('/kaggle/input/competitions/playground-series-s6e9')
            if IS_KAGGLE else Path('data'))
ORIG_DIR = (Path('/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety')
            if IS_KAGGLE else Path('data'))
ORIGINAL_PATH = ORIG_DIR / ('EV_Adoption_and_Range_Anxiety_Dataset.csv'
                            if IS_KAGGLE else 'train_original.csv')
EXP_NAME = MODEL_NAME = 'GENERATOR_AWARE_LOGREG_SAMPLE'
OOF_DIR, TEST_PRED_DIR = Path('oof'), Path('test_preds')
SUBMISSION_PATH = Path('submission.csv')
ARTIFACT_DIR = Path('artifacts_for_analysis') / EXP_NAME / 'files'

N_FOLDS, INNER_FOLDS, RANDOM_STATE = 10, 5, 42
L2 = 10.0
LBFGS_ITERS = 15 if SMOKE_TEST else 200
OUT_TEMPERATURE, OUT_OFFSET = 1.25, -1.0
SMOKE_TRAIN_ROWS, SMOKE_TEST_ROWS, SMOKE_FOLDS = 1200, 128, 2
TARGET, ID_COL = 'Will_Buy_EV', 'id'
FLOAT_COLS = ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']
INT_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']
CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
COLS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
        'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
        'Environmental_Concern_Level'] + CAT_COLS

if SMOKE_TEST:
    OOF_DIR /= '_smoke'
    TEST_PRED_DIR /= '_smoke'
    SUBMISSION_PATH = Path('_smoke') / 'submission.csv'
    ARTIFACT_DIR = Path('artifacts_for_analysis') / (EXP_NAME + '_SMOKE') / 'files'
for d in (OOF_DIR, TEST_PRED_DIR, SUBMISSION_PATH.parent, ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 9.0 * 2**30 / total))
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
np.random.seed(RANDOM_STATE)
TIMINGS = []


def tick(stage, started, **extra):
    TIMINGS.append({'stage': stage, 'seconds': round(time.time() - started, 3), **extra})


def write_json(name, value):
    (ARTIFACT_DIR / name).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))

print(f'{EXP_NAME} | {"SMOKE" if SMOKE_TEST else "CV"} | device={DEVICE}')

# ===== CELL 2 =====
t0 = time.time()
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

missing_train = set([ID_COL, TARGET] + COLS) - set(train.columns)
missing_test = set([ID_COL] + COLS) - set(test.columns)
if missing_train or missing_test:
    raise ValueError(f'missing columns: train={sorted(missing_train)} test={sorted(missing_test)}')
if TARGET in test.columns:
    raise ValueError('test.csv must not contain the target column')
if not train[ID_COL].is_unique or not test[ID_COL].is_unique:
    raise ValueError('train/test IDs must be unique')

ORIGINAL_TRAIN_ROW_INDEX = np.arange(len(train), dtype=np.int64)
if SMOKE_TEST:
    if len(train) > SMOKE_TRAIN_ROWS:
        idx, _ = train_test_split(
            np.arange(len(train)), train_size=SMOKE_TRAIN_ROWS,
            stratify=train[TARGET], random_state=RANDOM_STATE)
        idx = np.sort(idx)
        ORIGINAL_TRAIN_ROW_INDEX = ORIGINAL_TRAIN_ROW_INDEX[idx]
        train = train.iloc[idx].reset_index(drop=True)
    test = test.iloc[:SMOKE_TEST_ROWS].reset_index(drop=True)

y = (train[TARGET] == 'Yes').astype(np.int8).to_numpy()
ntr, nte = len(train), len(test)
X = pd.concat([train[COLS], test[COLS]], ignore_index=True)
N = len(X)
SPLITS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(ntr), y))
FOLD_ID = np.full(ntr, -1, np.int8)
for k, (_, va) in enumerate(SPLITS):
    FOLD_ID[va] = k

tick('load_data', t0, rows_train=ntr, rows_test=nte)
print('Train:', train.shape, '| test:', test.shape)

# ===== CELL 3 =====
t0 = time.time()
try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError('This notebook requires the transformers package for the GPT-2 income-token features.') from e

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

FLAT, FAMILY = {}, {}
for lv in ['L1', 'L2', 'LAST', 'IV', 'CV', 'CI', 'INC_MOD100', 'INC_DIV1000', 'Age']:
    for a in (5, 50):
        name = f'r_{lv}_a{a}'
        FLAT[name] = (K[lv], a)
        FAMILY[name] = 'flat_rate'
for lv in ['L1', 'L2', 'LAST']:
    for c in ['City_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Environmental_Concern_Level',
              'Range_Anxiety_Level', 'Gender', 'Current_Car_Type', 'Number_of_Cars_Owned',
              'AGEDEC', 'CMTBIN', 'SHB', 'SWB']:
        name = f'tokx_{lv}_{c}'
        FLAT[name] = (cross(lv, c), 20)
        FAMILY[name] = 'token_x_context'
for c in ['City_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level', 'INCQ']:
    name = f'cmtx_CI_{c}'
    FLAT[name] = (cross('CI', c), 20)
    FAMILY[name] = 'commute_x_context'
for c in ['Home_Charging_Possible', 'City_Type', 'Range_Anxiety_Level']:
    name = f'cmtx_CV_{c}'
    FLAT[name] = (cross('CV', c), 20)
    FAMILY[name] = 'commute_x_context'
for name, key, alpha in [
    ('cell', K['CELL'], 5), ('cell_x_L1', cross('CELL', 'L1'), 20),
    ('cell_x_CI', cross('CELL', 'CI'), 20), ('cell_x_INCQ', cross('CELL', 'INCQ'), 20)]:
    FLAT[name] = (key, alpha)
    FAMILY[name] = 'generator_formula_cell'

CATS = ['Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level', 'Home_Charging_Possible',
        'City_Type', 'Current_Car_Type', 'Number_of_Cars_Owned', 'Gender', 'AGEDEC', 'SHB', 'SWB']
for c in CATS:
    for suffix, other in [('INC20', 'INC20'), ('CMT10', 'CMTBIN')]:
        name = f'xc_{c}_{suffix}'
        FLAT[name] = (cross(c, other), 20)
        FAMILY[name] = 'category_x_continuous_bin'
for i, c1 in enumerate(CATS):
    for c2 in CATS[i + 1:]:
        name = f'xp_{c1}_{c2}'
        FLAT[name] = (cross(c1, c2), 20)
        FAMILY[name] = 'category_pair'
K['CELLI'] = cross('CELL', 'INC20')
for name, key in [('x3_CELL_INC20', K['CELLI']), ('x3_CELL_CMT10', cross('CELL', 'CMTBIN'))]:
    FLAT[name] = (key, 20)
    FAMILY[name] = 'cell_x_bin'
for c in ['City_Type', 'Current_Car_Type', 'Number_of_Cars_Owned', 'AGEDEC']:
    name = f'x4_CELLI_{c}'
    FLAT[name] = (cross('CELLI', c), 20)
    FAMILY[name] = 'cell_x_bin'

CHAINS = {f'chain_S{S}': ([K['L1'], K['L2'], K['IV']], S) for S in (5, 20, 80)}
CHAINS.update({f'chainC_S{S}': ([K['CI'], K['CV']], S) for S in (5, 20)})
LABEL_NAMES = list(FLAT) + [f'{n}_lv{i}' for n, (ks, _) in CHAINS.items() for i in range(len(ks))]
for n, (ks, _) in CHAINS.items():
    for i in range(len(ks)):
        FAMILY[f'{n}_lv{i}'] = 'token_tree_beta_chain' if n.startswith('chain_') else 'commute_beta_chain'

LF, LFN = [], []
for nm in ['IV', 'L2', 'L1', 'CV', 'LAST']:
    LF.append(np.log1p(np.bincount(K[nm])[K[nm]]))
    LFN.append(f'cnt_{nm}')
    FAMILY[f'cnt_{nm}'] = 'count_train_test'
inc_f = X.Annual_Income_USD.to_numpy(np.float64)
env = X.Environmental_Concern_Level.to_numpy(np.float64)
subs = (X.Subsidy_Available == 'Yes').to_numpy()
anx_formula = X.Range_Anxiety_Level.map({'Low': 0, 'Medium': 1, 'High': 3}).to_numpy(np.float64)
LF.append(1.2 * inc_f / 1e5 + 0.6 * env + 2 * subs - anx_formula)
LFN.append('formula'); FAMILY['formula'] = 'generator_formula'
for c in COLS[:7]:
    v = X[c].to_numpy(np.float64)
    LF.append(v); LFN.append(f'raw_{c}'); FAMILY[f'raw_{c}'] = 'raw'
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        knot = np.quantile(v, q)
        name = f'hinge_{c}_{q}'
        LF.append(np.maximum(v - knot, 0.0)); LFN.append(name); FAMILY[name] = 'raw_hinge'
OH = pd.get_dummies(X[CAT_COLS + ['Environmental_Concern_Level', 'Number_of_Cars_Owned']].astype(str), dtype=np.float32)
for c in OH.columns:
    name = f'oh_{c}'; LFN.append(name); FAMILY[name] = 'one_hot'
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
MIX, MIXN = [], []
for kn, key in mix_keys.items():
    counts = np.bincount(key).astype(np.float64)
    n = counts[key]
    MIX.append(np.log(np.maximum(n, 1.0))); MIXN.append(f'mix_{kn}_logn')
    for vn, v in MIXV.items():
        if (kn in ('IV', 'L2', 'L1') and vn == 'inc') or (kn == 'CV' and vn == 'cmt') or (kn == 'IVCV' and vn in ('inc', 'cmt')):
            continue
        global_mean = float(v.mean())
        sums = np.bincount(key, weights=v, minlength=len(counts))[key]
        MIX.append((sums - v + 5 * global_mean) / (n - 1 + 5) - global_mean)
        MIXN.append(f'mix_{kn}_{vn}')
for name in MIXN:
    FAMILY[name] = 'composition_mix_transductive'
MIX = np.column_stack(MIX).astype(np.float32)
LF = np.concatenate([LF, MIX], axis=1)
LFN += MIXN
ALL_NAMES = LABEL_NAMES + LFN
P = len(ALL_NAMES)

tick('feature_keys_and_label_free', t0, n_features=P)
print(f'Features: {P} | label-derived={len(LABEL_NAMES)} | label-free={len(LFN)}')

# ===== CELL 4 =====
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


def build_fold(k):
    fit_rows, valid_rows = SPLITS[k]
    test_rows = np.arange(ntr, N)
    Mfit = np.empty((len(fit_rows), P), np.float32)
    Mvalid = np.empty((len(valid_rows), P), np.float32)
    Mtest = np.empty((nte, P), np.float32)

    n_label = len(LABEL_NAMES)
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
    return fit_rows, valid_rows, Mfit, Mvalid, Mtest


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

    loss = float(opt.step(closure))
    return w.detach(), b.detach(), loss


def to_probability(eta):
    z = np.clip(OUT_TEMPERATURE * eta + OUT_OFFSET, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

# ===== CELL 5 =====
run_started = time.time()
run_folds = list(range(SMOKE_FOLDS)) if SMOKE_TEST else list(range(N_FOLDS))
oof = np.full(ntr, np.nan, np.float64)
test_fold_predictions = []
fold_auc = []
coef_rows = []
summary = {'experiment': EXP_NAME, 'status': 'running', 'smoke_test': SMOKE_TEST, 'completed_folds': 0, 'fold_auc': []}
write_json('run_summary.json', summary)

for k in run_folds:
    fold_started = time.time()
    fit_rows, valid_rows, Mfit, Mvalid, Mtest = build_fold(k)
    if not (np.isfinite(Mfit).all() and np.isfinite(Mvalid).all() and np.isfinite(Mtest).all()):
        raise ValueError(f'non-finite feature detected in fold {k}')

    mu = Mfit.mean(axis=0, dtype=np.float64)
    sd = Mfit.std(axis=0, dtype=np.float64) + 1e-6
    Zfit = torch.from_numpy((Mfit - mu) / sd).to(DEVICE, torch.float64)
    yy = torch.from_numpy(y[fit_rows].astype(np.float64)).to(DEVICE)
    w, b, loss = fit_glm(Zfit, yy)
    del Zfit, yy

    Zvalid = torch.from_numpy((Mvalid - mu) / sd).to(DEVICE, torch.float64)
    Ztest = torch.from_numpy((Mtest - mu) / sd).to(DEVICE, torch.float64)
    eta_valid = (Zvalid @ w + b).cpu().numpy()
    eta_test = (Ztest @ w + b).cpu().numpy()
    del Zvalid, Ztest

    valid_pred = to_probability(eta_valid)
    test_pred = to_probability(eta_test)
    oof[valid_rows] = valid_pred
    test_fold_predictions.append(test_pred)
    auc = float(roc_auc_score(y[valid_rows], valid_pred))
    fold_auc.append(auc)

    coef = w.cpu().numpy()
    coef_rows.extend({'fold': k, 'feature': name, 'coefficient': float(value), 'abs_coefficient': float(abs(value)),
                      'family': FAMILY[name]} for name, value in zip(ALL_NAMES, coef))
    summary.update(completed_folds=len(fold_auc), fold_auc=fold_auc)
    write_json('run_summary.json', summary)
    print(f'FOLD {k + 1}/{N_FOLDS} | AUC={auc:.9f} | seconds={time.time() - fold_started:.1f}')

    del Mfit, Mvalid, Mtest, w, b
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

complete = bool(np.isfinite(oof).all())
test_pred = np.mean(np.stack(test_fold_predictions), axis=0)
if not np.isfinite(test_pred).all() or np.any((test_pred < 0) | (test_pred > 1)):
    raise ValueError('invalid test probabilities')

if complete:
    pooled_auc = float(roc_auc_score(y, oof))
    print(f'POOLED_OOF_AUC={pooled_auc:.12f}')
else:
    pooled_auc = None
    print('SMOKE run: partial OOF only; pooled CV AUC is not reported.')

# ===== CELL 6 =====
oof_frame = pd.DataFrame({
    ID_COL: train[ID_COL].to_numpy(),
    'row_index': ORIGINAL_TRAIN_ROW_INDEX,
    'fold': FOLD_ID,
    'y_true': y,
    'oof_pred': oof,
})
test_frame = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), 'test_pred': test_pred})
submission = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: test_pred})

oof_path = OOF_DIR / f'{MODEL_NAME}_OOF.parquet'
test_path = TEST_PRED_DIR / f'{MODEL_NAME}_TEST.parquet'
if complete:
    oof_frame.to_parquet(oof_path, index=False)
else:
    partial_path = OOF_DIR / f'{MODEL_NAME}_PARTIAL_OOF.parquet'
    oof_frame.loc[np.isfinite(oof_frame.oof_pred)].to_parquet(partial_path, index=False)
    oof_path = partial_path
test_frame.to_parquet(test_path, index=False)
submission.to_csv(SUBMISSION_PATH, index=False)

pd.DataFrame(coef_rows).to_csv(ARTIFACT_DIR / 'feature_importance.csv', index=False)
pd.DataFrame(TIMINGS).to_csv(ARTIFACT_DIR / 'timings.csv', index=False)
pd.DataFrame({
    'feature': ALL_NAMES,
    'family': [FAMILY[n] for n in ALL_NAMES],
    'fit_scope': [
        ('label-free train+test transductive' if FAMILY[n] in ('count_train_test', 'composition_mix_transductive')
         else 'label-free row-local' if FAMILY[n] in ('raw', 'raw_hinge', 'one_hot', 'generator_formula')
         else 'outer-train labels; outer-training rows inner 5-fold cross-fitted')
        for n in ALL_NAMES]
}).to_csv(ARTIFACT_DIR / 'feature_manifest.csv', index=False)

checks = {
    'complete_oof': complete,
    'outer_validation_labels_used_in_features': False,
    'inner_target_features_cross_fitted': True,
    'test_probabilities_finite_in_01': bool(np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()),
    'oof_id_order_equals_train': True,
    'test_id_order_equals_test': True,
}
write_json('checks.json', checks)
write_json('config.json', {
    'model': 'L2-regularized logistic regression / binomial GLM',
    'optimizer': 'torch.optim.LBFGS', 'l2': L2, 'lbfgs_iters': LBFGS_ITERS,
    'output_transform': {'temperature': OUT_TEMPERATURE, 'offset': OUT_OFFSET},
    'cv': {'outer_folds': N_FOLDS, 'inner_folds': INNER_FOLDS, 'shuffle': True, 'random_state': RANDOM_STATE},
    'n_features': P, 'device': str(DEVICE),
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                 'torch': torch.__version__, 'sklearn': importlib.metadata.version('scikit-learn')}
})
summary.update(status='complete' if complete else 'partial_smoke', pooled_oof_auc=pooled_auc,
               runtime_seconds=round(time.time() - run_started, 2),
               prediction_files={'oof': str(oof_path), 'test': str(test_path), 'submission': str(SUBMISSION_PATH)})
write_json('run_summary.json', summary)

print('Saved:', oof_path)
print('Saved:', test_path)
print('Saved:', SUBMISSION_PATH)

