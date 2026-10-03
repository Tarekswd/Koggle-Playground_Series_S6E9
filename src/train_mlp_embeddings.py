"""
=============================================================================
ENTITY-EMBEDDING MLP — Genuinely Orthogonal to GBDT Ensemble
=============================================================================
Architecture:
  - Entity embeddings for 6 categorical features
  - StandardScaler normalization for 77 numeric features
  - Residual MLP: [512 → 256 → 128] with BatchNorm + GELU + Dropout
  - BCEWithLogitsLoss, AdamW + CosineAnnealing

Why this can help:
  - All 38 GBDTs use axis-aligned splits → they struggle with rotated boundaries
  - MLP learns dense, non-linear representations across ALL features simultaneously
  - Entity embeddings capture semantic similarity between categorical values
  - Truly orthogonal model → diversity in ensemble blending

Saves: mlp_oof.npy, mlp_test.npy, submissions/submission_mlp_blend.csv
=============================================================================
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# ── Categorical feature specs ─────────────────────────────────────────────────
CAT_COLS = [
    'Gender', 'City_Type', 'Current_Car_Type',
    'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level'
]

NUM_COLS_RAW = [
    'Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
    'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
    'Environmental_Concern_Level'
]


def clamp(preds, df):
    inc_raw = df['Annual_Income_USD'].values
    i = inc_raw / 100000.0
    e = df['Environmental_Concern_Level'].values.astype(float)
    s = (df['Subsidy_Available'] == 'Yes').astype(float).values
    m = (df['Range_Anxiety_Level'] == 'Medium').astype(float).values
    h = (df['Range_Anxiety_Level'] == 'High').astype(float).values
    score = 1.2*i + 0.6*e + 2.0*s - 1.0*m - 3.0*h
    c = preds.copy()
    c[inc_raw >= 170537] = 1.0
    c[score < 0.702596]  = 0.0
    c[score > 7.03543]   = 1.0
    return c


def smoothed_te(train_series, train_y, apply_series, global_prior, smoothing=10):
    stats = (pd.DataFrame({'k': train_series.astype(str), 'y': train_y})
               .groupby('k')['y'].agg(['count', 'mean']))
    te_map = ((stats['count'] * stats['mean'] + smoothing * global_prior) /
              (stats['count'] + smoothing)).to_dict()
    return apply_series.astype(str).map(te_map).fillna(global_prior).astype(np.float32).values


# ── Feature engineering ────────────────────────────────────────────────────────
RAW_COLS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
            'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
            'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

INTERACTION_PAIRS = [
    ('Subsidy_Available', 'Range_Anxiety_Level'),
    ('Subsidy_Available', 'City_Type'),
    ('Home_Charging_Possible', 'City_Type'),
    ('Subsidy_Available', 'Environmental_Concern_Level'),
]


def build_cat_codes(df, cat_encoders=None):
    """Returns dict {col: int_codes_array} and fitted encoders."""
    codes = {}
    encoders = {}
    for c in CAT_COLS:
        if cat_encoders is not None:
            # Apply fitted encoding
            enc = cat_encoders[c]
            codes[c] = df[c].map(enc).fillna(0).astype(np.int64).values
        else:
            uniq = sorted(df[c].astype(str).unique())
            enc = {v: i for i, v in enumerate(uniq)}
            encoders[c] = enc
            codes[c] = df[c].astype(str).map(enc).fillna(0).astype(np.int64).values
    return codes, encoders


def build_numeric_features(df, y, tr_idx, global_prior, trans_freq_maps, is_test=False, train_df=None):
    """Build numeric feature matrix (no categoricals). Returns np.float32 array."""
    X = {}
    ref = train_df if is_test else df

    # 1. Raw numeric features (will be scaled)
    for c in NUM_COLS_RAW:
        X[c] = df[c].values.astype(np.float32)

    # 2. Digit decompositions
    inc_int = df['Annual_Income_USD'].astype(int).values
    X['inc_d0'] = (inc_int % 10).astype(np.float32)
    X['inc_d1'] = ((inc_int // 10) % 10).astype(np.float32)
    X['inc_d2'] = ((inc_int // 100) % 10).astype(np.float32)
    X['inc_d3'] = ((inc_int // 1000) % 10).astype(np.float32)
    X['inc_is_170k'] = (inc_int >= 170537).astype(np.float32)
    X['inc_mod100']  = (inc_int % 100).astype(np.float32)
    X['inc_mod1000'] = (inc_int % 1000).astype(np.float32)

    age = df['Age'].values
    X['age_d0'] = (age % 10).astype(np.float32)
    X['age_d1'] = ((age // 10) % 10).astype(np.float32)

    commute = df['Daily_Commute_km'].values
    X['cm_d_m1'] = (np.round(commute * 10).astype(int) % 10).astype(np.float32)
    X['cm_d0']   = (commute.astype(int) % 10).astype(np.float32)
    X['cm_d1']   = ((commute.astype(int) // 10) % 10).astype(np.float32)

    st_home = df['Charging_Stations_Near_Home'].values
    X['sth_d0'] = (st_home % 10).astype(np.float32)
    X['sth_d1'] = ((st_home // 10) % 10).astype(np.float32)

    st_work = df['Charging_Stations_Near_Work'].values
    X['stw_d0'] = (st_work % 10).astype(np.float32)
    X['stw_d1'] = ((st_work // 10) % 10).astype(np.float32)

    # 3. Transductive freq maps
    for c in RAW_COLS:
        X[f'{c}_freq'] = df[c].astype(str).map(trans_freq_maps[c]).fillna(0.0).astype(np.float32).values

    # 4. Extra transductive digit freqs
    X['inc_d0_freq']   = (inc_int % 10).astype(str)
    X['inc_d0_freq']   = pd.Series(inc_int % 10).map(trans_freq_maps['inc_digit0_freq']).fillna(0.0).astype(np.float32).values
    X['inc_mod100_freq']  = pd.Series(inc_int % 100).map(trans_freq_maps['inc_mod100_freq']).fillna(0.0).astype(np.float32).values
    X['inc_mod1000_freq'] = pd.Series(inc_int % 1000).map(trans_freq_maps['inc_mod1000_freq']).fillna(0.0).astype(np.float32).values
    sub_ra = df['Subsidy_Available'].astype(str) + '__' + df['Range_Anxiety_Level'].astype(str)
    X['sub_ra_freq'] = sub_ra.map(trans_freq_maps['subsidy_ra_freq']).fillna(0.0).astype(np.float32).values

    # 5. Smoothed target encodings (OOF for train, full for test)
    tr_ref = ref if is_test else df.iloc[tr_idx]
    y_ref  = y if is_test else y[tr_idx]
    for c in RAW_COLS:
        te_vals = smoothed_te(tr_ref[c], y_ref, df[c], global_prior, 10)
        X[f'{c}_te10'] = te_vals
        te_auto = smoothed_te(tr_ref[c], y_ref, df[c], global_prior,
                              np.sqrt(len(tr_ref[c])) / len(tr_ref[c].astype(str).unique()))
        X[f'{c}_te_auto'] = te_auto

    # 6. Joint interaction TEs
    for c1, c2 in INTERACTION_PAIRS:
        pair_all = df[c1].astype(str) + '__' + df[c2].astype(str)
        pair_tr  = tr_ref[c1].astype(str) + '__' + tr_ref[c2].astype(str)
        X[f'te_{c1}_{c2}'] = smoothed_te(pair_tr, y_ref, pair_all, global_prior, 20)

    return np.column_stack(list(X.values()))  # shape (N, n_features)


# ── MLP Architecture ───────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class EVAdoptionMLP(nn.Module):
    def __init__(self, cat_vocab_sizes, n_numeric, hidden=384, dropout=0.25):
        super().__init__()
        # Entity embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, max(2, n_cats // 2))
            for n_cats in cat_vocab_sizes
        ])
        emb_total = sum(max(2, n // 2) for n in cat_vocab_sizes)

        input_dim = emb_total + n_numeric
        self.bn_input = nn.BatchNorm1d(input_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = ResBlock(hidden, dropout)
        self.block2 = ResBlock(hidden, dropout * 0.8)
        self.block3 = ResBlock(hidden // 2, dropout * 0.6) if hidden >= 256 else ResBlock(hidden, dropout * 0.6)
        self.down = nn.Linear(hidden, hidden // 2) if hidden >= 256 else nn.Identity()
        self.bn_down = nn.BatchNorm1d(hidden // 2) if hidden >= 256 else nn.Identity()
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        x = self.bn_input(x)
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.nn.functional.gelu(self.bn_down(self.down(x)))
        x = self.block3(x)
        return self.head(x).squeeze(1)


# ── Training loop ──────────────────────────────────────────────────────────────
def train_one_fold(X_cat_tr, X_num_tr, y_tr, X_cat_va, X_num_va, y_va,
                   cat_vocab_sizes, n_numeric, n_epochs=80, batch_size=8192, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EVAdoptionMLP(cat_vocab_sizes, n_numeric, hidden=384, dropout=0.25).to(DEVICE)

    # Class imbalance weight
    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    tr_ds = TensorDataset(
        torch.from_numpy(X_cat_tr).long(),
        torch.from_numpy(X_num_tr).float(),
        torch.from_numpy(y_tr).float()
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    X_cat_va_t = torch.from_numpy(X_cat_va).long().to(DEVICE)
    X_num_va_t = torch.from_numpy(X_num_va).float().to(DEVICE)

    best_auc   = 0.0
    best_state = None
    patience   = 0
    PATIENCE   = 15

    for epoch in range(n_epochs):
        model.train()
        for xc, xn, yb in tr_loader:
            xc, xn, yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xc, xn)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_cat_va_t, X_num_va_t).cpu().numpy()
        val_probs = expit(val_logits)
        auc = roc_auc_score(y_va, val_probs)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_auc


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("ENTITY-EMBEDDING MLP — ORTHOGONAL TO GBDT ENSEMBLE", flush=True)
    print("=" * 80, flush=True)

    BASE = 'playground-series-s6e9'
    train = pd.read_csv(f'{BASE}/train.csv')
    test  = pd.read_csv(f'{BASE}/test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    N_TRAIN, N_TEST = len(train), len(test)
    global_prior = y.mean()
    print(f"Train: {N_TRAIN:,} | Test: {N_TEST:,} | Buy rate: {global_prior:.4f}", flush=True)

    # ── Build transductive freq maps ──────────────────────────────────────────
    train_f = train.drop(columns=['Will_Buy_EV', 'id'], errors='ignore')
    test_f  = test.drop(columns=['id'], errors='ignore')
    combined = pd.concat([train_f, test_f], ignore_index=True)
    trans_freq_maps = {}
    for c in RAW_COLS:
        trans_freq_maps[c] = combined[c].astype(str).value_counts(normalize=True).to_dict()
    ci = combined['Annual_Income_USD'].astype(int)
    trans_freq_maps['inc_digit0_freq']  = (ci % 10).value_counts(normalize=True).to_dict()
    trans_freq_maps['inc_mod100_freq']  = (ci % 100).value_counts(normalize=True).to_dict()
    trans_freq_maps['inc_mod1000_freq'] = (ci % 1000).value_counts(normalize=True).to_dict()
    combined['_sub_ra'] = combined['Subsidy_Available'].astype(str) + '__' + combined['Range_Anxiety_Level'].astype(str)
    trans_freq_maps['subsidy_ra_freq'] = combined['_sub_ra'].value_counts(normalize=True).to_dict()
    print(f"Built {len(trans_freq_maps)} transductive freq maps", flush=True)

    # ── Categorical encodings (fitted on full train+test for vocab sizes) ──────
    _, cat_encoders = build_cat_codes(combined)
    cat_vocab_sizes = []
    for c in CAT_COLS:
        enc = cat_encoders[c]
        cat_vocab_sizes.append(len(enc) + 1)  # +1 for OOV
    print(f"Cat vocab sizes: {dict(zip(CAT_COLS, cat_vocab_sizes))}", flush=True)

    # ── Build FULL numeric feature matrix (fold-independent parts) ─────────────
    print("\nBuilding feature matrices...", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(train, y))

    # Get train cat codes
    X_cat_train, _ = build_cat_codes(train, cat_encoders)
    X_cat_train = np.column_stack(list(X_cat_train.values()))  # (N_TRAIN, 6)
    X_cat_test,  _ = build_cat_codes(test, cat_encoders)
    X_cat_test  = np.column_stack(list(X_cat_test.values()))   # (N_TEST, 6)

    # ── 5-Fold OOF ───────────────────────────────────────────────────────────
    mlp_oof  = np.zeros(N_TRAIN)
    mlp_test = np.zeros(N_TEST)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        print(f"\n  --- Fold {fold+1}/5 ---", flush=True)
        t_fold = time.time()

        # Build numeric features (fold-specific TEs for train, full for test)
        X_num_tr_fold = build_numeric_features(
            train, y, tr_idx, global_prior, trans_freq_maps, is_test=False)
        X_num_te_fold = build_numeric_features(
            test, y, tr_idx, global_prior, trans_freq_maps, is_test=True, train_df=train)

        n_numeric = X_num_tr_fold.shape[1]
        if fold == 0:
            print(f"  Numeric features: {n_numeric}", flush=True)

        # Scale numeric features (fit on train fold, apply to val and test)
        scaler = StandardScaler()
        X_num_scaled_tr = scaler.fit_transform(X_num_tr_fold[tr_idx]).astype(np.float32)
        X_num_scaled_va = scaler.transform(X_num_tr_fold[val_idx]).astype(np.float32)
        X_num_scaled_te = scaler.transform(X_num_te_fold).astype(np.float32)

        # NaN check
        X_num_scaled_tr = np.nan_to_num(X_num_scaled_tr, nan=0.0, posinf=5.0, neginf=-5.0)
        X_num_scaled_va = np.nan_to_num(X_num_scaled_va, nan=0.0, posinf=5.0, neginf=-5.0)
        X_num_scaled_te = np.nan_to_num(X_num_scaled_te, nan=0.0, posinf=5.0, neginf=-5.0)

        X_cat_tr_fold = X_cat_train[tr_idx]
        X_cat_va_fold = X_cat_train[val_idx]
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[val_idx]

        model, best_auc = train_one_fold(
            X_cat_tr_fold, X_num_scaled_tr, y_tr,
            X_cat_va_fold, X_num_scaled_va, y_va,
            cat_vocab_sizes, n_numeric,
            n_epochs=80, batch_size=8192, seed=fold * 7 + 42
        )

        # OOF predictions
        model.eval()
        with torch.no_grad():
            val_logits = model(
                torch.from_numpy(X_cat_va_fold).long().to(DEVICE),
                torch.from_numpy(X_num_scaled_va).float().to(DEVICE)
            ).cpu().numpy()
        mlp_oof[val_idx] = expit(val_logits)

        # Test predictions (average across folds)
        with torch.no_grad():
            te_logits = model(
                torch.from_numpy(X_cat_test).long().to(DEVICE),
                torch.from_numpy(X_num_scaled_te).float().to(DEVICE)
            ).cpu().numpy()
        mlp_test += expit(te_logits) / 5.0

        fold_auc = roc_auc_score(y_va, mlp_oof[val_idx])
        print(f"  Fold {fold+1} val AUC: {fold_auc:.6f} (best during training: {best_auc:.6f}) [{time.time()-t_fold:.1f}s]", flush=True)

    mlp_auc = roc_auc_score(y, mlp_oof)
    print(f"\nMLP OOF AUC: {mlp_auc:.6f}", flush=True)
    np.save('mlp_oof.npy', mlp_oof)
    np.save('mlp_test.npy', mlp_test)

    # ── Meta-stack with workspace engines ─────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("META-STACKING MLP + WORKSPACE ENGINES", flush=True)
    print("=" * 80, flush=True)

    def to_logit(p): return logit(np.clip(p, 1e-6, 1-1e-6))

    engines = {
        'mlp':         (mlp_oof,  mlp_test),
        'ws_stacker':  (np.load('stacking_logit_pinnacle_oof.npy'),  np.load('stacking_logit_pinnacle_test.npy')),
        'ws_lgb75':    (np.load('oof_predictions.npy'),               np.load('test_predictions.npy')),
        'ws_xgb_meta': (np.load('xgb_elefante_meta_oof.npy'),        np.load('xgb_elefante_meta_test.npy')),
        'ws_lgb_white':(np.load('lgb75_multiseed_whitened_oof.npy'), np.load('lgb75_multiseed_whitened_test.npy')),
        'ws_optuna':   (np.load('optuna_lgb_oof.npy'),               np.load('optuna_lgb_test.npy')),
        'ws_ladder':   (np.load('ladder_lgb_oof.npy'),               np.load('ladder_lgb_test_preds.npy')),
    }

    for name, (oo, _) in engines.items():
        print(f"  {name:15s}: OOF AUC = {roc_auc_score(y, oo):.6f}", flush=True)

    Z_tr = np.column_stack([to_logit(v[0]) for v in engines.values()])
    Z_te = np.column_stack([to_logit(v[1]) for v in engines.values()])

    meta = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
    meta.fit(Z_tr, y)

    final_oof  = clamp(meta.predict_proba(Z_tr)[:, 1], train)
    final_test = clamp(meta.predict_proba(Z_te)[:, 1], test)
    final_auc  = roc_auc_score(y, final_oof)

    print("\n" + "*" * 80, flush=True)
    print(f"*** MLP BLEND FINAL OOF AUC: {final_auc:.6f} ***", flush=True)
    print(f"    Previous best: 0.946072", flush=True)
    print(f"    Net OOF Advance: {final_auc - 0.946072:+.6f}", flush=True)
    print(f"    Expected LB: ~{0.94621 + (final_auc - 0.946072):.5f}", flush=True)
    print(f"    MLP weight in meta: {meta.coef_[0][0]:.4f}", flush=True)
    print("*" * 80, flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)

    out_csv = 'submissions/submission_mlp_blend.csv'
    sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test})
    sub.to_csv(out_csv, index=False)
    assert len(sub) == N_TEST and not sub['Will_Buy_EV'].isnull().any()
    print(f"Saved: {out_csv}", flush=True)


if __name__ == '__main__':
    main()
