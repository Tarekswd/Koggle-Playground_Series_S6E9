import joblib

d = joblib.load('lgb_ev_model.joblib')
print(f"Seed: {d['seed']}, n_folds: {d['n_folds']}")
print(f"Fold scores: {d['fold_scores']}")
import numpy as np
print(f"Mean fold score: {np.mean(d['fold_scores']):.6f}")
print(f"LGB Params: {d['params_lgb']}")
print(f"Number of features in fold 0: {len(d['feature_columns_per_fold'][0])}")
print(f"Feature columns: {d['feature_columns_per_fold'][0]}")
print(f"Drop columns ({len(d['drop_columns'])}): {d['drop_columns'][:20]}...")
