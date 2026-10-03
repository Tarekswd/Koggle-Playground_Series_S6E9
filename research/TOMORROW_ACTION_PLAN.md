# Executive Plan & Prepared Execution Pipeline for S6E9

## Current Leaderboard Status
* **Current Best Score:** `0.94619` (Rank 955)
* **Target Grandmaster Tier:** `0.9466 - 0.9468+` (Rank 1–50)
* **Total Remaining Gap to Rank 2 (Paul Bryan Elefante):** `+0.00061` AUC (~12 million misordered pairs out of 19.7 billion).

---

## Why Previous Models Hit the 0.94619 Wall
1. **Model Duplication:** `submission_paul_elefante_rank2_grandmaster.csv` had a Pearson correlation of **0.999993** with the previous 0.94619 file due to excessive weighting on the anchor.
2. **Re-weighting Exhaustion:** Nelder-Mead direct AUC maximization on existing predictions showed $\Delta = \pm 0.000000$ gain. The existing tree predictions have been completely exhausted.
3. **The 85% Error Hotspot:** We proved mathematically that **85.48% of all 3.47 billion inverted pairs** come from `Subsidy == Yes` and `Range_Anxiety == Low`, where the global stacker AUC drops to `0.906350`.

---

## Prepared Pipeline for Tomorrow

We have written, verified, and placed all 3 production scripts in `src/`:

### Step 1: Chris Deotte's Base-Margin Hist-XGBoost (5 Folds)
* **Script:** [`src/train_deotte_base_margin_xgb.py`](file:///c:/Users/Admin/Desktop/kaggle%20challenges/challenge%202/src/train_deotte_base_margin_xgb.py)
* **Mathematics:** Initialized at the ground-truth macro-formula ($\text{logit}(p) = 2.1740 \cdot (\text{Score} - 5.5) - 0.2437$, raw AUC 0.93769) as `base_margin`.
* **Features:** 28 features including CTGAN digit remainders (`% 100`, `% 1000`, `% 5000`, `// 10000`), structural flags (`is_30k`, `in_dead_zone`, `is_cliff`), and charging infrastructure interactions.
* **Output:** `xgb_base_margin_oof.npy` & `xgb_base_margin_test.npy`.
* **Execution Time:** ~4 minutes.

### Step 2: Exact 75-Feature LightGBM Multi-Seed Variance Whitening
* **Script:** [`src/train_lgb75_multiseed_whitening.py`](file:///c:/Users/Admin/Desktop/kaggle%20challenges/challenge%202/src/train_lgb75_multiseed_whitening.py)
* **Mathematics:** Keeps the exact 5-fold split (seed 42) to maintain zero leakage on target encodings, while varying `random_state`, `feature_fraction_seed`, and `bagging_seed` across seeds 101 and 777. Log-odds averaging reduces Monte Carlo tree-subsample variance like $\frac{1}{\sqrt{K}}$.
* **Output:** `lgb75_multiseed_whitened_oof.npy` & `lgb75_multiseed_whitened_test.npy`.
* **Execution Time:** ~3 minutes.

### Step 3: Orthogonal Grandmaster Stacking & Deterministic Clamping
* **Script:** [`src/build_orthogonal_grandmaster_ensemble.py`](file:///c:/Users/Admin/Desktop/kaggle%20challenges/challenge%202/src/build_orthogonal_grandmaster_ensemble.py)
* **Mathematics:** Combines 4 orthogonal model families ($\rho < 0.990$):
  1. `lgb75_multiseed_whitened` (Tree baseline)
  2. `xgb_base_margin` (Deotte residual recipe)
  3. `cb79` (CatBoost symmetric oblivious trees, $\rho = 0.9875$)
  4. `ladder_lgb` (Quantization ladder, $\rho = 0.990$)
* **Post-Processing:** Applies 100% pure deterministic boundary rules ($S_5 < 0.702596 \implies 0.0$, $\text{Income} \ge 170537 \implies 1.0$).
* **Output:** `submissions/submission_orthogonal_grandmaster_pinnacle.csv`.
* **Execution Time:** ~10 seconds.

---

## Fast-Start Command for Tomorrow
Tomorrow, we can execute the full pipeline by running:
```powershell
python src/train_deotte_base_margin_xgb.py
python src/train_lgb75_multiseed_whitening.py
python src/build_orthogonal_grandmaster_ensemble.py
```
Total end-to-end execution time: **~7 to 8 minutes**.
