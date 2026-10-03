# Competition Discoveries — Kaggle S6E9: Predicting Electric Vehicle Purchases

## Discovery 1: The Underlying Synthetic Target Generator Formula
- **Hypothesis:** Target `Will_Buy_EV` is generated via a latent additive score with a decision threshold.
- **Empirical Confirmation:**
  $$\text{Buy Score} = 1.2 \times \left(\frac{\text{Annual\_Income\_USD}}{100,000}\right) + 0.6 \times \text{Environmental\_Concern\_Level} + 2.0 \times \text{Subsidy\_Available} - 1.0 \times (\text{Range\_Anxiety == Medium}) - 3.0 \times (\text{Range\_Anxiety == High})$$
- **Result:** Raw score alone achieves an astonishing **0.937690 ROC-AUC** with 0 tree splits.
- **Interpretation:** The synthetic dataset generator (based on Omkar Kadam's 10,000-row EV Adoption dataset and scaled with CTGAN) directly used this latent buying propensity formula plus stochastic wobble around a ~5.5 threshold.

## Discovery 2: Deterministic Hard Clifts & Generator Artifacts
- **Millionaire Cliff:** Every single record with `Annual_Income_USD >= 170537` in the training set (393 out of 393, 100.000%) has `Will_Buy_EV == 'Yes'`. In the test set, exactly 156 rows meet this condition.
- **Zero Probability Floor:** Records with $\text{Buy Score} < 0.70260$ have 0 positive cases (5,595 out of 5,595 are `Will_Buy_EV == 'No'`). In the test set, 2,334 records fall into this zero region.
- **High Probability Ceiling:** Records with $\text{Buy Score} > 7.03543$ have 100% positive rate (230 out of 230 are `Will_Buy_EV == 'Yes'`). In the test set, 92 records fall into this region.

## Discovery 3: Inspection of Unzipped Models
- `buddy-scikitlearn-ev-v1.tar.gz`:
  - Contains `lgb_ev_model.joblib`, `oof_predictions.npy`, and `test_predictions.npy`.
  - Architecture: 5-fold Stratified `LGBMClassifier` using 75 engineered features, including **Digit Decomposition** (`inc_digit_*`, `commute_digit_*`), frequency encodings, and **Dual Target Encoding** (`smooth_10` and `smooth_auto`).
  - OOF ROC-AUC: **0.945832** across 668,665 training rows.
  - Test predictions: 286,571 probabilities matching `test.csv`.
- `heart-disease-xgboost-classifier-other-xgboost-v1.tar.gz`:
  - Contains `heart_disease_xgb_model.json`.
  - Trained on the UCI Cleveland Heart Disease dataset (features: `age`, `sex`, `cp`, `trestbps`, `chol`, `thalach`, etc.). Not related to EV dataset.

## Discovery 4: Empirical Ablation — Base Margin vs Tree Flexibility
- Providing the latent score as an explicit `init_score` / `base_margin` yielded Fold 0 AUC of **0.942160** vs **0.942438** without base margin ($\Delta = -0.000277$).
- **Reason:** Forcing a rigid linear base margin restricts tree gradient splits from fitting CTGAN non-linear density shifts and digit artifacts.
- **Conclusion:** The optimal strategy is using the formula features and post-processing boundaries in combination with high-capacity tree ensembles.
