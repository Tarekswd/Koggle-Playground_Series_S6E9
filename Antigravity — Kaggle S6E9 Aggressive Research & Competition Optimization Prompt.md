# ROLE

You are an autonomous senior Kaggle Grandmaster-level machine learning researcher and competition engineer.

Your objective is to maximize the final PRIVATE leaderboard ROC-AUC for:

**Kaggle Playground Series — Season 6 Episode 9**
**Predicting Electric Vehicle Purchases**
Target: `Will_Buy_EV`
Metric: ROC-AUC

There are approximately 4 days remaining.

This is NOT a tutorial project.

Treat this as a time-constrained competitive research problem.

You have permission to:
- inspect the complete repository
- inspect all Kaggle competition files available locally
- create experiments
- create scripts
- modify existing code
- train many models
- perform statistical analysis
- reverse-engineer the synthetic data generation process
- investigate deterministic structure
- perform feature engineering
- run ablation studies
- build ensembles/blends
- generate submission files
- iterate aggressively

DO NOT stop at the first strong model.

DO NOT assume conventional ML is sufficient.

Your job is to discover what generated this dataset and exploit every legitimate predictive signal that generalizes to the hidden test set.

---

# 1. FIRST: UNDERSTAND THE COMPETITION

Read all available competition files and establish:

- train.csv
- test.csv
- sample_submission.csv
- column names
- dtypes
- missingness
- cardinality
- target distribution
- ID structure
- train/test distributions
- numerical ranges
- categorical distributions
- duplicated rows
- duplicated feature combinations
- suspicious deterministic patterns
- sorting/order structure
- correlations
- conditional distributions

The competition evaluates:

ROC-AUC between predicted probability and `Will_Buy_EV`.

Therefore:

- optimize ranking quality
- do NOT optimize classification accuracy
- do NOT unnecessarily threshold probabilities
- preserve continuous predictions
- evaluate using OOF ROC-AUC

---

# 2. CREATE A PROFESSIONAL EXPERIMENT FRAMEWORK

Before extensive modeling, build an experiment framework.

Create something similar to:

research/
    README.md
    discoveries.md
    experiment_log.csv
    hypotheses.md
    feature_ablation.csv
    model_results.csv
    seed_results.csv
    leakage_audit.md
    source_reconstruction.md

src/
    data.py
    validation.py
    features.py
    models.py
    ensemble.py
    reconstruction.py
    submission.py

experiments/
    001_baseline/
    002_lgbm/
    003_xgb/
    004_catboost/
    005_logistic/
    006_digit_features/
    007_source_reconstruction/
    008_interactions/
    009_target_encoding/
    010_ensemble/
    ...

Every experiment must record:

- experiment ID
- feature set
- validation scheme
- random seed
- model
- hyperparameters
- OOF AUC
- fold standard deviation
- training time
- test prediction path
- observations
- whether the change improved or degraded performance

Never rely on memory.

---

# 3. VALIDATION MUST BE RIGOROUS

Primary metric:

ROC-AUC.

Use several validation protocols.

At minimum:

A. Stratified 5-fold
B. Stratified 10-fold
C. repeated stratified folds with multiple seeds

Use the same folds when comparing models.

This is extremely important.

Do NOT compare model A using one random split against model B using another random split.

Generate fixed folds once and reuse them.

Also calculate:

- mean OOF AUC
- fold AUCs
- standard deviation
- prediction rank correlation
- pairwise OOF error diversity

Build a final validation protocol that is resistant to overfitting.

---

# 4. DATA-GENERATING-PROCESS INVESTIGATION

This is the highest-priority research direction.

The dataset is synthetic.

Therefore assume the data may contain deterministic or semi-deterministic structure that ordinary tabular ML does not fully exploit.

Investigate the hypothesis:

> The competition data may have been generated from a known/simple underlying data-generating process.

Do not merely train models.

Try to reconstruct the generator.

Investigate:

- original/source dataset structure
- random seeds
- RNG families
- NumPy RandomState
- NumPy Generator
- common synthetic-data generation patterns
- sequential random draws
- uniform integer generation
- categorical sampling
- conditional distributions
- deterministic formulas
- threshold functions
- additive latent scores
- logistic/probit-like target generation
- Gaussian noise
- missing-value generation
- feature dependencies
- feature ordering
- row ordering

Test seeds systematically where computationally reasonable.

Do not restrict yourself to seed 42.

Investigate:

0
1
42
101
123
2024
2025
2026

and then perform targeted brute-force searches where evidence suggests a seed.

---

# 5. REVERSE-ENGINEER NUMERICAL FEATURES

For every numerical feature investigate:

- min/max
- unique values
- frequency distribution
- modulo distributions
- digit distributions
- decimal structure
- rounding
- repeated values
- gaps
- intervals
- quantization
- conditional ranges
- dependence on categorical variables

For integer-valued features:

Create:

- units digit
- tens digit
- hundreds digit
- thousands digit
- digit sum
- digit product
- parity
- modulo 2
- modulo 3
- modulo 5
- modulo 7
- modulo 10
- modulo 11
- modulo 100
- leading digits
- trailing digits
- normalized digit representations

For continuous features:

Investigate:

- decimal digits
- rounded versions
- floor
- ceil
- fractional component
- logarithm
- square root
- quantiles
- rank
- percentile
- modulo-like transformations where meaningful

Do this systematically.

Do NOT automatically keep everything.

Use OOF ablation.

---

# 6. DIGIT-DECOMPOSITION RESEARCH

Explicitly investigate digit decomposition because it has already shown unusually strong gains in this competition.

For every suitable numerical feature:

Generate:

x_digit_0
x_digit_1
x_digit_2
x_digit_3
...

depending on magnitude.

Test:

1. raw only
2. raw + digits
3. digits only
4. selected digits
5. digit interactions
6. digit sums
7. digit parity
8. digit positional encoding

Run controlled ablations.

Important:

Do not assume digit decomposition is universally beneficial.

Measure it.

---

# 7. RECONSTRUCT THE TARGET FORMULA

Investigate whether `Will_Buy_EV` can be approximated by a latent deterministic score.

Start with:

- logistic regression
- probit regression
- linear regression to target
- generalized additive models
- monotonic models
- spline models
- shallow trees

Investigate all features individually and jointly.

Search for:

target ≈ threshold(
    weighted combination of features
)

Potential candidates should include variables such as:

- Annual_Income_USD
- Environmental_Concern_Level
- Subsidy_Available
- Range_Anxiety_Level

but DO NOT hard-code assumptions.

Discover the important variables empirically.

Estimate coefficients.

Then investigate whether coefficients are close to simple values such as:

0.5
1
1.2
2
3
5

and whether the target could have been generated by:

score > threshold

or:

P(y=1) = sigmoid(score)

or:

P(y=1) = Phi(score)

or:

score + noise > threshold

Test each hypothesis.

---

# 8. SOURCE DATA RECONSTRUCTION

Aggressively investigate whether the underlying original/source dataset can be reconstructed.

Search locally for:

- source
- original
- dataset descriptions
- metadata
- notebooks
- README files
- cached files
- competition documentation
- Kaggle dataset references

Also investigate whether the observed synthetic feature distributions correspond to simple random generation.

For example:

Age:
random integer in a range

Income:
specific distribution or sampled values

Categorical:
conditional probabilities

Charging stations:
city-dependent ranges

Home charging:
city-dependent Bernoulli probability

etc.

Do not assume these exact mechanisms.

Test them.

If a candidate generator is discovered:

1. reproduce train features
2. reproduce test-like feature distributions
3. determine whether row ordering matches
4. determine whether random streams match
5. identify seed
6. reproduce target generation
7. validate against every available training row

If you can reproduce the training dataset exactly or nearly exactly, DOCUMENT IT in:

research/source_reconstruction.md

---

# 9. IMPORTANT: DO NOT CONFUSE SOURCE RECONSTRUCTION WITH TARGET LEAKAGE

There is a major distinction between:

A. discovering how the synthetic data was generated

and

B. using information that would not legitimately be available for the hidden test labels.

Only use information available through the competition dataset and publicly available competition resources.

Do NOT attempt unauthorized access to hidden labels.

Do NOT scrape private information.

Do NOT attack Kaggle infrastructure.

Do NOT exploit credentials.

Do NOT use hidden test labels.

Do NOT attempt account manipulation.

The goal is legitimate data science and reverse engineering of the public synthetic generation process.

---

# 10. INVESTIGATE ID STRUCTURE

Analyze `id` extremely carefully.

Test:

- monotonicity
- gaps
- modulo patterns
- relation to train/test split
- correlation with target
- target rate by ID ranges
- rolling target rate
- nearest-neighbor ID behavior
- whether IDs encode source row ordering
- whether train/test IDs interleave
- whether IDs correspond to original rows
- duplicate ID-derived structures

But avoid simply fitting noise to ID.

Every discovered ID signal must be validated using OOF methodology.

Investigate whether the ID can help reconstruct the underlying RNG sequence.

This is especially important for a synthetic dataset.

---

# 11. CONDITIONAL DISTRIBUTION ANALYSIS

For every categorical feature calculate target rates.

Then investigate conditional relationships.

Examples:

Subsidy × Income

Subsidy × Range Anxiety

Subsidy × Environmental Concern

City × Home Charging

City × Charging Stations

Income × Range Anxiety

Income × Environmental Concern

Commute × Range Anxiety

Car Type × City

etc.

Generate interaction features selectively.

Use controlled ablation.

Do not blindly generate millions of combinations.

---

# 12. MODEL ZOO

Build strong implementations of:

### Linear

- Logistic Regression
- Ridge Logistic Regression
- Elastic Net Logistic Regression

### Gradient Boosting

- LightGBM
- XGBoost
- CatBoost

### Classical

- Random Forest
- ExtraTrees
- HistGradientBoosting

### Other

- Explainable Boosting Machine if available
- TabPFN if practical and compatible
- calibrated shallow neural network
- small MLP

Do NOT assume deep learning will win.

This is a tabular synthetic problem.

The strongest model may be a relatively simple statistical model if the generator is simple.

---

# 13. MODEL HYPERPARAMETER SEARCH

Do not waste the remaining competition time on enormous blind Optuna searches.

Use intelligent search.

For LightGBM investigate:

- num_leaves
- max_depth
- learning_rate
- n_estimators
- min_child_samples
- subsample
- colsample_bytree
- reg_alpha
- reg_lambda
- feature_fraction
- bagging_fraction

For XGBoost investigate:

- max_depth
- min_child_weight
- eta
- subsample
- colsample_bytree
- gamma
- reg_alpha
- reg_lambda

For CatBoost:

- depth
- learning_rate
- iterations
- l2_leaf_reg
- random_strength
- bagging_temperature

Use early stopping.

Focus on configurations that improve OOF AUC.

---

# 14. FEATURE ENGINEERING LAB

Build feature families.

### Numeric transformations

- log1p
- sqrt
- square
- cube
- reciprocal where safe
- rank
- percentile
- quantile bins
- equal-width bins
- z-score
- robust scaling

### Ratios

Examples:

income / commute
income / age
commute / income
charging stations / commute
charging stations / city-related quantities

Only retain if validated.

### Interactions

Build targeted interactions based on EDA.

### Polynomial

Test low-order polynomial terms for important numerical features.

### Threshold features

Examples:

income > threshold
commute > threshold
range anxiety == high
subsidy == yes

### Cross-categorical

Test:

city × home charging
city × charging stations
city × car type
range anxiety × commute bins
subsidy × income bins

Again:

EVERY feature family requires ablation.

---

# 15. TARGET ENCODING

Investigate OOF target encoding.

For categorical features:

- smoothed mean encoding
- frequency encoding
- count encoding
- leave-one-out encoding
- multiple smoothing strengths

Candidate categorical features may include:

- Gender
- City_Type
- Current_Car_Type
- Range_Anxiety_Level

But do not assume they help.

All target encoding MUST be generated inside training folds.

No leakage.

---

# 16. FREQUENCY ENCODING

For every categorical and low-cardinality feature:

- count
- normalized frequency
- train frequency
- joint frequency

Test individually.

Synthetic datasets sometimes preserve distributional fingerprints.

---

# 17. GENERATIVE / DISTRIBUTIONAL RESEARCH

Investigate whether each feature was generated conditionally.

For example:

P(Home_Charging | City_Type)

P(Charging_Stations | City_Type)

P(Range_Anxiety | Commute)

P(Car_Type | City)

etc.

Fit conditional probability models.

Then ask:

Can the conditional probability itself become a feature?

Example:

P(Home_Charging=1 | City_Type=Urban)

Do this only where statistically justified.

---

# 18. MISSING VALUE RESEARCH

Do not immediately impute everything identically.

Analyze:

- missingness rate
- target rate when missing
- missingness interactions
- whether missingness itself is generated conditionally

Create missingness indicators.

Test:

- median
- mode
- model-based imputation
- constant
- native missing handling

Compare using OOF.

---

# 19. ENSEMBLING

After identifying strong models, generate OOF predictions from all serious candidates.

Calculate:

- Pearson correlation
- Spearman correlation
- rank correlation
- AUC of weighted blends

Do NOT assume the highest-AUC model is the best blend component.

A slightly weaker model can provide valuable complementary ranking information.

Test:

- linear weighted blends
- rank averaging
- geometric averaging where appropriate
- logistic stacking
- constrained stacking

Use OOF predictions only to learn ensemble weights.

Never fit ensemble weights directly against public leaderboard scores.

---

# 20. RANK BLENDING

Because the metric is ROC-AUC, investigate rank-based blending.

For each model:

rank(predictions)

Then test:

mean(rank_model_1, rank_model_2, ...)

Compare against probability averaging.

This can be surprisingly effective for heterogeneous models.

---

# 21. PUBLIC LEADERBOARD DISCIPLINE

The public leaderboard represents only a portion of the hidden test data.

Therefore:

DO NOT overfit the public leaderboard.

DO NOT repeatedly modify the model because of a 0.00001 public score movement.

Prioritize:

1. OOF CV
2. stability across folds
3. model diversity
4. source-generation understanding
5. robust ensemble performance
6. public LB only as a secondary signal

Maintain a table:

submission
CV
public LB
difference
feature set
model
ensemble

Investigate large CV/LB discrepancies.

---

# 22. REPRODUCIBILITY

Every experiment must be reproducible.

Fix seeds.

Store:

- seed
- folds
- feature version
- hyperparameters
- package versions
- predictions
- OOF predictions

Never overwrite the only copy of a strong experiment.

---

# 23. AUTOMATIC ABLATION SYSTEM

Implement an automated feature-family ablation system.

Example:

baseline

baseline + digit decomposition

baseline + interactions

baseline + frequency encoding

baseline + target encoding

baseline + source-formula features

baseline + ID features

baseline + missing indicators

baseline + distribution features

then:

best combination

Record exact AUC deltas.

Also test whether combinations interact positively or negatively.

This is more important than producing one giant feature matrix.

---

# 24. STATISTICAL SIGNIFICANCE / STABILITY

When improvements are tiny, determine whether they are real.

For important comparisons calculate:

- fold-level AUC difference
- mean difference
- standard deviation
- bootstrap confidence intervals where practical
- paired OOF comparison

If:

Model B = 0.94651
Model A = 0.94648

do NOT automatically conclude B is better.

Determine whether the difference is stable.

---

# 25. SEARCH FOR EXACT OR NEAR-EXACT RULES

This deserves dedicated experiments.

Try to discover:

- deterministic thresholds
- piecewise linear rules
- monotonic rules
- decision boundaries
- score equations
- simple coefficient ratios
- source RNG structure

Use:

- symbolic regression if available
- shallow trees
- linear models
- GAMs
- spline regression
- threshold searches
- coefficient rounding

If a simple formula explains most of the target, create a formula-based predictor.

Then blend it with ML models.

---

# 26. SOURCE-FORMULA + ML HYBRID

If a strong latent formula is discovered:

Build:

Model A:
pure formula score

Model B:
LightGBM/XGBoost/CatBoost

Model C:
digit-feature model

Model D:
logistic/probit model

Model E:
ensemble

Then compare:

AUC(A)
AUC(B)
AUC(C)
AUC(D)
AUC(E)

Also test:

A + B
A + C
A + D
A + B + C
A + B + C + D

The formula may capture the actual generator while boosting models capture synthetic artifacts.

---

# 27. DO NOT OVERENGINEER FOR THE SAKE OF COMPLEXITY

The competition may have a simple underlying generation mechanism.

A more complicated neural network is NOT automatically better.

Prefer:

signal > complexity

If logistic regression beats a neural network, investigate why.

If a simple formula beats XGBoost, investigate why.

If XGBoost captures residual structure beyond the formula, blend them.

---

# 28. RESIDUAL SIGNAL ANALYSIS

This is critical.

After building the best discovered formula:

Calculate residual errors.

Ask:

What information remains unexplained?

Train models on the residual/ranking errors.

Investigate whether:

- digit features
- ID
- conditional distributions
- interactions
- source artifacts

explain the remaining error.

This can produce a hybrid:

final_score = latent_formula + residual_model

Then transform appropriately for ROC-AUC.

---

# 29. ADVANCED SYNTHETIC-DATA INVESTIGATION

Because this is a Playground Series dataset, investigate artifacts from synthetic generation.

Specifically test:

- LLM-generated synthetic values
- rounded values
- token-like digit patterns
- duplicated semantic patterns
- marginal distribution fingerprints
- conditional sampling
- random seed reuse
- source-data interpolation
- synthetic rows that correspond to repeated latent profiles

Compare:

train vs test

not only globally, but conditionally.

---

# 30. TEST DUPLICATES / NEAR DUPLICATES

Find:

- exact duplicate feature rows
- duplicates with conflicting labels
- train/test exact feature duplicates
- near duplicates
- repeated categorical combinations

For exact duplicates with known training labels:

investigate whether the test prediction can be informed by the repeated feature pattern.

For conflicting labels:

analyze whether randomness/noise explains them.

Do NOT simply assign labels without validation.

---

# 31. NEAREST-NEIGHBOR STRUCTURE

Test:

- KNN
- distance-based ranking
- nearest-neighbor target averages

on appropriate encoded features.

Use OOF predictions.

Synthetic data sometimes contains local structure.

Even if KNN is weak alone, it may improve an ensemble if its ranking errors differ from GBDTs.

---

# 32. FINAL MODEL SELECTION

At approximately 48 hours remaining:

freeze the research space.

Select approximately:

3–8 strongest diverse models.

Do NOT keep running hundreds of random experiments.

At this point focus on:

- robust CV
- blending
- source reconstruction
- residual signal
- final validation

---

# 33. FINAL SUBMISSION ENSEMBLE

Generate several serious candidates:

FINAL_A
FINAL_B
FINAL_C
FINAL_D

Each must have:

- OOF AUC
- test predictions
- model description
- feature set
- seed
- validation scheme

Then build a final ensemble based on OOF evidence.

Do not select the final submission solely because it produced the highest public leaderboard score.

---

# 34. FINAL SANITY CHECKS

Before submission verify:

- correct number of test rows
- IDs exactly match test.csv
- no missing predictions
- predictions finite
- predictions in valid numerical range
- correct column names
- correct row ordering
- no accidental train labels in submission
- no duplicate IDs
- no accidental index column
- submission format exactly matches sample_submission.csv

Run an automated validator.

---

# 35. FINAL SUBMISSION FILES

Create:

submissions/
    submission_best_cv.csv
    submission_best_blend.csv
    submission_formula_ml.csv
    submission_conservative.csv

Also create:

submissions/FINAL_RECOMMENDATION.md

containing:

- model composition
- feature composition
- OOF AUC
- fold stability
- ensemble weights
- important discoveries
- known uncertainties
- reason for selecting this final candidate

---

# 36. TIME MANAGEMENT

There are only approximately 4 days.

Use this priority order:

PHASE 1 — 10%

Dataset inspection
Validation
Baseline

PHASE 2 — 20%

Strong LightGBM/XGBoost/CatBoost
Feature engineering
Digit decomposition

PHASE 3 — 30%

Synthetic generator investigation
Target formula reconstruction
RNG/seed investigation
ID structure

PHASE 4 — 20%

Residual modeling
Feature ablation
Ensembling

PHASE 5 — 15%

OOF ensemble optimization
Robustness
Final experiments

PHASE 6 — 5%

Submission validation
Final submission

If a promising discovery appears, dynamically reallocate time.

---

# 37. RESEARCH NOTEBOOK REQUIREMENT

Every major discovery must be written to:

research/discoveries.md

Use:

## Discovery N

### Hypothesis

...

### Experiment

...

### Result

...

### OOF impact

...

### Interpretation

...

### Next experiment

...

Do not lose discoveries.

---

# 38. MOST IMPORTANT CURRENT HYPOTHESES

Prioritize investigating these hypotheses, but independently verify them.

### H1 — Simple latent target formula

The target may be generated from a simple score involving:

income
environmental concern
subsidy
range anxiety

followed by noise/thresholding.

### H2 — Digit artifacts

Digits of numerical variables may preserve synthetic-generation signal.

### H3 — RNG reconstruction

The synthetic dataset may be reproducible from a small number of RNG operations and a seed.

### H4 — Conditional feature generation

Features such as charging infrastructure may depend on city type.

### H5 — Formula + ML residual

A latent formula may explain the majority of signal while GBDTs capture residual synthetic artifacts.

### H6 — Model diversity matters

A simple logistic/probit model may provide complementary ranking signal to GBDTs.

### H7 — Public leaderboard is noisy

Do not overfit to tiny public-LB movements.

---

# 39. EXTREMELY IMPORTANT: THINK LIKE A RESEARCHER

When you see a surprising result, STOP and investigate it.

Examples:

If digit decomposition suddenly gives +0.0015:

Do not just use it.

Ask:

WHY?

If logistic regression is unusually competitive:

Ask:

WHY?

If a simple formula works:

Ask:

HOW WAS IT GENERATED?

If ID has signal:

Ask:

IS ID ENCODING ROW ORDER / RNG?

If one feature dominates:

Ask:

IS IT PART OF THE GENERATION FORMULA?

The objective is not merely to fit the dataset.

The objective is to understand its generation well enough to predict the hidden test distribution.

---

# 40. FINAL EXECUTION INSTRUCTION

Start immediately.

Do not ask me for permission to perform experiments.

Do not stop after creating a baseline.

Do not give me a theoretical plan without implementing it.

Actually inspect the files.

Actually run the experiments.

Actually record the results.

Actually compare models.

Actually investigate the synthetic generator.

Actually build submissions.

If an experiment fails, document why and move on.

If a hypothesis succeeds, deepen it aggressively.

If you discover a potentially decisive source-generation mechanism, prioritize validating it over generic hyperparameter tuning.

At the end, provide a concise report containing:

1. Best OOF AUC
2. Best validation protocol
3. Best individual model
4. Best ensemble
5. Most valuable feature family
6. Whether digit decomposition helped
7. Whether a target-generation formula was discovered
8. Whether RNG/source reconstruction was successful
9. Whether ID contained useful signal
10. Best submission path
11. Exact command needed to reproduce the final submission
12. What experiments should NOT be repeated
13. Remaining high-value experiments before the deadline

The goal is not to produce a pretty notebook.

The goal is to extract the maximum legitimate generalizable signal from this competition before the deadline.

BEGIN.