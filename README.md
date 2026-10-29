# BRFSS Heart Disease Prediction — ML Project 1

Binary classification of heart disease (`_MICHD`: ever told you had a heart
attack or coronary heart disease) on the BRFSS 2015 survey dataset

---

## Repository structure

| file | role |
|---|---|
| `helpers.py` | course-provided: `load_csv_data`, `read_header`, `one_hot`, `k_fold_split`, `create_csv_submission` |
| `implementations.py` | graded solvers: GD/SGD least squares, least squares, ridge, logistic regression (standard, L2-regularised, class-weighted) |
| `preprocessing.py` | the data pipeline: variable selection, sentinel decoding, imputation, NaN indicators, quantile clipping, one-hot encoding |
| `training.py` | `LogisticClassifier` — sklearn-style wrapper around the graded logistic solvers |
| `diagnostics.py` | `Diagnostics.evaluate / report / cross_validate` — metrics and the CV loop |
| `diagstat.py` | metric primitives: tie-correct `ROC_AUC`, `F1_score` |
| `workflow_notebook.ipynb` | experiment log: EDA, cleaning audit, model selection history (narrative, in notebook order) |
| `run.ipynb` | the final single-pass pipeline used to produce statistics and plots for the report: load → fold cache → hyperparameter tune → figures → threshold → submission |
| `data/dataset/` | `x_train.csv`, `y_train.csv`, `x_test.csv` (321 feature columns each) |
| `data/codebook15_full.txt` | BRFSS 2015 codebook — the ground truth for every sentinel value and variable meaning |

### Quickstart

Open `run.ipynb` in Jupyter and **Run All** (the notebook is linear and self-contained)

---

## Data preparation (`preprocessing.py`)

The dataset has 321 columns: 230 raw survey questions and 91 BRFSS
*calculated* variables. The pipeline:

### 1. Variable selection (321 → 53 columns)

| round | what | why |
|---|---|---|
| keep the 91 calculated variables | all analysis starts here | they are the codebook's cleaned, derived versions of the survey |
| drop 42 (`DROP_LIST`) | response-quality flags (`_MIS*`, `_FRUITEX`, ...), exact duplicates and rescales (`HTIN4`, `MAXVO2_`, `PA1MIN_`, the 6 fruit/vegetable components, ...), derived thresholds/merges (`_RFSMOK3`, `_RFBMI5`, `_BMI5CAT`, `_ASTHMS1`'s parents, the PA-recoding ladder, ...) | each is a function of other kept variables (verified row-level, not just from the codebook); keeping both adds collinearity, not information |
| drop 14 more (`FURTHER_DROP`) | near-duplicates of kept columns (`_MINAC11` vs `PAMIN11_`, `_PA150R2` vs `_PACAT1`, ...) and features with zero measured importance (`_HCVU651`, `_FLSHOT6`, `_AIDTST3`, ...) | permutation importance ≈ 0 + pairwise correlation 0.85–0.96 |
| add back 18 raw variables (`RAW_VARS`) | `CVDSTRK3, DIABETE3, DIFFWALK, GENHLTH, PHYSHLTH, MENTHLTH, SEX, CHCCOPD1, CHCKIDNY, USEEQUIP, BLIND, QLACTLM2, DECIDE, HAVARTH3, CHCSCNCR, CHCOCNCR, EMPLOY1, MARITAL` | the calculated block is a lossy compression; these raw comorbidity / disability / demographic questions carry signal absent from the calculated set (measured positive-rate lifts: `CVDSTRK3` 4.2×, `CHCKIDNY` 3.3×, `CHCCOPD1` 3.0×, ...). Verified none encodes the target definition (no leakage). |

Result: 53 columns, which are then grouped into a four-way taxonomy:

| group | variables | count |
|---|---|---|
| numeric | continuous measurements (`_AGE80`, `HTM4`, `WTKG3`, `_BMI5`, drinking/diet/activity minutes, `PHYSHLTH`, `MENTHLTH`, ...) | 16 |
| ordinal | codes whose order is meaningful (`_CHLDCNT`, `_EDUCAG`, `_INCOMG`, `ACTIN11_`, `_PACAT1`, `_SMOKER3`, `GENHLTH`) | 7 |
| nominal | unordered categories (`_CHOLCHK`, `_ASTHMS1`, `_RACE`, `_LMTACT1/2/3`, `DIABETE3`, `EMPLOY1`, `MARITAL`) | 9 |
| binary | yes/no questions (`_RF*` family, `CVDSTRK3`, `SEX`, the chronic-condition questions, ...) | 21 |

### 2. Missing-value handling (per group)

BRFSS encodes missingness numerically (7 = don't know, 9 = refused, 77/99,
and per-column sentinels such as `_DRNKWEK = 99900`, `DROCDY3_ = 900`,
`FC60_ = 999`; `PHYSHLTH/MENTHLTH = 88` means "zero days", not missing).
Each group gets its own rule:

| group | rule |
|---|---|
| numeric | per-column sentinel values → NaN → **median** fill (fitted on training rows only) |
| ordinal | code 9 → NaN → **mode** fill; exception `ACTIN11_`: NaN → 0 if `_PACAT1 == 4` (verifiably inactive), else mode |
| nominal | all unknowns → the code-9 category, **kept as its own one-hot level** (no imputation — assumption-free) |
| binary | 7/9 → NaN → **mode** fill (statistics stored pre-recode), then recode {1,2} → {0,1} |

**Split-fitted imputation:** every fill statistic is computed on the training
subset only (`fill_nan_train(..., fit_idx=...)`) and applied verbatim to
validation/test rows — the same leakage discipline as the standardizer.

### 3. NaN indicators (missingness as a feature)

For every non-nominal column missing on ≥5% of training rows, a 0/1
`<name>_isnan` column marks imputed rows (20 such columns). Missingness in
this survey is informative (no cholesterol test ⇒ younger/healthier; no
activity minutes ⇒ inactive), and the indicators let the model separate
imputed rows from genuinely-average ones.

### 4. Quantile clipping (winsorising)

Numeric columns are clipped at the (0.5%, 99.5%) percentiles (fitted on
training rows, applied frozen). Self-reported absurdities (e.g.
`PAMIN11_` = 53,460 minutes/week) would otherwise dominate the
standardizer and create gradient spikes.

### 5. Test-side processing

`run_test(data)` rebuilds the *identical* pipeline for `x_test` using only
training statistics (medians, modes, clip bounds, flag set, one-hot level
set). Nothing is ever refitted on test data.

---

## Feature generation

- **One-hot encoding** of the 9 nominal groups (full level set from
  training; test aligned to training's columns, absent levels padded with
  zeros). A level-9 ("refused/unknown") dummy is a real category.
- **NaN indicator columns** (see above) — 20 additional 0/1 features.
- **Standardization** (mean/sd) is fitted inside `LogisticClassifier.fit`
  on the fit rows only and reused for all predictions.
- Final matrix: **114 features** (94 one-hot/numeric + 20 indicators).

---

## Cross-validation (`diagnostics.py`, `run.ipynb`)

Protocol (5-fold, deterministic seed):

1. **Fold matrices** — for each fold, the *entire* preprocessing pipeline is
   refitted with `fit_idx = train-fold indices`, so imputation/clipping
   statistics never see the validation rows. Since these matrices do not
   depend on model hyperparameters, they are built **once** and reused by
   every configuration in the grid.
2. **Per fold and config**: a fresh classifier is fitted on the fold's
   train part; the decision threshold `t*` is tuned **on the train part**
   (the threshold is a fitted statistic); metrics (F1, precision, recall,
   AUC) are computed on the held-out part at that threshold.
3. **Selection**: mean validation F1 across folds (the graded metric; at
   8.83% prevalence accuracy is uninformative — the all-negative baseline
   scores 91.2%). AUC is the threshold-free sanity column.
4. **Pooled out-of-fold (OOF) scores**: the five folds' held-out predictions
   are scattered back to their original row positions, giving one honest
   score per respondent (each scored by a model that never trained on it).
   All report figures (ROC, F1-vs-threshold) and the final deployed
   threshold `t*` are computed on these pooled OOF scores — no refits.
5. **Final model**: refit on all training rows with the selected
   hyperparameters; predict `x_test` at the OOF-tuned threshold;
   `create_csv_submission` writes ±1 predictions.

### Model

Regularised logistic regression trained by gradient descent
(`implementations.py`).
Three variants are selected by hyperparameters:

| variant | parameters |
|---|---|
| standard (unregularised) | `lam=0, w_pos=1` |
| L2-regularised | `lam>0, w_pos=1` |
| class-weighted | `w_pos>1` (positive-class gradient weight; scores are then deliberately uncalibrated — the threshold is re-tuned) |

