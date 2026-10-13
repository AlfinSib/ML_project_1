# Data exploration and preprocessing

Status of the data work for Project 1 (BRFSS 2015, MICHD prediction).

## Files

| File | Role |
|---|---|
| `explore_data.py` | Read-only exploration. Writes `reports/column_summary.csv` and `reports/overview.png` |
| `preprocess.py` | Stateless cleaning from codebook rules (`stateless_clean`) |
| `data/raw/` | Original CSVs, never modified |
| `data/clean/` | Cache of the stateless output only |

`data/` and `reports/` are in `.gitignore`.

## What the exploration showed

- 328,135 training samples, 321 features, 8.8% positives. Accuracy is useless here, so the decision threshold is tuned on validation F1.
- 147 of 321 columns have more than 50% NaN, 99 of them above 90%. Most are survey modules asked of a small subset.
- 6 columns are constant and `LADULT` is all NaN.
- Train and test NaN rates differ by at most 0.004, so no visible shift.
- Strongest signals: `GENHLTH`, `CVDASPRN`, age columns (`_AGE80`, `_AGEG5YR`), `EMPLOY1`, `BPHIGH4`.
- `CVDINFR4`, `CVDCRHD4` and `_MICHD` are not in the dataset.

## What the stateless step does

Nothing is learned from data, so it is safe to apply to train, validation and test at once.

1. 7/9 to NaN in yes/no columns, 77/99 to NaN in count columns.
2. "None" codes to 0: 88 (`PHYSHLTH`, `MENTHLTH`, ...), 888 (`ALCDAY5`, `STRENGTH`), 555 (fruit and vegetables).
3. Mixed-unit frequencies decoded to a count per 30 days.
4. `WEIGHT2` to kg, `HEIGHT3` to cm, `BMI_calc` added.
5. Survey-admin columns dropped (dates, phone-line flags, sampling weights).

Usage:

```python
from preprocess import stateless_clean
x_clean, names, report = stateless_clean(x_train, names)
```

Check: `python preprocess.py data/raw/`

## Results of the check

- Shape: 321 columns before, 290 after.
- NaN share: 0.448 before, 0.473 after (special codes became NaN).
- `PHYSHLTH` and `ALCDAY5` now range 0 to 30, `BMI_calc` 10 to 80.
- Rules whose column was not found: none.
- 84 columns were detected by the yes/no fallback. Reviewed against the codebook, all plausible.

## Decisions

- **`HAREHAB1` is kept on purpose.** It is only asked of people who had a heart attack, which is part of the target definition, so its missingness reveals part of the label. This can inflate validation and leaderboard scores. Planned ablation: train with and without it and report both F1 values.
- **Calculated (`_`) columns are kept.** They include age, which has no raw column. Dropping them is a second ablation.
- **Raw files are never edited.** Every function returns a copy.

## Not done yet (needs the training fold only)

1. Drop columns above a NaN threshold (try 0.5 and 0.9).
2. Clip outliers to train quantiles (`FRUIT1` reaches 2970, `WEIGHT2` reaches 501 kg).
3. Impute: median for numeric, 0 plus a missing flag for skip-logic columns.
4. Standardize and add a bias column.
5. Convert labels from {-1, 1} to {0, 1}.
6. Review `STREHAB1`, `RDUCHART`, `RDUCSTRK` for the same skip-logic effect.

## Limits

Rules come from the part of the codebook read so far (about pages 1 to 74 of 137). Columns beyond that only get the yes/no fallback.
