# CS-433 Project 1: Heart attack prediction

Predict whether a person has had a heart attack or coronary heart disease (MICHD) from lifestyle and health survey answers. Data: BRFSS 2015 (CDC), 321 features, 328,135 training samples, 8.8% positives. Metric: F1 score.

Course: Machine Learning (CS-433), EPFL. Constraint: NumPy only for the models.

## Structure

```
data/raw/        original CSVs, never modified (not committed)
data/clean/      cleaned arrays saved by make_clean.py (not committed)
reports/         exploration outputs (not committed)
explore_data.py  explore the raw data
explore_clean.py explore the cleaned data
make_clean.py    explore before, clean, explore after, save
preprocess.py    stateless cleaning from the codebook rules
test_preprocess.py  tests for the cleaning rules
PREPROCESSING.md details, worked example and results of the cleaning
```

## Run

```
python test_preprocess.py   # unit tests of the cleaning rules
python make_clean.py        # clean data/raw/ and save to data/clean/
```

Put the original CSVs in `data/raw/` first. The data is not in git.

## Status

- Data exploration: done
- Stateless cleaning (codes, units, skip logic): done, see `PREPROCESSING.md`
- Learned preprocessing (NaN threshold, clipping, imputation, scaling): next
- Models and validation: not started

## Team

Alfin, Chema, Hamza. Each of us works on a branch named after us and merges into `main`.