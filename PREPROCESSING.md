# Data exploration and preprocessing

Status of the data work for Project 1 (BRFSS 2015, MICHD prediction).

## Files

| File | Role |
|---|---|
| `explore_data.py` | Read-only exploration. Writes `reports/column_summary.csv` and `reports/overview.png` |
| `preprocess.py` | Stateless cleaning from codebook rules (`stateless_clean`) |
| `explore_clean.py` | Same exploration on the cleaned data (no saving) |
| `make_clean.py` | Explore before, clean, explore after, save to `data/clean/` and `reports/` |
| `test_preprocess.py` | Small tests for every rule type. Run `python test_preprocess.py` |
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

## The problem the stateless step solves

BRFSS stores answers as codes, and some codes are not real values. Fed to a model as they are, "refused to answer" (99) is read as 99 days of illness. Many NaNs also mean "this question was not asked", which is different from "unknown".

`preprocess.py` translates those codes with rules from the codebook. Nothing is learned from the data, so it is safe to apply to train, validation and test at once. Raw arrays are never modified, every function returns a copy. Running it twice on the same data raises an error, because a second pass would corrupt the values (for example `WEIGHT2` would be converted from "pounds" again).

Usage:

```python
from preprocess import stateless_clean
x_clean, names, report = stateless_clean(x_train, names)
x_test_clean, _, _ = stateless_clean(x_test, names)
```

Switches for ablations: `drop_calculated=True`, `drop_redundant=True`, `add_bmi_col=True`, `extra_drop=[...]`, `dtype=np.float32` (halves the memory).

Check: `python preprocess.py data/raw/`

## Worked example

Three made-up people. The "after" values are the real output of `stateless_clean`.

- A: man, 55
- B: woman, 30
- C: woman, 60

| Rule | Column | A before | B before | C before | A after | B after | C after |
|---|---|---|---|---|---|---|---|
| 7/9 to NaN | `GENHLTH` | 2 | 9 | 3 | 2 | NaN | 3 |
| 7/9 to NaN (frozen yes/no list) | `_RFHLTH` | 1 | 9 | 2 | 1 | NaN | 2 |
| 88 = "None" to 0, 77/99 to NaN | `PHYSHLTH` | 88 | 5 | 99 | 0 | 5 | NaN |
| same | `MENTHLTH` | 88 | 30 | 77 | 0 | 30 | NaN |
| Own missing codes | `INCOME2` | 8 | 77 | 99 | 8 | NaN | NaN |
| same | `_AGEG5YR` (14 = unknown) | 9 | 14 | 12 | 9 | NaN | 12 |
| same | `SEATBELT` (8 = never drive) | 1 | 8 | 3 | 1 | NaN | 3 |
| "Never" (8) to the next step | `CHECKUP1` | 1 | 8 | 4 | 1 | 5 | 4 |
| Ordinal flag: 0 no, 0.5 borderline, 1 yes | `BPHIGH4` | 3 | 1 | 4 | 0 | 1 | 0.5 |
| same | `DIABETE3` | 3 | 1 | 4 | 0 | 1 | 0.5 |
| Question not asked, fill 0 | `BPMEDS` | NaN | 2 | NaN | 0 | 0 | 0 |
| same | `POORHLTH` | NaN | 4 | NaN | 0 | 4 | NaN |
| same | `AVEDRNK2` | NaN | 77 | 2 | 0 | NaN | 2 |
| same | `PREGNANT` | NaN | 9 | NaN | 0 | NaN | 0 |
| Unit decoding, days per 30 | `ALCDAY5` (1xx per week, 2xx per month, 888 none) | 888 | 215 | 105 | 0 | 15 | 21.43 |
| Unit decoding and plausibility limit | `FRUIT1` (1xx per day, 555 never) | 101 | 199 | 555 | 30 | NaN | 0 |
| Pounds or kg (9 prefix) to kg | `WEIGHT2` | 180 | 9075 | 150 | 81.65 | 75 | 68.04 |
| Feet/inches or cm (9 prefix) to cm | `HEIGHT3` | 507 | 9170 | 600 | 170.18 | 170 | 182.88 |
| Unverified code (999 to NaN) | `MAXVO2_` | 22.84 | 999 | 25.8 | 22.84 | NaN | 25.8 |
| Dropped | `IDATE`, `_CPRACE` | | | | gone | gone | gone |

How to read the trickier rows:

- **`BPMEDS`.** The question is only asked when `BPHIGH4` is yes. A had no hypertension (not asked), B has it and takes no medication, C is borderline (not asked). All three become 0, and the model still sees the difference through `BPHIGH4` (0, 1, 0.5). Keep the parent column in the model.
- **`POORHLTH`.** The question is skipped when both `PHYSHLTH` and `MENTHLTH` are "None". A gets 0. C stays NaN, because `PHYSHLTH` was refused, so we cannot tell whether the question applied.
- **`AVEDRNK2`.** A did not drink (888), so 0 drinks. B drinks but refused the count, so NaN.
- **`PREGNANT`.** Only asked of women aged 44 or less. A (man) and C (age 60) get 0. B refused, so NaN.
- **`BPHIGH4` and `DIABETE3`.** The raw codes (1 yes, 3 no, 4 borderline) are not in a sensible order, so they are recoded to 0, 0.5 and 1. "Yes, but only during pregnancy" counts as 0.

## All columns by rule

| Rule | Columns |
|---|---|
| 7/9 to NaN (explicit list) | `GENHLTH`, `HLTHPLN1`, `PERSDOC2`, `MEDCOST`, `CHECKUP1`, `BPHIGH4`, `BPMEDS`, `BLOODCHO`, `CHOLCHK`, `TOLDHI2`, `CVDSTRK3`, `ASTHMA3`, `CHCSCNCR`, `CHCOCNCR`, `CHCCOPD1`, `HAVARTH3`, `ADDEPEV2`, `CHCKIDNY`, `DIABETE3`, `RENTHOM1`, `VETERAN3`, `QLACTLM2`, `USEEQUIP`, `BLIND`, `DECIDE`, `DIFFWALK`, `DIFFDRES`, `DIFFALON`, `SMOKE100`, `SMOKDAY2`, `USENOW3`, `EXERANY2`, `FLUSHOT6`, `PNEUVAC3`, `HIVTST6`, `PDIABTST`, `PREDIAB1`, `INSULIN`, `DIABEYE`, `DIABEDU`, `INTERNET`, `CVDASPRN`, `DRADVISE`, `ARTHSOCL`, `CAREGIV1`, `EYEEXAM` |
| 7/9 to NaN (frozen yes/no list, 68 columns) | `ASTHNOW`, `SEX`, `PREGNANT`, `STOPSMK2`, `LMTJOIN3`, `ARTHDIS2`, `CRGVPERS`, `CRGVHOUS`, `CRGVEXPT`, `VIGLUMA2`, `VIMACDG2`, `CIMEMLOS`, `CDDISCUS`, `WTCHSALT`, `ASATTACK`, `HAREHAB1`, `STREHAB1`, `RLIVPAIN`, `RDUCHART`, `RDUCSTRK`, `ARTHWGT`, `ARTHEXER`, `ARTHEDU`, `SHINGLE2`, `HADMAM`, `HADPAP2`, `HPVTEST`, `HADHYST2`, `PROFEXAM`, `BLDSTOOL`, `HADSIGM3`, `HADSGCO1`, `PCPSAAD2`, `PCPSADI1`, `PCPSARE1`, `PSATEST1`, `RCSGENDR`, `CASTHDX2`, `CASTHNO2`, `MISTMNT`, `ADANXEV`, `_CHISPNC`, `_RFHLTH`, `_HCVU651`, `_RFHYPE5`, `_RFCHOL`, `_LTASTH1`, `_CASTHM1`, `_DRDXAR1`, `_HISPANC`, `_RACEG21`, `_RFBMI5`, `_RFSMOK3`, `DRNKANY5`, `_RFBING5`, `_RFDRHV5`, `_FRTLT1`, `_VEGLT1`, `_TOTINDA`, `_PAINDX1`, `_PA30021`, `_PASTRNG`, `_PASTAE1`, `_RFSEAT2`, `_RFSEAT3`, `_FLSHOT6`, `_PNEUMO2`, `_AIDTST3`. Printed once from the training data, then frozen so train and test are treated the same way |
| 9 to NaN (calculated, more than 2 categories) | `_CHOLCHK`, `_ASTHMS1`, `_RACE`, `_RACEGR3`, `_EDUCAG`, `_INCOMG`, `_SMOKER3`, `_CHLDCNT`, `_PACAT1`, `_PA150R2`, `_PA300R2`, `_PAREC1`, `_LMTACT1`, `_LMTWRK1`, `_LMTSCL1` |
| Own missing codes to NaN | `INCOME2`, `_PRACE1`, `_MRACE1`, `_CRACE1`, `LASTSMK2`, `JOINPAIN`, `IMFVPLAC`, `WHRTST10`, `AVEDRNK2`, `MAXDRNKS` (77, 99), `EDUCA`, `MARITAL`, `EMPLOY1` (9), `SEATBELT` (7, 8, 9), `_AGEG5YR` (14), `_AGE65YR` (3), `DIABAGE2` (98, 99), `_DRNKWEK` (99900) |
| Unverified codes to NaN | `MAXVO2_`, `FC60_` (999), `DROCDY3_` (900). Inferred from value ranges, the share of rows affected is printed by the check |
| 88 to 0, 77/99 to NaN | `PHYSHLTH`, `MENTHLTH`, `POORHLTH`, `CHILDREN`, `DRNK3GE5`, `DOCTDIAB`, `CHKHEMO3`, `FEETCHK` |
| Days per week or month, 888 to 0 | `ALCDAY5`, `STRENGTH` |
| Times per day, week or month, 555 to 0 | `FRUITJU1`, `FRUIT1`, `FVBEANS`, `FVGREEN`, `FVORANG`, `VEGETAB1` |
| Unit conversion | `WEIGHT2` to kg, `HEIGHT3` to cm |
| "Never" (8) to the next step (5) | `CHECKUP1`, `EYEEXAM` |
| Ordinal flag (0 no, 0.5 borderline, 1 yes) | `BPHIGH4`, `DIABETE3` |
| Yes/no recoded to 1/0 | `BPMEDS`, `TOLDHI2`, `ASTHNOW`, `LMTJOIN3`, `ARTHDIS2`, `PREGNANT` |
| Plausibility limits, outside becomes NaN | `WEIGHT2` 25 to 300 kg, `HEIGHT3` 90 to 250 cm, the six raw fruit and vegetable columns 0 to 300 per 30 days, `FTJUDA1_`, `FRUTDA1_`, `BEANDAY_`, `GRENDAY_`, `ORNGDAY_`, `VEGEDA1_` 0 to 10 per day |
| Dropped: survey admin | `FMONTH`, `IDATE`, `IMONTH`, `IDAY`, `IYEAR`, `DISPCODE`, `SEQNO`, `_PSU`, `CTELENUM`, `PVTRESD1`, `COLGHOUS`, `STATERES`, `CELLFON3`, `LADULT`, `CTELNUM1`, `CELLFON2`, `CADULT`, `PVTRESD2`, `CCLGHOUS`, `CSTATE`, `LANDLINE`, `NUMHHOL2`, `NUMPHON2`, `CPDEMO1`, `QSTVER`, `QSTLANG`, `HHADULT` |
| Dropped: sampling weights | `_STSTR`, `_STRWT`, `_RAWRAKE`, `_WT2RAKE`, `_LLCPWT`, `_DUALUSE`, `_CLLCPWT`, `_DUALCOR` |
| Dropped: raw exercise (calculated `PA*` columns cover them) | `EXRACT11`, `EXRACT21`, `EXEROFT1`, `EXEROFT2`, `EXERHMM1`, `EXERHMM2` |
| Dropped: dates | `FLSHTMY2`, `HIVTSTD3` |
| Dropped: sensitive, almost no signal | `SXORIENT`, `TRNSGNDR` |
| Dropped: exact duplicate | `_CPRACE` (same statistics as `_CRACE1`) |

### Questions that were not asked (fill with 0)

The codebook skips these questions when a parent question does not apply. The skipped cells are filled with 0. When the parent stays in a linear model, the skipped group has a constant child value, so any constant gives the same fit and 0 is the natural choice. If the parent itself is unknown (NaN), the child stays NaN.

| Child columns | Not asked when |
|---|---|
| `BPMEDS` | `BPHIGH4` is no, pregnancy only, or borderline |
| `CHOLCHK`, `TOLDHI2` | `BLOODCHO` is no |
| `AVEDRNK2`, `DRNK3GE5`, `MAXDRNKS` | `ALCDAY5` is 0 (no drinks in 30 days) |
| `POORHLTH` | `PHYSHLTH` and `MENTHLTH` are both 0 |
| `ASTHNOW` | `ASTHMA3` is no |
| `LMTJOIN3`, `ARTHDIS2`, `ARTHSOCL`, `JOINPAIN` | `HAVARTH3` is no |
| `PREGNANT` | `SEX` is male or `_AGE80` is above 44 |

## Results on the real data

Output of `python make_clean.py` (train 328,135 rows, test 109,379 rows).

| | Before | After |
|---|---|---|
| Columns | 321 | 275 (46 dropped) |
| Constant or all-NaN columns | 6 | 0 |
| Overall NaN share | 0.448 | 0.449 |
| Columns with NaN rate below 0.01 | 100 | 45 |
| Columns with NaN rate 0.01 to 0.20 | 43 | 92 |
| Columns with NaN rate 0.20 to 0.50 | 31 | 19 |
| Columns with NaN rate 0.50 to 0.90 | 48 | 24 |
| Columns with NaN rate above 0.90 | 99 | 95 |

Checks: `rules not found` is empty, `yes/no-like columns in no rule list` is empty, train and test have identical columns. Share of rows set to NaN by the unverified codes: `MAXVO2_` 1.2%, `FC60_` 1.2%, `DROCDY3_` 4.9%. The only leftover special code is `_AGE80` = 77, which is a real age.

The overall NaN share barely moves because two effects cancel. Special codes (7, 9, 77, 99) became NaN, and skip-logic cells were filled with 0. Cells filled as "question not asked": `BPMEDS` 195,416, `AVEDRNK2`, `DRNK3GE5` and `MAXDRNKS` 154,556 each, `POORHLTH` 159,860, `ASTHNOW` 283,169, `LMTJOIN3`, `ARTHDIS2`, `ARTHSOCL` and `JOINPAIN` 216,696 each, `PREGNANT` 278,980, `CHOLCHK` and `TOLDHI2` 36,861 each.

Top correlations with the target (rows with NaN skipped, share of rows filled in brackets):

| Before | | After | |
|---|---|---|---|
| `GENHLTH` | +0.244 | `CVDASPRN` (0.04) | -0.257 |
| `CVDASPRN` (0.04) | -0.228 | `GENHLTH` | +0.249 |
| `_AGE80` | +0.225 | `_RFHLTH` | +0.235 |
| `_AGEG5YR` | +0.219 | `BPMEDS` | +0.232 |
| `EMPLOY1` | +0.207 | `_AGEG5YR` | +0.227 |
| `BPHIGH4` | -0.205 | `_AGE80` | +0.225 |
| `_RFHLTH` | +0.172 | `FC60_` and `MAXVO2_` | -0.224 |
| `DIABETE3` | -0.161 | `BPHIGH4` | +0.217 |

What this shows:

- `BPHIGH4` flipped from -0.205 to +0.217 because 1 now means yes.
- `BPMEDS` jumped into the top 4. It was mostly NaN before, and the skip-logic fill made it usable.
- `CVDASPRN` (4% filled) and `DRADVISE` (9% filled) stay strong. Their signal comes partly from who was asked, the same effect as `HAREHAB1`. They go in the same ablation.
- `FC60_` and `MAXVO2_` are calculated fitness estimates, probably derived from age, sex and BMI. They are covered by the `drop_calculated=True` ablation.
- `_DRNKWEK` still reaches 53200 (two implied decimals, so 532 drinks a week). The quantile clipping on the train fold handles it.
- 119 columns remain above 50% NaN. A plain threshold would drop `CVDASPRN` and `DRADVISE`, so the next step adds a `was_asked` flag before any drop.

## Decisions

- **`HAREHAB1` is kept on purpose.** It is only asked of people who had a heart attack, which is part of the target definition, so its missingness reveals part of the label. This can inflate validation and leaderboard scores. Planned ablation: train with and without it and report both F1 values.
- **Calculated (`_`) columns are kept.** They include age, which has no raw column. Dropping them (`drop_calculated=True`, keeps `_AGE80` and `_STATE`) is a second ablation.
- **Near-duplicates are kept by default.** `WEIGHT2`, `HEIGHT3` and `_BMI5`, `WTKG3`, `HTM4` carry the same information. `drop_redundant=True` removes the first two.
- **No data-dependent rule.** An earlier version detected yes/no columns from the values it received, which could treat train and test differently. That list is now frozen.
- **Raw files are never edited.** Every function returns a copy.

## Check

Run `python test_preprocess.py` (12 tests), then `python preprocess.py data/raw/` and look at:

1. The shape should drop from 321 to 275 columns (46 dropped) for both train and test, with identical column names.
2. `rules whose column was not found` should be empty.
3. `yes/no-like columns in no rule list` should be empty. A name there is a column to add to `YES_NO_COLUMNS`.
4. `share of rows set to NaN by unverified rules` shows how many rows the guessed codes affect. Remove an entry from `UNVERIFIED_CODES` if the share looks too large.
5. The list of special codes still present in mostly filled columns. Each line is a rule that may be missing.
6. `double-application guard works` and `checks passed` at the end.

## Not done yet (needs the training fold only)

1. Drop columns above a NaN threshold (test 0.5 and 0.9 by ablation, since `CVDASPRN` is 96% NaN yet strongly related to the target).
2. Clip outliers to train quantiles (`_FRUTSUM`, `_VEGESUM`, `PAFREQ1_`, `STRFREQ_` still have extreme values).
3. Impute the remaining NaNs: median for numeric columns, plus a missing flag where missingness carries signal.
4. One-hot encode categorical columns, standardize, add a bias column.
5. Convert labels from {-1, 1} to {0, 1}.
6. Review `STREHAB1`, `RDUCHART`, `RDUCSTRK` for the same skip-logic effect as `HAREHAB1`.

## Limits

Rules come from the part of the codebook read so far (about pages 1 to 74 of 137) and from the value ranges in `reports/column_summary.csv`. The codes for `_AGEG5YR` (14), `_AGE65YR` (3) and `_PRACE1` (77, 99) follow the usual BRFSS convention and should be confirmed in the codebook.