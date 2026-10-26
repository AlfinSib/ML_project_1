"""

Step 1: keep only the columns listed below (calculated columns + 7 raw ones).
Step 2: turn special codes (7, 9, 77, 99, ...) into NaN and recode a few columns.

Usage:
    x_clean, names, report = stateless_clean(x_raw, header_names)
"""
import numpy as np

# 1. Columns to keep
# BRFSS calculated columns (the ones ending or starting with "_").
# Age, race, BMI, activity and drinking each appear once, duplicates are gone.
KEEP_CALCULATED = [
    "_RFHLTH", "_RFHYPE5", "_RFCHOL", "_CHOLCHK", "_ASTHMS1", "_RACE",
    "_AGE80", "HTM4", "WTKG3", "_BMI5", "_CHLDCNT", "_EDUCAG", "_INCOMG",
    "_SMOKER3", "DROCDY3_", "_RFBING5", "_DRNKWEK", "_RFDRHV5", "_FRUTSUM",
    "_VEGESUM", "_TOTINDA", "FC60_", "ACTIN11_", "PADUR1_", "PAFREQ1_",
    "STRFREQ_", "PAMIN11_", "PAVIG11_", "_PACAT1", "_LMTACT1", "_LMTWRK1",
    "_LMTSCL1", "_RFSEAT2", "_RFSEAT3", "_PNEUMO2",
]

# Raw columns added back because they predict heart disease well.
KEEP_RAW = ["GENHLTH", "DIABETE3", "CVDSTRK3", "DIFFWALK", "PHYSHLTH",
            "MENTHLTH", "SEX"]

KEEP = KEEP_CALCULATED + KEEP_RAW


# 2. Cleaning rules
# Code 9 means "unknown" in these columns.
MISSING_9 = ["_CHOLCHK", "_ASTHMS1", "_RACE", "_EDUCAG", "_INCOMG",
             "_SMOKER3", "_CHLDCNT", "_PACAT1", "_LMTACT1", "_LMTWRK1",
             "_LMTSCL1"]

# Other columns with their own missing codes.
MISSING_CODES = {
    "GENHLTH": [7, 9],
    "_DRNKWEK": [99900],
    "DROCDY3_": [900],
    "FC60_": [999],
    "STRFREQ_": [99],
    "PAFREQ1_": [99],
}

# Two-level columns: 1 -> 0, 2 -> 1, 9 -> NaN.
BINARY_1_2 = ["_RFHLTH", "_RFHYPE5", "_RFCHOL", "_RFBING5", "_RFDRHV5",
              "_TOTINDA", "_RFSEAT2", "_RFSEAT3", "_PNEUMO2", "SEX"]

# Yes/No questions: 1 = yes -> 1, 2 = no -> 0, 7 and 9 -> NaN.
YES_NO = ["CVDSTRK3", "DIFFWALK"]

# Number of days (0 to 30): 88 means none, 77 and 99 are unknown.
DAY_COUNTS = ["PHYSHLTH", "MENTHLTH"]

# Codes that should not survive in a filled column (used by leftover_codes).
SUSPECT_CODES = [7, 9, 77, 88, 99, 777, 888, 999, 9999, 99900]


def select_columns(x, names, keep):
    """Return only the wanted columns, plus the list of wanted names not found."""
    idx = {n: j for j, n in enumerate(names)}
    found = [n for n in keep if n in idx]
    missing = [n for n in keep if n not in idx]
    return x[:, [idx[n] for n in found]], found, missing


def stateless_clean(x, names):
    """Select columns then clean them. Returns (array, names, report)."""
    x, names, missing = select_columns(np.asarray(x), names, KEEP)
    x = x.astype(np.float64)  # private copy, the raw array is untouched
    idx = {n: j for j, n in enumerate(names)}

    # Guard against cleaning twice (raw PHYSHLTH reaches 99, cleaned reaches 30)
    if "PHYSHLTH" in idx and np.nanmax(x[:, idx["PHYSHLTH"]]) <= 30:
        raise ValueError("Data looks already cleaned. Pass the raw array.")

    def col(n):
        return x[:, idx[n]]

    # Special codes -> NaN
    for n in MISSING_9:
        if n in idx:
            col(n)[col(n) == 9] = np.nan
    for n, codes in MISSING_CODES.items():
        if n in idx:
            col(n)[np.isin(col(n), codes)] = np.nan

    # Binary and yes/no columns -> 0/1
    for n in BINARY_1_2:
        if n in idx:
            c = col(n).copy()
            x[:, idx[n]] = np.where(c == 1, 0.0, np.where(c == 2, 1.0, np.nan))
    for n in YES_NO:
        if n in idx:
            c = col(n).copy()
            x[:, idx[n]] = np.where(c == 1, 1.0, np.where(c == 2, 0.0, np.nan))

    # Day counts: 88 -> 0, 77 and 99 -> NaN
    for n in DAY_COUNTS:
        if n in idx:
            c = col(n).copy()
            c[c == 88] = 0
            c[np.isin(c, [77, 99])] = np.nan
            x[:, idx[n]] = c

    # Diabetes: 1 yes -> 1, 4 borderline -> 0.5, 2 or 3 no -> 0, else NaN
    if "DIABETE3" in idx:
        c = col("DIABETE3").copy()
        out = np.full_like(c, np.nan)
        out[c == 1] = 1.0
        out[c == 4] = 0.5
        out[np.isin(c, [2, 3])] = 0.0
        x[:, idx["DIABETE3"]] = out

    report = {"rules_not_found": missing, "unverified_nan_share": {}}
    return x, names, report


def leftover_codes(x, names, min_share=0.01, max_nan=0.5):
    """List special-looking codes still present in mostly filled columns."""
    out = []
    for j, n in enumerate(names):
        c = x[:, j]
        if np.isnan(c).mean() > max_nan:
            continue
        for v in SUSPECT_CODES:
            share = float(np.mean(c == v))
            if share >= min_share:
                out.append((n, v, round(share, 3)))
    return out