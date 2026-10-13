"""Stateless cleaning of the BRFSS 2015 data, rules taken from the codebook.

Nothing in this file learns from the data (no means, medians or thresholds),
so it is safe to apply to train, validation and test at once.
Raw arrays are never modified: every function works on a copy.

Learned steps (imputation, scaling, dropping by NaN rate) come later in a
separate fit/transform pair, computed on the training fold only.

Usage: python preprocess.py [path/to/data/]   (runs sanity checks)
"""
import sys

import numpy as np


# Yes/No style columns: 7 = don't know, 9 = refused -> NaN.
NAN_7_9 = [
    "GENHLTH", "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4",
    "BPMEDS", "BLOODCHO", "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3",
    "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY",
    "DIABETE3", "RENTHOM1", "VETERAN3", "QLACTLM2", "USEEQUIP", "BLIND",
    "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2",
    "USENOW3", "EXERANY2", "SEATBELT", "FLUSHOT6", "PNEUVAC3", "HIVTST6",
    "PDIABTST", "PREDIAB1", "INSULIN", "DIABEYE", "DIABEDU", "INTERNET",
    "CVDASPRN", "DRADVISE",
]

# Columns with their own missing codes (7 or 9 can be real answers there).
NAN_CODES = {
    "INCOME2": [77, 99],
    "EDUCA": [9],
    "MARITAL": [9],
    "EMPLOY1": [9],
}

# Counts where 88 means "None" (a real zero). 77, 99 (and 98) are missing.
ZERO_88 = {
    "PHYSHLTH": [77, 99],
    "MENTHLTH": [77, 99],
    "POORHLTH": [77, 99],
    "CHILDREN": [99],
    "DRNK3GE5": [77, 99],
    "DOCTDIAB": [77, 99],
    "CHKHEMO3": [77, 98, 99],
    "FEETCHK": [77, 99],
}

# Counts with no "None" code, only missing codes.
NAN_ONLY = {"AVEDRNK2": [77, 99], "MAXDRNKS": [77, 99]}

# Frequency columns: 1xx = per week, 2xx = per month, 888 = none (-> 0).
FREQ_WEEK_MONTH = ["ALCDAY5", "STRENGTH"]

# Fruit/vegetable columns: 1xx = per day, 2xx = per week, 300 = less than once
# a month, 3xx = per month, 555 = never (-> 0).
FREQ_FRUITVEG = ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"]

# Admin and survey-design columns, no information about the person.
ADMIN_DROP = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE",
    "LANDLINE", "NUMHHOL2", "NUMPHON2", "CPDEMO1", "QSTVER", "QSTLANG",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_DUALUSE",
]


# ---------------------------------------------------------------------------
# Column decoders
# ---------------------------------------------------------------------------

def decode_week_month(col):
    """ALCDAY5, STRENGTH -> number of days (or times) in 30 days."""
    out = col.copy()
    out[col == 888] = 0
    out[np.isin(col, [777, 999])] = np.nan
    m = (col >= 101) & (col <= 199)
    out[m] = (col[m] - 100) * 30 / 7
    m = (col >= 201) & (col <= 299)
    out[m] = col[m] - 200
    return out


def decode_fruitveg(col):
    """Fruit and vegetable columns -> times per 30 days."""
    out = col.copy()
    out[col == 555] = 0
    out[np.isin(col, [777, 999])] = np.nan
    m = (col >= 101) & (col <= 199)
    out[m] = (col[m] - 100) * 30
    m = (col >= 201) & (col <= 299)
    out[m] = (col[m] - 200) * 30 / 7
    out[col == 300] = 0.5
    m = (col >= 301) & (col <= 399)
    out[m] = col[m] - 300
    return out


def weight_to_kg(col):
    """WEIGHT2: 50-999 are pounds, 9000-9998 are kilograms (9 prefix)."""
    out = np.full_like(col, np.nan)
    lb = (col >= 50) & (col <= 999)
    kg = (col >= 9000) & (col <= 9998)
    out[lb] = col[lb] * 0.45359237
    out[kg] = col[kg] - 9000
    return out


def height_to_cm(col):
    """HEIGHT3: 200-711 are feet/inches (e.g. 507 = 5'07), 9000-9998 are cm."""
    out = np.full_like(col, np.nan)
    ftin = (col >= 200) & (col <= 711)
    cm = (col >= 9000) & (col <= 9998)
    out[ftin] = ((col[ftin] // 100) * 12 + col[ftin] % 100) * 2.54
    out[cm] = col[cm] - 9000
    return out


# ---------------------------------------------------------------------------
# Main stateless pipeline
# ---------------------------------------------------------------------------

def recode_special_codes(x, names):
    """Apply all codebook rules. Returns (new array, same names, report)."""
    x = np.asarray(x, dtype=float).copy()
    idx = {n: j for j, n in enumerate(names)}
    handled, not_found = set(), []

    def col_of(name):
        if name not in idx:
            not_found.append(name)
            return None
        handled.add(name)
        return idx[name]

    for n in NAN_7_9:
        j = col_of(n)
        if j is not None:
            x[np.isin(x[:, j], [7, 9]), j] = np.nan

    for n, codes in NAN_CODES.items():
        j = col_of(n)
        if j is not None:
            x[np.isin(x[:, j], codes), j] = np.nan

    for n, codes in ZERO_88.items():
        j = col_of(n)
        if j is not None:
            c = x[:, j].copy()
            x[c == 88, j] = 0
            x[np.isin(c, codes), j] = np.nan

    for n, codes in NAN_ONLY.items():
        j = col_of(n)
        if j is not None:
            x[np.isin(x[:, j], codes), j] = np.nan

    for n in FREQ_WEEK_MONTH:
        j = col_of(n)
        if j is not None:
            x[:, j] = decode_week_month(x[:, j])

    for n in FREQ_FRUITVEG:
        j = col_of(n)
        if j is not None:
            x[:, j] = decode_fruitveg(x[:, j])

    j = col_of("WEIGHT2")
    if j is not None:
        x[:, j] = weight_to_kg(x[:, j])
    j = col_of("HEIGHT3")
    if j is not None:
        x[:, j] = height_to_cm(x[:, j])

    # Fallback: untouched columns whose values are only {1, 2, 7, 9} are
    # yes/no questions with don't know / refused.
    auto = []
    for n, j in idx.items():
        if n in handled:
            continue
        vals = np.unique(x[~np.isnan(x[:, j]), j])
        if vals.size and set(vals) <= {1, 2, 7, 9} and 1 in vals:
            x[np.isin(x[:, j], [7, 9]), j] = np.nan
            auto.append(n)

    report = {"rules_not_found": not_found, "auto_yes_no": auto}
    return x, list(names), report


def add_bmi(x, names):
    """Append BMI = kg / m^2 (needs WEIGHT2 in kg and HEIGHT3 in cm).
    Values outside [10, 80] are set to NaN (fixed rule, nothing learned)."""
    idx = {n: j for j, n in enumerate(names)}
    kg, cm = x[:, idx["WEIGHT2"]], x[:, idx["HEIGHT3"]]
    bmi = kg / (cm / 100.0) ** 2
    bmi[(bmi < 10) | (bmi > 80)] = np.nan
    return np.column_stack([x, bmi]), list(names) + ["BMI_calc"]


def drop_columns(x, names, to_drop):
    """Drop the listed columns that exist. Returns (new array, new names)."""
    keep = [j for j, n in enumerate(names) if n not in set(to_drop)]
    return x[:, keep], [names[j] for j in keep]


def stateless_clean(x, names):
    """Full stateless step: recode, BMI, drop admin columns."""
    x, names, report = recode_special_codes(x, names)
    x, names = add_bmi(x, names)
    x, names = drop_columns(x, names, ADMIN_DROP)
    return x, names, report


# ---------------------------------------------------------------------------
# Sanity checks (read-only on the raw data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from helpers import load_csv_data

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/clean"
    x_train, x_test, y_train, _, _ = load_csv_data(data_dir)
    with open(data_dir.rstrip("/") + "/x_train.csv") as f:
        header = f.readline().strip().split(",")
    names = header[1:] if len(header) == x_train.shape[1] + 1 else header

    x_new, new_names, rep = stateless_clean(x_train, names)
    idx = {n: j for j, n in enumerate(new_names)}

    print("shape before:", x_train.shape, "after:", x_new.shape)
    print("raw array untouched:", not np.shares_memory(x_train, x_new))
    print("rules whose column was not found:", rep["rules_not_found"])
    print("auto-detected yes/no columns:", len(rep["auto_yes_no"]))
    print("NaN share before: %.3f, after: %.3f" % (np.isnan(x_train).mean(), np.isnan(x_new).mean()))

    def stats(n):
        c = x_new[:, idx[n]]
        return "min=%.1f max=%.1f nan=%.3f" % (np.nanmin(c), np.nanmax(c), np.isnan(c).mean())

    for n in ["PHYSHLTH", "ALCDAY5", "FRUIT1", "WEIGHT2", "HEIGHT3", "BMI_calc"]:
        if n in idx:
            print(f"{n:10s}", stats(n))
    if "PHYSHLTH" in idx:
        c = x_new[:, idx["PHYSHLTH"]]
        assert not np.isin(c, [77, 88, 99]).any(), "special codes left in PHYSHLTH"
    print("checks passed")
