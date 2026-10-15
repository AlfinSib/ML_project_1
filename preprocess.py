"""Stateless cleaning of the BRFSS 2015 data, rules taken from the codebook.

Nothing here learns from the data (no means, medians, thresholds or lists
decided from the values), so it is safe to apply to train, validation and
test at once. Raw arrays are never modified: every function works on a copy.
Do not run it twice on the same data (a guard raises an error).

Learned steps (imputation, scaling, dropping by NaN rate, quantile clipping)
come later in a separate fit/transform pair, computed on the training fold only.

Usage: python preprocess.py [path/to/data/raw/]   (runs sanity checks)
"""
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Rules from the codebook
# ---------------------------------------------------------------------------

# Yes/No style columns: 7 = don't know, 9 = refused -> NaN.
NAN_7_9 = [
    "GENHLTH", "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4",
    "BPMEDS", "BLOODCHO", "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3",
    "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY",
    "DIABETE3", "RENTHOM1", "VETERAN3", "QLACTLM2", "USEEQUIP", "BLIND",
    "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2",
    "USENOW3", "EXERANY2", "FLUSHOT6", "PNEUVAC3", "HIVTST6",
    "PDIABTST", "PREDIAB1", "INSULIN", "DIABEYE", "DIABEDU", "INTERNET",
    "CVDASPRN", "DRADVISE", "ARTHSOCL", "CAREGIV1", "EYEEXAM",
]

# More yes/no style columns (7 and 9 -> NaN). This list was printed once from
# the training data (columns whose values are only 1, 2, 7, 9) and is now
# frozen, so train, validation and test are always treated the same way.
YES_NO_COLUMNS = [
    "ASTHNOW", "SEX", "PREGNANT", "STOPSMK2", "LMTJOIN3", "ARTHDIS2",
    "CRGVPERS", "CRGVHOUS", "CRGVEXPT", "VIGLUMA2", "VIMACDG2", "CIMEMLOS",
    "CDDISCUS", "WTCHSALT", "ASATTACK", "HAREHAB1", "STREHAB1", "RLIVPAIN",
    "RDUCHART", "RDUCSTRK", "ARTHWGT", "ARTHEXER", "ARTHEDU", "SHINGLE2",
    "HADMAM", "HADPAP2", "HPVTEST", "HADHYST2", "PROFEXAM", "BLDSTOOL",
    "HADSIGM3", "HADSGCO1", "PCPSAAD2", "PCPSADI1", "PCPSARE1", "PSATEST1",
    "RCSGENDR", "CASTHDX2", "CASTHNO2", "MISTMNT", "ADANXEV", "_CHISPNC",
    "_RFHLTH", "_HCVU651", "_RFHYPE5", "_RFCHOL", "_LTASTH1", "_CASTHM1",
    "_DRDXAR1", "_HISPANC", "_RACEG21", "_RFBMI5", "_RFSMOK3", "DRNKANY5",
    "_RFBING5", "_RFDRHV5", "_FRTLT1", "_VEGLT1", "_TOTINDA", "_PAINDX1",
    "_PA30021", "_PASTRNG", "_PASTAE1", "_RFSEAT2", "_RFSEAT3", "_FLSHOT6",
    "_PNEUMO2", "_AIDTST3",
]

# Calculated (_) columns with more than 2 categories where 9 = unknown.
NAN_9_CALC = [
    "_CHOLCHK", "_ASTHMS1", "_RACE", "_RACEGR3", "_EDUCAG", "_INCOMG",
    "_SMOKER3", "_CHLDCNT", "_PACAT1", "_PA150R2", "_PA300R2", "_PAREC1",
    "_LMTACT1", "_LMTWRK1", "_LMTSCL1",
]

# Columns with their own missing codes (7 or 9 can be real answers there).
NAN_CODES = {
    "INCOME2": [77, 99],
    "EDUCA": [9],
    "MARITAL": [9],
    "EMPLOY1": [9],
    "SEATBELT": [7, 8, 9],  # 8 = never drive or ride, not a level of use
    "_AGEG5YR": [14],
    "_AGE65YR": [3],
    "_PRACE1": [77, 99],
    "_MRACE1": [77, 99],
    "_CRACE1": [77, 99],
    "DIABAGE2": [98, 99],
    "LASTSMK2": [77, 99],
    "JOINPAIN": [77, 99],
    "IMFVPLAC": [77, 99],
    "WHRTST10": [77, 99],
    "_DRNKWEK": [99900],
}
NAN_CODES.update({n: [9] for n in NAN_9_CALC})

# Inferred from the value ranges, not confirmed in the codebook.
# The share of rows turned into NaN is printed so you can judge it.
UNVERIFIED_CODES = {"MAXVO2_": [999], "FC60_": [999], "DROCDY3_": [900]}

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

# Ordinal scales where 8 = "never" sits outside the scale. Map it to the next step.
ORDINAL_NEVER = {"CHECKUP1": 5, "EYEEXAM": 5}

# Yes/no answers that are only asked when a parent question applies.
# They are recoded to 1 = yes, 0 = no so that "question skipped" can be 0 too.
BINARY_CHILDREN = ["BPMEDS", "TOLDHI2", "ASTHNOW", "LMTJOIN3", "ARTHDIS2", "PREGNANT"]

# Fixed plausibility limits (low, high) after unit conversion. Outside -> NaN.
# These are physiological rules, not learned values.
PLAUSIBLE = {"WEIGHT2": (25, 300), "HEIGHT3": (90, 250)}  # kg, cm
PLAUSIBLE.update({n: (0, 300) for n in FREQ_FRUITVEG})  # times per 30 days
# Calculated servings per day, built from the same raw answers (300 per 30 days = 10 per day).
PLAUSIBLE.update({n: (0, 10) for n in
                  ["FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_"]})

# Columns dropped without learning anything: survey admin and design columns,
# raw columns covered by calculated ones, dates, and two sensitive items with
# almost no correlation with the target.
ADMIN_DROP = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE",
    "LANDLINE", "NUMHHOL2", "NUMPHON2", "CPDEMO1", "QSTVER", "QSTLANG",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_DUALUSE",
    "HHADULT", "_CLLCPWT", "_DUALCOR",
    "EXRACT11", "EXRACT21", "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2",
    "FLSHTMY2", "HIVTSTD3", "SXORIENT", "TRNSGNDR",
]

# Exact duplicates (identical statistics in column_summary.csv).
DUPLICATE_DROP = ["_CPRACE"]

# Same information as WTKG3, HTM4 and _BMI5. Dropped only with drop_redundant=True.
REDUNDANT_DROP = ["WEIGHT2", "HEIGHT3", "BMI_calc"]

# Calculated columns for the drop_calculated ablation. Age and state stay.
CALCULATED_OTHER = {"HTIN4", "HTM4", "WTKG3", "DRNKANY5"}
KEEP_ALWAYS = {"_AGE80", "_STATE"}

# Special codes that should not survive in a column that is mostly filled.
SUSPECT_CODES = [77, 88, 99, 777, 888, 999, 7777, 9999, 99900]


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


def ordinal_flag(col):
    """BPHIGH4, DIABETE3: 1 = yes, 4 = borderline, 2 and 3 = no (2 = only
    during pregnancy). Result: 0 = no, 0.5 = borderline, 1 = yes, NaN = unknown."""
    out = np.full_like(col, np.nan)
    out[col == 1] = 1.0
    out[col == 4] = 0.5
    out[np.isin(col, [2, 3])] = 0.0
    return out


# ---------------------------------------------------------------------------
# Skip logic: questions that were not asked because a parent question
# does not apply (codebook skip notes)
# ---------------------------------------------------------------------------

def not_applicable_rules(x, idx):
    """List of (child columns, boolean mask of rows where the parent says the
    question does not apply). Unknown parents (NaN) are never counted as
    "does not apply", so the child stays NaN there.

    Why fill with 0: when the parent column stays in a linear model, the value
    of the skipped child is a constant for that group, so any constant gives
    the same fit. 0 is the natural neutral choice."""
    def col(n):
        return x[:, idx[n]]

    rules = []
    if "BPHIGH4" in idx:  # BPMEDS asked only if told high blood pressure (1)
        rules.append((["BPMEDS"], np.isin(col("BPHIGH4"), [2, 3, 4])))
    if "BLOODCHO" in idx:  # asked only if cholesterol was ever checked (1)
        rules.append((["CHOLCHK", "TOLDHI2"], col("BLOODCHO") == 2))
    if "ALCDAY5" in idx:  # asked only if there was a drink in the past 30 days
        rules.append((["AVEDRNK2", "DRNK3GE5", "MAXDRNKS"], col("ALCDAY5") == 0))
    if "PHYSHLTH" in idx and "MENTHLTH" in idx:  # skipped if both are "None"
        rules.append((["POORHLTH"], (col("PHYSHLTH") == 0) & (col("MENTHLTH") == 0)))
    if "ASTHMA3" in idx:
        rules.append((["ASTHNOW"], col("ASTHMA3") == 2))
    if "HAVARTH3" in idx:
        rules.append((["LMTJOIN3", "ARTHDIS2", "ARTHSOCL", "JOINPAIN"], col("HAVARTH3") == 2))
    if "SEX" in idx and "_AGE80" in idx:  # asked only of women aged 44 or less
        rules.append((["PREGNANT"], (col("SEX") == 1) | (col("_AGE80") > 44)))
    return rules


# ---------------------------------------------------------------------------
# Main stateless pipeline
# ---------------------------------------------------------------------------

def recode_special_codes(x, names, copy=True, check_raw=True):
    """Apply all codebook rules. Returns (array, same names, report)."""
    x = np.array(x, copy=True) if copy else x
    if x.dtype.kind != "f":
        x = x.astype(float)
    idx = {n: j for j, n in enumerate(names)}

    if check_raw and "WEIGHT2" in idx and np.nanmax(x[:, idx["WEIGHT2"]]) < 1000:
        raise ValueError("WEIGHT2 looks already cleaned (raw values reach 9999). "
                         "Pass the raw array, or use check_raw=False for tiny test arrays.")

    handled, not_found, unverified, filled = set(), set(), {}, {}

    def col_of(name):
        if name not in idx:
            not_found.add(name)
            return None
        handled.add(name)
        return idx[name]

    for n in NAN_7_9 + YES_NO_COLUMNS:
        j = col_of(n)
        if j is not None:
            x[np.isin(x[:, j], [7, 9]), j] = np.nan

    for n, codes in NAN_CODES.items():
        j = col_of(n)
        if j is not None:
            x[np.isin(x[:, j], codes), j] = np.nan

    for n, codes in UNVERIFIED_CODES.items():
        j = col_of(n)
        if j is not None:
            hit = np.isin(x[:, j], codes)
            unverified[n] = round(float(hit.mean()), 4)
            x[hit, j] = np.nan

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

    for n, nxt in ORDINAL_NEVER.items():
        j = col_of(n)
        if j is not None:
            x[x[:, j] == 8, j] = nxt

    for n, (lo, hi) in PLAUSIBLE.items():
        if n in idx:
            c = x[:, idx[n]]
            x[(c < lo) | (c > hi), idx[n]] = np.nan

    # Skip logic. Masks are computed before the parents are recoded below.
    rules = not_applicable_rules(x, idx)
    for n in BINARY_CHILDREN:
        j = col_of(n)
        if j is not None:
            c = x[:, j]
            x[:, j] = np.where(c == 1, 1.0, np.where(c == 2, 0.0, np.nan))
    for children, mask in rules:
        for n in children:
            if n in idx:
                c = x[:, idx[n]]
                fill = mask & np.isnan(c)
                filled[n] = int(fill.sum())
                x[fill, idx[n]] = 0.0

    for n in ("BPHIGH4", "DIABETE3"):
        j = col_of(n)
        if j is not None:
            x[:, j] = ordinal_flag(x[:, j])

    report = {
        "rules_not_found": sorted(not_found),
        "unlisted_yes_no_like": find_unlisted_yes_no(x, names, handled),
        "unverified_nan_share": unverified,
        "not_applicable_filled": filled,
    }
    return x, list(names), report


def find_unlisted_yes_no(x, names, handled):
    """Diagnostic only, changes nothing. Columns that look like yes/no questions
    (values only 1, 2, 7, 9) but are in no rule list. Add them to
    YES_NO_COLUMNS if they matter."""
    out = []
    for j, n in enumerate(names):
        if n in handled:
            continue
        vals = np.unique(x[~np.isnan(x[:, j]), j])
        if vals.size and set(vals.tolist()) <= {1.0, 2.0, 7.0, 9.0} and 1.0 in vals:
            out.append(n)
    return out


def add_bmi(x, names):
    """Append BMI = kg / m^2 (needs WEIGHT2 in kg and HEIGHT3 in cm).
    Values outside [10, 80] are set to NaN (fixed rule, nothing learned).
    The data already has _BMI5, so this is off by default."""
    idx = {n: j for j, n in enumerate(names)}
    kg, cm = x[:, idx["WEIGHT2"]], x[:, idx["HEIGHT3"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        bmi = kg / (cm / 100.0) ** 2
    bmi[(bmi < 10) | (bmi > 80)] = np.nan
    return np.column_stack([x, bmi]), list(names) + ["BMI_calc"]


def drop_columns(x, names, to_drop):
    """Drop the listed columns that exist. Returns (new array, new names).
    The result is always a copy, never a view."""
    drop = set(to_drop)
    keep = [j for j, n in enumerate(names) if n not in drop]
    return x[:, keep], [names[j] for j in keep]


def is_calculated(name):
    """BRFSS calculated variable (leading or trailing underscore and a few others)."""
    if name in KEEP_ALWAYS:
        return False
    return name.startswith("_") or name.endswith("_") or name in CALCULATED_OTHER


def stateless_clean(x, names, add_bmi_col=False, drop_redundant=False,
                    drop_calculated=False, extra_drop=(), dtype=np.float64,
                    check_raw=True):
    """Full stateless step. Returns (cleaned array, names, report).

    add_bmi_col      append BMI_calc (redundant with _BMI5)
    drop_redundant   also drop WEIGHT2, HEIGHT3, BMI_calc (same info as WTKG3, HTM4, _BMI5)
    drop_calculated  ablation: drop BRFSS calculated columns (keeps _AGE80 and _STATE)
    extra_drop       any other column names to drop, for ablations
    dtype            np.float32 halves the memory
    """
    x = np.asarray(x)
    x, names = drop_columns(x, names, ADMIN_DROP + DUPLICATE_DROP)  # private copy
    x = x.astype(dtype, copy=False)
    x, names, report = recode_special_codes(x, names, copy=False, check_raw=check_raw)
    if add_bmi_col:
        x, names = add_bmi(x, names)
    late = list(extra_drop)
    if drop_redundant:
        late += REDUNDANT_DROP
    if drop_calculated:
        late += [n for n in names if is_calculated(n)]
    if late:
        x, names = drop_columns(x, names, late)
    return x, names, report


def leftover_codes(x, names, min_share=0.01, max_nan=0.5):
    """List special-looking codes still present in mostly filled columns.
    Use it to spot rules that are still missing."""
    out = []
    for j, n in enumerate(names):
        c = x[:, j]
        if np.isnan(c).mean() > max_nan:
            continue
        for v in SUSPECT_CODES:
            s = float(np.mean(c == v))
            if s >= min_share:
                out.append((n, v, round(s, 3)))
    return out


# ---------------------------------------------------------------------------
# Sanity checks (read-only on the raw data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from helpers import load_csv_data

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/clean/"
    x_train, x_test, y_train, _, _ = load_csv_data(data_dir)
    with open(data_dir.rstrip("/") + "/x_train.csv") as f:
        header = f.readline().strip().split(",")
    names = header[1:] if len(header) == x_train.shape[1] + 1 else header

    x_new, new_names, rep = stateless_clean(x_train, names)
    x_te, te_names, _ = stateless_clean(x_test, names)
    idx = {n: j for j, n in enumerate(new_names)}

    print("train shape before:", x_train.shape, "after:", x_new.shape)
    print("test  shape before:", x_test.shape, "after:", x_te.shape)
    assert new_names == te_names, "train and test columns differ after cleaning"
    print("raw array untouched:", not np.shares_memory(x_train, x_new))
    print("rules whose column was not found:", rep["rules_not_found"])
    print("yes/no-like columns in no rule list:", rep["unlisted_yes_no_like"])
    print("share of rows set to NaN by unverified rules:", rep["unverified_nan_share"])
    print("NaN share before: %.3f, after: %.3f" % (np.isnan(x_train).mean(), np.isnan(x_new).mean()))
    print("cells filled as 'question not asked' (set to 0):")
    for n, k in rep["not_applicable_filled"].items():
        print(f"   {n:10s} {k:7d}")

    def stats(n):
        c = x_new[:, idx[n]]
        return "min=%.2f max=%.2f nan=%.3f" % (np.nanmin(c), np.nanmax(c), np.isnan(c).mean())

    for n in ["PHYSHLTH", "POORHLTH", "BPMEDS", "BPHIGH4", "DIABETE3", "ALCDAY5",
              "FRUIT1", "WEIGHT2", "HEIGHT3", "SEATBELT", "_AGEG5YR", "_DRNKWEK"]:
        if n in idx:
            print(f"{n:10s}", stats(n))

    left = leftover_codes(x_new, new_names)
    print("\nspecial codes still present in mostly filled columns (review these):")
    for item in left:
        print("  ", item)
    if not left:
        print("   none")

    try:
        stateless_clean(x_new, new_names)
        raise AssertionError("double-application guard did not trigger")
    except ValueError:
        print("\ndouble-application guard works")

    assert not np.isin(x_new[:, idx["PHYSHLTH"]], [77, 88, 99]).any(), "special codes left in PHYSHLTH"
    assert not np.isinf(x_new).any(), "inf in cleaned data"
    print("checks passed")
