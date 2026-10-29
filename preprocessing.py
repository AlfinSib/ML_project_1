"""Data preprocessing pipeline for the BRFSS-2015 heart disease project.

Reproduces exactly the cleaning steps developed in workflow_notebook.ipynb
(cells 3-34), organised as importable functions, plus the two v2 upgrades:
NaN indicators and quantile clipping (adopted from teammate H's pipeline).

    import preprocessing as pp
    data = pp.run_train("data/dataset")        # train matrix + metadata
    xt   = pp.run_test("data/dataset", data)   # test matrix, aligned to train

Design notes
------------
* One function per pipeline stage; each returns what the next stage needs.
* All fill statistics (medians, modes) are computed on TRAIN only and returned
  in `stats`; the test pipeline consumes them verbatim (never recomputed).
* Binary fill modes are stored PRE-recode (in {1, 2}), so the test pipeline
  needs no recode inversion.
* Stage asserts catch the known silent-failure classes (name typos -> empty
  np.isin matches; header/matrix misalignment).

v2 additions (teammate H's design, adapted):
* NaN indicators: for every column with >= FLAG_NAN_TSHARE missing rows BEFORE
  the fill, append a 0/1 `<name>_isnan` column (1 = value was imputed). The
  model can then learn a separate effect for imputed rows. Indicators are
  computed on the FULL matrix (row-wise property, no fitted statistic), but
  the >= threshold DECISION is made on the fit subset only when fit_idx is
  iven.
* Quantile clipping: numeric columns are winsorized at CLIP_Q percentiles
  (fitted on the fit subset), capping self-reported absurdities
  (PAMIN11_ = 53460 min/week) before they poison the standardizer / gradient.
  Only columns with enough distinct values are clipped (min_unique_clip),
  so ordinal codes are never touched.
"""

import os
import numpy as np
import helpers


# ============================================================================
# raw-data cache: the CSV read (~40s) dominates run_train; cache it per session
# ============================================================================

_RAW_CACHE = {}


def load_raw(data_path="data/dataset", cache=True):
    """Load (x_train, x_test, y, train_ids, test_ids, full_headers) with caching.

    The CSV parse is ~40s (96% of run_train's cost). With cache=True, the
    first load reads the files; every later call returns the SAME arrays
    (zero-copy) keyed on the files' identity (path + size + mtime), so a
    changed dataset is re-read automatically but re-runs in the same kernel
    are instant. cache=False forces a fresh read.

    IMPORTANT: the cached arrays are shared. All downstream stages copy
    (select_variables copies via np.delete/np.concatenate; fills copy via
    .copy()), so mutating a downstream matrix is safe - but do NOT assign
    into the arrays returned by load_raw itself.
    """
    paths = {n: os.path.join(data_path, n) for n in
             ("x_train.csv", "x_test.csv", "y_train.csv")}
    key = tuple((p, os.path.getsize(p), os.path.getmtime(p))
                for p in paths.values())
    if cache and key in _RAW_CACHE:
        return _RAW_CACHE[key]

    x_train, x_test, y, train_ids, test_ids = helpers.load_csv_data(data_path)
    full_headers = helpers.read_header(paths["x_train.csv"])
    out = (x_train, x_test, y, train_ids, test_ids, full_headers)
    if cache:
        _RAW_CACHE[key] = out
    return out


# ============================================================================
# constants: the single source of truth (copied from the notebook)
# ============================================================================

# --- round 1: calculated-variable audit (91 -> 49) ---
DROP_LIST = [
    # response quality
    "_MISVEGN", "_MISFRTN", "_FRTRESP", "_VEGRESP",
    "_FRUITEX", "_VEGETEX", "_FRT16", "_VEG23", "PAMISS1_",
    # age/race duplicates (keep _AGE80, _RACE)
    "_AGEG5YR", "_AGE_G", "_AGE65YR",
    "_PRACE1", "_MRACE1", "_HISPANC", "_RACEG21", "_RACEGR3", "_RACE_G1",
    # exact duplicates / rescales
    "HTIN4", "MAXVO2_",
    # overlapping info (exact sums / thresholds / merges)
    "PA1MIN_", "PA1VIGM_",
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "_RFSMOK3", "_RFBMI5", "_BMI5CAT",
    "_FRTLT1", "_VEGLT1", "DRNKANY5", "_PASTRNG",
    "_PAREC1", "_PAINDX1", "_PA30021", "_PASTAE1",
    "_DRDXAR1", "_LTASTH1", "_CASTHM1",
]

# --- round 2: permutation-importance / correlation pruning (49 -> 35) ---
FURTHER_DROP = [
    "_MINAC21", "_MINAC11", "_PA150R2", "_PA300R2", "METVL11_", "METVL21_",
    "PADUR2_", "PAFREQ2_", "PAMIN21_", "PAVIG21_", "ACTIN21_",
    "_HCVU651", "_FLSHOT6", "_AIDTST3",
]

# --- round 3: raw variables added back (35 + 18 = 53) ---
RAW_VARS = [
    # first round: comorbidity + general health
    "CVDSTRK3", "DIABETE3", "DIFFWALK", "GENHLTH", "PHYSHLTH", "MENTHLTH", "SEX",
    # second round: chronic conditions / disability / demographics
    "CHCCOPD1", "CHCKIDNY", "USEEQUIP", "BLIND", "QLACTLM2", "DECIDE",
    "HAVARTH3", "CHCSCNCR", "CHCOCNCR", "EMPLOY1", "MARITAL",
]
RAW_BINARIES = [                       # {1,2} + DK7/9; NOT incl. the nominals
    "CVDSTRK3", "DIFFWALK", "SEX",
    "CHCCOPD1", "CHCKIDNY", "USEEQUIP", "BLIND", "QLACTLM2", "DECIDE",
    "HAVARTH3", "CHCSCNCR", "CHCOCNCR",
]
RAW_NOMINALS = ["DIABETE3", "EMPLOY1", "MARITAL"]
RAW_NUMERIC = ["PHYSHLTH", "MENTHLTH"]          # 88 = "none" -> true 0
RAW_ORDINAL = ["GENHLTH"]                        # 7/9 -> mode

# Nominals whose level 7 is a REAL category (not DK): keep it!
# EMPLOY1: 7 = retired. MARITAL has no 7. All other nominals: 7 = DK -> 9.
NOMINALS_KEEP_7 = ("EMPLOY1", "MARITAL")

# numeric sentinels that mean "missing" (calculated variables only)
ENCODED_NAN = {
    "DROCDY3_": 900.0, "_DRNKWEK": 99900.0, "FC60_": 999.0,
    "STRFREQ_": 99.0, "PAFREQ1_": 99.0, "PAFREQ2_": 99.0,
}

# --- v2: NaN indicator + quantile clipping configuration ------------------
# Indicators are added for any column whose missing share (BEFORE fills,
# including sentinel-decoded NaN) on the fit subset is at least FLAG_NAN_SHARE.
# Nominals are excluded: their missingness already becomes the 9-level dummy.
FLAG_NAN_SHARE = 0.05
# Numerics (and only numerics) are clipped at these percentiles, fitted on
# the fit subset after sentinel decoding, applied to all rows.
CLIP_Q = (0.5, 99.5)
# Columns with fewer distinct observed values than this are not clipped
# (protects low-cardinality ordinals/counts from being crushed).
MIN_UNIQUE_CLIP = 10

# full variable taxonomy of the 91 calculated variables (pre-drop; cell 11)
HEADERS_NUMERIC_CAT_ALL = [
    "_AGEG5YR", "_AGE_G", "_BMI5CAT", "_CHLDCNT", "_EDUCAG", "_INCOMG",
    "_MISVEGN", "ACTIN11_", "ACTIN21_", "_PACAT1", "_PA150R2", "_PA300R2",
    "_SMOKER3",
]
HEADERS_CAT_ALL = [
    "_CHOLCHK", "_ASTHMS1", "_PRACE1", "_MRACE1", "_RACE", "_RACEGR3",
    "_RACE_G1", "_PAREC1", "_LMTACT1", "_LMTWRK1", "_LMTSCL1",
]
HEADERS_NUMERIC_ALL = [
    "_AGE80", "HTIN4", "HTM4", "WTKG3", "_BMI5", "DROCDY3_", "_DRNKWEK",
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "_FRUTSUM", "_VEGESUM", "METVL11_", "METVL21_", "MAXVO2_", "FC60_",
    "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "_MINAC11", "_MINAC21",
    "STRFREQ_", "PAMIN11_", "PAMIN21_", "PA1MIN_", "PAVIG11_", "PAVIG21_",
    "PA1VIGM_",
]

# The full 91 calculated-variable block in order, used to derive the implicit
# binary group exactly like the notebook does ("kept vars in no taxonomy list").
# Derived from the dataset header at call time (NOT hardcoded): the header is
# the ground truth, and hand-typed lists have already caused phantom-name bugs.
def calc_headers_after_drops(data_path="data/dataset"):
    """The 35 calculated variable names kept after both drop rounds."""
    all_headers = helpers.read_header(data_path + "/x_train.csv")
    calc = all_headers[-91:]
    drop_all = set(DROP_LIST) | set(FURTHER_DROP)
    return [h for h in calc if h not in drop_all]


# ============================================================================
# small utilities
# ============================================================================

def _col(x, headers, name):
    """Column index of `name` in the header array; errors loudly if absent."""
    hits = np.where(headers == name)[0]
    if len(hits) != 1:
        raise KeyError(f"header {name!r} not found (or found twice) in matrix")
    return int(hits[0])


def _mode(values):
    """Mode of a 1-D array (no scipy)."""
    v, c = np.unique(values, return_counts=True)
    return v[np.argmax(c)]


# ============================================================================
# stage 1: select calculated variables, drop rounds 1+2, append raw variables
# ============================================================================

def select_variables(x, full_headers, verbose=True):
    """91 calculated -> drop DROP_LIST -> drop FURTHER_DROP -> append RAW_VARS.

    Works identically on x_train and x_test (same header layout).
    Returns (x_sel, headers) where headers covers all kept columns.
    """
    calc = x[:, -91:]
    calc_headers = full_headers[-91:]

    idx = np.where(np.isin(calc_headers, DROP_LIST))[0]
    assert len(idx) == len(DROP_LIST), \
        f"DROP_LIST matched only {len(idx)}/{len(DROP_LIST)} names - typo?"
    calc = np.delete(calc, idx, axis=1)
    calc_headers = np.delete(calc_headers, idx)

    idx = np.where(np.isin(calc_headers, FURTHER_DROP))[0]
    assert len(idx) == len(FURTHER_DROP), \
        f"FURTHER_DROP matched only {len(idx)}/{len(FURTHER_DROP)} names - typo?"
    calc = np.delete(calc, idx, axis=1)
    calc_headers = np.delete(calc_headers, idx)

    raw_idx = [int(np.where(full_headers == h)[0][0]) for h in RAW_VARS]
    assert len(raw_idx) == len(RAW_VARS), "typo in RAW_VARS?"
    x_sel = np.concatenate((calc, x[:, raw_idx]), axis=1)
    headers = np.concatenate((calc_headers, np.array(RAW_VARS)))

    if verbose:
        print(f"selected {x_sel.shape[1]} columns "
              f"({len(calc_headers)} calculated + {len(RAW_VARS)} raw)")
    return x_sel, headers


def make_groups(data_path="data/dataset"):
    """The four taxonomy lists AFTER all drops and additions.

    Returns dict with keys 'num', 'num_cat', 'cat', 'binary' (binary = implicit
    group: kept calculated vars in none of the three lists + RAW_BINARIES).
    """
    drop_all = set(DROP_LIST) | set(FURTHER_DROP)
    num_cat = [h for h in HEADERS_NUMERIC_CAT_ALL if h not in drop_all]
    cat = [h for h in HEADERS_CAT_ALL if h not in drop_all]
    num = [h for h in HEADERS_NUMERIC_ALL if h not in drop_all]

    num += RAW_NUMERIC
    num_cat += RAW_ORDINAL
    cat += RAW_NOMINALS

    # The calculated binaries are the IMPLICIT group: kept calculated variables
    # that are in none of the three taxonomy lists. They must be derived from
    # the KEPT-HEADERS list (as the notebook does), not from the taxonomy lists
    # - variables like _RFHLTH appear in no list by design.
    kept_calc = calc_headers_after_drops(data_path)
    explicit = set(num_cat) | set(cat) | set(num)
    calculated_binaries = [h for h in kept_calc if h not in explicit]
    binary = calculated_binaries + RAW_BINARIES
    return {"num": num, "num_cat": num_cat, "cat": cat, "binary": binary}


def calc_headers_after_drops(data_path="data/dataset"):
    """The 35 calculated variable names kept after both drop rounds."""
    all_headers = helpers.read_header(data_path + "/x_train.csv")
    calc = all_headers[-91:]
    drop_all = set(DROP_LIST) | set(FURTHER_DROP)
    return [h for h in calc if h not in drop_all]


# ============================================================================
# stage 2: NaN handling (fills computed on the given matrix; train use)
# ============================================================================

def fill_nan_train(x, headers, groups, verbose=True, fit_idx=None):
    """Apply the per-group NaN rules + v2 additions to a TRAINING matrix.

    v2 additions:
    * NaN indicators (<name>_isnan 0/1 columns) for columns missing on at
      least FLAG_NAN_SHARE of the fit subset, except nominals.
    * Quantile clipping of numeric columns at CLIP_Q percentiles fitted on
      the fit subset.

    Split-fitted imputation: pass fit_idx (e.g. a CV fold's train-row indices)
    to compute every fill statistic (medians / modes / clip bounds / the
    >= FLAG_NAN_SHARE decision) on those rows ONLY, while the fills are still
    applied to the whole matrix. fit_idx=None fits on all rows (use for the
    final full-train model; use a fold's train part for leakage-free CV).

    Returns (x, stats) where x has the indicator columns APPENDED at the end
    and headers_new returned alongside in stats['flag_headers'] - use
    append_flags() to (re)attach headers. Use fit_nan_test() for the test side.
    """
    x = x.astype(float).copy()
    stats = {"median": {}, "mode_num_cat": {}, "mode_binary": {},
             "flags": [], "clip": {}}
    fit = np.arange(len(x)) if fit_idx is None else np.asarray(fit_idx)

    # ---- record missingness masks BEFORE any fills (indicators + clip fitting)
    # sentinels must be decoded first so the masks match the real missingness
    pre_masks = {}
    for h in groups["num"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h in ENCODED_NAN:
            col[col == ENCODED_NAN[h]] = np.nan
        if h in RAW_NUMERIC:
            col[col == 88.0] = 0.0
            col[np.isin(col, [77.0, 99.0])] = np.nan
        pre_masks[h] = np.isnan(col)
    for h in groups["num_cat"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h != "ACTIN11_":
            if h == "GENHLTH":
                col[np.isin(col, [7.0, 9.0])] = np.nan
            else:
                col[col == 9.0] = np.nan
        pre_masks[h] = np.isnan(col)
    for h in groups["binary"]:
        j = _col(x, headers, h)
        col = x[:, j]
        col[np.isin(col, [7.0, 9.0])] = np.nan
        pre_masks[h] = np.isnan(col)
    # nominals: missingness becomes the 9 level - no indicator needed,
    # but decode AFTER the mask snapshot is irrelevant for them (NaN->9 later).

    # ---- NaN indicators: columns missing on >= FLAG_NAN_SHARE of fit rows ----
    nominal_set = set(groups["cat"])
    for h, mask in pre_masks.items():
        if h in nominal_set:
            continue
        share = float(mask[fit].mean())
        if share >= FLAG_NAN_SHARE:
            stats["flags"].append(h)
            if verbose:
                print(f"  flag  {h}_isnan   ({share:.1%} missing on fit rows)")

    # ---- fills: numerics (medians) + clipping --------------------------------
    for h in groups["num"]:
        j = _col(x, headers, h)
        col = x[:, j]                      # sentinels already decoded above
        med = float(np.nanmedian(col[fit]))  # statistic from the fit subset
        n_nan = int(np.isnan(col).sum())
        # quantile clip bounds fitted on the fit subset, AFTER decoding,
        # BEFORE filling (so the fill value itself is never an outlier)
        obs = col[fit][~np.isnan(col[fit])]
        if np.unique(obs).size >= MIN_UNIQUE_CLIP:
            stats["clip"][h] = tuple(float(v) for v in np.percentile(obs, CLIP_Q))
            col[~np.isnan(col)] = np.clip(col[~np.isnan(col)], *stats["clip"][h])
        col[np.isnan(col)] = med             # fill applied to ALL rows
        stats["median"][h] = med
        if verbose:
            clip_note = f", clipped {stats['clip'][h]}" if h in stats["clip"] else ""
            print(f"  num   {h:12s}: {n_nan} NaNs -> median {med:.3f}{clip_note}")

    # ---- ordinals: ACTIN11_ first (needs RAW _PACAT1), then 9 -> mode -----
    # NB: _PACAT1's OWN fill happens inside this loop, so when fit_idx is set,
    # the ACTIN branch may read _PACAT1 rows that belong to the held-out part.
    # This is restricted to the ==4 test (a RAW observed value, never a fill),
    # so no fill statistic leaks: held-out rows only contribute their true codes.
    pacat_raw = x[:, _col(x, headers, "_PACAT1")].copy()   # BEFORE its fill
    for h in groups["num_cat"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h == "ACTIN11_":
            m = np.isnan(col)
            mode = float(_mode(col[fit][~np.isnan(col[fit])]))
            col[m & (pacat_raw == 4)] = 0.0     # verifiably inactive -> 0
            col[m & (pacat_raw != 4)] = mode    # unknown -> mode
            stats["mode_num_cat"][h] = mode
            if verbose:
                print(f"  ord   ACTIN11_   : {int((m & (pacat_raw == 4)).sum())} "
                      f"-> 0, {int((m & (pacat_raw != 4)).sum())} -> mode {mode}")
        else:
            if h == "GENHLTH":                  # raw ordinal also carries 7
                col[np.isin(col, [7.0, 9.0])] = np.nan
            else:
                col[col == 9.0] = np.nan
            mode = float(_mode(col[fit][~np.isnan(col[fit])]))
            col[np.isnan(col)] = mode
            stats["mode_num_cat"][h] = mode
            if verbose:
                print(f"  ord   {h:12s}: -> mode {mode}")

    # ---- nominals: NaN -> 9; DK-7 -> 9 unless 7 is a real level ----------
    for h in groups["cat"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h not in NOMINALS_KEEP_7:
            col[col == 7.0] = 9.0               # DK joins the refused code
        col[np.isnan(col)] = 9.0                # 9 = own one-hot level

    # ---- binaries: 7/9 -> NaN -> mode (fit subset; PRE-recode) -> recode ----
    for h in groups["binary"]:
        j = _col(x, headers, h)
        col = x[:, j]                        # sentinels already decoded above
        mode = float(_mode(col[fit][~np.isnan(col[fit])]))  # in {1, 2}
        col[np.isnan(col)] = mode
        stats["mode_binary"][h] = mode                 # pre-recode mode!
        col[col == 1.0] = 0.0                          # 1=Yes -> 0
        col[col == 2.0] = 1.0                          # 2=No  -> 1

    # ---- append NaN indicator columns ---------------------------------------
    if stats["flags"]:
        flag_cols = [pre_masks[h].astype(float).reshape(-1, 1)
                     for h in stats["flags"]]
        x = np.concatenate([x] + flag_cols, axis=1)
        if verbose:
            print(f"  added {len(stats['flags'])} NaN indicator columns")

    assert np.isnan(x).sum() == 0, "NaN survived the training fills"
    flag_headers = [f"{h}_isnan" for h in stats["flags"]]
    stats["flag_headers"] = flag_headers
    stats["headers_full"] = list(headers) + flag_headers
    return x, stats


def fill_nan_test(x, headers, groups, stats, verbose=False):
    """Apply the same rules to the TEST matrix, using TRAIN's fill constants,
    TRAIN's clip bounds, and TEST's own missingness (for the indicators).

    Produces column-for-column the same coding as fill_nan_train, including
    the same indicator columns in the same order.
    """
    x = x.astype(float).copy()

    # snapshot missingness (test's own masks, same sentinel decoding order)
    pre_masks = {}
    for h in groups["num"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h in ENCODED_NAN:
            col[col == ENCODED_NAN[h]] = np.nan
        if h in RAW_NUMERIC:
            col[col == 88.0] = 0.0
            col[np.isin(col, [77.0, 99.0])] = np.nan
        pre_masks[h] = np.isnan(col)
    for h in groups["num_cat"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h != "ACTIN11_":
            if h == "GENHLTH":
                col[np.isin(col, [7.0, 9.0])] = np.nan
            else:
                col[col == 9.0] = np.nan
        pre_masks[h] = np.isnan(col)
    for h in groups["binary"]:
        j = _col(x, headers, h)
        col = x[:, j]
        col[np.isin(col, [7.0, 9.0])] = np.nan
        pre_masks[h] = np.isnan(col)

    # numerics: clip with TRAIN's bounds, fill with TRAIN's medians
    for h in groups["num"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h in stats["clip"]:
            col[~np.isnan(col)] = np.clip(col[~np.isnan(col)], *stats["clip"][h])
        col[np.isnan(col)] = stats["median"][h]

    # ordinals: ACTIN11_ first against RAW _PACAT1 (same ordering as train!)
    pacat_raw = x[:, _col(x, headers, "_PACAT1")].copy()
    for h in groups["num_cat"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h == "ACTIN11_":
            m = np.isnan(col)
            col[m & (pacat_raw == 4)] = 0.0
            col[m & (pacat_raw != 4)] = stats["mode_num_cat"][h]
        else:
            if h == "GENHLTH":
                col[np.isin(col, [7.0, 9.0])] = np.nan
            else:
                col[col == 9.0] = np.nan
            col[np.isnan(col)] = stats["mode_num_cat"][h]

    # nominals
    for h in groups["cat"]:
        j = _col(x, headers, h)
        col = x[:, j]
        if h not in NOMINALS_KEEP_7:
            col[col == 7.0] = 9.0
        col[np.isnan(col)] = 9.0

    # binaries (train's PRE-recode mode is in {1,2}: fill, then recode)
    for h in groups["binary"]:
        j = _col(x, headers, h)
        col = x[:, j]                        # sentinels already decoded above
        col[np.isnan(col)] = stats["mode_binary"][h]
        col[col == 1.0] = 0.0
        col[col == 2.0] = 1.0

    # append the SAME indicator columns as train (same names, same order),
    # computed from TEST's own missingness (a row-wise property, no fitting)
    if stats["flags"]:
        flag_cols = [pre_masks[h].astype(float).reshape(-1, 1)
                     for h in stats["flags"]]
        x = np.concatenate([x] + flag_cols, axis=1)

    assert np.isnan(x).sum() == 0, "NaN survived the test fills"
    return x


# ============================================================================
# stage 3: one-hot encoding (train levels define the column set)
# ============================================================================

def encode(x, headers, groups, train_headers_1h=None, verbose=False):
    """One-hot the nominal group. For TEST, pass train's one-hot headers to
    realign (missing levels padded with zeros, extras dropped).
    NOTE: the NaN indicator columns (already 0/1) pass through untouched -
    they are appended BEFORE one-hot, so one_hot's np.isin(col_names) never
    matches them (their names end in _isnan, not in groups['cat'])."""
    x_1h, headers_1h = helpers.one_hot(x, groups["cat"], headers)
    if train_headers_1h is not None:
        if list(headers_1h) != list(train_headers_1h):
            colmap = {h: j for j, h in enumerate(headers_1h)}
            x_1h = np.column_stack([
                x_1h[:, colmap[h]] if h in colmap else np.zeros(len(x_1h))
                for h in train_headers_1h
            ])
            headers_1h = np.array(train_headers_1h)
    return x_1h, headers_1h


# ============================================================================
# top-level convenience (what teammates normally call)
# ============================================================================

def run_train(data_path="data/dataset", verbose=True, fit_idx=None):
    """Load + full train pipeline. Returns a dict:
       x           cleaned, NaN-free, NOT one-hot, NOT standardized (N x 53)
       headers     matching column names
       groups      taxonomy dict (num / num_cat / cat / binary)
       stats       fill constants (fitted on fit_idx if given; reuse for test!)
       x_1h, headers_1h   one-hot version (N x ~86) for modelling
       y           labels in {0, 1}
       plus ids / x_test / full_headers for later submission use.

    Split-fitted imputation: pass fit_idx (train-fold row indices) so the fill
    statistics are leakage-free for cross-validation. None = fit on all rows,
    which is what the final full-train model wants.
    """
    x_train, x_test, y, train_ids, test_ids, full_headers = load_raw(data_path)
    y01 = y.copy()
    y01[y01 == -1] = 0

    x_sel, headers = select_variables(x_train, full_headers, verbose)
    groups = make_groups(data_path)
    x_filled, stats = fill_nan_train(x_sel, headers, groups, verbose,
                                     fit_idx=fit_idx)
    # fill_nan_train appends the NaN indicator columns; headers must follow
    headers_full = np.array(stats["headers_full"])
    x_1h, headers_1h = encode(x_filled, headers_full, groups)

    if verbose:
        print(f"one-hot width: {x_1h.shape[1]} "
              f"(+{len(stats['flags'])} NaN flags already inside)")
    return {
        "x": x_filled, "headers": headers_full, "groups": groups, "stats": stats,
        "x_1h": x_1h, "headers_1h": headers_1h,
        "y": y01, "x_test_raw": x_test, "full_headers": full_headers,
        "train_ids": train_ids, "test_ids": test_ids,
    }


def run_test(data, verbose=False):
    """Test matrix through the SAME pipeline (train's fills + levels + clips).
    `data` = the dict returned by run_train(). Returns (x_test_1h, y-like)."""
    x_test = data["x_test_raw"]
    full_headers = data["full_headers"]
    x_sel, headers = select_variables(x_test, full_headers, verbose)
    x_filled = fill_nan_test(x_sel, headers, data["groups"], data["stats"],
                             verbose)
    # indicators already appended inside fill_nan_test; rebuild full headers
    headers_full = np.array(list(headers) + data["stats"]["flag_headers"])
    x_1h, _ = encode(x_filled, headers_full, data["groups"],
                     train_headers_1h=data["headers_1h"])
    assert x_1h.shape[1] == data["x_1h"].shape[1], "train/test width mismatch"
    return x_1h


if __name__ == "__main__":
    data = run_train()
    xt = run_test(data)
    print(f"\ntrain: {data['x_1h'].shape}, test: {xt.shape}")
    print("groups:", {k: len(v) for k, v in data["groups"].items()})
