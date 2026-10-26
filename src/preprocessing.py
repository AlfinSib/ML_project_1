from pathlib import Path
import csv

import numpy as np


# ---------------------------------------------------------------------
# Missing-value encodings used internally during preprocessing
# ---------------------------------------------------------------------

MISSING_CATEGORY = -1
NOT_APPLICABLE_CODE = -2


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

ALL_CALCULATED_FEATURES = [
    # General health
    "_RFHLTH",

    # Health care coverage
    "_HCVU651",

    # High blood pressure
    "_RFHYPE5",

    # Cholesterol
    "_CHOLCHK", "_RFCHOL",

    # Cardiovascular disease, asthma, arthritis
    "_LTASTH1", "_CASTHM1", "_ASTHMS1", "_DRDXAR1",

    # Demographics, BMI, education, income
    "_AGEG5YR", "_AGE65YR", "_AGE80", "_AGE_G",
    "HTIN4", "HTM4", "WTKG3",
    "_BMI5", "_BMI5CAT", "_RFBMI5",
    "_CHLDCNT", "_EDUCAG", "_INCOMG",

    # Tobacco
    "_SMOKER3", "_RFSMOK3",

    # Alcohol
    "DRNKANY5", "DROCDY3_", "_RFBING5",
    "_DRNKWEK", "_RFDRHV5",

    # Fruits and vegetables
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_",
    "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "_MISFRTN", "_MISVEGN",
    "_FRTRESP", "_VEGRESP",
    "_FRUTSUM", "_VEGESUM",
    "_FRTLT1", "_VEGLT1",
    "_FRT16", "_VEG23",
    "_FRUITEX", "_VEGETEX",

    # Physical activity
    "_TOTINDA",
    "METVL11_", "METVL21_",
    "MAXVO2_", "FC60_",
    "ACTIN11_", "ACTIN21_",
    "PADUR1_", "PADUR2_",
    "PAFREQ1_", "PAFREQ2_",
    "_MINAC11", "_MINAC21",
    "STRFREQ_", "PAMISS1_",
    "PAMIN11_", "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_", "PAVIG21_", "PA1VIGM_",
    "_PACAT1", "_PAINDX1",
    "_PA150R2", "_PA300R2", "_PA30021",
    "_PASTRNG", "_PAREC1", "_PASTAE1",

    # Arthritis-related limitations
    "_LMTACT1", "_LMTWRK1", "_LMTSCL1",

    # Seat belts
    "_RFSEAT2", "_RFSEAT3",

    # Vaccinations
    "_FLSHOT6", "_PNEUMO2",

    # HIV testing
    "_AIDTST3",

    # Race
    "_RACEGR3",
]


DROP_FEATURES = [
    # Response-quality variables
    "_MISVEGN",
    "_MISFRTN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRUITEX",
    "_VEGETEX",
    "_FRT16",
    "_VEG23",
    "PAMISS1_",

    # Redundant age representations
    "_AGEG5YR",
    "_AGE_G",
    "_AGE65YR",

    # Redundant / derived features
    "HTIN4",
    "MAXVO2_",
    "PA1MIN_",
    "PA1VIGM_",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_RFSMOK3",
    "_RFBMI5",
    "_BMI5CAT",
    "_FRTLT1",
    "_VEGLT1",
    "DRNKANY5",
    "_PASTRNG",
    "_PAREC1",
    "_PAINDX1",
    "_PA30021",
    "_PASTAE1",
    "_DRDXAR1",
    "_LTASTH1",
    "_CASTHM1",
]


CALCULATED_FEATURES = [
    feature
    for feature in ALL_CALCULATED_FEATURES
    if feature not in DROP_FEATURES
]


NUMERICAL_FEATURES = [
    "_AGE80",
    "HTM4",
    "WTKG3",
    "_BMI5",
    "_CHLDCNT",
    "DROCDY3_",
    "_DRNKWEK",
    "_FRUTSUM",
    "_VEGESUM",
    "METVL11_",
    "METVL21_",
    "FC60_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMIN11_",
    "PAMIN21_",
    "PAVIG11_",
    "PAVIG21_",
]


ORDINAL_FEATURES = [
    "_EDUCAG",
    "_INCOMG",
    "ACTIN11_",
    "ACTIN21_",
    "_PACAT1",
    "_PA150R2",
    "_PA300R2",
]


CATEGORICAL_FEATURES = [
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_CHOLCHK",
    "_RFCHOL",
    "_ASTHMS1",
    "_SMOKER3",
    "_RFBING5",
    "_RFDRHV5",
    "_TOTINDA",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
    "_RACEGR3",
]


STRUCTURAL_MISSING_FEATURES = [
    "_PNEUMO2",
    "_FLSHOT6",
]


# These ordinal variables use BLANK rather than code 9 for missingness.
ORDINAL_BLANK_MISSING = [
    "ACTIN11_",
    "ACTIN21_",
]


NUMERICAL_SPECIAL_MISSING = {
    "DROCDY3_": [900],
    "FC60_": [99900],
    "_DRNKWEK": [99900],
    "WTKG3": [99999],
    "_CHLDCNT": [9],
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def _header_to_index(headers):
    """Return a mapping from feature name to column index."""
    return {
        feature: idx
        for idx, feature in enumerate(headers)
    }


def select_features(x, headers, selected_features):
    """Select specified columns from a NumPy feature matrix."""
    header_to_idx = _header_to_index(headers)

    missing_features = [
        feature
        for feature in selected_features
        if feature not in header_to_idx
    ]

    if missing_features:
        raise ValueError(
            f"Unknown features: {missing_features}"
        )

    indices = np.array([
        header_to_idx[feature]
        for feature in selected_features
    ])

    return x[:, indices], headers[indices]


def _validate_feature_partition():
    """Check that the feature-type lists form an exact partition."""
    numerical = set(NUMERICAL_FEATURES)
    ordinal = set(ORDINAL_FEATURES)
    categorical = set(CATEGORICAL_FEATURES)

    if not numerical.isdisjoint(ordinal):
        raise ValueError(
            "NUMERICAL_FEATURES and ORDINAL_FEATURES overlap."
        )

    if not numerical.isdisjoint(categorical):
        raise ValueError(
            "NUMERICAL_FEATURES and CATEGORICAL_FEATURES overlap."
        )

    if not ordinal.isdisjoint(categorical):
        raise ValueError(
            "ORDINAL_FEATURES and CATEGORICAL_FEATURES overlap."
        )

    typed_features = numerical | ordinal | categorical
    selected_features = set(CALCULATED_FEATURES)

    if typed_features != selected_features:
        missing = selected_features - typed_features
        extra = typed_features - selected_features

        raise ValueError(
            "Feature-type lists do not match CALCULATED_FEATURES. "
            f"Missing from type lists: {sorted(missing)}. "
            f"Unexpected in type lists: {sorted(extra)}."
        )


# ---------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------

def encode_structural_missing(x, headers):
    """
    Encode structural BLANK values as NOT_APPLICABLE_CODE.

    This must be called before replace_missing_codes(), while original
    blank CSV cells are still represented as NaN.
    """
    x_clean = x.copy()
    header_to_idx = _header_to_index(headers)

    for feature in STRUCTURAL_MISSING_FEATURES:
        idx = header_to_idx[feature]
        mask = np.isnan(x_clean[:, idx])
        x_clean[mask, idx] = NOT_APPLICABLE_CODE

    return x_clean


def replace_missing_codes(x, headers):
    """
    Convert explicit BRFSS nonresponse/sentinel codes to np.nan.

    Rules
    -----
    - retained categorical features: 9 -> NaN
    - retained ordinal features: 9 -> NaN except ACTIN11_/ACTIN21_
    - numerical features: only explicitly configured sentinel values
    """
    x_clean = x.copy()
    header_to_idx = _header_to_index(headers)

    for feature in CATEGORICAL_FEATURES:
        idx = header_to_idx[feature]
        x_clean[x_clean[:, idx] == 9, idx] = np.nan

    for feature in ORDINAL_FEATURES:
        if feature not in ORDINAL_BLANK_MISSING:
            idx = header_to_idx[feature]
            x_clean[x_clean[:, idx] == 9, idx] = np.nan

    for feature, missing_codes in NUMERICAL_SPECIAL_MISSING.items():
        idx = header_to_idx[feature]

        for code in missing_codes:
            x_clean[x_clean[:, idx] == code, idx] = np.nan

    return x_clean


# ---------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------

def fit_numerical_imputer(x, headers, numerical_features):
    """Fit one median per numerical feature."""
    header_to_idx = _header_to_index(headers)
    medians = {}

    for feature in numerical_features:
        idx = header_to_idx[feature]
        median = np.nanmedian(x[:, idx])

        if np.isnan(median):
            raise ValueError(
                f"Cannot fit median for '{feature}': "
                "all training values are missing."
            )

        medians[feature] = median

    return medians


# Backwards-compatible name used by older notebooks.
numerical_medians = fit_numerical_imputer


def fit_ordinal_imputer(x, headers, ordinal_features):
    """Fit one mode per ordinal feature."""
    header_to_idx = _header_to_index(headers)
    modes = {}

    for feature in ordinal_features:
        idx = header_to_idx[feature]
        column = x[:, idx]

        non_missing = column[~np.isnan(column)]

        if len(non_missing) == 0:
            raise ValueError(
                f"Cannot fit mode for '{feature}': "
                "all training values are missing."
            )

        values, counts = np.unique(
            non_missing,
            return_counts=True,
        )

        modes[feature] = values[np.argmax(counts)]

    return modes


def apply_imputer(x, headers, fill_values):
    """Apply pre-fitted per-feature fill values."""
    x_out = x.copy()
    header_to_idx = _header_to_index(headers)

    for feature, fill_value in fill_values.items():
        idx = header_to_idx[feature]
        mask = np.isnan(x_out[:, idx])
        x_out[mask, idx] = fill_value

    return x_out


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def fit_standardizer(x, headers, features):
    """Fit mean and standard deviation on training data only."""
    header_to_idx = _header_to_index(headers)
    stats = {}

    for feature in features:
        idx = header_to_idx[feature]
        column = x[:, idx]

        stats[feature] = (
            np.mean(column),
            np.std(column),
        )

    return stats


# Backwards-compatible name used by older notebooks.
standardization_stats = fit_standardizer


def apply_standardization(x, headers, stats):
    """Apply pre-fitted standardization statistics."""
    x_out = x.copy()
    header_to_idx = _header_to_index(headers)

    for feature, (mean, std) in stats.items():
        idx = header_to_idx[feature]

        if std == 0:
            x_out[:, idx] = 0
        else:
            x_out[:, idx] = (
                x_out[:, idx] - mean
            ) / std

    return x_out


# ---------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------

def fit_categorical_encoder(
    x,
    headers,
    categorical_features,
):
    """
    Fit a one-hot encoder using categories observed in training only.

    Missing values are represented by MISSING_CATEGORY when present.
    One regular observed category is dropped as the reference category.

    Important:
    No artificial UNKNOWN_CATEGORY dummy is created. A category that was
    not seen during training therefore has no learned coefficient.
    """
    header_to_idx = _header_to_index(headers)
    encoder = {}

    for feature in categorical_features:
        idx = header_to_idx[feature]
        column = x[:, idx].copy()

        column[np.isnan(column)] = MISSING_CATEGORY
        categories = np.unique(column)

        regular_categories = [
            category
            for category in categories
            if category not in (
                MISSING_CATEGORY,
                NOT_APPLICABLE_CODE,
            )
        ]

        if len(regular_categories) == 0:
            raise ValueError(
                f"No regular observed category found for '{feature}'."
            )

        encoder[feature] = {
            "categories": categories,
            "reference": regular_categories[0],
        }

    return encoder


def apply_categorical_encoder(
    x,
    headers,
    categorical_features,
    encoder,
):
    """
    Apply a categorical encoder fitted on the training split.

    For each categorical feature:
    - missing values are converted to MISSING_CATEGORY;
    - one observed regular category is the reference and is dropped;
    - only categories observed during training receive dummy columns;
    - an unseen validation/test category matches no dummy column and is
      therefore represented by the all-zero/reference encoding.

    This avoids creating training columns that are guaranteed to be zero.
    """
    encoded_columns = []
    encoded_headers = []
    categorical_set = set(categorical_features)

    for idx, feature in enumerate(headers):
        column = x[:, idx].copy()

        if feature not in categorical_set:
            encoded_columns.append(
                column.reshape(-1, 1)
            )
            encoded_headers.append(feature)
            continue

        column[np.isnan(column)] = MISSING_CATEGORY

        categories = encoder[feature]["categories"]
        reference = encoder[feature]["reference"]

        for category in categories:
            if category == reference:
                continue

            dummy = (
                column == category
            ).astype(float)

            encoded_columns.append(
                dummy.reshape(-1, 1)
            )

            encoded_headers.append(
                f"{feature}_{category:g}"
            )

    return (
        np.hstack(encoded_columns),
        np.array(encoded_headers),
    )

REDUNDANT_ENCODED_COLUMNS = [
    "_LMTWRK1_3",
    "_LMTSCL1_4",
    "_RFSEAT3_-1",
    "_PNEUMO2_-2",
]


def drop_redundant_encoded_columns(x, headers):
    keep_mask = ~np.isin(
        headers,
        REDUNDANT_ENCODED_COLUMNS,
    )

    return x[:, keep_mask], headers[keep_mask]
    
def find_unseen_categories(
    x,
    headers,
    categorical_features,
    encoder,
):
    """
    Return categorical values present in x but absent from training.

    This is a diagnostic helper; it does not alter the encoding.
    """
    header_to_idx = _header_to_index(headers)
    unseen = {}

    for feature in categorical_features:
        idx = header_to_idx[feature]
        column = x[:, idx].copy()

        column[np.isnan(column)] = MISSING_CATEGORY
        known_categories = encoder[feature]["categories"]

        unseen_values = np.unique(
            column[
                ~np.isin(
                    column,
                    known_categories,
                )
            ]
        )

        if len(unseen_values) > 0:
            unseen[feature] = unseen_values

    return unseen


# ---------------------------------------------------------------------
# Train/validation split and intercept
# ---------------------------------------------------------------------

def stratified_train_val_split(
    x,
    y,
    val_ratio=0.2,
    seed=42,
):
    """Create a stratified train/validation split."""
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        n_val = int(
            len(label_indices) * val_ratio
        )

        val_indices.extend(
            label_indices[:n_val]
        )
        train_indices.extend(
            label_indices[n_val:]
        )

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        x[train_indices],
        x[val_indices],
        y[train_indices],
        y[val_indices],
    )


def add_intercept(x):
    """Prepend a column of ones."""
    return np.column_stack([
        np.ones(x.shape[0]),
        x,
    ])


# ---------------------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------------------

def prepare_data(
    data_dir,
    val_ratio=0.2,
    seed=42,
):
    """
    Load and preprocess BRFSS train/test data.

    All learned preprocessing parameters are fitted using the training
    split only and then reused unchanged on validation and test data.

    Returns
    -------
    x_train_final
    x_val_final
    x_test_final
    headers_final
    y_train_split
    y_val_split
    """
    _validate_feature_partition()

    data_dir = Path(data_dir)

    # --------------------------------------------------------------
    # 1. Read CSV header and locate retained features
    # --------------------------------------------------------------
    with open(
        data_dir / "x_train.csv",
        "r",
        newline="",
    ) as f:
        reader = csv.reader(f)
        all_headers = np.array(next(reader))

    header_to_idx = _header_to_index(
        all_headers
    )

    missing_features = [
        feature
        for feature in CALCULATED_FEATURES
        if feature not in header_to_idx
    ]

    if missing_features:
        raise ValueError(
            "Selected features missing from x_train.csv: "
            f"{missing_features}"
        )

    feature_indices = np.array([
        header_to_idx[feature]
        for feature in CALCULATED_FEATURES
    ])

    headers_calc = all_headers[
        feature_indices
    ]

    # --------------------------------------------------------------
    # 2. Load selected features and target
    # --------------------------------------------------------------
    x_train = np.genfromtxt(
        data_dir / "x_train.csv",
        delimiter=",",
        skip_header=1,
        usecols=feature_indices,
        dtype=np.float32,
    )

    x_test = np.genfromtxt(
        data_dir / "x_test.csv",
        delimiter=",",
        skip_header=1,
        usecols=feature_indices,
        dtype=np.float32,
    )

    y_train = np.genfromtxt(
        data_dir / "y_train.csv",
        delimiter=",",
        skip_header=1,
        usecols=1,
        dtype=int,
    )

    # --------------------------------------------------------------
    # 3. Preserve structural missingness before generic cleaning
    # --------------------------------------------------------------
    x_train = encode_structural_missing(
        x_train,
        headers_calc,
    )

    x_test = encode_structural_missing(
        x_test,
        headers_calc,
    )

    # --------------------------------------------------------------
    # 4. Convert explicit BRFSS missing codes to NaN
    # --------------------------------------------------------------
    x_train = replace_missing_codes(
        x_train,
        headers_calc,
    )

    x_test = replace_missing_codes(
        x_test,
        headers_calc,
    )

    # --------------------------------------------------------------
    # 5. Stratified train / validation split
    # --------------------------------------------------------------
    (
        x_train_split,
        x_val_split,
        y_train_split,
        y_val_split,
    ) = stratified_train_val_split(
        x_train,
        y_train,
        val_ratio=val_ratio,
        seed=seed,
    )

    # --------------------------------------------------------------
    # 6. Fit numerical imputation on training only
    # --------------------------------------------------------------
    numerical_fill_values = (
        fit_numerical_imputer(
            x_train_split,
            headers_calc,
            NUMERICAL_FEATURES,
        )
    )

    x_train_proc = apply_imputer(
        x_train_split,
        headers_calc,
        numerical_fill_values,
    )

    x_val_proc = apply_imputer(
        x_val_split,
        headers_calc,
        numerical_fill_values,
    )

    x_test_proc = apply_imputer(
        x_test,
        headers_calc,
        numerical_fill_values,
    )

    # --------------------------------------------------------------
    # 7. Fit ordinal imputation on training only
    # --------------------------------------------------------------
    ordinal_fill_values = (
        fit_ordinal_imputer(
            x_train_split,
            headers_calc,
            ORDINAL_FEATURES,
        )
    )

    x_train_proc = apply_imputer(
        x_train_proc,
        headers_calc,
        ordinal_fill_values,
    )

    x_val_proc = apply_imputer(
        x_val_proc,
        headers_calc,
        ordinal_fill_values,
    )

    x_test_proc = apply_imputer(
        x_test_proc,
        headers_calc,
        ordinal_fill_values,
    )

    # --------------------------------------------------------------
    # 8. Fit standardization on training only
    # --------------------------------------------------------------
    features_to_standardize = (
        NUMERICAL_FEATURES
        + ORDINAL_FEATURES
    )

    scaler_stats = fit_standardizer(
        x_train_proc,
        headers_calc,
        features_to_standardize,
    )

    x_train_proc = apply_standardization(
        x_train_proc,
        headers_calc,
        scaler_stats,
    )

    x_val_proc = apply_standardization(
        x_val_proc,
        headers_calc,
        scaler_stats,
    )

    x_test_proc = apply_standardization(
        x_test_proc,
        headers_calc,
        scaler_stats,
    )

    # --------------------------------------------------------------
    # 9. Fit categorical encoding on training only
    # --------------------------------------------------------------
    categorical_encoder = (
        fit_categorical_encoder(
            x_train_proc,
            headers_calc,
            CATEGORICAL_FEATURES,
        )
    )

    (
        x_train_encoded,
        headers_encoded,
    ) = apply_categorical_encoder(
        x_train_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
        categorical_encoder,
    )

    (
        x_val_encoded,
        headers_val,
    ) = apply_categorical_encoder(
        x_val_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
        categorical_encoder,
    )

    (
        x_test_encoded,
        headers_test,
    ) = apply_categorical_encoder(
        x_test_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
        categorical_encoder,
    )

    x_train_encoded, headers_encoded = drop_redundant_encoded_columns(
    x_train_encoded,
    headers_encoded,
    )
    
    x_val_encoded, headers_val = drop_redundant_encoded_columns(
        x_val_encoded,
        headers_val,
    )
    
    x_test_encoded, headers_test = drop_redundant_encoded_columns(
        x_test_encoded,
        headers_test,
    )

    if not np.array_equal(
        headers_encoded,
        headers_val,
    ):
        raise ValueError(
            "Train and validation encoded headers differ."
        )

    if not np.array_equal(
        headers_encoded,
        headers_test,
    ):
        raise ValueError(
            "Train and test encoded headers differ."
        )

    # --------------------------------------------------------------
    # 10. Add intercept
    # --------------------------------------------------------------
    x_train_final = add_intercept(
        x_train_encoded
    )

    x_val_final = add_intercept(
        x_val_encoded
    )

    x_test_final = add_intercept(
        x_test_encoded
    )

    headers_final = np.concatenate([
        np.array(["intercept"]),
        headers_encoded,
    ])

    # --------------------------------------------------------------
    # 11. Final sanity checks
    # --------------------------------------------------------------
    for name, matrix in [
        ("training", x_train_final),
        ("validation", x_val_final),
        ("test", x_test_final),
    ]:
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{name.capitalize()} matrix "
                "contains NaN or infinite values."
            )

    if (
        x_train_final.shape[1]
        != x_val_final.shape[1]
        or x_train_final.shape[1]
        != x_test_final.shape[1]
    ):
        raise ValueError(
            "Train, validation, and test matrices "
            "do not have the same number of columns."
        )

    return (
        x_train_final,
        x_val_final,
        x_test_final,
        headers_final,
        y_train_split,
        y_val_split,
    )
