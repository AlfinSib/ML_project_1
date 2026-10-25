import csv
import numpy as np

MISSING_CATEGORY = -1
NOT_APPLICABLE_CODE = -2
UNKNOWN_CATEGORY = -3

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
    # response-quality variables
    "_MISVEGN",
    "_MISFRTN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRUITEX",
    "_VEGETEX",
    "_FRT16",
    "_VEG23",
    "PAMISS1_",

    # redundant age representations
    "_AGEG5YR",
    "_AGE_G",
    "_AGE65YR",

    # redundant / derived features
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

def select_features(x, headers, selected_features):
    """
    Select specified columns from a NumPy feature matrix.

    Args:
        x (np.ndarray):
            Feature matrix of shape (n_samples, n_features).

        headers (np.ndarray):
            Names of columns in x.

        selected_features (list[str] or np.ndarray):
            Feature names to preserve.

    Returns:
        x_selected (np.ndarray):
            Matrix containing only selected columns.

        selected_headers (np.ndarray):
            Names of selected columns in the same order.
    """

    missing_features = [
        feature
        for feature in selected_features
        if feature not in headers
    ]

    if missing_features:
        raise ValueError(
            f"Unknown features: {missing_features}"
        )

    indices = np.array([
        np.where(headers == feature)[0][0]
        for feature in selected_features
    ])

    return x[:, indices], headers[indices]

NUMERICAL_FEATURES = ['_AGE80', 'HTM4', 'WTKG3', '_BMI5', '_CHLDCNT', 'DROCDY3_', '_DRNKWEK', '_FRUTSUM', '_VEGESUM', 'METVL11_', 'METVL21_', 'FC60_', 'PADUR1_', 'PADUR2_', 'PAFREQ1_', 'PAFREQ2_', '_MINAC11', '_MINAC21', 'STRFREQ_', 'PAMIN11_', 'PAMIN21_', 'PAVIG11_', 'PAVIG21_']
ORDINAL_FEATURES = [
    #"_AGEG5YR",   # increasing age groups
    #"_AGE65YR",   # 18-64, 65+
    #"_AGE_G",     # increasing age groups
    #"_BMI5CAT",   # underweight/normal/overweight/obese
    "_EDUCAG",    # increasing education attainment
    "_INCOMG",    # increasing income brackets

    "ACTIN11_",   # none < moderate < vigorous
    "ACTIN21_",

    "_PACAT1",    # highly active / active / insufficient / inactive
    "_PA150R2",   # >=150 min / 1-149 / 0
    "_PA300R2",   # analogous ordered physical-activity levels
]
CATEGORICAL_FEATURES = ['_RFHLTH', '_HCVU651', '_RFHYPE5', '_CHOLCHK', '_RFCHOL', '_ASTHMS1', '_SMOKER3', '_RFBING5', '_RFDRHV5', '_TOTINDA', '_LMTACT1', '_LMTWRK1', '_LMTSCL1', '_RFSEAT2', '_RFSEAT3', '_FLSHOT6', '_PNEUMO2', '_AIDTST3', '_RACEGR3']   
STRUCTURAL_MISSING_FEATURES = [
    "_PNEUMO2",
    "_FLSHOT6",
]

ORD_NOT_MISSING_9 = [
    "ACTIN11_",
    "ACTIN21_",
]

ORD_MISSING_9 = [
    feature
    for feature in ORDINAL_FEATURES
    if feature not in ORD_NOT_MISSING_9
]

NUMERICAL_SPECIAL_MISSING = {
    "DROCDY3_": [900],
    "FC60_": [99900],
    "_DRNKWEK": [99900],
    "WTKG3": [99999],
    "_CHLDCNT": [9],
}

def encode_structural_missing(x, headers):
    """
    Encode structural BLANK values as a meaningful category.

    Must be called before replace_missing_codes().
    """
    x_clean = x.copy()
    header_to_idx = {
        feature: idx
        for idx, feature in enumerate(headers)
    }

    for feature in STRUCTURAL_MISSING_FEATURES:
        idx = header_to_idx[feature]

        # Original BLANK values are loaded as NaN
        x_clean[np.isnan(x_clean[:, idx]), idx] = NOT_APPLICABLE_CODE

    return x_clean

def replace_missing_codes(x, headers):
    """
    Convert explicit BRFSS missing-value codes to np.nan.

    Rules:
    - all categorical features: 9 -> NaN
    - ordinal features except ORD_NOT_MISSING_9: 9 -> NaN
    - numerical features: only explicitly listed sentinel values
    """
    x_clean = x.copy()

    header_to_idx = {
        feature: idx
        for idx, feature in enumerate(headers)
    }

    # All categorical features use 9 as missing
    for feature in CATEGORICAL_FEATURES:
        idx = header_to_idx[feature]
        x_clean[x_clean[:, idx] == 9, idx] = np.nan

    # Ordinal features: all use 9 except the two blank-missing ones
    for feature in ORDINAL_FEATURES:
        if feature not in ORD_NOT_MISSING_9:
            idx = header_to_idx[feature]
            x_clean[x_clean[:, idx] == 9, idx] = np.nan

    # Numerical features with explicit sentinel codes
    for feature, missing_codes in NUMERICAL_SPECIAL_MISSING.items():
        idx = header_to_idx[feature]

        for code in missing_codes:
            x_clean[x_clean[:, idx] == code, idx] = np.nan

    return x_clean

# Numerical features imputation

def numerical_medians(x, headers, numerical_features):
    medians = {}
    for feature in numerical_features:
        idx = np.where(headers == feature)[0][0]
        medians[feature] = np.nanmedian(x[:, idx])
    return medians

def apply_imputer(x, headers, fill_values):
    x_out = x.copy()

    for feature, fill_value in fill_values.items():
        idx = np.where(headers == feature)[0][0]

        mask = np.isnan(x_out[:, idx])
        x_out[mask, idx] = fill_value

    return x_out


# Categories extraction from categorical features
def fit_categorical_encoder(x, headers, categorical_features):
    """
    Learn categorical encodings from the training set.

    Returns
    -------
    encoder : dict
        For each feature:
        - categories: categories that can be encoded
        - reference: category that will be dropped
    """
    encoder = {}

    for feature in categorical_features:
        idx = np.where(headers == feature)[0][0]
        column = x[:, idx].copy()

        # Treat missing as an explicit category
        column[np.isnan(column)] = MISSING_CATEGORY

        observed_categories = np.unique(column)

        # Always support missing and unseen categories,
        # even if they don't occur in training.
        categories = np.unique(
            np.append(
                observed_categories,
                [MISSING_CATEGORY, UNKNOWN_CATEGORY]
            )
        )

        # Pick a normal observed category as reference
        regular_categories = [
            category
            for category in observed_categories
            if category not in (
                MISSING_CATEGORY,
                UNKNOWN_CATEGORY,
                NOT_APPLICABLE_CODE,
            )
        ]

        if len(regular_categories) == 0:
            raise ValueError(
                f"No regular category found for '{feature}'."
            )

        reference = regular_categories[0]

        encoder[feature] = {
            "categories": categories,
            "reference": reference,
        }

    return encoder

def apply_categorical_encoder(
    x,
    headers,
    categorical_features,
    encoder,
):
    """
    One-hot encode categorical features using an encoder
    fitted on the training set.

    Unseen categories are mapped to UNKNOWN_CATEGORY.
    One reference category per feature is dropped.
    """
    encoded_columns = []
    encoded_headers = []

    categorical_set = set(categorical_features)

    for idx, feature in enumerate(headers):
        column = x[:, idx].copy()

        # Non-categorical feature: leave unchanged
        if feature not in categorical_set:
            encoded_columns.append(column.reshape(-1, 1))
            encoded_headers.append(feature)
            continue

        # NaN -> explicit missing category
        column[np.isnan(column)] = MISSING_CATEGORY

        feature_categories = encoder[feature]["categories"]
        reference = encoder[feature]["reference"]

        # Categories actually observed during training.
        known_categories = feature_categories[
            feature_categories != UNKNOWN_CATEGORY
        ]

        # Anything not known from training becomes UNKNOWN
        unseen_mask = ~np.isin(column, known_categories)

        column[unseen_mask] = UNKNOWN_CATEGORY

        # One-hot encode all categories except the reference
        for category in feature_categories:

            if category == reference:
                continue

            dummy = (column == category).astype(float)

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

# Ordinal imputation and standardization
def fit_ordinal_imputer(x, headers, ordinal_features):
    modes = {}

    for feature in ordinal_features:
        idx = np.where(headers == feature)[0][0]
        column = x[:, idx]

        non_missing = column[~np.isnan(column)]

        values, counts = np.unique(
            non_missing,
            return_counts=True
        )

        modes[feature] = values[np.argmax(counts)]

    return modes

# Standardization
def standardization_stats(x, headers, features):
    stats = {}
    for feature in features:
        idx = np.where(headers == feature)[0][0]
        col = x[:, idx]
        stats[feature] = (np.mean(col), np.std(col))
    return stats
    
def apply_standardization(x, headers, stats):
    x_out = x.copy()
    for feature, (mean, std) in stats.items():
        idx = np.where(headers == feature)[0][0]
        if std == 0:
            x_out[:, idx] = 0
        else:
            x_out[:, idx] = (x[:, idx] - mean) / std
    return x_out

def add_intercept(x):
    return np.column_stack([
        np.ones(x.shape[0]),
        x
    ])

def stratified_train_val_split(x, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        n_val = int(len(label_indices) * val_ratio)

        val_indices.extend(label_indices[:n_val])
        train_indices.extend(label_indices[n_val:])

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


def prepare_data(data_dir, val_ratio=0.2, seed=42):
    # --------------------------------------------------
    # 1. Read CSV headers
    # --------------------------------------------------
    with open(data_dir / "x_train.csv", "r") as f:
        reader = csv.reader(f)
        all_headers = np.array(next(reader))

    # Find indices of the features we decided to keep
    feature_indices = np.array([
        np.where(all_headers == feature)[0][0]
        for feature in CALCULATED_FEATURES
    ])

    headers_calc = all_headers[feature_indices]

    # --------------------------------------------------
    # 2. Load only selected features
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 3. Preserve structural missingness
    # --------------------------------------------------
    x_train = encode_structural_missing(
        x_train,
        headers_calc,
    )

    x_test = encode_structural_missing(
        x_test,
        headers_calc,
    )

    # --------------------------------------------------
    # 4. Convert explicit missing codes -> NaN
    # --------------------------------------------------
    x_train = replace_missing_codes(
        x_train,
        headers_calc,
    )

    x_test = replace_missing_codes(
        x_test,
        headers_calc,
    )

    # --------------------------------------------------
    # 5. Stratified train / validation split
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 6. Numerical imputation
    # Fit on training only
    # --------------------------------------------------
    num_medians = numerical_medians(
        x_train_split,
        headers_calc,
        NUMERICAL_FEATURES,
    )

    x_train_proc = apply_imputer(
        x_train_split,
        headers_calc,
        num_medians,
    )

    x_val_proc = apply_imputer(
        x_val_split,
        headers_calc,
        num_medians,
    )

    x_test_proc = apply_imputer(
        x_test,
        headers_calc,
        num_medians,
    )

    # --------------------------------------------------
    # 7. Ordinal imputation
    # Fit on training only
    # --------------------------------------------------
    ordinal_modes = fit_ordinal_imputer(
        x_train_split,
        headers_calc,
        ORDINAL_FEATURES,
    )

    x_train_proc = apply_imputer(
        x_train_proc,
        headers_calc,
        ordinal_modes,
    )

    x_val_proc = apply_imputer(
        x_val_proc,
        headers_calc,
        ordinal_modes,
    )

    x_test_proc = apply_imputer(
        x_test_proc,
        headers_calc,
        ordinal_modes,
    )

    # --------------------------------------------------
    # 8. Standardize numerical + ordinal features
    # --------------------------------------------------
    features_to_standardize = (
        NUMERICAL_FEATURES
        + ORDINAL_FEATURES
    )

    stats = standardization_stats(
        x_train_proc,
        headers_calc,
        features_to_standardize,
    )

    x_train_proc = apply_standardization(
        x_train_proc,
        headers_calc,
        stats,
    )

    x_val_proc = apply_standardization(
        x_val_proc,
        headers_calc,
        stats,
    )

    x_test_proc = apply_standardization(
        x_test_proc,
        headers_calc,
        stats,
    )

    # --------------------------------------------------
    # 9. Fit categorical encoder on training only
    # --------------------------------------------------
    categorical_encoder = fit_categorical_encoder(
        x_train_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
    )

    x_train_encoded, headers_encoded = apply_categorical_encoder(
        x_train_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
        categorical_encoder,
    )

    x_val_encoded, headers_val = apply_categorical_encoder(
        x_val_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
        categorical_encoder,
    )

    x_test_encoded, headers_test = apply_categorical_encoder(
        x_test_proc,
        headers_calc,
        CATEGORICAL_FEATURES,
        categorical_encoder,
    )

    assert np.array_equal(
        headers_encoded,
        headers_val,
    )

    assert np.array_equal(
        headers_encoded,
        headers_test,
    )

    # --------------------------------------------------
    # 10. Add intercept
    # --------------------------------------------------
    x_train_final = add_intercept(x_train_encoded)
    x_val_final = add_intercept(x_val_encoded)
    x_test_final = add_intercept(x_test_encoded)

    headers_final = np.concatenate([
        np.array(["intercept"]),
        headers_encoded,
    ])

    # --------------------------------------------------
    # 11. Final sanity checks
    # --------------------------------------------------
    assert not np.isnan(x_train_final).any()
    assert not np.isnan(x_val_final).any()
    assert not np.isnan(x_test_final).any()

    assert x_train_final.shape[1] == x_val_final.shape[1]
    assert x_train_final.shape[1] == x_test_final.shape[1]

    return (
        x_train_final,
        x_val_final,
        x_test_final,
        headers_final,
        y_train_split,
        y_val_split,
    )