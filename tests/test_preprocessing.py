"""Tests for the data preprocessing functions."""

import numpy as np

from helpers import one_hot

df = np.array([
    [10, 1, 0, 5],
    [20, 2, 1, 6],
    [30, 1, 1, 7],
    [40, 3, 0, 8],
])

headers = np.array([
    "age",
    "category_a",
    "category_b",
    "score",
])

col_names = np.array([
    "category_a",
    "category_b",
])


def test_original_inputs_unmodified():
    old_df = df.copy()
    old_headers = headers.copy()
    one_hot(df, col_names, headers)
    assert np.array_equal(df, old_df)
    assert np.array_equal(headers, old_headers)


def test_exactly_one_active_category_per_row():
    new_df, new_headers = one_hot(df, col_names, headers)
    
    for col_name in col_names:
        indices = np.where(
            np.char.startswith(
                new_headers.astype(str),
                f"{col_name}_"
            )
        )[0]

        assert np.all(new_df[:, indices].sum(axis=1) == 1)


def test_one_added_col_for_each_unique_value():
    new_df, new_headers = one_hot(df, col_names, headers)

    for col_name in col_names:
        idx = np.where(headers == col_name)[0][0]
        unique_values = np.unique(df[:, idx])

        generated_headers = [
            header
            for header in new_headers
            if str(header).startswith(f"{col_name}_")
        ]

        assert len(generated_headers) == len(unique_values)

        for value in unique_values:
            expected_header = f"{col_name}_{value}"
            assert expected_header in generated_headers   


def test_encoding_done_correctly():
    new_df, new_headers = one_hot(df, col_names, headers)

    for col_name in col_names:
        old_idx = np.where(headers == col_name)[0][0]
        unique_values = np.unique(df[:, old_idx])

        for value in unique_values:
            expected_header = f"{col_name}_{value}"
            new_idx = np.where(new_headers == expected_header)[0][0]
            expected_column = (df[:, old_idx] == value).astype(int)
            actual_column = new_df[:, new_idx]

            assert np.array_equal(actual_column, expected_column)
    