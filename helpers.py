"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def read_header(data_path):
    """
    This function read the headers (variable names) of the data.
    N.B. The Id column is excluded to match the given helper functions above.
    Args:
        data_path (str): datafolder path

    Returns:
        header (np.array): header of data
    """    
    with open(data_path, "r", newline="") as f:
        headers = next(csv.reader(f), [])
    
    headers = np.array(headers)
    return headers[1:]


def one_hot(df, col_names, headers):
    """
    This functions takes a dataset, selected column names and the current set of variables (headers)
    that are being maintained (there may be a better way to handle the headers).
    Performs one-hot conversion on selected columns:
    The original column will be removed and for each unique value of the orginal column,
    a new column (named old_column_name+unique_value) will be added to the dataframe
    whose entries are binary (1 if the value matches that unique value and 0 otherwise)
    Args:
        df (np.array): data frame 
        col_names: the names of columns on which one-hot conversion will be performed
        headers: Current set of variables inside df
    Returns:
        new_df (np.array): data frame which one-hot conversion done on selected columms
        current_headers (np.array): new set of variables
    """
    col_idx = np.where(np.isin(headers, col_names))[0]
    current_headers = headers
    new_df = df
    for idx in sorted(col_idx, reverse=True):
        unique_vals = np.unique(df[:, idx])
        old_header = current_headers[idx]
        new_headers = [f"{old_header}_{v}" for v in unique_vals]
        current_headers = np.concatenate((current_headers[:idx], new_headers, current_headers[idx+1:]))
        bool_cols = np.column_stack([df[:, idx] == val for val in unique_vals])
        new_df = np.concatenate((new_df[:,:idx], bool_cols,  new_df[:,idx+1:]), axis = 1)
    return new_df, current_headers
