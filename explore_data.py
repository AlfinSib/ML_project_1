"""Read-only exploration of the BRFSS data.

Changes nothing. Prints a report, saves a per-column summary CSV and a figure.
Usage: python explore_data.py [path/to/data/]
Only NumPy and matplotlib (for plots) are used, as required by the project.
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers import load_csv_data

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/clean"
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)


def read_header(path):
    with open(path) as f:
        return f.readline().strip().split(",")


def top_values(col, k=3):
    """Most frequent non-NaN values of a column with their share."""
    vals = col[~np.isnan(col)]
    if vals.size == 0:
        return []
    u, c = np.unique(vals, return_counts=True)
    order = np.argsort(-c)[:k]
    return [(u[i], c[i] / vals.size) for i in order]


def corr_with_target(x, y01, min_n=100):
    """Pearson correlation of each column with the target, NaN ignored."""
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        m = ~np.isnan(x[:, j])
        if m.sum() < min_n:
            continue
        a, b = x[m, j], y01[m]
        if a.std() == 0 or b.std() == 0:
            continue
        out[j] = np.corrcoef(a, b)[0, 1]
    return out


def rate_by_value(x, y01, names, col_name, max_values=15):
    """Positive rate for each value of one column. Call it from a shell."""
    j = names.index(col_name)
    for v in np.unique(x[~np.isnan(x[:, j]), j])[:max_values]:
        m = x[:, j] == v
        print(f"{col_name}={v:g}  n={m.sum():7d}  positive rate={y01[m].mean():.3f}")
    m = np.isnan(x[:, j])
    if m.any():
        print(f"{col_name}=NaN  n={m.sum():7d}  positive rate={y01[m].mean():.3f}")


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)
    header = read_header(os.path.join(DATA_DIR, "x_train.csv"))
    names = header[1:] if len(header) == x_train.shape[1] + 1 else header
    y01 = (y_train == 1).astype(float)

    # 1. Shapes and target
    print("== Shapes ==")
    print("x_train", x_train.shape, "| x_test", x_test.shape, "| y_train", y_train.shape)
    print("labels:", dict(zip(*np.unique(y_train, return_counts=True))))
    print(f"positive rate: {y01.mean():.4f}")

    # 2. Missing values
    nan_rate = np.isnan(x_train).mean(axis=0)
    print("\n== Missing values (train) ==")
    for lo, hi in [(0, 0.01), (0.01, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        n = int(((nan_rate >= lo) & (nan_rate < hi)).sum())
        print(f"columns with NaN rate in [{lo:.2f}, {min(hi, 1):.2f}): {n}")
    print("worst 10:", [(names[j], round(float(nan_rate[j]), 3)) for j in np.argsort(-nan_rate)[:10]])

    # 3. Constant columns
    const = [names[j] for j in range(x_train.shape[1]) if np.nanstd(x_train[:, j]) == 0]
    print(f"\n== Constant (or all-NaN) columns: {len(const)} ==")
    print(const)

    # 4. Train vs test shift in missingness
    shift = np.abs(nan_rate - np.isnan(x_test).mean(axis=0))
    print("\n== Largest NaN-rate gap between train and test ==")
    print([(names[j], round(float(shift[j]), 3)) for j in np.argsort(-shift)[:5]])

    # 5. Correlation with target
    corr = corr_with_target(x_train, y01)
    order = np.argsort(-np.nan_to_num(np.abs(corr)))[:15]
    print("\n== Top 15 columns by |corr| with target (crude, special codes included) ==")
    for j in order:
        print(f"{names[j]:15s} {corr[j]:+.3f}")

    # 6. Per-column summary file
    path = os.path.join(OUT_DIR, "column_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "nan_rate", "n_unique", "min", "max",
                    "top1", "top1_share", "top2", "top2_share", "top3", "top3_share", "corr_target"])
        for j, name in enumerate(names):
            col = x_train[:, j]
            tops = top_values(col)
            tops += [(np.nan, np.nan)] * (3 - len(tops))
            row = [name, f"{nan_rate[j]:.4f}", len(np.unique(col[~np.isnan(col)])),
                   np.nanmin(col) if nan_rate[j] < 1 else "", np.nanmax(col) if nan_rate[j] < 1 else ""]
            for v, s in tops:
                row += [v, f"{s:.4f}" if s == s else ""]
            row.append(f"{corr[j]:.4f}" if corr[j] == corr[j] else "")
            w.writerow(row)
    print(f"\nPer-column summary saved to {path}")

    # 7. Figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(nan_rate, bins=20)
    ax[0].set_xlabel("NaN rate of a column")
    ax[0].set_ylabel("number of columns")
    ax[1].bar(["no MICHD (-1)", "MICHD (+1)"], [(y_train == -1).sum(), (y_train == 1).sum()])
    ax[1].set_ylabel("number of samples")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "overview.png"), dpi=150)
    print(f"Figure saved to {os.path.join(OUT_DIR, 'overview.png')}")


if __name__ == "__main__":
    main()
