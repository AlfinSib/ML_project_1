"""Explore the RAW data (read-only). Run: python explore_data.py [data_dir]
Writes reports/column_summary.csv and reports/overview.png
"""
import os
import sys

import numpy as np

from helpers import load_csv_data

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/"
x_train, x_test, y_train, _, _ = load_csv_data(data_dir)
with open(os.path.join(data_dir, "x_train.csv")) as f:
    header = f.readline().strip().split(",")
names = header[1:] if len(header) == x_train.shape[1] + 1 else header

y01 = (y_train == 1).astype(float)
nan_tr = np.isnan(x_train).mean(0)
nan_te = np.isnan(x_test).mean(0)

print("== Shapes ==")
print("x_train", x_train.shape, "| x_test", x_test.shape, "| y_train", y_train.shape)
vals, cnt = np.unique(y_train, return_counts=True)
print("labels:", dict(zip(vals.astype(int).tolist(), cnt.tolist())))
print("positive rate: %.4f" % y01.mean())

print("\n== Missing values (train) ==")
for lo, hi in [(0, .01), (.01, .2), (.2, .5), (.5, .9), (.9, 1.01)]:
    k = int(((nan_tr >= lo) & (nan_tr < hi)).sum())
    print(f"columns with NaN rate in [{lo:.2f}, {min(hi, 1):.2f}): {k}")
order = np.argsort(-nan_tr)[:10]
print("worst 10:", [(names[j], round(float(nan_tr[j]), 3)) for j in order])

const = [names[j] for j in range(x_train.shape[1])
         if np.isnan(x_train[:, j]).all() or np.nanstd(x_train[:, j]) == 0]
print(f"\n== Constant (or all-NaN) columns: {len(const)} ==")
print(const)

gap = np.abs(nan_tr - nan_te)
print("\n== Largest NaN-rate gap between train and test ==")
print([(names[j], round(float(gap[j]), 3)) for j in np.argsort(-gap)[:5]])

corr = np.full(x_train.shape[1], np.nan)
for j in range(x_train.shape[1]):
    m = ~np.isnan(x_train[:, j])
    if m.sum() > 1000 and x_train[m, j].std() > 0:
        corr[j] = np.corrcoef(x_train[m, j], y01[m])[0, 1]
top = np.argsort(-np.nan_to_num(np.abs(corr)))[:15]
print("\n== Top 15 columns by |corr| with target (crude, special codes included) ==")
for j in top:
    print(f"{names[j]:14s}{corr[j]:+.3f}")

os.makedirs("reports", exist_ok=True)
with open("reports/column_summary.csv", "w") as f:
    f.write("name,nan_train,nan_test,n_unique,min,max,mean,std,corr_target\n")
    for j, n in enumerate(names):
        c = x_train[:, j]
        m = ~np.isnan(c)
        if m.any():
            u = len(np.unique(c[m]))
            row = (u, c[m].min(), c[m].max(), c[m].mean(), c[m].std())
        else:
            row = (0, np.nan, np.nan, np.nan, np.nan)
        f.write(f"{n},{nan_tr[j]:.4f},{nan_te[j]:.4f},{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{corr[j]}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(nan_tr, bins=20)
ax[0].set(title="NaN rate per column", xlabel="NaN rate", ylabel="columns")
ax[1].bar(["-1", "1"], [(y_train == -1).sum(), (y_train == 1).sum()])
ax[1].set(title="Label balance")
c = corr[top][::-1]
ax[2].barh([names[j] for j in top][::-1], c)
ax[2].set(title="Top 15 correlations with target")
fig.tight_layout()
fig.savefig("reports/overview.png", dpi=120)
print("\nPer-column summary saved to reports/column_summary.csv")
print("Figure saved to reports/overview.png")
