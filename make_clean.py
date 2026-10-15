"""Explore BEFORE, clean, explore AFTER, save the cleaned data. Raw files are only read.
Run: python make_clean.py [raw_dir] [clean_dir]
Writes: data/clean/*.npy, columns.txt, reports/before_after.csv, reports/before_after.png
"""
import os
import sys

import numpy as np

from helpers import load_csv_data
from preprocess import stateless_clean

raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/"
out = sys.argv[2] if len(sys.argv) > 2 else "data/clean/"
BANDS = [(0, .01), (.01, .2), (.2, .5), (.5, .9), (.9, 1.01)]


def summarize(x, y01, names, title):
    nan = np.isnan(x).mean(0)
    print(f"\n===== {title} =====")
    print("shape:", x.shape, "| overall NaN share: %.3f" % np.isnan(x).mean())
    for lo, hi in BANDS:
        print(f"  NaN in [{lo:.2f},{min(hi, 1):.2f}): {int(((nan >= lo) & (nan < hi)).sum())}")
    const = [n for j, n in enumerate(names) if np.isnan(x[:, j]).all() or np.nanstd(x[:, j]) == 0]
    print("  constant or all-NaN columns:", len(const), const)
    cor = []
    for j, n in enumerate(names):
        m = ~np.isnan(x[:, j])
        if m.sum() > 1000 and x[m, j].std() > 0:
            r = np.corrcoef(x[m, j], y01[m])[0, 1]
            cor.append((abs(r), r, n, m.mean()))
    print("  top 15 |corr| with target (NaN rows skipped): name, corr, share filled")
    for _, r, n, s in sorted(cor, reverse=True)[:15]:
        print(f"    {n:12s} {r:+.3f}  {s:.2f}")
    return nan


x_tr, x_te, y_tr, tr_ids, te_ids = load_csv_data(raw)
with open(os.path.join(raw, "x_train.csv")) as f:
    h = f.readline().strip().split(",")
names = h[1:] if len(h) == x_tr.shape[1] + 1 else h
y01 = (y_tr == 1).astype(float)

nan_before = summarize(x_tr, y01, names, "BEFORE (raw)")

xc, nc, rep = stateless_clean(x_tr, names)
xt, nt, _ = stateless_clean(x_te, names)
assert nc == nt, "train and test columns differ after cleaning"

nan_after = summarize(xc, y01, nc, "AFTER (stateless clean)")
print("\nrules whose column was not found:", rep["rules_not_found"])
print("yes/no-like columns in no rule list:", rep["unlisted_yes_no_like"])
print("dropped columns:", len(names) - len(nc))

os.makedirs("reports", exist_ok=True)
after = dict(zip(nc, nan_after))
with open("reports/before_after.csv", "w") as f:
    f.write("name,nan_before,nan_after,status\n")
    for n, b in zip(names, nan_before):
        a = after.get(n)
        f.write(f"{n},{b:.4f},{'' if a is None else format(a, '.4f')},{'dropped' if a is None else 'kept'}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ax[0].hist(nan_before, bins=20)
ax[0].set(title=f"Before: {len(names)} columns", xlabel="NaN rate", ylabel="columns")
ax[1].hist(nan_after, bins=20)
ax[1].set(title=f"After: {len(nc)} columns", xlabel="NaN rate")
fig.tight_layout()
fig.savefig("reports/before_after.png", dpi=120)

os.makedirs(out, exist_ok=True)
np.save(out + "x_train_clean.npy", xc)
np.save(out + "x_test_clean.npy", xt)
np.save(out + "y_train.npy", y_tr)
np.save(out + "train_ids.npy", tr_ids)
np.save(out + "test_ids.npy", te_ids)
with open(out + "columns.txt", "w") as f:
    f.write("\n".join(nc))
print("\nsaved to", out, xc.shape, xt.shape)
print("reports/before_after.csv and reports/before_after.png written")