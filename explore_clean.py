"""Explore the CLEANED data (after stateless_clean). Run: python explore_clean.py [data_dir]"""
import os
import sys

import numpy as np

from helpers import load_csv_data
from preprocess import stateless_clean

d = sys.argv[1] if len(sys.argv) > 1 else "data/clean/"
x, _, y, _, _ = load_csv_data(d)
with open(os.path.join(d, "x_train.csv")) as f:
    h = f.readline().strip().split(",")
names = h[1:] if len(h) == x.shape[1] + 1 else h

xc, nc, _ = stateless_clean(x, names)
y01 = (y == 1).astype(float)

nan = np.isnan(xc).mean(0)
print("shape:", xc.shape)
for lo, hi in [(0, .01), (.01, .2), (.2, .5), (.5, .9), (.9, 1.01)]:
    print(f"NaN in [{lo:.2f},{hi:.2f}):", int(((nan >= lo) & (nan < hi)).sum()))
print("constant cols:", [n for j, n in enumerate(nc) if np.nanstd(xc[:, j]) == 0])

cor = []
for j, n in enumerate(nc):
    m = ~np.isnan(xc[:, j])
    if m.sum() > 1000 and xc[m, j].std() > 0:
        r = np.corrcoef(xc[m, j], y01[m])[0, 1]
        cor.append((abs(r), r, n, m.mean()))
print("\nTop 15 |corr| (NaN rows skipped), name, corr, share filled")
for _, r, n, s in sorted(cor, reverse=True)[:15]:
    print(f"{n:12s} {r:+.3f}  {s:.2f}")
