"""Learned preprocessing: fit on training data only, then transform.

Includes median imputation, NaN indicators, quantile clipping,
one-hot encoding of nominal columns, and standardisation.

"""
import numpy as np


DEFAULT_ONEHOT = [
    "_RACE", "_ASTHMS1", "_CHOLCHK",
    "_LMTACT1", "_LMTWRK1", "_LMTSCL1",
]


def _build(x, p, dtype):
    """Build raw feature matrix (imputed, flagged, clipped, one-hot) before scaling."""
    ix = {n: j for j, n in enumerate(p["in_names"])}
    cols, names = [], []
    for n in p["num"]:
        c = x[:, ix[n]].astype(np.float64)
        miss = np.isnan(c)
        if n in p["flag"]:
            cols.append(miss.astype(dtype))
            names.append(n + "_isnan")
        c[miss] = p["median"][n]
        if n in p["clip"]:
            c = np.clip(c, *p["clip"][n])
        cols.append(c.astype(dtype))
        names.append(n)
    for n, cats in p["onehot"].items():
        c = np.where(np.isnan(x[:, ix[n]]), -1.0, x[:, ix[n]])
        for k in cats:
            cols.append((c == k).astype(dtype))
            names.append(f"{n}={k:g}")
    return np.column_stack(cols), names


def fit(x, names, onehot=DEFAULT_ONEHOT, max_nan=0.8, flag_nan=0.05,
        clip_q=(0.5, 99.5), min_unique_clip=10, dtype=np.float32):
    """Learn statistics from training data for later transformation.
    Returns a dict of parameters to pass to transform().""" 
    names = list(names)
    ix = {n: j for j, n in enumerate(names)}
    p = {"in_names": names, "onehot": {}, "num": [], "median": {},
         "flag": [], "clip": {}, "dtype": dtype}

    for n in onehot:
        if n in ix:
            c = np.where(np.isnan(x[:, ix[n]]), -1.0, x[:, ix[n]])
            p["onehot"][n] = np.unique(c)

    for n in names:
        if n in p["onehot"]:
            continue
        c = x[:, ix[n]]
        miss = np.isnan(c)
        if miss.mean() > max_nan:
            continue
        obs = c[~miss]
        if obs.size == 0 or obs.min() == obs.max():
            continue
        p["num"].append(n)
        p["median"][n] = float(np.median(obs))
        if miss.mean() >= flag_nan:
            p["flag"].append(n)
        if np.unique(obs).size > min_unique_clip:
            p["clip"][n] = tuple(float(v) for v in np.percentile(obs, clip_q))

    raw, feat = _build(x, p, dtype)
    mu = raw.mean(axis=0, dtype=np.float64)
    sd = raw.std(axis=0, dtype=np.float64)
    keep = sd > 1e-8
    p["mu"], p["sd"], p["keep"] = mu[keep], sd[keep], keep
    p["names"] = ["bias"] + [n for n, k in zip(feat, keep) if k]
    return p


def transform(x, p):
    """Apply frozen statistics. Returns (X with bias column at position 0, names).
    """
    if x.shape[1] != len(p["in_names"]):
        raise ValueError(f"x has {x.shape[1]} columns, fit saw {len(p['in_names'])}")
    raw, _ = _build(x, p, p["dtype"])
    X = ((raw[:, p["keep"]] - p["mu"]) / p["sd"]).astype(p["dtype"])
    return np.c_[np.ones(len(X), dtype=p["dtype"]), X], p["names"]