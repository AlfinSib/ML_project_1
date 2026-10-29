"""Model diagnostics for the BRFSS heart-disease project.

Metric calculator on (y_true, scores) — no fitting, no model state, so it
works with any classifier that produces probability scores (training.py's
LogisticClassifier or anything else).

Moved out of training.py: models live there, evaluation lives here.

Usage:
    import diagnostics as dg

    out = dg.Diagnostics.evaluate(y_val, p_val,
                                  metrics=["f1", "auc", "confusion"],
                                  threshold=0.2)
    dg.Diagnostics.report(out, title="validation")

    folds = dg.Diagnostics.cross_validate(
        x, y, clf_factory=lambda: LogisticClassifier(...), k=5)

Dependencies: numpy, diagstat (ROC_AUC, F1_score), helpers (k_fold_split,
imported lazily inside cross_validate).
"""

import numpy as np
import diagstat as ds


class Diagnostics:
    """Metric calculator on (y_true, scores). No fitting, no state.

    Usage:
        out = Diagnostics.evaluate(y_val, p_val,
                                   metrics=["f1", "auc", "confusion"],
                                   threshold=0.2)
        Diagnostics.report(out)
    """

    METRICS = ("f1", "precision", "recall", "auc", "accuracy",
               "roc", "confusion", "log_loss", "01_loss")

    @staticmethod
    def evaluate(y_true, scores, metrics=None, threshold=0.5):
        """Compute the requested metrics. Returns an ordered dict.

        metrics: list from METRICS (None = f1, precision, recall, auc).
        threshold: used by f1/precision/recall/confusion/accuracy/01_loss.
        'roc' returns (fpr, tpr) arrays; 'auc' a float.
        """
        y = (np.asarray(y_true) == 1)
        p = np.asarray(scores, dtype=float)
        metrics = list(metrics) if metrics is not None \
            else ["f1", "precision", "recall", "auc"]

        for m in metrics:
            if m not in Diagnostics.METRICS:
                raise ValueError(f"unknown metric {m!r}; choose from {Diagnostics.METRICS}")

        pred = p > threshold
        tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y))
        fn = int(np.sum(~pred & y)); tn = int(np.sum(~pred & ~y))

        out = {}
        for m in metrics:
            if m == "f1":
                out["f1"] = 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) else 0.0
            elif m == "precision":
                out["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
            elif m == "recall":
                out["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
            elif m == "auc":
                out["auc"] = float(ds.ROC_AUC(y.astype(float), p)[2])
            elif m == "accuracy":
                out["accuracy"] = (tp + tn) / max(len(y), 1)
            elif m == "01_loss":
                out["01_loss"] = 1.0 - (tp + tn) / max(len(y), 1)
            elif m == "confusion":
                out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            elif m == "roc":
                fpr, tpr, _ = ds.ROC_AUC(y.astype(float), p)
                out["roc"] = (fpr, tpr)
            elif m == "log_loss":
                # softplus form: y*softplus(-z) + (1-y)*softplus(z) is ALREADY
                # the positive cross-entropy (no leading minus, unlike log-sigma form)
                z = np.clip(np.log(p / np.clip(1 - p, 1e-12, None)), -30, 30)
                out["log_loss"] = float(np.mean(y * np.logaddexp(0, -z)
                                                 + (1 - y) * np.logaddexp(0, z)))
        out["threshold"] = threshold
        out["n"] = len(y)
        out["n_pos"] = int(y.sum())
        return out

    @staticmethod
    def report(results, title=None):
        """Pretty-print an evaluate() dict (or list of dicts, e.g. CV folds)."""
        dicts = results if isinstance(results, list) else [results]
        if title:
            print(f"--- {title} ---")
        for r in dicts:
            parts = []
            for k, v in r.items():
                if k in ("roc", "scores", "val_idx"):   # bulky arrays: skip in printouts
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print("  " + "  ".join(parts))
        return None

    @staticmethod
    def cross_validate(x, y, clf_factory, k=5, seed=0, tune_threshold=True,
                       metrics=None, verbose=True, fold_preprocess=None):
        """K-fold CV around any fit/predict_scores classifier. Fits, tunes t*
        on the train part (leakage-free), evaluates on the val part.
        Returns per-fold list of evaluate() dicts (plus 't*').

        clf_factory: zero-arg callable returning a FRESH classifier, e.g.
            lambda: LogisticClassifier(lam=1e-5, max_iters=500)
        The classifier only needs .fit(x, y), .decision_threshold(x, y)
        (when tune_threshold) and .predict_scores(x).

        fold_preprocess: optional callable(tr_idx, va_idx) -> (x_tr, x_va).
        When given, it REPLACES slicing x by the fold indices: use it to run
        the whole preprocessing pipeline per fold (fills fitted on the
        fold's train rows only), e.g. with preprocessing.py's cache:

            def fold_data(tr, va):
                d = pp.run_train(fit_idx=tr)        # fills from THIS fold's train
                return d["x_1h"][tr], d["x_1h"][va]

            dg.Diagnostics.cross_validate(None, y, clf_factory,
                                           fold_preprocess=fold_data)

        (x is ignored when fold_preprocess is given; pass None.)
        """
        import helpers
        folds = helpers.k_fold_split(len(y), k, seed=seed)
        out = []
        for i, (tr, va) in enumerate(folds):
            if fold_preprocess is not None:
                x_tr, x_va = fold_preprocess(tr, va)
            else:
                x_tr, x_va = x[tr], x[va]
            clf = clf_factory()
            clf.fit(x_tr, y[tr])
            if tune_threshold:
                t = clf.decision_threshold(x_tr, y[tr])
            else:
                t = 0.5
            p_va = clf.predict_scores(x_va)
            r = Diagnostics.evaluate(y[va], p_va, metrics=metrics,
                                     threshold=t)
            r["t*"] = t
            # keep the fold's predictions & indices: pooled-OOF plots and
            # threshold tuning reuse them WITHOUT refitting anything
            r["scores"] = p_va
            r["val_idx"] = va
            out.append(r)
            if verbose:
                Diagnostics.report([r], title=f"fold {i}")
        if verbose:
            key_metrics = metrics if metrics is not None \
                else ["f1", "precision", "recall", "auc"]
            means = {m: np.mean([r[m] for r in out]) for m in key_metrics}
            print("mean:", "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
        return out


if __name__ == "__main__":
    # smoke test: synthetic scores with known quality
    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.random(n) < 0.088).astype(float)            # 8.8% prevalence
    p = 1 / (1 + np.exp(-np.clip(rng.normal(0, 1, n) + 1.2 * (y == 1), -30, 30)))

    out = Diagnostics.evaluate(y, p, metrics=["f1", "precision", "recall",
                                              "auc", "accuracy", "confusion",
                                              "log_loss", "01_loss"],
                               threshold=0.2)
    Diagnostics.report(out, title="smoke test (all metrics)")

    # error path
    try:
        Diagnostics.evaluate(y, p, metrics=["nope"])
    except ValueError as e:
        print("error path OK:", str(e)[:60])
