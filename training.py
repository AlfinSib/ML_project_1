"""Model training for the BRFSS heart-disease project: sklearn-style wrapper.

Builds on implementations.py (graded functions, unchanged) and diagstat.py
(metrics). Pure numpy — course rule: stdlib + numpy + seaborn only.

Design
------
LogisticClassifier      sklearn-like regressor/classifier
  .fit(x, y)            -> self (stores weights, scaler, config in .result_)
  .predict_scores(x)    -> probabilities
  .predict(x)           -> {0,1} labels at the tuned threshold
  .decision_threshold(x, y)  -> tune t* on given (x, y) with a fine grid
  .result_              -> a FitResult object (weights, scaler, config, history)

FitResult               container for one fit's outputs
  .w, .intercept, .loss, .n_iter, .mu, .sd, .config, .convergence_

Metrics & diagnostics (Diagnostics.evaluate / report / cross_validate) live
in diagnostics.py — import them from there.

Typical use (with preprocessing.py):
    import preprocessing as pp
    import training as tr

    data = pp.run_train()
    clf = tr.LogisticClassifier(lam=1e-5, gamma=0.1, max_iters=1000)
    clf.fit(data["x_1h"], data["y"])

    p = clf.predict_scores(x_val)
    t = clf.decision_threshold(x_train, y_train)     # tuned on train!

    import diagnostics as dg
    dg.Diagnostics.evaluate(y_val, p, threshold=t)
"""

import numpy as np
import implementations as imp
import diagstat as ds


def sigmoid(z):
    """Numerically stable logistic function (tanh form)."""
    return 0.5 * (1 + np.tanh(0.5 * z))


# ============================================================================
# fit result container
# ============================================================================

class FitResult:
    """Outputs of one trained model. Returned as classifier.result_.

    Attributes
    ----------
    w : (d+1,) weight vector, w[0] = intercept (column standardized)
    intercept : float, w[0]
    loss : final (unweighted) logistic loss on the fit data
    n_iter : iterations run
    mu, sd : standardization constants fitted on the fit data
    config : dict of the hyperparameters used
    convergence_ : list of (iteration, loss) checkpoints (or None)
    """

    def __init__(self, w, loss, n_iter, mu, sd, config, convergence=None):
        self.w = w
        self.intercept = float(w[0])
        self.loss = loss
        self.n_iter = n_iter
        self.mu = mu
        self.sd = sd
        self.config = config
        self.convergence_ = convergence

    def predict_scores(self, x, standardize=True):
        """Probabilities for rows of x (raw features, no intercept column).
        Standardization uses THIS fit's mu/sd (never refit)."""
        if standardize:
            x = (x - self.mu) / self.sd
            return sigmoid(x @ self.w[1:] + self.w[0])
        return sigmoid(x @ self.w)          # x already has intercept, w already scaled

    def coefficients(self, headers=None):
        """(name, weight) list, intercept first; for bar plots."""
        names = ["intercept"] + list(headers) if headers is not None \
            else ["intercept"] + [f"x{j}" for j in range(len(self.w) - 1)]
        return list(zip(names, self.w))


# ============================================================================
# sklearn-style classifier
# ============================================================================

class LogisticClassifier:
    """Regularised (optionally class-weighted) logistic regression via GD.

    Mimics the sklearn API shape:
        clf = LogisticClassifier(lam=1e-5, gamma=0.1, max_iters=1000)
        clf.fit(x, y)                       # y in {0,1}; x WITHOUT intercept
        clf.predict(x)                      # {0,1} labels at threshold t*
        clf.predict_scores(x)               # probabilities

    Variant selection (the three models are nested, chosen by parameters):
        standard logistic       lam=0,   w_pos=1   -> imp.logistic_regression
        regularised (L2)        lam>0,   w_pos=1   -> imp.reg_logistic_regression
        class-weighted (+L2)    w_pos>1  (any lam) -> imp.reg_logistic_regression_weighted

    The standardizer (mu/sd) is fitted inside fit() on the fit rows and
    reused unchanged for all predictions — the same leakage discipline as
    preprocessing.py.

    Parameters
    ----------
    lam : L2 penalty strength (per-sample scale of implementations.py);
          lam=0 selects the UNREGULARISED model (routes to logistic_regression)
    gamma : gradient step size (0.1 recommended after standardization)
    max_iters : GD iterations
    w_pos : positive-class gradient weight (1.0 = unweighted)
    standardize : if False, x must already contain the intercept column
                  and be standardized (advanced use)
    track_every : if >0, record (iter, loss) every N iterations in
                  result_.convergence_
    """

    def __init__(self, lam=1e-5, gamma=0.1, max_iters=1000, w_pos=1.0,
                 standardize=True, track_every=0, seed=0):
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.max_iters = int(max_iters)
        self.w_pos = float(w_pos)
        self.standardize = standardize
        self.track_every = int(track_every)
        self.seed = seed
        self.result_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, x, y):
        """Train on (x, y). y must be {0,1}. Returns self."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert set(np.unique(y)) <= {0.0, 1.0}, \
            f"y must be coded {{0,1}}, got {np.unique(y)[:5]}"

        config = {"lam": self.lam, "gamma": self.gamma,
                  "max_iters": self.max_iters, "w_pos": self.w_pos}

        if self.standardize:
            mu = x.mean(axis=0)
            sd = x.std(axis=0)
            sd[sd == 0] = 1.0
            x_fit = np.c_[np.ones(len(x)), (x - mu) / sd]
        else:
            mu, sd = None, None
            x_fit = x                      # caller provides intercept + scaling

        # convergence tracking (diagnostic; costs one extra pass per check)
        convergence = None
        if self.track_every > 0:
            convergence = []
            w = np.zeros(x_fit.shape[1])
            for it in range(1, self.max_iters + 1):
                g = self._grad(x_fit, y, w)
                w = w - self.gamma * g
                if it % self.track_every == 0:
                    convergence.append((it, float(imp.logistic_loss(y, x_fit, w))))
            w_final, loss = w, imp.logistic_loss(y, x_fit, w)
        else:
            w_final, loss = self._fit_impl(x_fit, y)
            loss = float(loss)

        self.result_ = FitResult(w_final, loss, self.max_iters, mu, sd,
                                 config, convergence)
        return self

    # internal: single solvers (graded functions where signatures allow)
    def _grad(self, x_with_intercept, y, w):
        if self.w_pos == 1.0:
            return imp.grad_logistic(y, x_with_intercept, w) \
                + 2 * self.lam * w
        row_w = np.where(y == 1, self.w_pos, 1.0)
        sig = sigmoid(x_with_intercept.dot(w))
        return x_with_intercept.T.dot(row_w * (sig - y)) / row_w.sum() \
            + 2 * self.lam * w

    def _fit_impl(self, x_fit, y):
        w0 = np.zeros(x_fit.shape[1])
        if self.w_pos == 1.0 and self.lam == 0.0:
            # standard (unregularised) logistic regression
            return imp.logistic_regression(
                y, x_fit, w0, self.max_iters, self.gamma)
        if self.w_pos == 1.0:
            # L2-regularised logistic regression
            return imp.reg_logistic_regression(
                y, x_fit, self.lam, w0, self.max_iters, self.gamma)
        # class-weighted (2*lam*w is exactly 0 when lam=0, so the
        # unregularised weighted model is the same function with lam=0)
        return imp.reg_logistic_regression_weighted(
            y, x_fit, self.lam, w0, self.max_iters, self.gamma, self.w_pos)

    # -------------------------------------------------------------- predict
    def predict_scores(self, x):
        """Probabilities P(y=1|x) for raw (no-intercept) rows."""
        self._check_fitted()
        return self.result_.predict_scores(np.asarray(x, dtype=float),
                                           standardize=self.standardize)

    def predict(self, x, threshold=0.5):
        """{0,1} labels at `threshold` (tune first via decision_threshold)."""
        return (self.predict_scores(x) > threshold).astype(int)

    # ------------------------------------------------------------- threshold
    def decision_threshold(self, x, y, grid=None, criterion="f1"):
        """Tune the decision threshold on (x, y) — call on TRAIN data.

        Returns the best threshold; also stores it as self.threshold_.
        grid: np.linspace(0.02, 0.98, 97) by default (covers inflated
        weighted-model scores).
        """
        self._check_fitted()
        p = self.predict_scores(x)
        if grid is None:
            grid = np.linspace(0.02, 0.98, 97)
        best_t, best = 0.5, -np.inf
        for t in grid:
            if criterion == "f1":
                score = ds.F1_score(y, p, t)[0]
            elif criterion == "accuracy":
                score = -np.mean((p > t) != (np.asarray(y) == 1))
            else:
                raise ValueError(f"unknown criterion {criterion!r}")
            if score > best:
                best_t, best = t, score
        self.threshold_ = float(best_t)
        return self.threshold_

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("call fit() first")


if __name__ == "__main__":
    # smoke test: synthetic data, model classes only (metrics in diagnostics.py)
    import diagnostics as dg
    rng = np.random.default_rng(0)
    n, d = 20000, 8
    X = rng.normal(size=(n, d))
    w_true = np.array([0.0, 0.9, -0.7, 0.0, 0.0, 0.5, 0.0, 0.05])
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w_true)))).astype(float)

    clf = LogisticClassifier(lam=1e-6, gamma=0.1, max_iters=800)
    clf.fit(X, y)
    p = clf.predict_scores(X)
    t = clf.decision_threshold(X, y)
    print("fit ok: w shape", clf.result_.w.shape,
          "| loss", round(clf.result_.loss, 4),
          "| t*", t)
    dg.Diagnostics.report(
        dg.Diagnostics.evaluate(y, p, metrics=["f1", "auc"], threshold=t),
        title="smoke test")

    folds_out = dg.Diagnostics.cross_validate(
        X, y, lambda: LogisticClassifier(lam=1e-6, max_iters=400), k=3,
        verbose=True)
