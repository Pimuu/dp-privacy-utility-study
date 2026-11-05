"""Model factory: one place that knows how to build every model in the study.

Model families
--------------
majority            Always predicts the majority class (the "no learning" floor).
baseline_logreg     Non-private L2 logistic regression (scikit-learn, L-BFGS).
objpert_logreg      epsilon-DP logistic regression via *objective perturbation*
                    (Chaudhuri, Monteleoni & Sarwate, 2011) from IBM diffprivlib.
                    Pure epsilon-DP (delta = 0).
sgd_nonprivate      Our NumPy mini-batch SGD, no clipping / no noise. Matched
                    non-private reference for DP-SGD.
sgd_clip_only       Same as DP-SGD with per-example clipping but zero noise.
                    Ablation (not private) separating clipping bias from noise.
dpsgd_logreg        Our NumPy DP-SGD with RDP accounting: (epsilon, delta)-DP.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from . import config
from .dp_sgd import DPSGDLogisticRegression

try:  # diffprivlib is optional: DP-SGD (our own code) works without it.
    from diffprivlib.models import LogisticRegression as DPLogisticRegression
    HAVE_DIFFPRIVLIB = True
except Exception:  # ImportError, or incompatible scikit-learn version
    HAVE_DIFFPRIVLIB = False


def make_model(kind: str, epsilon: float | None = None, seed: int = 0,
               data_norm: float | None = None):
    """Build an unfitted model of the requested family."""
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "baseline_logreg":
        return LogisticRegression(C=config.LOGREG_C, max_iter=config.LOGREG_MAX_ITER)
    if kind == "objpert_logreg":
        if not HAVE_DIFFPRIVLIB:
            raise RuntimeError("diffprivlib not available")
        # data_norm MUST be a public bound; here it is the schema-derived bound.
        return DPLogisticRegression(epsilon=epsilon, data_norm=data_norm,
                                    C=config.LOGREG_C, max_iter=config.LOGREG_MAX_ITER,
                                    random_state=seed)
    if kind == "sgd_nonprivate":
        return DPSGDLogisticRegression(epsilon=None, random_state=seed, **config.DPSGD)
    if kind == "sgd_clip_only":
        return DPSGDLogisticRegression(epsilon=np.inf, random_state=seed, **config.DPSGD)
    if kind == "dpsgd_logreg":
        return DPSGDLogisticRegression(epsilon=epsilon, delta=config.DELTA,
                                       random_state=seed, **config.DPSGD)
    raise ValueError(f"unknown model kind: {kind}")


def fit_quietly(model, X, y):
    """Fit while silencing convergence chatter (high-noise fits rarely 'converge')."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y)


def scores(model, X) -> np.ndarray:
    """Continuous scores for ROC-AUC (probability of the positive class)."""
    return model.predict_proba(X)[:, 1]
