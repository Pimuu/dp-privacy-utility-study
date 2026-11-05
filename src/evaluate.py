"""Metrics and aggregation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Accuracy, F1 of the positive (>50K) class, and ROC-AUC.

    F1 complements accuracy on this imbalanced task (~24% positives): a model
    that collapses to "always <=50K" scores ~76% accuracy but F1 = 0.
    ROC-AUC is threshold-free; it is 0.5 for a constant scorer.
    """
    auc = roc_auc_score(y_true, y_score) if np.ptp(y_score) > 0 else 0.5
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def summarise(df: pd.DataFrame, keys=("model", "epsilon")) -> pd.DataFrame:
    """Mean, std and 95% normal-approx. CI half-width over seeds."""
    keys = list(keys)
    metrics = ["accuracy", "f1", "roc_auc"]
    g = df.groupby(keys, dropna=False)[metrics]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    for m in metrics:
        out[f"{m}_ci95"] = 1.96 * out[f"{m}_std"].fillna(0) / np.sqrt(out[f"{m}_count"])
        out = out.drop(columns=f"{m}_count")
    out["n_seeds"] = g.size()
    return out.reset_index()
