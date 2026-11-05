"""End-to-end experiment driver.

Usage (from the repository root):
    python -m src.run_experiments            # full study (~1-2 min on a laptop CPU)
    python -m src.run_experiments --quick    # 2 seeds, smoke test
    python -m src.run_experiments --synthetic  # force the synthetic dataset

Outputs (all in results/):
    metrics_raw.csv            one row per (model, epsilon, seed) run
    metrics_summary.csv        mean / std / 95% CI over seeds
    size_sweep_raw.csv         training-set-size experiment, per run
    size_sweep_summary.csv     ... aggregated
    run_info.json              dataset sizes, library versions, timings
    figures/*.png              produced by src/plots.py
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd

from . import config
from .data import load_dataset
from .evaluate import evaluate, summarise
from .models import HAVE_DIFFPRIVLIB, fit_quietly, make_model, scores


def run_one(kind, eps, seed, X_tr, y_tr, X_te, y_te, data_norm) -> dict:
    """Train one model and return a flat record of its metrics and DP metadata."""
    t0 = time.perf_counter()
    model = fit_quietly(make_model(kind, eps, seed, data_norm), X_tr, y_tr)
    rec = {"model": kind, "epsilon": eps, "seed": seed, "n_train": len(y_tr)}
    rec.update(evaluate(y_te, model.predict(X_te), scores(model, X_te)))
    rec["delta"] = (config.DELTA if kind == "dpsgd_logreg"
                    else 0.0 if kind == "objpert_logreg" else np.nan)
    rec["noise_multiplier"] = getattr(model, "sigma_", np.nan)
    rec["dpsgd_steps"] = getattr(model, "n_steps_", np.nan)
    rec["epsilon_spent"] = getattr(model, "spent_epsilon_", np.nan) if kind == "dpsgd_logreg" else eps
    rec["fit_seconds"] = time.perf_counter() - t0
    return rec


def main_sweep(ds, seeds, epsilons) -> pd.DataFrame:
    """Experiment 1: accuracy / F1 / AUC vs epsilon on the full training set."""
    args = (ds.X_train, ds.y_train, ds.X_test, ds.y_test, ds.max_row_norm)
    private = (["objpert_logreg"] if HAVE_DIFFPRIVLIB else []) + ["dpsgd_logreg"]
    rows = [run_one("majority", None, 0, *args),
            run_one("baseline_logreg", None, 0, *args)]      # deterministic: 1 run
    for seed in seeds:
        rows.append(run_one("sgd_nonprivate", None, seed, *args))
        rows.append(run_one("sgd_clip_only", None, seed, *args))
        for eps in epsilons:
            for kind in private:
                rows.append(run_one(kind, eps, seed, *args))
        print(f"  seed {seed} done")
    return pd.DataFrame(rows)


def size_sweep(ds, seeds) -> pd.DataFrame:
    """Experiment 2: how does the amount of training data change the trade-off?"""
    rows = []
    private = (["objpert_logreg"] if HAVE_DIFFPRIVLIB else []) + ["dpsgd_logreg"]
    n_full = len(ds.y_train)
    for n in config.SIZE_SWEEP_N:
        n = min(n, n_full)
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            idx = rng.choice(n_full, size=n, replace=False)   # random subset
            sub = (ds.X_train[idx], ds.y_train[idx], ds.X_test, ds.y_test, ds.max_row_norm)
            rows.append(run_one("baseline_logreg", None, seed, *sub))
            # Same SGD schedule without privacy: separates "too few steps at
            # small n" from "too much noise at small n".
            rows.append(run_one("sgd_nonprivate", None, seed, *sub))
            for eps in config.SIZE_SWEEP_EPSILONS:
                for kind in private:
                    rows.append(run_one(kind, eps, seed, *sub))
        print(f"  n={n} done")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="2 seeds, for smoke tests")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    t_start = time.perf_counter()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    ds = load_dataset(force_synthetic=a.synthetic)
    seeds = config.SEEDS[:2] if a.quick else config.SEEDS
    size_seeds = config.SIZE_SWEEP_SEEDS[:2] if a.quick else config.SIZE_SWEEP_SEEDS
    print(f"dataset={ds.source} train={ds.X_train.shape} test={ds.X_test.shape} "
          f"positives={ds.y_train.mean():.3f} data_norm={ds.max_row_norm:.3f} "
          f"diffprivlib={HAVE_DIFFPRIVLIB}")

    print("Experiment 1: epsilon sweep")
    raw = main_sweep(ds, seeds, config.EPSILONS)
    raw.to_csv(config.RESULTS_DIR / "metrics_raw.csv", index=False)
    summ = summarise(raw)
    extra = (raw.groupby(["model", "epsilon"], dropna=False)
             [["noise_multiplier", "dpsgd_steps", "delta"]].first().reset_index())
    summ = summ.merge(extra, on=["model", "epsilon"], how="left")
    summ.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False, float_format="%.5f")

    print("Experiment 2: training-set size sweep")
    sraw = size_sweep(ds, size_seeds)
    sraw.to_csv(config.RESULTS_DIR / "size_sweep_raw.csv", index=False)
    summarise(sraw, keys=("model", "epsilon", "n_train")).to_csv(
        config.RESULTS_DIR / "size_sweep_summary.csv", index=False, float_format="%.5f")

    import matplotlib, scipy, sklearn  # noqa: E401  (versions only)
    if HAVE_DIFFPRIVLIB:
        import diffprivlib
    info = {
        "dataset": ds.source, "n_train": int(len(ds.y_train)), "n_test": int(len(ds.y_test)),
        "n_features": int(ds.X_train.shape[1]), "positive_rate_train": float(ds.y_train.mean()),
        "data_norm_bound": ds.max_row_norm, "epsilons": config.EPSILONS, "delta": config.DELTA,
        "seeds": seeds, "dpsgd": config.DPSGD, "quick": a.quick,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "diffprivlib": diffprivlib.__version__ if HAVE_DIFFPRIVLIB else None,
        "matplotlib": matplotlib.__version__,
    }
    if not a.no_plots:
        from .plots import make_all_figures
        make_all_figures()
    info["total_seconds"] = round(time.perf_counter() - t_start, 1)
    (config.RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"done in {info['total_seconds']} s -> {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
