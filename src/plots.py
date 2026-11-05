"""Figures for the paper. Reads the CSVs in results/ (so it can be re-run alone:
``python -m src.plots``) and writes PNGs to results/figures/.

Visual conventions: one hue per training algorithm, used consistently in every
figure (blue = objective perturbation / L-BFGS family, orange = DP-SGD / SGD
family); non-private references are neutral grey lines; shaded bands are 95% CIs
over seeds; epsilon is on a log axis because budgets span three orders of
magnitude.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"
LABEL = {"objpert_logreg": "Objective perturbation (diffprivlib, ε-DP)",
         "dpsgd_logreg": "DP-SGD (ours, (ε, δ=1e-5)-DP)"}
COLOR = {"objpert_logreg": BLUE, "dpsgd_logreg": ORANGE}
METRIC_NAME = {"accuracy": "Test accuracy", "f1": "F1 (>50K class)", "roc_auc": "ROC-AUC"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _load():
    s = pd.read_csv(config.RESULTS_DIR / "metrics_summary.csv")
    return s


def _ref(s, model, metric):
    row = s[s.model == model]
    return float(row[f"{metric}_mean"].iloc[0]) if len(row) else None


def fig_metric_vs_epsilon(s):
    """Fig. 1 - accuracy, F1 and AUC as a function of the privacy budget."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, metric in zip(axes, ["accuracy", "f1", "roc_auc"]):
        for kind in ["objpert_logreg", "dpsgd_logreg"]:
            d = s[s.model == kind].sort_values("epsilon")
            if d.empty:
                continue
            m, ci = d[f"{metric}_mean"], d[f"{metric}_ci95"]
            ax.fill_between(d.epsilon, m - ci, m + ci, color=COLOR[kind], alpha=0.15, lw=0)
            ax.plot(d.epsilon, m, "-o", color=COLOR[kind], ms=6,
                    mec=SURFACE, mew=1.5, label=LABEL[kind])
        base = _ref(s, "baseline_logreg", metric)
        ax.axhline(base, color=INK_2, ls="--", lw=1.5, label="Non-private logistic regression")
        ax.annotate(f"non-private {base:.3f}", (config.EPSILONS[0], base),
                    xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=8.5)
        clip = _ref(s, "sgd_clip_only", metric)
        if clip is not None:
            ax.axhline(clip, color=ORANGE, ls="-.", lw=1, alpha=0.8,
                       label="SGD with clipping only (no noise, not private)")
        maj = _ref(s, "majority", metric)
        if metric == "accuracy":
            ax.axhline(maj, color=MUTED, ls=":", lw=1.5, label="Majority class (always ≤50K)")
            ax.annotate(f"majority class {maj:.3f}", (config.EPSILONS[-1], maj),
                        xytext=(0, -12), textcoords="offset points", color=MUTED,
                        fontsize=8.5, ha="right")
        ax.set_xscale("log")
        ax.set_xticks(config.EPSILONS, [f"{e:g}" for e in config.EPSILONS])
        ax.set_xlabel("Privacy budget ε (log scale; smaller = more private)")
        ax.set_title(METRIC_NAME[metric], loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Privacy–utility trade-off on Adult (mean over 10 seeds, 95% CI bands)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(config.FIG_DIR / "fig1_metrics_vs_epsilon.png", dpi=200)
    plt.close(fig)


def fig_baseline_vs_private(s):
    """Fig. 2 - grouped bars: each algorithm family at three budgets vs non-private."""
    groups = [("ε = 0.1", 0.1), ("ε = 1", 1.0), ("ε = 10", 10.0), ("non-private", None)]
    nonpriv = {"objpert_logreg": "baseline_logreg", "dpsgd_logreg": "sgd_nonprivate"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(len(groups))
    w = 0.38
    for ax, metric in zip(axes, ["accuracy", "f1"]):
        for j, kind in enumerate(["objpert_logreg", "dpsgd_logreg"]):
            vals, errs = [], []
            for _, eps in groups:
                if eps is None:
                    r = s[s.model == nonpriv[kind]]
                else:
                    r = s[(s.model == kind) & np.isclose(s.epsilon, eps)]
                vals.append(float(r[f"{metric}_mean"].iloc[0]))
                errs.append(float(r[f"{metric}_ci95"].fillna(0).iloc[0]))
            xs = x + (j - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=COLOR[kind], yerr=errs, capsize=3,
                   error_kw=dict(ecolor=INK_2, lw=1), label=LABEL[kind].split(" (")[0]
                   + (" / L-BFGS" if j == 0 else " / plain SGD") + " family")
            for xi, v, e in zip(xs, vals, errs):
                ax.text(xi, v + e + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        if metric == "accuracy":
            maj = _ref(s, "majority", "accuracy")
            ax.axhline(maj, color=MUTED, ls=":", lw=1.5)
            ax.annotate("majority class", (1.5, maj), ha="center", va="center",
                        color=INK_2, fontsize=8.5, zorder=5,
                        bbox=dict(fc=SURFACE, ec="none", pad=1.5))
            ax.set_ylim(0.4, 0.92)
        else:
            ax.set_ylim(0, 0.75)
        ax.set_xticks(x, [g for g, _ in groups])
        ax.set_title(METRIC_NAME[metric], loc="left")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Non-private vs. differentially private logistic regression",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(config.FIG_DIR / "fig2_baseline_vs_private.png", dpi=200)
    plt.close(fig)


def fig_size_sweep():
    """Fig. 3 - accuracy vs training-set size at two fixed budgets."""
    p = config.RESULTS_DIR / "size_sweep_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    eps_list = config.SIZE_SWEEP_EPSILONS
    fig, axes = plt.subplots(1, len(eps_list), figsize=(11, 4), sharey=True,
                             constrained_layout=True)
    for ax, eps in zip(np.atleast_1d(axes), eps_list):
        b = s[s.model == "baseline_logreg"].sort_values("n_train")
        ax.plot(b.n_train, b.accuracy_mean, "--", color=INK_2, lw=1.5,
                label="Non-private logistic regression")
        g = s[s.model == "sgd_nonprivate"].sort_values("n_train")
        if not g.empty:
            ax.plot(g.n_train, g.accuracy_mean, ":", color=INK_2, lw=1.5,
                    label="Non-private SGD (same schedule as DP-SGD)")
        for kind in ["objpert_logreg", "dpsgd_logreg"]:
            d = s[(s.model == kind) & np.isclose(s.epsilon, eps)].sort_values("n_train")
            if d.empty:
                continue
            ax.fill_between(d.n_train, d.accuracy_mean - d.accuracy_ci95,
                            d.accuracy_mean + d.accuracy_ci95, color=COLOR[kind], alpha=0.15, lw=0)
            ax.plot(d.n_train, d.accuracy_mean, "-o", color=COLOR[kind], ms=6,
                    mec=SURFACE, mew=1.5, label=LABEL[kind])
        ax.set_xscale("log")
        ax.set_xticks(b.n_train, [f"{int(n):,}" for n in b.n_train])
        ax.set_xlabel("Training-set size n (log scale)")
        ax.set_title(f"ε = {eps:g}", loc="left")
    np.atleast_1d(axes)[0].set_ylabel("Test accuracy")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("More data makes privacy cheaper (mean over 5 seeds, 95% CI bands)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(config.FIG_DIR / "fig3_size_sweep.png", dpi=200)
    plt.close(fig)


def fig_noise_multiplier(s):
    """Fig. 4 - the DP-SGD noise multiplier our accountant calibrates per epsilon."""
    d = s[s.model == "dpsgd_logreg"].sort_values("epsilon")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    ax.plot(d.epsilon, d.noise_multiplier, "-o", color=ORANGE, ms=6, mec=SURFACE, mew=1.5)
    for e, sig in zip(d.epsilon, d.noise_multiplier):
        ax.annotate(f"{sig:.2f}", (e, sig), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(config.EPSILONS, [f"{e:g}" for e in config.EPSILONS])
    ax.set_xlabel("Target ε (δ = 1e-5)")
    ax.set_ylabel("Noise multiplier σ")
    steps = int(d.dpsgd_steps.iloc[0])
    ax.set_title(f"DP-SGD noise calibrated by the RDP accountant\n"
                 f"(q = {config.DPSGD['batch_size']}/n, T = {steps} steps)", loc="left")
    fig.savefig(config.FIG_DIR / "fig4_noise_multiplier.png", dpi=200)
    plt.close(fig)


def make_all_figures():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = _load()
    fig_metric_vs_epsilon(s)
    fig_baseline_vs_private(s)
    fig_size_sweep()
    fig_noise_multiplier(s)
    print(f"  figures written to {config.FIG_DIR}")


if __name__ == "__main__":
    make_all_figures()
