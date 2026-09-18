"""Figure generation.

``text.usetex = False`` throughout: matplotlib's own mathtext renders ``$...$`` without a TeX
installation, so figures build in CI and on any node. Every figure is written as both PDF
(for the paper) and PNG (for quick inspection over SSH).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

BASE_C, RL_C = "#B45309", "#1D4ED8"


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out.with_suffix(".pdf")


def plot_passk_curves(boot: dict, out: Path, *, title: str = "", label_a="base", label_b="RLVR"):
    """The classic crossover plot, with the k=n point flagged as high-variance."""
    ks = list(boot["ks"])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(ks, [boot["curve_a"][k] for k in ks], "o-", color=BASE_C, label=label_a, ms=3.5)
    ax.plot(ks, [boot["curve_b"][k] for k in ks], "s-", color=RL_C, label=label_b, ms=3.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$k$")
    ax.set_ylabel("pass@$k$")
    ax.set_ylim(0, 1.02)
    if ks:
        # At k == n the unbiased estimator degenerates to "at least one correct": still
        # unbiased, but zero variance reduction. Readers should know which point that is.
        ax.axvline(ks[-1], color="0.7", lw=0.8, ls=":")
        ax.annotate("$k=n$", xy=(ks[-1], 0.04), fontsize=6.5, color="0.45", ha="right")
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, out)


def plot_gap_with_ci(boot: dict, out: Path, *, title: str = ""):
    ks = list(boot["ks"])
    g = np.array([boot["gap"][k] for k in ks])
    lo = np.array([boot["lo"][k] for k in ks])
    hi = np.array([boot["hi"][k] for k in ks])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.axhline(0, color="0.4", lw=0.9)
    ax.fill_between(ks, lo, hi, color=BASE_C, alpha=0.18, lw=0)
    ax.plot(ks, g, "o-", color=BASE_C, ms=3.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$k$")
    ax.set_ylabel("pass@$k$: base $-$ RLVR")
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, out)


def plot_shapley_waterfall(decomp, out: Path, *, ci: dict | None = None, title: str = ""):
    """The headline figure: how the raw gap decomposes, confound by confound.

    A true waterfall. The leftmost bar is the raw base-minus-RLVR gap; each confound bar is a
    step that removes its Shapley contribution; the rightmost bar is the unexplained residual,
    which is the estimate of genuine support shrinkage (C1). By Shapley efficiency the steps
    land exactly on the residual, so the figure is a visual proof of the additive identity.

    Error bars are drawn on each step's own magnitude at that step's position, not on the
    cumulative total -- a CI on a cumulative position would not be the quantity being estimated.
    """
    rows = [r for r in decomp.as_rows() if r["confound"] != "C1"]
    labels = [r["confound"] for r in rows]
    vals = [r["shapley"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    n = len(rows)
    xs = range(n + 2)

    # Anchor bar: the raw gap.
    ax.bar(0, decomp.raw_gap, color="0.55", width=0.62)

    running = decomp.raw_gap
    for i, v in enumerate(vals, start=1):
        bottom, top = running - v, running
        ax.bar(i, -v, bottom=running, width=0.62,
               color=BASE_C if v < 0 else RL_C, alpha=0.9)
        ax.plot([i - 0.31 - 0.18, i - 0.31], [top, top], color="0.6", lw=0.7)
        if ci and labels[i - 1] in ci:
            lo, hi = ci[labels[i - 1]]
            if np.isfinite(lo) and np.isfinite(hi):
                # The step ends at `bottom`; the CI on the step size maps to [running-hi, running-lo].
                ax.plot([i, i], [running - hi, running - lo], color="0.12", lw=1.1)
        running = bottom

    # Closing bar: the residual. Shapley efficiency means `running` already equals it.
    ax.bar(n + 1, decomp.residual_gap, color="0.3", width=0.62)
    if ci and "C1" in ci:
        lo, hi = ci["C1"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([n + 1, n + 1], [lo, hi], color="0.12", lw=1.1)

    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["raw\ngap", *labels, "C1\nresidual"], fontsize=7.5)
    ax.set_ylabel("contribution to base $-$ RLVR pass@$k$")
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, out)


def plot_hard_zero_2x2(tab, out: Path, *, title: str = ""):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    grid = np.array([[tab.neither, tab.expansion], [tab.lost, tab.both]], dtype=float)
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(1, grid.max()))
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{int(v)}", ha="center", va="center",
                color="white" if v > grid.max() * 0.55 else "0.15", fontsize=11)
    ax.set_xticks([0, 1], ["RLVR: never", "RLVR: solves"], fontsize=7.5)
    ax.set_yticks([0, 1], ["base: never", "base: solves"], fontsize=7.5)
    ax.set_title(title or "Hard-zero partition", fontsize=9)
    ax.grid(False)
    return _save(fig, out)


def plot_solve_rate_cdf(cdf_base: dict, cdf_rl: dict, out: Path, *, title: str = ""):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(cdf_base["grid"], cdf_base["cdf"], color=BASE_C, label="base")
    ax.plot(cdf_rl["grid"], cdf_rl["cdf"], color=RL_C, label="RLVR")
    ax.set_xlabel("per-problem solve rate $p_i$")
    ax.set_ylabel("empirical CDF")
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, out)


def plot_iso_compute(ks, tokens_base, tokens_rl, curve_base, curve_rl, out: Path, *, title=""):
    """Coverage against total sampled tokens rather than k -- the C6 view."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(np.array(ks) * tokens_base, curve_base, "o-", color=BASE_C, label="base", ms=3.5)
    ax.plot(np.array(ks) * tokens_rl, curve_rl, "s-", color=RL_C, label="RLVR", ms=3.5)
    ax.set_xscale("log")
    ax.set_xlabel("total sampled tokens")
    ax.set_ylabel("pass@(budget)")
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, out)
