"""Cluster bootstrap over problems, and the pass@k curve machinery.

With ~60 problems in Tier A, between-problem variance dominates everything else, so the
primary CI resamples *problems* with replacement. A parametric within-problem check is
provided to demonstrate that sampling noise at n=256 is negligible by comparison.

Do not resample completions with replacement and then apply the without-replacement
estimator -- that combination is incoherent.
"""

from __future__ import annotations

import numpy as np

from soe.analysis.passk import pass_at_k

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def curve(counts: np.ndarray, n: int, ks: tuple[int, ...] = DEFAULT_KS) -> dict[int, float]:
    """Mean pass@k over problems. ``counts`` is ``c[i]`` for one model."""
    out = {}
    for k in ks:
        if k > n:
            continue
        out[k] = float(np.mean([pass_at_k(n, int(c), k) for c in counts]))
    return out


def _curve_matrix(counts: np.ndarray, n: int, ks: tuple[int, ...]) -> np.ndarray:
    """``[P, K]`` per-problem pass@k. Precomputed once, then bootstraps are cheap reindexes."""
    usable = [k for k in ks if k <= n]
    lut = np.zeros((n + 1, len(usable)))
    for c in range(n + 1):
        for j, k in enumerate(usable):
            lut[c, j] = pass_at_k(n, c, k)
    return lut[np.asarray(counts, dtype=int)]


def bootstrap_gap(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_a: int,
    n_b: int,
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired cluster bootstrap of pass@k_a - pass@k_b over problems.

    Paired: the same resampled problem indices are used for both arms, which is far tighter
    than comparing two marginal CIs and is the right test when both models see the same
    problem set.
    """
    if len(counts_a) != len(counts_b):
        raise ValueError("paired bootstrap requires the same problems in both arms")
    usable = tuple(k for k in ks if k <= min(n_a, n_b))
    A = _curve_matrix(counts_a, n_a, usable)
    B = _curve_matrix(counts_b, n_b, usable)
    P = A.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n_boot, P))
    reps = A[idx].mean(axis=1) - B[idx].mean(axis=1)     # [n_boot, K]
    point = A.mean(axis=0) - B.mean(axis=0)

    lo = np.quantile(reps, alpha / 2, axis=0)
    hi = np.quantile(reps, 1 - alpha / 2, axis=0)
    return {
        "ks": usable,
        "gap": dict(zip(usable, point.tolist(), strict=True)),
        "lo": dict(zip(usable, lo.tolist(), strict=True)),
        "hi": dict(zip(usable, hi.tolist(), strict=True)),
        "reps": reps,
        "curve_a": dict(zip(usable, A.mean(axis=0).tolist(), strict=True)),
        "curve_b": dict(zip(usable, B.mean(axis=0).tolist(), strict=True)),
    }


def bootstrap_mean(
    counts: np.ndarray, n: int, *, ks=DEFAULT_KS, n_boot=2000, seed=0, alpha=0.05
) -> dict:
    usable = tuple(k for k in ks if k <= n)
    C = _curve_matrix(counts, n, usable)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, C.shape[0], size=(n_boot, C.shape[0]))
    reps = C[idx].mean(axis=1)
    return {
        "ks": usable,
        "mean": dict(zip(usable, C.mean(axis=0).tolist(), strict=True)),
        "lo": dict(zip(usable, np.quantile(reps, alpha / 2, axis=0).tolist(), strict=True)),
        "hi": dict(zip(usable, np.quantile(reps, 1 - alpha / 2, axis=0).tolist(), strict=True)),
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval. Used for the lower bound on the expansion set.

    With n samples and zero successes the upper bound is ~3/n (the rule of three), which is
    why "never solved in n samples" supports a *bound* on the expansion set rather than a
    claim that the problem is unreachable at any k.
    """
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
