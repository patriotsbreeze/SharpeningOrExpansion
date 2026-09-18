"""Estimating k*, the crossover point -- and being honest when there isn't one.

For each bootstrap replicate we find the smallest k at which the base curve overtakes the RL
curve. Some replicates have no crossover within k <= n. Reporting a CI over only the
replicates that *did* cross would be a lie by selection, so ``p_exists`` is reported
alongside, and the CI is explicitly conditional on existence.
"""

from __future__ import annotations

import numpy as np


def crossover_k(curve_base: np.ndarray, curve_rl: np.ndarray, ks: tuple[int, ...]) -> int | None:
    """Smallest k where base >= rl. ``None`` when the base never catches up within k <= n."""
    for j, k in enumerate(ks):
        if curve_base[j] >= curve_rl[j]:
            return k
    return None


def bootstrap_crossover(
    counts_base: np.ndarray,
    counts_rl: np.ndarray,
    n: int,
    *,
    ks: tuple[int, ...],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    from soe.analysis.bootstrap import _curve_matrix

    usable = tuple(k for k in ks if k <= n)
    A = _curve_matrix(counts_base, n, usable)
    B = _curve_matrix(counts_rl, n, usable)
    P = A.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n_boot, P))
    ma, mb = A[idx].mean(axis=1), B[idx].mean(axis=1)

    found = []
    for r in range(n_boot):
        k = crossover_k(ma[r], mb[r], usable)
        if k is not None:
            found.append(k)

    point = crossover_k(A.mean(axis=0), B.mean(axis=0), usable)
    out = {
        "k_star": point,
        "p_exists": len(found) / n_boot,
        "n_boot": n_boot,
        "ks": usable,
    }
    if found:
        arr = np.array(found)
        out["ci_conditional"] = (
            float(np.quantile(arr, alpha / 2)),
            float(np.quantile(arr, 1 - alpha / 2)),
        )
        out["median_conditional"] = float(np.median(arr))
    else:
        out["ci_conditional"] = None
        out["median_conditional"] = None
    return out
