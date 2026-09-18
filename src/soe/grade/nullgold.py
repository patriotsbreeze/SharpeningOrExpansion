"""Null-gold permutation calibration -- the primary C4 measure.

Grade every completion a second time against a *permuted* gold answer (another problem's
answer, matched for answer type). The resulting hit rate is a direct, empirical
false-positive rate ``phi`` for each (model x extraction policy x grader), and

    p_corrected = (p_observed - phi) / (1 - phi)

This replaces the subjective "the CoT contains no derivation" heuristic with something
quantitative that costs zero extra generation. It matters most exactly where the paper's
claim lives: an 8k rambling base-model completion contains dozens of integers, AIME has only
1000 possible answers, and at k=256 a per-sample false-positive rate of even 1% compounds
into a visible pass@256 difference.

The permutation is deterministic given the problem manifest, so the calibration is
reproducible and can be recomputed from stored artifacts.
"""

from __future__ import annotations

import numpy as np


def permuted_golds(golds: list[str], seed: int = 0) -> list[str]:
    """A derangement of ``golds``: no problem keeps its own answer.

    Duplicate answer values across problems are left alone -- if two problems genuinely share
    an answer, a "false positive" there is a real ambiguity in the benchmark and should be
    counted as one.
    """
    n = len(golds)
    if n < 2:
        raise ValueError("need at least two problems to build a derangement")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return [golds[i] for i in perm]
    # Deterministic fallback: a single cycle is always a derangement for n >= 2.
    return [golds[(i + 1) % n] for i in range(n)]


def correct_rate(observed: float, phi: float) -> float:
    """Invert the two-component mixture p_obs = p_true*(1) + (1-p_true)*phi.

    Clipped to [0, 1]: an observed rate below the false-positive floor means the model is not
    measurably above chance, not that it has negative ability.
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must be in [0, 1), got {phi}")
    return float(np.clip((observed - phi) / (1.0 - phi), 0.0, 1.0))


def correct_matrix(observed: np.ndarray, phi: np.ndarray | float) -> np.ndarray:
    """Vectorised correction. ``phi`` broadcasts against ``observed``."""
    phi = np.asarray(phi, dtype=np.float64)
    if np.any((phi < 0) | (phi >= 1)):
        raise ValueError("phi must be in [0, 1)")
    return np.clip((np.asarray(observed, dtype=np.float64) - phi) / (1.0 - phi), 0.0, 1.0)
