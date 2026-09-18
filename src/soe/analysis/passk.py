"""Unbiased pass@k, plus two independent reference implementations used by the tests.

There are two *distinct* claims about ``1 - C(n-c, k) / C(n, k)`` and conflating them is the
most common silent error in pass@k papers:

**(a) The combinatorial identity.** For a fixed set of ``n`` samples of which ``c`` are
correct, the expression equals the exact fraction of ``k``-subsets containing at least one
correct sample. This is checked by exhaustive enumeration.

**(b) Unbiasedness.** With ``c ~ Binomial(n, p)``, the *expectation* of the estimator equals
``1 - (1-p)**k``. This needs Monte Carlo over ``c`` and is the property that actually justifies
reporting the estimator. The naive plug-in ``1 - (1 - c/n)**k`` satisfies (a) trivially but
fails (b) badly at small ``n*p`` -- which is exactly the large-k tail where the
sharpening-vs-expansion claim lives.

``pass_at_k`` below uses the numerically stable product form (Chen et al., 2021), which never
forms a factorial and is exact to ~1e-15 even at n=100_000.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pass_at_k", "pass_at_k_vec", "pass_at_k_hypergeom", "pass_at_k_gammaln", "naive_plugin"]


def _validate(n: int, c: int, k: int) -> None:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must satisfy 0 <= c <= n, got c={c}, n={n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"pass@k requires n >= k; got n={n}, k={k}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k from ``c`` correct out of ``n`` i.i.d. samples.

    The samples must be i.i.d. draws from the same model/prompt/temperature. Duplicate
    completions produced by seed reuse violate that and bias the result -- see ``soe.verify``.
    """
    _validate(n, c, k)
    if n - c < k:
        # Every k-subset must contain a correct sample.
        return 1.0
    # prod_{i=n-c+1}^{n} (1 - k/i) == C(n-c, k) / C(n, k), computed without factorials.
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64)))


def pass_at_k_vec(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Vectorised over problems. ``n`` and ``c`` are integer arrays of equal shape."""
    n = np.asarray(n, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if n.shape != c.shape:
        raise ValueError(f"n and c must have the same shape, got {n.shape} and {c.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if np.any(k > n):
        raise ValueError(f"pass@k requires n >= k for every problem; k={k}, min n={n.min()}")
    if np.any((c < 0) | (c > n)):
        raise ValueError("c must satisfy 0 <= c <= n elementwise")

    out = np.ones(n.shape, dtype=np.float64)
    # Only entries with n - c >= k need the product; the rest are exactly 1.0.
    idx = np.flatnonzero((n - c) >= k)
    for j in idx:
        out.flat[j] = pass_at_k(int(n.flat[j]), int(c.flat[j]), k)
    return out


def pass_at_k_hypergeom(n: int, c: int, k: int) -> float:
    """Reference: 1 - P(draw zero correct) under a hypergeometric draw. Tests only."""
    from scipy.stats import hypergeom

    _validate(n, c, k)
    return float(1.0 - hypergeom.pmf(0, M=n, n=c, N=k))


def pass_at_k_gammaln(n: int, c: int, k: int) -> float:
    """Reference: log-gamma form of C(n-c, k) / C(n, k). Tests only."""
    from scipy.special import gammaln

    _validate(n, c, k)
    if n - c < k:
        return 1.0
    log_ratio = (
        gammaln(n - c + 1) - gammaln(n - c - k + 1) - gammaln(n + 1) + gammaln(n - k + 1)
    )
    return float(1.0 - np.exp(log_ratio))


def naive_plugin(n: int, c: int, k: int) -> float:
    """The biased estimator the literature sometimes uses. Kept so a test can prove it biased."""
    _validate(n, c, k)
    return float(1.0 - (1.0 - c / n) ** k)
