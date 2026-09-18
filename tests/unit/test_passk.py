"""Validation of the pass@k estimator: the identity, unbiasedness, references, and edges."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

from soe.analysis.passk import (
    naive_plugin,
    pass_at_k,
    pass_at_k_gammaln,
    pass_at_k_hypergeom,
    pass_at_k_vec,
)


def test_combinatorial_identity_exhaustive():
    """Claim (a): brute-force count of k-subsets containing >=1 of the first c indices."""
    for n in range(1, 13):
        for c in range(n + 1):
            for k in range(1, n + 1):
                subsets = itertools.combinations(range(n), k)
                hits = sum(1 for s in subsets if any(i < c for i in s))
                expected = hits / comb(n, k)
                assert pass_at_k(n, c, k) == pytest.approx(expected, abs=1e-12), (n, c, k)


def test_matches_hypergeom_and_gammaln():
    rng = np.random.default_rng(0)
    for _ in range(400):
        n = int(rng.integers(1, 300))
        c = int(rng.integers(0, n + 1))
        k = int(rng.integers(1, n + 1))
        v = pass_at_k(n, c, k)
        assert v == pytest.approx(pass_at_k_hypergeom(n, c, k), abs=1e-12)
        assert v == pytest.approx(pass_at_k_gammaln(n, c, k), abs=1e-10)


@pytest.mark.slow
def test_monte_carlo_subset_draws():
    """Claim (a) again, empirically: draw k of n without replacement."""
    rng = np.random.default_rng(12345)
    n, draws = 50, 200_000
    cells = [(c, k) for c in (1, 3, 7, 25) for k in (1, 5, 10, 25)]
    # Bonferroni over the grid: 4 sigma per cell is already ~1e-4 two-sided.
    for c, k in cells:
        correct = np.arange(c)
        hits = 0
        for _ in range(draws):
            s = rng.choice(n, size=k, replace=False)
            hits += bool(np.isin(s, correct).any())
        mc = hits / draws
        analytic = pass_at_k(n, c, k)
        tol = 4.0 * np.sqrt(max(analytic * (1 - analytic), 1e-9) / draws) + 1e-3
        assert abs(mc - analytic) < tol, (c, k, mc, analytic)


@pytest.mark.slow
def test_unbiasedness_and_that_naive_plugin_is_biased():
    """Claim (b): E_{c~Bin(n,p)}[estimator] == 1-(1-p)^k, and the plug-in fails the same test."""
    rng = np.random.default_rng(7)
    trials = 200_000
    naive_failed_somewhere = False

    for p in (0.01, 0.05, 0.2, 0.5):
        for n in (16, 64, 256):
            cs = rng.binomial(n, p, size=trials)
            for k in (1, 4, 16, 64):
                if k > n:
                    continue
                truth = 1.0 - (1.0 - p) ** k
                est = np.array([pass_at_k(n, int(c), k) for c in cs[:20_000]])
                se = est.std(ddof=1) / np.sqrt(est.size)
                assert abs(est.mean() - truth) < 4 * se + 2e-3, (p, n, k, est.mean(), truth)

                naive = np.array([naive_plugin(n, int(c), k) for c in cs[:20_000]])
                se_n = naive.std(ddof=1) / np.sqrt(naive.size)
                if abs(naive.mean() - truth) > 4 * se_n + 2e-3:
                    naive_failed_somewhere = True

    assert naive_failed_somewhere, (
        "the naive plug-in should be detectably biased somewhere on this grid; "
        "if it is not, the test grid is too easy to justify the unbiased estimator"
    )


def test_edges():
    assert pass_at_k(10, 0, 3) == 0.0
    assert pass_at_k(10, 10, 3) == 1.0
    assert pass_at_k(10, 1, 10) == 1.0
    assert pass_at_k(10, 8, 3) == 1.0  # n - c < k
    for bad in ((5, 2, 6), (5, 2, 0), (5, 6, 2), (5, -1, 2)):
        with pytest.raises(ValueError):
            pass_at_k(*bad)


def test_numerics_large_n():
    v = pass_at_k(100_000, 1, 1)
    assert v == pytest.approx(1e-5, rel=1e-9)
    assert np.isfinite(pass_at_k(100_000, 3, 1000))


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(3)
    n = np.full(64, 256)
    c = rng.integers(0, 257, size=64)
    for k in (1, 8, 64, 256):
        got = pass_at_k_vec(n, c, k)
        want = np.array([pass_at_k(256, int(ci), k) for ci in c])
        assert np.allclose(got, want, atol=1e-12)
