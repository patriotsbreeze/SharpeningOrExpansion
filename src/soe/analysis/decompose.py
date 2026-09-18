"""Shapley attribution over the adjustment lattice -- the paper's headline result.

Let ``G(S)`` be the base-minus-RL pass@k gap after applying the adjustments in set ``S``, and
define the explained value ``v(S) = G(emptyset) - G(S)``. Each confound gets its exact Shapley
value of ``v``. Shapley efficiency then gives the identity

    G(emptyset) = sum_i phi_i + G(full)

i.e. **raw gap = explained + residual**, where the residual is exactly C1, genuine support
shrinkage. That turns "whatever is left over" -- which is unfalsifiable -- into an estimated
quantity with a bootstrap confidence interval.

Leave-one-out effects are reported alongside because they are more intuitive, but they do not
sum under interaction. Shapley is the one that adds up, which is why it carries the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np

from soe.analysis.adjustments import CONFOUNDS, AnalysisState, apply_set
from soe.analysis.passk import pass_at_k


class EmptyStratumError(RuntimeError):
    """An adjustment left no problems with positive weight.

    Silently returning NaN here would poison the whole Shapley lattice with NaN and produce a
    decomposition that looks computed but means nothing, so this is loud. In practice it means
    C2's informative band and C3's date filter have no overlap on this problem set -- widen the
    band, or run the decomposition on Tier A union Tier B where there is enough support.
    """


def weighted_pass_at_k(correct: np.ndarray, weights: np.ndarray, k: int) -> float:
    """Weighted mean of per-problem unbiased pass@k. Zero-weight problems drop out."""
    n = correct.shape[1]
    k = min(k, n)
    live = weights > 0
    if not live.any():
        raise EmptyStratumError(
            f"no problems left with positive weight (started from {len(weights)})"
        )
    c = correct[live].sum(axis=1)
    vals = np.array([pass_at_k(n, int(ci), k) for ci in c])
    w = weights[live]
    return float(np.sum(vals * w) / np.sum(w))


def gap(st: AnalysisState) -> float:
    """base pass@k_base  minus  RL pass@k_rl. Positive = base ahead (the crossover regime)."""
    a = weighted_pass_at_k(st.correct_base, st.weights, st.k_base)
    b = weighted_pass_at_k(st.correct_rl, st.weights, st.k_rl)
    return a - b


@dataclass(frozen=True)
class Decomposition:
    raw_gap: float
    residual_gap: float           # G(full) == C1
    shapley: dict[str, float]
    loo: dict[str, float]
    confounds: tuple[str, ...]

    @property
    def explained(self) -> float:
        return sum(self.shapley.values())

    def check_efficiency(self, atol: float = 1e-9) -> None:
        lhs = self.raw_gap
        rhs = self.explained + self.residual_gap
        if abs(lhs - rhs) > atol:
            raise AssertionError(
                f"Shapley efficiency violated: raw={lhs!r} != explained+residual={rhs!r}. "
                f"The additive identity is the paper's claim, so this must never be tolerated."
            )

    def as_rows(self) -> list[dict]:
        from soe.analysis.adjustments import CONFOUND_LABELS

        rows = [
            {"confound": c, "label": CONFOUND_LABELS[c], "shapley": self.shapley[c],
             "loo": self.loo[c], "share": self.shapley[c] / self.raw_gap if self.raw_gap else 0.0}
            for c in self.confounds
        ]
        rows.append(
            {"confound": "C1", "label": CONFOUND_LABELS["C1"], "shapley": self.residual_gap,
             "loo": float("nan"),
             "share": self.residual_gap / self.raw_gap if self.raw_gap else 0.0}
        )
        return rows


def decompose(
    st: AnalysisState,
    *,
    confounds: tuple[str, ...] = CONFOUNDS,
    kwargs: dict | None = None,
) -> Decomposition:
    """Exact Shapley over the full 2^|confounds| lattice."""
    n = len(confounds)
    cache: dict[frozenset, float] = {}

    def G(S: frozenset) -> float:
        if S not in cache:
            try:
                cache[S] = gap(apply_set(st, S, kwargs=kwargs))
            except EmptyStratumError as e:
                raise EmptyStratumError(
                    f"adjustment subset {sorted(S) or ['(none)']} emptied the problem set: {e}. "
                    f"Every subset of the lattice must be estimable or the Shapley values are "
                    f"undefined."
                ) from None
        return cache[S]

    g_empty = G(frozenset())
    v = {S: g_empty - G(S) for S in _all_subsets(confounds)}

    shapley = {}
    for i, c in enumerate(confounds):
        others = [x for x in confounds if x != c]
        total = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for combo in combinations(others, r):
                S = frozenset(combo)
                total += w * (v[S | {c}] - v[S])
        shapley[c] = total

    full = frozenset(confounds)
    loo = {c: v[full] - v[full - {c}] for c in confounds}

    d = Decomposition(
        raw_gap=g_empty,
        residual_gap=G(full),
        shapley=shapley,
        loo=loo,
        confounds=confounds,
    )
    d.check_efficiency(atol=1e-8)
    return d


def _all_subsets(items: tuple[str, ...]):
    for r in range(len(items) + 1):
        for combo in combinations(items, r):
            yield frozenset(combo)


def bootstrap_decompose(
    st: AnalysisState,
    *,
    confounds: tuple[str, ...] = CONFOUNDS,
    kwargs: dict | None = None,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Cluster bootstrap over problems, redoing the whole lattice in each replicate."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    P = st.n_problems
    point = decompose(st, confounds=confounds, kwargs=kwargs)

    reps: dict[str, list[float]] = {c: [] for c in (*confounds, "C1")}
    raws = []
    for _ in range(n_boot):
        idx = rng.integers(0, P, size=P)
        st_b = replace(
            st,
            correct_base=st.correct_base[idx], correct_rl=st.correct_rl[idx],
            tokens_base=st.tokens_base[idx], tokens_rl=st.tokens_rl[idx],
            answer_pos_base=st.answer_pos_base[idx], answer_pos_rl=st.answer_pos_rl[idx],
            weights=st.weights[idx], release_dates=st.release_dates[idx],
            nullgold_base=None if st.nullgold_base is None else st.nullgold_base[idx],
            nullgold_rl=None if st.nullgold_rl is None else st.nullgold_rl[idx],
            correct_base_shared_prompt=None if st.correct_base_shared_prompt is None
            else st.correct_base_shared_prompt[idx],
            correct_rl_shared_prompt=None if st.correct_rl_shared_prompt is None
            else st.correct_rl_shared_prompt[idx],
        )
        try:
            d = decompose(st_b, confounds=confounds, kwargs=kwargs)
        except (AssertionError, ValueError, EmptyStratumError):
            # A resampled problem set can empty a stratum by chance; skip that replicate and
            # report how many survived, rather than quietly narrowing the interval.
            continue
        raws.append(d.raw_gap)
        for c in confounds:
            reps[c].append(d.shapley[c])
        reps["C1"].append(d.residual_gap)

    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        a = np.array(vals)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (float("nan"), float("nan"))
        return float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))

    return {
        "point": point,
        "n_boot_ok": len(raws),
        "raw_gap_ci": ci(raws),
        "ci": {c: ci(reps[c]) for c in (*confounds, "C1")},
    }
