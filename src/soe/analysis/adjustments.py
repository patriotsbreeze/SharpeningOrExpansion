"""The seven confounds, each as a uniform operator on a single analysis state.

Making every confound the same *type* of thing is what makes the decomposition tractable:
2^7 = 128 lattice evaluations per bootstrap replicate, each a few array operations on a
precomputed tensor. If any adjustment needed a re-grade or a re-sample, the decomposition
would be unaffordable.

    C1  support shrinkage        -- the residual; no operator
    C2  saturation/overtraining  -- problem-weight op
    C3  contamination            -- problem-weight op
    C4  lucky guessing           -- correctness op (null-gold correction)
    C5  truncation               -- correctness op (retro-truncate to the base's budget)
    C6  matched-k vs matched-compute -- comparison-rule op
    C7  prompt/template mismatch -- correctness swap to a shared variant
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np

CONFOUNDS = ("C2", "C3", "C4", "C5", "C6", "C7")  # C1 is the residual, never an operator

CONFOUND_LABELS = {
    "C2": "Saturation (overtraining)",
    "C3": "Contamination",
    "C4": "Lucky guessing",
    "C5": "Truncation",
    "C6": "Matched compute",
    "C7": "Prompt mismatch",
    "C1": "Support shrinkage (residual)",
}


@dataclass(frozen=True)
class AnalysisState:
    """Everything an adjustment may touch."""

    correct_base: np.ndarray      # [P, N] bool
    correct_rl: np.ndarray        # [P, N] bool
    tokens_base: np.ndarray       # [P, N] int
    tokens_rl: np.ndarray         # [P, N] int
    answer_pos_base: np.ndarray   # [P, N] int, -1 = no answer
    answer_pos_rl: np.ndarray
    weights: np.ndarray           # [P] float, problem weights (need not sum to 1)
    release_dates: np.ndarray     # [P] object(date)
    k_base: int                   # comparison budget for base
    k_rl: int                     # comparison budget for RL (differs only under C6)

    # Alternates, precomputed so adjustments stay pure array ops.
    nullgold_base: np.ndarray | None = None
    nullgold_rl: np.ndarray | None = None
    correct_base_shared_prompt: np.ndarray | None = None
    correct_rl_shared_prompt: np.ndarray | None = None

    @property
    def n_problems(self) -> int:
        return self.correct_base.shape[0]

    @property
    def n_samples(self) -> int:
        return self.correct_base.shape[1]


def apply_C2(st: AnalysisState, *, lo: float = 0.05, hi: float = 0.80) -> AnalysisState:
    """Drop problems already saturated (or hopeless) for the base model.

    2606.15455's claim is that the aggregate high-k decline is dominated by problems whose
    contribution to the metric has already saturated. Restricting to the informative band is
    the direct test of that.
    """
    p = st.correct_base.mean(axis=1)
    keep = (p >= lo) & (p <= hi)
    return replace(st, weights=st.weights * keep.astype(float))


def apply_C3(st: AnalysisState, *, cutoff: date) -> AnalysisState:
    """Keep only problems that postdate the cutoff (every checkpoint's public release)."""
    keep = np.array([d > cutoff for d in st.release_dates], dtype=float)
    return replace(st, weights=st.weights * keep)


def apply_C4(st: AnalysisState) -> AnalysisState:
    """Remove samples that the null-gold calibration marks as answer-matching false positives.

    A sample whose extracted answer also matches a *permuted* gold is evidence that the match
    channel fires on this completion regardless of content, so it should not count as a solve.
    """
    if st.nullgold_base is None or st.nullgold_rl is None:
        return st
    return replace(
        st,
        correct_base=st.correct_base & ~st.nullgold_base,
        correct_rl=st.correct_rl & ~st.nullgold_rl,
    )


def apply_C5(st: AnalysisState, *, budget: int | None = None) -> AnalysisState:
    """Retro-truncate both arms to a common generation budget.

    Free and exact: a sample counts only if its answer appeared within ``budget`` tokens. The
    default is the base arm's own observed ceiling, which is the comparison Yue et al.'s setup
    implicitly makes without controlling for it.
    """
    if budget is None:
        valid = st.answer_pos_base[st.answer_pos_base >= 0]
        budget = int(valid.max()) if valid.size else 0
    return replace(
        st,
        correct_base=st.correct_base & (st.answer_pos_base >= 0) & (st.answer_pos_base <= budget),
        correct_rl=st.correct_rl & (st.answer_pos_rl >= 0) & (st.answer_pos_rl <= budget),
    )


def apply_C6(st: AnalysisState, *, k_reference: int | None = None) -> AnalysisState:
    """Compare at equal total sampled tokens instead of equal k.

    Matched-k flatters whichever model emits shorter chains of thought. The RL budget becomes
    ``k_rl = max{k : k * E[tokens_rl] <= B}`` where ``B`` is the base arm's token spend at
    ``k_base``.
    """
    k_ref = k_reference or st.k_base
    mean_base = float(st.tokens_base.mean()) or 1.0
    mean_rl = float(st.tokens_rl.mean()) or 1.0
    budget = k_ref * mean_base
    k_rl = int(max(1, min(st.n_samples, np.floor(budget / mean_rl))))
    return replace(st, k_base=k_ref, k_rl=k_rl)


def apply_C7(st: AnalysisState) -> AnalysisState:
    """Put both arms on the shared prompt variant.

    Template choice dominates base-model behaviour, so a base model scored under a mismatched
    template against an RL model under its native one can manufacture the entire crossover.
    """
    if st.correct_base_shared_prompt is None or st.correct_rl_shared_prompt is None:
        return st
    return replace(
        st,
        correct_base=st.correct_base_shared_prompt,
        correct_rl=st.correct_rl_shared_prompt,
    )


ADJUSTMENTS = {
    "C2": apply_C2,
    "C3": apply_C3,
    "C4": apply_C4,
    "C5": apply_C5,
    "C6": apply_C6,
    "C7": apply_C7,
}


def apply_set(st: AnalysisState, names, *, kwargs: dict | None = None) -> AnalysisState:
    """Apply a set of adjustments.

    Order is fixed (weight ops, then correctness ops, then the comparison rule) so the lattice
    is order-independent and the Shapley values are well defined.
    """
    kwargs = kwargs or {}
    for name in ("C2", "C3", "C7", "C4", "C5", "C6"):
        if name in names:
            st = ADJUSTMENTS[name](st, **kwargs.get(name, {}))
    return st
