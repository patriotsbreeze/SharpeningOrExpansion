"""Shapley axioms. The additive identity is the paper's claim, so it is tested directly."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from soe.analysis.adjustments import AnalysisState
from soe.analysis.decompose import decompose, gap, weighted_pass_at_k


def _state(P=40, N=32, seed=0, **over):
    rng = np.random.default_rng(seed)
    pb = rng.beta(0.9, 2.0, size=P)
    pr = np.clip(pb + 0.12 * (1 - pb), 0, 0.98)
    base = dict(
        correct_base=rng.random((P, N)) < pb[:, None],
        correct_rl=rng.random((P, N)) < pr[:, None],
        tokens_base=rng.integers(200, 400, size=(P, N)),
        tokens_rl=rng.integers(400, 800, size=(P, N)),
        answer_pos_base=rng.integers(10, 300, size=(P, N)),
        answer_pos_rl=rng.integers(10, 700, size=(P, N)),
        weights=np.ones(P),
        release_dates=np.array([date(2026, 2, 5)] * P, dtype=object),
        k_base=N, k_rl=N,
        nullgold_base=np.zeros((P, N), bool),
        nullgold_rl=np.zeros((P, N), bool),
    )
    base.update(over)
    return AnalysisState(**base)


KW = {"C3": {"cutoff": date(2025, 12, 1)}}


def test_efficiency_identity_holds():
    d = decompose(_state(), kwargs=KW)
    assert d.raw_gap == pytest.approx(d.explained + d.residual_gap, abs=1e-12)
    d.check_efficiency(atol=1e-12)


def test_null_player_gets_exactly_zero():
    """C7 with identical shared-prompt matrices changes nothing, so it must score zero."""
    st = _state()
    st = type(st)(**{**st.__dict__,
                     "correct_base_shared_prompt": st.correct_base,
                     "correct_rl_shared_prompt": st.correct_rl})
    d = decompose(st, kwargs=KW)
    assert d.shapley["C7"] == pytest.approx(0.0, abs=1e-12)


def test_c3_is_a_null_player_when_no_problem_is_filtered():
    """Every problem postdates the cutoff, so contamination filtering is a no-op."""
    d = decompose(_state(), kwargs={"C3": {"cutoff": date(2020, 1, 1)}})
    assert d.shapley["C3"] == pytest.approx(0.0, abs=1e-12)


def test_injected_truncation_artifact_is_attributed_to_C5():
    """A gap caused purely by late RL answers must land on C5, not on the residual."""
    P, N = 60, 32
    rng = np.random.default_rng(5)
    # Solve rates sit inside C2's informative band, so no subset of the lattice empties the
    # problem set; the ONLY asymmetry between the arms is where the answer appears.
    hit = rng.random((P, N)) < 0.5
    st = _state(
        P=P, N=N,
        correct_base=hit, correct_rl=hit.copy(),
        answer_pos_base=np.full((P, N), 50),      # base answers early
        answer_pos_rl=np.full((P, N), 5000),      # RL answers far past the base budget
        tokens_base=np.full((P, N), 100),
        tokens_rl=np.full((P, N), 100),           # equal tokens, so C6 is inert
    )
    d = decompose(st, kwargs=KW)
    shares = {k: abs(v) for k, v in d.shapley.items()}
    assert max(shares, key=shares.get) == "C5", d.shapley
    d.check_efficiency(atol=1e-9)


def test_symmetry_two_identical_confounds_score_equally():
    """C4 and C5 configured to remove the same samples must receive equal credit."""
    P, N = 30, 16
    mask = np.zeros((P, N), bool)
    mask[:, ::2] = True
    st = _state(
        P=P, N=N,
        correct_base=np.ones((P, N), bool), correct_rl=np.ones((P, N), bool),
        nullgold_base=mask, nullgold_rl=np.zeros((P, N), bool),
        answer_pos_base=np.where(mask, 5000, 10), answer_pos_rl=np.full((P, N), 10),
        tokens_base=np.full((P, N), 100), tokens_rl=np.full((P, N), 100),
    )
    d = decompose(st, confounds=("C4", "C5"), kwargs={})
    assert d.shapley["C4"] == pytest.approx(d.shapley["C5"], abs=1e-9)


def test_empty_stratum_raises_instead_of_poisoning_the_lattice_with_nan():
    """An adjustment that removes every problem must fail loudly, not return NaN."""
    from soe.analysis.decompose import EmptyStratumError

    P, N = 20, 8
    st = _state(P=P, N=N,
                correct_base=np.ones((P, N), bool),   # p_base == 1.0, outside C2's band
                correct_rl=np.ones((P, N), bool))
    with pytest.raises(EmptyStratumError, match="emptied the problem set"):
        decompose(st, confounds=("C2",), kwargs={})


def test_weighted_passk_ignores_zero_weight_problems():
    P, N = 10, 8
    correct = np.zeros((P, N), bool)
    correct[0] = True
    w = np.zeros(P)
    w[0] = 1.0
    assert weighted_pass_at_k(correct, w, 4) == pytest.approx(1.0)


def test_gap_sign_convention():
    """Positive gap means base ahead -- the crossover regime."""
    P, N = 10, 8
    st = _state(P=P, N=N,
                correct_base=np.ones((P, N), bool),
                correct_rl=np.zeros((P, N), bool))
    assert gap(st) > 0
