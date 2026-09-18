"""Extraction is a treatment, not a detail -- so it gets a golden table."""

from __future__ import annotations

import numpy as np
import pytest

from soe.grade.extract import extract, extract_all, iter_boxed
from soe.grade.graders import FastIntGrader
from soe.grade.nullgold import correct_matrix, correct_rate, permuted_golds
from soe.grade.pool import clip_tail
from soe.grade.tokenpos import NO_ANSWER, answer_token_pos, correct_at_budget, find_answer_char

GOLDEN = [
    (r"so \boxed{399}.",                   "boxed_strict", "399"),
    (r"\boxed{\boxed{5}}",                 "boxed_strict", "5"),
    (r"\boxed{\boxed{\boxed{5}}}",         "boxed_strict", "5"),
    (r"\boxed{\frac{1}{2}}",               "boxed_strict", r"\frac{1}{2}"),
    (r"\boxed{ 007 }",                     "boxed_strict", "007"),
    (r"$\boxed{5}$",                       "boxed_strict", "5"),
    (r"\boxed{}",                          "boxed_strict", None),
    (r"truncated \boxed{12",               "boxed_strict", None),
    (r"\boxed{1} and \boxed{2}",           "boxed_strict", None),
    (r"\boxed{1} and \boxed{2}",           "boxed_last",   "2"),
    ("no box at all",                      "boxed_last",   None),
    ("first 7 then 42.",                   "last_int",     "42"),
    ("value is 1,234 total",               "last_int",     "1234"),
    ("no digits here",                     "last_int",     None),
    ("The final answer is 137.",           "answer_is",    "137"),
    ("the answer: 42",                     "answer_is",    "42"),
    (r"\boxed {88}",                       "boxed_strict", "88"),
]


@pytest.mark.parametrize("text,policy,expected", GOLDEN)
def test_extraction_golden_table(text, policy, expected):
    assert extract(text, policy) == expected


def test_nested_box_is_one_answer_not_two():
    assert iter_boxed(r"\boxed{\boxed{5}}") == [r"\boxed{5}"]
    assert len(iter_boxed(r"\boxed{1} x \boxed{2}")) == 2


def test_rambling_completion_gives_different_answers_per_policy():
    """boxed_strict and last_int disagree -- which is exactly why both are recorded.

    This is the false-positive channel null-gold calibration is designed to measure: an
    unboxed ramble still hands ``last_int`` a number, and against a 1000-value answer space
    that number is right about 0.1% of the time by accident, compounding over k=256.
    """
    text = "We try " + " ".join(str(i) for i in range(200)) + " and stop."
    ex = extract_all(text)
    assert ex["boxed_strict"] is None
    assert ex["last_int"] == "199"

    boxed_then_noise = r"hence \boxed{7}. For reference we also computed 199."
    ex2 = extract_all(boxed_then_noise)
    assert ex2["boxed_strict"] == "7"
    assert ex2["last_int"] == "199"
    assert ex2["boxed_strict"] != ex2["last_int"]


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="unknown extraction policy"):
        extract("x", "vibes")


def test_fastint_normalises_leading_zeros_and_commas():
    g = FastIntGrader()
    assert g.grade(gold="7", pred="007")
    assert g.grade(gold="1234", pred="1,234")
    assert not g.grade(gold="7", pred="8")
    assert not g.grade(gold="7", pred=r"\frac{1}{2}")


def test_permuted_golds_is_a_derangement():
    golds = [str(i) for i in range(20)]
    perm = permuted_golds(golds, seed=3)
    assert all(a != b for a, b in zip(golds, perm, strict=True))
    assert sorted(perm) == sorted(golds)


def test_permutation_is_deterministic():
    g = [str(i) for i in range(20)]
    assert permuted_golds(g, seed=1) == permuted_golds(g, seed=1)


def test_nullgold_correction_recovers_a_known_rate():
    p_true, phi = 0.30, 0.12
    observed = p_true + (1 - p_true) * phi
    assert correct_rate(observed, phi) == pytest.approx(p_true, abs=1e-9)


def test_nullgold_clips_below_the_floor():
    assert correct_rate(0.05, 0.12) == 0.0
    assert np.allclose(correct_matrix(np.array([0.0, 1.0]), 0.1), [0.0, 1.0])


def test_nullgold_rejects_degenerate_phi():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            correct_rate(0.5, bad)


def test_token_budget_ladder_is_monotone():
    text = "a" * 100 + "\\boxed{7}"
    pos = answer_token_pos(text, find_answer_char(text, "7"), 200)
    assert pos > 0
    seen = [correct_at_budget(True, pos, b) for b in range(0, 220, 10)]
    assert seen == sorted(seen), "correct_at_budget must be monotone non-decreasing in budget"


def test_no_answer_never_counts_at_any_budget():
    assert answer_token_pos("nothing here", None, 100) == NO_ANSWER
    assert not correct_at_budget(True, NO_ANSWER, 10**9)


def test_clip_tail_keeps_the_end():
    text = "x" * 10_000 + "ANSWER"
    assert clip_tail(text, 4096).endswith("ANSWER")
    assert len(clip_tail(text, 4096)) == 4096
