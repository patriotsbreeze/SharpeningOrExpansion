"""Where in the completion the answer appears -- the substrate for the C5 budget ladder.

Recording ``answer_token_pos`` once at grading time makes the entire truncation analysis free
and exact afterwards: ``correct_at_budget(T) == correct AND 0 <= answer_token_pos <= T``, which
is monotone in T and needs no regeneration. This is why retro-truncation is the *primary* C5
control and the RoPE-extended base arm is only a robustness appendix -- one costs nothing and
is exact, the other costs GPU-hours and introduces an extrapolation confound.
"""

from __future__ import annotations

NO_ANSWER = -1


def answer_token_pos(
    text: str, answer_span_start: int | None, n_completion_tokens: int
) -> int:
    """Approximate token index at which the extracted answer begins.

    Uses a character-fraction estimate rather than a real tokenizer: the budget ladder only
    needs monotonicity and rough calibration, and requiring a tokenizer here would drag
    ``transformers`` into the analysis import graph. ``grade/runner.py`` can override this
    with exact offsets when a fast tokenizer is available on the node.
    """
    if answer_span_start is None or not text:
        return NO_ANSWER
    frac = min(1.0, max(0.0, answer_span_start / len(text)))
    return int(round(frac * n_completion_tokens))


def find_answer_char(text: str, extracted: str | None) -> int | None:
    """Character offset where the reported answer appears (last occurrence)."""
    if not extracted:
        return None
    idx = text.rfind(extracted)
    return idx if idx >= 0 else None


def correct_at_budget(correct: bool, pos: int, budget: int) -> bool:
    """Would this sample still have been correct under a ``budget``-token generation cap?"""
    return bool(correct) and pos != NO_ANSWER and pos <= budget
