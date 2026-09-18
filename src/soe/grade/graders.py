"""Three graders, all recorded. Their disagreement rate is a *result*, not an inconvenience.

* ``FastIntGrader``  -- regex + integer compare. Handles the great majority of AIME-style
  answers in microseconds and needs no timeout machinery.
* ``QwenGrader``     -- vendored from QwenLM/Qwen2.5-Math. SimpleRL, Oat-Zero and PRIME all
  report numbers with it, so running it is what makes our numbers comparable to theirs.
* ``MathVerifyGrader`` -- HuggingFace math-verify.

``math_verify.verify`` is **not symmetric**: gold first, prediction second. Reversing the
arguments changes verdicts. The signature here is keyword-only so a call site cannot get it
wrong silently, and a test pins a pair that flips under swap.
"""

from __future__ import annotations

import re
from typing import Protocol

GRADERS = ("fastint", "qwen", "mathverify")


class Grader(Protocol):
    name: str

    def grade(self, *, gold: str, pred: str) -> bool: ...


def _as_int(s: str) -> int | None:
    s = s.strip().replace(",", "").replace("$", "").strip()
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s).strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    return None


class FastIntGrader:
    name = "fastint"

    def grade(self, *, gold: str, pred: str) -> bool:
        g, p = _as_int(gold), _as_int(pred)
        if g is None or p is None:
            return False
        return g == p


class MathVerifyGrader:
    name = "mathverify"

    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds

    def grade(self, *, gold: str, pred: str) -> bool:
        from math_verify import parse, verify

        gold_parsed = parse(f"${gold}$")
        pred_parsed = parse(pred)
        if not gold_parsed or not pred_parsed:
            return False
        # Argument order is load-bearing: verify(gold, target).
        return bool(verify(gold_parsed, pred_parsed))


class QwenGrader:
    name = "qwen"

    def grade(self, *, gold: str, pred: str) -> bool:
        try:
            from soe.grade.qwen_vendor.grader import math_equal
        except ImportError:
            # Vendoring is a deliberate, recorded step (see qwen_vendor/PROVENANCE.md).
            # Falling back silently to a different grader would corrupt the comparability
            # argument, so this is loud.
            raise RuntimeError(
                "Qwen grader not vendored. Run scripts/vendor_qwen_grader.sh on a machine "
                "with network access, or drop 'qwen' from the grading config deliberately."
            ) from None
        return bool(math_equal(pred, gold))


def make_grader(name: str) -> Grader:
    if name == "fastint":
        return FastIntGrader()
    if name == "mathverify":
        return MathVerifyGrader()
    if name == "qwen":
        return QwenGrader()
    raise ValueError(f"unknown grader {name!r}; expected one of {GRADERS}")


def available_graders() -> list[str]:
    """Which graders can actually run here. Used by the smoke path, never by a real run."""
    out = ["fastint"]
    try:
        import math_verify  # noqa: F401

        out.append("mathverify")
    except ImportError:
        pass
    try:
        from soe.grade.qwen_vendor import grader  # noqa: F401

        out.append("qwen")
    except ImportError:
        pass
    return out
