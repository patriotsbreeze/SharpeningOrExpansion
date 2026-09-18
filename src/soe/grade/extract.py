"""Answer extraction policies.

Extraction leniency is a **treatment, not a detail**. ``last_int`` applied to an 8k rambling
base-model completion containing 200 integers, against an answer space of only 1000 values,
has a large false-positive rate. ``boxed_strict`` and ``last_int`` can give different *signs*
for the crossover on the base model -- which is a finding, but only if it is measured. So
every policy is computed for every completion (it is cheap) and the null-gold false-positive
rate is reported per (model x policy).
"""

from __future__ import annotations

import re

POLICIES = ("boxed_strict", "boxed_last", "last_int", "answer_is")

_BOXED = re.compile(r"\\boxed\s*\{")
_INT = re.compile(r"-?\d[\d,]*")
_ANSWER_IS = re.compile(
    r"(?:final answer|answer)\s*(?:is|:)\s*\$?\\?\(?\s*(-?[\d,]+|[^\s.$]{1,40})", re.IGNORECASE
)


def _balanced_after(text: str, open_idx: int) -> str | None:
    """Return the contents of a brace group starting at ``open_idx`` (the '{')."""
    depth, out = 0, []
    for ch in text[open_idx:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        if depth >= 1:
            out.append(ch)
    return None  # unbalanced -- truncated mid-\boxed{


def iter_boxed(text: str) -> list[str]:
    """Contents of each **outermost** balanced ``\\boxed{...}``, in order of appearance.

    Nested boxes are one answer, not two. Counting ``\\boxed{\\boxed{5}}`` as two would make
    ``boxed_strict`` reject it as ambiguous, turning a formatting quirk into a scored miss --
    and models emit this often enough for that to move pass@k.
    """
    out: list[str] = []
    consumed_to = -1
    for m in _BOXED.finditer(text):
        if m.start() < consumed_to:
            continue  # inside a box we already took
        brace = text.find("{", m.start())
        if brace == -1:
            continue
        content = _balanced_after(text, brace)
        if content is None:
            continue  # unbalanced: truncated mid-\boxed{
        out.append(content.strip())
        consumed_to = brace + len(content) + 2
    return out


def _unwrap_nested(s: str) -> str:
    """``\\boxed{\\boxed{5}}`` -> ``5``. Models do this; it should not count as a miss."""
    for _ in range(4):
        inner = iter_boxed(s)
        if len(inner) == 1 and _BOXED.match(s.strip()):
            s = inner[0]
        else:
            break
    return s.strip()


def extract(text: str, policy: str) -> str | None:
    if policy not in POLICIES:
        raise ValueError(f"unknown extraction policy {policy!r}; expected one of {POLICIES}")

    if policy in ("boxed_strict", "boxed_last"):
        boxes = iter_boxed(text)
        if not boxes:
            return None
        if policy == "boxed_strict" and len(boxes) > 1:
            # Strict: ambiguity is a miss, not a guess.
            return None
        return _unwrap_nested(boxes[-1]) or None

    if policy == "last_int":
        ints = _INT.findall(text)
        if not ints:
            return None
        return ints[-1].replace(",", "")

    if policy == "answer_is":
        matches = _ANSWER_IS.findall(text)
        if not matches:
            return None
        # Last statement wins: models often restate the answer after a correction.
        return matches[-1].replace(",", "").strip() or None

    return None


def extract_all(text: str) -> dict[str, str | None]:
    return {p: extract(text, p) for p in POLICIES}
