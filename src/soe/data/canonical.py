"""Canonical problem representation and text normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

_WS = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Normalization used for the content hash only -- never for what the model sees.

    Stable under whitespace reflow and Unicode form so that a cosmetic upstream re-push does
    not look like problem drift, while a genuine edit does.
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("−", "-").replace(" ", " ")
    return _WS.sub(" ", t).strip()


def normalize_answer(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("−", "-").replace(" ", " ")
    t = t.strip().strip("$").strip()
    return _WS.sub(" ", t)


@dataclass(frozen=True, slots=True)
class Problem:
    problem_idx: int          # position in the FROZEN manifest; feeds the seed
    problem_uid: str          # content hash; detects upstream drift
    dataset_key: str
    question: str             # verbatim, as shown to the model
    answer: str               # verbatim gold
    release_date: date
    answer_type: str

    def to_manifest_row(self) -> dict:
        return {
            "problem_idx": self.problem_idx,
            "problem_uid": self.problem_uid,
            "dataset_key": self.dataset_key,
            "question": self.question,
            "answer": self.answer,
            "release_date": self.release_date.isoformat(),
            "answer_type": self.answer_type,
        }

    @staticmethod
    def from_manifest_row(row: dict) -> Problem:
        return Problem(
            problem_idx=row["problem_idx"],
            problem_uid=row["problem_uid"],
            dataset_key=row["dataset_key"],
            question=row["question"],
            answer=row["answer"],
            release_date=date.fromisoformat(row["release_date"]),
            answer_type=row["answer_type"],
        )
