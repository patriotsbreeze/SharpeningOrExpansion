"""Stable identifiers. Every id is a pure function of logical identity, never of wall-clock
time, worker index, or filesystem order -- so a resumed run reproduces exactly the same ids."""

from __future__ import annotations

import hashlib


def _h(*parts: object, n: int = 12) -> str:
    blob = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def sample_uid(
    exp_id: str, model_key: str, dataset_key: str, variant: str,
    sampling_id: str, problem_idx: int, sample_idx: int,
) -> str:
    return _h(exp_id, model_key, dataset_key, variant, sampling_id, problem_idx, sample_idx, n=16)


def unit_id(
    exp_id: str, model_key: str, dataset_key: str, variant: str,
    sampling_id: str, pshard: int, sample_lo: int, sample_hi: int,
) -> str:
    return _h(exp_id, model_key, dataset_key, variant, sampling_id, pshard, sample_lo, sample_hi)


def problem_uid(question: str) -> str:
    """Content hash of the normalized question. Detects upstream dataset drift."""
    from soe.data.canonical import normalize_question

    return hashlib.sha256(normalize_question(question).encode()).hexdigest()[:16]


def prompt_sha(rendered: str) -> str:
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]
