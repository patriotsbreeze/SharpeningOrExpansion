"""The generation boundary.

This module imports nothing but the standard library. That is load-bearing: it is what keeps
``vllm`` and ``torch`` out of the analysis and grading import graph, so the whole pipeline
stays testable on a CPU box with no GPU packages installed. ``tests/unit/test_import_hygiene.py``
asserts it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenRequest:
    request_id: str                          # == sample_uid
    prompt: str | list[dict[str, str]]       # completion text, or chat messages
    seed: int
    max_tokens: int                          # ALREADY clamped to the context budget by the caller
    temperature: float
    top_p: float
    top_k: int
    stop: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenSample:
    request_id: str
    text: str
    n_prompt_tokens: int
    n_completion_tokens: int
    finish_reason: str                       # "stop" | "length" | "abort"
    stop_str: str | None


class GenerationBackend(Protocol):
    def load(self, spec) -> None: ...
    @property
    def fingerprint(self) -> str: ...
    def count_prompt_tokens(self, payloads: Sequence[str | list[dict]]) -> list[int]: ...
    def generate(self, reqs: Sequence[GenRequest]) -> list[GenSample]: ...
    def close(self) -> None: ...
