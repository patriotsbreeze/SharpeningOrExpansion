"""Replays a recorded corpus keyed by request_id.

Exists so that regression tests can run against *real* base-model pathologies -- the megabyte
unbalanced-LaTeX completions that break SymPy graders -- without weights, a GPU, or network.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from soe.gen.backend import GenRequest, GenSample


class ReplayBackend:
    def __init__(self, corpus_path: str | Path) -> None:
        self.corpus: dict[str, dict] = {}
        for line in Path(corpus_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                self.corpus[row["request_id"]] = row

    def load(self, spec) -> None:
        self._spec = spec

    @property
    def fingerprint(self) -> str:
        return "replay|v1"

    def count_prompt_tokens(self, payloads: Sequence[str | list[dict]]) -> list[int]:
        out = []
        for p in payloads:
            text = p if isinstance(p, str) else " ".join(m["content"] for m in p)
            out.append(max(1, len(text) // 4))
        return out

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenSample]:
        out = []
        for r in reqs:
            row = self.corpus.get(r.request_id)
            if row is None:
                raise KeyError(
                    f"replay corpus has no entry for request_id {r.request_id!r}; the corpus "
                    f"must be regenerated when the plan changes"
                )
            out.append(
                GenSample(
                    request_id=r.request_id, text=row["text"],
                    n_prompt_tokens=row.get("n_prompt_tokens", 1),
                    n_completion_tokens=row.get("n_completion_tokens", len(row["text"]) // 4),
                    finish_reason=row.get("finish_reason", "stop"),
                    stop_str=row.get("stop_str"),
                )
            )
        return out

    def close(self) -> None:
        self._spec = None
