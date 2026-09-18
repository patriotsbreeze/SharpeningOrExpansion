"""CPU backend with *known* ground truth -- the most valuable test asset in the repo.

Each problem gets a latent solve probability ``p_i``, exposed via ``true_p``. Because the
truth is known analytically (``pass@k == 1-(1-p_i)**k``), the entire pipeline -- planner,
seeding, sharding, grading, estimator, bootstrap, decomposition -- can be validated end to end
on a laptop in seconds, before a single GPU-hour is spent.

It also injects the pathologies the real run will produce, so the downstream code is exercised
against them rather than discovering them at hour three:

* length truncation (``finish_reason="length"``, no boxed answer)
* lucky guesses: a correct boxed answer with no derivation -- this is what C4 must catch
* degenerate repetition, the thing that makes SymPy graders hang
* malformed and nested ``\\boxed``
* role-dependent lognormal token counts, so C5 and C6 have real signal
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from soe.gen.backend import GenRequest, GenSample


class MockBackend:
    def __init__(
        self,
        *,
        base_p_alpha: float = 0.8,
        base_p_beta: float = 2.0,
        rl_p_boost: float = 0.18,
        truncation_rate: float = 0.06,
        guess_rate: float = 0.04,
        degenerate_rate: float = 0.02,
        malformed_rate: float = 0.02,
        answers: dict[int, str] | None = None,
    ) -> None:
        self.base_p_alpha = base_p_alpha
        self.base_p_beta = base_p_beta
        self.rl_p_boost = rl_p_boost
        self.truncation_rate = truncation_rate
        self.guess_rate = guess_rate
        self.degenerate_rate = degenerate_rate
        self.malformed_rate = malformed_rate
        self.answers = answers or {}
        self._spec = None

    # -- backend protocol ------------------------------------------------------------
    def load(self, spec) -> None:
        self._spec = spec

    @property
    def fingerprint(self) -> str:
        return "mock|v1"

    def count_prompt_tokens(self, payloads: Sequence[str | list[dict]]) -> list[int]:
        out = []
        for p in payloads:
            text = p if isinstance(p, str) else " ".join(m["content"] for m in p)
            out.append(max(1, len(text) // 4))
        return out

    def close(self) -> None:
        self._spec = None

    # -- ground truth ----------------------------------------------------------------
    def true_p(self, problem_idx: int) -> float:
        """The latent solve probability. Deterministic given (model role, problem)."""
        role = getattr(self._spec, "role", "base")
        h = hashlib.sha256(f"p|{problem_idx}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        p = float(rng.beta(self.base_p_alpha, self.base_p_beta))
        if role != "base":
            # Sharpening-like: concentrate mass, but genuinely lift a few hard problems so the
            # "newly solvable" bucket is non-empty and the decomposition has something to find.
            p = min(0.98, p + self.rl_p_boost * (1.0 - p))
        return p

    # -- generation ------------------------------------------------------------------
    def generate(self, reqs: Sequence[GenRequest]) -> list[GenSample]:
        out = []
        for r in reqs:
            out.append(self._one(r))
        return out

    def _one(self, r: GenRequest) -> GenSample:
        rng = np.random.default_rng(r.seed)
        problem_idx = self._pidx(r)
        p = self.true_p(problem_idx)
        gold = self.answers.get(problem_idx, "42")
        role = getattr(self._spec, "role", "base")

        n_tok = int(np.clip(rng.lognormal(6.6 if role == "base" else 7.6, 0.6), 32, r.max_tokens * 4))
        truncated = rng.random() < self.truncation_rate or n_tok > r.max_tokens
        n_tok = min(n_tok, r.max_tokens)

        if truncated:
            body = "We begin. " + "Consider the quantity $x$. " * max(1, n_tok // 24)
            return GenSample(r.request_id, body[: n_tok * 4], self._pt(r), n_tok, "length", None)

        if rng.random() < self.degenerate_rate:
            body = "\\frac{" * 400 + "1" + "}" * 200  # unbalanced on purpose
            return GenSample(r.request_id, body, self._pt(r), n_tok, "stop", None)

        correct = rng.random() < p
        if rng.random() < self.guess_rate:
            # A lucky guess: right answer, no derivation. C4's null-gold calibration must see
            # these as false positives rather than capability.
            body = f"The answer is \\boxed{{{gold}}}."
            return GenSample(r.request_id, body, self._pt(r), n_tok, "stop", None)

        shown = gold if correct else str((int(gold) + int(rng.integers(1, 97))) % 1000
                                         if gold.isdigit() else rng.integers(0, 1000))
        steps = " ".join(f"Step {i}: we compute ${i}\\cdot 2={2*i}$." for i in range(1, 6))
        if rng.random() < self.malformed_rate:
            body = f"{steps} Therefore \\boxed{{\\boxed{{{shown}}}}}."
        else:
            body = f"{steps} Therefore the final answer is \\boxed{{{shown}}}."
        return GenSample(r.request_id, body, self._pt(r), n_tok, "stop", None)

    def _pidx(self, r: GenRequest) -> int:
        from soe.seeding import unpack

        return unpack(r.seed).problem_idx

    def _pt(self, r: GenRequest) -> int:
        text = r.prompt if isinstance(r.prompt, str) else " ".join(m["content"] for m in r.prompt)
        return max(1, len(text) // 4)
