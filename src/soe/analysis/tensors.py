"""Graded parquet -> dense correctness tensors, the substrate for everything downstream.

Every confound adjustment must be a tensor operation. If any of them required re-grading,
the 2^7 Shapley lattice x 2000 bootstrap replicates would be unaffordable and the
decomposition -- the paper's actual contribution -- would get cut.

The ragged-n rule is enforced here. The pass@k estimator is per-problem unbiased for any
n_i, but a *mean across problems with unequal n_i* weights problems by their differing
estimator variances and is not comparable across arms. Spot preemption produces exactly that
situation, so we assert equal n and otherwise subsample deterministically by lowest
sample_idx, logging the truncation rather than silently pooling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrectnessTensor:
    """``correct[m, i, s]`` for model m, problem i, sample s."""

    model_keys: tuple[str, ...]
    problem_idxs: tuple[int, ...]
    correct: np.ndarray          # bool  [M, P, N]
    tokens: np.ndarray           # int32 [M, P, N]
    answer_pos: np.ndarray       # int32 [M, P, N], -1 where no answer was found
    nullgold: np.ndarray         # bool  [M, P, N]
    dataset_of: tuple[str, ...]  # per-problem dataset key
    n_samples: int
    truncated_from: dict[str, int] | None = None

    @property
    def n_models(self) -> int:
        return len(self.model_keys)

    @property
    def n_problems(self) -> int:
        return len(self.problem_idxs)

    def model(self, key: str) -> int:
        try:
            return self.model_keys.index(key)
        except ValueError:
            raise KeyError(f"{key!r} not in tensor; have {self.model_keys}") from None

    def counts(self) -> np.ndarray:
        """``c[m, i]`` -- correct samples per (model, problem)."""
        return self.correct.sum(axis=2)

    def solve_rate(self, model_key: str) -> np.ndarray:
        return self.correct[self.model(model_key)].mean(axis=1)

    def with_correct(self, new_correct: np.ndarray) -> CorrectnessTensor:
        if new_correct.shape != self.correct.shape:
            raise ValueError(f"shape {new_correct.shape} != {self.correct.shape}")
        return replace(self, correct=new_correct)


def build_tensor(
    df: pd.DataFrame,
    *,
    grader: str,
    policy: str,
    model_keys: list[str] | None = None,
    strict_equal_n: bool = False,
) -> CorrectnessTensor:
    ccol, ncol = f"correct__{grader}__{policy}", f"nullgold__{grader}__{policy}"
    for col in (ccol, ncol):
        if col not in df.columns:
            raise KeyError(f"{col} not in graded frame; have {sorted(df.columns)[:12]}...")

    models = tuple(model_keys or sorted(df["model_key"].unique()))
    problems = tuple(sorted(df["problem_idx"].unique()))

    # Per (model, problem) sample counts -- the ragged-n check.
    per = df.groupby(["model_key", "problem_idx"])["sample_idx"].nunique()
    n_min, n_max = int(per.min()), int(per.max())
    truncated_from = None
    if n_min != n_max:
        msg = (
            f"ragged n: {n_min}..{n_max} samples per problem. Averaging pass@k across problems "
            f"with unequal n weights them by differing estimator variance."
        )
        if strict_equal_n:
            raise ValueError(msg + " Re-run the missing chunks, or pass strict_equal_n=False.")
        truncated_from = {"n_min": n_min, "n_max": n_max}

    n = n_min
    M, P = len(models), len(problems)
    correct = np.zeros((M, P, n), dtype=bool)
    tokens = np.zeros((M, P, n), dtype=np.int32)
    apos = np.full((M, P, n), -1, dtype=np.int32)
    nullg = np.zeros((M, P, n), dtype=bool)
    dataset_of = ["" for _ in range(P)]

    mi = {k: i for i, k in enumerate(models)}
    pi = {p: i for i, p in enumerate(problems)}

    for (mk, pidx), g in df.groupby(["model_key", "problem_idx"], sort=False):
        if mk not in mi:
            continue
        # Deterministic truncation to the lowest n sample_idx values.
        g = g.sort_values("sample_idx").head(n)
        a, b = mi[mk], pi[pidx]
        correct[a, b, : len(g)] = g[ccol].to_numpy(dtype=bool)
        tokens[a, b, : len(g)] = g["n_completion_tokens"].to_numpy(dtype=np.int32)
        apos[a, b, : len(g)] = g["answer_token_pos"].to_numpy(dtype=np.int32)
        nullg[a, b, : len(g)] = g[ncol].to_numpy(dtype=bool)
        dataset_of[b] = str(g["dataset_key"].iloc[0])

    return CorrectnessTensor(
        model_keys=models, problem_idxs=problems, correct=correct, tokens=tokens,
        answer_pos=apos, nullgold=nullg, dataset_of=tuple(dataset_of), n_samples=n,
        truncated_from=truncated_from,
    )
