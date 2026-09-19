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
    # (dataset_key, problem_idx) -- problem_idx alone is a position within ONE dataset's
    # manifest and is not globally unique, so it cannot key the axis on its own.
    problem_idxs: tuple[tuple[str, int], ...]
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


def _require_usable_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return a bool array, or raise. Never coerce.

    ``.to_numpy(dtype=bool)`` on an unvalidated column is never safe here. If two grader
    configurations are ever concatenated, the narrower one contributes NaN for the columns it
    lacks -- and ``bool(float('nan'))`` is ``True``, so every one of those samples is scored
    CORRECT under a grader it was never run through. That drives pass@k toward 1.0 for
    whichever arm lacked the column, and corrupts the null-gold false-positive rate in the same
    direction, so the one diagnostic that might have caught it reads like a grader problem
    instead of a merge problem.
    """
    if col not in df.columns:
        raise KeyError(f"{col} not in graded frame; have {sorted(df.columns)[:12]}...")
    s = df[col]
    if s.isna().any():
        n_bad = int(s.isna().sum())
        raise ValueError(
            f"{col} has {n_bad}/{len(s)} null values. This is what mixing grader "
            f"configurations looks like: the narrower one has no such column, and a null "
            f"would be silently coerced to True -- i.e. scored correct. Select a single "
            f"gradecfg instead of pooling them."
        )
    if s.dtype != bool:
        try:
            return s.to_numpy(dtype=bool)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{col} is dtype {s.dtype} and not coercible to bool: {e}") from None
    return s.to_numpy(dtype=bool)


def build_tensor(
    df: pd.DataFrame,
    *,
    grader: str,
    policy: str,
    model_keys: list[str] | None = None,
    strict_equal_n: bool = True,
    require_problems: set[tuple[str, int]] | None = None,
) -> CorrectnessTensor:
    """Dense correctness tensor over an ARM-SCOPED problem axis.

    The problem axis is ``(dataset_key, problem_idx)``, not ``problem_idx`` alone.
    ``problem_idx`` is a position within one dataset's manifest and is not globally unique, so
    keying on it alone collapses every arm of a model onto the same cells. On the real Stage 1
    config ``qwen25m7b_base`` has six arms -- two prompt variants across four datasets -- all of
    which contain ``problem_idx`` 0. Pooling them would average the deliberately contaminated
    AIME 2024 control together with the clean 2026 set, which is precisely the comparison C3
    exists to make, and it would do so without any ragged-n signal.
    """
    ccol, ncol = f"correct__{grader}__{policy}", f"nullgold__{grader}__{policy}"

    models = tuple(model_keys or sorted(df["model_key"].unique()))
    # Restrict BEFORE deriving anything: an arm nobody asked about must not set n for the
    # arms they did ask about, nor widen the problem axis.
    df = df[df["model_key"].isin(models)]
    if df.empty:
        raise ValueError(f"no graded rows for any of {models}")

    missing_models = [m for m in models if not (df["model_key"] == m).any()]
    if missing_models:
        raise ValueError(
            f"no graded rows at all for {missing_models}. Left unchecked this yields an "
            f"all-False arm rather than an error -- every problem scored wrong, no exception."
        )

    # One variant and one sampling config per (model, dataset), or the cells are a blend.
    for keys, g in df.groupby(["model_key", "dataset_key"], sort=False):
        for col in ("variant", "sampling_id"):
            if col in g.columns and g[col].nunique() > 1:
                raise ValueError(
                    f"{keys}: graded frame mixes {col}s {sorted(g[col].unique())}. Scope the "
                    f"load to one, or the tensor silently averages across them."
                )

    correct_all = _require_usable_column(df, ccol)
    nullg_all = _require_usable_column(df, ncol)
    df = df.assign(_correct=correct_all, _nullgold=nullg_all)

    problems: tuple[tuple[str, int], ...] = tuple(
        sorted({(str(d), int(p)) for d, p in zip(df["dataset_key"], df["problem_idx"], strict=True)})
    )
    if require_problems is not None:
        absent = set(require_problems) - set(problems)
        if absent:
            raise ValueError(
                f"{len(absent)} expected problems are absent from the graded frame "
                f"(e.g. {sorted(absent)[:3]}). An entire ungraded pshard looks exactly like "
                f"this: the problems vanish, every survivor keeps a full sample count, and no "
                f"ragged-n warning fires. Re-grade the assembled tree."
            )

    # Equal n is required WITHIN an arm. Across datasets it is legitimate and deliberate --
    # Stage 1 samples n=512 on the Tier A base arms and n=256 on the controls -- so a global
    # equality check would make the real config unrunnable.
    per = df.groupby(["model_key", "dataset_key", "problem_idx"])["sample_idx"].nunique()
    ragged_arms = {}
    for (mk, dk), g in per.groupby(level=[0, 1]):
        if g.min() != g.max():
            short = [int(i[2]) for i in g[g < g.max()].index[:3]]
            ragged_arms[f"{mk}/{dk}"] = (int(g.min()), int(g.max()), short)
    if ragged_arms:
        msg = "ragged n within arm(s): " + "; ".join(
            f"{a}: {lo}..{hi} (short problems e.g. {ex})" for a, (lo, hi, ex) in ragged_arms.items()
        )
        if strict_equal_n:
            raise ValueError(
                msg + ". Averaging pass@k over problems with unequal n weights them by "
                "differing estimator variance. Re-grade the missing chunks, or pass "
                "strict_equal_n=False deliberately."
            )

    n = int(per.min())
    truncated_from = None
    if per.min() != per.max():
        truncated_from = {"n_min": int(per.min()), "n_max": int(per.max()),
                          "ragged_arms": ragged_arms or None}

    M, P = len(models), len(problems)
    correct = np.zeros((M, P, n), dtype=bool)
    tokens = np.zeros((M, P, n), dtype=np.int32)
    apos = np.full((M, P, n), -1, dtype=np.int32)
    nullg = np.zeros((M, P, n), dtype=bool)
    dataset_of = [d for d, _ in problems]

    mi = {k: i for i, k in enumerate(models)}
    pi = {k: i for i, k in enumerate(problems)}

    for (mk, dk, pidx), g in df.groupby(["model_key", "dataset_key", "problem_idx"], sort=False):
        key = (str(dk), int(pidx))
        if mk not in mi or key not in pi:
            continue
        # Deterministic truncation to the lowest n sample_idx values.
        g = g.sort_values("sample_idx").head(n)
        a, b = mi[mk], pi[key]
        correct[a, b, : len(g)] = g["_correct"].to_numpy(dtype=bool)
        tokens[a, b, : len(g)] = g["n_completion_tokens"].to_numpy(dtype=np.int32)
        apos[a, b, : len(g)] = g["answer_token_pos"].to_numpy(dtype=np.int32)
        nullg[a, b, : len(g)] = g["_nullgold"].to_numpy(dtype=bool)

    return CorrectnessTensor(
        model_keys=models, problem_idxs=problems, correct=correct, tokens=tokens,
        answer_pos=apos, nullgold=nullg, dataset_of=tuple(dataset_of), n_samples=n,
        truncated_from=truncated_from,
    )
