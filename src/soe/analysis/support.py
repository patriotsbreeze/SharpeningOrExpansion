"""Direct measures of support change.

The Shapley residual estimates how much of the gap is *not* explained by the measured
confounds, but a residual alone is weak evidence for a mechanism. These measures speak to
support change directly, and all of them are free from data already retained.

The hard-zero 2x2 is likely the single strongest figure in the paper: it is a direct count of
problems the base never solves but the RL model does (expansion), against problems the base
solves but the RL model never does (lost support), with exact binomial bounds on both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soe.analysis.bootstrap import clopper_pearson


@dataclass(frozen=True)
class HardZero2x2:
    both: int            # both models solve at least once
    expansion: int       # base never solves, RL does
    lost: int            # base solves, RL never does
    neither: int
    n_samples: int
    # (dataset_key, problem_idx) pairs -- see CorrectnessTensor.problem_idxs.
    expansion_problem_idxs: tuple[tuple[str, int], ...]
    lost_problem_idxs: tuple[tuple[str, int], ...]

    @property
    def n_problems(self) -> int:
        return self.both + self.expansion + self.lost + self.neither

    def bounds(self) -> dict:
        """What 'never solved in n samples' actually licenses.

        Zero successes in n draws bounds the per-sample probability at roughly 3/n, so these
        are problems the base reaches with probability below that floor -- not problems it
        provably cannot reach at any k. Reporting it this way is the honest version of the
        'newly solvable' claim.
        """
        p_hi = clopper_pearson(0, self.n_samples)[1]
        return {
            "per_sample_upper_bound_when_zero": p_hi,
            "expansion_frac": self.expansion / max(1, self.n_problems),
            "lost_frac": self.lost / max(1, self.n_problems),
            "expansion_ci": clopper_pearson(self.expansion, self.n_problems),
            "lost_ci": clopper_pearson(self.lost, self.n_problems),
        }


def hard_zero_2x2(correct_base: np.ndarray, correct_rl: np.ndarray, problem_idxs) -> HardZero2x2:
    """``correct_*`` are ``[P, N]`` boolean matrices over the same problems."""
    if correct_base.shape != correct_rl.shape:
        raise ValueError(f"shape mismatch: {correct_base.shape} vs {correct_rl.shape}")
    b = correct_base.any(axis=1)
    r = correct_rl.any(axis=1)
    idxs = list(problem_idxs)
    return HardZero2x2(
        both=int((b & r).sum()),
        expansion=int((~b & r).sum()),
        lost=int((b & ~r).sum()),
        neither=int((~b & ~r).sum()),
        n_samples=correct_base.shape[1],
        expansion_problem_idxs=tuple(k for k, keep in zip(idxs, ~b & r, strict=True) if keep),
        lost_problem_idxs=tuple(k for k, keep in zip(idxs, b & ~r, strict=True) if keep),
    )


def solve_rate_cdf(correct: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Empirical CDF of per-problem solve rates.

    Sharpening and expansion have different signatures here: sharpening moves mass out of the
    middle toward both extremes, while expansion moves mass off zero.
    """
    p = correct.mean(axis=1)
    grid = np.linspace(0, 1, 101) if grid is None else grid
    return {
        "p": p,
        "grid": grid,
        "cdf": np.array([(p <= g).mean() for g in grid]),
        "frac_zero": float((p == 0).mean()),
        "frac_one": float((p == 1).mean()),
        "mean": float(p.mean()),
    }


def answer_breadth(extracted: list[list[str | None]], correct: np.ndarray) -> dict:
    """Distinct *incorrect* answers per problem, and the entropy of the answer distribution.

    Sharpening should collapse both; genuine expansion should not. This is the measure least
    confounded by the pass@k machinery, since it never touches k at all.
    """
    breadth, entropies = [], []
    for i, answers in enumerate(extracted):
        wrong = [a for a, c in zip(answers, correct[i], strict=True) if a is not None and not c]
        breadth.append(len(set(wrong)))
        if wrong:
            _, counts = np.unique(np.array(wrong, dtype=object), return_counts=True)
            q = counts / counts.sum()
            entropies.append(float(-(q * np.log(q)).sum()))
        else:
            entropies.append(0.0)
    return {
        "distinct_wrong": np.array(breadth),
        "answer_entropy": np.array(entropies),
        "mean_distinct_wrong": float(np.mean(breadth)),
        "mean_entropy": float(np.mean(entropies)),
    }
