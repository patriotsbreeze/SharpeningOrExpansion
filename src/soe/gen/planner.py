"""Turn an ExperimentConfig into a deterministic, resumable list of WorkUnits.

Determinism is the whole point: the plan is regenerable from config alone, so resume is
"plan minus existing markers" with no shared state between workers. Two calls on the same
config must produce byte-identical unit_ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from soe.config import ExperimentConfig
from soe.ids import unit_id
from soe.paths import ChunkRef, gen_chunk, plan_file
from soe.registry import load_models


@dataclass(frozen=True, slots=True)
class WorkUnit:
    exp_id: str
    model_key: str
    dataset_key: str
    variant: str
    sampling_id: str
    pshard: int
    chunk_idx: int
    problem_idxs: tuple[int, ...]
    sample_lo: int
    sample_hi: int
    n_total: int
    chunk_size: int

    @property
    def unit_id(self) -> str:
        return unit_id(
            self.exp_id, self.model_key, self.dataset_key, self.variant,
            self.sampling_id, self.pshard, self.sample_lo, self.sample_hi,
        )

    @property
    def chunk_ref(self) -> ChunkRef:
        return ChunkRef(
            self.model_key, self.dataset_key, self.variant, self.sampling_id,
            self.pshard, self.chunk_idx,
        )

    def to_row(self) -> dict:
        return {
            "unit_id": self.unit_id, "exp_id": self.exp_id, "model_key": self.model_key,
            "dataset_key": self.dataset_key, "variant": self.variant,
            "sampling_id": self.sampling_id, "pshard": self.pshard,
            "chunk_idx": self.chunk_idx, "problem_idxs": list(self.problem_idxs),
            "sample_lo": self.sample_lo, "sample_hi": self.sample_hi,
            "n_total": self.n_total, "chunk_size": self.chunk_size,
        }

    @staticmethod
    def from_row(row: dict) -> WorkUnit:
        return WorkUnit(
            exp_id=row["exp_id"], model_key=row["model_key"], dataset_key=row["dataset_key"],
            variant=row["variant"], sampling_id=row["sampling_id"], pshard=row["pshard"],
            chunk_idx=row["chunk_idx"], problem_idxs=tuple(row["problem_idxs"]),
            sample_lo=row["sample_lo"], sample_hi=row["sample_hi"],
            n_total=row["n_total"], chunk_size=row["chunk_size"],
        )


def partition_problems(n_problems: int, n_workers: int) -> list[list[int]]:
    """Round-robin so each worker gets a mix of easy and hard problems.

    Contiguous blocks would correlate a worker's wall-clock with problem difficulty, which
    makes stragglers worse and makes a lost worker lose a *contiguous* region of the dataset.
    """
    shards: list[list[int]] = [[] for _ in range(n_workers)]
    for i in range(n_problems):
        shards[i % n_workers].append(i)
    return shards


def build_plan(cfg: ExperimentConfig, n_problems: dict[str, int]) -> list[WorkUnit]:
    models = load_models()
    units: list[WorkUnit] = []
    for arm in sorted(cfg.arms, key=lambda a: (a.model_key, a.dataset_key, a.variant)):
        spec = models[arm.model_key]
        if arm.variant not in spec.prompt_variants:
            raise ValueError(
                f"{arm.model_key}: variant {arm.variant!r} not permitted; "
                f"allowed {spec.prompt_variants}"
            )
        samp = cfg.sampling_for(arm)
        if arm.dataset_key not in n_problems:
            raise KeyError(f"no problem count for dataset {arm.dataset_key!r}")
        shards = partition_problems(n_problems[arm.dataset_key], cfg.n_workers)
        n_chunks = samp.n_total // samp.chunk_size
        for w, pidxs in enumerate(shards):
            if not pidxs:
                continue
            for c in range(n_chunks):
                lo = c * samp.chunk_size
                units.append(
                    WorkUnit(
                        exp_id=cfg.exp_id, model_key=arm.model_key,
                        dataset_key=arm.dataset_key, variant=arm.variant,
                        sampling_id=samp.sampling_id, pshard=w, chunk_idx=c,
                        problem_idxs=tuple(pidxs), sample_lo=lo,
                        sample_hi=lo + samp.chunk_size, n_total=samp.n_total,
                        chunk_size=samp.chunk_size,
                    )
                )
    return units


def write_plan(root: Path | str, cfg: ExperimentConfig, units: list[WorkUnit]) -> Path:
    from soe.io.shards import _atomic_write

    path = plan_file(root, cfg.exp_id)
    body = "\n".join(json.dumps(u.to_row(), sort_keys=True, separators=(",", ":")) for u in units)
    _atomic_write(path, (body + "\n").encode())
    return path


def read_plan(root: Path | str, exp_id: str) -> list[WorkUnit]:
    path = plan_file(root, exp_id)
    return [WorkUnit.from_row(json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()]


def remaining_units(root: Path | str, units: list[WorkUnit], worker: int | None = None) -> list[WorkUnit]:
    """Plan minus existing markers. With ``worker`` set, only that worker's own pshard.

    Call with ``worker=None`` for the steal pass, after a worker's own queue drains -- that is
    what picks up the work of a GPU that died mid-run.
    """
    from soe.io.markers import is_done

    out = []
    for u in units:
        if worker is not None and u.pshard != worker:
            continue
        if not is_done(gen_chunk(root, u.exp_id, u.chunk_ref)):
            out.append(u)
    return out
