"""Artifact path construction.

The local root and the object-store prefix are identical below the root, so syncing is a
plain recursive copy in either direction, with no path rewriting to get wrong. Every builder
has a matching parser and ``parse(build(x)) == x`` is a test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

GEN_CHUNK_RE = re.compile(
    r"gen/model=(?P<model>[^/]+)/dataset=(?P<dataset>[^/]+)/variant=(?P<variant>[^/]+)"
    r"/samp=(?P<samp>[^/]+)/pshard=(?P<pshard>\d+)/chunk=(?P<chunk>\d+)\.jsonl\.zst$"
)

GRADE_CHUNK_RE = re.compile(
    r"grade/gradecfg=(?P<gradecfg>[^/]+)"
    r"/model=(?P<model>[^/]+)/dataset=(?P<dataset>[^/]+)/variant=(?P<variant>[^/]+)"
    r"/samp=(?P<samp>[^/]+)/pshard=(?P<pshard>\d+)/chunk=(?P<chunk>\d+)\.parquet$"
)


@dataclass(frozen=True, slots=True)
class ChunkRef:
    model_key: str
    dataset_key: str
    variant: str
    sampling_id: str
    pshard: int
    chunk_idx: int


def exp_root(root: Path | str, exp_id: str) -> Path:
    return Path(root) / f"exp={exp_id}"


def gen_chunk(root: Path | str, exp_id: str, ref: ChunkRef) -> Path:
    return (
        exp_root(root, exp_id)
        / "gen"
        / f"model={ref.model_key}"
        / f"dataset={ref.dataset_key}"
        / f"variant={ref.variant}"
        / f"samp={ref.sampling_id}"
        / f"pshard={ref.pshard:02d}"
        / f"chunk={ref.chunk_idx:04d}.jsonl.zst"
    )


def marker_for(data_path: Path) -> Path:
    """The .done.json marker beside a shard. Written LAST; its presence means data is complete."""
    name = data_path.name
    for suffix in (".jsonl.zst", ".parquet"):
        if name.endswith(suffix):
            return data_path.with_name(name[: -len(suffix)] + ".done.json")
    raise ValueError(f"unrecognised artifact name: {name}")


def grade_chunk(
    root: Path | str, exp_id: str, grading_id: str, ref: ChunkRef
) -> Path:
    return (
        exp_root(root, exp_id)
        / "grade"
        / f"gradecfg={grading_id}"
        / f"model={ref.model_key}"
        / f"dataset={ref.dataset_key}"
        / f"variant={ref.variant}"
        / f"samp={ref.sampling_id}"
        / f"pshard={ref.pshard:02d}"
        / f"chunk={ref.chunk_idx:04d}.parquet"
    )


def gradecfg_dir(root: Path | str, exp_id: str, grading_id: str) -> Path:
    return exp_root(root, exp_id) / "grade" / f"gradecfg={grading_id}"


def list_gradecfgs(root: Path | str, exp_id: str) -> list[str]:
    """Every grading configuration present in the tree, sorted.

    More than one is normal -- ``grading_id`` hashes the grader and policy SET, so re-grading
    with a different grader produces a second directory. Callers must therefore CHOOSE one
    rather than consuming the union: concatenating them duplicates every sample, and where the
    two configurations disagree on columns it silently scores blanks as correct.
    """
    base = exp_root(root, exp_id) / "grade"
    if not base.is_dir():
        return []
    return sorted(
        d.name.split("=", 1)[1] for d in base.iterdir()
        if d.is_dir() and d.name.startswith("gradecfg=")
    )


def parse_grade_chunk(path: Path | str, root: Path | str, exp_id: str) -> tuple[str, ChunkRef]:
    """Inverse of :func:`grade_chunk`. Returns ``(grading_id, ref)``."""
    rel = str(Path(path).relative_to(exp_root(root, exp_id))).replace("\\", "/")
    m = GRADE_CHUNK_RE.search(rel)
    if not m:
        raise ValueError(f"not a grade chunk path: {rel}")
    return m["gradecfg"], ChunkRef(
        model_key=m["model"],
        dataset_key=m["dataset"],
        variant=m["variant"],
        sampling_id=m["samp"],
        pshard=int(m["pshard"]),
        chunk_idx=int(m["chunk"]),
    )


def parse_gen_chunk(path: Path | str, root: Path | str, exp_id: str) -> ChunkRef:
    rel = str(Path(path).relative_to(exp_root(root, exp_id))).replace("\\", "/")
    m = GEN_CHUNK_RE.search(rel)
    if not m:
        raise ValueError(f"not a generation chunk path: {rel}")
    return ChunkRef(
        model_key=m["model"],
        dataset_key=m["dataset"],
        variant=m["variant"],
        sampling_id=m["samp"],
        pshard=int(m["pshard"]),
        chunk_idx=int(m["chunk"]),
    )


def problems_manifest(root: Path | str, exp_id: str, dataset_key: str) -> Path:
    return exp_root(root, exp_id) / "problems" / f"{dataset_key}.manifest.jsonl"


def plan_file(root: Path | str, exp_id: str) -> Path:
    return exp_root(root, exp_id) / "plan.jsonl"


def manifest_file(root: Path | str, exp_id: str) -> Path:
    return exp_root(root, exp_id) / "manifest.json"
