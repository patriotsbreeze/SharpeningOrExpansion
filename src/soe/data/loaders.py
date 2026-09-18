"""Dataset loading and the frozen problem manifest.

``problem_idx`` is the position in the manifest, and it feeds the seed. The manifest is
written once and thereafter treated as immutable: on every later load the content hashes are
re-derived and compared, so an upstream re-push becomes a hard error rather than a silent
seed remap.

The column names below are guesses for several MathArena repos (this sandbox cannot reach
huggingface.co). ``soe doctor --registry`` resolves them for real; ``FIELD_CANDIDATES`` is
deliberately permissive so a naming difference is a warning at prepare time, not a crash at
hour three of a paid run.
"""

from __future__ import annotations

import json
from pathlib import Path

from soe.config import DatasetSpec
from soe.data.canonical import Problem, normalize_answer
from soe.ids import problem_uid

QUESTION_FIELDS = ("problem", "question", "Problem", "prompt", "problem_statement")
ANSWER_FIELDS = ("answer", "Answer", "solution", "gold", "final_answer")


def _pick(row: dict, candidates: tuple[str, ...], kind: str, dataset_key: str) -> str:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return str(row[c])
    raise KeyError(
        f"{dataset_key}: no {kind} column found. Tried {candidates}; row has {sorted(row)}. "
        f"Add the real column name to loaders.py."
    )


def load_hf_problems(spec: DatasetSpec) -> list[Problem]:
    """Download and normalize. Only ever called on the node -- HF is unreachable locally."""
    if spec.is_mock:
        return load_mock_problems(spec)
    from datasets import load_dataset

    ds = load_dataset(
        spec.hf_path, spec.hf_config, split=spec.split, revision=spec.revision
    )
    problems = []
    for i, row in enumerate(ds):
        q = _pick(row, QUESTION_FIELDS, "question", spec.key)
        a = _pick(row, ANSWER_FIELDS, "answer", spec.key)
        problems.append(
            Problem(
                problem_idx=i,
                problem_uid=problem_uid(q),
                dataset_key=spec.key,
                question=q,
                answer=normalize_answer(a),
                release_date=spec.release_date,
                answer_type=spec.answer_type,
            )
        )
    return problems


def load_mock_problems(spec: DatasetSpec) -> list[Problem]:
    path = Path(__file__).resolve().parents[3] / "fixtures" / "problems_mini.jsonl"
    problems = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        problems.append(
            Problem(
                problem_idx=i,
                problem_uid=problem_uid(row["problem"]),
                dataset_key=spec.key,
                question=row["problem"],
                answer=normalize_answer(row["answer"]),
                release_date=spec.release_date,
                answer_type=spec.answer_type,
            )
        )
    return problems


def write_manifest(path: Path, problems: list[Problem]) -> None:
    from soe.io.shards import _atomic_write

    body = "\n".join(
        json.dumps(p.to_manifest_row(), sort_keys=True, separators=(",", ":")) for p in problems
    )
    _atomic_write(path, (body + "\n").encode())


def read_manifest(path: Path) -> list[Problem]:
    """Read the frozen manifest and re-verify every content hash."""
    problems = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = Problem.from_manifest_row(json.loads(line))
        recomputed = problem_uid(p.question)
        if recomputed != p.problem_uid:
            raise ValueError(
                f"{path}: problem {p.problem_idx} content hash drifted "
                f"({p.problem_uid} -> {recomputed}). The upstream dataset changed under a "
                f"pinned revision, or the manifest was hand-edited. Seeds derive from "
                f"problem_idx, so continuing would silently mix two different problem sets."
            )
        problems.append(p)
    for i, p in enumerate(problems):
        if p.problem_idx != i:
            raise ValueError(f"{path}: manifest is not densely ordered at row {i}")
    return problems
