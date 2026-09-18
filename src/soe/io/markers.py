"""Completion markers -- the resume primitive.

Resume is: sync only ``*.done.json`` from S3 (kilobytes), then take ``plan.jsonl`` minus the
markers that already exist. No lock server, no database, no coordination between workers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

MARKER_SCHEMA = 3


def write_marker(
    data_path: Path,
    *,
    unit_id: str,
    integrity: dict,
    problem_idxs: list[int],
    seed_lo: int,
    seed_hi: int,
    chunk_size: int,
    engine_fingerprint: str,
    git_sha: str,
    wall_s: float,
    started_at: str,
) -> Path:
    """Write the marker. Callers MUST have already durably written the data file."""
    from soe.paths import marker_for
    from soe.io.shards import _atomic_write

    if not data_path.exists():
        raise RuntimeError(
            f"refusing to write a marker for missing data file {data_path}; the marker is the "
            f"promise that the data is complete"
        )
    payload = {
        "unit_id": unit_id,
        "schema": MARKER_SCHEMA,
        "n_rows": integrity["n_rows"],
        "sha256": integrity["sha256"],
        "bytes": integrity["bytes"],
        "problem_idxs": problem_idxs,
        "seed_lo": seed_lo,
        "seed_hi": seed_hi,
        "chunk_size": chunk_size,
        "engine_fingerprint": engine_fingerprint,
        "git_sha": git_sha,
        "wall_s": round(wall_s, 3),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    mpath = marker_for(data_path)
    _atomic_write(mpath, json.dumps(payload, sort_keys=True, indent=1).encode())
    return mpath


def read_marker(data_path: Path) -> dict | None:
    from soe.paths import marker_for

    m = marker_for(data_path)
    if not m.exists():
        return None
    payload = json.loads(m.read_text())
    if payload.get("schema") != MARKER_SCHEMA:
        raise ValueError(
            f"{m}: marker schema {payload.get('schema')} != reader schema {MARKER_SCHEMA}. "
            f"Artifacts were written by a different code version; do not mix them."
        )
    return payload


def is_done(data_path: Path) -> bool:
    from soe.paths import marker_for

    return marker_for(data_path).exists()


def iter_markers(exp_dir: Path):
    yield from sorted(exp_dir.rglob("*.done.json"))
