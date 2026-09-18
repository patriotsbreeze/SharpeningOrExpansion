"""Atomic zstd-JSONL shard writer/reader.

Write protocol, in this exact order:

    tmp -> fsync -> rename -> (upload data) -> write+rename marker -> (upload marker)

**The marker is always written last**, so its presence implies the data is complete. A shard
is never appended to and never patched; a crash leaves at most a ``.tmp`` file, which is
garbage collected at startup. That invariant is what makes resume-after-preemption safe
without a lock server or a database.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import zstandard as zstd

COMPRESSION_LEVEL = 10


class ShardExistsError(RuntimeError):
    """Raised when writing a shard that already has a completion marker."""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync the directory so the rename itself survives a power loss / instance reclaim.
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_shard(path: Path, rows: Iterable[dict]) -> dict:
    """Write rows as zstd-compressed JSONL. Returns integrity metadata for the marker."""
    from soe.paths import marker_for

    if marker_for(path).exists():
        raise ShardExistsError(
            f"{path} already has a completion marker; shards are immutable. Delete the marker "
            f"deliberately if you intend to regenerate."
        )
    rows = list(rows)
    raw = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows)
    if rows:
        raw += "\n"
    payload = zstd.ZstdCompressor(level=COMPRESSION_LEVEL).compress(raw.encode())
    _atomic_write(path, payload)
    return {
        "n_rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def read_shard(path: Path) -> Iterator[dict]:
    payload = path.read_bytes()
    raw = zstd.ZstdDecompressor().decompress(payload).decode()
    for line in raw.splitlines():
        if line:
            yield json.loads(line)


def verify_shard(path: Path, expected_sha: str, expected_rows: int) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"{path}: sha256 mismatch (marker {expected_sha}, file {actual})")
    n = sum(1 for _ in read_shard(path))
    if n != expected_rows:
        raise ValueError(f"{path}: row count mismatch (marker {expected_rows}, file {n})")


def sweep_tmp(root: Path) -> int:
    """Delete orphaned .tmp files left by a crash. Returns how many were removed."""
    removed = 0
    for p in root.rglob("*.tmp"):
        p.unlink(missing_ok=True)
        removed += 1
    return removed
