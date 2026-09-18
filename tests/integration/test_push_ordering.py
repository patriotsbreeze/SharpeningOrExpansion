"""The push must never leave a marker in the store without its shard.

This is the invariant all resume correctness rests on, and it is the one an earlier two-pass
implementation got wrong: pass 2 re-enumerated the markers, so a chunk completing after pass 1
had walked its directory got its marker uploaded with its shard left behind. An eviction then
destroys the only copy of that shard, and the next worker skips the chunk forever on the
strength of the marker -- recorded complete, data nowhere.

Pushes run every SYNC_INTERVAL while generation is still writing, so a tree that mutates
mid-push is the normal operating condition, not an edge case. These tests mutate it on purpose.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OBJSTORE = REPO / "scripts" / "objstore.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")


def _push(local_dir: Path, store: Path) -> subprocess.CompletedProcess:
    script = (
        f'source "{OBJSTORE}"; '
        f'export SOE_OBJSTORE=file SOE_FILE_STORE="{store}"; '
        f'objstore_push "{local_dir}" "exp=t"'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=300)


def _orphans(store: Path) -> list[str]:
    """Markers in the store whose shard is absent -- each one is a permanently lost chunk."""
    out = []
    for m in store.rglob("*.done.json"):
        if not m.with_name(m.name.replace(".done.json", ".jsonl.zst")).exists():
            out.append(str(m.relative_to(store)))
    return out


def _tree(root: Path, n_bulk: int = 4000) -> Path:
    local = root / "exp=t"
    (local / "gen" / "aaa").mkdir(parents=True)
    for i in range(n_bulk):
        (local / "gen" / f"zzz_{i}.jsonl.zst").write_text("data\n")
    return local


def test_no_orphan_marker_when_a_chunk_completes_mid_push(tmp_path):
    """A chunk finishing while the push is in flight must not get its marker uploaded alone."""
    store = tmp_path / "store"
    store.mkdir()
    local = _tree(tmp_path)
    d = local / "gen" / "aaa"

    def write_late_chunk():
        time.sleep(0.15)
        # Local order is always shard-then-marker; run_unit guarantees it.
        (d / "chunk=9999.jsonl.zst").write_text("shard\n")
        (d / "chunk=9999.done.json").write_text("marker\n")

    writer = threading.Thread(target=write_late_chunk)
    writer.start()
    r = _push(local, store)
    writer.join()

    assert r.returncode == 0, r.stderr
    assert not _orphans(store), f"orphaned markers after a mid-push write: {_orphans(store)}"


def test_repeated_pushes_converge_and_never_orphan(tmp_path):
    """Across several cycles with concurrent writes, the store is never left inconsistent."""
    store = tmp_path / "store"
    store.mkdir()
    local = _tree(tmp_path, n_bulk=1500)
    d = local / "gen" / "aaa"

    stop = threading.Event()

    def generator():
        i = 0
        while not stop.is_set():
            (d / f"chunk={i:04d}.jsonl.zst").write_text("shard\n")
            (d / f"chunk={i:04d}.done.json").write_text("marker\n")
            i += 1
            time.sleep(0.05)

    g = threading.Thread(target=generator)
    g.start()
    try:
        for _ in range(3):
            r = _push(local, store)
            assert r.returncode == 0, r.stderr
            assert not _orphans(store), f"orphaned markers mid-run: {_orphans(store)}"
    finally:
        stop.set()
        g.join()

    # A final quiescent push must then complete the picture.
    assert _push(local, store).returncode == 0
    assert not _orphans(store)
    local_markers = {str(p.relative_to(local.parent)) for p in local.rglob("*.done.json")}
    stored = {str(p.relative_to(store)) for p in store.rglob("*.done.json")}
    assert local_markers == stored, "a quiescent push must upload every marker"


def test_push_reports_failure_instead_of_silently_copying_nothing(tmp_path):
    """`find -exec` swallows per-file exit status, so a push can copy nothing and return 0.

    That is the worst possible failure here: the run looks durable, the VM is evicted, and
    everything it produced is gone. The destination is made a regular file rather than a
    read-only directory because these tests may run as root, which ignores mode bits.
    """
    store = tmp_path / "store"
    store.write_text("not a directory\n")
    local = _tree(tmp_path, n_bulk=5)
    r = _push(local, store)
    assert r.returncode != 0, "a push that could not write anything reported success"
