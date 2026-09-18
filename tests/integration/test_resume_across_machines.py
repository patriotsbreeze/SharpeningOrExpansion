"""End-to-end resume across machine boundaries, driven through the real shell launcher.

This is the path that cannot be tested from Python alone and is the one most likely to lose
work: a spot VM finishes some chunks, pushes them, and is evicted with its local disk
destroyed. A fresh VM must pull only the markers, conclude those chunks are done, and finish
the rest -- without re-generating anything and without treating the absent shards as damage.

The `file` object-store backend stands in for blob storage with identical semantics, including
the two-pass marker-last ordering.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LAUNCH = REPO / "scripts" / "launch_node.sh"
CONFIG = "configs/experiments/smoke_mock.yaml"
EXP = "smoke_mock"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not (REPO / ".venv" / "bin" / "python").exists(),
    reason="needs bash and the project venv",
)


def _run(root: Path, store: Path, worker: int) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PATH": f"{REPO / '.venv' / 'bin'}:{os.environ['PATH']}",
        "PYTHONPATH": str(REPO / "src"),
        "SOE_ROOT": str(root),
        "SOE_WORKER": str(worker),
        "SOE_OBJSTORE": "file",
        "SOE_FILE_STORE": str(store),
        "GRADERS": "fastint",
        # No boot-time watcher in a test; point the sentinel somewhere that never appears.
        "PREEMPT_SENTINEL": str(root / "never"),
        "SYNC_INTERVAL": "100000",
    }
    return subprocess.run(
        ["bash", str(LAUNCH), CONFIG], cwd=REPO, env=env, capture_output=True, text=True,
        timeout=900,
    )


def _shards(root: Path) -> set[str]:
    base = root / f"exp={EXP}"
    return {str(p.relative_to(base)) for p in (base / "gen").rglob("*.jsonl.zst")}


def _gen_markers(root: Path) -> set[str]:
    """Generation markers only. Grading writes its own markers under grade/."""
    base = root / f"exp={EXP}"
    return {str(p.relative_to(base)) for p in (base / "gen").rglob("*.done.json")}


def _grade_markers(root: Path) -> set[str]:
    base = root / f"exp={EXP}"
    g = base / "grade"
    return {str(p.relative_to(base)) for p in g.rglob("*.done.json")} if g.exists() else set()


def test_fresh_machine_resumes_from_markers_alone(tmp_path):
    store = tmp_path / "blob"
    vm_a = tmp_path / "vmA"

    # --- VM A: does its shard, pushes, then is "evicted" ---
    a = _run(vm_a, store, worker=0)
    assert a.returncode == 0, a.stdout + a.stderr
    a_shards = _shards(vm_a)
    assert a_shards, "VM A produced no shards"
    assert _gen_markers(vm_a) == {s.replace(".jsonl.zst", ".done.json") for s in a_shards}
    assert _grade_markers(vm_a), "VM A graded nothing"

    # Everything VM A made must be in the store before it dies.
    stored = {
        str(p.relative_to(store / f"exp={EXP}"))
        for p in (store / f"exp={EXP}").rglob("*.jsonl.zst")
    }
    assert a_shards <= stored, "shards missing from the store; eviction would lose them"

    # Eviction with --eviction-policy Delete: the disk is gone.
    shutil.rmtree(vm_a)

    # --- VM B: brand new machine, same experiment ---
    vm_b = tmp_path / "vmB"
    b = _run(vm_b, store, worker=1)
    assert b.returncode == 0, b.stdout + b.stderr

    # It must have learned about VM A's work from the markers alone...
    expected = {s.replace(".jsonl.zst", ".done.json") for s in a_shards}
    assert expected <= _gen_markers(vm_b), (
        "VM B did not pick up VM A's generation markers: missing "
        f"{sorted(expected - _gen_markers(vm_b))[:3]}"
    )

    # ...without any of VM A's shards being present locally, which is the whole point:
    # resume transfers kilobytes of markers, not hundreds of GB of shards.
    assert not (_shards(vm_b) & a_shards), (
        f"VM B re-generated already-complete chunks: {sorted(_shards(vm_b) & a_shards)[:3]}"
    )


def test_the_run_completes_across_a_full_fleet_rotation(tmp_path):
    """Every worker in turn, each on a fresh machine, must add up to a complete run."""
    store = tmp_path / "blob"
    n_workers = 2

    for w in range(n_workers):
        vm = tmp_path / f"vm{w}"
        r = _run(vm, store, worker=w)
        assert r.returncode == 0, r.stdout + r.stderr
        shutil.rmtree(vm)

    # Assemble the complete tree the way the analysis machine does, then deep-verify it.
    final = tmp_path / "analysis"
    final.mkdir()
    shutil.copytree(store / f"exp={EXP}", final / f"exp={EXP}")

    env = {
        **os.environ,
        "PATH": f"{REPO / '.venv' / 'bin'}:{os.environ['PATH']}",
        "PYTHONPATH": str(REPO / "src"),
    }
    v = subprocess.run(
        ["python", "-m", "soe.cli", "verify", CONFIG, "--root", str(final)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert v.returncode == 0, (
        "deep verify of the assembled tree failed -- the fleet did not produce a coherent "
        f"run:\n{v.stdout}\n{v.stderr}"
    )
    assert "verify: PASS" in v.stdout
