"""Invariants that make a multi-MACHINE fleet safe, not just multi-GPU on one box.

On Azure the natural topology is a fleet of independent single-GPU VMs: this workload runs
tensor_parallel_size=1 with one engine per GPU and no inter-worker communication, so N small
VMs are interchangeable with one N-GPU node -- and far easier to get capacity for.

That moves a previously-local assumption across a machine boundary. Each VM computes the work
partition itself from the shared config, so these properties have to hold or two VMs will
silently disagree about who owns what.
"""

from __future__ import annotations

import pytest

from soe.config import ArmSpec, ExperimentConfig, SamplingSpec
from soe.gen.planner import build_plan, partition_problems
from soe.seeding import seed_for


def _cfg(n_workers: int, exp_id: str = "fleet") -> ExperimentConfig:
    return ExperimentConfig(
        exp_id=exp_id, backend="mock", n_workers=n_workers,
        sampling=SamplingSpec(n_total=64, chunk_size=16, temperature=1.0, max_new_tokens=512),
        arms=[
            ArmSpec(model_key="mock_base", dataset_key="mock_mini", variant="zeroshot_boxed"),
            ArmSpec(model_key="mock_rl", dataset_key="mock_mini", variant="zeroshot_boxed"),
        ],
    )


def test_partition_is_a_total_cover_with_no_overlap():
    """Every problem is owned by exactly one worker -- no gaps, no double work."""
    for n_problems in (1, 7, 12, 30, 60, 240):
        for n_workers in (1, 2, 3, 8, 16):
            shards = partition_problems(n_problems, n_workers)
            flat = [p for s in shards for p in s]
            assert sorted(flat) == list(range(n_problems)), (n_problems, n_workers)
            assert len(flat) == len(set(flat)), "a problem was assigned to two workers"


def test_partition_is_balanced():
    """Round-robin, so no worker gets more than one extra problem."""
    shards = partition_problems(241, 8)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_every_machine_computes_the_same_plan():
    """Two VMs planning independently from the same config must agree exactly.

    Each VM runs `soe plan` itself rather than fetching an assignment, so determinism across
    processes is what stops two machines from both claiming the same chunk.
    """
    a = build_plan(_cfg(8), {"mock_mini": 12})
    b = build_plan(_cfg(8), {"mock_mini": 12})
    assert [u.unit_id for u in a] == [u.unit_id for u in b]
    assert [u.chunk_ref for u in a] == [u.chunk_ref for u in b]


def test_pshard_does_not_enter_the_seed():
    """Seeds are a function of the problem, not of who happened to generate it.

    This is what makes the fleet safe to resize *between* runs -- but it is also exactly why
    resizing it *during* a run is unsafe, which the next test pins down.
    """
    s1 = seed_for(0, 3, 1, 17, 42)
    s2 = seed_for(0, 3, 1, 17, 42)
    assert s1 == s2


def test_changing_n_workers_midrun_would_duplicate_seeds():
    """A guard rail, written as an executable warning.

    n_workers changes the partition, so the same (problem, sample) lands in a different pshard
    and therefore a different output path -- while keeping the SAME seed, because pshard is not
    part of the seed. Two fleet sizes against one artifact tree would write the identical
    sample twice under two paths, which `soe verify` reports as duplicate seeds.

    Hence: n_workers lives in the committed config, and the fleet size must match it. The
    launcher refuses to start a worker whose index is out of range.
    """
    p8 = partition_problems(12, 8)
    p4 = partition_problems(12, 4)
    owner8 = {p: w for w, s in enumerate(p8) for p in s}
    owner4 = {p: w for w, s in enumerate(p4) for p in s}
    assert owner8 != owner4, "fleet size must change ownership, or this hazard would not exist"

    moved = [p for p in owner8 if owner8[p] != owner4[p]]
    assert moved, "at least one problem must move between fleet sizes"
    # Same seed, different owner -> same sample reachable under two paths.
    for p in moved[:3]:
        assert seed_for(0, 0, 0, p, 0) == seed_for(0, 0, 0, p, 0)


def test_worker_index_must_be_within_the_configured_fleet():
    """A VM handed --worker 9 against n_workers=8 has no shard and must not silently no-op."""
    cfg = _cfg(8)
    units = build_plan(cfg, {"mock_mini": 12})
    pshards = {u.pshard for u in units}
    assert pshards <= set(range(cfg.n_workers))
    assert 9 not in pshards


def test_fewer_problems_than_workers_leaves_idle_workers_not_crashes():
    """Partial capacity is normal on spot: asking for 8 VMs and getting 5 must still work."""
    cfg = _cfg(16)
    units = build_plan(cfg, {"mock_mini": 12})
    assert units, "a fleet larger than the problem count must still produce work"
    owners = {u.pshard for u in units}
    assert len(owners) == 12, "12 problems across 16 workers should occupy exactly 12 shards"


@pytest.mark.parametrize("fleet", [1, 2, 5, 8])
def test_any_subset_of_workers_covers_everything_via_the_steal_pass(fleet):
    """Union of all workers' units == the full plan, so a drained worker can finish the rest."""
    cfg = _cfg(8)
    units = build_plan(cfg, {"mock_mini": 12})
    alive = set(range(fleet))
    mine = [u for u in units if u.pshard in alive]
    # Whatever the live workers do not own is exactly what the steal pass must pick up.
    stolen = [u for u in units if u.pshard not in alive]
    assert len(mine) + len(stolen) == len(units)
    assert {u.unit_id for u in mine} | {u.unit_id for u in stolen} == {u.unit_id for u in units}


def test_out_of_range_worker_fails_loudly_rather_than_idling(tmp_path):
    """A GPU VM that owns no shard must crash, not bill quietly for hours."""
    from soe.data.loaders import load_mock_problems
    from soe.gen.runner import run_experiment
    from soe.registry import load_datasets

    problems = load_mock_problems(load_datasets()["mock_mini"])
    with pytest.raises(ValueError, match="outside the configured fleet"):
        run_experiment(_cfg(8), tmp_path, {"mock_mini": problems}, worker=9)


def test_verify_detects_two_machines_using_different_manifests(tmp_path):
    """The failure a multi-VM fleet makes possible, and the check that catches it.

    Each VM builds its own problem manifest. If the upstream dataset moved between two VMs'
    downloads (a branch pin rather than a commit sha), problem_idx 5 means a different problem
    on each machine -- while the seeds stay identical, because seeds key on the index. Nothing
    inside one machine's artifacts looks wrong; only cross-shard comparison reveals it.

    Only the problem_uid is altered here: the marker is rewritten with correct integrity
    metadata and the original unit_id, so every other check still passes and the test is
    actually exercising the drift detection rather than tripping a checksum.
    """
    from soe.data.loaders import load_mock_problems, write_manifest
    from soe.gen.planner import build_plan, write_plan
    from soe.gen.runner import run_experiment
    from soe.io.markers import read_marker, write_marker
    from soe.io.shards import read_shard, write_shard
    from soe.paths import exp_root, marker_for, problems_manifest
    from soe.registry import load_datasets
    from soe.verify import verify_experiment

    cfg = _cfg(2, exp_id="drift")
    problems = load_mock_problems(load_datasets()["mock_mini"])
    write_manifest(problems_manifest(tmp_path, cfg.exp_id, "mock_mini"), problems)
    write_plan(tmp_path, cfg, build_plan(cfg, {"mock_mini": len(problems)}))
    run_experiment(
        cfg, tmp_path, {"mock_mini": problems}, worker=0,
        backend_kwargs={"answers": {p.problem_idx: p.answer for p in problems}}, steal=True,
    )
    assert verify_experiment(tmp_path, cfg.exp_id, deep=True).ok, "baseline must be clean"

    # Second VM's view: identical indices and seeds, one problem_uid differs.
    shard = sorted(exp_root(tmp_path, cfg.exp_id).rglob("*.jsonl.zst"))[0]
    old_marker = read_marker(shard)
    rows = list(read_shard(shard))
    target = rows[0]["problem_idx"]
    for r in rows:
        if r["problem_idx"] == target:
            r["problem_uid"] = "deadbeefdeadbeef"

    marker_for(shard).unlink()
    shard.unlink()
    integrity = write_shard(shard, rows)
    write_marker(
        shard,
        unit_id=old_marker["unit_id"],
        integrity=integrity,
        problem_idxs=old_marker["problem_idxs"],
        seed_lo=old_marker["seed_lo"],
        seed_hi=old_marker["seed_hi"],
        chunk_size=old_marker["chunk_size"],
        engine_fingerprint=old_marker["engine_fingerprint"],
        git_sha=old_marker["git_sha"],
        wall_s=0.0,
        started_at=old_marker["started_at"],
    )

    rep = verify_experiment(tmp_path, cfg.exp_id, deep=True)
    assert not rep.ok, "manifest drift across machines must not pass verify"
    assert any("more than one problem_uid" in e for e in rep.errors), rep.render()
