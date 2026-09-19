"""Grading coverage, and the silent ways an ungraded shard changes published numbers.

A VM evicted mid-run never reaches its own grading step: `scripts/launch_node.sh` grades after
the generate wait loop, over whatever is on that VM's local disk, and resume pulls markers but
never shards. So its shards stay ungraded forever.

That is not a missing file so much as a missing ROW. `build_tensor` derives its problem axis
from whatever is in the graded frame, so an entire ungraded pshard makes those problems vanish
from the analysis with NO ragged-n warning -- every surviving problem still has a full sample
count. `soe verify` is the only place that can catch it.
"""

from __future__ import annotations

from pathlib import Path

from soe.config import ArmSpec, ExperimentConfig, SamplingSpec
from soe.data.loaders import load_mock_problems, write_manifest
from soe.gen.planner import build_plan, write_plan
from soe.gen.runner import run_experiment
from soe.grade.runner import grade_shard
from soe.paths import (
    ChunkRef,
    exp_root,
    grade_chunk,
    list_gradecfgs,
    marker_for,
    parse_gen_chunk,
    parse_grade_chunk,
    problems_manifest,
)
from soe.registry import load_datasets
from soe.verify import verify_experiment


def _run(root: Path, exp_id: str = "cov", graders: tuple[str, ...] = ("fastint",)):
    cfg = ExperimentConfig(
        exp_id=exp_id, backend="mock", n_workers=2,
        sampling=SamplingSpec(n_total=32, chunk_size=16, temperature=1.0, max_new_tokens=512),
        arms=[ArmSpec(model_key="mock_base", dataset_key="mock_mini", variant="zeroshot_boxed")],
    )
    probs = load_mock_problems(load_datasets()["mock_mini"])
    write_manifest(problems_manifest(root, cfg.exp_id, "mock_mini"), probs)
    write_plan(root, cfg, build_plan(cfg, {"mock_mini": len(probs)}))
    run_experiment(
        cfg, root, {"mock_mini": probs}, worker=0,
        backend_kwargs={"answers": {p.problem_idx: p.answer for p in probs}}, steal=True,
    )
    pmap = {p.problem_idx: p for p in probs}
    for g in sorted(exp_root(root, cfg.exp_id).joinpath("gen").rglob("*.jsonl.zst")):
        grade_shard(
            g, root=root, exp_id=cfg.exp_id, ref=parse_gen_chunk(g, root, cfg.exp_id),
            problems=pmap, graders=list(graders),
        )
    return cfg


def test_grade_path_round_trips():
    """`paths.py` documents that every builder has a matching parser; it now does."""
    ref = ChunkRef("qwen25m7b_base", "aime_2026", "zeroshot_boxed", "1fc90270", 3, 17)
    p = grade_chunk("/tmp/runs", "tierA", "7cefe9d7", ref)
    gid, back = parse_grade_chunk(p, "/tmp/runs", "tierA")
    assert gid == "7cefe9d7"
    assert back == ref


def test_a_fully_graded_run_passes(tmp_path):
    cfg = _run(tmp_path)
    gid = list_gradecfgs(tmp_path, cfg.exp_id)[0]
    rep = verify_experiment(tmp_path, cfg.exp_id, deep=True, target_gradecfg=gid)
    assert rep.ok, rep.render()
    assert "(target)" in rep.stats[f"gradecfg[{gid}]"]


def test_an_ungraded_chunk_fails_verify(tmp_path):
    """The regression test for the whole defect: this passed before the coverage check."""
    cfg = _run(tmp_path)
    gid = list_gradecfgs(tmp_path, cfg.exp_id)[0]
    graded = sorted(exp_root(tmp_path, cfg.exp_id).joinpath("grade").rglob("*.parquet"))
    assert graded
    marker_for(graded[0]).unlink()
    graded[0].unlink()

    rep = verify_experiment(tmp_path, cfg.exp_id, deep=True, target_gradecfg=gid)
    assert not rep.ok, "an ungraded chunk must not pass the gate"
    assert any("ungraded" in e for e in rep.errors), rep.render()


def test_a_marker_without_its_parquet_is_not_coverage(tmp_path):
    """The parquet is renamed without an fsync while its marker is fsynced.

    So a marker can outlive the data it vouches for, and coverage keyed on markers alone is
    precisely the check that misses it.
    """
    cfg = _run(tmp_path)
    gid = list_gradecfgs(tmp_path, cfg.exp_id)[0]
    graded = sorted(exp_root(tmp_path, cfg.exp_id).joinpath("grade").rglob("*.parquet"))
    graded[0].unlink()  # marker survives, data does not

    rep = verify_experiment(tmp_path, cfg.exp_id, deep=True, target_gradecfg=gid)
    assert not rep.ok, "a marker without its parquet must not count as covered"


def test_incomplete_generation_downgrades_coverage_to_a_warning(tmp_path):
    """Per-VM verify runs with require_complete off, and every VM's gradecfg is partial.

    `soe grade` walks the shards on that VM's local disk while resume pulls markers but never
    shards, so a partial gradecfg is the healthy intermediate state. Failing on it would fire
    on every worker in the fleet.
    """
    cfg = _run(tmp_path)
    gid = list_gradecfgs(tmp_path, cfg.exp_id)[0]
    graded = sorted(exp_root(tmp_path, cfg.exp_id).joinpath("grade").rglob("*.parquet"))
    marker_for(graded[0]).unlink()
    graded[0].unlink()

    rep = verify_experiment(
        tmp_path, cfg.exp_id, deep=True, require_complete=False, target_gradecfg=gid
    )
    assert rep.ok, rep.render()
    assert any("ungraded" in w for w in rep.warnings)


def test_no_grading_at_all_fails_a_complete_run_but_warns_mid_flight(tmp_path):
    """An ungraded tree cannot be analysed, so passing it would be the same silent lie."""
    import shutil

    cfg = _run(tmp_path)
    shutil.rmtree(exp_root(tmp_path, cfg.exp_id) / "grade")

    strict = verify_experiment(tmp_path, cfg.exp_id, deep=True)
    assert not strict.ok
    assert any("no graded output" in e for e in strict.errors)

    midflight = verify_experiment(tmp_path, cfg.exp_id, deep=True, require_complete=False)
    assert midflight.ok, midflight.render()
    assert midflight.stats["gradecfgs"] == "(none)"


def test_a_partial_NON_target_gradecfg_is_reported_not_failed(tmp_path):
    """A cheap pass over everything plus an expensive pass over one arm is a legitimate shape.

    Only the target configuration must be complete; the rest are data, not errors.
    """
    cfg = _run(tmp_path)
    target = list_gradecfgs(tmp_path, cfg.exp_id)[0]

    probs = load_mock_problems(load_datasets()["mock_mini"])
    pmap = {p.problem_idx: p for p in probs}
    one = sorted(exp_root(tmp_path, cfg.exp_id).joinpath("gen").rglob("*.jsonl.zst"))[0]
    grade_shard(
        one, root=tmp_path, exp_id=cfg.exp_id, ref=parse_gen_chunk(one, tmp_path, cfg.exp_id),
        problems=pmap, graders=["fastint", "qwen"],
    )
    cfgs = list_gradecfgs(tmp_path, cfg.exp_id)
    assert len(cfgs) == 2
    other = next(c for c in cfgs if c != target)

    rep = verify_experiment(tmp_path, cfg.exp_id, deep=True, target_gradecfg=target)
    assert rep.ok, rep.render()
    assert rep.stats[f"gradecfg[{other}]"].startswith("1/")
