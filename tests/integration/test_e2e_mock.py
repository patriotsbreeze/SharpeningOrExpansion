"""The keystone test: plan -> generate -> grade -> verify -> analyze -> figures, on CPU.

The mock backend exposes each problem's latent solve probability, so the recovered pass@k
curve can be checked against the analytic ``1-(1-p)^k``. That validates the planner, the
seeding scheme, sharding, grading, the estimator and the bootstrap *simultaneously*, in
seconds, before any GPU-hour is committed.
"""

from __future__ import annotations

import numpy as np
import pytest

from soe.analysis.passk import pass_at_k
from soe.analysis.pipeline import load_graded, run_analysis
from soe.analysis.tensors import build_tensor
from soe.config import ArmSpec, ExperimentConfig, SamplingSpec
from soe.data.loaders import load_mock_problems, read_manifest, write_manifest
from soe.gen.factory import make_backend
from soe.gen.planner import build_plan, write_plan
from soe.gen.runner import run_experiment
from soe.paths import problems_manifest
from soe.registry import load_datasets, load_models
from soe.verify import verify_experiment

# Pathologies off, so the mock's analytic truth holds exactly and the assertion is sharp.
CLEAN = dict(truncation_rate=0.0, guess_rate=0.0, degenerate_rate=0.0, malformed_rate=0.0)


def _cfg(exp_id="e2e", n_total=128, chunk=32, workers=2):
    return ExperimentConfig(
        exp_id=exp_id, backend="mock", n_workers=workers,
        sampling=SamplingSpec(
            n_total=n_total, chunk_size=chunk, temperature=1.0,
            max_new_tokens=2048, max_prompt_tokens=1024,
        ),
        arms=[
            ArmSpec(model_key="mock_base", dataset_key="mock_mini", variant="zeroshot_boxed"),
            ArmSpec(model_key="mock_rl", dataset_key="mock_mini", variant="zeroshot_boxed"),
        ],
    )


def _prepare(root, cfg):
    problems = load_mock_problems(load_datasets()["mock_mini"])
    write_manifest(problems_manifest(root, cfg.exp_id, "mock_mini"), problems)
    write_plan(root, cfg, build_plan(cfg, {"mock_mini": len(problems)}))
    return problems


def _run(root, cfg, problems, backend_kwargs, workers=(0, 1)):
    for w in workers:
        run_experiment(
            cfg, root, {"mock_mini": problems}, worker=w,
            backend_kwargs={"answers": {p.problem_idx: p.answer for p in problems},
                            **backend_kwargs},
            steal=False,
        )


def _grade(root, cfg, problems):
    from soe.grade.runner import grade_shard
    from soe.paths import exp_root, parse_gen_chunk

    pmap = {p.problem_idx: p for p in problems}
    for path in sorted(exp_root(root, cfg.exp_id).joinpath("gen").rglob("*.jsonl.zst")):
        ref = parse_gen_chunk(path, root, cfg.exp_id)
        grade_shard(path, root=root, exp_id=cfg.exp_id, ref=ref,
                    problems=pmap, graders=["fastint"])


def test_recovers_analytic_passk(tmp_path):
    """Measured pass@k must match the mock's known 1-(1-p_i)^k within sampling error."""
    cfg = _cfg(n_total=128)
    problems = _prepare(tmp_path, cfg)
    _run(tmp_path, cfg, problems, CLEAN)
    _grade(tmp_path, cfg, problems)

    rep = verify_experiment(tmp_path, cfg.exp_id, deep=True)
    assert rep.ok, rep.render()

    df = load_graded(tmp_path, cfg.exp_id)
    t = build_tensor(df, grader="fastint", policy="boxed_last",
                     model_keys=["mock_base", "mock_rl"])
    n = t.n_samples

    for model_key in ("mock_base", "mock_rl"):
        backend = make_backend("mock", **CLEAN)
        backend.load(load_models()[model_key])
        truth_p = np.array([backend.true_p(i) for i in t.problem_idxs])
        counts = t.correct[t.model(model_key)].sum(axis=1)

        for k in (1, 8, 64):
            measured = float(np.mean([pass_at_k(n, int(c), k) for c in counts]))
            analytic = float(np.mean(1.0 - (1.0 - truth_p) ** k))
            # Binomial sampling error over P problems x n samples, generously bounded.
            tol = 4.0 * np.sqrt(0.25 / (len(counts) * n)) + 0.06
            assert abs(measured - analytic) < tol, (
                f"{model_key} pass@{k}: measured {measured:.4f} vs analytic {analytic:.4f}"
            )


def test_full_pipeline_with_pathologies(tmp_path):
    """With truncation, lucky guesses and degenerate output switched on, nothing crashes."""
    cfg = _cfg(exp_id="e2e_dirty", n_total=64, chunk=16)
    problems = _prepare(tmp_path, cfg)
    _run(tmp_path, cfg, problems, {})
    _grade(tmp_path, cfg, problems)

    res = run_analysis(tmp_path, cfg, base_key="mock_base", rl_key="mock_rl",
                       grader="fastint", policy="boxed_last", n_boot=200)
    s = res["summary"]
    assert 0.0 <= s["pass_at_1"]["base"] <= 1.0
    assert s["crossover"]["p_exists"] is not None
    assert s["hard_zero"]["both"] + s["hard_zero"]["expansion"] + s["hard_zero"]["lost"] \
        + s["hard_zero"]["neither"] == s["n_problems"]
    # Shapley efficiency must hold on real pipeline output, not just synthetic data.
    res["decomp"]["point"].check_efficiency(atol=1e-8)

    from soe.analysis.pipeline import write_outputs

    paths = write_outputs(res, tmp_path / "results")
    assert all(p.exists() for p in paths)


def test_resume_is_a_noop(tmp_path):
    cfg = _cfg(exp_id="e2e_resume", n_total=64, chunk=16)
    problems = _prepare(tmp_path, cfg)
    _run(tmp_path, cfg, problems, CLEAN)
    before = sorted(p.name for p in (tmp_path).rglob("*.jsonl.zst"))
    _run(tmp_path, cfg, problems, CLEAN)
    after = sorted(p.name for p in (tmp_path).rglob("*.jsonl.zst"))
    assert before == after


@pytest.mark.parametrize("chunk_a,chunk_b", [(16, 32)])
def test_rechunking_after_preemption_never_duplicates_a_seed(tmp_path, chunk_a, chunk_b):
    """The property the whole seeding design exists to guarantee.

    Generate part of a run at one chunk size, wipe some shards to simulate preemption, then
    resume at a *different* chunk size. Seeds must still be globally unique -- duplicates
    would make c and n non-i.i.d. and bias the estimator.
    """
    from soe.io.shards import read_shard
    from soe.paths import exp_root, marker_for

    cfg_a = _cfg(exp_id="e2e_chunk", n_total=64, chunk=chunk_a)
    problems = _prepare(tmp_path, cfg_a)
    _run(tmp_path, cfg_a, problems, CLEAN, workers=(0,))

    shards = sorted(exp_root(tmp_path, "e2e_chunk").rglob("*.jsonl.zst"))
    assert shards, "expected some shards before simulating preemption"
    for p in shards[: max(1, len(shards) // 2)]:
        marker_for(p).unlink()
        p.unlink()

    cfg_b = _cfg(exp_id="e2e_chunk", n_total=64, chunk=chunk_b)
    write_plan(tmp_path, cfg_b, build_plan(cfg_b, {"mock_mini": len(problems)}))
    _run(tmp_path, cfg_b, problems, CLEAN, workers=(0, 1))

    seen: dict[tuple, set] = {}
    for p in sorted(exp_root(tmp_path, "e2e_chunk").rglob("*.jsonl.zst")):
        for row in read_shard(p):
            arm = (row["model_key"], row["dataset_key"], row["variant"])
            bucket = seen.setdefault(arm, set())
            assert row["seed"] not in bucket, (
                f"duplicate seed {row['seed']} in {arm} after re-chunking "
                f"{chunk_a} -> {chunk_b}"
            )
            bucket.add(row["seed"])


def test_manifest_read_detects_drift(tmp_path):
    cfg = _cfg(exp_id="e2e_drift")
    problems = _prepare(tmp_path, cfg)
    path = problems_manifest(tmp_path, cfg.exp_id, "mock_mini")
    assert len(read_manifest(path)) == len(problems)
    path.write_text(path.read_text().replace("least positive", "smallest positive"))
    with pytest.raises(ValueError, match="content hash drifted"):
        read_manifest(path)
