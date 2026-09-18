"""soe -- command line entry point.

Import hygiene matters here: this module (and everything it imports at module scope) must not
pull in ``vllm`` or ``torch``, so that grading, analysis and figures run on a CPU box with no
GPU packages installed. ``tests/unit/test_import_hygiene.py`` enforces it.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False, help="Sharpening or Expansion? experiment harness.")
registry_app = typer.Typer(help="Frozen registry management.")
app.add_typer(registry_app, name="registry")

DEFAULT_ROOT = Path("runs")


def _load_cfg(path: Path):
    from soe.config import ExperimentConfig

    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))


@registry_app.command("freeze")
def registry_freeze():
    """Recompute and commit the registry index hashes."""
    from soe.registry import write_freeze

    for k, v in write_freeze().items():
        typer.echo(f"{k}: {v}")


@registry_app.command("check")
def registry_check():
    from soe.registry import check_freeze

    check_freeze(strict=True)
    typer.echo("registry freeze OK")


@app.command()
def doctor(registry: bool = typer.Option(False, "--registry", help="Resolve every HF id.")):
    """Preflight. Run this BEFORE booking GPU time -- a typo found here costs nothing.

    Every id in the registries came from a project README rather than the Hub itself, so this
    is the step that turns "probably right" into "verified".
    """
    from soe.registry import check_freeze, load_datasets, load_models

    check_freeze(strict=True)
    models, datasets = load_models(), load_datasets()
    typer.echo(f"registries: {len(models)} models, {len(datasets)} datasets, freeze OK")

    if not registry:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    bad = []
    for m in models.values():
        if m.is_mock:
            continue
        try:
            info = api.model_info(m.hf_id, revision=m.revision)
            typer.echo(f"  OK   model   {m.hf_id}@{m.revision}  sha={info.sha[:12]}")
        except Exception as e:
            bad.append(f"model {m.hf_id}@{m.revision}: {type(e).__name__}: {e}")
            typer.echo(f"  FAIL model   {m.hf_id}@{m.revision}  {type(e).__name__}")
    for d in datasets.values():
        if d.is_mock:
            continue
        try:
            info = api.dataset_info(d.hf_path, revision=d.revision)
            typer.echo(f"  OK   dataset {d.hf_path}@{d.revision}  sha={info.sha[:12]}")
        except Exception as e:
            bad.append(f"dataset {d.hf_path}@{d.revision}: {type(e).__name__}: {e}")
            typer.echo(f"  FAIL dataset {d.hf_path}@{d.revision}  {type(e).__name__}")
    if bad:
        typer.echo("\n".join(["", "UNRESOLVED:", *bad]))
        raise typer.Exit(1)
    typer.echo("all ids resolve")


@app.command()
def prepare(config: Path, root: Path = DEFAULT_ROOT):
    """Download datasets and freeze the problem manifests. Runs on the node."""
    from soe.data.loaders import load_hf_problems, write_manifest
    from soe.paths import problems_manifest
    from soe.registry import load_datasets

    cfg = _load_cfg(config)
    datasets = load_datasets()
    for key in sorted({a.dataset_key for a in cfg.arms}):
        problems = load_hf_problems(datasets[key])
        spec = datasets[key]
        if spec.expected_rows and len(problems) != spec.expected_rows:
            typer.echo(
                f"  WARN {key}: {len(problems)} rows, registry expected "
                f"{spec.expected_rows}. Update expected_rows deliberately."
            )
        path = problems_manifest(root, cfg.exp_id, key)
        write_manifest(path, problems)
        typer.echo(f"  {key}: {len(problems)} problems -> {path}")


@app.command()
def plan(config: Path, root: Path = DEFAULT_ROOT):
    """Build the deterministic work plan."""
    from soe.data.loaders import read_manifest
    from soe.gen.planner import build_plan, write_plan
    from soe.paths import problems_manifest

    cfg = _load_cfg(config)
    counts = {}
    for key in sorted({a.dataset_key for a in cfg.arms}):
        counts[key] = len(read_manifest(problems_manifest(root, cfg.exp_id, key)))
    units = build_plan(cfg, counts)
    path = write_plan(root, cfg, units)
    typer.echo(f"{len(units)} units -> {path}")


@app.command()
def generate(
    config: Path,
    root: Path = DEFAULT_ROOT,
    worker: int = typer.Option(0, help="This worker's problem shard (one per GPU)."),
    steal: bool = typer.Option(True, help="After draining, pick up units abandoned by others."),
):
    """Run generation for one worker. Resumable: already-marked chunks are skipped."""
    from soe.data.loaders import read_manifest
    from soe.gen.runner import run_experiment
    from soe.io.shards import sweep_tmp
    from soe.paths import exp_root, problems_manifest

    cfg = _load_cfg(config)
    swept = sweep_tmp(exp_root(root, cfg.exp_id))
    if swept:
        typer.echo(f"swept {swept} orphaned .tmp files from a previous crash")

    problems = {
        key: read_manifest(problems_manifest(root, cfg.exp_id, key))
        for key in sorted({a.dataset_key for a in cfg.arms})
    }
    kwargs = {}
    if cfg.backend == "mock":
        kwargs["answers"] = {
            p.problem_idx: p.answer for ps in problems.values() for p in ps
        }
    written = run_experiment(
        cfg, root, problems, worker=worker, backend_kwargs=kwargs, steal=steal
    )
    typer.echo(f"worker {worker}: wrote {len(written)} shards")


@app.command()
def grade(
    config: Path,
    root: Path = DEFAULT_ROOT,
    graders: str = typer.Option("fastint,mathverify", help="Comma-separated grader names."),
):
    """Grade every generation shard. Re-runnable without re-sampling."""
    from soe.data.loaders import read_manifest
    from soe.grade.runner import grade_shard
    from soe.paths import exp_root, parse_gen_chunk, problems_manifest

    cfg = _load_cfg(config)
    names = [g.strip() for g in graders.split(",") if g.strip()]
    problems = {
        key: {p.problem_idx: p for p in read_manifest(problems_manifest(root, cfg.exp_id, key))}
        for key in sorted({a.dataset_key for a in cfg.arms})
    }
    n = 0
    for path in sorted(exp_root(root, cfg.exp_id).joinpath("gen").rglob("*.jsonl.zst")):
        ref = parse_gen_chunk(path, root, cfg.exp_id)
        if grade_shard(
            path, root=root, exp_id=cfg.exp_id, ref=ref,
            problems=problems[ref.dataset_key], graders=names,
        ):
            n += 1
    typer.echo(f"graded {n} shards with {names}")


@app.command()
def verify(config: Path, root: Path = DEFAULT_ROOT, deep: bool = True,
           require_complete: bool = True):
    """Integrity gate. Analysis refuses to run until this passes."""
    from soe.verify import verify_experiment

    cfg = _load_cfg(config)
    rep = verify_experiment(root, cfg.exp_id, deep=deep, require_complete=require_complete)
    typer.echo(rep.render())
    if not rep.ok:
        raise typer.Exit(1)


@app.command()
def analyze(
    config: Path,
    root: Path = DEFAULT_ROOT,
    base: str = typer.Option(..., help="Base model key."),
    rl: str = typer.Option(..., help="RLVR model key."),
    grader: str = "fastint",
    policy: str = "boxed_last",
    n_boot: int = 2000,
):
    """Curves, crossover, hard-zero 2x2, and the Shapley decomposition."""
    from soe.analysis.pipeline import run_analysis

    cfg = _load_cfg(config)
    res = run_analysis(
        root, cfg, base_key=base, rl_key=rl, grader=grader, policy=policy, n_boot=n_boot
    )
    out = Path(root) / f"exp={cfg.exp_id}" / "results" / "metrics" / f"{base}__vs__{rl}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res["summary"], indent=2, default=str))
    typer.echo(json.dumps(res["summary"], indent=2, default=str))
    typer.echo(f"\nwrote {out}")


@app.command()
def figures(
    config: Path,
    root: Path = DEFAULT_ROOT,
    base: str = typer.Option(...),
    rl: str = typer.Option(...),
    grader: str = "fastint",
    policy: str = "boxed_last",
    n_boot: int = 2000,
):
    """Regenerate every figure and table from stored artifacts."""
    from soe.analysis.pipeline import run_analysis, write_outputs

    cfg = _load_cfg(config)
    res = run_analysis(
        root, cfg, base_key=base, rl_key=rl, grader=grader, policy=policy, n_boot=n_boot
    )
    paths = write_outputs(res, Path(root) / f"exp={cfg.exp_id}" / "results")
    for p in paths:
        typer.echo(f"  {p}")


if __name__ == "__main__":
    app()
