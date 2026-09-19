"""soe -- command line entry point.

Import hygiene matters here: this module (and everything it imports at module scope) must not
pull in ``vllm`` or ``torch``, so that grading, analysis and figures run on a CPU box with no
GPU packages installed. ``tests/unit/test_import_hygiene.py`` enforces it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False, help="Sharpening or Expansion? experiment harness.")
registry_app = typer.Typer(help="Frozen registry management.")
app.add_typer(registry_app, name="registry")

DEFAULT_ROOT = Path("runs")


def _resolve_gradecfg(gradecfg: str, graders: str) -> str | None:
    """A gradecfg hash, or one derived from a grader set, or None to auto-select."""
    if gradecfg:
        return gradecfg
    if graders:
        from soe.grade.extract import POLICIES
        from soe.grade.runner import grading_id

        return grading_id([g.strip() for g in graders.split(",") if g.strip()], list(POLICIES))
    return None


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
def doctor(
    registry: bool = typer.Option(False, "--registry", help="Resolve every HF id."),
    azure: bool = typer.Option(False, "--azure", help="Check the Azure account can run this."),
    location: str = typer.Option("eastus2", help="Region to check quota and offering in."),
    vm_size: str = typer.Option("Standard_NC40ads_H100_v5", help="SKU to check."),
):
    """Preflight. Run this BEFORE booking GPU time -- a typo found here costs nothing.

    Every id in the registries came from a project README rather than the Hub itself, so this
    is the step that turns "probably right" into "verified".
    """
    from soe.registry import check_freeze, load_datasets, load_models

    check_freeze(strict=True)
    models, datasets = load_models(), load_datasets()
    typer.echo(f"registries: {len(models)} models, {len(datasets)} datasets, freeze OK")

    if azure:
        _doctor_azure(location, vm_size)

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


def _doctor_azure(location: str, vm_size: str) -> None:
    """Preflight the Azure account. Every failure names the runbook step that fixes it.

    Three gates fail differently and are easy to confuse: subscription ELIGIBILITY (Free Trial
    cannot raise GPU quota and is excluded from Spot), QUOTA (on-demand and spot are separate
    pools requiring separate requests), and subscription OFFERING (a SKU can be quota-approved,
    have capacity, and still not be offered to you). This checks all three before any GPU bills.
    """
    import json as _json
    import shutil as _shutil
    import subprocess as _sp

    bad: list[str] = []

    def _az(*args: str) -> tuple[int, str]:
        r = _sp.run(["az", *args], capture_output=True, text=True, timeout=120)
        return r.returncode, (r.stdout or r.stderr).strip()

    if not _shutil.which("az"):
        typer.echo("  FAIL az CLI not on PATH -- see azure/README.md step 0")
        raise typer.Exit(1)

    rc, out = _az("account", "show", "-o", "json")
    if rc != 0:
        typer.echo("  FAIL not logged in: run `az login`  [README step 0]")
        raise typer.Exit(1)
    acct = _json.loads(out)
    quota_id = (acct.get("subscriptionPolicies") or {}).get("quotaId", "")
    typer.echo(f"  OK   subscription {acct.get('name')} ({acct.get('state')})")
    if "FreeTrial" in quota_id:
        bad.append(
            f"subscription is Free Trial (quotaId={quota_id}): cannot raise GPU quota and is "
            f"excluded from Spot entirely  [README step 0]"
        )
        typer.echo("  FAIL Free Trial subscription")

    rc, out = _az("vm", "list-skus", "-l", location, "--size", vm_size, "--all", "-o", "json")
    if rc != 0:
        bad.append(f"could not list SKUs in {location}: {out[:160]}")
        typer.echo(f"  FAIL vm list-skus {vm_size}")
    else:
        skus = _json.loads(out or "[]")
        if not skus:
            bad.append(f"{vm_size} is not offered in {location} at all  [README step 2]")
            typer.echo(f"  FAIL {vm_size} not offered in {location}")
        else:
            restr = skus[0].get("restrictions") or []
            if restr:
                reasons = ",".join(r.get("reasonCode", "?") for r in restr)
                bad.append(
                    f"{vm_size} in {location} is restricted ({reasons}). NotAvailableForSubscription "
                    f"is a THIRD failure mode, distinct from quota and capacity  [README step 2]"
                )
                typer.echo(f"  FAIL {vm_size} restricted: {reasons}")
            else:
                typer.echo(f"  OK   {vm_size} offered in {location}, no restrictions")
            fam = skus[0].get("family", "")
            if fam:
                typer.echo(f"  OK   ARM quota family: {fam}  (never hardcode this)")

    rc, out = _az("vm", "list-usage", "-l", location, "-o", "json")
    if rc != 0:
        bad.append(f"could not read quota in {location}: {out[:160]}")
        typer.echo("  FAIL vm list-usage")
    else:
        usage = _json.loads(out or "[]")
        def _limit(pred) -> int | None:
            for u in usage:
                name = (u.get("name") or {}).get("value", "")
                if pred(name.lower()):
                    return int(u.get("limit", 0))
            return None

        dedicated = _limit(lambda n: "h100" in n and "low" not in n and "spot" not in n)
        spot = _limit(lambda n: "lowpriority" in n or "spot" in n)
        for label, val, step in (("on-demand", dedicated, "3"), ("spot", spot, "3")):
            if val is None:
                typer.echo(f"  WARN could not identify the {label} quota row; read the table by hand")
            elif val <= 0:
                bad.append(f"{label} quota is {val} in {location} -- request it  [README step {step}]")
                typer.echo(f"  FAIL {label} quota = {val}")
            else:
                typer.echo(f"  OK   {label} quota = {val} vCPU")

    container = os.environ.get("AZ_CONTAINER_URL", "")
    if not container:
        typer.echo("  WARN AZ_CONTAINER_URL unset; skipping the blob check  [README step 5]")
    elif not _shutil.which("azcopy"):
        typer.echo("  WARN azcopy not on PATH; skipping the blob check")
    else:
        r = _sp.run(["azcopy", "list", container, "--output-level=essential"],
                    capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            bad.append(f"blob container unreachable: {(r.stderr or r.stdout)[:160]}  [README step 5]")
            typer.echo("  FAIL blob container unreachable")
        else:
            typer.echo("  OK   blob container reachable")

    if bad:
        typer.echo("\n".join(["", "BLOCKERS:", *(f"  - {b}" for b in bad)]))
        raise typer.Exit(1)
    typer.echo("azure preflight OK")


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
    from soe.io.markers import is_done

    n = skipped = 0
    for path in sorted(exp_root(root, cfg.exp_id).joinpath("gen").rglob("*.jsonl.zst")):
        # A shard without its marker is uncertified: generation crashed between writing the
        # data and writing the marker. Grading it would feed unverified rows into the tensor
        # while `verify` still counts the unit as not-done.
        if not is_done(path):
            skipped += 1
            continue
        ref = parse_gen_chunk(path, root, cfg.exp_id)
        if grade_shard(
            path, root=root, exp_id=cfg.exp_id, ref=ref,
            problems=problems[ref.dataset_key], graders=names,
        ):
            n += 1
    msg = f"graded {n} shards with {names}"
    if skipped:
        msg += f" ({skipped} uncertified shards skipped -- no completion marker)"
    typer.echo(msg)


@app.command()
def verify(
    config: Path,
    root: Path = DEFAULT_ROOT,
    deep: bool = True,
    require_complete: bool = True,
    graders: str = typer.Option(
        "", help="Grader set whose grading must be COMPLETE, e.g. 'fastint,mathverify'."
    ),
    gradecfg: str = typer.Option("", help="Target gradecfg hash directly, instead of --graders."),
):
    """Integrity gate. Analysis refuses to run until this passes."""
    from soe.grade.extract import POLICIES
    from soe.grade.runner import grading_id
    from soe.verify import verify_experiment

    cfg = _load_cfg(config)
    target = gradecfg or None
    if not target and graders:
        names = [g.strip() for g in graders.split(",") if g.strip()]
        target = grading_id(names, list(POLICIES))
    rep = verify_experiment(
        root, cfg.exp_id, deep=deep, require_complete=require_complete, target_gradecfg=target
    )
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
    gradecfg: str = typer.Option(
        "", help="Which grading configuration to analyse. Required when several exist."
    ),
    graders: str = typer.Option("", help="Derive the gradecfg from a grader set instead."),
):
    """Curves, crossover, hard-zero 2x2, and the Shapley decomposition."""
    from soe.analysis.pipeline import run_analysis

    cfg = _load_cfg(config)
    res = run_analysis(
        root, cfg, base_key=base, rl_key=rl, grader=grader, policy=policy, n_boot=n_boot,
        grading_id=_resolve_gradecfg(gradecfg, graders),
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
    gradecfg: str = typer.Option(
        "", help="Which grading configuration to analyse. Required when several exist."
    ),
    graders: str = typer.Option("", help="Derive the gradecfg from a grader set instead."),
):
    """Regenerate every figure and table from stored artifacts."""
    from soe.analysis.pipeline import run_analysis, write_outputs

    cfg = _load_cfg(config)
    res = run_analysis(
        root, cfg, base_key=base, rl_key=rl, grader=grader, policy=policy, n_boot=n_boot,
        grading_id=_resolve_gradecfg(gradecfg, graders),
    )
    paths = write_outputs(res, Path(root) / f"exp={cfg.exp_id}" / "results")
    for p in paths:
        typer.echo(f"  {p}")


if __name__ == "__main__":
    app()
