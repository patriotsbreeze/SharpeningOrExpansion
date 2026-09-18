"""Integrity gate. Analysis must not run until this passes.

Each check corresponds to a specific way the run can produce numbers that look fine and are
wrong. The expensive one -- global seed uniqueness -- is exactly the check that catches
duplicate samples introduced by resume-after-preemption, which is the only failure mode that
genuinely biases the pass@k estimator rather than merely widening its interval.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from soe.gen.planner import WorkUnit, read_plan
from soe.io.markers import read_marker
from soe.io.shards import read_shard, verify_shard
from soe.paths import exp_root, gen_chunk


@dataclass
class VerifyReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        lines = [f"verify: {'PASS' if self.ok else 'FAIL'}"]
        for k, v in sorted(self.stats.items()):
            lines.append(f"  {k}: {v}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for e in self.errors:
            lines.append(f"  ERROR {e}")
        return "\n".join(lines)


def verify_experiment(
    root: Path | str, exp_id: str, *, deep: bool = True, require_complete: bool = True
) -> VerifyReport:
    rep = VerifyReport()
    root = Path(root)

    try:
        from soe.registry import check_freeze

        check_freeze(strict=True)
    except Exception as e:
        rep.fail(f"registry freeze: {e}")

    try:
        units = read_plan(root, exp_id)
    except FileNotFoundError:
        rep.fail(f"no plan.jsonl for exp={exp_id}; run `soe plan` first")
        return rep

    by_id = {u.unit_id: u for u in units}
    rep.stats["planned_units"] = len(units)

    # 1. plan <-> marker bijection
    done, missing = [], []
    for u in units:
        (done if _marker(root, u) else missing).append(u)
    rep.stats["completed_units"] = len(done)
    if missing and require_complete:
        rep.fail(f"{len(missing)} planned units have no marker (e.g. {missing[0].unit_id})")
    elif missing:
        rep.warn(f"{len(missing)} units still pending")

    marker_ids = set()
    for mpath in exp_root(root, exp_id).joinpath("gen").rglob("*.done.json"):
        try:
            marker_ids.add(json.loads(mpath.read_text())["unit_id"])
        except Exception as e:
            rep.fail(f"unreadable marker {mpath}: {e}")
    orphans = marker_ids - set(by_id)
    if orphans:
        rep.fail(
            f"{len(orphans)} markers have no planned unit (stale artifacts from a different "
            f"config mixed into this directory): {sorted(orphans)[:3]}"
        )

    # 2-5. per-shard integrity and seed accounting
    seeds_by_arm: dict[tuple, Counter] = defaultdict(Counter)
    n_by_arm_problem: dict[tuple, Counter] = defaultdict(Counter)
    fingerprints: dict[tuple, set] = defaultdict(set)
    # (dataset_key, problem_idx) -> the set of problem_uids any shard claims for it. On a
    # fleet of independent VMs each machine builds its own problem manifest, so this is the
    # check that catches two machines disagreeing about WHICH problem an index refers to.
    uid_by_problem: dict[tuple, set] = defaultdict(set)
    total_rows = 0

    for u in done:
        path = gen_chunk(root, exp_id, u.chunk_ref)
        m = read_marker(path)
        if m is None:
            rep.fail(f"marker vanished for {path}")
            continue
        fingerprints[(u.model_key, u.dataset_key, u.variant)].add(
            m["engine_fingerprint"].split("|")[0]
        )
        if deep:
            try:
                verify_shard(path, m["sha256"], m["n_rows"])
            except Exception as e:
                rep.fail(str(e))
                continue
            arm = (u.model_key, u.dataset_key, u.variant, u.sampling_id)
            for row in read_shard(path):
                seeds_by_arm[arm][row["seed"]] += 1
                n_by_arm_problem[arm][row["problem_idx"]] += 1
                uid_by_problem[(u.dataset_key, row["problem_idx"])].add(row["problem_uid"])
                total_rows += 1

    rep.stats["rows"] = total_rows

    if deep:
        for arm, counter in seeds_by_arm.items():
            dupes = [s for s, c in counter.items() if c > 1]
            if dupes:
                rep.fail(
                    f"{arm}: {len(dupes)} duplicate seeds. Samples are not i.i.d. and the "
                    f"pass@k estimator is biased. Most likely a resume wrote a chunk twice "
                    f"under different chunking. Delete the affected shards and regenerate."
                )

        drifted = {k: v for k, v in uid_by_problem.items() if len(v) > 1}
        if drifted:
            (dataset_key, pidx), uids = next(iter(drifted.items()))
            rep.fail(
                f"{len(drifted)} problem indices resolve to more than one problem_uid "
                f"(e.g. {dataset_key} problem_idx={pidx} -> {sorted(uids)}). Two machines "
                f"built different problem manifests, so the SAME seed denotes DIFFERENT "
                f"problems in different shards. Pin the dataset by commit sha rather than a "
                f"branch, distribute one manifest to the whole fleet, and regenerate the "
                f"affected arms -- these samples cannot be pooled."
            )

        for arm, counter in n_by_arm_problem.items():
            vals = set(counter.values())
            if len(vals) > 1:
                rep.fail(
                    f"{arm}: ragged n per problem {sorted(vals)}. Averaging pass@k over "
                    f"problems with unequal n weights them by differing estimator variance "
                    f"and is not comparable across arms."
                )
            expected = next((u.n_total for u in done if u.model_key == arm[0]), None)
            if expected and vals and next(iter(vals)) != expected:
                rep.warn(f"{arm}: n={next(iter(vals))} but plan says n_total={expected}")

    for arm, fps in fingerprints.items():
        if len(fps) > 1:
            rep.fail(
                f"{arm} was generated across {len(fps)} engine versions {sorted(fps)}; an arm "
                f"split across an engine upgrade is not a controlled comparison"
            )

    return rep


def _marker(root: Path, u: WorkUnit) -> bool:
    from soe.io.markers import is_done

    return is_done(gen_chunk(root, u.exp_id, u.chunk_ref))
