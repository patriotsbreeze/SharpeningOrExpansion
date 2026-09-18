"""Execute WorkUnits: render, generate, write shard, write marker, upload.

Two invariants live here and nowhere else:

1. ``max_tokens_effective = min(max_new_tokens, context_len - n_prompt_tokens)`` is computed
   by *us* and recorded per row. Qwen2.5-Math-7B's 4096 is a TOTAL budget; a 4-shot prompt is
   ~900 tokens, so asking for 4096 new tokens either errors or gets silently clamped by the
   engine. A silent clamp would make the C5/C6 token accounting wrong for the single most
   important arm in the paper.
2. The marker is written only after the data file is durably on disk (see io/shards).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from soe.config import ExperimentConfig, ModelSpec, SamplingSpec
from soe.data.canonical import Problem
from soe.fingerprint import engine_fingerprint, git_sha
from soe.gen.backend import GenerationBackend, GenRequest
from soe.gen.planner import WorkUnit
from soe.ids import sample_uid
from soe.io.markers import is_done, write_marker
from soe.io.shards import write_shard
from soe.paths import gen_chunk
from soe.prompts.render import render
from soe.seeding import seed_for


class BudgetError(RuntimeError):
    """Raised when a prompt leaves no room for generation inside the model's context."""


def effective_max_tokens(spec: ModelSpec, samp: SamplingSpec, n_prompt_tokens: int) -> int:
    room = spec.context_len - n_prompt_tokens
    if room <= 0:
        raise BudgetError(
            f"{spec.key}: prompt is {n_prompt_tokens} tokens but context_len is "
            f"{spec.context_len}; nothing left to generate."
        )
    return min(samp.max_new_tokens, room)


def run_unit(
    unit: WorkUnit,
    *,
    root: Path | str,
    spec: ModelSpec,
    samp: SamplingSpec,
    problems: dict[int, Problem],
    backend: GenerationBackend,
    on_shard: Callable[[Path], None] | None = None,
) -> Path | None:
    """Generate one chunk. Returns the shard path, or None if it was already done."""
    out_path = gen_chunk(root, unit.exp_id, unit.chunk_ref)
    if is_done(out_path):
        return None

    started = datetime.now(UTC).isoformat()
    t0 = perf_counter()

    payloads, metas = [], []
    for pidx in unit.problem_idxs:
        problem = problems[pidx]
        payload = render(problem, spec, unit.variant)
        payloads.append(payload.text if payload.text is not None else payload.messages)
        metas.append((pidx, problem, payload))

    prompt_tokens = backend.count_prompt_tokens(payloads)
    if any(pt > samp.max_prompt_tokens for pt in prompt_tokens):
        worst = max(prompt_tokens)
        raise BudgetError(
            f"{spec.key}/{unit.dataset_key}: prompt of {worst} tokens exceeds the configured "
            f"max_prompt_tokens={samp.max_prompt_tokens}. Raise it deliberately -- it is the "
            f"number the context-budget validation is computed against."
        )

    reqs, rows = [], []
    for (pidx, problem, payload), p_tokens, prompt in zip(metas, prompt_tokens, payloads, strict=True):
        max_tok = effective_max_tokens(spec, samp, p_tokens)
        for s_idx in range(unit.sample_lo, unit.sample_hi):
            seed = seed_for(samp.seed_epoch, spec.seed_idx, _dataset_idx(unit.dataset_key),
                            pidx, s_idx)
            uid = sample_uid(unit.exp_id, unit.model_key, unit.dataset_key, unit.variant,
                             unit.sampling_id, pidx, s_idx)
            reqs.append(
                GenRequest(
                    request_id=uid, prompt=prompt, seed=seed, max_tokens=max_tok,
                    temperature=samp.temperature, top_p=samp.top_p, top_k=samp.top_k,
                    stop=tuple(spec.stop),
                )
            )
            rows.append(
                {
                    "sample_uid": uid, "exp_id": unit.exp_id, "model_key": unit.model_key,
                    "model_revision": spec.revision, "dataset_key": unit.dataset_key,
                    "problem_idx": pidx, "problem_uid": problem.problem_uid,
                    "variant": unit.variant, "prompt_sha": payload.prompt_sha,
                    "sampling_id": unit.sampling_id, "sample_idx": s_idx, "seed": seed,
                    "n_prompt_tokens": p_tokens, "max_tokens_effective": max_tok,
                }
            )

    samples = {s.request_id: s for s in backend.generate(reqs)}
    if len(samples) != len(reqs):
        raise RuntimeError(
            f"backend returned {len(samples)} unique samples for {len(reqs)} requests; "
            f"request_ids must round-trip exactly"
        )
    for row in rows:
        s = samples[row["sample_uid"]]
        row.update(
            text=s.text,
            n_completion_tokens=s.n_completion_tokens,
            finish_reason=s.finish_reason,
            stop_str=s.stop_str,
        )

    integrity = write_shard(out_path, rows)
    seeds = [r["seed"] for r in rows]
    write_marker(
        out_path, unit_id=unit.unit_id, integrity=integrity,
        problem_idxs=list(unit.problem_idxs), seed_lo=min(seeds), seed_hi=max(seeds),
        chunk_size=unit.chunk_size,
        engine_fingerprint=engine_fingerprint(backend.fingerprint),
        git_sha=git_sha(), wall_s=perf_counter() - t0, started_at=started,
    )
    if on_shard:
        on_shard(out_path)
    return out_path


def _dataset_idx(dataset_key: str) -> int:
    from soe.registry import load_datasets

    return load_datasets()[dataset_key].seed_idx


def run_experiment(
    cfg: ExperimentConfig,
    root: Path | str,
    problems_by_dataset: dict[str, list[Problem]],
    *,
    worker: int | None = None,
    backend_kwargs: dict | None = None,
    steal: bool = True,
    on_shard: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Run this worker's units, then steal any still-missing units from dead workers."""
    from soe.gen.factory import make_backend
    from soe.gen.planner import build_plan, remaining_units
    from soe.registry import load_models

    models = load_models()
    n_problems = {k: len(v) for k, v in problems_by_dataset.items()}
    units = build_plan(cfg, n_problems)

    written: list[Path] = []
    passes = [worker] + ([None] if steal and worker is not None else [])
    for w in passes:
        todo = remaining_units(root, units, worker=w)
        by_arm: dict[tuple[str, str, str], list[WorkUnit]] = {}
        for u in todo:
            by_arm.setdefault((u.model_key, u.dataset_key, u.variant), []).append(u)

        for (model_key, dataset_key, variant), arm_units in sorted(by_arm.items()):
            spec = models[model_key]
            backend = make_backend(cfg.backend, **(backend_kwargs or {}))
            backend.load(spec)
            try:
                problems = {p.problem_idx: p for p in problems_by_dataset[dataset_key]}
                samp = next(
                    cfg.sampling_for(a) for a in cfg.arms
                    if (a.model_key, a.dataset_key, a.variant) == (model_key, dataset_key, variant)
                )
                for u in sorted(arm_units, key=lambda x: (x.pshard, x.chunk_idx)):
                    path = run_unit(
                        u, root=root, spec=spec, samp=samp, problems=problems,
                        backend=backend, on_shard=on_shard,
                    )
                    if path:
                        written.append(path)
            finally:
                backend.close()
    return written
