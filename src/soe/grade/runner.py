"""Grading pass: generation shards -> wide parquet.

Grading is deliberately a separate pass from generation. Graders have bugs, extraction
policies are a treatment to be varied, and C4 needs the raw CoT -- so re-grading must never
require re-sampling. The output is *wide* over (policy x grader) so that every downstream
adjustment is a tensor operation; if any adjustment required a re-grade, the Shapley
decomposition would become unaffordable and get cut.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from soe.data.canonical import Problem
from soe.grade.extract import POLICIES, extract_all
from soe.grade.graders import make_grader
from soe.grade.nullgold import permuted_golds
from soe.grade.pool import clip_tail
from soe.grade.tokenpos import answer_token_pos, find_answer_char
from soe.io.markers import is_done, write_marker
from soe.io.shards import read_shard
from soe.paths import ChunkRef, grade_chunk


def grading_id(graders: list[str], policies: list[str]) -> str:
    blob = json.dumps({"graders": sorted(graders), "policies": sorted(policies)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def grade_rows(
    rows: list[dict],
    problems: dict[int, Problem],
    *,
    graders: list[str],
    policies: list[str] | None = None,
    nullgold_seed: int = 0,
) -> pd.DataFrame:
    policies = list(POLICIES) if policies is None else policies
    idxs = sorted(problems)
    golds = [problems[i].answer for i in idxs]
    null_map = dict(zip(idxs, permuted_golds(golds, seed=nullgold_seed), strict=True))

    grader_objs = {g: make_grader(g) for g in graders}
    out = []
    for r in rows:
        text = r["text"]
        tail = clip_tail(text)
        gold = problems[r["problem_idx"]].answer
        null_gold = null_map[r["problem_idx"]]
        ex = extract_all(tail)

        rec = {
            "sample_uid": r["sample_uid"],
            "model_key": r["model_key"],
            "dataset_key": r["dataset_key"],
            "variant": r["variant"],
            "problem_idx": r["problem_idx"],
            "sample_idx": r["sample_idx"],
            "n_completion_tokens": r["n_completion_tokens"],
            "finish_reason": r["finish_reason"],
            "gold": gold,
        }
        for p in policies:
            rec[f"extracted__{p}"] = ex[p]

        primary = ex.get("boxed_last") or ex.get("boxed_strict")
        rec["answer_token_pos"] = answer_token_pos(
            tail, find_answer_char(tail, primary), r["n_completion_tokens"]
        )
        rec["n_distinct_ints"] = len({m for m in _ints(tail)})
        rec["repetition_ratio"] = _repetition_ratio(tail)

        for gname, gobj in grader_objs.items():
            status = "ok"
            for p in policies:
                pred = ex[p]
                if pred is None:
                    rec[f"correct__{gname}__{p}"] = False
                    rec[f"nullgold__{gname}__{p}"] = False
                    continue
                try:
                    rec[f"correct__{gname}__{p}"] = gobj.grade(gold=gold, pred=pred)
                    rec[f"nullgold__{gname}__{p}"] = gobj.grade(gold=null_gold, pred=pred)
                except Exception:
                    rec[f"correct__{gname}__{p}"] = False
                    rec[f"nullgold__{gname}__{p}"] = False
                    status = "error"
            rec[f"grade_status__{gname}"] = status
        out.append(rec)
    return pd.DataFrame(out)


def _ints(text: str) -> list[str]:
    import re

    return re.findall(r"-?\d+", text)


def _repetition_ratio(text: str, window: int = 40) -> float:
    """Fraction of ``window``-char blocks that are exact duplicates. Flags degenerate loops."""
    if len(text) < window * 2:
        return 0.0
    blocks = [text[i : i + window] for i in range(0, len(text) - window, window)]
    if not blocks:
        return 0.0
    return 1.0 - len(set(blocks)) / len(blocks)


def grade_shard(
    gen_path: Path,
    *,
    root: Path | str,
    exp_id: str,
    ref: ChunkRef,
    problems: dict[int, Problem],
    graders: list[str],
    policies: list[str] | None = None,
) -> Path | None:
    policies = list(POLICIES) if policies is None else policies
    gid = grading_id(graders, policies)
    out_path = grade_chunk(root, exp_id, gid, ref)
    if is_done(out_path):
        return None
    rows = list(read_shard(gen_path))
    df = grade_rows(rows, problems, graders=graders, policies=policies)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    integrity = {
        "n_rows": len(df),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "bytes": out_path.stat().st_size,
    }
    write_marker(
        out_path, unit_id=f"grade:{gid}:{ref.pshard}:{ref.chunk_idx}", integrity=integrity,
        problem_idxs=sorted({int(i) for i in df["problem_idx"]}),
        seed_lo=0, seed_hi=0, chunk_size=len(df),
        engine_fingerprint=f"grade|{gid}", git_sha="", wall_s=0.0, started_at="",
    )
    return out_path
