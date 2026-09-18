"""End-to-end analysis: graded shards -> curves, crossover, 2x2, decomposition, figures.

Everything here reads only from stored artifacts, so the whole analysis can be re-run after a
grader fix or a policy change without touching a GPU. That separation is what makes the
robustness axes (three graders x four extraction policies) affordable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from soe.analysis.adjustments import AnalysisState
from soe.analysis.bootstrap import DEFAULT_KS, bootstrap_gap
from soe.analysis.crossover import bootstrap_crossover
from soe.analysis.decompose import bootstrap_decompose
from soe.analysis.support import answer_breadth, hard_zero_2x2, solve_rate_cdf
from soe.analysis.tensors import build_tensor


def load_graded(root: Path | str, exp_id: str) -> pd.DataFrame:
    parts = sorted((Path(root) / f"exp={exp_id}" / "grade").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no graded parquet under exp={exp_id}; run `soe grade` first")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def run_analysis(
    root: Path | str,
    cfg,
    *,
    base_key: str,
    rl_key: str,
    grader: str = "fastint",
    policy: str = "boxed_last",
    n_boot: int = 2000,
    contamination_cutoff: date = date(2025, 12, 1),
    require_verify: bool = True,
) -> dict:
    if require_verify:
        from soe.verify import verify_experiment

        rep = verify_experiment(root, cfg.exp_id, deep=True, require_complete=False)
        if not rep.ok:
            raise RuntimeError(
                "verify failed; refusing to analyze possibly-corrupt artifacts:\n"
                + rep.render()
            )

    df = load_graded(root, cfg.exp_id)
    t = build_tensor(df, grader=grader, policy=policy, model_keys=[base_key, rl_key])
    bi, ri = t.model(base_key), t.model(rl_key)
    n = t.n_samples

    cb, cr = t.correct[bi], t.correct[ri]
    boot = bootstrap_gap(cb.sum(1), cr.sum(1), n, n, ks=DEFAULT_KS, n_boot=n_boot)
    cross = bootstrap_crossover(cb.sum(1), cr.sum(1), n, ks=DEFAULT_KS, n_boot=n_boot)
    tab = hard_zero_2x2(cb, cr, t.problem_idxs)

    # Per-problem release dates, for the C3 difference-in-differences.
    from soe.registry import load_datasets

    ds = load_datasets()
    dates = np.array([ds[k].release_date for k in t.dataset_of], dtype=object)

    st = AnalysisState(
        correct_base=cb, correct_rl=cr,
        tokens_base=t.tokens[bi], tokens_rl=t.tokens[ri],
        answer_pos_base=t.answer_pos[bi], answer_pos_rl=t.answer_pos[ri],
        weights=np.ones(t.n_problems), release_dates=dates,
        k_base=min(256, n), k_rl=min(256, n),
        nullgold_base=t.nullgold[bi], nullgold_rl=t.nullgold[ri],
        # Same-variant alternates are filled in by the C7 arm when it exists; passing the
        # observed matrices makes C7 an exact null player rather than silently dropping it.
        correct_base_shared_prompt=cb, correct_rl_shared_prompt=cr,
    )
    decomp = bootstrap_decompose(
        st, kwargs={"C3": {"cutoff": contamination_cutoff}},
        n_boot=min(n_boot, 500),
    )

    k_top = max(boot["ks"])
    summary = {
        "exp_id": cfg.exp_id, "base": base_key, "rl": rl_key,
        "grader": grader, "policy": policy,
        "n_problems": t.n_problems, "n_samples": n,
        "ragged_n": t.truncated_from,
        "pass_at_1": {"base": boot["curve_a"][1], "rl": boot["curve_b"][1]},
        f"pass_at_{k_top}": {"base": boot["curve_a"][k_top], "rl": boot["curve_b"][k_top]},
        f"gap_at_{k_top}": boot["gap"][k_top],
        f"gap_ci_at_{k_top}": [boot["lo"][k_top], boot["hi"][k_top]],
        "crossover": {
            "k_star": cross["k_star"],
            "p_exists": cross["p_exists"],
            "ci_conditional": cross["ci_conditional"],
        },
        "hard_zero": {
            "both": tab.both, "expansion": tab.expansion,
            "lost": tab.lost, "neither": tab.neither, **tab.bounds(),
        },
        "decomposition": {
            "raw_gap": decomp["point"].raw_gap,
            "residual_C1": decomp["point"].residual_gap,
            "shapley": decomp["point"].shapley,
            "ci": decomp["ci"],
        },
        "nullgold_phi": {
            "base": float(t.nullgold[bi].mean()),
            "rl": float(t.nullgold[ri].mean()),
        },
        "truncation_rate": {
            "base": float((t.answer_pos[bi] < 0).mean()),
            "rl": float((t.answer_pos[ri] < 0).mean()),
        },
        "mean_completion_tokens": {
            "base": float(t.tokens[bi].mean()), "rl": float(t.tokens[ri].mean()),
        },
    }
    return {
        "summary": summary, "boot": boot, "cross": cross, "tab": tab,
        "decomp": decomp, "tensor": t, "base_key": base_key, "rl_key": rl_key,
        "cdf_base": solve_rate_cdf(cb), "cdf_rl": solve_rate_cdf(cr),
    }


def write_outputs(res: dict, results_dir: Path) -> list[Path]:
    from soe.analysis import figures as F
    from soe.analysis.tables import decomposition_table, write_table

    figdir = results_dir / "figures"
    tabdir = results_dir / "tables"
    tag = f"{res['base_key']}__vs__{res['rl_key']}"
    out = [
        F.plot_passk_curves(res["boot"], figdir / f"curves_{tag}",
                            label_a=res["base_key"], label_b=res["rl_key"]),
        F.plot_gap_with_ci(res["boot"], figdir / f"gap_{tag}"),
        F.plot_shapley_waterfall(res["decomp"]["point"], figdir / f"waterfall_{tag}",
                                 ci=res["decomp"]["ci"]),
        F.plot_hard_zero_2x2(res["tab"], figdir / f"hardzero_{tag}"),
        F.plot_solve_rate_cdf(res["cdf_base"], res["cdf_rl"], figdir / f"cdf_{tag}"),
    ]
    out.append(
        write_table(
            decomposition_table(res["decomp"]["point"], ci=res["decomp"]["ci"]),
            tabdir / f"decomposition_{tag}",
            caption=(
                "Shapley decomposition of the base-minus-RLVR pass@$k$ gap. "
                "C1 is the unexplained residual, i.e. the estimate of genuine support "
                "shrinkage. Contributions sum to the raw gap by Shapley efficiency."
            ),
            label=f"tab:decomp:{tag}",
        )
    )
    return out
