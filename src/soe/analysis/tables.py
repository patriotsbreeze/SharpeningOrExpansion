"""LaTeX table fragments and CSV siblings.

Fragments only -- nothing here compiles LaTeX (there is no TeX toolchain in the authoring
environment, and there need not be one on the node). A CPU test checks brace balance, column
counts, and the absence of nan/inf, so a malformed table fails in CI rather than at
submission time.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _fmt(x, nd: int = 3) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "--"
        if math.isinf(x):
            raise ValueError("inf in a table cell; fix the upstream computation")
        return f"{x:.{nd}f}"
    return str(x)


def write_table(df: pd.DataFrame, out: Path, *, caption: str, label: str, nd: int = 3) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)

    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(str(c).replace("_", "\\_") for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[c], nd) for c in cols) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ]
    text = "\n".join(lines)
    if text.count("{") != text.count("}"):
        raise ValueError(f"unbalanced braces in generated table {out}")
    out.with_suffix(".tex").write_text(text)
    return out.with_suffix(".tex")


def decomposition_table(decomp, ci: dict | None = None) -> pd.DataFrame:
    rows = []
    for r in decomp.as_rows():
        row = {
            "confound": r["confound"],
            "explanation": r["label"],
            "shapley": r["shapley"],
            "share": r["share"],
            "loo": r["loo"],
        }
        if ci and r["confound"] in ci:
            row["ci_lo"], row["ci_hi"] = ci[r["confound"]]
        rows.append(row)
    return pd.DataFrame(rows)
