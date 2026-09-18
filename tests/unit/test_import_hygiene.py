"""The CPU-only constraint must be real, not aspirational."""

from __future__ import annotations

import subprocess
import sys

MODULES = [
    "soe.cli",
    "soe.analysis.pipeline",
    "soe.analysis.decompose",
    "soe.grade.runner",
    "soe.gen.planner",
    "soe.gen.runner",
    "soe.verify",
]


def test_no_gpu_packages_imported():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in MODULES)
        + "bad = sorted({'vllm','torch'} & set(sys.modules))\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", (
        f"{out} reached sys.modules via the CPU import path. Grading, analysis and figures "
        f"must run on a box with no GPU packages installed; move the import inside a function."
    )
