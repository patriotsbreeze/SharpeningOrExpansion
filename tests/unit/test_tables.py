"""Generated LaTeX must be well-formed. A malformed table should fail in CI, not at submission."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soe.analysis.tables import _fmt, write_table


def test_table_is_balanced_and_has_consistent_columns(tmp_path):
    df = pd.DataFrame({"confound": ["C2", "C5"], "shapley": [0.1234, -0.5], "share": [0.2, -0.8]})
    out = write_table(df, tmp_path / "t", caption="Cap", label="tab:x")
    text = out.read_text()
    assert text.count("{") == text.count("}")
    body = [ln for ln in text.splitlines() if ln.endswith(r"\\")]
    assert len({ln.count("&") for ln in body}) == 1, "inconsistent column counts"
    assert (tmp_path / "t.csv").exists(), "CSV sibling must always be written"


def test_nan_renders_as_a_dash_and_inf_is_refused(tmp_path):
    assert _fmt(float("nan")) == "--"
    with pytest.raises(ValueError, match="inf in a table cell"):
        write_table(pd.DataFrame({"a": [np.inf]}), tmp_path / "t", caption="c", label="l")


def test_negative_zero_is_not_printed_as_negative():
    """'-0.000' in a results table reads as a real negative effect that rounds away."""
    assert _fmt(-0.0000001) == "0.000"
    assert not _fmt(-1e-9).startswith("-")
    assert _fmt(-0.5) == "-0.500"


def test_underscores_in_headers_are_escaped(tmp_path):
    out = write_table(pd.DataFrame({"ci_lo": [0.1]}), tmp_path / "t", caption="c", label="l")
    assert r"ci\_lo" in out.read_text()
