"""Blob URL construction, and the two dangerous Azure defaults the scripts must not use.

These are shell-level checks driven from pytest so they run in the same CI pass as everything
else. The URL logic is worth pinning because every failure mode here is silent: a path spliced
after a SAS query string authenticates against the wrong thing rather than erroring clearly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OBJSTORE = REPO / "scripts" / "objstore.sh"
LAUNCH = REPO / "scripts" / "launch_node.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")


def _url(base: str, sub: str = "", suffix: str = "") -> str:
    script = f'source "{OBJSTORE}"; AZ_CONTAINER_URL="{base}"; _az_url "{sub}" "{suffix}"'
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


BASE = "https://acct.blob.core.windows.net/soe"
SAS = "https://acct.blob.core.windows.net/soe?sv=2024-11-04&sig=abc%2Fdef"


def test_plain_container_url():
    assert _url(BASE) == BASE
    assert _url(BASE, "exp=s1") == f"{BASE}/exp=s1"
    assert _url(BASE, "exp=s1", "/*") == f"{BASE}/exp=s1/*"


def test_path_is_spliced_before_the_sas_query_not_after():
    """The failure this prevents: '...?sig=x/exp=s1' targets the wrong prefix silently."""
    got = _url(SAS, "exp=s1")
    assert got == "https://acct.blob.core.windows.net/soe/exp=s1?sv=2024-11-04&sig=abc%2Fdef"
    assert "?" in got
    assert got.index("exp=s1") < got.index("?"), "path must precede the query string"


def test_wildcard_is_also_spliced_before_the_query():
    got = _url(SAS, "exp=s1", "/*")
    assert got.endswith("?sv=2024-11-04&sig=abc%2Fdef")
    assert "/exp=s1/*?" in got


def test_trailing_slash_on_the_base_does_not_double_up():
    assert _url(BASE + "/", "exp=s1") == f"{BASE}/exp=s1"
    assert "//exp" not in _url(BASE + "/", "exp=s1")


def test_sas_signature_is_not_mangled():
    """A percent-encoded signature must survive verbatim -- re-encoding invalidates it."""
    assert "sig=abc%2Fdef" in _url(SAS, "exp=s1", "/*")


def test_unconfigured_backend_refuses_rather_than_silently_skipping():
    script = f'source "{OBJSTORE}"; unset AZ_CONTAINER_URL; objstore_require'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode != 0
    assert "AZ_CONTAINER_URL is unset" in r.stderr


def _code_lines(path: Path) -> list[str]:
    """Script lines with comments and blanks removed.

    The scripts deliberately NAME the dangerous commands in comments to explain why they are
    avoided, so these checks have to look at code rather than prose.
    """
    out = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(line.split(" #", 1)[0])
    return out


def test_scripts_never_use_the_destructive_sync_commands():
    """`az storage blob sync` and `azcopy sync` both default --delete-destination=true.

    With N workers pushing overlapping subtrees into one container, either command would
    delete the other workers' shards. Neither default is mentioned in `--help`, so this
    check is the guard rail.
    """
    for path in (OBJSTORE, LAUNCH):
        code = "\n".join(_code_lines(path))
        assert "az storage blob sync" not in code, f"{path.name} uses az storage blob sync"
        assert "azcopy sync" not in code, f"{path.name} uses azcopy sync"


def test_every_azcopy_copy_of_a_directory_passes_recursive():
    """`azcopy copy` defaults --recursive to FALSE and exits 0 having uploaded almost nothing."""
    code = _code_lines(OBJSTORE)
    found = 0
    for i, line in enumerate(code):
        if "azcopy copy" not in line:
            continue
        # Single-file copies are the one legitimate non-recursive case.
        if "${local_file}" in line:
            continue
        found += 1
        block = " ".join(code[i : i + 3])
        assert "--recursive" in block, f"a tree copy lacks --recursive: {line.strip()}"
    assert found >= 3, f"expected several directory copies to check, found {found}"


def test_bash_syntax_is_valid():
    for path in (OBJSTORE, LAUNCH):
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
