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


def test_push_snapshots_markers_before_uploading_anything():
    """Two passes alone are not enough; the marker list must be snapshotted first.

    Pushes run every SYNC_INTERVAL while generation is still writing, so the tree is not
    quiescent. If the markers pass re-enumerated the tree, a chunk completing after the first
    pass walked its directory would have its marker uploaded with its shard left behind -- an
    eviction then destroys the only copy, and the next worker skips the chunk forever on the
    strength of that marker. Snapshotting first means every marker sent is one whose shard
    provably existed before the first pass enumerated.

    The behavioural counterpart is tests/integration/test_push_ordering.py, which mutates the
    tree mid-push and asserts the store never holds an orphaned marker.
    """
    code = _code_lines(OBJSTORE)
    start = next(i for i, ln in enumerate(code) if ln.startswith("objstore_push()"))
    end = next(i for i, ln in enumerate(code[start + 1 :], start + 1) if ln.startswith("}"))
    body = code[start : end + 1]
    joined = "\n".join(body)

    snap = next(i for i, ln in enumerate(body) if "snapshot" in ln and "find" in ln)
    first_upload = next(
        i
        for i, ln in enumerate(body)
        if any(c in ln for c in ("azcopy copy", "aws s3 sync", "_fs_copy_all"))
    )
    assert snap < first_upload, (
        "the marker snapshot must be taken BEFORE any upload begins, or a marker written "
        "mid-push can be sent without its shard"
    )
    assert "-name '*.done.json'" in body[snap]

    # Every backend's first pass must exclude markers, and its second must be driven by the
    # snapshot rather than a fresh enumeration.
    assert '--exclude-pattern "*.done.json"' in joined, "azure pass 1 must exclude markers"
    assert '--list-of-files "${snapshot}"' in joined, "azure pass 2 must use the snapshot"
    assert '--exclude "*.done.json"' in joined, "s3 pass 1 must exclude markers"
    assert 'read -r rel' in joined, "s3/file pass 2 must iterate the snapshot"


def test_push_propagates_copy_failures():
    """`find -exec` discards per-file exit status, so a push can copy nothing and return 0.

    That is the worst failure available here: the run looks durable, the VM is evicted, and
    everything it produced is gone.
    """
    code = "\n".join(_code_lines(OBJSTORE))
    assert "-exec cp" not in code, "find -exec swallows the exit status of every copy"
    assert "_fs_copy_all" in code, "the file backend must use the error-propagating helper"


def test_manifest_publication_is_write_once():
    """problem_idx is a position in the manifest and feeds every seed.

    A second publisher overwriting it mid-run repartitions the experiment underneath the
    workers already generating against the first version. Write-once also makes leadership
    takeover safe: whoever wins, everyone consumes the winner's bytes.
    """
    code = "\n".join(_code_lines(OBJSTORE))
    assert "objstore_publish_once" in code
    assert "--overwrite=false" in code, "the manifest must not be overwritable"
    launch = "\n".join(_code_lines(LAUNCH))
    assert "objstore_publish_once" in launch
    assert "objstore_push_file" not in launch, "the overwriting publisher must not be used"


def test_marker_pull_never_fetches_shards():
    """Resume must stay cheap: markers are kilobytes, shards are hundreds of GB."""
    code = _code_lines(OBJSTORE)
    start = next(i for i, ln in enumerate(code) if ln.startswith("objstore_pull_markers()"))
    end = next(i for i, ln in enumerate(code[start + 1 :], start + 1) if ln.startswith("}"))
    body = " ".join(code[start : end + 1])
    assert '--include-pattern "*.done.json"' in body
    assert '--include "*.done.json"' in body


def test_per_vm_verify_is_not_deep():
    """A resumed VM holds everyone's markers but only its own shards.

    Deep verify checksums every shard it has a marker for, so on a resumed VM it would fail
    on each absent one and bury any real problem. The deep pass belongs on the analysis
    machine against the complete tree.
    """
    code = " ".join(_code_lines(LAUNCH))
    assert "soe.cli verify" in code
    assert "--no-deep" in code, "the per-VM verify must be marker-level only"


def test_background_helpers_do_not_inherit_the_callers_stdout():
    """A background job that inherits stdout keeps the pipe open after the script exits.

    Any caller that captures output -- CI, `az vm run-command`,
    subprocess.run(capture_output=True) -- then blocks until the helper happens to die rather
    than when the run finishes. The periodic sync sleeps for minutes at a time, so in practice
    that reads as a hang. Both helpers must redirect their own output.
    """
    code = _code_lines(LAUNCH)
    backgrounded = [ln for ln in code if ln.rstrip().endswith("&")]
    assert backgrounded, "expected some backgrounded helpers to check"
    for ln in backgrounded:
        # The generate workers already redirect to per-worker logs.
        assert ">" in ln, f"backgrounded without redirecting stdout: {ln.strip()}"
