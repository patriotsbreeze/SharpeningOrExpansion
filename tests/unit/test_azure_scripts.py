"""The Azure layer, which has never run against a real subscription.

Until now nothing under `azure/` was referenced by any test -- not even `bash -n`. It is also
the most expensive code here to get wrong, because a malformed flag surfaces only once GPUs are
billing. These tests run the real scripts with a fake `az` on PATH that records its argv, so
flag construction is checked at laptop speed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = [
    REPO / "azure" / "launch.sh",
    REPO / "azure" / "reap.sh",
    REPO / "azure" / "teardown.sh",
    REPO / "azure" / "bootstrap.sh",
    REPO / "scripts" / "azwrap.sh",
]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")

MUTATING = (
    ("group", "create"), ("group", "delete"),
    ("vm", "create"), ("vm", "delete"),
    ("role", "assignment", "create"),
)


def _fake_az(tmp_path: Path) -> tuple[Path, Path]:
    """A stub `az` that records argv and answers the read-only queries launch.sh makes."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "az.log"
    (bindir / "az").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        'case "$1 $2" in\n'
        '  "account show") echo "00000000-0000-0000-0000-000000000000" ;;\n'
        '  "storage account") echo "soe-data" ;;\n'
        '  "vm show") echo "11111111-1111-1111-1111-111111111111" ;;\n'
        '  "vm list") : ;;\n'
        '  "resource list") : ;;\n'
        "esac\n"
        "exit 0\n"
    )
    (bindir / "az").chmod(0o755)
    return bindir, log


def _launch(tmp_path: Path, **env_over) -> tuple[subprocess.CompletedProcess, list[str]]:
    bindir, log = _fake_az(tmp_path)
    key = tmp_path / "id_rsa.pub"
    key.write_text("ssh-rsa AAAA test\n")
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "DRY_RUN": "1",
        "LOCATION": "eastus2",
        "STORAGE_ACCOUNT": "soeacct",
        "AZ_CONTAINER_URL": "https://soeacct.blob.core.windows.net/soe",
        "SSH_KEY": str(key),
        "FLEET": "3",
        "RG": "soe-test",
        **env_over,
    }
    r = subprocess.run(
        ["bash", str(REPO / "azure" / "launch.sh")], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=180,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax_is_valid(script):
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_dry_run_mutates_nothing(tmp_path):
    r, calls = _launch(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    for call in calls:
        parts = call.split()
        for verb in MUTATING:
            assert tuple(parts[: len(verb)]) != verb, f"DRY_RUN executed a mutation: {call}"
    assert "[dry-run]" in r.stdout


def test_spot_flags_are_present_for_spot(tmp_path):
    r, _ = _launch(tmp_path, PRIORITY="Spot")
    creates = [ln for ln in r.stdout.splitlines() if "vm create" in ln]
    assert len(creates) == 3, r.stdout
    for ln in creates:
        assert "--priority Spot" in ln
        assert "--eviction-policy Delete" in ln
        assert "--max-price -1" in ln


def test_spot_flags_are_absent_for_pay_as_you_go(tmp_path):
    """The runbook's budget-safe fallback. az rejects these flags outside Spot, so passing
    them unconditionally made PRIORITY=Regular impossible to launch at all."""
    r, _ = _launch(tmp_path, PRIORITY="Regular")
    creates = [ln for ln in r.stdout.splitlines() if "vm create" in ln]
    assert len(creates) == 3, r.stdout
    for ln in creates:
        assert "--priority Regular" in ln
        assert "--eviction-policy" not in ln
        assert "--max-price" not in ln


def test_every_vm_gets_its_own_worker_index_and_a_shared_lead(tmp_path):
    r, _ = _launch(tmp_path)
    assert "lead worker" in r.stdout
    # The custom-data is written to a temp file, so assert on the reported election instead.
    assert "lead worker (publishes the problem manifest): 0" in r.stdout


def test_identity_and_role_scope_are_requested(tmp_path):
    r, _ = _launch(tmp_path)
    creates = [ln for ln in r.stdout.splitlines() if "vm create" in ln]
    assert all("--assign-identity" in ln for ln in creates)
    roles = [ln for ln in r.stdout.splitlines() if "role assignment create" in ln]
    assert len(roles) == 3
    assert all("Storage Blob Data Contributor" in ln for ln in roles)
    assert all("/providers/Microsoft.Storage/storageAccounts/soeacct" in ln for ln in roles)


def test_public_ip_and_disks_are_marked_for_deletion(tmp_path):
    """az vm delete does not cascade; an orphaned Standard IP bills indefinitely."""
    r, _ = _launch(tmp_path)
    for ln in [x for x in r.stdout.splitlines() if "vm create" in x]:
        assert "--os-disk-delete-option Delete" in ln
        assert "--nic-delete-option Delete" in ln
        assert "--public-ip-address-delete-option Delete" in ln


def test_teardown_refuses_non_interactively_without_force(tmp_path):
    bindir, _ = _fake_az(tmp_path)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "RG": "soe-test"}
    r = subprocess.run(
        ["bash", str(REPO / "azure" / "teardown.sh")], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
    )
    assert r.returncode != 0
    assert "FORCE=1" in r.stderr


def test_teardown_does_not_report_an_az_failure_as_nothing_to_do(tmp_path):
    """Reporting a failed call as "nothing to tear down" leaves GPU VMs billing."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "az").write_text("#!/usr/bin/env bash\necho 'AuthorizationFailed' >&2\nexit 1\n")
    (bindir / "az").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "RG": "soe-test", "FORCE": "1"}
    r = subprocess.run(
        ["bash", str(REPO / "azure" / "teardown.sh")], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0
    assert "NOT assuming it is empty" in r.stderr


def test_reap_uses_the_shared_url_builder(tmp_path):
    """A private copy of the SAS-splicing logic is not covered by the objstore tests."""
    text = (REPO / "azure" / "reap.sh").read_text()
    code = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert not any(ln.strip().startswith("_url()") for ln in code)
    assert "_az_url" in "\n".join(code)


def _doctor(tmp_path: Path, az_body: str, **env_over) -> subprocess.CompletedProcess:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "az").write_text("#!/usr/bin/env bash\n" + az_body)
    (bindir / "az").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "PYTHONPATH": str(REPO / "src"),
        **env_over,
    }
    env.pop("AZ_CONTAINER_URL", None)
    return subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-m", "soe.cli", "doctor", "--azure"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )


HEALTHY = """
case "$1 $2" in
  "account show") echo '{"name":"sub","state":"Enabled","subscriptionPolicies":{"quotaId":"PayAsYouGo_2014-09-01"}}' ;;
  "vm list-skus") echo '[{"family":"standardNCADSH100v5Family","restrictions":[]}]' ;;
  "vm list-usage") echo '[{"name":{"value":"standardNCADSH100v5Family"},"limit":80},{"name":{"value":"lowPriorityCores"},"limit":80}]' ;;
esac
exit 0
"""


@pytest.mark.skipif(
    not (REPO / ".venv" / "bin" / "python").exists(), reason="needs the project venv"
)
class TestDoctorAzure:
    def test_healthy_account_passes(self, tmp_path):
        r = _doctor(tmp_path, HEALTHY)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "azure preflight OK" in r.stdout

    def test_free_trial_is_a_blocker(self, tmp_path):
        """Free Trial cannot raise GPU quota and is excluded from Spot entirely."""
        body = HEALTHY.replace("PayAsYouGo_2014-09-01", "FreeTrial_2014-09-01")
        r = _doctor(tmp_path, body)
        assert r.returncode == 1
        assert "Free Trial" in r.stdout

    def test_a_restricted_sku_is_a_blocker(self, tmp_path):
        """NotAvailableForSubscription is a third failure mode, distinct from quota."""
        body = HEALTHY.replace(
            '"restrictions":[]',
            '"restrictions":[{"reasonCode":"NotAvailableForSubscription"}]',
        )
        r = _doctor(tmp_path, body)
        assert r.returncode == 1
        assert "restricted" in r.stdout
        assert "NotAvailableForSubscription" in r.stdout

    def test_zero_spot_quota_is_a_blocker(self, tmp_path):
        """On-demand and spot are separate pools needing separate requests."""
        body = HEALTHY.replace('{"name":{"value":"lowPriorityCores"},"limit":80}',
                               '{"name":{"value":"lowPriorityCores"},"limit":0}')
        r = _doctor(tmp_path, body)
        assert r.returncode == 1
        assert "spot quota = 0" in r.stdout

    def test_not_logged_in_fails_clearly(self, tmp_path):
        r = _doctor(tmp_path, "echo 'Please run az login' >&2\nexit 1\n")
        assert r.returncode == 1
        assert "az login" in r.stdout
