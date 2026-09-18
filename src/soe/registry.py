"""Load and validate the frozen registries.

The freeze hash is the only thing standing between the experiment and a silent seed remap
caused by reordering a YAML file. See ``configs/registry/FREEZE.md``.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import yaml

from soe.config import DatasetSpec, ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "configs" / "registry"
FREEZE_FILE = REGISTRY_DIR / "FREEZE.md"


class RegistryFreezeError(RuntimeError):
    """Raised when a registry's index assignment no longer matches the committed hash."""


def _index_hash(pairs: list[tuple[str, int]]) -> str:
    blob = "\n".join(f"{k}={i}" for k, i in sorted(pairs))
    return hashlib.sha256(blob.encode()).hexdigest()


@lru_cache(maxsize=1)
def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    raw = yaml.safe_load((path or REGISTRY_DIR / "models.yaml").read_text())
    specs = [ModelSpec.model_validate(m) for m in raw["models"]]
    out: dict[str, ModelSpec] = {}
    seen_idx: dict[int, str] = {}
    for s in specs:
        if s.key in out:
            raise ValueError(f"duplicate model key {s.key!r}")
        if s.seed_idx in seen_idx:
            raise ValueError(
                f"seed_idx {s.seed_idx} used by both {seen_idx[s.seed_idx]!r} and {s.key!r}; "
                f"seed_idx must be unique or seeds collide across models"
            )
        seen_idx[s.seed_idx] = s.key
        out[s.key] = s
    for s in out.values():
        if s.base_of and s.base_of not in out:
            raise ValueError(f"{s.key}: base_of={s.base_of!r} is not a registered model")
    return out


@lru_cache(maxsize=1)
def load_datasets(path: Path | None = None) -> dict[str, DatasetSpec]:
    raw = yaml.safe_load((path or REGISTRY_DIR / "datasets.yaml").read_text())
    specs = [DatasetSpec.model_validate(d) for d in raw["datasets"]]
    out: dict[str, DatasetSpec] = {}
    seen_idx: dict[int, str] = {}
    for s in specs:
        if s.key in out:
            raise ValueError(f"duplicate dataset key {s.key!r}")
        if s.seed_idx in seen_idx:
            raise ValueError(
                f"seed_idx {s.seed_idx} used by both {seen_idx[s.seed_idx]!r} and {s.key!r}"
            )
        seen_idx[s.seed_idx] = s.key
        out[s.key] = s
    return out


def current_hashes() -> dict[str, str]:
    return {
        "models_sha256": _index_hash([(m.key, m.seed_idx) for m in load_models().values()]),
        "datasets_sha256": _index_hash([(d.key, d.seed_idx) for d in load_datasets().values()]),
    }


def frozen_hashes() -> dict[str, str]:
    text = FREEZE_FILE.read_text()
    out = {}
    for name in ("models_sha256", "datasets_sha256"):
        m = re.search(rf"^{name}:\s*(\S+)\s*$", text, re.MULTILINE)
        if not m:
            raise RegistryFreezeError(f"{name} not found in {FREEZE_FILE}")
        out[name] = m.group(1)
    return out


def check_freeze(strict: bool = True) -> None:
    """Raise if the registries' index assignment drifted from the committed hash."""
    cur, froz = current_hashes(), frozen_hashes()
    if froz["models_sha256"] == "PENDING" or froz["datasets_sha256"] == "PENDING":
        if strict:
            raise RegistryFreezeError(
                "registries are not frozen yet -- run `soe registry freeze` and commit the hashes"
            )
        return
    bad = [k for k in cur if cur[k] != froz[k]]
    if bad:
        raise RegistryFreezeError(
            f"registry index hash mismatch for {bad}. A key was reordered, renumbered or "
            f"removed. This remaps EVERY seed in the experiment and will produce duplicate "
            f"samples against existing shards. Expected {froz}, computed {cur}. "
            f"See configs/registry/FREEZE.md."
        )


def write_freeze() -> dict[str, str]:
    """Re-freeze the registries, rewriting the hash block in FREEZE.md."""
    cur = current_hashes()
    text = FREEZE_FILE.read_text()
    block = (
        "<!-- BEGIN FROZEN HASHES -- machine-managed, do not hand-edit -->\n"
        f"models_sha256: {cur['models_sha256']}\n"
        f"datasets_sha256: {cur['datasets_sha256']}\n"
        "<!-- END FROZEN HASHES -->"
    )
    new = re.sub(
        r"<!-- BEGIN FROZEN HASHES.*?<!-- END FROZEN HASHES -->",
        block,
        text,
        flags=re.DOTALL,
    )
    FREEZE_FILE.write_text(new)
    return cur
