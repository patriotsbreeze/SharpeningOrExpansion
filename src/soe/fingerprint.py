"""Engine fingerprint, recorded per shard.

Seeds pin sampling decisions *given logits*, but logits are not bitwise invariant to batch
composition, prefix-cache hits or attention backend. So reproducibility is distributional, not
byte-identical -- and the honest way to support that claim in the paper is to record exactly
what produced each shard and assert one engine major version per arm.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def engine_fingerprint(backend_fp: str, prefix_caching: bool = True) -> str:
    return f"{backend_fp}|pfxcache={int(prefix_caching)}"
