"""Bit-packed seed allocation.

Every sample in the experiment has exactly one integer seed, it is a pure function of the
sample's logical identity, and the map is injective *by construction* -- a bit-field packing,
not a hash, so there is no birthday argument to make.

Three properties carry the experiment's correctness:

1. **Chunk invariance.** The seed depends on the global ``sample_idx`` and never on
   ``chunk_idx`` or ``chunk_size``. Re-chunking after a spot preemption therefore cannot
   reissue a seed, which is what stops duplicate completions from entering ``c`` and ``n``
   and biasing the pass@k estimator.
2. **Block disjointness.** Each ``(epoch, model, dataset, problem)`` owns a contiguous
   ``2**SAMPLE_BITS`` block, so common random numbers across problems are structurally
   impossible and the cluster bootstrap's i.i.d.-problems assumption holds by design.
3. **Frozen indices.** ``model_idx`` and ``dataset_idx`` come from the append-only registry.
   Renumbering the registry remaps every seed in the experiment; ``registry.py`` guards
   against that with a committed hash.

Note on reproducibility: fixed seeds pin *sampling decisions given logits*. vLLM's logits are
not bitwise invariant to batch composition, prefix-cache hits, or attention backend, so seeds
give distributional -- not byte-identical -- reproducibility. The paper must say so, and the
per-shard ``EngineFingerprint`` makes the claim auditable.
"""

from __future__ import annotations

from typing import NamedTuple

EPOCH_BITS = 6
MODEL_BITS = 6
DATASET_BITS = 6
PROBLEM_BITS = 10
SAMPLE_BITS = 20

MAX_EPOCH = 1 << EPOCH_BITS
MAX_MODEL = 1 << MODEL_BITS
MAX_DATASET = 1 << DATASET_BITS
MAX_PROBLEM = 1 << PROBLEM_BITS
MAX_SAMPLE = 1 << SAMPLE_BITS

TOTAL_BITS = EPOCH_BITS + MODEL_BITS + DATASET_BITS + PROBLEM_BITS + SAMPLE_BITS  # 48


class SeedKey(NamedTuple):
    epoch: int
    model_idx: int
    dataset_idx: int
    problem_idx: int
    sample_idx: int


def _check(epoch: int, model_idx: int, dataset_idx: int, problem_idx: int) -> None:
    if not 0 <= epoch < MAX_EPOCH:
        raise ValueError(f"epoch {epoch} out of range [0, {MAX_EPOCH})")
    if not 0 <= model_idx < MAX_MODEL:
        raise ValueError(f"model_idx {model_idx} out of range [0, {MAX_MODEL})")
    if not 0 <= dataset_idx < MAX_DATASET:
        raise ValueError(f"dataset_idx {dataset_idx} out of range [0, {MAX_DATASET})")
    if not 0 <= problem_idx < MAX_PROBLEM:
        raise ValueError(f"problem_idx {problem_idx} out of range [0, {MAX_PROBLEM})")


def seed_base(epoch: int, model_idx: int, dataset_idx: int, problem_idx: int) -> int:
    """Start of the contiguous seed block owned by this (epoch, model, dataset, problem)."""
    _check(epoch, model_idx, dataset_idx, problem_idx)
    packed = epoch
    packed = (packed << MODEL_BITS) | model_idx
    packed = (packed << DATASET_BITS) | dataset_idx
    packed = (packed << PROBLEM_BITS) | problem_idx
    return packed << SAMPLE_BITS


def seed_for(
    epoch: int, model_idx: int, dataset_idx: int, problem_idx: int, sample_idx: int
) -> int:
    """The seed for one sample. Injective over the whole key space."""
    if not 0 <= sample_idx < MAX_SAMPLE:
        raise ValueError(f"sample_idx {sample_idx} out of range [0, {MAX_SAMPLE})")
    return seed_base(epoch, model_idx, dataset_idx, problem_idx) + sample_idx


def unpack(seed: int) -> SeedKey:
    """Inverse of :func:`seed_for`. ``unpack(seed_for(*k)) == k`` for every valid key."""
    if not 0 <= seed < (1 << TOTAL_BITS):
        raise ValueError(f"seed {seed} outside the packed range [0, 2**{TOTAL_BITS})")
    sample_idx = seed & (MAX_SAMPLE - 1)
    rest = seed >> SAMPLE_BITS
    problem_idx = rest & (MAX_PROBLEM - 1)
    rest >>= PROBLEM_BITS
    dataset_idx = rest & (MAX_DATASET - 1)
    rest >>= DATASET_BITS
    model_idx = rest & (MAX_MODEL - 1)
    rest >>= MODEL_BITS
    epoch = rest
    return SeedKey(epoch, model_idx, dataset_idx, problem_idx, sample_idx)
