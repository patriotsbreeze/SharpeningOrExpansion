from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from soe.seeding import (
    MAX_DATASET, MAX_EPOCH, MAX_MODEL, MAX_PROBLEM, MAX_SAMPLE,
    SeedKey, seed_base, seed_for, unpack,
)

# Golden vectors. These exist to catch an accidental change to the bit layout, which would
# silently remap every seed in the experiment and produce duplicates against existing shards.
GOLDEN = [
    ((0, 0, 0, 0, 0), 0),
    ((0, 0, 0, 0, 1), 1),
    ((0, 0, 0, 1, 0), 1 << 20),
    ((0, 0, 1, 0, 0), 1 << 30),
    ((0, 1, 0, 0, 0), 1 << 36),
    ((1, 0, 0, 0, 0), 1 << 42),
    ((3, 5, 2, 17, 999), 13539902227431),
]


@pytest.mark.parametrize("key,expected", GOLDEN)
def test_golden_vectors(key, expected):
    assert seed_for(*key) == expected


@given(
    st.integers(0, MAX_EPOCH - 1), st.integers(0, MAX_MODEL - 1),
    st.integers(0, MAX_DATASET - 1), st.integers(0, MAX_PROBLEM - 1),
    st.integers(0, MAX_SAMPLE - 1),
)
@settings(max_examples=500, deadline=None)
def test_roundtrip(e, m, d, p, s):
    assert unpack(seed_for(e, m, d, p, s)) == SeedKey(e, m, d, p, s)


def test_injective_over_dense_subgrid():
    seen = {}
    for m in range(4):
        for d in range(4):
            for p in range(12):
                for s in range(16):
                    k = (0, m, d, p, s)
                    v = seed_for(*k)
                    assert v not in seen, f"collision {k} vs {seen[v]}"
                    seen[v] = k


def test_blocks_are_disjoint_across_problems():
    """Common random numbers across problems must be structurally impossible."""
    n = 1000
    a = {seed_for(0, 1, 1, 5, s) for s in range(n)}
    b = {seed_for(0, 1, 1, 6, s) for s in range(n)}
    assert not (a & b)


def test_chunk_invariance():
    """The seed depends only on the GLOBAL sample_idx, never on how it was chunked."""
    flat = [seed_for(0, 2, 3, 7, s) for s in range(64)]
    for chunk in (8, 16, 32):
        rebuilt = []
        for c in range(64 // chunk):
            rebuilt += [seed_for(0, 2, 3, 7, s) for s in range(c * chunk, (c + 1) * chunk)]
        assert rebuilt == flat


def test_seed_fits_in_int64():
    assert seed_for(MAX_EPOCH - 1, MAX_MODEL - 1, MAX_DATASET - 1,
                    MAX_PROBLEM - 1, MAX_SAMPLE - 1) < 2**63


@pytest.mark.parametrize("bad", [
    (MAX_EPOCH, 0, 0, 0, 0), (0, MAX_MODEL, 0, 0, 0), (0, 0, MAX_DATASET, 0, 0),
    (0, 0, 0, MAX_PROBLEM, 0), (0, 0, 0, 0, MAX_SAMPLE), (-1, 0, 0, 0, 0),
])
def test_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        seed_for(*bad)


def test_seed_base_is_block_start():
    assert seed_for(1, 2, 3, 4, 0) == seed_base(1, 2, 3, 4)
