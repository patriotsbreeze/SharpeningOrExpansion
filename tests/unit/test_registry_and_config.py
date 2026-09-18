from __future__ import annotations

from datetime import date

import pytest

from soe.config import ModelSpec, SamplingSpec
from soe.registry import check_freeze, current_hashes, frozen_hashes, load_datasets, load_models


def test_registries_load_and_are_frozen():
    check_freeze(strict=True)
    assert current_hashes() == frozen_hashes()


def test_seed_indices_unique_and_lineage_resolves():
    models = load_models()
    assert len({m.seed_idx for m in models.values()}) == len(models)
    for m in models.values():
        if m.base_of:
            assert m.base_of in models
    datasets = load_datasets()
    assert len({d.seed_idx for d in datasets.values()}) == len(datasets)


def test_olmo_step_checkpoints_have_distinct_seed_indices():
    """Each RL step revision is its own model, or the whole sweep draws correlated samples."""
    models = load_models()
    steps = [m for m in models.values() if m.role == "rl_ckpt"]
    assert len(steps) >= 2
    assert len({m.seed_idx for m in steps}) == len(steps)
    assert len({m.rl_step for m in steps}) == len(steps)


def test_qwen_base_context_is_4096():
    """The C5 confound in person. If this ever changes, the token budgeting must be revisited."""
    assert load_models()["qwen25m7b_base"].native_context_len == 4096


def test_r1_distill_is_not_labelled_rlvr():
    """It is SFT on R1 traces; pooling it with the zero-RL arms would conflate treatments."""
    assert load_models()["r1_distill_q7b"].role == "distill"


def test_context_beyond_native_requires_rope_scaling():
    with pytest.raises(ValueError, match="rope_scaling"):
        ModelSpec(
            key="x", seed_idx=0, hf_id="a", revision="main", family="f", role="base",
            native_context_len=4096, context_len=32768, release_date=date(2024, 1, 1),
            prompt_variants=["zeroshot_boxed"],
        )


def test_sampling_id_is_chunk_invariant_but_temperature_sensitive():
    a = SamplingSpec(n_total=256, chunk_size=16, temperature=1.0, max_new_tokens=3196)
    b = SamplingSpec(n_total=256, chunk_size=32, temperature=1.0, max_new_tokens=3196)
    c = SamplingSpec(n_total=256, chunk_size=16, temperature=0.6, max_new_tokens=3196)
    assert a.sampling_id == b.sampling_id
    assert a.sampling_id != c.sampling_id


def test_n_total_must_divide_evenly_into_chunks():
    with pytest.raises(ValueError, match="multiple of chunk_size"):
        SamplingSpec(n_total=100, chunk_size=32, temperature=1.0, max_new_tokens=512)


def test_extra_keys_rejected():
    with pytest.raises(Exception):
        SamplingSpec(n_total=64, chunk_size=16, temperature=1.0, max_new_tokens=512, typo=1)
