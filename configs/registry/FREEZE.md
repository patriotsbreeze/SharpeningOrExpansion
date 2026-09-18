# Registry freeze rules

`configs/registry/models.yaml` and `datasets.yaml` assign `seed_idx` values that are packed
directly into every sample seed (see `src/soe/seeding.py`). They are **append-only**.

## The rules

1. **Never renumber or reorder.** A one-line reorder remaps every seed in the experiment. Any
   resumed run would then draw *different* samples under the *same* `sample_idx`, producing
   duplicates that make `c` and `n` non-i.i.d. and genuinely bias the pass@k estimator. This is
   the most dangerous silent edit in the codebase.
2. **To add an entry**, append it with the next free `seed_idx` and re-freeze deliberately
   (`soe registry freeze`), committing the updated hash below in the same commit.
3. **To remove an entry**, don't. Mark it unused; its `seed_idx` stays retired forever.
4. `soe verify` refuses to run analysis if the computed hash disagrees with the frozen one.

## Frozen hashes

Computed over the sorted `(key, seed_idx)` pairs of each registry. Regenerate with:

```
soe registry freeze
```

<!-- BEGIN FROZEN HASHES -- machine-managed, do not hand-edit -->
models_sha256: 220481064f24e62ce6ec892c6dfac2a5dd7bd8b0a709ac3afa693fb00947058a
datasets_sha256: 57dae3a2d8b74bce3ea4bbc1497235a9c7c033f7c20378cccce813c7abd175b4
<!-- END FROZEN HASHES -->

## Unverified entries to resolve before the first paid run

`soe doctor --registry` resolves every non-mock `hf_id` / `hf_path` against the Hub and fails
loudly. Run it *before* booking GPU time. Known open questions:

- `MathArena/aime_2026` row count: 30 (I+II) or 15 (I only)? A sibling `aime_2026_I` exists,
  which suggests the unsuffixed repo is the combined set, but this is unconfirmed.
- `MathArena/hmmt_feb_2026` row count.
- `MathArena/final_answer_comps` -- does not match MathArena's usual per-competition naming.
  Confirm it exists and exactly which 2025 competitions it aggregates.
- `opencompass/AIME2025` is split into I/II subsets; pin which config is intended.
- `allenai/Olmo-3-7B-RL-Zero-Math` vs `...RLZero...` casing, and the step-revision format
  (`step_250` vs `stepXXXX-tokensYYYB`).
- Whether `Qwen/Qwen2.5-Math-7B` ships a `chat_template` in `tokenizer_config.json`. If it
  does, applying it to a base model is a silent catastrophe -- the golden prompt fixtures
  under `fixtures/prompts_golden/` exist to catch exactly that.

## Licensing

MathArena datasets are **CC BY-NC-SA 4.0**. The NonCommercial clause matters if any author has
commercial affiliation; ShareAlike may propagate to derived datasets released alongside the
paper. MathArena's *code* is MIT. Cite arXiv 2505.23281 and the 2026 follow-up 2605.00674.
