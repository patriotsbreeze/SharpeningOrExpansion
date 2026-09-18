# Sharpening, Expansion, or Artifact?

A confound decomposition of the base-vs-RLVR **pass@k crossover**.

## The question

Yue et al. (arXiv:2504.13837, NeurIPS 2025 Oral) report that RLVR models lead at pass@1 but
are matched or overtaken by their base models as *k* grows, and read that crossover as
evidence that RLVR only *sharpens* an existing distribution rather than *expanding* the
reasoning boundary. Since then at least **seven distinct explanations** for the same crossover
have been proposed. No study measures their relative contributions on a common setup, and four
have never been isolated at all.

| | Explanation | Proposed by | Isolated before? |
|---|---|---|---|
| C1 | Support shrinkage (sharpening) | Yue et al.; Invisible Leash | — (the residual) |
| C2 | Overtraining on already-saturated problems | 2606.15455 | against nothing |
| C3 | **Contamination** inflating base large-k coverage | implied only | **no** |
| C4 | Answer-only scoring rewarding lucky guesses | 2506.14245 | against nothing |
| C5 | **Truncation** (4k base vs. 32k+ descendants) | **nobody** | **no** |
| C6 | **Matched-k ≠ matched-compute** | **nobody** | **no** |
| C7 | **Prompt/template mismatch** | **nobody** | **no** |

## What this repo does

Holds models, problems and sampling budget fixed; removes each confound in turn; reports how
much of the base-minus-RLVR gap at large *k* each one accounts for.

Each confound is an operator on a single precomputed correctness tensor, so each gets an exact
**Shapley value** over the 2⁷ adjustment lattice. Shapley efficiency gives the identity

```
raw gap  =  explained  +  residual
```

where the residual **is** C1 — making genuine support shrinkage an estimated quantity with a
bootstrap confidence interval instead of an unfalsifiable remainder.

The design is deliberately **direction-agnostic**: if the crossover survives every control,
that is a strong confirmation of the sharpening view and the paper still stands.

## Quickstart (CPU, no GPU, no network)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
export PYTHONPATH=src

pytest -q                                  # 78 tests, ~90s
make smoke                                 # full pipeline on the mock backend
```

`make smoke` runs plan → generate → grade → verify → analyze → figures against a backend whose
latent per-problem solve probability is *known*, and asserts the recovered pass@k curve matches
analytic `1-(1-p)^k`. That validates the planner, seeding, sharding, grading, estimator and
bootstrap simultaneously — before a single GPU-hour is spent.

## Running for real

See **[aws/README.md](aws/README.md)**. Short version:

```bash
soe doctor --registry                              # resolve every HF id first. Costs nothing.
./aws/launch.sh                                    # p5.48xlarge spot, on-demand fallback
scripts/launch_node.sh configs/experiments/pilot_gpu.yaml 1        # ~$2 shakeout
scripts/launch_node.sh configs/experiments/stage1_tierA.yaml       # the headline
```

## Design decisions that carry the results

- **Never vLLM's `n>1`.** Every sample is an `n=1` request with an explicit recorded seed. The
  child-seed derivation for `n>1` is an engine implementation detail that has changed across
  releases; prefix caching makes the repeated prompt nearly free anyway. The payoff is that
  `soe verify` can *prove* seed uniqueness from the artifacts rather than assuming it.
- **Bit-packed, chunk-invariant seeding.** A seed depends on the global `sample_idx`, never on
  how work was chunked, so resuming after a spot preemption at a different chunk size cannot
  reissue a seed. Duplicate samples are the one failure that genuinely *biases* the pass@k
  estimator rather than merely widening its interval.
- **Marker-last writes.** A `.done.json` is written only after its shard is durably on disk, so
  its presence means the data is complete. Resume is "plan minus markers" with no lock server.
- **Generation and grading are separate passes.** Graders have bugs, extraction policy is a
  treatment to vary, and C4 needs the raw chain of thought. Re-grading never re-samples.
- **Registries are frozen.** `seed_idx` values are packed into every seed, so reordering a YAML
  file would remap the entire experiment. A committed hash makes that a hard error.
- **Equal *n* per problem is enforced.** pass@k is per-problem unbiased at any *n*, but
  averaging over problems with unequal *n* weights them by differing estimator variance — and
  preemption produces exactly that.

## Evaluation set

| Tier | Contents | ~n | Role |
|---|---|---|---|
| A | AIME 2026, HMMT Feb 2026 | 60 | Headline — postdates every checkpoint's public release |
| B | MathArena 2025 final-answer, HMMT Nov 2025 | 180 | Power for the decomposition |
| Control | AIME 2024 | 30 | **Deliberately contaminated positive control** |
| Anchor | AIME 2025 | 30 | Comparability with published numbers |

The contamination argument is **release-date dominance**, not stated training cutoffs: three of
four model families publish no official cutoff, and the dominant leakage path is post-training
(R1-Distill trains on R1 traces; open RL corpora contain AIME ≤ 2024). "These problems did not
exist when the weights shipped" does not depend on anyone being candid.

AMC is excluded (multiple choice gives a 20% guessing floor that swamps the measured quantity)
and proof-based sets are excluded from pass@k (they need human grading, and LLM-jury noise is
correlated across the k samples, which breaks the estimator).

> **License note.** MathArena datasets are CC BY-NC-SA 4.0. The NonCommercial clause matters
> with commercial affiliation, and ShareAlike may propagate to derived datasets.

## Status

Harness complete and tested; **no GPU run has been executed yet.** Registry ids were verified
against project READMEs but not against the Hub itself — `soe doctor --registry` is the gate,
and `configs/registry/FREEZE.md` lists the open questions.
