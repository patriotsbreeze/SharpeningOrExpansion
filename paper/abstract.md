# ICLR 2027 submission — title + abstract

> **Paste into OpenReview by 18 Sept 2026, 11:59pm AoE.**
> Titles and abstracts can be edited until the full-paper deadline (25 Sept); registering
> now is what preserves the right to submit at all. Written to remain accurate whichever
> direction the results land.

## Title

**Sharpening, Expansion, or Artifact? A Confound Decomposition of the pass@k Crossover in RLVR**

### Backup titles
- *What Actually Causes the pass@k Crossover? Decomposing Seven Explanations on Uncontaminated Competition Math*
- *How Much of the RLVR Reasoning Boundary Is Measurement?*

## Abstract

Whether reinforcement learning with verifiable rewards (RLVR) expands a language model's
reasoning boundary or merely sharpens the base model's existing distribution remains the central
open question in RL for reasoning. The empirical anchor of the sharpening view is a *crossover*:
RLVR models lead at pass@1 but are matched or overtaken by their base models as k grows. Since
that result, at least seven distinct explanations for the crossover have been proposed —
genuine support shrinkage, overtraining on already-saturated problems, benchmark contamination
inflating base-model coverage, answer-only scoring rewarding lucky guesses, context-length
truncation, matched-k comparisons that are not matched-compute, and prompt-template mismatch —
but no study has measured their relative contributions on a common setup, and four have never
been isolated at all.

We provide that measurement. Holding models, problems, and sampling budget fixed, we treat each
explanation as an adjustment operator on a single precomputed correctness tensor and assign each
one an exact Shapley value of the reduction it produces in the base-minus-RLVR gap at large k.
Shapley efficiency yields an additive identity — raw gap = explained + residual — so genuine
support shrinkage becomes an estimated quantity with a confidence interval rather than an
unfalsifiable remainder. We evaluate base/RLVR pairs whose post-training lineage is public,
including a family with matched context length on both sides and released intermediate RL
checkpoints, which lets us trace the crossover as a function of RL steps rather than comparing
two endpoints. To separate contamination from capability we use a difference-in-differences
design over competition problems that postdate every checkpoint's public release, with a
deliberately contaminated pre-release set as a positive control, and we calibrate answer-matching
false positives directly by grading against permuted gold answers.

Our contribution is an adjudication rather than a new training method: a reproducible
decomposition that says how much of a widely cited phenomenon is a property of RLVR and how much
is a property of how it is measured. We release the harness, per-sample generations, and
per-confound attributions.

---

## Notes for the submission form

- **Primary area:** reinforcement learning / evaluation & benchmarks (check the ICLR 2027 area list).
- **Keywords:** RLVR, pass@k, reasoning boundary, contamination, evaluation methodology, Shapley attribution.
- **TL;DR:** We decompose the base-vs-RLVR pass@k crossover into seven candidate confounds and
  report how much of it each explains.
- **Claims deliberately hedged:** nothing above asserts a direction for the result. If the
  crossover survives every control, the abstract still reads correctly and the paper becomes a
  strong confirmation of the sharpening view.
- **Mandatory AI-use statement** is required at full-paper submission and does not count against
  the 9-page limit.
