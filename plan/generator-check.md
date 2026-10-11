# Generator-capability check (specified before running)

Specified 2026-10-02. **Post hoc**; reported as exploratory.

## Question

Is the generator's inability to tell when the AI is wrong a capability limit of the 14B model? In v3, the qwen2.5:14b verdict's AUROC for death was about 0.50 within each label stratum. The information-ceiling test shows the timeline adds little beyond the model's aggregates. But the model's own probability does carry within-label signal (AUROC 0.70 within HIGH and 0.84 within LOW), so a capable explainer could at least convey that.

## Design

- **Generator:** qwen2.5:32b (Q4_K_M), with the identical `EXPLAIN_V3` prompt, inputs, decoding settings (temperature 0, seed 42), and verdict parsing as v3.
- **Cases:** from the v3 main sample, every case with an incorrect label (93 HIGH, 23 LOW), plus an equal number of correct-label cases from the same label stratum, drawn at random (seed 11). That gives 232 cases, enriched so that within-stratum discrimination can be estimated.
- **Arms:** Static LLM and TAXAI. No judge is run; this isolates the generator.
- **Comparator:** the existing qwen2.5:14b v3 verdicts on the same cases.

## Endpoints

1. **Verdict discrimination:** within each label stratum, the AUROC for in-hospital death of the verdict score (too high = −1, about right = 0, too low = +1), with bootstrap 95% CIs. The 32B and 14B models are compared on the same cases (paired bootstrap).
2. **Verdict accuracy** (Amendment A1 definition) and the verdict distribution.

## Decision rule (fixed now)

A capability bottleneck is supported if the 32B within-HIGH AUROC is ≥ 0.60 with its CI excluding 0.50, and the paired difference from 14B has a CI excluding 0. HIGH is the stratum with enough incorrect labels.

If 32B also stays near 0.50, the limit is not specific to model size in this family. The explanation's inability to convey the AI's errors then stands as a robust finding.
