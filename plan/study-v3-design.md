# Study v3: a fair test of trajectory-grounded explanations

Status: **design (before pilot)**, 2026-10-01. The confirmatory analysis plan will be frozen after the pilot and before the main run (`plan/analysis-plan-v3.md`).

## Why v3

The pre-registered v2 study found no reliance difference between TAXAI and Static LLM, but it was a weak test of the timeline hypothesis:

1. **Weak manipulation.** The v2 timeline added no laboratory values beyond the summary (median 0 extra) and only about 5 vital-sign points per case. In effect it tested timestamps, not trajectories.
2. **Ambiguous endpoint.** The accept/override decision does not say in which direction the judge disagrees, and the judge overrode about 96% of labels by default.
3. **No timeline measures.** Nothing checked whether temporal information reached the explanations or mediated decisions.

v2 is reported as conducted. v3 is a new study with its own frozen plan.

## Research question

When the clinician has the full chart, does an explanation grounded in the chronological trajectory move the clinician's mortality estimate toward the truth more than an explanation grounded in an aggregate summary?

## Design decisions (user-approved 2026-10-01)

| Element | v2 | v3 |
|---|---|---|
| Judge's view of the patient | aggregate summary | **full chronological chart, identical in all arms** |
| Primary endpoint | BAR (accept/override vs label correctness) | **ΔBrier: judge's own risk estimate after vs before the explanation** |
| Timeline content | first and last lab values, 8-h vital medians | **all lab values, 2-h vital medians, medication starts, imaging** |
| Generator verdict | consistent / partly / inconsistent (99.8% "partly") | **directional: the AI estimate appears too high / about right / too low** |
| Timeline measures | none | **input contrast (pre-launch gate), temporal content, temporal-claim faithfulness, mediation** |

## Arms (same 500 cases as v2; same base model, threshold, and prompts unless stated)

1. **Empty:** no explanation. Also controls for the effect of simply being asked twice.
2. **Random:** another patient's summary.
3. **SHAP:** top-10 attributions with values.
4. **Static LLM:** explanation generated from the aggregate summary only.
5. **TAXAI:** explanation generated from the full chronological timeline plus summary.

Arms 4 and 5 receive the identical instruction.

## Judge protocol (two turns, one conversation, temperature 0, fixed seed)

- **Turn 1:** the chart (full timeline) and the AI prediction (probability and label). The judge answers "Estimate the probability (0–100%) that this patient dies in hospital." The prompt is identical across arms, so it is run once per case and per judge.
- **Turn 2:** the explanation for the arm (or "No explanation provided"). The judge answers "Re-estimate the probability."

Endpoints per case and arm:
- pre = turn-1 estimate; post = turn-2 estimate (as probabilities);
- ΔBrier = (post − y)² − (pre − y)², where negative is better.

## Endpoints

- **Primary (H1):** mean ΔBrier(TAXAI) − mean ΔBrier(Static LLM). Paired case-level bootstrap (2,000 resamples), two-sided α = 0.05.
- **Secondary** (Bonferroni across the set):
  - ΔBrier, TAXAI vs Empty, Random, and SHAP;
  - post-explanation AUROC;
  - "appropriate shift" rate: the estimate moves toward y by ≥ 5 points;
  - BAR derived from whether the post-estimate agrees with the label's side of the threshold.
- **Timeline measures:**
  - input contrast;
  - temporal content of explanations (time references, sequence terms, verified temporal claims);
  - temporal-claim faithfulness: the value matches the variable at the cited ICU hour ± 1 h.
- **Mediation (secondary, pre-specified):** within TAXAI and Static LLM, the association between the number of verified temporal claims and ΔBrier (mixed or OLS model adjusted for arm).
- **Robustness:** phi4 as a second judge; H1 sign agreement.

## Pilot (40 test cases, disjoint from the 500; seed 7)

The pilot may change only prompt formatting and parsing. It may not change endpoints, arms, or content.

Go/no-go criteria, fixed now:

| # | Criterion | Threshold |
|---|---|---|
| G1 | Input contrast: data points in the TAXAI timeline not in the summary | median ≥ 30 |
| G2 | Judge parse rate (numeric estimate extracted) | ≥ 95% of turns |
| G3 | Judge pre-estimates are informative | SD ≥ 0.10, and AUROC is reported |
| G4 | Prompt fits the context window (num_ctx 8192) | max prompt ≤ 6,000 tokens (estimated) |

If G1 fails, the timeline is enriched before freezing. If G2 or G4 fails, formatting is fixed. G3 failing means the judge cannot do the task; that would be reported, and the study would not proceed as designed.

## Compute

Per case: 2 generations, plus 1 pre-turn and 5 post-turns per judge. That is about 4,000 calls per judge for 500 cases, roughly 4–5 h per judge on the A4500.
