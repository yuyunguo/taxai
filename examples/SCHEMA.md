# Data schema (Gate 2 inputs and caches)

The real files contain MIMIC-IV-derived values and are not distributed. They are available only to credentialed PhysioNet users. The synthetic files in this folder use the same formats.

## `imv_sample.csv` (one row per case)

| Column | Type | Meaning |
|---|---|---|
| `hadm_id` | int | Admission identifier (synthetic files use 900000001 and up) |
| `y` | 0/1 | Invasive ventilation started within 24 h after the prediction time |
| `split` | str | `val` or `test` (temporal split) |
| `pA`, `pB` | float | Model A and model B predicted probabilities |
| `level` | str | `LOW` or `HIGH` (model A threshold 0.0647) |
| `in_prior_sample` | 0/1 | Case was in the earlier exploratory sample |
| `rule_score` | -1/+1 | Rule control: +1 ("too low") if pB > pA, else -1 |

## Explanation caches (`gen_imv_<model>.jsonl`; append-only, one JSON object per line)

| Field | Meaning |
|---|---|
| `method` | Condition: `static_llm`, `taxai`, `taxai_hint` |
| `hadm_id` | Admission identifier |
| `response` | Full model response; the verdict is parsed from its final sentence |

Real caches also record the prompt, model digest, and inference options.

## Addendum caches (`gen_tierC_<model>.jsonl`)

Same record format as above. `method` is the addendum condition (`C1a`, `C1b`, `C2`, `C3`). For C1a and C1b, the elicited probability is parsed from the final sentence ("… is X%.") by `parse_elicited` in `src/pipeline_v2/prompts_tierC.py`.

## What the synthetic demo covers

- Verdict parsing and the registered H1 rule (bootstrap CIs, open/closed/inconclusive).
- The rule control.
- Addendum C1: probability parsing and the Firth-penalized test of information beyond model A (likelihood-ratio test, 99% CI of the coefficient).

The synthetic data contain deliberately planted signal, so their results illustrate the code path; they are not study results.
