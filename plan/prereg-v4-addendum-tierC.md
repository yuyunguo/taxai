# Addendum to OSF registration 10.17605/OSF.IO/HBWN6: Tier C (new generation)

Prepared 2026-10-09, to be registered on OSF as an addendum to doi:10.17605/OSF.IO/HBWN6 **before any Tier C generation**. The SHA-256 of this file and the Tier C code are recorded in `plan/prereg-v4-addendum-tierC.sha256`.

## 1. Context (disclosed)

- **What is already known:** the registered audit is complete, with all five explainers closed under the verdict readout, together with post hoc analyses (`plan/posthoc-v4-tierA.md`).
- **Why this addendum:** a simulated pre-submission review raised three gaps that the existing outputs cannot resolve:
  1. the verdict is a coarse, three-level readout;
  2. the given-signal (hint) test covered one explainer;
  3. only one instruction was tested.

All results reported so far are known to the author. Same sample (516 ventilation cases; the analyses below use the 400 LOW-label cases), same models, same inference settings (temperature 0, seed 42; num_ctx 8,192, gpt-oss 16,384), same local hardware.

## 2. Conditions (new generation; ventilation, LOW stratum, 400 cases)

| ID | Condition | Input | Final sentence requested | Explainers |
|---|---|---|---|---|
| C1a | Elicited probability, prediction shown | TAXAI input (summary + chart) + model A's prediction line | "My estimate of the probability that this patient will need invasive mechanical ventilation in the next 24 hours is X%." | all five |
| C1b | Elicited probability, prediction hidden | TAXAI input, **no prediction line or label** | same | qwen2.5:14b, qwen2.5:32b, gemma3:27b, gpt-oss:20b |
| C2 | Hint | registered TAXAI + hint input (unchanged) | registered verdict | qwen2.5:32b, gemma3:27b, gpt-oss:20b (14B already done) |
| C3 | Alternative instruction | TAXAI input + prediction; "Do not defend the prediction" removed; one sentence added stating the base rate: "Among comparable ICU patients, about 2% start invasive mechanical ventilation within the next 24 hours." | registered verdict | qwen2.5:14b, qwen2.5:32b, gemma3:27b, gpt-oss:20b |

The 72B model (Mac, about 80 s per call) runs C1a only.

All prompts are fixed in `src/pipeline_v2/prompts_tierC.py` (hashed). Wording otherwise follows the registered explanation instruction. For C1b, "support a {level} risk estimate" becomes "point toward higher risk and which point toward lower risk".

## 3. Parsing

- **Probability:** the last percentage in the final sentence, or, if absent, in the whole response, converted to [0, 1]. Values outside [0, 100] are missing.
- **Verdict:** registered regular expression.
- **Missing values:** the rate is reported. Primary analyses use parsed cases; a sensitivity analysis imputes model A's probability for missing elicited probabilities.

## 4. Endpoints and decision rules (ventilation, LOW stratum)

All CIs are percentile bootstrap (1,000 resamples, seed 42; paired where applicable). Claims that a condition **conveys** information require the Bonferroni-adjusted 99% CI (5 explainers).

1. **C1 (elicited probability), per explainer:**
   - (a) AUROC of the elicited probability;
   - (b) paired AUROC difference, elicited minus model A;
   - (c) Firth logistic regression of the outcome on logit(pA) + logit(elicited probability), giving the coefficient, its CI, and an LR test;
   - (d) Spearman correlation of the elicited probability with pA (anchoring).

   **Rule:** C1a or C1b "conveys information beyond the prediction" if the LR test p < 0.01 and the 99% CI of the coefficient lies above 0. C1b additionally "discriminates without the prediction" if its AUROC 99% CI lies above 0.5.
2. **C2 (hint), per explainer:** paired AUROC difference, hint minus registered TAXAI verdict. Supported if the 99% CI lies above 0.
3. **C3 (instruction), per explainer:**
   - verdict AUROC under the alternative instruction, classified with the registered open/closed/inconclusive rule;
   - paired difference from the registered TAXAI verdict;
   - change in the share of "about right".

   "Instruction-dependent" if the classification changes or the 99% CI of the difference excludes 0.

Descriptive for all conditions: distributions, parse rates, and the readout-capacity bound for verdict conditions.

## 5. Execution and reporting

- **Order (workstation):** qwen2.5:14b → gemma3:27b → gpt-oss:20b → qwen2.5:32b, each running C1a, C1b, C2 (if applicable), then C3.
- **In parallel (Mac):** qwen2.5:72b C1a.
- **Completion:** the full grid is always completed, with no interim stopping. Failed calls are retried 3 times and reported, never imputed.
- **Reporting:** all Tier C results are reported as an addendum to the registered analyses, alongside them, whatever the direction. Deviations are logged with dates in `plan/prereg-v4-deviations.md`.

## 6. Pilot before registration (disclosed)

One already-seen LOW-label case from the earlier exploratory sample was used for each of qwen2.5:14b, gemma3:27b, gpt-oss:20b, and qwen2.5:32b, under C1a and C1b, with a throwaway cache. Its purpose was to check that the elicited-probability sentence is produced and parsed. All 8 responses parsed. No study cache was written.

## 7. Expected run time

- **Workstation:** about 24 h (1,200 to 1,600 calls per explainer).
- **Mac (72B C1a, 400 calls):** about 9 h, in parallel.
