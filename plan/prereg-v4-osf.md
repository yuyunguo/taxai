# Pre-registration: explainer-ceiling audit of LLM explanations for ICU risk predictions

Prepared 2026-10-08, to be registered on OSF before any of the analyses below are run. The SHA-256 of this file and of the analysis code is recorded in `plan/prereg-v4.sha256`, and the same hashes are entered in the OSF registration.

## 1. Background and what is already known (disclosed)

These results exist and were seen before this registration. They are **not** confirmed by it:

- **Information ceiling (Gate 1), MIMIC-IV test split:** model A (aggregates) vs model B (aggregates plus trajectory dynamics).
  - Mortality: ΔAUROC +0.004 (95% CI −0.001 to 0.008).
  - 24-h invasive mechanical ventilation (IMV): ΔAUROC +0.066 (0.037 to 0.095).
- **Explainer ceiling (Gate 2), exploratory:**
  - IMV, 120 test LOW cases: qwen2.5:14b TAXAI verdict AUROC 0.51 (0.42 to 0.59), against 0.80 for model B.
  - Mortality, 232 cases: within label strata, the qwen2.5:14b verdict AUROCs ranged from 0.46 to 0.57 with CIs including 0.5. qwen2.5:32b TAXAI in LOW was 0.33 (0.22 to 0.43), below chance.
- **Earlier work:** reader (Gate 3) studies with LLM judges on mortality, and an exploratory Gate 3 test on IMV.

This registration extends the Gate 2 evidence with:

- a larger IMV sample;
- positive controls that check the instrument;
- additional explainer families and scales.

The cohort, feature, model, label, and chart code are unchanged from the earlier runs. There are two technical additions: a `--split` option in the chart builder, and an optional host argument in the LLM client.

## 2. Hypotheses

- **H1 (primary):** in the IMV task, among patients labeled LOW risk by model A, an LLM explainer's verdict on the prediction does not usefully discriminate who is ventilated within 24 h.
  - The verdicts are "too high", "about right", and "too low", scored −1, 0, and +1. AUROC counts ties as half.
  - 0.60 is the smallest discrimination we consider useful. Model B reaches about 0.80 in this stratum.
  - Each explainer is classified by its TAXAI verdict AUROC and 95% CI:
    - **Gate 2 open:** AUROC ≥ 0.60 and the CI lower bound > 0.5.
    - **Gate 2 closed:** the CI upper bound < 0.60, which rules out useful discrimination.
    - **Inconclusive:** otherwise.
  - H1 is supported for an explainer if Gate 2 is closed.
- **H2 (instrument validity):** a rule explainer whose verdict is sign(pB − pA) reaches a verdict AUROC with a 95% CI above 0.5 in the same sample. If H2 fails, H1 is uninterpretable and will be reported as such.
- **H3 (using signal that is given):** qwen2.5:14b, given model B's estimate in its input (the hint condition), reaches a higher verdict AUROC than without it (paired difference, 95% CI above 0). This separates transmitting signal that is handed over from finding it in the chart.
- **H4 (failure modes; descriptive, no inferential test):** explainers fail in different ways, either a directional bias (mostly "too high" or mostly "too low") or deference (mostly "about right"). Described by the verdict distribution and the mean verdict score in non-events and in events.

## 3. Data and sample

- **Source and models:**
  - MIMIC-IV v3.2, with the cohort and features of the earlier studies.
  - The IMV label as before (`gate1_outcomes.outcome_labels`): invasive ventilation started within 24 h after the prediction time, excluding patients already ventilated.
  - Models A and B as before (XGBoost, fixed seed).
- **Splits:** the validation split was used only for early stopping and for the HIGH/LOW threshold (maximizing F1), and none of the LLMs saw it. Gate 2 cases are drawn from **validation + test**.
- **Sample (seed 11):**
  - Within each stratum (HIGH, LOW), events are taken, up to 200. When there are more than 200, they are drawn at random.
  - An equal number of non-events is drawn at random.
  - The sample is new and not a superset of the earlier 148 cases. Overlapping cases are kept and flagged, and a sensitivity analysis excludes them.
- **Realized sample:** counts from a sampler dry-run; no explanation or verdict was generated. Threshold 0.0647.
  - LOW pool: 212 events and 10,842 non-events, so 200 + 200 are sampled.
  - HIGH pool: 58 events and 412 non-events, so 58 + 58 are sampled.
  - Total 516 cases (301 validation, 215 test), of which 75 overlap the earlier sample.
- **Mortality (secondary):** the existing 232-case error-enriched sample (the generator check) is reused. Explanations from qwen2.5:14b and qwen2.5:32b already exist and were seen, so only the new explainers' mortality results are new.

## 4. Explainers and conditions

| Explainer | Role | IMV conditions | Mortality |
|---|---|---|---|
| qwen2.5:14b (Q4_K_M) | Base, as in the earlier studies | Static LLM, TAXAI, TAXAI + hint | Existing (seen) |
| qwen2.5:32b (Q4_K_M) | Scale, same family | Static LLM, TAXAI | Existing (seen) |
| qwen2.5:72b-instruct (Q3_K_M) | Scale, same family. Q4 does not fit in 48 GB; served on a researcher-owned Apple-silicon Mac | TAXAI, LOW stratum only | TAXAI |
| gemma3:27b (Q4_K_M) | Second family | Static LLM, TAXAI | Static LLM, TAXAI |
| gpt-oss:20b (MXFP4) | Third family; reasoning model | Static LLM, TAXAI | Static LLM, TAXAI |
| Rule | Positive control | sign(pB − pA) | sign(pB − pA) |

- **Inputs:**
  - Static LLM: the aggregate trajectory summary.
  - TAXAI: the summary plus the full chronological chart (the Study 3 chart).
- **Prompt:** the Study 3 explanation prompt, with "mortality" replaced by the IMV wording for IMV. It ends with "Overall, the AI estimate appears [too high / about right / too low] for this patient."
- **Hint condition:** after the prediction line, the input adds "A second model that also accounts for the timing of recent changes estimates this risk at X%." Nothing else changes.
- **Settings:**
  - All LLMs run locally via Ollama, on the workstation or the researcher-owned Mac on the same LAN. No external service is used (PhysioNet DUA).
  - Temperature 0, seed 42. Other sampling parameters are left at the Ollama model defaults, and the full model information is recorded.
  - gpt-oss runs at its default reasoning effort (medium).
- **Context:** the longest TAXAI prompt over all 17,494 validation and test charts is 6,551 tokens for qwen2.5 and 6,563 for gemma3. num_ctx is 8,192 for all models except gpt-oss, which gets 16,384 because its reasoning tokens count against the same window.
- **Failed calls:** each call is retried up to 3 times. A case that still fails is reported, never imputed.

## 5. Endpoints and analysis

All CIs are percentile bootstrap (1,000 resamples, seed 42).

1. **Primary (H1):** the TAXAI verdict AUROC per explainer in IMV LOW, classified by the H1 rule. Missing verdicts are scored 0, and their rate is reported.
2. **H2:** the rule-control verdict AUROC in IMV LOW.
3. **H3:** the paired AUROC difference, qwen2.5:14b TAXAI + hint minus TAXAI, in IMV LOW.
4. **Secondary:**
   - the TAXAI − Static LLM paired difference per explainer;
   - the scale ladder: qwen2.5 14B → 32B → 72B, with paired differences 72B − 14B and 72B − 32B. The 72B rung differs in quantization (Q3_K_M vs Q4_K_M), which is reported as a caveat;
   - benchmarks: within-stratum AUROC of pA and pB, reported separately for test and validation, because validation probabilities are slightly optimistic through early stopping;
   - logistic regression of the event on logit(pA) and the TAXAI verdict score (the verdict's incremental value);
   - a sensitivity analysis excluding cases that overlap the earlier sample;
   - the HIGH stratum, reported descriptively.
5. **H4:** per explainer, the verdict distribution, the share of "about right", and the mean score among non-events and among events.
6. **Mortality (secondary):** the same endpoints within each label stratum, for the new explainers.

**Multiplicity.** Five LLM explainers are tested on H1. A claim that an explainer *opens* Gate 2 also requires its Bonferroni-adjusted 99% CI lower bound to exceed 0.5. Classifying an explainer as closed needs no adjustment.

## 6. Execution

- **Order:** on the workstation: build (sample and rule control) → 14B → gemma3:27b (IMV, then mortality) → gpt-oss:20b (IMV, then mortality) → 32B. In parallel, on the Mac: 72B (IMV, then mortality).
- **Interrupted runs:** a run cut short by hardware failure is resumed from its cache. If it can't be completed, it is reported as incomplete, never dropped silently.
- **Pilots before registration (already-seen cases from the earlier sample; nothing written to the study caches):**
  - gemma3:27b and gpt-oss:20b: 1 case each, to check that the verdict parses.
  - qwen2.5:72b-instruct Q3_K_M: 3 cases. Prompt reading ran at about 90 tok/s and output at 5.2 tok/s, about 75 s per explanation, and every verdict parsed.
  - qwen:72b (Qwen1.5) was abandoned: it returned degenerate output and the model spilled onto the CPU.

## 7. Deviations, data, and code

- **Deviations:** any deviation is logged with its date in `plan/prereg-v4-deviations.md` and reported in the manuscript.
- **Data:** MIMIC-IV is available through PhysioNet credentialed access. LLM outputs contain patient-derived content and will be shared only with credentialed PhysioNet users.
- **Code:** released with the paper. The hashes in this registration identify the exact version that was run.
