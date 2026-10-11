# Limited Gate 2 test on invasive ventilation (specified before running)

Specified 2026-10-02. **Exploratory and post hoc.** This follows the limited Gate 1 test, where Gate 1 opened for invasive ventilation (ΔAUROC +0.066).

## Question

When the trajectory carries outcome information beyond the model's aggregates, can an LLM explainer tell which ventilation predictions are wrong? And does giving the explainer the chronological chart (TAXAI) help it more than an aggregate summary (Static LLM)?

## Task and model

- **Outcome:** start of invasive mechanical ventilation within 24 h after the prediction time (ICU admission + 24 h). Patients ventilated earlier are excluded. The cohort and split are as in `plan/gate1-other-outcomes.md`.
- **Model:** model A (XGBoost on the 209 aggregate features) trained for this outcome. Its predicted probability is shown to the explainer.
- **Threshold:** HIGH/LOW at the threshold maximizing F1 on validation.
- **Reference:** model B (aggregates + dynamics) is reported only to indicate the signal that is available.

## Sample (test split, seed 11)

- In each label stratum (HIGH, LOW), all patients with the event are taken, plus an equal number without it, drawn at random.
- **Cap:** at most 120 cases per stratum (60 events + 60 non-events). If a stratum has fewer events, all of them are used.
- **Design:** the sample is enriched so that discrimination within each stratum can be estimated.

## Explanations

- **Generator:** qwen2.5:14b, temperature 0, seed 42, as in Study 3.
- **Prompt:** the Study 3 instruction with "mortality" replaced by "the need for invasive mechanical ventilation in the next 24 hours". It ends with "Overall, the AI estimate appears [too high / about right / too low]".
- **Inputs:** Static LLM gets the aggregate summary; TAXAI gets the summary plus the full chronological chart. No judge is used.

## Endpoints

1. **Verdict AUROC for the event within each stratum** (too high = −1, about right = 0, too low = +1), for Static LLM and TAXAI, with bootstrap 95% CIs (1,000 resamples).
2. **TAXAI − Static LLM paired difference** within each stratum.
3. **Benchmarks:** within-stratum AUROC of the model A probability and of the model B probability.
4. **Incremental value:** logistic regression of the event on logit(model A probability) and the TAXAI verdict score, reporting the verdict coefficient and p.

## Decision rule (fixed now)

- **Gate 2 open (for TAXAI):** within-stratum verdict AUROC ≥ 0.60 with the 95% CI above 0.5, in the stratum with more events.
- **Timeline helps the explainer:** the TAXAI − Static LLM difference has a 95% CI above 0 in that stratum.
