# Limited Gate 3 test on invasive ventilation (specified before running)

Specified 2026-10-02. **Exploratory and post hoc.** This follows the limited Gate 1 test (open) and Gate 2 test (closed) for 24-h invasive ventilation.

## Question

Do readers update toward the outcome when an explanation carries real information? We separate Gate 3 from the Gate 2 failure with informative positive-control arms.

## Sample

The same 148 cases as `plan/gate2-imv.md`: an outcome-enriched test sample, 28 HIGH and 120 LOW, with the same model A probabilities and threshold.

## Arms (shown in turn 2)

1. **Empty:** "No explanation provided."
2. **Static LLM:** the existing Gate 2 explanation (qwen2.5:14b, summary input).
3. **TAXAI:** the existing Gate 2 explanation (qwen2.5:14b, summary plus chart).
4. **Model B (informative control):** "A second model that additionally accounts for the timing and trajectory of the patient's recent changes estimates this risk at X%." Here X is model B's probability, which is more discriminative than model A (AUROC 0.80 vs 0.74 within LOW labels).
5. **Trajectory direction (informative control, direction only):** "Accounting for the timing and trajectory of recent changes, this patient's risk appears higher (or lower) than the AI estimate." The direction is the sign of (model B − model A).

## Protocol

The Study 3 two-turn judge protocol, adapted to the outcome:

- **Turn 1:** the judge sees the full chart and model A's prediction, and estimates the probability (0–100%) of invasive mechanical ventilation in the next 24 h. This turn is identical across arms and asked once per case.
- **Turn 2:** the judge sees the arm's text and re-estimates.
- **Judges:** qwen2.5:14b (primary) and phi4, at temperature 0 with seed 42, run sequentially.

## Endpoints, per judge and arm

- **Anchoring:** correlation of the turn-1 estimate with model A's probability.
- **ΔBrier** = (post − y)² − (pre − y)², with bootstrap 95% CIs (1,000 resamples). We also report the decomposition into movement, information, and direction.
- **Contrasts versus Empty:** paired bootstrap 95% CIs for each arm's ΔBrier minus Empty's.
- **Discrimination:** post-explanation AUROC.

## Decision rule (fixed now; primary judge)

- **Gate 3 open for informative input:** the Model B arm's ΔBrier is lower than Empty's, with the 95% CI of the difference entirely below 0.
- **LLM explanations harmful:** an LLM arm's ΔBrier is higher than Empty's, with the CI entirely above 0.
- **Robustness:** results under phi4 are reported as a robustness check.

## Caveat

Because the sample is outcome-enriched by design, Brier values are not population estimates. The contrasts between arms within the sample are the quantities of interest.
