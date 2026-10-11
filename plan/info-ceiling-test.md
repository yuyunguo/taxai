# Information-ceiling test (specified before running)

Specified 2026-10-02, before any computation. This is a **post-hoc** analysis relative to the frozen v2 and v3 plans, and it will be labeled as such.

## Question

Does the patient's chronological trajectory carry information about in-hospital death **beyond** the base model's 209 aggregate features? This tests Proposition 1's condition, I(𝒯; Y | X) > 0. If it fails, explanations grounded in the timeline cannot, in principle, tell a clinician when this model is wrong.

## Data

- **Cohort and splits:** the MIMIC-IV cohort and splits as in v2 (train 2008–2016, validation 2017–2019, test 2020–2022).
- **Observation window:** the same as the base model, [max(admission, ICU − 24 h), ICU + 24 h].
- **Variables:** the 15 laboratory tests and 10 vital-sign variables used by the base model.

## Timeline-only (dynamics) features

Per admission and variable, with times in hours relative to ICU admission and prediction time at +24 h:

1. mean in the early window (t < 6), mean in the late window (t ≥ 18), and late − early;
2. acceleration: OLS slope for t ≥ 12 minus OLS slope for t < 12;
3. volatility: mean absolute difference between successive values;
4. hours from the last measurement to the prediction time;
5. time of the maximum value and time of the minimum value;
6. number of measurements in the last 12 h.

These describe *when* and *how* values change, which the aggregate features (mean, SD, min, max, first, last, overall slope, count) do not.

## Models (XGBoost, identical hyperparameters to the base model, seed 42)

To compare like with like, both A and B use native missing-value handling and no imputer.

- **A:** 209 aggregate features.
- **B:** A plus the dynamics features.

## Endpoints, evaluated on the test split (n = 7,353)

1. **ΔAUROC, ΔAUPRC, and ΔBrier (B − A),** with paired case-bootstrap 95% CIs (1,000 resamples, seed 42).
2. **Error detection:** AUROC for death *within* each label stratum (HIGH or LOW at τ = 0.296 under A), for A vs B. This asks whether the timeline helps identify which labels are wrong.
3. **Incremental value over the deployed model:** a logistic regression of outcome on logit(p_A) and logit(p_B), fitted on validation and evaluated on test (AUROC and Brier).

## Decision rule (fixed now)

- **Meaningful residual information:** ΔAUROC ≥ 0.010 *and* its 95% CI excludes 0, *or* a within-stratum AUROC gain ≥ 0.02 with CI excluding 0.
- **Otherwise:** we conclude that, at this resolution, the timeline adds little outcome information beyond the aggregates. That would support the information-ceiling explanation of the v2/v3 nulls.

## Caveat

Engineered dynamics features are not the only way to read a trajectory. A negative result bounds what *these* summaries add; it does not rule out signal that a sequence model could extract. A sequence-model check (GRU) is the natural follow-up if the result is borderline.
