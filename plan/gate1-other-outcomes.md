# Limited Gate 1 test on timing-driven outcomes (specified before running)

Specified 2026-10-02. **Exploratory and post hoc.** LLM-free; Gate 1 only.

## Question

Does the patient trajectory carry outcome information beyond the 209 aggregate features for *short-horizon, timing-driven* outcomes, where the framework predicts Gate 1 is more likely to open than for in-hospital mortality?

## Outcomes

Both use the same cohort, prediction time (ICU admission + 24 h), and input window as the main study. Each is defined over the next 24 h, from the prediction time to prediction time + 24 h, on the first ICU stay.

1. **Vasopressor initiation:** any norepinephrine infusion starting in the horizon (`mimic_derived.norepinephrine`). Patients with any norepinephrine before the prediction time are excluded.
2. **Invasive ventilation start:** any `InvasiveVent` episode starting in the horizon (`mimic_derived.ventilation`). Patients with invasive ventilation or tracheostomy before the prediction time are excluded.

Sepsis onset is not tested, because Sepsis-3 labels are derived from the same laboratory and SOFA inputs as the model, which risks circularity.

## Models and evaluation

Identical to the mortality information-ceiling test (`plan/info-ceiling-test.md`):

- **Models:** XGBoost with the 209 aggregate features (A) versus A plus 216 dynamics features (B). Same hyperparameters, same temporal split, native missing-value handling.
- **Endpoints, on the test split:** ΔAUROC and ΔAUPRC with paired bootstrap 95% CIs (1,000 resamples, seed 42).

## Decision rule (fixed now)

- **Gate 1 open:** ΔAUROC ≥ 0.010 with its 95% CI excluding 0.
- **Comparison:** results are reported alongside mortality (ΔAUROC +0.004, CI −0.001 to 0.008).

## Scope

This is a limited feasibility check, not a new explanation study. If Gate 1 opens for an outcome, a full pre-registered explanation study on that outcome would be the follow-up. If neither opens, the information-ceiling result generalizes beyond mortality.
