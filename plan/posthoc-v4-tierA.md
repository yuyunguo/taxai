# Post hoc analyses after the simulated review (Tier A)

Specified 2026-10-09, before computation. **Post hoc.** These analyses respond to the simulated review (`plan/review-synthesis-v4.md`). They do not replace the registered analyses (doi:10.17605/OSF.IO/HBWN6): the registered classifications are reported unchanged, and everything here is labelled post hoc in the manuscript. No new LLM generation; existing responses and data only.

## P1. Readout capacity (R1-M1, R2-M1/M4, R3-M1)

Per explainer and condition, ventilation LOW (n = 400):
- dissent rate (share not "about right") among events and among non-events;
- **attainable maximum AUROC** given the explainer's observed verdict counts (the most favourable assignment of its verdicts to events and non-events);
- a "non-responsive" flag if more than 90% of verdicts are one category.

## P2. Comparison with the positive control (R1-M1, R2-M1, R3-M1)

Paired bootstrap AUROC difference, each LLM's TAXAI verdict minus the rule control, in the same 400 cases (1,000 resamples, seed 42).

## P3. Results by split (R2-M6, R3-M4, R1-m10)

LOW stratum, separately for validation and test cases, and overall:
- n and events;
- AUROC of model A, model B, the rule control, and each explainer's TAXAI verdict;
- paired B − A AUROC difference with bootstrap 95% CI.

## P4. Information beyond the prediction (R1-M2, R3-M2)

Per explainer, TAXAI, ventilation LOW:
- (a) Firth-penalized logistic regression of the outcome on logit(pA) plus verdict indicators ("too low" and "too high" vs "about right"), with profile-free Wald 95% CIs from the penalized fit, and a likelihood-ratio test against logit(pA) alone.
- (b) Spearman correlation between the verdict score and pA.
- (c) Verdict AUROC within tertiles of pA, combined as a stratified (van Elteren-style) average weighted by events × non-events per tertile.

## P5. Source of the Gate 1 ventilation signal (R1-M4, R2-M5, R3-M7)

Test set, ventilation, XGBoost with registered settings, paired bootstrap ΔAUROC versus model A (1,000 resamples, seed 42). Four models:
1. model A + **value dynamics** only (early/late means and difference, slope change, volatility, timing of max and min);
2. model A + **measurement-process** features only (hours since last measurement, number of measurements in the last 12 h);
3. model B (all dynamics, as registered);
4. **capacity control:** model A + all dynamics features with rows permuted within each split (destroys the patient link, keeps feature count and distributions).

Also: Gate 1 within the ventilation LOW and HIGH strata on the test set.

## P6. Ventilation model performance and calibration (R2-m3, R1-m6, R3-m3)

Model A, ventilation:
- test AUROC with a CI, and validation AUROC;
- Brier score;
- calibration intercept and slope;
- observed vs mean predicted risk overall and in the LOW stratum.

## P7. Ventilation event characterization (R1-M3, R1-m15, R3-m11)

Among eligible patients:
- (a) time from the prediction time to the start of invasive ventilation (events);
- (b) share of events preceded by non-invasive ventilation or high-flow oxygen (MIMIC-IV derived ventilation table) in the 24 h before or after the prediction time;
- (c) DNR/DNI/comfort-measures status recorded before the prediction time (chartevents "Code Status"): counts among events and non-events;
- (d) **sensitivity analysis:** Gate 1 (ΔAUROC) and Gate 2 (each explainer's TAXAI AUROC, LOW stratum) excluding patients with a DNI, DNR/DNI, or comfort-measures status before the prediction time;
- (e) death or ICU discharge within 24 h without intubation (competing events): counts.

Note on reintubation: by construction, patients with any invasive ventilation or tracheostomy earlier in the same ICU stay were excluded, so in-stay reintubations are not events. This is stated, not re-analysed.

## Interpretation rules (fixed now)

- No result here changes a registered classification.
- **P2:** an LLM is described as "below the positive control" only if the 95% CI of its difference from the rule control lies entirely below 0.
- **P4:** a verdict is described as adding information beyond the prediction only if the LR test gives p < 0.05 **and** the "too low" coefficient's 95% CI excludes 0 in the expected direction. Otherwise: "no evidence of added information".
- **P5:** the signal is attributed to a feature family if that family alone recovers at least half of model B's ΔAUROC with a CI excluding 0. The capacity control must show no gain (CI including 0); otherwise the Gate 1 result is reported as possibly capacity-driven.
- **P7d:** if any Gate 1 or Gate 2 conclusion changes after excluding DNI and comfort-care patients, the change is reported in the main text.
