# Pre-specified analysis plan: TAXAI v2 (MIMIC-IV)

**Frozen:** 2026-10-01, ~18:25 EDT. At this point explanation generation was in progress (75 of ~1,050 generation calls done). No judge output (DC′, TAS, or reliance) had been produced or viewed. Integrity: the SHA-256 of this file is recorded in `plan/analysis-plan-v2.sha256`, and a copy is stored on the workstation at `~/workspace/taxai_v2/analysis-plan-v2.md`.

Any analysis not listed here will be labeled **exploratory** in the manuscript.

## 1. Context and motivation

In v1 (MIMIC-III), the primary decision metric was `1 − override rate`. A post-hoc analysis of v1 showed that TAXAI explanations lowered overrides mainly by raising acceptance of *wrong* predictions, with balanced appropriate reliance of 0.454 for TAXAI vs 0.681 for SHAP. v2 therefore replaces the decision endpoint with appropriate reliance and fixes leakage, calibration, prompt advocacy, and judge outcome exposure (`plan/root-cause-fix-plan.md`). Because the endpoint was chosen after seeing v1, the v1 findings are reported as the motivating analysis, and **only v2 results are confirmatory**.

## 2. Design (fixed)

- **Sample:** 500 MIMIC-IV test admissions (2020–2022): 250 predicted HIGH and 250 predicted LOW at the validation-chosen threshold 0.296; seed 42; 180 deaths.
- **Methods:** Empty, Random, SHAP, Static LLM, TAXAI. The two LLM methods use identical instructions and differ only by the chronological timeline.
- **Judges:**
  - **Primary:** qwen2.5:14b.
  - **Robustness:** phi4.
  - **Decoding:** temperature 0, seed 42.
  - **Inputs:** each judge sees the summary, the prediction with its label, and the explanation, but no ICD codes and no outcome.
- **Label correctness:** a HIGH label is correct if the patient died in hospital; a LOW label is correct if the patient survived.

## 3. Endpoints

| Role | Endpoint | Definition |
|---|---|---|
| **Primary** | BAR | 0.5·[P(accept \| label correct) + P(override \| label wrong)]; 0.5 = any constant policy |
| Secondary | CU | 0.4·DC′ + 0.3·TAS + 0.3·BAR |
| Secondary | DC′ | Rank-weighted match of the judge's top-3 diagnoses to hidden discharge ICD titles |
| Secondary | TAS | Keyword-based treatment appropriateness (v1 lists) |
| Secondary | Faithfulness | Share of numeric values cited in an explanation that are supported by the patient's data (`faithfulness.py`); LLM methods only |
| Descriptive | OR; P(accept \| correct); P(override \| wrong) | Overall and by HIGH/LOW label |
| Descriptive | v1-style CU | 0.4·DC′ + 0.3·TAS + 0.3·(1 − OR), to show how the metric choice changes rankings |

## 4. Hypotheses and tests

- **H1 (primary):** BAR(TAXAI) ≠ BAR(Static LLM) under the qwen2.5:14b judge. This contrast isolates the chronological timeline.
  - Test: paired case-level bootstrap (2,000 resamples, seed 42) of the BAR difference; two-sided α = 0.05; a single test with no multiplicity adjustment.
  - Report the difference with its 95% CI.
- **H2 (secondary):** TAXAI vs each of Empty, Random, and SHAP on BAR.
- **H3 (secondary):** TAXAI vs each baseline on CU, DC′, TAS, and OR.
  - Tests: Wilcoxon signed-rank for DC′ and TAS; McNemar with continuity correction for OR; paired bootstrap for BAR and CU.
  - Bonferroni correction over all 20 secondary tests (4 comparisons × 5 metrics).
- **H4 (secondary):** faithfulness of TAXAI vs Static LLM, tested with a Wilcoxon signed-rank test on per-explanation supported-value rates.
- **Robustness:** H1 is repeated under phi4. Agreement across judges is assessed by (a) the sign of the H1 difference, (b) Spearman correlation of method CU, and (c) Cohen's κ on per-case override decisions.

## 5. Interpretation rules (fixed in advance)

- **H1 CI entirely > 0:** "Adding the chronological timeline improved appropriate reliance of the LLM-simulated clinician."
- **H1 CI includes 0:** "No evidence that the chronological timeline changed appropriate reliance." Equivalence will **not** be claimed.
- **H1 CI entirely < 0:** report that the timeline *worsened* appropriate reliance.
- **Any LLM method with BAR below Empty:** report as explanation-induced over-reliance (automation bias), whatever the CU ranking.
- **qwen and phi4 disagree on the sign of H1:** report the result as judge-dependent and not robust.
- **Effect too small to matter:** a BAR difference below 0.03 is described as practically negligible, even if statistically significant.
- **All endpoints in Section 3 are reported regardless of direction.** The v1 results remain in the paper as the motivating negative finding.

## 6. Things that will not change after judge outputs are seen

Sample, threshold, prompts, judge models, decoding settings, DC′ matching rule (≥ 50% token overlap with synonym map), TAS keyword lists, CU weights, BAR definition, and the bootstrap settings. Fixes for crashes or parsing failures are allowed only if applied blind to method labels, and they will be documented.

## 7. Planned exploratory analyses (labeled as such)

- BAR by predicted label (HIGH vs LOW) and by risk decile.
- Relationship between faithfulness and acceptance (do unfaithful explanations get accepted?).
- Qualitative review of the 25 multi-granularity outputs (clinician, patient, regulator), including cross-level contradictions.
- Optional eICU external validation, if run, with the same plan.
