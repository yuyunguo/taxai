# Pre-specified analysis plan: study v3 (fair test of the timeline)

**Status: FROZEN 2026-10-01 23:18:19 EDT.** Frozen after all four pilot gates passed (G1 input contrast median 73.5; G2 parse rate 100%; G3 pre-estimate SD 0.29; G4 max prompt about 2,500 tokens) and before any v3 main-sample explanation or judge output existed. SHA-256 values are in `plan/frozen-artifacts-v3.sha256`. Pilot outcome comparisons were not inspected; only the gate metrics and code smoke tests were. The design rationale is in `plan/study-v3-design.md`.

## 1. Design (fixed)

- **Sample:** the same 500 MIMIC-IV test admissions as v2 (250 HIGH, 250 LOW at τ = 0.296; 180 deaths). Pilot cases (40) are disjoint and excluded.
- **Chart:**
  - **Content:** the full chronological record. All in-window lab values (one per test per hour), 2-h vital-sign medians (heart rate, SBP, MAP, respiratory rate, SpO₂, temperature), the first 15 medication starts, and imaging references. PO₂ is withheld.
  - **Who sees it:** the judge, identically in all arms.
- **Arms:** Empty, Random, SHAP, Static LLM, TAXAI.
  - **Generator inputs:** Static LLM gets the aggregate summary; TAXAI gets the summary plus the chart.
  - **Instruction:** identical for both (`EXPLAIN_V3`), ending with a directional verdict.
- **Judge:**
  - **Models:** qwen2.5:14b (primary); phi4 (robustness).
  - **Decoding:** temperature 0, seed 42.
  - **Protocol:** turn 1 estimates mortality from the chart and the AI prediction; turn 2 re-estimates after the explanation.
- **Code:** `src/pipeline_v2/{timeline_v3, prompts_v3, explain_v3, evaluate_v3, faithfulness_v3, stats_v3}.py`.

## 2. Endpoints and tests

Per-case Brier change: ΔBrier = (post − y)² − (pre − y)², where y is in-hospital death and a negative value is better.

| ID | Endpoint | Test |
|---|---|---|
| **H1 (primary)** | mean ΔBrier(TAXAI) − mean ΔBrier(Static LLM) | paired case bootstrap (2,000; seed 42); two-sided α = 0.05; no adjustment |
| S1–S3 | ΔBrier, TAXAI vs Empty / Random / SHAP | paired bootstrap |
| S4 | post-explanation AUROC, TAXAI vs Static LLM | paired bootstrap |
| S5 | appropriate shift (moves ≥ 5 points toward y), TAXAI vs Static LLM | McNemar |
| S6 | derived BAR (post estimate on the label's side of τ), TAXAI vs Static LLM | paired bootstrap |
| S7 | verified temporal claims per explanation, TAXAI vs Static LLM | Wilcoxon signed-rank |
| S8 | mediation: ΔBrier ~ verified temporal claims + arm | OLS, case-clustered SEs |

- **Multiplicity:** S1–S8 are Bonferroni-corrected (×8).
- **Missing estimates:** cases with an unparsed estimate in either compared arm are excluded pairwise; counts are reported.

**Timeline measures (descriptive, all reported):**
- input contrast per case;
- time references and sequence terms per explanation;
- value and temporal-claim faithfulness per arm;
- distribution of the generator's directional verdict, and its agreement with label correctness.

**Robustness:** H1 is repeated with phi4. We report sign agreement and both CIs.

## 3. Interpretation rules (fixed in advance)

- **H1 CI entirely < 0:** "With the full chart available, trajectory-grounded explanations moved the simulated clinician's estimates closer to the outcome than summary-grounded explanations."
- **H1 CI includes 0:** "No evidence that trajectory grounding changed the accuracy of the simulated clinician's estimates." Equivalence will not be claimed.
- **H1 CI entirely > 0:** report that trajectory grounding *worsened* estimates.
- **Practical relevance:** a |difference| < 0.005 in mean ΔBrier is described as practically negligible.
- **Arm worse than Empty:** any arm with mean ΔBrier > Empty's (S1 for TAXAI) is reported as an explanation that worsened the judge's estimates.
- **Judges disagree:** if qwen2.5:14b and phi4 disagree on the sign of H1, the result is reported as judge-dependent.
- **Mediation (S8)** is interpreted as an association, not a causal effect.
- **All endpoints are reported regardless of direction,** and v2 is reported alongside v3.

## 4. Fixed after freezing

The sample, chart construction, prompts, judges, decoding, parsing rules, endpoint definitions, the ±5-point shift threshold, the faithfulness tolerances (±1 h for labs, ±2 h for vitals), the bootstrap settings, and the interpretation rules.

Crash or parse fixes are allowed only if applied blind to arm, and they will be documented.

## 5. Amendment A1 (2026-10-01 23:24:08 EDT), made before any v3 main-sample output existed

Made after proofreading the frozen plan against the frozen code. Verified at the time of the amendment: `data_v2/v3/` contained only `pilot/`, and the main run was still queued behind the v2 phi4 judge. **No hypothesis, endpoint, test, threshold, or frozen analysis code is changed.** `stats_v3.py` and the other frozen code keep their original hashes.

1. **Implementation of the promised descriptive measures.** Section 2's timeline measures had no implementing code. They are now implemented in `src/pipeline_v2/timeline_measures_v3.py`: input contrast; time references and sequence terms per explanation; verdict distribution and verdict accuracy. The script was smoke-tested on the pilot with its output not inspected.
2. **Definition of verdict accuracy (previously unspecified).**
   - When the label is wrong: "too high" is the right verdict for a wrong HIGH label, and "too low" is right for a wrong LOW label.
   - When the label is correct: "about right" is the right verdict.
3. **Clarification of H1.** Turn 1 is shared across arms, so the pre-explanation estimate is identical within a case in every arm. H1 is therefore numerically equal to the difference in post-explanation Brier between TAXAI and Static LLM.
4. **Renaming of S8.** S8 is an arm-adjusted *association* between verified temporal claims and ΔBrier (OLS with case-clustered SEs), not a formal mediation analysis. It is reported as "temporal-claim association".
5. **S7 is a manipulation check.** Static LLM's input contains no hour stamps, so a TAXAI excess on S7 is expected by design. S7 confirms that temporal content reached the explanations; it is not evidence of benefit.
6. **The "worse than Empty" rule.** It is tested only for TAXAI (S1). For the other arms it is applied descriptively to point estimates.
7. **Precision.** n = 500 was fixed by the v2 sample; no power calculation was made from pilot outcomes. The achieved CI half-width for H1 will be reported. The 0.005 practical-relevance threshold was set a priori, as a small fraction of a typical pre-explanation Brier score in a sample with 36% mortality.
8. **Measurement limitation.** The faithfulness checker has no alias for temperature, so temperature values are not attributable and are excluded from value and temporal-claim faithfulness.
