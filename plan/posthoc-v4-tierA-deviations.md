# Deviations from the post hoc Tier A specification

Spec: `plan/posthoc-v4-tierA.md` (hashed 2026-10-09 16:13:21 EDT).

| Date | Deviation | Reason | Effect |
|---|---|---|---|
| 2026-10-09 21:20 EDT | P4(a): the Firth model's reference category is "about right" (missing pooled with it) **unless it has fewer than 5 cases**, in which case the most frequent other category is the reference. | qwen2.5:14b Static LLM had 1 "about right" verdict (TAXAI: 2), which made the model singular or unstable. Found on first execution, before any P4 result was inspected. | Applies to qwen2.5:14b (Static LLM, TAXAI, and TAXAI + hint if < 5). The "too low" coefficient keeps its direction; the decision rule is unchanged. |
| 2026-10-09 21:25 EDT | P4(a), refinement of the row above: any other verdict category with fewer than 5 cases is pooled with the reference category. | qwen2.5:14b Static LLM still had a 1-case category after the reference switch, and the fit failed. | Pooled categories are listed in the output (`pooled_into_reference`). Decision rule unchanged. |

**Disclosure on the second refinement.** Before the pooling refinement was introduced (to fix the qwen2.5:14b Static LLM fit), the run under the first deviation had produced, and the author had seen, the P4(a) result for gpt-oss:20b TAXAI: LR p = 0.068, "too low" coefficient not meeting the rule (added information: no). Under the pooled rule, the two "too high" verdicts are merged with the reference, giving LR p = 0.0099 and a "too low" coefficient of +2.30 (95% CI 0.17 to 4.43), which meets the rule (added information: yes). Because the refinement changed this classification after a result was seen, **both results are reported**, and the manuscript treats the gpt-oss finding as sensitive to category handling, not as established.
