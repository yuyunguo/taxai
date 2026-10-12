# OSF submission text: Tier C addendum

Paste the fields below into an OSF **Open-Ended Registration** created from project djmfc.

## Title

Addendum to "Explainer-ceiling audit of large language model explanations for ICU risk predictions" (doi:10.17605/OSF.IO/HBWN6): elicited probabilities, given-signal test, and alternative instruction

## Description

Addendum to OSF registration 10.17605/OSF.IO/HBWN6. It specifies three further conditions in the registered ventilation LOW-label sample (400 cases). The conditions are an elicited-probability readout (prediction shown or hidden), the given-signal (hint) test for three more explainers, and an alternative instruction. For each condition, the addendum fixes the endpoints and Bonferroni-adjusted decision rules.

## Summary (Open-Ended Registration field)

This addendum extends OSF registration 10.17605/OSF.IO/HBWN6 (registered 8 October 2026).

Timing. The addendum was specified after the registered analyses were complete. The attached plan (prereg-v4-addendum-tierC.md) and analysis code were frozen and hashed (SHA-256) on 2026-10-09 at 21:33:53 EDT. This was before any addendum output was generated: generation started at 21:43:05 EDT, after the deployed code was verified identical to these hashes. Generation and the hashed analysis were completed on 2026-10-10. The plan and code are deposited here unchanged, after completion; the hashes below allow verification. All results are reported in the associated manuscript, whatever their direction.

Content. The conditions are:
- C1a/C1b: an elicited probability, with model A's prediction shown or hidden;
- C2: the hint, for qwen2.5:32b, gemma3:27b, and gpt-oss:20b;
- C3: an alternative instruction stating the base rate, without "Do not defend the prediction".

Endpoints and decision rules are in section 4 of the attached plan.

SHA-256 (computed 2026-10-09 21:33:53 EDT):
- prereg-v4-addendum-tierC.md 39143620f04251527470aa9f4161c67a4bd406c4d084871b356ee776d051ab42
- prompts_tierC.py e267d48692499424c23d75f9bbf9d8a91f66d403a073f83815157340e1923311
- tierC_v4.py 44e323d7805a8769fcc11c82249b8bc5fe72ec52b4f829d8c73e7c9390fcf0e9
- tierC_v4_analysis.py 7ebed990f5d8e05107f8f266d5fff191806170d5f09087cbcee4062200c08ed6
- posthoc_v4_gate2.py 142754752b89c32fcf0870fc4bd637d6e62f6453ef74b8b4672605897c2ad8b7
- run_tierC.sh 92164ed3ee8fec2275b6dd4fb1e1e932a133d36caf9fc170adcb33e2b4dcbb35
- run_tierC_mac.sh 05f109056eea97f2b5f9e328bd990362d50e6390da7d69703d492f9097feba7e

No patient-level data are included (PhysioNet data use agreement).

## Files to attach (code and plan only; no data)

- plan/prereg-v4-addendum-tierC.md
- plan/prereg-v4-addendum-tierC.sha256
- src/pipeline_v2/prompts_tierC.py
- src/pipeline_v2/tierC_v4.py
- src/pipeline_v2/tierC_v4_analysis.py
- src/pipeline_v2/posthoc_v4_gate2.py
- src/pipeline_v2/run_tierC.sh
- src/pipeline_v2/run_tierC_mac.sh

Do not attach anything from data_v2/, results, logs, or *.jsonl.
