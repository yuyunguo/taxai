# Deviations from the v4 pre-registration

Registered: OSF Preregistration, 2026-10-08, https://doi.org/10.17605/OSF.IO/HBWN6 (associated project https://osf.io/djmfc; registration public after OSF's 48-h moderation window). Hash file: `plan/prereg-v4.sha256` (see file for timestamp).

| Date | Deviation | Reason | Effect on inference |
|---|---|---|---|

## Run log (not deviations)

- 2026-10-08 12:48:33 EDT: both queues launched (`run_v4.sh` on the workstation; `run_v4_mac.sh` driving the Mac). Deployed code verified byte-identical to the registered SHA-256 hashes immediately before launch; `data_v2/v4/` did not exist before launch.
- 2026-10-09 05:57 EDT: workstation queue finished (V4_DONE); Mac queue finished 02:41 EDT (V4_MAC_DONE). 6,204/6,204 registered generations; no failed calls; no incomplete explainer. Registered analysis re-run on complete data 2026-10-09 ~08:20 (deterministic; same code hash d48ab7bc…).

No deviations from the registered plan.
- 2026-10-09 21:43:05 EDT: Tier C queues launched after the addendum was hashed (21:33:53 EDT). Deployed code verified identical to plan/prereg-v4-addendum-tierC.sha256 immediately before launch; no Tier C cache existed. **Correction (2026-10-10):** this entry originally said the launch followed OSF registration of the addendum; the OSF API showed no addendum at that time. The addendum was deposited unchanged as an approved update to HBWN6, submitted 2026-10-10 20:56 EDT (schema response 6acadcbba4ed734b80ca44ca), after the Tier C analyses. Pre-generation timing rests on the hash file.
- 2026-10-10 06:24:06 EDT: Mac Tier C queue finished (TIERC_MAC_DONE; qwen2.5:72b C1a, 400 calls).
- 2026-10-10 19:35:03 EDT: workstation Tier C queue finished (TIERC_DONE; 6,000 calls) and the hashed analysis wrote `tierC_v4.json`. 6,400/6,400 addendum generations, all HTTP 200, no retries, no incomplete explainer. 1 of 3,600 elicited probabilities unparsable (gpt-oss:20b C1a); imputed sensitivity analysis unchanged (LR p 0.1602 vs 0.1605).
- Reporting convention (not a deviation): the addendum analysis computes the registered "about right" share with missing verdicts scored as about right (gpt-oss:20b 97.5%); the manuscript reports explicit verdicts only (97.0%), as in Table 2. Footnoted in Supplementary Table for Note 10.

No deviations from the Tier C addendum.
- 2026-10-10 21:16 EDT: sensitivity analysis (not part of the addendum), `src/pipeline_v2/tierC_v4_boot10k.py`: hashed analysis code run with N_BOOT = 10,000, output to `tierC_v4_boot10k.json` (primary file unchanged, SHA-256 116724f7…). 0 of 16 classifications changed; max change in any 99% limit 0.0093; Firth results identical.
