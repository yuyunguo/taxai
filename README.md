# TAXAI: a two-ceiling audit of language-model explanations for ICU risk predictions

Code, analysis plans, and aggregate results for:

> Yu Y. *A two-ceiling audit finds language-model verdicts miss useful trajectory signal in ICU risk prediction.* (Manuscript submitted.)

**Pre-registration:** OSF, doi:[10.17605/OSF.IO/HBWN6](https://doi.org/10.17605/OSF.IO/HBWN6), registered 8 October 2026. The pre-specified addendum was added as an update on 10 October 2026.

## What the audit measures

An explanation can improve a clinician's judgment of a risk prediction only if two conditions hold. Both are measured here before any reader is involved.

1. **Information ceiling (Gate 1).** The patient's trajectory must hold outcome information that the prediction has not used. This is tested by comparing model A (209 aggregate features) with model B (model A plus 216 trajectory-dynamics features).
2. **Explainer ceiling (Gate 2).** The explainer's verdict on the prediction ("too high / about right / too low") must discriminate outcomes among patients with the same risk label. A rule-based positive control checks that the audit can register signal.

## Data access

All analyses use **MIMIC-IV v3.2**, **MIMIC-IV-Note**, and, for external validation, the **eICU Collaborative Research Database**. These are available to credentialed users through [PhysioNet](https://physionet.org/).

**This repository contains no patient-level data.** The `.gitignore` excludes every derived data file. Language-model prompts and responses contain values derived from these databases. They are shared only with credentialed PhysioNet users, on request.

All language models were run locally through [Ollama](https://ollama.com/). No data were sent to external services.

## Repository layout

| Path | Contents |
|---|---|
| `src/pipeline_v2/` | Cohort extraction, features, models A and B, charts, explanation generation, and all analyses |
| `scripts/` | Figure and table generation from the aggregate results |
| `results_v2/` | Aggregate results only (JSON); these are the inputs to `scripts/` |
| `plan/` | Registered and hashed analysis plans, hash files, and deviation logs |
| `tests/` | Unit tests for the faithfulness measures |

## Environment

Python 3.12. Install the packages with `pip install -r requirements.txt`.

| Variable | Purpose |
|---|---|
| `PGPASSWORD` | PostgreSQL password for the MIMIC-IV database; never stored in code |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` | Database connection (defaults: `localhost`, `5432`, `mimiciv`, `postgres`) |
| `TAXAI_V2_DATA` | Directory for derived data (default `./data_v2`) |
| `MAC_OLLAMA_HOST` | Ollama endpoint for the 72B explainer, which ran on a second machine |
| `PY` | Python interpreter used by the shell scripts |

## Reproducing the analyses

Run from the repository root, in order:

```bash
# 1. Cohort, features, and models (needs database access)
PGPASSWORD=... python -m src.pipeline_v2.extract
bash src/pipeline_v2/run_all.sh

# 2. Gate 1: information ceiling (post hoc)
python -m src.pipeline_v2.info_ceiling                     # mortality
PGPASSWORD=... python -m src.pipeline_v2.gate1_outcomes    # ventilation, norepinephrine

# 3. Secondary analysis: language-model readers on mortality (locally hashed plan)
bash src/pipeline_v2/run_pilot_v3.sh
bash src/pipeline_v2/run_main_v3.sh
bash src/pipeline_v2/run_measures_v3.sh

# 4. Gate 2: explainer ceiling (registered)
bash src/pipeline_v2/run_v4.sh                             # workstation explainers
MAC_OLLAMA_HOST=http://<host>:11434 bash src/pipeline_v2/run_v4_mac.sh   # 72B explainer

# 5. Post hoc analyses
PGPASSWORD=... python -m src.pipeline_v2.posthoc_v4_gate1
python -m src.pipeline_v2.posthoc_v4_gate2

# 6. Pre-specified addendum, plus the 10,000-resample sensitivity analysis
bash src/pipeline_v2/run_tierC.sh
MAC_OLLAMA_HOST=http://<host>:11434 bash src/pipeline_v2/run_tierC_mac.sh
python -m src.pipeline_v2.tierC_v4_boot10k

# 7. Figures 1-4 and tables (aggregate results only)
python scripts/gen_figures_v4.py
python scripts/gen_posthoc_tables.py
python scripts/gen_supp_note2.py
```

LLM generation is deterministic (temperature 0, seed 42) and cached in append-only files, so re-running an analysis step does not regenerate responses.

## Verifying the registered code

Every plan and script that was frozen before execution is listed with its SHA-256 hash in `plan/*.sha256`. The same hashes appear in the registration and in Supplementary Note 5 of the manuscript.

```bash
shasum -a 256 -c <(grep -E '^[0-9a-f]{64}' plan/prereg-v4.sha256)
shasum -a 256 -c <(grep -E '^[0-9a-f]{64}' plan/prereg-v4-addendum-tierC.sha256)
```

| Hash file | Frozen | Covers |
|---|---|---|
| `prereg-v4.sha256` | 2026-10-08 12:04 EDT | Registered Gate 2 plan and code |
| `prereg-v4-addendum-tierC.sha256` | 2026-10-09 21:33 EDT | Pre-specified addendum plan and code |
| `posthoc-v4-tierA.sha256` | 2026-10-09 16:13 EDT | Post hoc analysis plan |
| `frozen-artifacts-v3.sha256` | 2026-10-01 23:18 EDT, Amendment A1 23:24 EDT | Language-model reader analysis |
| `info-ceiling-test.sha256`, `gate1-other-outcomes.sha256` | before computation | Gate 1 plans |
| `generator-check.sha256`, `gate2-imv.sha256` | before computation | Exploratory Gate 2 tests, disclosed in the registration |

Hash files list files as of their freeze. Where a plan was later amended, the file matches its latest listed hash. There are two such cases:

- **`plan/analysis-plan-v3.md`** matches its Amendment A1 hash.
- **`src/pipeline_v2/timeline_v3.py`** was edited on 8 October 2026 at 10:46 EDT, after the reader analysis and before the v4 registration. The validation-set charts for the registered sample were generated with it at 10:50. It therefore matches the v4 hash (`prereg-v4.sha256`), not the v3 freeze hash. The edit did not change the reader-analysis inputs. Regenerating all 7,353 test-set charts with the current code reproduced the stored charts exactly (verified 2026-10-10).

## Citation

If you use this code, please cite the article (citation to be added on publication) and the registration (doi:10.17605/OSF.IO/HBWN6).

## License

MIT (see `LICENSE`). The license covers the code only. MIMIC-IV and eICU data remain subject to their PhysioNet data use agreements.
