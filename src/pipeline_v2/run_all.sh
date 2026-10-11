#!/usr/bin/env bash
# Full v2 pipeline on the workstation. Resumable: LLM calls are cached.
# Requires PGPASSWORD only if extraction must be re-run.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TAXAI_V2_DATA="${TAXAI_V2_DATA:-$PWD/data_v2}"
PY="${PY:-$HOME/workspace/taxai/.venv/bin/python}"
$PY -m src.pipeline_v2.build
$PY -m src.pipeline_v2.explain
$PY -m src.pipeline_v2.evaluate --judge qwen2.5:14b
$PY -m src.pipeline_v2.evaluate --judge phi4
echo "ALL DONE"
