#!/usr/bin/env bash
# Amendment A1: descriptive timeline measures, run once the v3 main explanations exist (CPU only).
set -euo pipefail
cd "$(dirname "$0")/../.."
export TAXAI_V2_DATA="${TAXAI_V2_DATA:-$PWD/data_v2}"
PY="${PY:-$HOME/workspace/taxai/.venv/bin/python}"
until [ -f "$TAXAI_V2_DATA/v3/main/explanations.json" ]; do sleep 120; done
$PY -m src.pipeline_v2.timeline_measures_v3 --set main
echo "MEASURES DONE"
