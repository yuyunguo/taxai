#!/usr/bin/env bash
# v3 main run, queued after the v2 phi4 judge to avoid GPU model thrashing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TAXAI_V2_DATA="${TAXAI_V2_DATA:-$PWD/data_v2}"
PY="${PY:-$HOME/workspace/taxai/.venv/bin/python}"
while pgrep -f resume_phi4_v2.sh >/dev/null; do sleep 60; done
$PY -m src.pipeline_v2.explain_v3 --set main
$PY -m src.pipeline_v2.faithfulness_v3 --set main
$PY -m src.pipeline_v2.evaluate_v3 --set main --judge qwen2.5:14b
$PY -m src.pipeline_v2.stats_v3 --set main --judge qwen2.5:14b
echo "V3 QWEN DONE"
$PY -m src.pipeline_v2.evaluate_v3 --set main --judge phi4
$PY -m src.pipeline_v2.stats_v3 --set main --judge phi4
echo "V3 ALL DONE"
