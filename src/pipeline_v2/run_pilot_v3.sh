#!/usr/bin/env bash
# v3 pilot: build charts, generate explanations, judge (qwen2.5:14b), evaluate go/no-go gates.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TAXAI_V2_DATA="${TAXAI_V2_DATA:-$PWD/data_v2}"
PY="${PY:-$HOME/workspace/taxai/.venv/bin/python}"
[ -f "$TAXAI_V2_DATA/trajectories_test_v3.pkl" ] || $PY -m src.pipeline_v2.timeline_v3
$PY -m src.pipeline_v2.explain_v3 --set pilot
$PY -m src.pipeline_v2.evaluate_v3 --set pilot --judge qwen2.5:14b
$PY -m src.pipeline_v2.pilot_gates_v3 --judge qwen2.5:14b
echo "PILOT DONE"
