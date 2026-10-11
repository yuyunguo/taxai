#!/usr/bin/env bash
# Resume the v2 phi4 judge after the v3 pilot finishes (avoids GPU model thrashing).
set -euo pipefail
cd "$(dirname "$0")/../.."
export TAXAI_V2_DATA="${TAXAI_V2_DATA:-$PWD/data_v2}"
PY="${PY:-$HOME/workspace/taxai/.venv/bin/python}"
while pgrep -f run_pilot_v3.sh >/dev/null; do sleep 30; done
$PY -m src.pipeline_v2.evaluate --judge phi4
$PY -m src.pipeline_v2.stats
echo "PHI4 V2 DONE"
