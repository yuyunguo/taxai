#!/usr/bin/env bash
# v4 72B arm (plan/prereg-v4-osf.md): runs on the database host, generation served by the Mac's Ollama.
# Start after run_v4.sh has finished its build step (needs data_v2/v4/imv_sample.csv):
#   MAC_OLLAMA_HOST=http://<mac>:11434 nohup setsid bash src/pipeline_v2/run_v4_mac.sh > logs/v4_mac.log 2>&1 < /dev/null &
set -euo pipefail
: "${MAC_OLLAMA_HOST:?set MAC_OLLAMA_HOST}"
export TAXAI_V2_DATA="$PWD/data_v2"
PY="$HOME/workspace/taxai/.venv/bin/python"
until [ -f data_v2/v4/imv_sample.csv ]; do sleep 30; done
for task in imv mortality; do
  echo "=== $(date '+%F %T') qwen2.5:72b-instruct-q3_K_M $task"
  "$PY" -m src.pipeline_v2.gate2_v4 --explainer qwen2.5:72b-instruct-q3_K_M --task "$task"
done
"$PY" -m src.pipeline_v2.gate2_v4_analysis
echo "=== $(date '+%F %T') V4_MAC_DONE"
