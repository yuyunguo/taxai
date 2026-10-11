#!/usr/bin/env bash
# Tier C 72B arm (C1a), generated on the Mac, driven from the workstation. Launch only after the OSF addendum:
#   MAC_OLLAMA_HOST=http://<mac>:11434 nohup setsid bash src/pipeline_v2/run_tierC_mac.sh > logs/tierC_mac.log 2>&1 < /dev/null &
set -euo pipefail
: "${MAC_OLLAMA_HOST:?set MAC_OLLAMA_HOST}"
export TAXAI_V2_DATA="$PWD/data_v2"
PY="$HOME/workspace/taxai/.venv/bin/python"
echo "=== $(date '+%F %T') qwen2.5:72b-instruct-q3_K_M C1a"
"$PY" -m src.pipeline_v2.tierC_v4 --explainer qwen2.5:72b-instruct-q3_K_M
"$PY" -m src.pipeline_v2.tierC_v4_analysis
echo "=== $(date '+%F %T') TIERC_MAC_DONE"
