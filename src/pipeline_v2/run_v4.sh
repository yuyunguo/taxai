#!/usr/bin/env bash
# v4 explainer-ceiling audit queue (plan/prereg-v4-osf.md). Run on the database host:
#   PGPASSWORD=... nohup setsid bash src/pipeline_v2/run_v4.sh > logs/v4.log 2>&1 < /dev/null &
# Order is pre-registered; the analysis is refreshed after every step so partial results exist.
set -euo pipefail
export TAXAI_V2_DATA="$PWD/data_v2"
PY="$HOME/workspace/taxai/.venv/bin/python"
step() { echo "=== $(date '+%F %T') $*"; "$PY" -m src.pipeline_v2.gate2_v4 "$@"; "$PY" -m src.pipeline_v2.gate2_v4_analysis; }

echo "=== $(date '+%F %T') build"; "$PY" -m src.pipeline_v2.gate2_v4 --build
step --explainer qwen2.5:14b  --task imv
step --explainer gemma3:27b   --task imv
step --explainer gemma3:27b   --task mortality
step --explainer gpt-oss:20b  --task imv
step --explainer gpt-oss:20b  --task mortality
step --explainer qwen2.5:32b  --task imv
echo "=== $(date '+%F %T') V4_DONE"
