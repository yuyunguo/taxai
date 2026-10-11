#!/usr/bin/env bash
# Tier C queue on the workstation (plan/prereg-v4-addendum-tierC.md). Launch only after the OSF addendum:
#   nohup setsid bash src/pipeline_v2/run_tierC.sh > logs/tierC.log 2>&1 < /dev/null &
set -euo pipefail
export TAXAI_V2_DATA="$PWD/data_v2"
PY="$HOME/workspace/taxai/.venv/bin/python"
for m in qwen2.5:14b gemma3:27b gpt-oss:20b qwen2.5:32b; do
  echo "=== $(date '+%F %T') $m"
  "$PY" -m src.pipeline_v2.tierC_v4 --explainer "$m"
  "$PY" -m src.pipeline_v2.tierC_v4_analysis
done
echo "=== $(date '+%F %T') TIERC_DONE"
