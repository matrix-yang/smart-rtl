#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
module=${1:-signed_shifter}
run_root=${MAGE_RUN_ROOT:-$project_root/artifacts/experiment/mage-chipverilog-chipmate-smoke}
chipverilog_root=${CHIPVERILOG_ROOT:-/nfs/FM/yangyifan/rtl_bench/ChipVerilog}
image=${MAGE_CHIPVERILOG_IMAGE:-mage-chipverilog:chipmate-v9b}
model_name=${MAGE_RESULT_MODEL_NAME:-mage_chipmate_v9b_smoke}

mkdir -p "$run_root/output" "$run_root/logs"
docker run --rm \
  -e PYTHONPATH=/opt/mage-chipverilog/src \
  -e MAGE_API_BASE_URL=${MAGE_API_BASE_URL:-http://192.168.16.17:30000/v1} \
  -e MAGE_API_KEY=${MAGE_API_KEY:-EMPTY} \
  -e MAGE_MODEL=${MAGE_MODEL:-ChipMATE-V-9B} \
  -e MAGE_API_TIMEOUT=${MAGE_API_TIMEOUT:-1800} \
  -e MAGE_API_RETRIES=${MAGE_API_RETRIES:-3} \
  -v "$script_dir:/opt/mage-chipverilog:ro" \
  -v "$chipverilog_root:/chipverilog" \
  -v "$run_root:/experiment" \
  "$image" python3 /opt/mage-chipverilog/scripts/run_chipverilog.py \
    --chipverilog-root /chipverilog \
    --output-root /experiment/output \
    --log-root /experiment/logs \
    --model-name "$model_name" \
    --model-id ${MAGE_MODEL:-ChipMATE-V-9B} \
    --api-base-url ${MAGE_API_BASE_URL:-http://192.168.16.17:30000/v1} \
    --context-window ${MAGE_CONTEXT_WINDOW:-32768} \
    --spec-char-budget ${MAGE_SPEC_CHAR_BUDGET:-48000} \
    --llm-parallel-requests ${MAGE_LLM_PARALLEL_REQUESTS:-1} \
    --max-tokens ${MAGE_MAX_OUTPUT_TOKENS:-8192} \
    --modules "$module"

