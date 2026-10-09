#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
benchmark_root=${CVDP_BENCHMARK_ROOT:-/home/yangyifan/ai4chipdesign/cvdp_benchmark}
dataset=${CVDP_DATASET:-/home/yangyifan/ai4chipdesign/cvdp-benchmark-dataset/cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl}
task_id=${1:-cvdp_agentic_binary_to_gray_0003}
work_dir=${CVDP_WORK_DIR:-$project_root/artifacts/experiment/mage-cvdp-chipmate-smoke}
agent_image=${CVDP_AGENT_IMAGE:-cvdp-mage-chipmate-v9b:latest}

export MAGE_API_BASE_URL=${MAGE_API_BASE_URL:-http://192.168.16.17:30000/v1}
export MAGE_API_KEY=${MAGE_API_KEY:-EMPTY}
export MAGE_MODEL=${MAGE_MODEL:-ChipMATE-V-9B}
export MAGE_API_TIMEOUT=${MAGE_API_TIMEOUT:-1800}
export MAGE_API_RETRIES=${MAGE_API_RETRIES:-3}
export MAGE_MAX_OUTPUT_TOKENS=${MAGE_MAX_OUTPUT_TOKENS:-8192}
export MAGE_CONTEXT_CHAR_BUDGET=${MAGE_CONTEXT_CHAR_BUDGET:-48000}
export MAGE_TEMPERATURE=${MAGE_TEMPERATURE:-0.2}
export MAGE_TOP_P=${MAGE_TOP_P:-0.95}
export CVDP_AGENT_ENV=MAGE_API_BASE_URL,MAGE_API_KEY,MAGE_MODEL,MAGE_API_TIMEOUT,MAGE_API_RETRIES,MAGE_MAX_OUTPUT_TOKENS,MAGE_CONTEXT_CHAR_BUDGET,MAGE_TEMPERATURE,MAGE_TOP_P
unset CVDP_AGENT_MOUNTS || true
export PYTHONPATH="$benchmark_root/.deps${PYTHONPATH:+:$PYTHONPATH}"

cd "$benchmark_root"
python3 run_benchmark.py -f "$dataset" -i "$task_id" -l \
  -g "$agent_image" -p "$work_dir"

