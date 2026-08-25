#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

platform_slug=${QUALIFICATION_PLATFORM:-$(hostname | tr -d '\n' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-')}
model_tag=${QUALIFICATION_MODEL:-qwen3.5:4b-q4_K_M}

run_case() {
  local mode=$1
  local number=$2
  local output=$3
  local telemetry=()
  if [[ $mode == on ]]; then
    telemetry=(--memory-telemetry)
  fi
  echo "Running pair $number telemetry ${mode^^}..."
  LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC=$interval \
    bash run_bench.sh \
      --ui none \
      --tests llm \
      --llm-models "$model_tag" \
      --max-prompt-tokens 2048 \
      --warmup 2 \
      --runs 3 \
      "${telemetry[@]}" \
      --out "$output"
}

run_interval() {
  interval=$1
  local interval_dir=$2
  local output_dir="results/qualification/$platform_slug/$interval_dir"
  mkdir -p "$output_dir"

  for pair in $(seq 1 20); do
    local number
    number=$(printf '%02d' "$pair")
    local off_file="$output_dir/off-$number.json"
    local on_file="$output_dir/on-$number.json"

    if [[ -f $off_file && -f $on_file ]]; then
      echo "Pair $number already complete - skipping."
      continue
    fi
    if [[ -f $off_file || -f $on_file ]]; then
      echo "A partial pair exists in $output_dir. Remove or relocate pair $number, then retry." >&2
      return 3
    fi

    if (( pair % 2 == 1 )); then
      run_case off "$number" "$off_file"
      sleep 30
      run_case on "$number" "$on_file"
    else
      run_case on "$number" "$on_file"
      sleep 30
      run_case off "$number" "$off_file"
    fi
    if (( pair < 20 )); then
      sleep 30
    fi
  done
  echo "Qualification interval $interval completed."
}

run_selected_interval() {
  case $1 in
    1.0) run_interval 1.0 1s ;;
    0.5) run_interval 0.5 0.5s ;;
    0.25) run_interval 0.25 0.25s ;;
    *) echo "Usage: bash qual.sh [1.0|0.5|0.25]" >&2; return 2 ;;
  esac
}

if (( $# > 1 )); then
  echo "Usage: bash qual.sh [1.0|0.5|0.25]" >&2
  exit 2
fi
if (( $# == 1 )); then
  run_selected_interval "$1"
else
  run_selected_interval 1.0
  sleep 30
  run_selected_interval 0.5
  sleep 30
  run_selected_interval 0.25
  echo "All qualification intervals completed."
fi
