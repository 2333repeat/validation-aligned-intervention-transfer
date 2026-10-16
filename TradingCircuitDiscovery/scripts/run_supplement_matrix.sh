#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:?Policy-Value root is required}"
BASE_OUTPUT_ROOT="${2:?Base full-matrix output root is required}"
OUTPUT_ROOT="${3:?Supplement output root is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUPPLEMENT_WORKERS="${SUPPLEMENT_WORKERS:-1}"
SUPPLEMENT_THREADS_PER_WORKER="${SUPPLEMENT_THREADS_PER_WORKER:-4}"
SUPPLEMENT_SKIP_GPU_GATE="${SUPPLEMENT_SKIP_GPU_GATE:-0}"

if ! [[ "$SUPPLEMENT_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUPPLEMENT_WORKERS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$SUPPLEMENT_THREADS_PER_WORKER" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUPPLEMENT_THREADS_PER_WORKER must be a positive integer" >&2
  exit 2
fi

export OMP_NUM_THREADS="$SUPPLEMENT_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$SUPPLEMENT_THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$SUPPLEMENT_THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$SUPPLEMENT_THREADS_PER_WORKER"

mkdir -p "$OUTPUT_ROOT"

gpu_gate_args=()
if [[ "$SUPPLEMENT_SKIP_GPU_GATE" == "1" ]]; then
  gpu_gate_args+=(--skip-gpu-gate)
elif [[ "$SUPPLEMENT_SKIP_GPU_GATE" != "0" ]]; then
  echo "SUPPLEMENT_SKIP_GPU_GATE must be 0 or 1" >&2
  exit 2
fi

if [[ "$SUPPLEMENT_WORKERS" -eq 1 ]]; then
  "$PYTHON_BIN" -m TradingCircuitDiscovery.experiments.run_supplement_matrix \
    --project-root "$PROJECT_ROOT" \
    --base-output-root "$BASE_OUTPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --config TradingCircuitDiscovery/configs/supplement_experiment.yaml \
    --max-gpu-utilization 20 \
    "${gpu_gate_args[@]}" \
    --resume
else
  pids=()
  for ((shard_index = 0; shard_index < SUPPLEMENT_WORKERS; shard_index++)); do
    manifest_args=()
    if [[ "$shard_index" -ne 0 ]]; then
      manifest_args+=(--skip-manifest-write)
    fi
    "$PYTHON_BIN" -u -m TradingCircuitDiscovery.experiments.run_supplement_matrix \
      --project-root "$PROJECT_ROOT" \
      --base-output-root "$BASE_OUTPUT_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --config TradingCircuitDiscovery/configs/supplement_experiment.yaml \
      --max-gpu-utilization 20 \
      "${gpu_gate_args[@]}" \
      --shard-count "$SUPPLEMENT_WORKERS" \
      --shard-index "$shard_index" \
      "${manifest_args[@]}" \
      --resume > "$OUTPUT_ROOT/shard_${shard_index}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one supplement shard failed" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m TradingCircuitDiscovery.experiments.aggregate_supplement \
  --output-root "$OUTPUT_ROOT" \
  --base-output-root "$BASE_OUTPUT_ROOT" \
  --manifest "$OUTPUT_ROOT/manifest.json" \
  --cluster-bootstrap-repeats 2000
