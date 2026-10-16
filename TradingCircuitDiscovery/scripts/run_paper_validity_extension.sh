#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:?Policy-Value root is required}"
BASE_OUTPUT_ROOT="${2:?Base full-matrix output root is required}"
SUPPLEMENT_OUTPUT_ROOT="${3:?Supplement output root is required}"
TRANSFER_OUTPUT_ROOT="${4:?Transfer output root is required}"
PAPER_OUTPUT_ROOT="${5:?Paper-validity output root is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRANSFER_WORKERS="${TRANSFER_WORKERS:-1}"
TRANSFER_THREADS_PER_WORKER="${TRANSFER_THREADS_PER_WORKER:-4}"
TRANSFER_SKIP_GPU_GATE="${TRANSFER_SKIP_GPU_GATE:-0}"
TRANSFER_RANDOM_CONTROLS="${TRANSFER_RANDOM_CONTROLS:-50}"
TRANSFER_BOOTSTRAP_REPEATS="${TRANSFER_BOOTSTRAP_REPEATS:-20000}"
TRANSFER_BOOTSTRAP_WORKERS="${TRANSFER_BOOTSTRAP_WORKERS:-16}"

if ! [[ "$TRANSFER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRANSFER_WORKERS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TRANSFER_THREADS_PER_WORKER" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRANSFER_THREADS_PER_WORKER must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TRANSFER_RANDOM_CONTROLS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRANSFER_RANDOM_CONTROLS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TRANSFER_BOOTSTRAP_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRANSFER_BOOTSTRAP_REPEATS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TRANSFER_BOOTSTRAP_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRANSFER_BOOTSTRAP_WORKERS must be a positive integer" >&2
  exit 2
fi

export OMP_NUM_THREADS="$TRANSFER_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$TRANSFER_THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$TRANSFER_THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$TRANSFER_THREADS_PER_WORKER"

mkdir -p "$TRANSFER_OUTPUT_ROOT" "$PAPER_OUTPUT_ROOT"

gpu_gate_args=()
if [[ "$TRANSFER_SKIP_GPU_GATE" == "1" ]]; then
  gpu_gate_args+=(--skip-gpu-gate)
elif [[ "$TRANSFER_SKIP_GPU_GATE" != "0" ]]; then
  echo "TRANSFER_SKIP_GPU_GATE must be 0 or 1" >&2
  exit 2
fi

if [[ "$TRANSFER_WORKERS" -eq 1 ]]; then
  "$PYTHON_BIN" -m TradingCircuitDiscovery.experiments.run_transfer_matrix \
    --project-root "$PROJECT_ROOT" \
    --base-output-root "$BASE_OUTPUT_ROOT" \
    --supplement-output-root "$SUPPLEMENT_OUTPUT_ROOT" \
    --output-root "$TRANSFER_OUTPUT_ROOT" \
    --random-controls "$TRANSFER_RANDOM_CONTROLS" \
    --max-gpu-utilization 20 \
    "${gpu_gate_args[@]}" \
    --resume
else
  pids=()
  for ((shard_index = 0; shard_index < TRANSFER_WORKERS; shard_index++)); do
    manifest_args=()
    if [[ "$shard_index" -ne 0 ]]; then
      manifest_args+=(--skip-manifest-write)
    fi
    "$PYTHON_BIN" -u -m TradingCircuitDiscovery.experiments.run_transfer_matrix \
      --project-root "$PROJECT_ROOT" \
      --base-output-root "$BASE_OUTPUT_ROOT" \
      --supplement-output-root "$SUPPLEMENT_OUTPUT_ROOT" \
      --output-root "$TRANSFER_OUTPUT_ROOT" \
      --random-controls "$TRANSFER_RANDOM_CONTROLS" \
      --max-gpu-utilization 20 \
      "${gpu_gate_args[@]}" \
      --shard-count "$TRANSFER_WORKERS" \
      --shard-index "$shard_index" \
      "${manifest_args[@]}" \
      --resume > "$TRANSFER_OUTPUT_ROOT/shard_${shard_index}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one transfer shard failed" >&2
    exit 1
  fi
fi

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"$PYTHON_BIN" -m TradingCircuitDiscovery.experiments.aggregate_paper_validity \
  --supplement-output-root "$SUPPLEMENT_OUTPUT_ROOT" \
  --transfer-output-root "$TRANSFER_OUTPUT_ROOT" \
  --output-root "$PAPER_OUTPUT_ROOT" \
  --manifest "$TRANSFER_OUTPUT_ROOT/manifest.json" \
  --bootstrap-repeats "$TRANSFER_BOOTSTRAP_REPEATS" \
  --bootstrap-workers "$TRANSFER_BOOTSTRAP_WORKERS"
