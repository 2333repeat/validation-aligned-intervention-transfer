#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:?Policy-Value root is required}"
OUTPUT_ROOT="${2:?Output root is required}"

python -m TradingCircuitDiscovery.experiments.run_full_matrix \
  --project-root "$PROJECT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --config TradingCircuitDiscovery/configs/full_experiment.yaml \
  --resume
