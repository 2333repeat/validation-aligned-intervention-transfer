#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-3}"
MODE="${MODE:-full}"
MARKETS="${MARKETS:-HSI,N225,DJI,SP500,KS11,IXIC}"
ALGOS="${ALGOS:-PPO,DDQN,A2C,SAC}"
SEEDS="${SEEDS:-1,2,3}"
TRAIN_START="${TRAIN_START:-2012-01-01}"
TRAIN_END="${TRAIN_END:-2021-12-31}"
TEST_START="${TEST_START:-2022-01-01}"
TEST_END="${TEST_END:-2024-12-31}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-200000}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10000}"
USE_TMUX="${USE_TMUX:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --markets) MARKETS="$2"; shift 2 ;;
    --algos) ALGOS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --steps) TOTAL_ENV_STEPS="$2"; shift 2 ;;
    --eval-every) EVAL_EVERY_STEPS="$2"; shift 2 ;;
    --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
    --no-tmux) USE_TMUX=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "$MODE" == "smoke" ]]; then
  TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-512}"
  EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-128}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
job_dir="${ROOT_DIR}/result/job_manifests/${timestamp}"
mkdir -p "${job_dir}"
jobs_file="${job_dir}/jobs.txt"
: > "${jobs_file}"

IFS=',' read -r -a market_arr <<< "${MARKETS}"
IFS=',' read -r -a algo_arr <<< "${ALGOS}"
IFS=',' read -r -a seed_arr <<< "${SEEDS}"

for algo in "${algo_arr[@]}"; do
  for market in "${market_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      exp_name="${algo}-${market}-${DEVICE}-s${seed}"
      exp_name="${exp_name}${RUN_SUFFIX}"
      cmd="${PYTHON_EXE} train_cli.py --algorithm ${algo} --dataset ${market} --seed ${seed} --device ${DEVICE} --exp-name ${exp_name} --train-start ${TRAIN_START} --train-end ${TRAIN_END} --test-start ${TEST_START} --test-end ${TEST_END} --total-env-steps ${TOTAL_ENV_STEPS} --eval-every-steps ${EVAL_EVERY_STEPS} --random-reset"
      if [[ "$MODE" == "smoke" ]]; then
        cmd="${cmd} --smoke"
      fi
      echo "${cmd}" >> "${jobs_file}"
    done
  done
done

echo "[run_baselines] jobs=${jobs_file}"
echo "[run_baselines] workers=${WORKERS} device=${DEVICE} mode=${MODE}"

if [[ "${USE_TMUX}" == "1" ]] && command -v tmux >/dev/null 2>&1; then
  session="pv_baselines_${timestamp}"
  tmux new-session -d -s "${session}" -n "w1"
  for worker in $(seq 1 "${WORKERS}"); do
    worker_file="${job_dir}/worker_${worker}.txt"
    awk -v w="${worker}" -v n="${WORKERS}" 'NR % n == (w % n) {print}' "${jobs_file}" > "${worker_file}"
    if [[ "${worker}" -gt 1 ]]; then
      tmux new-window -t "${session}" -n "w${worker}"
    fi
    tmux send-keys -t "${session}:$((worker-1))" "cd '${ROOT_DIR}' && bash -lc 'while IFS= read -r cmd; do if [[ -n \"\$cmd\" ]]; then echo \"[worker ${worker}] \$cmd\"; eval \"\$cmd\"; fi; done < \"${worker_file}\"; echo worker_${worker}_done'" C-m
  done
  echo "[run_baselines] tmux_session=${session}"
  echo "[run_baselines] attach: tmux attach -t ${session}"
else
  xargs -I{} -P "${WORKERS}" bash -lc "cd '${ROOT_DIR}' && echo \"[worker] {}\" && eval \"{}\"" < "${jobs_file}"
fi
