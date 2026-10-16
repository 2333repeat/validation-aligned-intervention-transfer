# Validation-Aligned Intervention Transfer

Source code and daily market data for **Validation-Aligned Intervention Transfer: Auditing Explanation Portability in Financial Reinforcement Learning**.

Validation-Aligned Intervention Transfer (VAIT) evaluates whether an intervention on a compact hidden-unit subset transfers between independently trained financial reinforcement learning policies. It learns unit correspondences from validation-period activations, transfers subset membership, and replaces selected activations with target-policy discovery means. Later observations test both intervention sensitivity and correspondence with a subset selected directly in the target policy.

## Repository structure

| Path | Contents |
| --- | --- |
| `Policy-Value/` | PPO, A2C, DDQN, and discrete SAC agents, training, and trading environment |
| `Policy-Value/Data/` | Daily market CSV files for 15 equities and indices |
| `TradingCircuitDiscovery/` | Activation collection, subset discovery, alignment, interventions, and evaluation |
| `TradingCircuitDiscovery/configs/` | Discovery and supplementary-analysis settings |
| `TradingCircuitDiscovery/experiments/` | Experiment runners and statistical aggregation |

## Installation

Use Python 3.10 or later. The code has been checked with Python 3.12. From the repository root, install the package and its dependencies:

```bash
python -m pip install -e .
```

Dependencies are also listed in `requirements.txt`. Full-matrix execution uses an NVIDIA GPU, a CUDA-enabled PyTorch installation, and `nvidia-smi` available on the command path. Individual policy training also accepts `--device cpu`. Start from the repository root; multiline examples use Bash syntax.

## Data and audit periods

Market files are stored as `Policy-Value/Data/<market>.csv`:

```text
AAPL  AMD   AMZN  DJI   FCHI
GE    GOOGL HSI   INTC  IXIC
KS11  MSFT  N225  NVDA  SP500
```

The daily Yahoo Finance data contain `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. `SP500.csv` additionally contains `Adj Close`. File date ranges vary; each run selects its configured date window.

| Period | Role |
| --- | --- |
| 2012–2021 | Policy training |
| 2022 | Concept probes, neuron ranking, and replacement means |
| 2023 | Layer and subset selection; source–target alignment |
| 2024 | Intervention-response evaluation |

The policies use rolling 20-day observations with seven features: OHLCV and 5- and 10-day moving averages. VAIT estimates subsets and mappings from the 2022–2023 observations and evaluates their intervention responses on the 2024 observations.

The training entry point evaluates checkpoints over 2022–2024 by default and saves the checkpoint with the best PnL. The 2024 audit period is held out from subset and mapping selection; it overlaps this default checkpoint-evaluation window.

## Train policies

Train one PPO policy on HSI:

```bash
cd Policy-Value
python train_cli.py \
  --algorithm PPO --dataset HSI --seed 1 --device cuda \
  --total-env-steps 200000 --eval-every-steps 10000 --random-reset
cd ..
```

To train the complete matrix of four algorithms, 15 markets, and three seeds:

```bash
bash Policy-Value/scripts/run_baselines.sh \
  --markets AAPL,AMD,AMZN,DJI,FCHI,GE,GOOGL,HSI,INTC,IXIC,KS11,MSFT,N225,NVDA,SP500 \
  --algos PPO,DDQN,A2C,SAC --seeds 1,2,3 \
  --device cuda --workers 1 --steps 200000 --eval-every 10000 --no-tmux
```

Training writes checkpoints and account files under `Policy-Value/result/`. The matrix runners use the `ALGORITHM-MARKET-cuda-sSEED` naming convention produced by this batch command. Complete all 180 training runs before starting the matrix audit.

## Run the VAIT audit

The following stages use the trained checkpoints in order.

### 1. Collect activations and discover subsets

```bash
python -m TradingCircuitDiscovery.experiments.run_full_matrix \
  --project-root Policy-Value \
  --output-root outputs/discovery \
  --config TradingCircuitDiscovery/configs/full_experiment.yaml \
  --resume
```

### 2. Select compact subsets and run supplementary analyses

```bash
python -m TradingCircuitDiscovery.experiments.run_supplement_matrix \
  --project-root Policy-Value \
  --base-output-root outputs/discovery \
  --output-root outputs/supplement \
  --config TradingCircuitDiscovery/configs/supplement_experiment.yaml \
  --resume
```

### 3. Transfer interventions between policies

```bash
python -m TradingCircuitDiscovery.experiments.run_transfer_matrix \
  --project-root Policy-Value \
  --base-output-root outputs/discovery \
  --supplement-output-root outputs/supplement \
  --output-root outputs/transfer \
  --random-controls 50 --resume
```

The transfer runner evaluates eight mapping rules: absolute- and signed-correlation Hungarian matching, Procrustes, CCA, optimal transport, weight matching, ridge mapping, and identity. Comparisons use same-index subsets, size-matched random subsets, and subsets selected directly in the target policy.

### 4. Aggregate results

```bash
python -m TradingCircuitDiscovery.experiments.aggregate_paper_validity \
  --supplement-output-root outputs/supplement \
  --transfer-output-root outputs/transfer \
  --manifest outputs/discovery/manifest.json \
  --output-root outputs/summary \
  --bootstrap-repeats 20000
```

Outputs contain per-policy records, transfer responses, aggregate tables, and statistical summaries. `--resume` continues the matrix stages using their completed-run records. Training and analysis outputs are excluded from Git by `.gitignore`.

## Main implementation files

- [Policy training](Policy-Value/train_cli.py)
- [Subset discovery](TradingCircuitDiscovery/discovery/circuit_search.py)
- [Alignment methods](TradingCircuitDiscovery/evaluation/alignment_methods.py)
- [Target-policy interventions](TradingCircuitDiscovery/experiments/run_transfer_checkpoint.py)
- [Dependence-aware inference](TradingCircuitDiscovery/evaluation/dependence_bootstrap.py)
- [Result aggregation](TradingCircuitDiscovery/experiments/aggregate_paper_validity.py)
