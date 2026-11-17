# Validation-Aligned Intervention Transfer

Code and reproducibility artifacts for **Validation-Aligned Intervention Transfer: Auditing Explanation Portability in Financial Reinforcement Learning**.

VAIT tests whether a compact hidden-unit intervention transfers between independently trained financial reinforcement learning policies. A validation-fitted correspondence transfers subset membership; intervention values are recomputed from the target policy's discovery observations. Evaluation separates intervention strength from correspondence to a subset selected independently in the target.

## Current experiment: 600 frozen policies

The current release uses four algorithms (A2C, PPO, DDQN, discrete SAC), 15 markets, and ten seeds per condition. It reuses terminal checkpoints trained for 200,000 steps through 2022. It does not select checkpoints by 2025 or 2026 returns.

| Period | Role |
| --- | --- |
| 2012–2022 | Archived policy training |
| 2023 | Concept discovery, candidate ranking, replacement means |
| 2024 | Layer/subset selection and source–target alignment |
| 2025 | First evaluation period |
| January–October 2026 | Second evaluation period; last observation 30 October |

This extension is retrospective. Subsets and maps were frozen before evaluation inference; the historical outcomes already existed. Both periods reuse the same policies and maps.

## Install and obtain the frozen artifact

Python 3.12 is the tested environment (Python 3.11 or later is required by the release tools). The current audit runs on CPU; a GPU is not required to replay the frozen models.

```bash
python -m pip install -e ".[test]"
python tools/fetch_vait_artifact.py
python tools/replay_vait_recent_snapshot.py --verify-only
```

The download script retrieves and joins the versioned [600-policy artifact] parts(https://github.com/2333repeat/validation-aligned-intervention-transfer/releases/tag/v0.2.0-vait600), verifies its SHA-256 against [release.json](evidence/vait600/release.json), and extracts it to `artifacts/vait600/`. It includes the 600 checkpoints, their policy implementation, fixed inputs, discovery/selection/mapping records, row-level results, and aggregate tables. Per-file checksums are in `MANIFEST.json` inside the artifact.

An existing `artifacts/vait600/` is preserved. To use an artifact stored elsewhere, pass `--artifact-root /path/to/vait600` to the replay script.

## Replay the experiment

From the repository root:

```bash
python tools/replay_vait_recent_snapshot.py --workers 2
```

This runs preparation, discovery and alignment, evaluation, then 20,000 bootstrap resamples per endpoint. It uses the frozen market snapshots without fetching prices and writes to `replay_output/`, preserving the published snapshot. Full replay performs the complete 600-policy audit; it is not an instant demonstration.

To check input preparation without running the full audit:

```bash
python tools/replay_vait_recent_snapshot.py --prepare-only --output replay_preparation
```

Compare the completed `replay_output/aggregate/facts.json` and CSV tables with `artifacts/vait600/snapshot/aggregate/`. Numerical-library versions and degenerate assignment ties may cause floating-point differences. Exact package versions used for the original run are recorded in [requirements-observed.txt](requirements-observed.txt); its CUDA-specific PyTorch build is provenance, not a requirement for CPU inference.

The public configuration is [vait_600_recent_20261117.json](TradingCircuitDiscovery/configs/vait_600_recent_20261117.json). `VAIT_RECENT_CONFIG` can select an alternative configuration file. The replay wrapper resolves artifact paths and writes a separate configuration for the new run. The preparation module's direct entry point can download data when no snapshot exists; use the replay wrapper for reproduction from the published snapshots.

## Results and interpretation

Each evaluation period has 32,400 nominal directed concept transfers, 28,042 eligible transfers, and 22,760 primary transfers after excluding the complementary sideways label. Eight mappings produce 448,672 eligible method evaluations across both periods; these are dependent records, not independent samples.

| Primary map minus same-index | Mean sensitivity difference | 95% multiway bootstrap interval |
| --- | ---: | --- |
| 2025 | 0.035735 | [0.031417, 0.040154] |
| 2026 January–October | 0.037633 | [0.033152, 0.042051] |

Four of six response-correspondence endpoints improve under the reported approximate bootstrap analysis. Both Jensen–Shannon-divergence contrasts have intervals spanning zero. Stronger sensitivity can accompany poorer distributional correspondence, supporting partial response portability in this policy matrix.

The [evidence directory](evidence/vait600/) contains aggregate estimates, uncertainty intervals, eligibility counts, and model/data manifests. The artifact contains the row-level results. Missing response values are omitted from their paired endpoint, never replaced with zero. Conditional action-redirection agreement uses 22,610 records in 2025 and 22,577 in 2026. BH adjustments are grouped jointly across years within the documented families; their formal calibration under the complete dependence structure has not been demonstrated.

The audit measures commands and normalized output scores. DDQN uses unit-temperature softmax Q scores, not calibrated trading probabilities; commands are not position transitions. Financial diagnostics include weak policies and 13 nonpositive-equity replay paths in 2026 and do not establish investment gains from explanation transfer.

## Implementation and checks

| File | Role |
| --- | --- |
| `experiments/prepare_vait_recent.py` | Checkpoint manifests and chronological observation caches |
| `experiments/vait_recent_core.py` | Batched interventions, selection and eight correspondence rules |
| `experiments/run_vait_recent.py` | Discovery, alignment and global freeze |
| `experiments/evaluate_vait_recent.py` | Evaluation on frozen choices and shared observations |
| `experiments/aggregate_vait_recent.py` | Hierarchical estimates, resampling and BH families |
| `evaluation/dependence_bootstrap.py` | Market-first, reciprocal-dyad and multiway resampling |

Paths in this table are relative to `TradingCircuitDiscovery/`. With the artifact extracted, run:

```bash
python -m pytest TradingCircuitDiscovery/tests/test_vait_recent.py -q
```

These checks cover original-adapter equivalence for all four algorithms, metrics and mapping projection, accounting, temporal input isolation and rejection of modified frozen inputs. The public artifact relocates machine-specific metadata and updates the dependent metadata hashes; [PACKAGING.json](evidence/vait600/PACKAGING.json) records the original and distributed digests. Scientific configuration, weights, observations, selections, mappings and numerical CSV results are preserved.

Market data originate from Yahoo Finance; retrieval provenance and snapshot hashes are retained. Third-party data keep their source terms. The [legacy 180-policy instructions](docs/legacy_180_policy.md) document the earlier 2024 experiment and are not the protocol for this release.
