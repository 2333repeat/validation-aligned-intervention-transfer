from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

from TradingCircuitDiscovery.collect.dataset import ActivationBank, flatten_activation_dataset
from TradingCircuitDiscovery.discovery.linear_probe import train_temporal_probe
from TradingCircuitDiscovery.discovery.neuron_ranking import rank_neurons_by_activation_difference
from TradingCircuitDiscovery.evaluation.statistics import (
    benjamini_hochberg,
    paired_bootstrap_ci,
    shuffled_labels,
)


def aggregate_full_results(
    output_root: str | Path,
    expected_run_ids: list[str],
    shuffle_repeats: int = 20,
    market_data_root: str | Path | None = None,
    transition_cost: float = 0.003,
) -> dict[str, int]:
    root = Path(output_root)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    statuses_by_id = {}
    metrics_by_id = {}
    circuits_by_id = {}
    distribution_effects_by_id = {}
    run_dirs_by_id = {}
    for status_path in root.glob("runs/*/*/seed_*/status.json"):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        statuses_by_id[payload["run_id"]] = payload
        run_dirs_by_id[payload["run_id"]] = status_path.parent
        metrics_path = status_path.parent / "metrics.json"
        if payload.get("status") == "completed" and metrics_path.is_file():
            metrics_by_id[payload["run_id"]] = json.loads(metrics_path.read_text(encoding="utf-8"))
            circuits_path = status_path.parent / "neuron_circuits.json"
            if circuits_path.is_file():
                circuits_by_id[payload["run_id"]] = json.loads(circuits_path.read_text(encoding="utf-8"))
            distribution_path = status_path.parent / "distribution_effects.json"
            if distribution_path.is_file():
                distribution_effects_by_id[payload["run_id"]] = json.loads(
                    distribution_path.read_text(encoding="utf-8")
                )
    status_rows = []
    for run_id in expected_run_ids:
        payload = statuses_by_id.get(run_id, {"run_id": run_id, "status": "missing", "error": "missing status"})
        status_rows.append({"run_id": run_id, "status": payload.get("status"), "error": payload.get("error")})
    status_frame = pd.DataFrame(status_rows)
    status_frame.to_csv(aggregate_dir / "run_status.csv", index=False)
    failures = status_frame[status_frame["status"] != "completed"].copy()
    failures.to_csv(aggregate_dir / "failures.csv", index=False)

    result_rows = []
    control_rows = []
    sae_rows = []
    performance_rows = []
    distribution_rows = []
    for run_id, payload in metrics_by_id.items():
        baseline = None
        for concept_payload in payload.get("concepts", {}).values():
            candidate = concept_payload.get("economic", {}).get("baseline", {})
            if candidate:
                baseline = candidate
                break
        if baseline:
            performance_rows.append(
                {
                    "run_id": run_id,
                    "algorithm": payload.get("algorithm"),
                    "market": payload.get("market"),
                    "seed": payload.get("seed"),
                    "pnl": baseline.get("pnl"),
                    "sharpe": baseline.get("sharpe"),
                    "mdd": baseline.get("mdd"),
                    "turnover": baseline.get("turnover"),
                    "final_capital": baseline.get("final_capital"),
                }
            )
        for concept, concept_payload in payload.get("concepts", {}).items():
            if concept_payload.get("status") == "skipped":
                continue
            probe_test = concept_payload.get("probe", {}).get("test", {})
            necessity = concept_payload.get("necessity", {})
            patching = concept_payload.get("patching", {})
            control = concept_payload.get("random_control", {})
            economic_delta = concept_payload.get("economic", {}).get("delta", {})
            circuit = circuits_by_id.get(run_id, {}).get(concept, {})
            distribution_payload = (
                distribution_effects_by_id.get(run_id, {}).get("concepts", {}).get(concept, {})
            )
            distribution_necessity = distribution_payload.get("necessity", {})
            distribution_patching = distribution_payload.get("patching", {})
            result_rows.append(
                {
                    "run_id": run_id,
                    "algorithm": payload.get("algorithm"),
                    "market": payload.get("market"),
                    "seed": payload.get("seed"),
                    "concept": concept,
                    "layer": concept_payload.get("layer"),
                    "circuit_size": len(circuit.get("neurons", [])),
                    "probe_balanced_accuracy": probe_test.get("balanced_accuracy"),
                    "probe_auroc": probe_test.get("auroc"),
                    "necessity_margin_change": necessity.get("mean_absolute_margin_change"),
                    "necessity_jsd": necessity.get("mean_jsd"),
                    "action_flip_rate": necessity.get("action_flip_rate"),
                    "patching_margin_change": patching.get("mean_absolute_margin_change"),
                    "patching_jsd": patching.get("mean_jsd"),
                    "distribution_necessity_margin_change": distribution_necessity.get(
                        "mean_absolute_margin_change"
                    ),
                    "distribution_necessity_jsd": distribution_necessity.get("mean_jsd"),
                    "distribution_patching_margin_change": distribution_patching.get(
                        "mean_absolute_margin_change"
                    ),
                    "distribution_patching_jsd": distribution_patching.get("mean_jsd"),
                    "random_control_mean": control.get("mean"),
                    "empirical_p": control.get("empirical_p"),
                    "delta_pnl": economic_delta.get("pnl"),
                    "delta_sharpe": economic_delta.get("sharpe"),
                    "delta_mdd": economic_delta.get("mdd"),
                    "delta_turnover": economic_delta.get("turnover"),
                }
            )
            if distribution_payload:
                distribution_rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": payload.get("algorithm"),
                        "market": payload.get("market"),
                        "seed": payload.get("seed"),
                        "concept": concept,
                        "layer": distribution_payload.get("layer"),
                        "distribution_semantics": distribution_effects_by_id[run_id].get(
                            "distribution_semantics"
                        ),
                        "necessity_action_flip_rate": distribution_necessity.get("action_flip_rate"),
                        "necessity_margin_change": distribution_necessity.get(
                            "mean_absolute_margin_change"
                        ),
                        "necessity_jsd": distribution_necessity.get("mean_jsd"),
                        "patching_action_flip_rate": distribution_patching.get("action_flip_rate"),
                        "patching_margin_change": distribution_patching.get(
                            "mean_absolute_margin_change"
                        ),
                        "patching_jsd": distribution_patching.get("mean_jsd"),
                    }
                )
            for control_index, effect in enumerate(control.get("effects", [])):
                control_rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": payload.get("algorithm"),
                        "market": payload.get("market"),
                        "seed": payload.get("seed"),
                        "concept": concept,
                        "control_index": control_index,
                        "control_effect": effect,
                        "observed_effect": necessity.get("mean_absolute_margin_change"),
                    }
                )
        for layer, sae_payload in payload.get("sae", {}).items():
            selected_dim = sae_payload.get("latent_dim")
            for candidate in sae_payload.get("candidates", []):
                sae_rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": payload.get("algorithm"),
                        "market": payload.get("market"),
                        "seed": payload.get("seed"),
                        "layer": layer,
                        **candidate,
                        "selected": candidate.get("latent_dim") == selected_dim,
                    }
                )
    main_results = pd.DataFrame(result_rows)
    main_results.to_csv(aggregate_dir / "main_results.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(aggregate_dir / "distribution_effects.csv", index=False)
    performance = pd.DataFrame(
        performance_rows,
        columns=[
            "run_id", "algorithm", "market", "seed", "pnl", "sharpe", "mdd",
            "turnover", "final_capital",
        ],
    )
    performance.to_csv(aggregate_dir / "policy_performance.csv", index=False)
    if len(performance):
        performance_summary = (
            performance.groupby(["algorithm", "market"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                mean_pnl=("pnl", "mean"),
                std_pnl=("pnl", "std"),
                mean_sharpe=("sharpe", "mean"),
                std_sharpe=("sharpe", "std"),
                mean_mdd=("mdd", "mean"),
                std_mdd=("mdd", "std"),
                mean_turnover=("turnover", "mean"),
                std_turnover=("turnover", "std"),
            )
            .reset_index()
        )
    else:
        performance_summary = pd.DataFrame(
            columns=[
                "algorithm", "market", "seeds", "mean_pnl", "std_pnl", "mean_sharpe",
                "std_sharpe", "mean_mdd", "std_mdd", "mean_turnover", "std_turnover",
            ]
        )
    performance_summary.to_csv(aggregate_dir / "policy_performance_summary.csv", index=False)
    pd.DataFrame(
        control_rows,
        columns=[
            "run_id", "algorithm", "market", "seed", "concept", "control_index",
            "control_effect", "observed_effect",
        ],
    ).to_csv(aggregate_dir / "random_controls.csv", index=False)
    pd.DataFrame(sae_rows).to_csv(aggregate_dir / "sae_ablation.csv", index=False)

    stability_rows = []
    circuit_groups: dict[tuple[str, str, str], list[tuple[int, set[str]]]] = {}
    for run_id, concepts in circuits_by_id.items():
        payload = metrics_by_id[run_id]
        for concept, circuit in concepts.items():
            key = (payload.get("algorithm"), payload.get("market"), concept)
            circuit_groups.setdefault(key, []).append(
                (int(payload.get("seed")), set(circuit.get("neurons", [])))
            )
    for (algorithm, market, concept), seed_circuits in sorted(circuit_groups.items()):
        similarities = []
        for (_, left), (_, right) in combinations(seed_circuits, 2):
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
        stability_rows.append(
            {
                "algorithm": algorithm,
                "market": market,
                "concept": concept,
                "seeds": len(seed_circuits),
                "seed_pairs": len(similarities),
                "mean_jaccard": np.mean(similarities) if similarities else np.nan,
                "min_jaccard": np.min(similarities) if similarities else np.nan,
                "max_jaccard": np.max(similarities) if similarities else np.nan,
            }
        )
    pd.DataFrame(
        stability_rows,
        columns=[
            "algorithm", "market", "concept", "seeds", "seed_pairs",
            "mean_jaccard", "min_jaccard", "max_jaccard",
        ],
    ).to_csv(aggregate_dir / "circuit_stability.csv", index=False)

    temporal_rows = []
    probe_control_rows = []
    probe_control_summaries = {}
    market_test_ranges = {}
    period_order = ("discovery", "validation", "test")
    for run_id, concepts in circuits_by_id.items():
        activation_path = run_dirs_by_id[run_id] / "activation" / "oos.pt"
        if not activation_path.is_file():
            continue
        bank = ActivationBank.load(activation_path)
        partitions = np.asarray([record.get("partition", "excluded") for record in bank.records])
        payload = metrics_by_id[run_id]
        test_dates = pd.to_datetime(
            [
                record.get("date")
                for record in bank.records
                if record.get("partition") == "test" and record.get("date") is not None
            ]
        )
        if len(test_dates):
            current = market_test_ranges.get(payload.get("market"))
            lower, upper = test_dates.min(), test_dates.max()
            market_test_ranges[payload.get("market")] = (
                min(current[0], lower) if current else lower,
                max(current[1], upper) if current else upper,
            )
        for concept, circuit in concepts.items():
            layer = circuit.get("layer")
            if not layer:
                continue
            try:
                activations, labels = flatten_activation_dataset(bank.records, layer=layer, concept=concept)
            except ValueError:
                continue
            if len(activations) != len(partitions):
                continue
            discovery_mask = partitions == "discovery"
            observed_probe = (
                payload.get("concepts", {})
                .get(concept, {})
                .get("probe", {})
                .get("test", {})
                .get("balanced_accuracy")
            )
            null_scores = []
            if observed_probe is not None and len(np.unique(labels[discovery_mask])) == 2:
                for repeat in range(int(shuffle_repeats)):
                    null_labels = labels.copy()
                    null_labels[discovery_mask] = shuffled_labels(
                        labels[discovery_mask],
                        random_state=int(payload.get("seed", 1)) * 1000 + repeat,
                    )
                    null_probe = train_temporal_probe(
                        activations,
                        null_labels,
                        partitions,
                        random_state=int(payload.get("seed", 1)) * 1000 + repeat,
                    )
                    null_score = null_probe.test_metrics["balanced_accuracy"]
                    null_scores.append(null_score)
                    probe_control_rows.append(
                        {
                            "run_id": run_id,
                            "algorithm": payload.get("algorithm"),
                            "market": payload.get("market"),
                            "seed": payload.get("seed"),
                            "concept": concept,
                            "layer": layer,
                            "repeat": repeat,
                            "observed_balanced_accuracy": observed_probe,
                            "shuffled_balanced_accuracy": null_score,
                        }
                    )
            if null_scores:
                probe_control_summaries[(run_id, concept)] = {
                    "mean": float(np.mean(null_scores)),
                    "std": float(np.std(null_scores)),
                    "empirical_p": float(
                        (1 + sum(score >= observed_probe for score in null_scores))
                        / (1 + len(null_scores))
                    ),
                }
            ranked_by_period = {}
            for period in period_order:
                mask = partitions == period
                if not mask.any() or len(np.unique(labels[mask])) != 2:
                    continue
                ranking = rank_neurons_by_activation_difference(
                    activations[mask],
                    labels[mask],
                    top_k=min(8, activations.shape[1]),
                )
                ranked_by_period[period] = {item.index for item in ranking}
            for period_a, period_b in combinations(period_order, 2):
                if period_a not in ranked_by_period or period_b not in ranked_by_period:
                    continue
                left = ranked_by_period[period_a]
                right = ranked_by_period[period_b]
                union = left | right
                temporal_rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": payload.get("algorithm"),
                        "market": payload.get("market"),
                        "seed": payload.get("seed"),
                        "concept": concept,
                        "layer": layer,
                        "period_a": period_a,
                        "period_b": period_b,
                        "top_k": len(left),
                        "jaccard": len(left & right) / len(union) if union else 1.0,
                    }
                )
    pd.DataFrame(
        temporal_rows,
        columns=[
            "run_id", "algorithm", "market", "seed", "concept", "layer",
            "period_a", "period_b", "top_k", "jaccard",
        ],
    ).to_csv(aggregate_dir / "temporal_feature_stability.csv", index=False)
    pd.DataFrame(
        probe_control_rows,
        columns=[
            "run_id", "algorithm", "market", "seed", "concept", "layer", "repeat",
            "observed_balanced_accuracy", "shuffled_balanced_accuracy",
        ],
    ).to_csv(aggregate_dir / "probe_shuffle_controls.csv", index=False)

    market_baseline_rows = []
    if market_data_root is not None:
        data_root = Path(market_data_root)
        for market, (start_date, end_date) in sorted(market_test_ranges.items()):
            data_path = data_root / f"{market}.csv"
            if not data_path.is_file():
                continue
            market_data = pd.read_csv(data_path)
            market_data["Date"] = pd.to_datetime(market_data["Date"])
            aligned = market_data.loc[
                (market_data["Date"] >= start_date) & (market_data["Date"] <= end_date),
                ["Date", "Close"],
            ].sort_values("Date")
            if len(aligned) < 2:
                continue
            closes = aligned["Close"].astype(float).to_numpy()
            daily_returns = closes[1:] / closes[:-1] - 1.0
            equity = closes / closes[0]
            peak = np.maximum.accumulate(equity)
            market_baseline_rows.append(
                {
                    "market": market,
                    "start_date": aligned["Date"].iloc[0].date().isoformat(),
                    "end_date": aligned["Date"].iloc[-1].date().isoformat(),
                    "observations": len(aligned),
                    "gross_pnl": float(closes[-1] / closes[0] - 1.0),
                    "net_pnl": float(
                        closes[-1] * (1.0 - transition_cost)
                        / (closes[0] * (1.0 + transition_cost))
                        - 1.0
                    ),
                    "sharpe": float(
                        np.mean(daily_returns)
                        / (np.std(daily_returns) + 1e-12)
                        * np.sqrt(252)
                    ),
                    "mdd": float(np.min(equity / np.maximum(peak, 1e-12) - 1.0)),
                    "transition_cost": float(transition_cost),
                }
            )
    market_baselines = pd.DataFrame(
        market_baseline_rows,
        columns=[
            "market", "start_date", "end_date", "observations", "gross_pnl", "net_pnl",
            "sharpe", "mdd", "transition_cost",
        ],
    )
    market_baselines.to_csv(aggregate_dir / "market_baselines.csv", index=False)
    baseline_for_merge = market_baselines.rename(
        columns={
            "gross_pnl": "buy_hold_gross_pnl",
            "net_pnl": "buy_hold_net_pnl",
            "sharpe": "buy_hold_sharpe",
            "mdd": "buy_hold_mdd",
        }
    )[["market", "buy_hold_gross_pnl", "buy_hold_net_pnl", "buy_hold_sharpe", "buy_hold_mdd"]]
    performance = performance.drop(
        columns=[
            "buy_hold_gross_pnl", "buy_hold_net_pnl", "buy_hold_sharpe", "buy_hold_mdd",
            "excess_pnl", "excess_sharpe",
        ],
        errors="ignore",
    ).merge(baseline_for_merge, on="market", how="left")
    performance["excess_pnl"] = performance["pnl"] - performance["buy_hold_net_pnl"]
    performance["excess_sharpe"] = performance["sharpe"] - performance["buy_hold_sharpe"]
    performance.to_csv(aggregate_dir / "policy_performance.csv", index=False)
    if len(performance):
        performance_summary = (
            performance.groupby(["algorithm", "market"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                mean_pnl=("pnl", "mean"),
                std_pnl=("pnl", "std"),
                mean_sharpe=("sharpe", "mean"),
                std_sharpe=("sharpe", "std"),
                mean_mdd=("mdd", "mean"),
                std_mdd=("mdd", "std"),
                mean_turnover=("turnover", "mean"),
                std_turnover=("turnover", "std"),
                mean_excess_pnl=("excess_pnl", "mean"),
                mean_excess_sharpe=("excess_sharpe", "mean"),
            )
            .reset_index()
        )
        performance_summary.to_csv(aggregate_dir / "policy_performance_summary.csv", index=False)
    if len(main_results):
        main_results["probe_shuffle_mean"] = [
            probe_control_summaries.get((row.run_id, row.concept), {}).get("mean")
            for row in main_results.itertuples()
        ]
        main_results["probe_shuffle_std"] = [
            probe_control_summaries.get((row.run_id, row.concept), {}).get("std")
            for row in main_results.itertuples()
        ]
        main_results["probe_shuffle_empirical_p"] = [
            probe_control_summaries.get((row.run_id, row.concept), {}).get("empirical_p")
            for row in main_results.itertuples()
        ]
        main_results.to_csv(aggregate_dir / "main_results.csv", index=False)
    if len(main_results):
        statistic_rows = []
        for (algorithm, market, concept), group in main_results.groupby(
            ["algorithm", "market", "concept"], dropna=False
        ):
            paired = group[["necessity_margin_change", "random_control_mean"]].dropna()
            if len(paired):
                bootstrap = paired_bootstrap_ci(
                    paired["necessity_margin_change"].to_numpy(),
                    paired["random_control_mean"].to_numpy(),
                    random_state=1,
                )
            else:
                bootstrap = None
            run_p_values = group["empirical_p"].dropna().to_numpy(dtype=float)
            combined_p = (
                float(combine_pvalues(np.clip(run_p_values, 1e-12, 1.0), method="fisher")[1])
                if len(run_p_values)
                else 1.0
            )
            probe_p_values = group["probe_shuffle_empirical_p"].dropna().to_numpy(dtype=float)
            probe_combined_p = (
                float(combine_pvalues(np.clip(probe_p_values, 1e-12, 1.0), method="fisher")[1])
                if len(probe_p_values)
                else 1.0
            )
            statistic_rows.append(
                {
                    "algorithm": algorithm,
                    "market": market,
                    "concept": concept,
                    "n": int(group["seed"].count()),
                    "mean_probe_balanced_accuracy": group["probe_balanced_accuracy"].mean(),
                    "mean_necessity_margin_change": group["necessity_margin_change"].mean(),
                    "mean_random_control": group["random_control_mean"].mean(),
                    "mean_effect_over_control": bootstrap.mean_difference if bootstrap else np.nan,
                    "bootstrap_ci_low": bootstrap.ci_low if bootstrap else np.nan,
                    "bootstrap_ci_high": bootstrap.ci_high if bootstrap else np.nan,
                    "mean_empirical_p": group["empirical_p"].mean(),
                    "combined_p": combined_p,
                    "probe_combined_p": probe_combined_p,
                }
            )
        statistics = pd.DataFrame(statistic_rows)
        statistics["fdr_q"] = benjamini_hochberg(statistics["combined_p"].to_numpy())
        statistics["probe_fdr_q"] = benjamini_hochberg(statistics["probe_combined_p"].to_numpy())
    else:
        statistics = pd.DataFrame(
            columns=[
                "algorithm", "market", "concept", "n", "mean_probe_balanced_accuracy",
                "mean_necessity_margin_change", "mean_random_control", "mean_effect_over_control",
                "bootstrap_ci_low", "bootstrap_ci_high", "mean_empirical_p", "combined_p", "fdr_q",
                "probe_combined_p", "probe_fdr_q",
            ]
        )
    statistics.to_csv(aggregate_dir / "statistical_tests.csv", index=False)
    summary = {
        "expected": len(expected_run_ids),
        "completed": int((status_frame["status"] == "completed").sum()),
        "failed": int((status_frame["status"] == "failed").sum()),
        "missing": int((status_frame["status"] == "missing").sum()),
    }
    report = (
        "# Full Circuit Discovery Aggregate\n\n"
        f"- Expected runs: {summary['expected']}\n"
        f"- Completed runs: {summary['completed']}\n"
        f"- Failed runs: {summary['failed']}\n"
        f"- Missing runs: {summary['missing']}\n"
        f"- Result rows: {len(main_results)}\n"
        f"- Random-control rows: {len(control_rows)}\n"
        f"- SAE candidate rows: {len(sae_rows)}\n"
        f"- Stability groups: {len(stability_rows)}\n"
        f"- Temporal-stability rows: {len(temporal_rows)}\n"
        f"- Probe shuffled-label rows: {len(probe_control_rows)}\n"
        f"- Policy-performance rows: {len(performance_rows)}\n"
        f"- Market-baseline rows: {len(market_baseline_rows)}\n"
        f"- Distribution-effect rows: {len(distribution_rows)}\n"
    )
    (aggregate_dir / "report.md").write_text(report, encoding="utf-8")
    return summary
