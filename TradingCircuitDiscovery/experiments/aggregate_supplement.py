from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from TradingCircuitDiscovery.collect.dataset import ActivationBank
from TradingCircuitDiscovery.evaluation.representation_stability import (
    aligned_jaccard,
    hungarian_neuron_alignment,
    linear_cka,
    orthogonal_procrustes_residual,
)
from TradingCircuitDiscovery.evaluation.statistics import benjamini_hochberg


EFFECT_METRICS = (
    "mean_absolute_margin_change",
    "mean_jsd",
    "action_flip_rate",
    "mean_signed_margin_change",
)
ECONOMIC_METRICS = ("pnl", "sharpe", "mdd", "turnover")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    ordered = columns + [column for column in frame.columns if column not in columns]
    frame.loc[:, ordered].to_csv(path, index=False)


def _effect_fields(payload: dict, prefix: str = "") -> dict:
    return {
        f"{prefix}{metric}": payload.get(metric)
        for metric in EFFECT_METRICS
        if metric in payload
    }


def _identity(payload: dict, concept: str | None = None) -> dict:
    row = {
        "run_id": payload["run_id"],
        "algorithm": payload["algorithm"],
        "market": payload["market"],
        "seed": int(payload["seed"]),
        "policy_label": payload.get("validation_policy", {}).get("label"),
    }
    if concept is not None:
        row["concept"] = concept
    return row


def _jaccard(left, right) -> float:
    left_set = {int(value) for value in left}
    right_set = {int(value) for value in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _activation_matrix_by_key(bank: ActivationBank, layer: str) -> dict[tuple[str, str], np.ndarray]:
    values = {}
    for index, record in enumerate(bank.records):
        if layer not in record.get("activations", {}):
            continue
        key = (str(record.get("partition", "")), str(record.get("date", index)))
        activation = record["activations"][layer]
        if hasattr(activation, "detach"):
            activation = activation.detach().cpu().numpy()
        values[key] = np.asarray(activation, dtype=np.float64).reshape(-1)
    return values


def _aligned_representation_pair(
    reference_bank: ActivationBank,
    target_bank: ActivationBank,
    layer: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference = _activation_matrix_by_key(reference_bank, layer)
    target = _activation_matrix_by_key(target_bank, layer)
    common = sorted(set(reference) & set(target))
    if len(common) < 2:
        raise ValueError("cross-seed alignment requires at least two common dated observations")
    return np.vstack([reference[key] for key in common]), np.vstack([target[key] for key in common])


def _representation_tables(
    payloads: dict[str, dict],
    base_output_root: Path,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[tuple[dict, dict]]] = {}
    for payload in payloads.values():
        for concept, concept_payload in payload.get("concepts", {}).items():
            if concept_payload.get("status") != "completed":
                continue
            key = (payload["algorithm"], payload["market"], concept)
            grouped.setdefault(key, []).append((payload, concept_payload))
    bank_cache: dict[str, ActivationBank] = {}
    stability_rows = []
    similarity_rows = []
    for (algorithm, market, concept), items in grouped.items():
        for (reference, reference_concept), (target, target_concept) in combinations(
            sorted(items, key=lambda item: int(item[0]["seed"])), 2
        ):
            reference_layer = reference_concept["selected_layer"]
            target_layer = target_concept["selected_layer"]
            base = {
                "algorithm": algorithm,
                "market": market,
                "concept": concept,
                "reference_seed": int(reference["seed"]),
                "target_seed": int(target["seed"]),
                "reference_layer": reference_layer,
                "target_layer": target_layer,
            }
            if reference_layer != target_layer:
                stability_rows.append({**base, "status": "skipped_layer_mismatch"})
                similarity_rows.append({**base, "status": "skipped_layer_mismatch"})
                continue
            try:
                for payload in (reference, target):
                    if payload["run_id"] not in bank_cache:
                        bank_path = (
                            base_output_root
                            / "runs"
                            / payload["algorithm"]
                            / payload["market"]
                            / f"seed_{int(payload['seed'])}"
                            / "activation"
                            / "oos.pt"
                        )
                        bank_cache[payload["run_id"]] = ActivationBank.load(bank_path)
                reference_values, target_values = _aligned_representation_pair(
                    bank_cache[reference["run_id"]],
                    bank_cache[target["run_id"]],
                    reference_layer,
                )
                alignment = hungarian_neuron_alignment(reference_values, target_values)
                reference_indices = reference_concept["final_circuit"]["indices"]
                target_indices = target_concept["final_circuit"]["indices"]
                stability_rows.append(
                    {
                        **base,
                        "status": "completed",
                        "raw_jaccard": _jaccard(reference_indices, target_indices),
                        "aligned_jaccard": aligned_jaccard(
                            reference_indices,
                            target_indices,
                            alignment["target_to_reference"],
                        ),
                        "mean_absolute_correlation": alignment["mean_absolute_correlation"],
                        "min_absolute_correlation": alignment["min_absolute_correlation"],
                        "common_observations": int(len(reference_values)),
                    }
                )
                similarity_rows.append(
                    {
                        **base,
                        "status": "completed",
                        "linear_cka": linear_cka(reference_values, target_values),
                        "procrustes_residual": orthogonal_procrustes_residual(
                            reference_values, target_values
                        ),
                        "common_observations": int(len(reference_values)),
                    }
                )
            except Exception as exc:
                stability_rows.append({**base, "status": "failed", "error": str(exc)})
                similarity_rows.append({**base, "status": "failed", "error": str(exc)})
    return stability_rows, similarity_rows


def _cluster_bootstrap_rows(
    frame: pd.DataFrame,
    repeats: int,
    random_state: int = 1,
) -> list[dict]:
    if frame.empty:
        return [{"method": "market_cluster_bootstrap", "term": "insufficient_data"}]
    markets = sorted(frame["market"].dropna().unique().tolist())
    if not markets:
        return [{"method": "market_cluster_bootstrap", "term": "insufficient_data"}]
    groups = [("overall", None)]
    for column in ("algorithm", "concept", "policy_label"):
        groups.extend((f"{column}={value}", (column, value)) for value in sorted(frame[column].dropna().unique()))
    rng = np.random.default_rng(random_state)
    samples = {term: [] for term, _selector in groups}
    for _repeat in range(int(repeats)):
        selected_markets = rng.choice(markets, size=len(markets), replace=True)
        sampled = pd.concat(
            [frame.loc[frame["market"] == market] for market in selected_markets],
            ignore_index=True,
        )
        for term, selector in groups:
            selected = sampled if selector is None else sampled.loc[sampled[selector[0]] == selector[1]]
            if not selected.empty:
                samples[term].append(float(selected["effect"].mean()))
    rows = []
    for term, values in samples.items():
        if not values:
            continue
        rows.append(
            {
                "method": "market_cluster_bootstrap",
                "term": term,
                "estimate": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "p_value": float(
                    2.0 * min(np.mean(np.asarray(values) <= 0), np.mean(np.asarray(values) >= 0))
                ),
                "markets": len(markets),
                "observations": len(frame),
            }
        )
    return rows


def _mixedlm_rows(frame: pd.DataFrame) -> list[dict]:
    required_counts = (
        len(frame) >= 12
        and frame["market"].nunique() >= 3
        and frame["algorithm"].nunique() >= 2
    )
    if not required_counts:
        return [
            {
                "method": "mixedlm_market_random_intercept",
                "term": "insufficient_data",
                "status": "not_fitted",
                "observations": len(frame),
                "markets": int(frame["market"].nunique()) if not frame.empty else 0,
            }
        ]
    try:
        worker = Path(__file__).resolve().parents[1] / "evaluation" / "hierarchical_statistics.py"
        with tempfile.TemporaryDirectory(prefix="circuit_mixedlm_") as directory:
            input_path = Path(directory) / "effects.csv"
            output_path = Path(directory) / "mixedlm.json"
            frame.to_csv(input_path, index=False)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0 or not output_path.is_file():
                raise RuntimeError(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"MixedLM worker exited with code {completed.returncode}"
                )
            rows = _read_json(output_path)
        if not isinstance(rows, list) or not rows:
            raise ValueError("MixedLM worker returned no result rows")
        return rows
    except Exception as exc:
        return [
            {
                "method": "mixedlm_market_random_intercept",
                "term": "fit_failed",
                "status": "failed",
                "error": str(exc),
                "observations": len(frame),
                "markets": int(frame["market"].nunique()),
            }
        ]


def _hierarchical_rows(effect_rows: list[dict], repeats: int) -> list[dict]:
    frame = pd.DataFrame(effect_rows)
    rows = _mixedlm_rows(frame) + _cluster_bootstrap_rows(frame, repeats=repeats)
    finite = [index for index, row in enumerate(rows) if np.isfinite(row.get("p_value", np.nan))]
    if finite:
        adjusted = benjamini_hochberg([rows[index]["p_value"] for index in finite])
        for index, q_value in zip(finite, adjusted):
            rows[index]["fdr_q"] = float(q_value)
    return rows


def _policy_stratification(effect_rows: list[dict], economic_rows: list[dict]) -> list[dict]:
    effect_frame = pd.DataFrame(effect_rows)
    if effect_frame.empty:
        return []
    economic_frame = pd.DataFrame(economic_rows)
    default_economic = pd.DataFrame()
    if not economic_frame.empty:
        default_economic = economic_frame.loc[np.isclose(economic_frame["transition_cost"], 0.003)]
    rows = []
    columns = ["algorithm", "concept", "policy_label"]
    for keys, selected in effect_frame.groupby(columns, dropna=False):
        row = dict(zip(columns, keys))
        row.update(
            {
                "runs": int(len(selected)),
                "mean_causal_effect": float(selected["effect"].mean()),
                "std_causal_effect": float(selected["effect"].std(ddof=1)) if len(selected) > 1 else 0.0,
            }
        )
        if not default_economic.empty:
            matching = default_economic
            for column, value in row.items():
                if column in columns:
                    matching = matching.loc[matching[column] == value]
            row["mean_delta_pnl_at_cost_0_003"] = (
                float(matching["delta_pnl"].mean()) if not matching.empty else np.nan
            )
        rows.append(row)
    return rows


def aggregate_supplement_results(
    output_root: str | Path,
    base_output_root: str | Path,
    expected_run_ids: list[str],
    cluster_bootstrap_repeats: int = 2000,
) -> dict[str, int]:
    root = Path(output_root)
    base_root = Path(base_output_root)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    statuses = {}
    payloads = {}
    for status_path in root.glob("runs/*/*/seed_*/status.json"):
        status = _read_json(status_path)
        run_id = status.get("run_id")
        supplement_path = status_path.parent / "supplement.json"
        if status.get("status") == "completed" and supplement_path.is_file():
            payloads[run_id] = _read_json(supplement_path)
        elif status.get("status") == "completed":
            status = {**status, "status": "invalid", "error": "missing supplement.json"}
        statuses[run_id] = status
    status_rows = []
    for run_id in expected_run_ids:
        status = statuses.get(run_id, {"run_id": run_id, "status": "missing", "error": "missing status"})
        status_rows.append(
            {"run_id": run_id, "status": status.get("status"), "error": status.get("error")}
        )
    _write_table(aggregate_dir / "run_status.csv", status_rows, ["run_id", "status", "error"])

    sensitivity_rows = []
    intervention_rows = []
    attribution_rows = []
    bootstrap_rows = []
    sae_rows = []
    sae_intervention_rows = []
    specificity_rows = []
    economic_rows = []
    economic_bootstrap_rows = []
    hierarchical_effect_rows = []
    for payload in payloads.values():
        for layer, layer_payload in payload.get("layers", {}).items():
            sae = layer_payload.get("sae", {})
            for candidate in sae.get("candidates", []):
                sae_rows.append({**_identity(payload), "layer": layer, **candidate})
            for concept, feature in sae.get("feature_interventions", {}).items():
                sae_intervention_rows.append(
                    {**_identity(payload, concept), "layer": layer, **feature}
                )
        for concept, concept_payload in payload.get("concepts", {}).items():
            if concept_payload.get("status") != "completed":
                continue
            identity = {
                **_identity(payload, concept),
                "family": concept_payload.get("family"),
                "selected_layer": concept_payload.get("selected_layer"),
            }
            for row in concept_payload.get("circuit_sensitivity", []):
                sensitivity_rows.append(
                    {
                        **identity,
                        "max_size": row.get("max_size"),
                        "min_relative_gain": row.get("min_relative_gain"),
                        "actual_size": len(row.get("indices", [])),
                        "validation_effect": row.get("validation_effect"),
                        **_effect_fields(row.get("test_effect", {}), "test_"),
                        "random_control_mean": row.get("random_control", {}).get("mean"),
                        "random_control_empirical_p": row.get("random_control", {}).get("empirical_p"),
                    }
                )
            for mode, row in concept_payload.get("intervention_robustness", {}).items():
                output = {**identity, "mode": mode, **row}
                intervention_rows.append(output)
                if mode == "discovery_mean" and row.get("mean_absolute_margin_change") is not None:
                    hierarchical_effect_rows.append(
                        {**identity, "effect": float(row["mean_absolute_margin_change"])}
                    )
            for row in concept_payload.get("attribution_baselines", []):
                attribution_rows.append(
                    {
                        **identity,
                        "method": row.get("method"),
                        "requested_size": row.get("requested_size"),
                        "actual_size": len(row.get("indices", [])),
                        **_effect_fields(row.get("effect", {})),
                        "random_control_mean": row.get("random_control_mean"),
                        "empirical_p": row.get("empirical_p"),
                    }
                )
            stability = concept_payload.get("bootstrap_stability", {})
            for row in stability.get("rows", []):
                bootstrap_rows.append(
                    {
                        **identity,
                        "repeat": row.get("repeat"),
                        "status": row.get("status"),
                        "circuit_size": len(row.get("indices", [])),
                        "jaccard": row.get("jaccard"),
                        "validation_effect": row.get("validation_effect"),
                    }
                )
            specificity = concept_payload.get("concept_specificity", {})
            for row in specificity.get("rows", []):
                specificity_rows.append(
                    {
                        **identity,
                        **row,
                        "target_effect": specificity.get("target_effect"),
                        "non_target_mean": specificity.get("non_target_mean"),
                        "specificity_gap": specificity.get("specificity_gap"),
                    }
                )
            for row in concept_payload.get("economic_robustness", []):
                economic = {
                    **identity,
                    "transition_cost": row.get("transition_cost"),
                }
                for source in ("baseline", "intervention", "delta"):
                    for metric in ECONOMIC_METRICS:
                        economic[f"{source}_{metric}"] = row.get(source, {}).get(metric)
                economic_rows.append(economic)
                for metric, interval in row.get("block_bootstrap", {}).items():
                    economic_bootstrap_rows.append(
                        {
                            **identity,
                            "transition_cost": row.get("transition_cost"),
                            "metric": metric,
                            **interval,
                        }
                    )

    aligned_rows, similarity_rows = _representation_tables(payloads, base_root)
    policy_rows = _policy_stratification(hierarchical_effect_rows, economic_rows)
    hierarchical_rows = _hierarchical_rows(
        hierarchical_effect_rows,
        repeats=int(cluster_bootstrap_repeats),
    )

    identity_columns = ["run_id", "algorithm", "market", "seed", "concept", "policy_label"]
    _write_table(aggregate_dir / "circuit_size_sensitivity.csv", sensitivity_rows, identity_columns)
    _write_table(aggregate_dir / "intervention_robustness.csv", intervention_rows, identity_columns)
    _write_table(aggregate_dir / "attribution_baselines.csv", attribution_rows, identity_columns)
    _write_table(aggregate_dir / "bootstrap_stability.csv", bootstrap_rows, identity_columns)
    _write_table(
        aggregate_dir / "aligned_seed_stability.csv",
        aligned_rows,
        ["algorithm", "market", "concept", "reference_seed", "target_seed", "status"],
    )
    _write_table(
        aggregate_dir / "representation_similarity.csv",
        similarity_rows,
        ["algorithm", "market", "concept", "reference_seed", "target_seed", "status"],
    )
    _write_table(aggregate_dir / "sae_sparsity_pareto.csv", sae_rows, ["run_id", "algorithm", "market", "seed", "layer"])
    _write_table(aggregate_dir / "sae_feature_interventions.csv", sae_intervention_rows, identity_columns)
    _write_table(aggregate_dir / "concept_specificity.csv", specificity_rows, identity_columns)
    _write_table(
        aggregate_dir / "policy_stratification.csv",
        policy_rows,
        ["algorithm", "concept", "policy_label", "runs", "mean_causal_effect"],
    )
    _write_table(aggregate_dir / "economic_cost_sensitivity.csv", economic_rows, identity_columns)
    _write_table(aggregate_dir / "economic_block_bootstrap.csv", economic_bootstrap_rows, identity_columns)
    _write_table(
        aggregate_dir / "hierarchical_effects.csv",
        hierarchical_rows,
        ["method", "term", "estimate", "std_error", "ci_low", "ci_high", "p_value", "fdr_q", "status"],
    )

    completed = sum(row["status"] == "completed" for row in status_rows)
    missing = sum(row["status"] == "missing" for row in status_rows)
    failed = len(status_rows) - completed - missing
    report = "\n".join(
        [
            "# Supplementary validity experiment report",
            "",
            f"- Expected runs: {len(status_rows)}",
            f"- Completed runs: {completed}",
            f"- Failed or invalid runs: {failed}",
            f"- Missing runs: {missing}",
            f"- Valid concept-level effects: {len(hierarchical_effect_rows)}",
            f"- Cross-seed aligned comparisons: {sum(row.get('status') == 'completed' for row in aligned_rows)}",
            "",
            "All selection decisions use discovery/validation data; test-period rows are reporting-only.",
        ]
    )
    (aggregate_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    return {
        "expected": len(status_rows),
        "completed": completed,
        "failed": failed,
        "missing": missing,
    }


def _manifest_run_ids(path: str | Path) -> list[str]:
    payload = _read_json(Path(path))
    return [row["run_id"] for row in payload]


def supplement_summary_succeeded(summary: dict[str, int]) -> bool:
    return (
        int(summary.get("expected", -1)) > 0
        and int(summary.get("completed", -1)) == int(summary.get("expected", -1))
        and int(summary.get("failed", -1)) == 0
        and int(summary.get("missing", -1)) == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cluster-bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    summary = aggregate_supplement_results(
        args.output_root,
        base_output_root=args.base_output_root,
        expected_run_ids=_manifest_run_ids(args.manifest),
        cluster_bootstrap_repeats=args.cluster_bootstrap_repeats,
    )
    print(json.dumps(summary))
    return 0 if supplement_summary_succeeded(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
