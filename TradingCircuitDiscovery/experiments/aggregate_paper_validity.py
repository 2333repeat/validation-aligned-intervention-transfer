from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from TradingCircuitDiscovery.evaluation.economic_decomposition import (
    decompose_economic_channels,
)
from TradingCircuitDiscovery.evaluation.date_block_bootstrap import (
    moving_block_bootstrap_summary,
)
from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
    IDENTITY_COLUMNS,
    complete_cell_transfer_frame,
    dependence_bootstrap_summary,
    hierarchical_transfer_estimate,
)
from TradingCircuitDiscovery.evaluation.statistics import benjamini_hochberg


EFFECT_METRICS = (
    "mean_absolute_margin_change",
    "action_flip_rate",
    "mean_jsd",
    "mean_signed_margin_change",
)
PROFILE_METRICS = (
    "signed_margin_correlation_to_native",
    "signed_margin_sign_agreement_to_native",
    "conditional_transition_agreement_to_native",
    "transition_jsd_to_native",
    "intervened_distribution_jsd_to_native",
    "probability_shift_cosine_to_native",
)
LOWER_IS_BETTER_PROFILE_METRICS = {
    "transition_jsd_to_native",
    "intervened_distribution_jsd_to_native",
}
ECONOMIC_METRICS = ("pnl", "sharpe", "mdd", "turnover")
MARKET_GROUPS = {
    "us_equities": {"AAPL", "AMD", "AMZN", "GE", "GOOGL", "INTC", "MSFT", "NVDA"},
    "us_indices": {"DJI", "IXIC", "SP500"},
    "international_indices": {"FCHI", "HSI", "KS11", "N225"},
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    ordered = columns + [column for column in frame.columns if column not in columns]
    frame.loc[:, ordered].to_csv(path, index=False)


def _supplement_payloads(root: str | Path) -> dict[str, dict]:
    payloads = {}
    for path in Path(root).glob("runs/*/*/seed_*/supplement.json"):
        payload = _read_json(path)
        payloads[payload["run_id"]] = payload
    return payloads


def _supplement_tables(payloads: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    effect_rows = []
    economic_rows = []
    for payload in payloads.values():
        validation = payload.get("validation_policy", {})
        validation_metrics = validation.get("metrics", {})
        buy_hold = validation.get("buy_hold", {})
        excess_pnl = validation.get("excess_pnl")
        if excess_pnl is None:
            excess_pnl = validation_metrics.get("pnl", np.nan) - buy_hold.get(
                "net_pnl", np.nan
            )
        excess_sharpe = validation_metrics.get("sharpe", np.nan) - buy_hold.get(
            "sharpe", np.nan
        )
        for concept, concept_payload in payload.get("concepts", {}).items():
            if concept_payload.get("status") != "completed":
                continue
            identity = {
                "run_id": payload["run_id"],
                "algorithm": payload["algorithm"],
                "market": payload["market"],
                "seed": int(payload["seed"]),
                "concept": concept,
                "policy_label": validation.get("label"),
                "family": concept_payload.get("family"),
                "selected_layer": concept_payload.get("selected_layer"),
            }
            discovery_mean = concept_payload.get("intervention_robustness", {}).get(
                "discovery_mean", {}
            )
            effect = discovery_mean.get("mean_absolute_margin_change")
            if effect is not None:
                effect_rows.append(
                    {
                        **identity,
                        "effect": float(effect),
                        "validation_excess_pnl": float(excess_pnl),
                        "validation_excess_sharpe": float(excess_sharpe),
                        "validation_policy_pnl": validation_metrics.get("pnl"),
                        "validation_policy_sharpe": validation_metrics.get("sharpe"),
                        "validation_buy_hold_pnl": buy_hold.get("net_pnl"),
                        "validation_buy_hold_sharpe": buy_hold.get("sharpe"),
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
    return effect_rows, economic_rows


def _flatten_transfer_row(row: dict) -> dict:
    nested = {
        "target_native_effect",
        "aligned_transfer_effect",
        "raw_index_effect",
        "random_control",
        "alignment_null",
        "observation_effects",
        "mapping_stability",
        "aligned_profile_to_native",
        "raw_profile_to_native",
    }
    output = {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in row.items()
        if key not in nested and key != "traceback"
    }
    if row.get("status") != "completed":
        return output
    for prefix, source in (
        ("native", row.get("target_native_effect", {})),
        ("aligned", row.get("aligned_transfer_effect", {})),
        ("raw", row.get("raw_index_effect", {})),
    ):
        for metric in EFFECT_METRICS:
            output[f"{prefix}_{metric}"] = source.get(metric)
    for prefix, source in (
        ("aligned_profile", row.get("aligned_profile_to_native", {})),
        ("raw_profile", row.get("raw_profile_to_native", {})),
    ):
        for metric in PROFILE_METRICS:
            output[f"{prefix}_{metric}"] = source.get(metric)
    random_control = row.get("random_control", {})
    random_mean_effect = random_control.get("mean_effect", {})
    random_mean_profile = random_control.get("mean_profile_to_native", {})
    output.update(
        {
            "random_control_repeats": random_control.get("repeats"),
            "random_control_mean_effect": random_control.get("mean"),
            "random_control_std_effect": random_control.get("std"),
            "aligned_random_empirical_p": random_control.get("empirical_p"),
            "random_control_effects": json.dumps(random_control.get("effects", [])),
            "random_control_indices": json.dumps(random_control.get("indices", [])),
        }
    )
    for metric in EFFECT_METRICS:
        value = random_mean_effect.get(metric)
        if value is None and metric == "mean_absolute_margin_change":
            value = random_control.get("mean")
        output[f"random_{metric}"] = value
    for metric in PROFILE_METRICS:
        output[f"random_profile_{metric}"] = random_mean_profile.get(metric)
    alignment_null = row.get("alignment_null") or {}
    alignment_null_mean_effect = alignment_null.get("mean_effect", {})
    alignment_null_mean_profile = alignment_null.get("mean_profile_to_native", {})
    output.update(
        {
            "alignment_null_type": alignment_null.get("null_type"),
            "alignment_null_repeats": alignment_null.get("repeats"),
            "alignment_null_mean_effect": alignment_null.get("mean"),
            "alignment_null_std_effect": alignment_null.get("std"),
            "alignment_null_empirical_p": alignment_null.get("empirical_p"),
            "alignment_null_effects": json.dumps(alignment_null.get("effects", [])),
        }
    )
    for metric in EFFECT_METRICS:
        value = alignment_null_mean_effect.get(metric)
        if value is None and metric == "mean_absolute_margin_change":
            value = alignment_null.get("mean")
        output[f"alignment_null_{metric}"] = value
    for metric in PROFILE_METRICS:
        output[f"alignment_null_profile_{metric}"] = (
            alignment_null_mean_profile.get(metric)
        )
    mapping_stability = row.get("mapping_stability") or {}
    output.update(
        {
            "mapping_stability_blocks": mapping_stability.get("blocks"),
            "mapping_exact_match_rate": mapping_stability.get("exact_match_rate"),
            "mapping_mean_jaccard": mapping_stability.get("mean_jaccard"),
        }
    )
    aligned = output.get("aligned_mean_absolute_margin_change")
    raw = output.get("raw_mean_absolute_margin_change")
    native = output.get("native_mean_absolute_margin_change")
    random_mean = output.get("random_control_mean_effect")
    output["aligned_minus_raw_effect"] = (
        float(aligned - raw) if aligned is not None and raw is not None else np.nan
    )
    output["aligned_minus_random_effect"] = (
        float(aligned - random_mean)
        if aligned is not None and random_mean is not None
        else np.nan
    )
    output["aligned_minus_native_effect"] = (
        float(aligned - native)
        if aligned is not None and native is not None
        else np.nan
    )
    alignment_null_mean = output.get("alignment_null_mean_effect")
    output["aligned_minus_alignment_null_effect"] = (
        float(aligned - alignment_null_mean)
        if aligned is not None and alignment_null_mean is not None
        else np.nan
    )
    for metric in EFFECT_METRICS:
        aligned_value = output.get(f"aligned_{metric}")
        raw_value = output.get(f"raw_{metric}")
        random_value = output.get(f"random_{metric}")
        alignment_null_value = output.get(f"alignment_null_{metric}")
        output[f"aligned_minus_raw_{metric}"] = (
            float(aligned_value - raw_value)
            if aligned_value is not None and raw_value is not None
            else np.nan
        )
        output[f"aligned_minus_random_{metric}"] = (
            float(aligned_value - random_value)
            if aligned_value is not None and random_value is not None
            else np.nan
        )
        output[f"aligned_minus_alignment_null_{metric}"] = (
            float(aligned_value - alignment_null_value)
            if aligned_value is not None and alignment_null_value is not None
            else np.nan
        )
    for metric in PROFILE_METRICS:
        aligned_value = output.get(f"aligned_profile_{metric}")
        raw_value = output.get(f"raw_profile_{metric}")
        random_value = output.get(f"random_profile_{metric}")
        alignment_null_value = output.get(f"alignment_null_profile_{metric}")
        direction = -1.0 if metric in LOWER_IS_BETTER_PROFILE_METRICS else 1.0
        output[f"aligned_profile_improvement_over_raw_{metric}"] = (
            float(direction * (aligned_value - raw_value))
            if aligned_value is not None and raw_value is not None
            else np.nan
        )
        output[f"aligned_profile_improvement_over_random_{metric}"] = (
            float(direction * (aligned_value - random_value))
            if aligned_value is not None and random_value is not None
            else np.nan
        )
        output[f"aligned_profile_improvement_over_alignment_null_{metric}"] = (
            float(direction * (aligned_value - alignment_null_value))
            if aligned_value is not None and alignment_null_value is not None
            else np.nan
        )
    return output


def _observation_transfer_rows(payload: dict) -> list[dict]:
    dates = [str(value) for value in payload.get("test_dates", [])]
    rows = []
    for row in payload.get("rows", []):
        if row.get("status") != "completed":
            continue
        if row.get("alignment_method", "hungarian_abs") != "hungarian_abs":
            continue
        effects = row.get("observation_effects") or {}
        aligned = effects.get("aligned", {})
        raw = effects.get("raw", {})
        native = effects.get("native", {})
        aligned_absolute = aligned.get("absolute_margin_change", [])
        raw_absolute = raw.get("absolute_margin_change", [])
        native_absolute = native.get("absolute_margin_change", [])
        lengths = {
            len(dates),
            len(aligned_absolute),
            len(raw_absolute),
            len(native_absolute),
        }
        if lengths == {0}:
            continue
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError(
                f"observation effects do not match test dates for {row.get('target_run_id')}"
            )
        identity = {
            column: row.get(column)
            for column in IDENTITY_COLUMNS
        }
        for index, date in enumerate(dates):
            rows.append(
                {
                    **identity,
                    "date": date,
                    "aligned_absolute_margin_change": float(aligned_absolute[index]),
                    "raw_absolute_margin_change": float(raw_absolute[index]),
                    "native_absolute_margin_change": float(native_absolute[index]),
                    "aligned_minus_raw_absolute_margin_change": float(
                        aligned_absolute[index] - raw_absolute[index]
                    ),
                }
            )
    return rows


def _transfer_tables(
    transfer_root: str | Path,
    expected_target_run_ids: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    root = Path(transfer_root)
    statuses = {}
    payloads = {}
    for status_path in root.glob("runs/*/*/seed_*/status.json"):
        status = _read_json(status_path)
        run_id = status.get("run_id")
        transfer_path = status_path.parent / "transfer.json"
        if status.get("status") == "completed" and transfer_path.is_file():
            payloads[run_id] = _read_json(transfer_path)
        elif status.get("status") == "completed":
            status = {**status, "status": "invalid", "error": "missing transfer.json"}
        statuses[run_id] = status
    status_rows = []
    for run_id in expected_target_run_ids:
        status = statuses.get(
            run_id,
            {"run_id": run_id, "status": "missing", "error": "missing status"},
        )
        status_rows.append(
            {
                "run_id": run_id,
                "status": status.get("status"),
                "error": status.get("error"),
            }
        )
    transfer_rows = []
    observation_rows = []
    for payload in payloads.values():
        transfer_rows.extend(_flatten_transfer_row(row) for row in payload.get("rows", []))
        observation_rows.extend(_observation_transfer_rows(payload))
    return status_rows, transfer_rows, observation_rows


def _date_block_sensitivity_rows(
    observation_rows: list[dict],
    repeats: int,
    random_state: int = 1,
    block_length: int = 20,
) -> list[dict]:
    frame = pd.DataFrame(observation_rows)
    metric = "aligned_minus_raw_absolute_margin_change"
    if frame.empty or metric not in frame:
        return []
    frame = frame.loc[frame["concept"].astype(str) != "sideway"].copy()
    groups = [("overall", "overall", frame)]
    groups.extend(
        ("algorithm", str(algorithm), selected)
        for algorithm, selected in frame.groupby("algorithm", sort=True)
    )
    rows = []
    for group_index, (group_type, group_value, selected) in enumerate(groups):
        daily = []
        for _date, date_frame in selected.groupby("date", sort=True):
            daily.append(
                hierarchical_transfer_estimate(
                    date_frame,
                    metric=metric,
                    excluded_concepts=(),
                )
            )
        if len(daily) < 2:
            continue
        summary = moving_block_bootstrap_summary(
            daily,
            repeats=int(repeats),
            block_length=int(block_length),
            random_state=int(random_state) + group_index * 1009,
        )
        rows.append(
            {
                "specification": "primary_no_sideway",
                "group_type": group_type,
                "group_value": group_value,
                "endpoint": metric,
                "bootstrap_method": "calendar_circular_moving_block",
                "observed_path_conditional": True,
                **summary,
            }
        )
    return rows


def _reliability_interval_rows(
    transfer_rows: list[dict],
    repeats: int,
    random_state: int = 1,
) -> list[dict]:
    frame = pd.DataFrame(transfer_rows)
    if frame.empty or "status" not in frame:
        return []
    selected = frame.loc[frame["status"] == "completed"].copy()
    if "alignment_method" in selected:
        selected = selected.loc[
            selected["alignment_method"].fillna("hungarian_abs") == "hungarian_abs"
        ].copy()
    selected = selected.loc[selected["concept"].astype(str) != "sideway"].copy()
    if selected.empty:
        return []
    selected["gain_over_raw"] = (
        pd.to_numeric(selected["aligned_minus_raw_effect"], errors="coerce") > 0
    ).astype(float)
    selected["gain_over_membership_control"] = (
        pd.to_numeric(selected["aligned_minus_random_effect"], errors="coerce")
        > 0
    ).astype(float)
    selected["gain_over_both_controls"] = (
        (selected["gain_over_raw"] > 0)
        & (selected["gain_over_membership_control"] > 0)
    ).astype(float)
    selected["membership_control_empirical_p_le_0_05"] = (
        pd.to_numeric(selected["aligned_random_empirical_p"], errors="coerce")
        <= 0.05
    ).astype(float)
    selected["alignment_null_empirical_p_le_0_05"] = (
        pd.to_numeric(selected["alignment_null_empirical_p"], errors="coerce")
        <= 0.05
    ).astype(float)
    selected["mapping_exact_match_rate"] = pd.to_numeric(
        selected["mapping_exact_match_rate"], errors="coerce"
    )
    selected["mapping_mean_jaccard"] = pd.to_numeric(
        selected["mapping_mean_jaccard"], errors="coerce"
    )
    endpoints = (
        "gain_over_raw",
        "gain_over_membership_control",
        "gain_over_both_controls",
        "membership_control_empirical_p_le_0_05",
        "alignment_null_empirical_p_le_0_05",
        "mapping_exact_match_rate",
        "mapping_mean_jaccard",
    )
    rows = []
    for endpoint_index, endpoint in enumerate(endpoints):
        task_frame = selected.loc[:, list(IDENTITY_COLUMNS) + [endpoint]].copy()
        summaries = dependence_bootstrap_summary(
            task_frame,
            metric=endpoint,
            repeats=int(repeats),
            random_state=int(random_state) + endpoint_index * 1009,
        )
        rows.extend(
            {
                "specification": "primary_no_sideway",
                "endpoint": endpoint,
                **summary,
            }
            for summary in summaries
        )
    return rows


def _market_group_leaveout_rows(
    transfer_rows: list[dict],
    repeats: int,
    random_state: int = 1,
) -> list[dict]:
    frame = pd.DataFrame(transfer_rows)
    if frame.empty or "status" not in frame:
        return []
    selected = frame.loc[frame["status"] == "completed"].copy()
    if "alignment_method" in selected:
        selected = selected.loc[
            selected["alignment_method"].fillna("hungarian_abs") == "hungarian_abs"
        ].copy()
    selected = selected.loc[selected["concept"].astype(str) != "sideway"].copy()
    endpoint = "aligned_minus_raw_effect"
    methods = ("market_first", "dyadic", "multiway_pigeonhole")
    rows = []
    for group_index, (group_name, markets) in enumerate(MARKET_GROUPS.items()):
        retained = selected.loc[~selected["market"].astype(str).isin(markets)].copy()
        if retained.empty:
            rows.extend(
                {
                    "specification": "primary_no_sideway",
                    "excluded_market_group": group_name,
                    "endpoint": endpoint,
                    "bootstrap_method": method,
                    "status": "no_markets_remaining",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "markets": 0,
                }
                for method in methods
            )
            continue
        task_frame = retained.loc[:, list(IDENTITY_COLUMNS) + [endpoint]].copy()
        summaries = dependence_bootstrap_summary(
            task_frame,
            metric=endpoint,
            repeats=int(repeats),
            random_state=int(random_state) + group_index * 1009,
        )
        rows.extend(
            {
                "specification": "primary_no_sideway",
                "excluded_market_group": group_name,
                "endpoint": endpoint,
                "status": "completed",
                **summary,
            }
            for summary in summaries
        )
    return rows


def _cluster_bootstrap_samples(
    selected: pd.DataFrame,
    metric: str,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cluster_totals = (
        selected.groupby("target_run_id", sort=True)[metric]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )
    cluster_count = len(cluster_totals)
    if cluster_count == 0:
        return np.asarray([], dtype=float)
    chosen = rng.choice(
        cluster_count,
        size=(int(repeats), cluster_count),
        replace=True,
    )
    sums = cluster_totals["sum"].to_numpy(dtype=float)
    counts = cluster_totals["count"].to_numpy(dtype=float)
    sampled_counts = counts[chosen].sum(axis=1)
    return sums[chosen].sum(axis=1) / sampled_counts


def _bootstrap_transfer_summary(
    transfer_rows: list[dict],
    repeats: int,
    random_state: int = 1,
) -> list[dict]:
    frame = pd.DataFrame(transfer_rows)
    if frame.empty or "status" not in frame:
        return []
    frame = frame.loc[frame["status"] == "completed"].copy()
    if "alignment_method" in frame and frame["alignment_method"].notna().any():
        frame = frame.loc[frame["alignment_method"] == "hungarian_abs"].copy()
    if frame.empty:
        return []
    group_specs = [("overall", "overall", frame)]
    for algorithm, selected in frame.groupby("algorithm"):
        group_specs.append(("algorithm", str(algorithm), selected))
    for concept, selected in frame.groupby("concept"):
        group_specs.append(("concept", str(concept), selected))
    for (algorithm, concept), selected in frame.groupby(["algorithm", "concept"]):
        group_specs.append(("algorithm_concept", f"{algorithm}:{concept}", selected))
    rng = np.random.default_rng(random_state)
    rows = []
    metrics = ("aligned_minus_raw_effect", "aligned_minus_random_effect")
    for group_type, group_value, selected in group_specs:
        clusters = sorted(selected["target_run_id"].dropna().unique().tolist())
        if not clusters:
            continue
        for metric in metrics:
            observed = float(selected[metric].mean())
            sample_array = _cluster_bootstrap_samples(
                selected,
                metric=metric,
                repeats=repeats,
                rng=rng,
            )
            rows.append(
                {
                    "method": "target_policy_cluster_bootstrap",
                    "group_type": group_type,
                    "group_value": group_value,
                    "metric": metric,
                    "estimate": observed,
                    "ci_low": float(np.quantile(sample_array, 0.025)),
                    "ci_high": float(np.quantile(sample_array, 0.975)),
                    "p_value": float(
                        2.0
                        * min(
                            np.mean(sample_array <= 0),
                            np.mean(sample_array >= 0),
                        )
                    ),
                    "target_policies": len(clusters),
                    "transfer_rows": len(selected),
                    "aligned_empirical_p_lt_0_05_rate": float(
                        np.mean(selected["aligned_random_empirical_p"] < 0.05)
                    ),
                    "mean_aligned_to_native_ratio": float(
                        selected["aligned_to_native_ratio"].dropna().mean()
                    )
                    if "aligned_to_native_ratio" in selected
                    and selected["aligned_to_native_ratio"].notna().any()
                    else np.nan,
                }
            )
    finite = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(row.get("p_value", np.nan))
    ]
    if finite:
        adjusted = benjamini_hochberg([rows[index]["p_value"] for index in finite])
        for index, q_value in zip(finite, adjusted):
            rows[index]["fdr_q"] = float(q_value)
    return rows


def _dependence_endpoint_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "aligned_minus_raw_effect",
        "aligned_minus_random_effect",
        "aligned_minus_raw_mean_signed_margin_change",
        "aligned_minus_random_mean_signed_margin_change",
        "aligned_minus_raw_action_flip_rate",
        "aligned_minus_random_action_flip_rate",
        "aligned_minus_raw_mean_jsd",
        "aligned_minus_random_mean_jsd",
        "aligned_minus_alignment_null_mean_absolute_margin_change",
        "aligned_minus_alignment_null_mean_signed_margin_change",
        "aligned_minus_alignment_null_action_flip_rate",
        "aligned_minus_alignment_null_mean_jsd",
    ]
    for metric in PROFILE_METRICS:
        candidates.extend(
            [
                f"aligned_profile_improvement_over_raw_{metric}",
                f"aligned_profile_improvement_over_random_{metric}",
                f"aligned_profile_improvement_over_alignment_null_{metric}",
            ]
        )
    return [
        column
        for column in candidates
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _paired_alignment_comparisons(
    frame: pd.DataFrame,
    reference_method: str = "hungarian_abs",
) -> tuple[pd.DataFrame, list[str]]:
    identity_candidates = (
        "algorithm",
        "market",
        "concept",
        "source_run_id",
        "target_run_id",
        "source_layer",
        "target_layer",
    )
    identity = [column for column in identity_candidates if column in frame]
    value_columns = [
        *(f"aligned_{metric}" for metric in EFFECT_METRICS),
        *(f"aligned_profile_{metric}" for metric in PROFILE_METRICS),
    ]
    value_columns = [
        column
        for column in value_columns
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if not identity or not value_columns:
        return pd.DataFrame(), []
    reference = frame.loc[frame["alignment_method"] == reference_method].copy()
    methods = frame.loc[frame["alignment_method"] != reference_method].copy()
    if reference.empty or methods.empty:
        return pd.DataFrame(), []
    reference = reference.loc[:, identity + value_columns].rename(
        columns={column: f"reference_{column}" for column in value_columns}
    )
    paired = methods.merge(reference, on=identity, how="inner", validate="many_to_one")
    endpoints = []
    for metric in EFFECT_METRICS:
        method_column = f"aligned_{metric}"
        reference_column = f"reference_{method_column}"
        if method_column not in paired or reference_column not in paired:
            continue
        endpoint = f"method_minus_hungarian_{metric}"
        paired[endpoint] = pd.to_numeric(
            paired[method_column], errors="coerce"
        ) - pd.to_numeric(paired[reference_column], errors="coerce")
        endpoints.append(endpoint)
    for metric in PROFILE_METRICS:
        method_column = f"aligned_profile_{metric}"
        reference_column = f"reference_{method_column}"
        if method_column not in paired or reference_column not in paired:
            continue
        endpoint = f"method_improvement_over_hungarian_{metric}"
        direction = -1.0 if metric in LOWER_IS_BETTER_PROFILE_METRICS else 1.0
        paired[endpoint] = direction * (
            pd.to_numeric(paired[method_column], errors="coerce")
            - pd.to_numeric(paired[reference_column], errors="coerce")
        )
        endpoints.append(endpoint)
    return paired, endpoints


def _execute_dependence_tasks(
    tasks: list[tuple[pd.DataFrame, str, int, tuple[str, ...], dict]],
    repeats: int,
    workers: int,
) -> list[dict]:
    if int(workers) <= 0:
        raise ValueError("bootstrap workers must be positive")

    if int(workers) == 1:
        results = map(
            _execute_dependence_task,
            ((*task, int(repeats)) for task in tasks),
        )
    else:
        executor = ProcessPoolExecutor(max_workers=int(workers))
        results = executor.map(
            _execute_dependence_task,
            ((*task, int(repeats)) for task in tasks),
            chunksize=1,
        )
    try:
        return [row for task_rows in results for row in task_rows]
    finally:
        if int(workers) != 1:
            executor.shutdown(wait=True)


def _execute_dependence_task(task) -> list[dict]:
    selected, endpoint, seed, excluded, identity, repeats = task
    summaries = dependence_bootstrap_summary(
        selected,
        metric=endpoint,
        repeats=int(repeats),
        random_state=int(seed),
        excluded_concepts=excluded,
    )
    return [{**identity, "endpoint": endpoint, **row} for row in summaries]


def _add_dependence_fdr(rows: list[dict]) -> None:
    families: dict[tuple, list[int]] = {}
    for index, row in enumerate(rows):
        family = (
            row.get("specification"),
            row.get("alignment_method"),
            row.get("reference_alignment_method") or "controls",
            row.get("group_type"),
            row.get("group_value"),
            row.get("bootstrap_method"),
        )
        families.setdefault(family, []).append(index)
    for indices in families.values():
        finite = [
            index
            for index in indices
            if np.isfinite(rows[index].get("p_value", np.nan))
        ]
        adjusted = benjamini_hochberg([rows[index]["p_value"] for index in finite])
        for index, q_value in zip(finite, adjusted):
            rows[index]["fdr_q"] = float(q_value)


def _robust_transfer_tables(
    transfer_rows: list[dict],
    repeats: int,
    random_state: int = 1,
    workers: int = 1,
) -> tuple[list[dict], list[dict], list[dict]]:
    frame = pd.DataFrame(transfer_rows)
    if frame.empty or "status" not in frame:
        return [], [], []
    frame = frame.loc[frame["status"] == "completed"].copy()
    if frame.empty:
        return [], [], []
    if "alignment_method" not in frame:
        frame["alignment_method"] = "hungarian_abs"
    frame["alignment_method"] = frame["alignment_method"].fillna("hungarian_abs")
    directional_rows = frame.to_dict(orient="records")
    specifications = (
        ("primary_no_sideway", ("sideway",), False),
        ("primary_no_sideway_complete_cells", ("sideway",), True),
        ("all_concepts", (), False),
    )
    endpoints = _dependence_endpoint_columns(frame)
    tasks = []
    for specification, excluded, complete_only in specifications:
        for alignment_method, method_frame in frame.groupby(
            "alignment_method", sort=True
        ):
            groups = [("overall", "overall", method_frame)]
            groups.extend(
                ("algorithm", str(algorithm), selected)
                for algorithm, selected in method_frame.groupby("algorithm", sort=True)
            )
            for group_type, group_value, selected in groups:
                for endpoint_index, endpoint in enumerate(endpoints):
                    task_frame = selected.loc[
                        :, list(IDENTITY_COLUMNS) + [endpoint]
                    ].copy()
                    if not pd.to_numeric(
                        task_frame[endpoint], errors="coerce"
                    ).notna().any():
                        continue
                    task_excluded = excluded
                    if complete_only:
                        task_frame = complete_cell_transfer_frame(
                            task_frame,
                            metric=endpoint,
                            excluded_concepts=excluded,
                        )
                        task_excluded = ()
                    tasks.append(
                        (
                            task_frame,
                            endpoint,
                            int(random_state)
                            + endpoint_index * 1009
                            + sum(ord(value) for value in str(alignment_method))
                            + sum(ord(value) for value in group_value),
                            task_excluded,
                            {
                                "specification": specification,
                                "alignment_method": str(alignment_method),
                                "group_type": group_type,
                                "group_value": group_value,
                            },
                        )
                    )
    paired_frame, paired_endpoints = _paired_alignment_comparisons(frame)
    for specification, excluded, complete_only in specifications:
        for alignment_method, method_frame in paired_frame.groupby(
            "alignment_method", sort=True
        ):
            groups = [("overall", "overall", method_frame)]
            groups.extend(
                ("algorithm", str(algorithm), selected)
                for algorithm, selected in method_frame.groupby("algorithm", sort=True)
            )
            for group_type, group_value, selected in groups:
                for endpoint_index, endpoint in enumerate(paired_endpoints):
                    task_frame = selected.loc[
                        :, list(IDENTITY_COLUMNS) + [endpoint]
                    ].copy()
                    task_excluded = excluded
                    if complete_only:
                        task_frame = complete_cell_transfer_frame(
                            task_frame,
                            metric=endpoint,
                            excluded_concepts=excluded,
                        )
                        task_excluded = ()
                    tasks.append(
                        (
                            task_frame,
                            endpoint,
                            int(random_state)
                            + 100_003
                            + endpoint_index * 1009
                            + sum(ord(value) for value in str(alignment_method))
                            + sum(ord(value) for value in group_value),
                            task_excluded,
                            {
                                "specification": specification,
                                "alignment_method": str(alignment_method),
                                "reference_alignment_method": "hungarian_abs",
                                "group_type": group_type,
                                "group_value": group_value,
                            },
                        )
                    )
    dependence_rows = _execute_dependence_tasks(tasks, repeats, workers)
    _add_dependence_fdr(dependence_rows)
    comparison_rows = [
        row.copy()
        for row in dependence_rows
        if row["specification"] == "primary_no_sideway"
        and row["group_type"] == "overall"
        and row["bootstrap_method"] == "multiway_pigeonhole"
    ]
    return directional_rows, dependence_rows, comparison_rows


def continuous_quality_model_rows(
    effect_rows: list[dict],
    quality_metrics: tuple[str, ...] = (
        "validation_excess_sharpe",
        "validation_excess_pnl",
    ),
    include_sensitivity: bool = True,
) -> list[dict]:
    frame = pd.DataFrame(effect_rows)
    specifications = [("primary_no_sideway", ("sideway",))]
    if include_sensitivity:
        specifications.append(("sensitivity_all_concepts", ()))
    worker = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "continuous_quality_statistics.py"
    )
    rows = []
    for quality_metric in quality_metrics:
        for model_name, excluded in specifications:
            try:
                with tempfile.TemporaryDirectory(prefix="circuit_quality_") as directory:
                    input_path = Path(directory) / "effects.csv"
                    output_path = Path(directory) / "model.json"
                    frame.to_csv(input_path, index=False)
                    command = [
                        sys.executable,
                        str(worker),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--quality-column",
                        quality_metric,
                        "--model-name",
                        model_name,
                    ]
                    for concept in excluded:
                        command.extend(["--exclude-concept", concept])
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if completed.returncode != 0 or not output_path.is_file():
                        raise RuntimeError(
                            completed.stderr.strip()
                            or completed.stdout.strip()
                            or f"continuous quality worker exited with {completed.returncode}"
                        )
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                if not isinstance(result, list) or not result:
                    raise ValueError("continuous quality worker returned no rows")
                rows.extend(result)
            except Exception as exc:
                rows.append(
                    {
                        "method": "continuous_quality_mixedlm",
                        "model_name": model_name,
                        "quality_metric": quality_metric,
                        "excluded_concepts": ",".join(excluded),
                        "term": "fit_failed",
                        "status": "failed",
                        "error": str(exc),
                        "observations": len(frame),
                    }
                )
    finite_indices = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(row.get("p_value", np.nan))
    ]
    if finite_indices:
        adjusted = benjamini_hochberg([rows[index]["p_value"] for index in finite_indices])
        for index, q_value in zip(finite_indices, adjusted):
            rows[index]["fdr_q"] = float(q_value)
    return rows


def aggregate_paper_validity(
    supplement_output_root: str | Path,
    transfer_output_root: str | Path,
    output_root: str | Path,
    expected_target_run_ids: list[str],
    bootstrap_repeats: int = 20_000,
    bootstrap_workers: int = 1,
    fit_quality_models: bool = True,
) -> dict[str, int]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    payloads = _supplement_payloads(supplement_output_root)
    effect_rows, economic_rows = _supplement_tables(payloads)
    model_rows = continuous_quality_model_rows(effect_rows) if fit_quality_models else []
    economic_frame = decompose_economic_channels(pd.DataFrame(economic_rows))
    status_rows, transfer_rows, observation_rows = _transfer_tables(
        transfer_output_root,
        expected_target_run_ids,
    )
    transfer_summary = _bootstrap_transfer_summary(
        transfer_rows,
        repeats=int(bootstrap_repeats),
    )
    directional_rows, dependence_rows, comparison_rows = _robust_transfer_tables(
        transfer_rows,
        repeats=int(bootstrap_repeats),
        workers=int(bootstrap_workers),
    )
    date_block_rows = _date_block_sensitivity_rows(
        observation_rows,
        repeats=int(bootstrap_repeats),
    )
    reliability_rows = _reliability_interval_rows(
        transfer_rows,
        repeats=int(bootstrap_repeats),
    )
    market_group_rows = _market_group_leaveout_rows(
        transfer_rows,
        repeats=int(bootstrap_repeats),
    )
    identity = ["run_id", "algorithm", "market", "seed", "concept"]
    _write_table(output / "continuous_policy_quality_data.csv", effect_rows, identity)
    _write_table(
        output / "continuous_policy_quality_models.csv",
        model_rows,
        [
            "method",
            "model_name",
            "quality_metric",
            "term",
            "estimate",
            "std_error",
            "ci_low",
            "ci_high",
            "p_value",
            "fdr_q",
            "status",
        ],
    )
    economic_frame.to_csv(output / "economic_channel_decomposition.csv", index=False)
    _write_table(
        output / "cross_seed_transfer.csv",
        transfer_rows,
        [
            "algorithm",
            "market",
            "concept",
            "source_run_id",
            "source_seed",
            "target_run_id",
            "target_seed",
            "status",
        ],
    )
    _write_table(
        output / "cross_seed_transfer_summary.csv",
        transfer_summary,
        [
            "method",
            "group_type",
            "group_value",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "fdr_q",
        ],
    )
    _write_table(
        output / "transfer_directional_metrics.csv",
        directional_rows,
        [
            "algorithm",
            "market",
            "concept",
            "source_run_id",
            "target_run_id",
            "alignment_method",
            "status",
        ],
    )
    _write_table(
        output / "transfer_dependence_sensitivity.csv",
        dependence_rows,
        [
            "specification",
            "alignment_method",
            "group_type",
            "group_value",
            "endpoint",
            "bootstrap_method",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "fdr_q",
        ],
    )
    _write_table(
        output / "alignment_method_comparison.csv",
        comparison_rows,
        [
            "specification",
            "alignment_method",
            "group_type",
            "group_value",
            "endpoint",
            "bootstrap_method",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "fdr_q",
        ],
    )
    _write_table(
        output / "transfer_date_block_sensitivity.csv",
        date_block_rows,
        [
            "specification",
            "group_type",
            "group_value",
            "endpoint",
            "bootstrap_method",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "observations",
            "block_length",
            "observed_path_conditional",
        ],
    )
    _write_table(
        output / "transfer_unit_reliability_intervals.csv",
        reliability_rows,
        [
            "specification",
            "endpoint",
            "bootstrap_method",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "repeats",
            "markets",
            "algorithms",
            "policies",
            "dyads",
            "transfer_rows",
        ],
    )
    _write_table(
        output / "market_group_leaveout_sensitivity.csv",
        market_group_rows,
        [
            "specification",
            "excluded_market_group",
            "endpoint",
            "bootstrap_method",
            "status",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "markets",
            "algorithms",
            "policies",
            "dyads",
            "transfer_rows",
        ],
    )
    _write_table(output / "run_status.csv", status_rows, ["run_id", "status", "error"])
    completed_targets = sum(row["status"] == "completed" for row in status_rows)
    missing_targets = sum(row["status"] == "missing" for row in status_rows)
    failed_targets = len(status_rows) - completed_targets - missing_targets
    completed_transfer_rows = sum(row.get("status") == "completed" for row in transfer_rows)
    failed_transfer_rows = sum(row.get("status") == "failed" for row in transfer_rows)
    skipped_transfer_rows = (
        len(transfer_rows) - completed_transfer_rows - failed_transfer_rows
    )
    summary = {
        "supplement_payloads": len(payloads),
        "expected_targets": len(status_rows),
        "completed_targets": completed_targets,
        "failed_targets": failed_targets,
        "missing_targets": missing_targets,
        "completed_transfer_rows": completed_transfer_rows,
        "skipped_transfer_rows": skipped_transfer_rows,
        "failed_transfer_rows": failed_transfer_rows,
    }
    report = "\n".join(
        [
            "# Paper validity extension report",
            "",
            f"- Supplement payloads: {summary['supplement_payloads']}",
            f"- Expected target policies: {summary['expected_targets']}",
            f"- Completed target policies: {summary['completed_targets']}",
            f"- Failed target policies: {summary['failed_targets']}",
            f"- Missing target policies: {summary['missing_targets']}",
            f"- Completed directed transfer rows: {summary['completed_transfer_rows']}",
            f"- Skipped transfer rows: {summary['skipped_transfer_rows']}",
            f"- Failed transfer rows: {summary['failed_transfer_rows']}",
            f"- Continuous quality observations: {len(effect_rows)}",
            f"- Economic decomposition rows: {len(economic_frame)}",
            f"- Directional transfer rows: {len(directional_rows)}",
            f"- Dependence-robust summaries: {len(dependence_rows)}",
            f"- Alignment comparison summaries: {len(comparison_rows)}",
            f"- Date-block sensitivity summaries: {len(date_block_rows)}",
            f"- Reliability interval summaries: {len(reliability_rows)}",
            f"- Market-group leave-out summaries: {len(market_group_rows)}",
            "",
            "Alignment is fitted on validation activations only; intervention effects use test states only.",
        ]
    )
    (output / "report.md").write_text(report + "\n", encoding="utf-8")
    return summary


def _manifest_run_ids(path: str | Path) -> list[str]:
    return [row["run_id"] for row in _read_json(Path(path))]


def aggregate_summary_succeeded(summary: dict[str, int]) -> bool:
    expected = int(summary.get("expected_targets", -1))
    return (
        expected > 0
        and int(summary.get("supplement_payloads", -1)) == expected
        and int(summary.get("completed_targets", -1)) == expected
        and int(summary.get("failed_targets", -1)) == 0
        and int(summary.get("missing_targets", -1)) == 0
        and int(summary.get("failed_transfer_rows", -1)) == 0
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-output-root", required=True)
    parser.add_argument("--transfer-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    parser.add_argument("--skip-quality-models", action="store_true")
    args = parser.parse_args()
    summary = aggregate_paper_validity(
        supplement_output_root=args.supplement_output_root,
        transfer_output_root=args.transfer_output_root,
        output_root=args.output_root,
        expected_target_run_ids=_manifest_run_ids(args.manifest),
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_workers=args.bootstrap_workers,
        fit_quality_models=not args.skip_quality_models,
    )
    print(json.dumps(summary))
    if not aggregate_summary_succeeded(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
