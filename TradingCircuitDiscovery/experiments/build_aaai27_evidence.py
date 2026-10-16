from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


import hashlib
import json
from pathlib import Path
from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
    hierarchical_transfer_estimate,
)
import argparse
ALGORITHM_ORDER = ("A2C", "PPO", "DDQN", "SAC")
EFFECT_COLUMNS = {
    "native_effect": "native_mean_absolute_margin_change",
    "raw_effect": "raw_mean_absolute_margin_change",
    "aligned_effect": "aligned_mean_absolute_margin_change",
    "random_effect": "random_mean_absolute_margin_change",
}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def summarize_primary_transfer(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "algorithm",
        "market",
        "concept",
        "status",
        "alignment_method",
        *EFFECT_COLUMNS.values(),
    }
    _require_columns(frame, required)
    primary = frame.loc[
        frame["status"].eq("completed")
        & frame["alignment_method"].eq("hungarian_abs")
        & frame["concept"].ne("sideway")
    ].copy()
    if primary.empty:
        raise ValueError("primary_no_sideway transfer subset is empty")

    rows = []
    present = set(primary["algorithm"].astype(str))
    algorithms = [name for name in ALGORITHM_ORDER if name in present]
    algorithms.extend(sorted(present.difference(algorithms)))
    for algorithm in [*algorithms, "All"]:
        group = primary if algorithm == "All" else primary.loc[primary["algorithm"].eq(algorithm)]
        row = {"algorithm": algorithm, "n_units": int(len(group))}
        for output_name, source_name in EFFECT_COLUMNS.items():
            row[output_name] = hierarchical_transfer_estimate(group, source_name)
        row["aligned_minus_raw"] = row["aligned_effect"] - row["raw_effect"]
        row["aligned_minus_random"] = row["aligned_effect"] - row["random_effect"]
        rows.append(row)
    return pd.DataFrame(rows)


def select_primary_dependence(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, {"specification", "alignment_method"})
    selected = frame.loc[
        frame["specification"].eq("primary_no_sideway")
        & frame["alignment_method"].eq("hungarian_abs")
    ].copy()
    if selected.empty:
        raise ValueError("formal primary dependence subset is empty")
    order = [
        column
        for column in ("group_type", "group_value", "endpoint", "bootstrap_method")
        if column in selected.columns
    ]
    if order:
        selected = selected.sort_values(order, kind="stable")
    return selected.reset_index(drop=True)


def validate_permutation_results(
    manifest: dict,
    summary: pd.DataFrame,
    *,
    expected_random_controls: int = 50,
) -> dict:
    required = {
        "requested_runs",
        "completed_runs",
        "failed_runs",
        "unit_rows",
        "expected_unit_rows",
        "trial_rows",
        "expected_trial_rows",
        "random_controls",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"manifest missing required fields: {', '.join(missing)}")
    checks = {
        "completed_runs": manifest["completed_runs"] == manifest["requested_runs"],
        "failed_runs": manifest["failed_runs"] == 0,
        "unit_rows": manifest["unit_rows"] == manifest["expected_unit_rows"],
        "trial_rows": manifest["trial_rows"] == manifest["expected_trial_rows"],
        "random_controls": manifest["random_controls"] == expected_random_controls,
    }
    for name, passed in checks.items():
        if not passed:
            raise ValueError(f"permutation audit failed: {name}")

    _require_columns(
        summary,
        {
            "algorithm",
            "n_runs",
            "n_units",
            "n_trials",
            "mapping_accuracy_min",
            "exact_mapping_fraction",
            "random_separation_fraction",
        },
    )
    overall = summary.loc[summary["algorithm"].eq("All")]
    if len(overall) != 1:
        raise ValueError("permutation summary must contain exactly one All row")
    row = overall.iloc[0]
    summary_checks = {
        "summary n_runs": int(row["n_runs"]) == int(manifest["completed_runs"]),
        "summary n_units": int(row["n_units"]) == int(manifest["unit_rows"]),
        "summary n_trials": int(row["n_trials"]) == int(manifest["trial_rows"]),
        "mapping_accuracy_min": np.isclose(float(row["mapping_accuracy_min"]), 1.0),
        "exact_mapping_fraction": np.isclose(float(row["exact_mapping_fraction"]), 1.0),
        "random_separation_fraction": np.isclose(float(row["random_separation_fraction"]), 1.0),
    }
    for name, passed in summary_checks.items():
        if not passed:
            raise ValueError(f"permutation audit failed: {name}")
    return {
        "completed_runs": int(manifest["completed_runs"]),
        "unit_rows": int(manifest["unit_rows"]),
        "trial_rows": int(manifest["trial_rows"]),
        "random_controls": int(manifest["random_controls"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_evidence_tables(
    *,
    transfer_path: str | Path,
    dependence_path: str | Path,
    permutation_root: str | Path,
    output_root: str | Path,
) -> dict:
    transfer_path = Path(transfer_path).expanduser().resolve()
    dependence_path = Path(dependence_path).expanduser().resolve()
    permutation_root = Path(permutation_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    permutation_manifest_path = permutation_root / "permutation_sanity_manifest.json"
    permutation_summary_path = permutation_root / "permutation_sanity_summary.csv"

    transfer = pd.read_csv(transfer_path)
    dependence = pd.read_csv(dependence_path)
    permutation_summary = pd.read_csv(permutation_summary_path)
    permutation_manifest = json.loads(
        permutation_manifest_path.read_text(encoding="utf-8")
    )
    primary_summary = summarize_primary_transfer(transfer)
    primary_dependence = select_primary_dependence(dependence)
    permutation_audit = validate_permutation_results(
        permutation_manifest,
        permutation_summary,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    primary_summary_path = output_root / "primary_transfer_summary.csv"
    primary_dependence_path = output_root / "primary_dependence_sensitivity.csv"
    sanitized_permutation_summary_path = output_root / "permutation_sanity_summary.csv"
    sanitized_permutation_audit_path = output_root / "permutation_sanity_audit.json"
    primary_summary.to_csv(primary_summary_path, index=False)
    primary_dependence.to_csv(primary_dependence_path, index=False)
    permutation_summary.to_csv(sanitized_permutation_summary_path, index=False)

    overall = permutation_summary.loc[permutation_summary["algorithm"].eq("All")].iloc[0]
    sanitized_audit = {
        **permutation_audit,
        "failed_runs": int(permutation_manifest["failed_runs"]),
        "expected_unit_rows": int(permutation_manifest["expected_unit_rows"]),
        "expected_trial_rows": int(permutation_manifest["expected_trial_rows"]),
        "checkpoint_manifest_sha256": permutation_manifest.get(
            "checkpoint_manifest_sha256"
        ),
        "source_archive_sha256": permutation_manifest.get("source_archive_sha256"),
        "source_sha256": permutation_manifest.get("source_sha256"),
        "mapping_accuracy_min": float(overall["mapping_accuracy_min"]),
        "exact_mapping_fraction": float(overall["exact_mapping_fraction"]),
        "random_separation_fraction": float(overall["random_separation_fraction"]),
    }
    for name in (
        "max_output_error_max",
        "mapped_intervened_output_error_max",
        "mapped_effect_error_max",
    ):
        if name in overall.index:
            sanitized_audit[name] = float(overall[name])
    _write_json(sanitized_permutation_audit_path, sanitized_audit)

    outputs = (
        primary_summary_path,
        primary_dependence_path,
        sanitized_permutation_summary_path,
        sanitized_permutation_audit_path,
    )
    evidence_manifest = {
        "schema_version": 1,
        "primary_protocol": {
            "status": "completed",
            "alignment_method": "hungarian_abs",
            "excluded_concepts": ["sideway"],
            "estimator": "equal-weight concepts within markets, markets within algorithms, then algorithms",
        },
        "primary_transfer_units": int(
            primary_summary.loc[primary_summary["algorithm"].eq("All"), "n_units"].iloc[0]
        ),
        "primary_dependence_rows": int(len(primary_dependence)),
        "permutation": permutation_audit,
        "sources": {
            "cross_seed_transfer": {
                "filename": transfer_path.name,
                "rows": int(len(transfer)),
                "sha256": _sha256(transfer_path),
            },
            "transfer_dependence_sensitivity": {
                "filename": dependence_path.name,
                "rows": int(len(dependence)),
                "sha256": _sha256(dependence_path),
            },
            "permutation_manifest": {
                "filename": permutation_manifest_path.name,
                "sha256": _sha256(permutation_manifest_path),
            },
            "permutation_summary": {
                "filename": permutation_summary_path.name,
                "rows": int(len(permutation_summary)),
                "sha256": _sha256(permutation_summary_path),
            },
        },
        "outputs": {
            path.name: {"sha256": _sha256(path)}
            for path in outputs
        },
    }
    _write_json(output_root / "evidence_manifest.json", evidence_manifest)
    return evidence_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build audited AAAI-27 evidence tables from frozen results."
    )
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--dependence", required=True)
    parser.add_argument("--permutation-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    evidence = build_evidence_tables(
        transfer_path=args.transfer,
        dependence_path=args.dependence,
        permutation_root=args.permutation_root,
        output_root=args.output_root,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
