from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from TradingCircuitDiscovery.collect.dataset import ActivationBank
from TradingCircuitDiscovery.experiments.permutation_sanity import (
    evaluate_known_permutation_controls,
    permute_hidden_units,
)
from TradingCircuitDiscovery.integrations.policy_value import (
    build_policy_value_adapter,
)
from TradingCircuitDiscovery.intervention.activation_patch import resolve_module


@dataclass(frozen=True)
class AlgorithmSanitySpec:
    algorithm: str
    network_attr: str
    layer_prefix: str
    linear_layers: tuple[str, ...]
    probability_scores: bool

    @property
    def hidden_layers(self) -> tuple[str, ...]:
        return self.linear_layers[:-1]

    @property
    def adapter_layers(self) -> tuple[str, ...]:
        return tuple(f"{self.layer_prefix}{name}" for name in self.hidden_layers)


_SPECS = {
    "PPO": AlgorithmSanitySpec(
        algorithm="PPO",
        network_attr="actor",
        layer_prefix="actor.",
        linear_layers=("fc1", "fc2"),
        probability_scores=True,
    ),
    "A2C": AlgorithmSanitySpec(
        algorithm="A2C",
        network_attr="actor",
        layer_prefix="actor.",
        linear_layers=("fc1", "fc2"),
        probability_scores=True,
    ),
    "DDQN": AlgorithmSanitySpec(
        algorithm="DDQN",
        network_attr="policy_net",
        layer_prefix="policy_net.",
        linear_layers=("fc1", "fc2", "fc3"),
        probability_scores=False,
    ),
    "SAC": AlgorithmSanitySpec(
        algorithm="SAC",
        network_attr="actor",
        layer_prefix="actor.",
        linear_layers=("fc1", "fc2"),
        probability_scores=False,
    ),
}


def algorithm_spec(algorithm: str) -> AlgorithmSanitySpec:
    normalized = str(algorithm).upper().replace("_", "-")
    aliases = {
        "DQN": "DDQN",
        "QR-DQN": "DDQN",
        "QRDQN": "DDQN",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return _SPECS[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported policy algorithm: {algorithm}") from error


@contextmanager
def strict_float32_matmul():
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)


def expected_unit_count_failure(
    observed_unit_rows: int,
    expected_unit_rows: int | None,
) -> dict | None:
    if expected_unit_rows is None or int(observed_unit_rows) == int(expected_unit_rows):
        return None
    return {
        "run_id": "__global__",
        "algorithm": None,
        "market": None,
        "seed": None,
        "error_type": "FullCoverageError",
        "error": (
            "full run unit coverage mismatch: "
            f"{int(observed_unit_rows)}/{int(expected_unit_rows)}"
        ),
    }
def partition_states(records: Sequence[Mapping]) -> dict[str, torch.Tensor]:
    partitions: dict[str, torch.Tensor] = {}
    for name in ("discovery", "validation", "test"):
        rows = [
            np.asarray(record["state"], dtype=np.float32).reshape(-1)
            for record in records
            if str(record.get("partition", "")) == name
        ]
        if not rows:
            raise ValueError(f"activation bank is missing the {name} partition")
        matrix = np.stack(rows)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} states contain non-finite values")
        partitions[name] = torch.from_numpy(matrix)
    widths = {int(values.shape[1]) for values in partitions.values()}
    if len(widths) != 1:
        raise ValueError("chronological state partitions have inconsistent widths")
    return partitions


def _linear_width(network: nn.Module, layer: str) -> int:
    module = resolve_module(network, layer)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{layer!r} must resolve to torch.nn.Linear")
    return int(module.out_features)


def _non_identity_permutation(
    width: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if width < 2:
        raise ValueError("known-permutation sanity requires hidden width >= 2")
    values = rng.permutation(width)
    if np.array_equal(values, np.arange(width)):
        values = np.roll(values, 1)
    return tuple(int(value) for value in values)


def build_permuted_target(
    reference: nn.Module,
    *,
    linear_layers: Sequence[str],
    permutation_seed: int,
) -> tuple[nn.Module, tuple[tuple[int, ...], ...]]:
    names = tuple(str(name) for name in linear_layers)
    rng = np.random.default_rng(int(permutation_seed))
    permutations = tuple(
        _non_identity_permutation(_linear_width(reference, name), rng)
        for name in names[:-1]
    )
    target = permute_hidden_units(
        reference,
        linear_layers=names,
        target_to_source=permutations,
    )
    return target, permutations


def _bare_layer(layer: str, prefix: str, hidden_layers: Sequence[str]) -> str:
    value = str(layer)
    if prefix and value.startswith(prefix):
        value = value[len(prefix) :]
    if value not in hidden_layers:
        allowed = ", ".join(f"{prefix}{name}" for name in hidden_layers)
        raise ValueError(f"circuit layer {layer!r} is not one of: {allowed}")
    return value


def validate_known_permutation_result(result: Mapping) -> dict[str, float | bool]:
    tolerance_by_dtype = {
        "torch.float16": (5e-3, 5e-3),
        "torch.bfloat16": (1e-2, 1e-2),
        "torch.float32": (1e-5, 1e-5),
        "torch.float64": (1e-10, 1e-8),
    }
    dtype_name = str(result.get("numeric_dtype", ""))
    if dtype_name not in tolerance_by_dtype:
        raise ValueError(
            f"unsupported numeric dtype for equivalence audit: {dtype_name}"
        )
    atol, rtol = tolerance_by_dtype[dtype_name]
    numeric_keys = (
        "mapping_accuracy",
        "output_scale",
        "mapped_intervened_output_scale",
        "max_output_error",
        "mapped_intervened_output_error",
        "mapped_effect_error",
        "random_mapping_accuracy",
        "random_intervened_output_error",
        "random_effect_error",
    )
    values = {key: float(result[key]) for key in numeric_keys}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("numeric equivalence audit contains non-finite values")
    if values["mapping_accuracy"] != 1.0:
        raise ValueError(
            f"mapping equivalence failed: accuracy={values['mapping_accuracy']:.9g}"
        )

    output_limit = atol + rtol * max(1.0, values["output_scale"])
    intervention_limit = atol + rtol * max(
        1.0,
        values["mapped_intervened_output_scale"],
    )
    effect_limit = 2.0 * intervention_limit
    if values["max_output_error"] > output_limit:
        raise ValueError(
            "function equivalence failed: "
            f"error={values['max_output_error']:.9g}, limit={output_limit:.9g}"
        )
    if values["mapped_intervened_output_error"] > intervention_limit:
        raise ValueError(
            "intervention equivalence failed: "
            f"error={values['mapped_intervened_output_error']:.9g}, "
            f"limit={intervention_limit:.9g}"
        )
    if values["mapped_effect_error"] > effect_limit:
        raise ValueError(
            "effect equivalence failed: "
            f"error={values['mapped_effect_error']:.9g}, limit={effect_limit:.9g}"
        )
    return {
        "numeric_valid": True,
        "numeric_atol": float(atol),
        "numeric_rtol": float(rtol),
        "output_equivalence_limit": float(output_limit),
        "intervention_equivalence_limit": float(intervention_limit),
        "effect_equivalence_limit": float(effect_limit),
    }


def evaluate_policy_circuits(
    *,
    reference: nn.Module,
    records: Sequence[Mapping],
    circuits: Mapping[str, Mapping],
    linear_layers: Sequence[str],
    layer_prefix: str,
    probability_scores: bool,
    permutation_seed: int,
    random_seeds: Sequence[int],
    metadata: Mapping | None = None,
) -> list[dict]:
    partitions = partition_states(records)
    names = tuple(str(name) for name in linear_layers)
    target, permutations = build_permuted_target(
        reference,
        linear_layers=names,
        permutation_seed=int(permutation_seed),
    )
    controls = tuple(int(seed) for seed in random_seeds)
    if not controls:
        raise ValueError("at least one random control is required")
    if not circuits:
        raise ValueError("circuit payload must contain at least one eligible concept")

    base = dict(metadata or {})
    trials: list[dict] = []
    for concept in sorted(circuits):
        payload = circuits[concept]
        if not isinstance(payload, Mapping):
            raise TypeError(f"circuit payload for {concept!r} must be a mapping")
        hidden_layer = _bare_layer(
            str(payload.get("layer", "")),
            str(layer_prefix),
            names[:-1],
        )
        source_indices = tuple(int(index) for index in payload.get("indices", ()))
        if len(set(source_indices)) != len(source_indices):
            raise ValueError(f"circuit {concept!r} contains duplicate indices")
        width = _linear_width(reference, hidden_layer)
        if not source_indices or len(source_indices) >= width:
            raise ValueError(
                f"circuit {concept!r} must be a non-empty strict subset of {hidden_layer}"
            )
        if min(source_indices) < 0 or max(source_indices) >= width:
            raise ValueError(f"circuit {concept!r} contains an out-of-range index")

        known_mapping = {
            target_index: source_index
            for target_index, source_index in enumerate(
                permutations[names[:-1].index(hidden_layer)]
            )
        }
        raw_results = evaluate_known_permutation_controls(
            reference,
            target,
            discovery_states=partitions["discovery"],
            validation_states=partitions["validation"],
            test_states=partitions["test"],
            hidden_layer=hidden_layer,
            source_indices=source_indices,
            probability_scores=bool(probability_scores),
            random_seeds=controls,
        )
        results = [
            {**result, **validate_known_permutation_result(result)}
            for result in raw_results
        ]
        if len(results) != len(controls):
            raise RuntimeError(
                "random-control result count does not match its seed count"
            )
        for control_index, (random_seed, result) in enumerate(
            zip(controls, results, strict=True)
        ):
            row = {
                **base,
                "concept": str(concept),
                "layer": f"{layer_prefix}{hidden_layer}",
                "hidden_width": width,
                "circuit_size": len(source_indices),
                "circuit_indices": json.dumps(source_indices, separators=(",", ":")),
                "known_target_to_source": json.dumps(
                    known_mapping,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "permutation_seed": int(permutation_seed),
                "random_control_index": int(control_index),
                "random_seed": int(random_seed),
                "n_discovery": int(partitions["discovery"].shape[0]),
                "n_validation": int(partitions["validation"].shape[0]),
                "n_test": int(partitions["test"].shape[0]),
                **result,
            }
            trials.append(row)
    expected_rows = len(circuits) * len(controls)
    if len(trials) != expected_rows:
        raise RuntimeError(
            f"coverage identity failed: {len(trials)} != {len(circuits)} x {len(controls)}"
        )
    return trials


def _unit_key(row: Mapping) -> tuple:
    return (
        row.get("run_id"),
        row.get("concept"),
        row.get("layer"),
    )


def summarize_units(trials: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple, list[Mapping]] = {}
    for row in trials:
        grouped.setdefault(_unit_key(row), []).append(row)

    output = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        rows = grouped[key]
        first = rows[0]
        output.append(
            {
                "run_id": first.get("run_id"),
                "algorithm": first.get("algorithm"),
                "market": first.get("market"),
                "seed": first.get("seed"),
                "concept": first.get("concept"),
                "layer": first.get("layer"),
                "hidden_width": first.get("hidden_width"),
                "circuit_size": first.get("circuit_size"),
                "n_random_controls": len(rows),
                "mapping_accuracy": float(first["mapping_accuracy"]),
                "max_output_error": float(first["max_output_error"]),
                "mapped_intervened_output_error": float(
                    first["mapped_intervened_output_error"]
                ),
                "source_effect": float(first["source_effect"]),
                "mapped_effect": float(first["mapped_effect"]),
                "mapped_effect_error": float(first["mapped_effect_error"]),
                "random_mapping_accuracy_mean": float(
                    np.mean([float(row["random_mapping_accuracy"]) for row in rows])
                ),
                "random_intervened_output_error_mean": float(
                    np.mean(
                        [float(row["random_intervened_output_error"]) for row in rows]
                    )
                ),
                "random_effect_mean": float(
                    np.mean([float(row["random_effect"]) for row in rows])
                ),
                "random_effect_error_mean": float(
                    np.mean([float(row["random_effect_error"]) for row in rows])
                ),
                "random_separation_fraction": float(
                    np.mean(
                        [
                            float(row["random_intervened_output_error"])
                            > float(row["mapped_intervened_output_error"]) + 1e-12
                            for row in rows
                        ]
                    )
                ),
            }
        )
    return output


def _summary_row(algorithm: str, rows: Sequence[Mapping]) -> dict:
    units = summarize_units(rows)
    return {
        "algorithm": algorithm,
        "n_runs": len({row.get("run_id") for row in rows}),
        "n_units": len(units),
        "n_trials": len(rows),
        "mapping_accuracy_mean": float(
            np.mean([float(row["mapping_accuracy"]) for row in units])
        ),
        "mapping_accuracy_min": float(
            np.min([float(row["mapping_accuracy"]) for row in units])
        ),
        "exact_mapping_fraction": float(
            np.mean([float(row["mapping_accuracy"]) == 1.0 for row in units])
        ),
        "max_output_error_max": float(
            np.max([float(row["max_output_error"]) for row in units])
        ),
        "mapped_intervened_output_error_max": float(
            np.max([float(row["mapped_intervened_output_error"]) for row in units])
        ),
        "mapped_effect_error_mean": float(
            np.mean([float(row["mapped_effect_error"]) for row in units])
        ),
        "mapped_effect_error_max": float(
            np.max([float(row["mapped_effect_error"]) for row in units])
        ),
        "random_intervened_output_error_mean": float(
            np.mean(
                [float(row["random_intervened_output_error_mean"]) for row in units]
            )
        ),
        "random_effect_error_mean": float(
            np.mean([float(row["random_effect_error_mean"]) for row in units])
        ),
        "random_separation_fraction": float(
            np.mean([float(row["random_separation_fraction"]) for row in units])
        ),
    }


def summarize_trials(trials: Sequence[Mapping]) -> list[dict]:
    if not trials:
        return []
    by_algorithm: dict[str, list[Mapping]] = {}
    for row in trials:
        by_algorithm.setdefault(str(row.get("algorithm", "Unknown")), []).append(row)
    summary = [
        _summary_row(algorithm, by_algorithm[algorithm])
        for algorithm in sorted(by_algorithm)
    ]
    if len(by_algorithm) > 1:
        summary.append(_summary_row("All", list(trials)))
    return summary


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_experiment_archive(
    archive_path: str | Path,
    extraction_root: str | Path,
) -> Path:
    import tarfile

    archive = Path(archive_path).resolve()
    destination = Path(extraction_root).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"experiment archive missing: {archive}")
    archive_hash = _sha256(archive)
    marker = destination / ".archive_manifest.json"
    if marker.exists():
        payload = _read_json(marker)
        experiment_root = destination / str(payload.get("experiment_root", ""))
        if (
            payload.get("archive_sha256") == archive_hash
            and experiment_root.is_dir()
            and (experiment_root / "runs").is_dir()
        ):
            return experiment_root
        raise ValueError(
            "existing archive extraction does not match the requested archive"
        )

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(
            f"archive extraction root must be empty or verified by a marker: {destination}"
        )
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"linked archive member is not allowed: {member.name}")
        handle.extractall(destination, members=members, filter="data")

    candidates = [
        path
        for path in destination.iterdir()
        if path.is_dir() and (path / "runs").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"archive must contain exactly one experiment root; found {len(candidates)}"
        )
    experiment_root = candidates[0]
    _write_json(
        marker,
        {
            "archive_path": str(archive),
            "archive_sha256": archive_hash,
            "experiment_root": experiment_root.name,
        },
    )
    return experiment_root


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join((str(int(base_seed)), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_checkpoint_manifest(path: str | Path) -> dict[str, dict]:
    manifest_path = Path(path)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("checkpoint hash manifest is empty")
    required = {"run_id", "algorithm", "market", "seed", "checkpoint_sha256"}
    if not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0]))
        raise ValueError(f"checkpoint hash manifest is missing columns: {missing}")
    indexed: dict[str, dict] = {}
    for row in rows:
        run_id = str(row["run_id"])
        if run_id in indexed:
            raise ValueError(f"duplicate checkpoint hash row: {run_id}")
        digest = str(row["checkpoint_sha256"]).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid checkpoint SHA-256 for {run_id}")
        indexed[run_id] = {
            **row,
            "seed": int(row["seed"]),
            "checkpoint_sha256": digest,
        }
    return indexed


def _write_json(path: str | Path, payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_directory(experiment_root: Path, run: Mapping) -> Path:
    return (
        experiment_root
        / "runs"
        / str(run["algorithm"])
        / str(run["market"])
        / f"seed_{int(run['seed'])}"
    )


def _resolve_checkpoint(
    run: Mapping,
    project_root: Path,
    checkpoint_root: Path | None,
) -> Path:
    run_id = str(run["run_id"])
    if checkpoint_root is not None:
        candidate = checkpoint_root / run_id / "agent" / "model" / "best_model.pth"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"checkpoint unavailable under explicit root for {run_id}: {candidate}"
        )

    recorded = Path(str(run["checkpoint_path"]))
    if recorded.exists():
        return recorded
    candidate = project_root / "result" / run_id / "agent" / "model" / "best_model.pth"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"checkpoint unavailable for {run_id}: {recorded} or {candidate}"
    )


def run_permutation_sanity_suite(
    *,
    manifest_path: str | Path,
    checkpoint_manifest_path: str | Path,
    project_root: str | Path,
    experiment_root: str | Path,
    output_dir: str | Path,
    checkpoint_root: str | Path | None = None,
    source_archive_path: str | Path | None = None,
    random_controls: int = 50,
    base_seed: int = 2701,
    algorithms: Sequence[str] | None = None,
    markets: Sequence[str] | None = None,
    limit: int | None = None,
    expected_unit_rows: int | None = None,
    fail_fast: bool = False,
    allow_failures: bool = False,
) -> dict:
    if int(random_controls) < 1:
        raise ValueError("random_controls must be positive")
    manifest_file = Path(manifest_path).resolve()
    checkpoint_manifest_file = Path(checkpoint_manifest_path).resolve()
    policy_value_root = Path(project_root).resolve()
    experiment_dir = Path(experiment_root).resolve()
    destination = Path(output_dir).resolve()
    checkpoint_dir = Path(checkpoint_root).resolve() if checkpoint_root else None
    source_archive = (
        Path(source_archive_path).resolve() if source_archive_path else None
    )
    destination.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(manifest_file)
    if not isinstance(manifest, list):
        raise TypeError("experiment manifest must contain a list of runs")
    checkpoint_manifest = _load_checkpoint_manifest(checkpoint_manifest_file)
    manifest_by_id = {str(run["run_id"]): run for run in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("experiment manifest contains duplicate run ids")
    if set(manifest_by_id) != set(checkpoint_manifest):
        missing_hashes = sorted(set(manifest_by_id).difference(checkpoint_manifest))
        extra_hashes = sorted(set(checkpoint_manifest).difference(manifest_by_id))
        raise ValueError(
            "checkpoint manifest run ids do not match the experiment manifest: "
            f"missing={missing_hashes[:5]}, extra={extra_hashes[:5]}"
        )
    for run_id, run in manifest_by_id.items():
        expected = checkpoint_manifest[run_id]
        observed_metadata = (
            str(run["algorithm"]),
            str(run["market"]),
            int(run["seed"]),
        )
        expected_metadata = (
            str(expected["algorithm"]),
            str(expected["market"]),
            int(expected["seed"]),
        )
        if observed_metadata != expected_metadata:
            raise ValueError(f"checkpoint metadata mismatch for {run_id}")

    algorithm_filter = {value.upper() for value in algorithms or ()}
    market_filter = {str(value) for value in markets or ()}
    selected = [
        run
        for run in manifest
        if (not algorithm_filter or str(run["algorithm"]).upper() in algorithm_filter)
        and (not market_filter or str(run["market"]) in market_filter)
    ]
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no manifest runs match the requested filters")

    expected_concepts = {
        "trend",
        "high_volatility",
        "sideway",
        "long_action",
        "short_action",
        "risk_off",
    }
    trials: list[dict] = []
    failures: list[dict] = []
    completed_runs: list[str] = []
    run_coverage: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for run in selected:
        run_id = str(run["run_id"])
        try:
            spec = algorithm_spec(str(run["algorithm"]))
            checkpoint = _resolve_checkpoint(
                run,
                policy_value_root,
                checkpoint_dir,
            )
            expected_checkpoint_hash = checkpoint_manifest[run_id]["checkpoint_sha256"]
            actual_checkpoint_hash = _sha256(checkpoint)
            if actual_checkpoint_hash.lower() != expected_checkpoint_hash.lower():
                raise ValueError(
                    f"frozen checkpoint SHA-256 mismatch for {run_id}: "
                    f"expected {expected_checkpoint_hash}, got {actual_checkpoint_hash}"
                )

            run_dir = _run_directory(experiment_dir, run)
            bank_path = run_dir / "activation" / "oos.pt"
            circuits_path = run_dir / "neuron_circuits.json"
            if not bank_path.exists():
                raise FileNotFoundError(f"activation bank missing: {bank_path}")
            if not circuits_path.exists():
                raise FileNotFoundError(f"circuit file missing: {circuits_path}")
            circuits = _read_json(circuits_path)
            if not isinstance(circuits, dict) or not circuits:
                raise ValueError(f"circuit file is empty or malformed: {circuits_path}")
            unknown_concepts = sorted(set(circuits).difference(expected_concepts))
            if unknown_concepts:
                raise ValueError(
                    f"unexpected circuit concepts for {run_id}: {unknown_concepts}"
                )

            adapter = build_policy_value_adapter(
                algorithm=spec.algorithm,
                agent_config_path=f"configs/{spec.algorithm}.yaml",
                checkpoint_path=checkpoint,
                layers=list(spec.adapter_layers),
                project_root=policy_value_root,
            )
            network = getattr(adapter.agent, spec.network_attr)
            network.eval()
            bank = ActivationBank.load(bank_path)
            permutation_seed = _stable_seed(base_seed, run_id, "permutation")
            random_seeds = tuple(
                _stable_seed(base_seed, run_id, "random", str(index))
                for index in range(int(random_controls))
            )
            circuit_hash = _sha256(circuits_path)
            activation_hash = _sha256(bank_path)
            with strict_float32_matmul():
                run_trials = evaluate_policy_circuits(
                    reference=network,
                    records=bank.records,
                    circuits=circuits,
                    linear_layers=spec.linear_layers,
                    layer_prefix=spec.layer_prefix,
                    probability_scores=spec.probability_scores,
                    permutation_seed=permutation_seed,
                    random_seeds=random_seeds,
                    metadata={
                        "run_id": run_id,
                        "algorithm": spec.algorithm,
                        "market": str(run["market"]),
                        "seed": int(run["seed"]),
                        "checkpoint_sha256": actual_checkpoint_hash,
                        "expected_checkpoint_sha256": expected_checkpoint_hash,
                        "activation_bank_sha256": activation_hash,
                        "circuit_file_sha256": circuit_hash,
                    },
                )
            eligible_count = len(circuits)
            expected_trial_count = eligible_count * int(random_controls)
            if len(run_trials) != expected_trial_count:
                raise RuntimeError(
                    f"run coverage identity failed for {run_id}: "
                    f"{len(run_trials)} != {eligible_count} x {random_controls}"
                )
            if not all(bool(row.get("numeric_valid")) for row in run_trials):
                raise RuntimeError(f"numeric acceptance missing for {run_id}")

            trials.extend(run_trials)
            completed_runs.append(run_id)
            run_coverage.append(
                {
                    "run_id": run_id,
                    "algorithm": spec.algorithm,
                    "market": str(run["market"]),
                    "seed": int(run["seed"]),
                    "expected_concepts": len(expected_concepts),
                    "eligible_concepts": eligible_count,
                    "evaluated_concepts": eligible_count,
                    "excluded_concepts": len(expected_concepts) - eligible_count,
                    "random_controls": int(random_controls),
                    "trial_rows": len(run_trials),
                    "checkpoint_sha256": actual_checkpoint_hash,
                    "activation_bank_sha256": activation_hash,
                    "circuit_file_sha256": circuit_hash,
                    "status": "completed",
                }
            )
            _write_csv(destination / "permutation_sanity_trials.csv", trials)
        except Exception as error:
            failures.append(
                {
                    "run_id": run_id,
                    "algorithm": run.get("algorithm"),
                    "market": run.get("market"),
                    "seed": run.get("seed"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    units = summarize_units(trials)
    observed_unit_rows = sum(row["eligible_concepts"] for row in run_coverage)
    expected_trial_rows = observed_unit_rows * int(random_controls)
    if len(units) != observed_unit_rows or len(trials) != expected_trial_rows:
        failures.append(
            {
                "run_id": "__global__",
                "algorithm": None,
                "market": None,
                "seed": None,
                "error_type": "CoverageIdentityError",
                "error": (
                    f"global coverage failed: units {len(units)}/{observed_unit_rows}, "
                    f"trials {len(trials)}/{expected_trial_rows}"
                ),
            }
        )
    if len(selected) == len(manifest) == 180 and not failures:
        coverage_failure = expected_unit_count_failure(
            observed_unit_rows,
            expected_unit_rows,
        )
        if coverage_failure is not None:
            failures.append(coverage_failure)

    summary = summarize_trials(trials)
    _write_csv(destination / "permutation_sanity_trials.csv", trials)
    _write_csv(destination / "permutation_sanity_units.csv", units)
    _write_csv(destination / "permutation_sanity_summary.csv", summary)
    _write_csv(destination / "permutation_sanity_run_coverage.csv", run_coverage)
    _write_csv(destination / "permutation_sanity_failures.csv", failures)
    audit = {
        "schema_version": 2,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256(manifest_file),
        "checkpoint_manifest_path": str(checkpoint_manifest_file),
        "checkpoint_manifest_sha256": _sha256(checkpoint_manifest_file),
        "project_root": str(policy_value_root),
        "checkpoint_root": str(checkpoint_dir) if checkpoint_dir else None,
        "experiment_root": str(experiment_dir),
        "source_archive_path": str(source_archive) if source_archive else None,
        "source_archive_sha256": _sha256(source_archive) if source_archive else None,
        "random_controls": int(random_controls),
        "base_seed": int(base_seed),
        "requested_runs": len(selected),
        "completed_runs": len(completed_runs),
        "failed_runs": len(failures),
        "trial_rows": len(trials),
        "unit_rows": len(units),
        "expected_unit_rows": int(expected_unit_rows or observed_unit_rows),
        "expected_trial_rows": expected_trial_rows,
        "excluded_concepts": sum(row["excluded_concepts"] for row in run_coverage),
        "algorithms": sorted({str(run["algorithm"]) for run in selected}),
        "completed_run_ids": completed_runs,
        "run_coverage": run_coverage,
        "failures": failures,
        "algorithm_specs": {
            key: asdict(value) for key, value in sorted(_SPECS.items())
        },
        "numeric_execution": {
            "float32_matmul_precision": "highest",
            "cuda_allow_tf32": False,
        },
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(destination / "permutation_sanity_manifest.json", audit)
    if failures and not allow_failures:
        raise RuntimeError(
            f"known-permutation suite failed for {len(failures)} audit entries"
        )
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full known-permutation alignment and intervention sanity checks."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--project-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-root")
    source.add_argument("--experiment-archive")
    parser.add_argument("--extraction-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--random-controls", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=2701)
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--markets", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-unit-rows", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_archive = None
    if args.experiment_archive:
        extraction_root = args.extraction_root or str(
            Path(args.output_dir).resolve().parent / "permutation_sanity_source"
        )
        experiment_root = extract_experiment_archive(
            args.experiment_archive,
            extraction_root,
        )
        source_archive = args.experiment_archive
    else:
        experiment_root = args.experiment_root

    audit = run_permutation_sanity_suite(
        manifest_path=args.manifest,
        checkpoint_manifest_path=args.checkpoint_manifest,
        project_root=args.project_root,
        experiment_root=experiment_root,
        output_dir=args.output_dir,
        checkpoint_root=args.checkpoint_root,
        source_archive_path=source_archive,
        random_controls=args.random_controls,
        base_seed=args.base_seed,
        algorithms=args.algorithms,
        markets=args.markets,
        limit=args.limit,
        expected_unit_rows=args.expected_unit_rows,
        fail_fast=args.fail_fast,
        allow_failures=args.allow_failures,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
