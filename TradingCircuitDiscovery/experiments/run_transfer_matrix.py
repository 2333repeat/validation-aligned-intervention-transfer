from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from TradingCircuitDiscovery.evaluation.alignment_methods import ALIGNMENT_METHODS
from TradingCircuitDiscovery.experiments.manifest import (
    build_full_manifest,
    write_manifest,
)
from TradingCircuitDiscovery.experiments.run_checkpoint import _write_json, run_directory
from TradingCircuitDiscovery.experiments.run_supplement_matrix import wait_for_gpu_idle
from TradingCircuitDiscovery.experiments.run_transfer_checkpoint import (
    run_transfer_checkpoint,
)


def resolve_alignment_methods(values) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(str(value) for value in values))
    if not methods:
        return ALIGNMENT_METHODS
    unknown = set(methods) - set(ALIGNMENT_METHODS)
    if unknown:
        raise ValueError(f"unknown alignment methods: {sorted(unknown)}")
    return methods


def select_target_shard(runs, shard_count: int = 1, shard_index: int = 0):
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [run for index, run in enumerate(runs) if index % shard_count == shard_index]


def run_transfer_matrix(
    target_runs,
    output_root,
    execute_target,
    resume: bool = True,
    source_pool=None,
    gpu_gate=None,
) -> dict[str, int]:
    root = Path(output_root)
    all_sources = list(source_pool if source_pool is not None else target_runs)
    counts = {"completed": 0, "failed": 0, "skipped": 0, "total": len(target_runs)}
    for target in target_runs:
        target_dir = run_directory(root, target)
        status_path = target_dir / "status.json"
        transfer_path = target_dir / "transfer.json"
        if resume and status_path.is_file() and transfer_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            if status.get("status") == "completed":
                counts["skipped"] += 1
                continue
        if gpu_gate is not None:
            gpu_gate()
        sources = [
            run
            for run in all_sources
            if run.algorithm == target.algorithm
            and run.market == target.market
            and run.seed != target.seed
        ]
        target_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            status_path,
            {"run_id": target.run_id, "status": "running", "error": None},
        )
        try:
            result = execute_target(target, sources)
            if result.get("transfer") is not None:
                _write_json(transfer_path, result["transfer"])
            _write_json(
                status_path,
                {"run_id": target.run_id, "status": "completed", "error": None},
            )
            counts["completed"] += 1
        except Exception as exc:
            _write_json(
                status_path,
                {
                    "run_id": target.run_id,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            counts["failed"] += 1
    return counts


def matrix_summary_succeeded(summary: dict[str, int]) -> bool:
    return (
        int(summary.get("failed", -1)) == 0
        and int(summary.get("completed", 0)) + int(summary.get("skipped", 0))
        == int(summary.get("total", -1))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--base-output-root", required=True)
    parser.add_argument("--supplement-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--random-controls", type=int, default=50)
    parser.add_argument(
        "--alignment-method",
        action="append",
        default=[],
        choices=ALIGNMENT_METHODS,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--skip-gpu-gate", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-manifest-write", action="store_true")
    args = parser.parse_args()
    alignment_methods = resolve_alignment_methods(args.alignment_method)
    all_runs = build_full_manifest(args.project_root, validate_paths=True)
    output_root = Path(args.output_root)
    if not args.skip_manifest_write:
        write_manifest(all_runs, output_root / "manifest.json")
    targets = all_runs
    if args.run_id:
        selected = set(args.run_id)
        targets = [run for run in targets if run.run_id in selected]
        missing = selected - {run.run_id for run in targets}
        if missing:
            raise ValueError(f"Unknown run ids: {sorted(missing)}")
    targets = select_target_shard(targets, args.shard_count, args.shard_index)
    gate = None if args.skip_gpu_gate else lambda: wait_for_gpu_idle(args.max_gpu_utilization)
    summary = run_transfer_matrix(
        targets,
        output_root=output_root,
        source_pool=all_runs,
        execute_target=lambda target, sources: run_transfer_checkpoint(
            target,
            source_runs=sources,
            project_root=args.project_root,
            base_output_root=args.base_output_root,
            supplement_output_root=args.supplement_output_root,
            output_root=output_root,
            random_controls=args.random_controls,
            alignment_methods=alignment_methods,
        ),
        resume=args.resume,
        gpu_gate=gate,
    )
    print(json.dumps(summary))
    if not matrix_summary_succeeded(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
