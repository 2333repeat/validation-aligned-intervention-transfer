from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from TradingCircuitDiscovery.experiments.manifest import build_full_manifest, write_manifest
from TradingCircuitDiscovery.experiments.run_checkpoint import _write_json, run_directory
from TradingCircuitDiscovery.experiments.run_supplement_checkpoint import run_supplement_checkpoint
from TradingCircuitDiscovery.utils.config import load_config


def select_run_shard(runs, shard_count: int = 1, shard_index: int = 0):
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [run for index, run in enumerate(runs) if index % shard_count == shard_index]


def run_supplement_matrix(runs, output_root, execute_run, resume: bool = True, gpu_gate=None) -> dict[str, int]:
    root = Path(output_root)
    counts = {"completed": 0, "failed": 0, "skipped": 0, "total": len(runs)}
    for run in runs:
        run_dir = run_directory(root, run)
        status_path = run_dir / "status.json"
        if resume and status_path.is_file() and (run_dir / "supplement.json").is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            if status.get("status") == "completed":
                counts["skipped"] += 1
                continue
        if gpu_gate:
            gpu_gate()
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(status_path, {"run_id": run.run_id, "status": "running", "error": None})
        try:
            result = execute_run(run)
            if result.get("supplement") is not None:
                _write_json(run_dir / "supplement.json", result["supplement"])
            _write_json(status_path, {"run_id": run.run_id, "status": "completed", "error": None})
            counts["completed"] += 1
        except Exception as exc:
            _write_json(
                status_path,
                {
                    "run_id": run.run_id,
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


def gpu_utilization() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def wait_for_gpu_idle(max_utilization: int = 20, poll_seconds: int = 60, consecutive_checks: int = 2) -> None:
    idle_checks = 0
    while idle_checks < consecutive_checks:
        utilization = gpu_utilization()
        idle_checks = idle_checks + 1 if utilization <= max_utilization else 0
        if idle_checks < consecutive_checks:
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--base-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default="TradingCircuitDiscovery/configs/supplement_experiment.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--skip-gpu-gate", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-manifest-write", action="store_true")
    args = parser.parse_args()
    runs = build_full_manifest(args.project_root, validate_paths=True)
    output_root = Path(args.output_root)
    if not args.skip_manifest_write:
        write_manifest(runs, output_root / "manifest.json")
    if args.run_id:
        selected = set(args.run_id)
        runs = [run for run in runs if run.run_id in selected]
        missing = selected - {run.run_id for run in runs}
        if missing:
            raise ValueError(f"Unknown run ids: {sorted(missing)}")
    runs = select_run_shard(runs, args.shard_count, args.shard_index)
    config = load_config(args.config)
    gate = None if args.skip_gpu_gate else lambda: wait_for_gpu_idle(args.max_gpu_utilization)
    summary = run_supplement_matrix(
        runs,
        output_root,
        execute_run=lambda run: run_supplement_checkpoint(
            run,
            project_root=args.project_root,
            base_output_root=args.base_output_root,
            output_root=output_root,
            config=config,
        ),
        resume=args.resume,
        gpu_gate=gate,
    )
    print(json.dumps(summary))
    if not matrix_summary_succeeded(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
