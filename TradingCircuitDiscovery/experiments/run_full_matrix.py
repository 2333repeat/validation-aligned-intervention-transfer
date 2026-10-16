from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from TradingCircuitDiscovery.experiments.manifest import build_full_manifest, write_manifest
from TradingCircuitDiscovery.experiments.run_checkpoint import _write_json, run_checkpoint_experiment, run_directory
from TradingCircuitDiscovery.utils.config import load_config


def run_full_matrix(runs, output_root, execute_run, resume: bool = True, gpu_gate=None) -> dict[str, int]:
    output_root = Path(output_root)
    counts = {"completed": 0, "failed": 0, "skipped": 0, "total": len(runs)}
    for run in runs:
        run_dir = run_directory(output_root, run)
        status_path = run_dir / "status.json"
        if resume and status_path.is_file():
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
            if current.get("status") == "completed" and (run_dir / "metrics.json").is_file():
                counts["skipped"] += 1
                continue
        if gpu_gate:
            gpu_gate()
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(status_path, {"run_id": run.run_id, "status": "running", "error": None})
        try:
            result = execute_run(run)
            if result.get("metrics") is not None:
                _write_json(run_dir / "metrics.json", result["metrics"])
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


def wait_for_gpu(min_free_mib: int = 4096, max_utilization: int = 95, poll_seconds: int = 20) -> None:
    import torch

    while torch.cuda.is_available():
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_mib = free_bytes // (1024 * 1024)
        if free_mib >= min_free_mib:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="TradingCircuitDiscovery/configs/full_experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runs = build_full_manifest(args.project_root, validate_paths=True)
    write_manifest(runs, Path(args.output_root) / "manifest.json")
    config = load_config(args.config)
    summary = run_full_matrix(
        runs,
        args.output_root,
        execute_run=lambda run: run_checkpoint_experiment(
            run,
            project_root=args.project_root,
            output_root=args.output_root,
            config=config,
        ),
        resume=args.resume,
        gpu_gate=wait_for_gpu,
    )
    print(json.dumps(summary))
    if not matrix_summary_succeeded(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
