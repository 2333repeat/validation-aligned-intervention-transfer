"""有阶段冻结及可恢复状态的VAIT 600策略实验入口。"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback

import torch
from threadpoolctl import threadpool_limits

from TradingCircuitDiscovery.experiments.prepare_vait_recent import CONFIG, write_json, now
from TradingCircuitDiscovery.experiments.vait_recent_core import select_group, map_group, freeze_pretest


def execute(cfg, stage, algorithm, market):
    start = time.monotonic()
    try:
        fn = {"selection": select_group, "mapping": map_group}[stage]
        result = fn(cfg, algorithm, market)
        print(json.dumps(dict(stage=stage, group=f"{algorithm}_{market}", status="completed", seconds=round(time.monotonic()-start,2),
                              counts=result.get("counts")), ensure_ascii=False), flush=True)
        return result
    except Exception:
        failure = dict(stage=stage, algorithm=algorithm, market=market, at=now(), traceback=traceback.format_exc())
        write_json(Path(cfg["output_root"])/"failures"/f"{stage}_{algorithm}_{market}.json", failure)
        print(failure["traceback"], flush=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage", choices=["selection", "mapping", "pretest", "freeze"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--group", action="append", default=[])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    groups = [(a,m) for a in cfg["algorithms"] for m in cfg["markets"] if not args.group or f"{a}_{m}" in args.group]
    if not groups:
        raise ValueError("no matching groups")
    root = Path(cfg["output_root"])
    if args.stage == "freeze":
        freeze_pretest(cfg)
        return
    stages = ["selection", "mapping"] if args.stage == "pretest" else [args.stage]
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=args.workers) as pool:
        for stage in stages:
            write_json(root/"active_stage.json", dict(stage=stage, started_at=now(), requested_groups=len(groups), workers=args.workers))
            futures = [pool.submit(execute, cfg, stage, a, m) for a,m in groups]
            results = [f.result() for f in as_completed(futures)]
            write_json(root/f"{stage}_stage_complete.json", dict(completed_at=now(), groups=len(results), results=results))
    if args.stage == "pretest" and not args.group:
        freeze_pretest(cfg)
    print("REQUESTED_STAGES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
