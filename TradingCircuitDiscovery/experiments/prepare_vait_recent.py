"""准备600策略VAIT批次的冻结权重清单与独立时间段观察，不运行测试推理。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from TradingCircuitDiscovery.concepts.market_regime import label_market_regime
from TradingCircuitDiscovery.integrations.policy_value_env import fit_training_regime_thresholds

CONFIG = Path(os.environ.get("VAIT_RECENT_CONFIG", "TradingCircuitDiscovery/configs/vait_600_recent_20261117.json"))
SYMBOLS = {"DJI": "^DJI", "FCHI": "^FCHI", "HSI": "^HSI", "IXIC": "^IXIC",
           "KS11": "^KS11", "N225": "^N225", "SP500": "^GSPC"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def build_observations(frame, start, end, thresholds, policy_root):
    """原环境预处理和get_state；前90日仅暖启动，保留评价期最后交易日。"""
    sys.path.insert(0, str(Path(policy_root).resolve()))
    from TradingEnv import TradingEnv

    warmup = (pd.Timestamp(start) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    env = TradingEnv(frame, warmup, end)
    env.init_market_data()
    dates = pd.to_datetime(env.account.Date)
    indices = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    if len(indices) < 120 or indices[0] < env.window_size - 1:
        raise ValueError(f"insufficient dated observations: {start}..{end}: {len(indices)}")
    states = np.stack([env.get_state(int(i)) for i in indices]).astype(np.float32)
    if states.shape[1] != 140 or not np.isfinite(states).all():
        raise ValueError("invalid policy observations")
    labels = [label_market_regime(env.account, int(i), thresholds, 20) for i in indices]
    return dict(states=states, dates=dates.iloc[indices].dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
                prices=env.trend.to_numpy(dtype=np.float64)[indices],
                trend=np.asarray([x["trend"] for x in labels], dtype=np.int8),
                high_volatility=np.asarray([x["high_volatility"] for x in labels], dtype=np.int8),
                sideway=np.asarray([x["sideway"] for x in labels], dtype=np.int8))


def download_market(market, output):
    import yfinance as yf

    symbol = SYMBOLS.get(market, market)
    for attempt in range(3):
        frame = yf.download(symbol, start="2024-10-01", end="2026-11-01", auto_adjust=False,
                            progress=False, threads=False, timeout=25)
        if len(frame):
            break
        time.sleep(2 * (attempt + 1))
    if frame.empty:
        raise RuntimeError(f"empty Yahoo Finance series: {symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    frame.Date = pd.to_datetime(frame.Date).dt.strftime("%Y-%m-%d")
    frame.to_csv(output, index=False)
    return dict(provider="Yahoo Finance via yfinance", symbol=symbol, auto_adjust=False,
                download_start="2024-10-01", download_end_exclusive="2026-11-01", access_time=now())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root, policy = Path(cfg["output_root"]), Path(cfg["policy_root"])
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != cfg:
            raise ValueError("frozen protocol differs; refusing overwrite")
    else:
        write_json(frozen, cfg)
    archived = {r["path"]: r["sha256"] for r in json.loads(Path(cfg["archive_manifest"]).read_text(encoding="utf-8"))}
    models = []
    for algo in cfg["algorithms"]:
        for market in cfg["markets"]:
            for seed in cfg["seeds"]:
                run_id = f"{algo}-{market}-update2022-cuda-s{seed}"
                rel = f"source/Policy-Value/result/{run_id}/agent/model/best_model.pth"
                path = policy / "result" / run_id / "agent/model/best_model.pth"
                digest = sha256(path)
                if digest != archived[rel]:
                    raise ValueError(f"checkpoint hash mismatch: {run_id}")
                models.append(dict(run_id=run_id, algorithm=algo, market=market, seed=seed,
                                   path=str(path), sha256=digest, training_end=cfg["training_end"]))
    if len(models) != 600 or len({r["sha256"] for r in models}) != 600:
        raise ValueError("600 unique checkpoints required")
    write_json(root / "model_manifest.json", models)
    all_provenance = []
    for market in cfg["markets"]:
        out = root / "data" / market
        out.mkdir(parents=True, exist_ok=True)
        old_path = policy / "Data" / f"{market}.csv"
        expected = archived[f"source/Policy-Value/Data/{market}.csv"]
        if sha256(old_path) != expected:
            raise ValueError(f"original data hash mismatch: {market}")
        old = pd.read_csv(old_path)
        future = out / "future_ohlcv.csv"
        source_info = out / "future_source.json"
        if not future.exists():
            info = download_market(market, future)
            write_json(source_info, info)
        fresh = pd.read_csv(future)
        if pd.to_datetime(fresh.Date).max() < pd.Timestamp("2026-10-28"):
            raise ValueError(f"missing late October 2026 data: {market}")
        # 仅在训练时期原始行情上拟合概念阈值。
        thresholds = fit_training_regime_thresholds(old, cfg["training_end"], cfg["lookback"], cfg["threshold_quantile"])
        provenance = dict(market=market, training_data_path=str(old_path), training_data_sha256=expected,
                          threshold_training_end=cfg["training_end"], thresholds=thresholds,
                          future_data_sha256=sha256(future), source=json.loads(source_info.read_text(encoding="utf-8")), periods={})
        for period, (start, end) in cfg["periods"].items():
            frame = fresh if period.startswith("test_") else old
            values = build_observations(frame, start, end, thresholds, policy)
            output = out / f"{period}.npz"
            if output.exists():
                previous = np.load(output)
                if any(not np.array_equal(previous[k], v) for k, v in values.items()):
                    raise ValueError(f"prepared data changed: {output}")
            else:
                np.savez_compressed(output, **values)
            provenance["periods"][period] = dict(rows=len(values["dates"]), first_date=str(values["dates"][0]),
                                                last_date=str(values["dates"][-1]), sha256=sha256(output))
        write_json(out / "provenance.json", provenance)
        all_provenance.append(provenance)
        print(market, provenance["periods"], flush=True)
    write_json(root / "data_manifest.json", all_provenance)
    write_json(root / "preparation_complete.json", dict(completed_at=now(), models=len(models), markets=len(all_provenance),
               protocol_sha256=sha256(frozen), model_manifest_sha256=sha256(root/"model_manifest.json"),
               data_manifest_sha256=sha256(root/"data_manifest.json")))
    print("PREPARATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
