"""只消费全局冻结选择的2025/2026 VAIT评价，不允许反馈更新选集或映射。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from TradingCircuitDiscovery.experiments.prepare_vait_recent import CONFIG, now, sha256, write_json
from TradingCircuitDiscovery.experiments.vait_recent_core import (
    load_adapter, load_period, concepts, BatchedIntervention, batch_metrics, margin,
)


def account_trace(prices, actions, cost, initial_cash=500000., execute_last=False):
    """原离散全仓/整数股交易规则；最后价格作估值，不调用未来五步训练奖励。"""
    cash, position, units = float(initial_cash), 0, 0
    capitals, turnover, trades = [], 0., 0
    for i, (price, action) in enumerate(zip(prices, actions)):
        if i < len(prices)-1 or execute_last:
            if action == 1 and position == 0:
                units = int(cash / (price*(1+cost)))
                cash -= units*price*(1+cost); position = 1
                turnover += abs(units*price); trades += 1
            elif action == 1 and position == -1:
                cash -= units*price*(1+cost); position = 0
                turnover += abs(units*price); trades += 1
            elif action == 2 and position == 1:
                cash += units*price*(1-cost); position = 0
                turnover += abs(units*price); units = 0; trades += 1
            elif action == 2 and position == 0:
                units = int(np.floor(cash / (price*(1+cost))))
                cash += units*price*(1-cost); position = -1
                turnover += abs(units*price); trades += 1
        capitals.append(cash + position*units*price)
    capitals = np.asarray(capitals, dtype=float)
    previous = np.r_[initial_cash, capitals[:-1]]
    returns = np.divide(capitals-previous, previous, out=np.zeros_like(capitals), where=np.abs(previous)>1e-12)
    peaks = np.maximum.accumulate(np.r_[initial_cash,capitals])[1:]
    drawdown = np.divide(peaks-capitals, peaks, out=np.zeros_like(capitals), where=peaks>0)
    volatility = returns.std(ddof=0)
    metrics = dict(net_return=float(capitals[-1]/initial_cash-1), net_pnl=float(capitals[-1]-initial_cash),
                   sharpe=float(np.sqrt(252)*returns.mean()/volatility) if volatility>1e-12 else 0.,
                   max_drawdown=float(drawdown.max()), annualized_volatility=float(np.sqrt(252)*volatility),
                   turnover_notional=float(turnover), executed_trades=trades,
                   action_switch_rate=float(np.mean(np.diff(actions)!=0)), nonpositive_equity=bool(np.any(capitals<=0)))
    return metrics, capitals


def verify_pretest(cfg):
    root = Path(cfg["output_root"])
    path = root/"pretest_freeze.json"
    if not path.exists():
        raise ValueError("all selections and mappings must be frozen before test inference")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    frozen_protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    path_keys = {"output_root", "policy_root", "archive_manifest"}
    if {k:v for k,v in cfg.items() if k not in path_keys} != {k:v for k,v in frozen_protocol.items() if k not in path_keys}:
        raise ValueError("scientific configuration differs from the frozen protocol")
    for rel, expected in freeze["files"].items():
        if sha256(root/rel.replace("\\", "/")) != expected:
            raise ValueError(f"frozen input modified: {rel}")
    for market in json.loads((root/"data_manifest.json").read_text(encoding="utf-8")):
        for period, record in market["periods"].items():
            actual = root/"data"/market["market"]/f"{period}.npz"
            if sha256(actual) != record["sha256"]:
                raise ValueError(f"frozen observation data modified: {actual}")
    for model in json.loads((root/"model_manifest.json").read_text(encoding="utf-8")):
        actual = Path(cfg["policy_root"])/"result"/model["run_id"]/"agent/model/best_model.pth"
        if sha256(actual) != model["sha256"]:
            raise ValueError(f"frozen checkpoint modified: {model['run_id']}")
    return freeze


def evaluate_group(cfg, algorithm, market):
    root = Path(cfg["output_root"])
    group = f"{algorithm}_{market}"
    out = root/"evaluation"/group
    out.mkdir(parents=True, exist_ok=True)
    frozen_hash = sha256(root/"pretest_freeze.json")
    status_path = out/"complete.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["pretest_freeze_sha256"] != frozen_hash:
            raise ValueError("evaluation resume freeze mismatch")
        return status
    selections = json.loads((root/"selection"/group/"selection.json").read_text(encoding="utf-8"))
    with gzip.open(root/"mapping"/group/"mappings.json.gz", "rt", encoding="utf-8") as f:
        mappings = json.load(f)["records"]
    period_data = {period: load_period(cfg,market,period) for period in ("test_2025", "test_2026", "validation")}
    transfer_rows, financial_rows, native_rows, control_arrays, observation_arrays = [], [], [], {}, {}
    for target in selections:
        adapter, checkpoint = load_adapter(cfg, algorithm, market, target["seed"])
        if sha256(checkpoint) != target["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after selection")
        target_rows = [r for r in mappings if r["target_seed"] == target["seed"]]
        for period, data in period_data.items():
            baseline = adapter.get_action_distribution(data["states"]).numpy()
            labels = concepts(data,baseline)
            engines = {layer: BatchedIntervention(adapter,data["states"],layer,np.asarray(mean,dtype=np.float32))
                       for layer,mean in target["means"].items()}
            native_distributions = {}
            native_actions = {}
            for concept, choice in target["concepts"].items():
                if choice["status"] != "completed":
                    continue
                probabilities = engines[choice["layer"]].probabilities(choice["final"]["indices"])
                native_distributions[concept] = probabilities
                native_actions[concept] = probabilities.argmax(axis=1)
                metric = batch_metrics(baseline,probabilities,probabilities)
                effect_by_day = np.abs(margin(probabilities,baseline.argmax(axis=1))-margin(baseline,baseline.argmax(axis=1)))[0]
                concept_mask = labels[concept].astype(bool)
                gap = (float(effect_by_day[concept_mask].mean()-effect_by_day[~concept_mask].mean())
                       if concept_mask.any() and (~concept_mask).any() else np.nan)
                native_rows.append(dict(algorithm=algorithm,market=market,seed=target["seed"],run_id=target["run_id"],
                    period=period,concept=concept,layer=choice["layer"],size=len(choice["final"]["indices"]),
                    validation_effect=choice["final"]["validation_effect"], native_effect=float(metric["mean_absolute_margin_change"][0]),
                    specificity_gap=gap, prevalence=float(concept_mask.mean()), rows=len(data["dates"])))
            policies = {"original": baseline.argmax(axis=1), **native_actions}
            buy_hold = np.zeros(len(data["prices"]),dtype=int); buy_hold[0] = 1
            policies["buy_hold"] = buy_hold
            for policy_name, actions in policies.items():
                for cost in (cfg["transaction_costs"] if period != "validation" else [0.003]):
                    metrics, _ = account_trace(data["prices"],actions,cost,cfg["initial_cash"])
                    financial_rows.append(dict(algorithm=algorithm,market=market,seed=target["seed"],run_id=target["run_id"],
                                               period=period,policy=policy_name,cost=cost,**metrics))
            if period == "validation":
                continue
            for row in target_rows:
                metadata = {k: row[k] for k in ("algorithm","market","concept","source_seed","target_seed","source_run_id","target_run_id","status")}
                metadata.update(period=period,test_observations=len(data["dates"]),first_date=str(data["dates"][0]),last_date=str(data["dates"][-1]))
                if row["status"] != "completed":
                    transfer_rows.append(metadata)
                    continue
                engine = engines[row["layer"]]
                methods = cfg["alignment_methods"]
                sets = [row["target_indices"],row["source_indices"]] + [row["mappings"][m] for m in methods] + row["random_sets"] + row["alignment_null_sets"]
                probabilities = engine.probabilities_many(sets)
                metrics = batch_metrics(baseline,probabilities,probabilities[0])
                random_start = 2+len(methods)
                null_start = random_start+cfg["random_controls"]
                random_effects = metrics["mean_absolute_margin_change"][random_start:null_start]
                null_effects = metrics["mean_absolute_margin_change"][null_start:]
                identifier = f"{period}_s{row['source_seed']}_t{row['target_seed']}_{row['concept']}"
                control_arrays[identifier] = np.stack([random_effects,null_effects])
                main_index = 2+methods.index("hungarian_abs")
                observation_arrays[identifier] = np.stack([margin(probabilities[j],baseline.argmax(axis=1))[0]-margin(baseline,baseline.argmax(axis=1))[0]
                                                           for j in (main_index,1,0)]).astype(np.float32)
                for j, method in enumerate(methods, start=2):
                    record = dict(metadata,alignment_method=method,layer=row["layer"],source_circuit_size=len(row["source_indices"]),
                                  target_native_size=len(row["target_indices"]),mapping_jaccard=row["mapping_jaccard"],mapping_exact=row["mapping_exact"],
                                  mean_absolute_alignment_correlation=row["mean_absolute_alignment_correlation"],
                                  min_absolute_alignment_correlation=row["min_absolute_alignment_correlation"])
                    for key, values in metrics.items():
                        record[f"aligned_{key}"] = float(values[j])
                        record[f"raw_{key}"] = float(values[1])
                        record[f"native_{key}"] = float(values[0])
                        finite = values[random_start:null_start]
                        record[f"random_{key}"] = float(np.nanmean(finite)) if np.isfinite(finite).any() else np.nan
                        finite = values[null_start:]
                        record[f"null_{key}"] = float(np.nanmean(finite)) if np.isfinite(finite).any() else np.nan
                    observed = metrics["mean_absolute_margin_change"][j]
                    record.update(random_empirical_p=float((1+np.sum(random_effects>=observed))/(len(random_effects)+1)),
                                  null_empirical_p=float((1+np.sum(null_effects>=observed))/(len(null_effects)+1)))
                    transfer_rows.append(record)
    pd.DataFrame(transfer_rows).to_csv(out/"transfer.csv.gz",index=False,compression="gzip")
    pd.DataFrame(financial_rows).to_csv(out/"financial.csv.gz",index=False,compression="gzip")
    pd.DataFrame(native_rows).to_csv(out/"native.csv.gz",index=False,compression="gzip")
    np.savez_compressed(out/"control_effects.npz",**control_arrays)
    np.savez_compressed(out/"observation_signed_margins.npz",**observation_arrays)
    completed = sum(r["status"] == "completed" for r in transfer_rows)
    result = dict(stage="evaluation",algorithm=algorithm,market=market,completed_at=now(),completed_method_rows=completed,
                  exclusions=sum(r["status"] != "completed" for r in transfer_rows), failed_rows=0,
                  pretest_freeze_sha256=frozen_hash, artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!="complete.json"})
    write_json(status_path,result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--group",action="append",default=[])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    verify_pretest(cfg)
    root=Path(cfg["output_root"])
    write_json(root/"test_access_manifest.json",dict(opened_at=now(),pretest_freeze_sha256=sha256(root/"pretest_freeze.json"),periods=["test_2025","test_2026"]))
    groups=[(a,m) for a in cfg["algorithms"] for m in cfg["markets"] if not args.group or f"{a}_{m}" in args.group]
    def execute(a,m):
        start=time.monotonic()
        try:
            result=evaluate_group(cfg,a,m)
            print(json.dumps(dict(group=f"{a}_{m}",seconds=round(time.monotonic()-start,2),rows=result["completed_method_rows"])),flush=True)
            return result
        except Exception:
            write_json(root/"failures"/f"evaluation_{a}_{m}.json",dict(at=now(),traceback=traceback.format_exc()))
            raise
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=[f.result() for f in as_completed([pool.submit(execute,a,m) for a,m in groups])]
    write_json(root/"evaluation_stage_complete.json",dict(completed_at=now(),groups=len(results),results=results))
    verify_pretest(cfg)
    print("EVALUATION_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
