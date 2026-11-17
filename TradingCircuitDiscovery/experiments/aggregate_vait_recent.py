"""复用VAIT既有依赖重采样定义，生成新批次的证据表与论文事实。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
    hierarchical_transfer_estimate, dependence_bootstrap_summary,
    _market_first_samples, _multiway_samples, _add_one_two_sided_p,
)
from TradingCircuitDiscovery.experiments.prepare_vait_recent import CONFIG, write_json, sha256, now
from TradingCircuitDiscovery.experiments.vait_recent_core import PROFILE_KEYS, stable_seed

EFFECT = "mean_absolute_margin_change"
PROFILE_SIGNS = {key: (-1 if "jsd" in key else 1) for key in PROFILE_KEYS}
SENSITIVITY = {"mapped_minus_same_index": f"raw_{EFFECT}", "mapped_minus_random": f"random_{EFFECT}",
               "mapped_minus_alignment_null": f"null_{EFFECT}"}


def estimate(frame, key):
    selected=frame.loc[np.isfinite(frame[key])]
    return hierarchical_transfer_estimate(selected,key) if len(selected) else float("nan")


def interval(frame,key,repeats,seed,method="multiway_pigeonhole"):
    selected=frame.loc[np.isfinite(frame[key])].copy()
    if selected.empty:
        return dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p_value=np.nan,bootstrap_method=method,transfer_rows=0)
    generator = _multiway_samples if method=="multiway_pigeonhole" else _market_first_samples
    samples=generator(selected,key,repeats,np.random.default_rng(seed))
    return dict(estimate=estimate(selected,key),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),
                p_value=_add_one_two_sided_p(samples),bootstrap_method=method,transfer_rows=len(selected),repeats=repeats,random_state=seed)


def aggregate(cfg):
    root=Path(cfg["output_root"]); out=root/"aggregate"; out.mkdir(exist_ok=True)
    folders=[root/"evaluation"/f"{a}_{m}" for a in cfg["algorithms"] for m in cfg["markets"]]
    hashes={}
    for folder in folders:
        status=json.loads((folder/"complete.json").read_text(encoding="utf-8"))
        if status["failed_rows"] or status["pretest_freeze_sha256"]!=sha256(root/"pretest_freeze.json"):
            raise ValueError(f"invalid evaluation status: {folder}")
        for name,expected in status["artifacts"].items():
            if sha256(folder/name)!=expected:raise ValueError(f"evaluation hash mismatch: {folder/name}")
        hashes[str(folder.relative_to(root))]=status["artifacts"]["transfer.csv.gz"]
    keep={"algorithm","market","concept","source_seed","target_seed","source_run_id","target_run_id","status","period","alignment_method",
          "source_circuit_size","target_native_size","test_observations","first_date","last_date","mapping_jaccard","mapping_exact",
          "mean_absolute_alignment_correlation","min_absolute_alignment_correlation","random_empirical_p","null_empirical_p"}
    keep.update(f"{prefix}_{EFFECT}" for prefix in ("aligned","raw","native","random","null"))
    keep.update(f"{prefix}_{key}" for prefix in ("aligned","raw") for key in PROFILE_KEYS)
    frame=pd.concat([pd.read_csv(p/"transfer.csv.gz",usecols=lambda c:c in keep) for p in folders],ignore_index=True)
    complete=frame.loc[frame.status.eq("completed")].copy()
    for name,reference in SENSITIVITY.items():complete[name]=complete[f"aligned_{EFFECT}"]-complete[reference]
    for key,sign in PROFILE_SIGNS.items():complete[f"gain_{key}"]=sign*(complete[f"aligned_{key}"]-complete[f"raw_{key}"])
    complete["beats_same_index"]=(complete.mapped_minus_same_index>0).astype(float)
    complete["beats_random"]=(complete.mapped_minus_random>0).astype(float)
    complete["beats_both"]=((complete.mapped_minus_same_index>0)&(complete.mapped_minus_random>0)).astype(float)
    complete["random_rank_below_05"]=(complete.random_empirical_p<.05).astype(float)
    point_rows, method_rows, interval_rows, comparison_rows=[],[],[],[]
    nominal={}
    for period in ("test_2025","test_2026"):
        current=complete.loc[complete.period.eq(period)]
        main=current.loc[current.alignment_method.eq("hungarian_abs")]
        primary=main.loc[~main.concept.isin(cfg["primary_excluded_concepts"])]
        raw=frame.loc[frame.period.eq(period)]
        exclusions=raw.loc[~raw.status.eq("completed")].status.value_counts().to_dict()
        nominal[period]=dict(nominal_transfers=600*9*6,eligible_transfers=len(main),primary_transfers=len(primary),
                             alignment_evaluations=len(current),exclusions={k:int(v) for k,v in exclusions.items()})
        for algorithm in [*cfg["algorithms"],"All"]:
            selected=primary if algorithm=="All" else primary.loc[primary.algorithm.eq(algorithm)]
            row=dict(period=period,algorithm=algorithm,transfers=len(selected))
            for prefix in ("native","raw","aligned","random","null"):
                row[f"{prefix}_effect"]=estimate(selected,f"{prefix}_{EFFECT}")
            for key in [*SENSITIVITY,"beats_same_index","beats_random","beats_both","random_rank_below_05",
                        "mapping_jaccard","mean_absolute_alignment_correlation","min_absolute_alignment_correlation"]:
                row[key]=estimate(selected,key)
            row["mapped_to_native_ratio"]=row["aligned_effect"]/row["native_effect"] if row["native_effect"] else np.nan
            point_rows.append(row)
        for method in cfg["alignment_methods"]:
            selected=current.loc[current.alignment_method.eq(method)&~current.concept.isin(cfg["primary_excluded_concepts"])]
            row=dict(period=period,alignment_method=method,transfers=len(selected))
            for key in [*SENSITIVITY,*[f"gain_{k}" for k in PROFILE_KEYS]]:
                row[key]=estimate(selected,key)
            row["mapped_effect"]=estimate(selected,f"aligned_{EFFECT}")
            method_rows.append(row)
        endpoints=["mapped_minus_same_index","mapped_minus_random",*[f"gain_{k}" for k in PROFILE_KEYS],"mapped_minus_alignment_null"]
        for endpoint in endpoints:
            family=("response" if endpoint.startswith("gain_") else "alignment_null" if endpoint.endswith("alignment_null") else "sensitivity")
            cache=out/f"bootstrap_{period}_{endpoint}.json"
            if cache.exists():
                rows=json.loads(cache.read_text(encoding="utf-8"))
            else:
                selected=primary.loc[np.isfinite(primary[endpoint])]
                rows=dependence_bootstrap_summary(selected,endpoint,repeats=cfg["bootstrap_repeats"],
                        random_state=stable_seed(cfg["random_seed"],period,endpoint))
                write_json(cache,rows)
            interval_rows.extend([dict(period=period,algorithm="All",family=family,endpoint=endpoint,specification="primary_no_sideway",**r) for r in rows])
            print(period,endpoint,"bootstrap complete",flush=True)
        for algorithm in cfg["algorithms"]:
            selected=primary.loc[primary.algorithm.eq(algorithm)]
            for endpoint in ("mapped_minus_same_index","mapped_minus_random"):
                cache=out/f"bootstrap_{period}_{algorithm}_{endpoint}.json"
                if cache.exists():row=json.loads(cache.read_text(encoding="utf-8"))
                else:
                    row=interval(selected,endpoint,cfg["bootstrap_repeats"],stable_seed(cfg["random_seed"],period,algorithm,endpoint))
                    write_json(cache,row)
                interval_rows.append(dict(period=period,algorithm=algorithm,family="algorithm_sensitivity",endpoint=endpoint,specification="primary_no_sideway",**row))
        for endpoint in ("mapped_minus_same_index","mapped_minus_random"):
            cache=out/f"bootstrap_{period}_all_concepts_{endpoint}.json"
            if cache.exists():rows=json.loads(cache.read_text(encoding="utf-8"))
            else:
                rows=dependence_bootstrap_summary(main,endpoint,cfg["bootstrap_repeats"],stable_seed(cfg["random_seed"],period,"all",endpoint))
                write_json(cache,rows)
            interval_rows.extend([dict(period=period,algorithm="All",family="all_concepts",endpoint=endpoint,specification="all_concepts",**r) for r in rows])
        join=["algorithm","market","concept","source_run_id","target_run_id"]
        for method in cfg["alignment_methods"]:
            if method=="hungarian_abs":continue
            selected=current.loc[current.alignment_method.eq(method)&~current.concept.isin(cfg["primary_excluded_concepts"])]
            for endpoint in ("mapped_minus_same_index","gain_intervened_distribution_jsd_to_native"):
                paired=selected[join+[endpoint]].merge(primary[join+[endpoint]],on=join,suffixes=("_candidate","_hungarian"),validate="one_to_one")
                paired["difference"]=paired[endpoint+"_candidate"]-paired[endpoint+"_hungarian"]
                row=interval(paired,"difference",cfg["bootstrap_repeats"],stable_seed(cfg["random_seed"],period,method,endpoint),"market_first")
                comparison_rows.append(dict(period=period,alignment_method=method,endpoint=endpoint,**row))
        pd.DataFrame(point_rows).to_csv(out/"sensitivity.csv",index=False)
        pd.DataFrame(method_rows).to_csv(out/"alignment_methods.csv",index=False)
        pd.DataFrame(interval_rows).to_csv(out/"inference_progress.csv",index=False)
    inference=pd.DataFrame(interval_rows)
    # 两个年份共同进入各自endpoint家族；重采样方案分别校正。
    inference["q_value"]=np.nan
    for _,indices in inference.groupby(["family","bootstrap_method"]).groups.items():
        valid=inference.loc[indices,"p_value"].dropna()
        inference.loc[valid.index,"q_value"]=multipletests(valid,method="fdr_bh")[1]
    inference.to_csv(out/"inference.csv",index=False)
    comparisons=pd.DataFrame(comparison_rows)
    comparisons["q_value"]=multipletests(comparisons.p_value,method="fdr_bh")[1]
    comparisons.to_csv(out/"paired_method_comparisons.csv",index=False)
    native=pd.concat([pd.read_csv(p/"native.csv.gz") for p in folders],ignore_index=True)
    financial=pd.concat([pd.read_csv(p/"financial.csv.gz") for p in folders],ignore_index=True)
    original=financial.loc[financial.policy.eq("original")]
    hold=financial.loc[financial.policy.eq("buy_hold")]
    join=["algorithm","market","seed","run_id","period","cost"]
    quality=original.merge(hold[join+["net_pnl"]],on=join,suffixes=("","_buy_hold"),validate="one_to_one")
    quality["excess_pnl"]=quality.net_pnl-quality.net_pnl_buy_hold
    quality["validation_success"]=(quality.excess_pnl>0)&(quality.sharpe>0)
    quality.to_csv(out/"policy_quality.csv",index=False)
    selected_finance=financial.loc[~financial.policy.isin(["original","buy_hold"])].merge(original[join+["net_return","turnover_notional","action_switch_rate"]],on=join,suffixes=("","_original"),validate="many_to_one")
    for key in ("net_return","turnover_notional","action_switch_rate"):
        selected_finance["delta_"+key]=selected_finance[key]-selected_finance[key+"_original"]
    economic=selected_finance.groupby(["period","cost","algorithm","market","policy"])[["delta_net_return","delta_turnover_notional","delta_action_switch_rate"]].mean().groupby(["period","cost","algorithm","market"]).mean().groupby(["period","cost","algorithm"]).mean().groupby(["period","cost"]).mean().reset_index()
    economic.to_csv(out/"economic_diagnostics.csv",index=False)
    native["family"]=np.where(native.concept.isin(["trend","high_volatility","sideway"]),"market_state","output_derived")
    native.loc[native.concept.ne("sideway")].groupby(["period","family"]).agg(runs=("specificity_gap","count"),mean_gap=("specificity_gap","mean"),mean_size=("size","mean"),ceiling_fraction=("size",lambda x:float((x==8).mean())),positive_fraction=("specificity_gap",lambda x:float((x.dropna()>0).mean()))).reset_index().to_csv(out/"native_diagnostics.csv",index=False)
    selections=[]
    for a in cfg["algorithms"]:
        for m in cfg["markets"]:selections.extend(json.loads((root/"selection"/f"{a}_{m}"/"selection.json").read_text(encoding="utf-8")))
    eligible=sum(c["status"]=="completed" for r in selections for c in r["concepts"].values())
    facts=dict(completed_at=now(),policies=600,algorithms=4,markets=15,seeds=10,nominal_concept_runs=3600,
               eligible_concept_runs=eligible,periods=cfg["periods"],counts=nominal,
               validation_success_policies=int(quality.loc[quality.period.eq("validation"),"validation_success"].sum()),
               pretest_freeze_sha256=sha256(root/"pretest_freeze.json"),input_transfer_hashes=hashes,
               methods=cfg["alignment_methods"],bootstrap_repeats=cfg["bootstrap_repeats"],
               completed_groups=60,failed_groups=0,failed_rows=0,
               inference_family_rule="BH jointly across years within endpoint family and resampling scheme; algorithm contrasts are a separate family",
               sensitivity=pd.DataFrame(point_rows).to_dict("records"))
    write_json(out/"facts.json",facts)
    print("AGGREGATION_COMPLETE",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=CONFIG);args=parser.parse_args()
    with threadpool_limits(limits=1):aggregate(json.loads(args.config.read_text(encoding="utf-8")))
