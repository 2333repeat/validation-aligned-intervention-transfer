"""600策略VAIT批次：只读冻结权重、独立选择、映射冻结及批量真实干预。"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import yaml

from TradingCircuitDiscovery.discovery.circuit_search import CandidateNeuron, greedy_circuit_search
from TradingCircuitDiscovery.discovery.linear_probe import _fit_model, _metrics
from TradingCircuitDiscovery.discovery.neuron_ranking import rank_neurons_by_activation_difference
from TradingCircuitDiscovery.evaluation import alignment_methods as am
from TradingCircuitDiscovery.evaluation.intervention_robustness import select_minimal_near_max
from TradingCircuitDiscovery.experiments.neuro_response_data import JointIntervention
from TradingCircuitDiscovery.experiments.prepare_vait_recent import sha256, write_json, now
from TradingCircuitDiscovery.models.model_adapter import A2CAdapter, PPOAdapter, DQNAdapter, SACAdapter


def identity(algorithm, market, seed):
    return f"{algorithm}-{market}-update2022-cuda-s{seed}"


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def layers(algorithm):
    return ["policy_net.fc1", "policy_net.fc2"] if algorithm == "DDQN" else ["actor.fc1"]


def load_adapter(cfg, algorithm, market, seed):
    policy = Path(cfg["policy_root"]).resolve()
    if str(policy) not in sys.path:
        sys.path.insert(0, str(policy))
    from agents.A2CAgent import A2C
    from agents.PPOAgent import PPO
    from agents.DQNagent import DDQN
    from agents.SACAgent import SAC
    agent_type, adapter_type = {"A2C": (A2C, A2CAdapter), "PPO": (PPO, PPOAdapter),
                                "DDQN": (DDQN, DQNAdapter), "SAC": (SAC, SACAdapter)}[algorithm]
    settings = yaml.safe_load((policy / "configs" / f"{algorithm}.yaml").read_text(encoding="utf-8-sig"))
    settings["device"] = cfg["device"]
    agent = agent_type(SimpleNamespace(**settings))
    checkpoint = policy / "result" / identity(algorithm, market, seed) / "agent/model/best_model.pth"
    agent.load_model(str(checkpoint))
    return adapter_type(agent, layers=layers(algorithm)), checkpoint


def load_period(cfg, market, period):
    path = Path(cfg["output_root"]) / "data" / market / f"{period}.npz"
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def concepts(data, probabilities):
    actions = probabilities.argmax(axis=1)
    return {**{key: data[key].astype(int) for key in ("trend", "high_volatility", "sideway")},
            "long_action": (actions == 1).astype(int), "short_action": (actions == 2).astype(int),
            "risk_off": ((actions == 0) & data["high_volatility"].astype(bool)).astype(int)}


def margin(probabilities, actions):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 2:
        p = p[None]
    selected = np.take_along_axis(p, np.broadcast_to(actions[None, :, None], (len(p), len(actions), 1)), axis=2)[..., 0]
    mask = np.eye(p.shape[-1], dtype=bool)[actions]
    return selected - np.where(mask[None], -np.inf, p).max(axis=2)


class BatchedIntervention(JointIntervention):
    """缓存前缀后执行原网络后缀，批量只减少Python调用，不线性化网络。"""

    def probabilities_many(self, sets, batch_size=32):
        results = []
        with torch.no_grad():
            for start in range(0, len(sets), batch_size):
                current = sets[start:start + batch_size]
                hidden = self.hidden.unsqueeze(0).expand(len(current), -1, -1).clone()
                for j, indices in enumerate(current):
                    hidden[j, :, indices] = self.means[indices]
                values = torch.softmax(self._suffix(hidden), dim=-1).cpu().numpy()
                results.append(values)
        return np.concatenate(results)


def select_group(cfg, algorithm, market):
    """函数只打开discovery/validation缓存，无法接收测试观察。"""
    root = Path(cfg["output_root"])
    out = root / "selection" / f"{algorithm}_{market}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text(encoding="utf-8"))
    discovery = load_period(cfg, market, "discovery")
    validation = load_period(cfg, market, "validation")
    output, activation_cache = [], {}
    for seed in cfg["seeds"]:
        adapter, checkpoint = load_adapter(cfg, algorithm, market, seed)
        discovery_probs = adapter.get_action_distribution(discovery["states"]).numpy()
        validation_probs = adapter.get_action_distribution(validation["states"]).numpy()
        discovery_labels = concepts(discovery, discovery_probs)
        validation_labels = concepts(validation, validation_probs)
        hidden_d = adapter.get_hidden(discovery["states"])
        hidden_v = adapter.get_hidden(validation["states"])
        result = dict(run_id=identity(algorithm, market, seed), algorithm=algorithm, market=market, seed=seed,
                      checkpoint_sha256=sha256(checkpoint), concepts={}, means={})
        per_layer = {}
        for layer in layers(algorithm):
            d, v = hidden_d[layer].numpy(), hidden_v[layer].numpy()
            mean = d.mean(axis=0)
            result["means"][layer] = mean.tolist()
            activation_cache[f"s{seed}_{layer}"] = v
            engine = BatchedIntervention(adapter, validation["states"], layer, mean)
            original_margin = margin(engine.baseline, engine.baseline.argmax(axis=1))[0]
            per_layer[layer] = {}
            for concept in cfg["concepts"]:
                yd, yv = discovery_labels[concept], validation_labels[concept]
                if len(np.unique(yd)) != 2:
                    per_layer[layer][concept] = dict(status="ineligible", reason="discovery_missing_class")
                    continue
                probe = _fit_model(d.astype(np.float64), yd, seed)
                probe_metrics = _metrics(yv, probe.predict(v.astype(np.float64)), probe.predict_proba(v.astype(np.float64))[:, 1])
                # 单类验证AUROC未定义；层选择仍沿用balanced_accuracy。
                probe_metrics = {k: float(x) if np.isfinite(x) else None for k, x in probe_metrics.items()}
                ranking = rank_neurons_by_activation_difference(d, yd, min(cfg["top_candidates"], d.shape[1]))
                candidates = [CandidateNeuron(layer, x.index, x.score) for x in ranking]
                def effect_fn(selected):
                    probabilities = engine.probabilities([x.index for x in selected])
                    return float(np.abs(margin(probabilities, engine.baseline.argmax(axis=1))[0] - original_margin).mean())
                searches = []
                for size in cfg["circuit_sizes"]:
                    for gain in cfg["min_relative_gains"]:
                        search = greedy_circuit_search(candidates, effect_fn, size, gain)
                        searches.append(dict(indices=[x.index for x in search.neurons], validation_effect=float(search.effect),
                                             max_size=size, min_relative_gain=gain))
                final = select_minimal_near_max(searches, cfg["near_max_fraction"])
                per_layer[layer][concept] = dict(status="completed", layer=layer, probe=probe_metrics,
                    discovery_prevalence=float(yd.mean()), validation_prevalence=float(yv.mean()),
                    ranking=[x.index for x in ranking], sensitivity=searches, final=final)
        for concept in cfg["concepts"]:
            valid = [per_layer[layer][concept] for layer in layers(algorithm) if per_layer[layer][concept]["status"] == "completed"]
            result["concepts"][concept] = (max(valid, key=lambda x: x["probe"]["balanced_accuracy"]) if valid
                                            else dict(status="ineligible", reason="discovery_missing_class"))
        result["all_layers"] = per_layer
        output.append(result)
    write_json(out / "selection.json", output)
    np.savez_compressed(out / "validation_activations.npz", **activation_cache)
    complete = dict(stage="selection", algorithm=algorithm, market=market, models=len(output), completed_at=now(),
                    selection_sha256=sha256(out/"selection.json"), activations_sha256=sha256(out/"validation_activations.npz"),
                    protocol_sha256=sha256(root/"protocol.json"))
    write_json(out / "complete.json", complete)
    return complete


def correlation(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=0, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=0, keepdims=True), 1e-12)
    return left.T @ right


def assignment(similarity, absolute=True):
    objective = np.abs(similarity) if absolute else similarity
    source, target = linear_sum_assignment(-objective)
    mapping = np.empty(len(source), dtype=int)
    mapping[source] = target
    return mapping


class FrozenMaps:
    """一个策略对/层拟合一次；按原VAIT规则把完整映射投影为集合。"""
    def __init__(self, reference, target, source_signatures, target_signatures):
        self.left, self.right = am._paired_matrices(reference, target)
        corr = correlation(self.left, self.right)
        self.corr = corr
        self.direct = {"identity": np.arange(self.left.shape[1]), "hungarian_abs": assignment(corr),
                       "hungarian_signed": assignment(corr, False)}
        ls, rs = am._standardize(source_signatures), am._standardize(target_signatures)
        self.direct["weight_matching"] = assignment(correlation(ls.T, rs.T), False)
        u, singular, vt = torch.linalg.svd(torch.from_numpy(self.left.T @ self.right), full_matrices=False)
        self.dense = {"procrustes": (u @ vt).numpy()}
        self.dense["cca"], cca_meta = am._cca_map(self.left, self.right, 16)
        self.dense["ridge"], ridge_meta = am._ridge_map(self.left, self.right, (1e-4, 1e-2, 1., 100.), .25)
        self.transport, iterations = am._sinkhorn_plan(1. - np.abs(corr), .05)
        self.metadata = dict(cca=cca_meta, ridge=ridge_meta, ot_iterations=iterations,
                             assignment_solver="scipy.optimize.linear_sum_assignment",
                             tie_rule="solver-deterministic maximum objective; no perturbation")
        chosen = np.abs(corr)[np.arange(len(self.direct["hungarian_abs"])), self.direct["hungarian_abs"]]
        self.quality = dict(mean_absolute_alignment_correlation=float(chosen.mean()), min_absolute_alignment_correlation=float(chosen.min()))
        self.null_cache = {}
        self.block_maps = []
        for block in np.array_split(np.arange(len(self.left)), 4):
            keep = np.ones(len(self.left), dtype=bool); keep[block] = False
            bl, br = am._paired_matrices(self.left[keep], self.right[keep])
            self.block_maps.append(assignment(correlation(bl, br)))

    def indices(self, method, indices):
        if method in self.direct:
            return sorted(self.direct[method][indices].tolist())
        if method == "optimal_transport":
            return am._top_k(self.transport[indices].sum(axis=0), len(indices))
        return am._top_k(np.sqrt(np.square(self.dense[method][indices]).sum(axis=0)), len(indices))

    def nulls(self, indices, repeats, seed):
        shifts = np.random.default_rng(seed).integers(1, len(self.left), size=repeats)
        for shift in set(shifts):
            if int(shift) not in self.null_cache:
                self.null_cache[int(shift)] = assignment(correlation(self.left, np.roll(self.right, int(shift), axis=0)))
        return [sorted(self.null_cache[int(s)][indices].tolist()) for s in shifts], shifts.tolist()


def map_group(cfg, algorithm, market):
    root = Path(cfg["output_root"])
    source_dir = root / "selection" / f"{algorithm}_{market}"
    out = root / "mapping" / f"{algorithm}_{market}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text(encoding="utf-8"))
    selections = json.loads((source_dir / "selection.json").read_text(encoding="utf-8"))
    with np.load(source_dir / "validation_activations.npz") as values:
        activations = {key: values[key] for key in values.files}
    signatures = {}
    for r in selections:
        checkpoint = Path(cfg["policy_root"]) / "result" / r["run_id"] / "agent/model/best_model.pth"
        for layer in layers(algorithm):
            signatures[(r["seed"], layer)] = am.checkpoint_weight_signatures(checkpoint, layer)
    records, fit_meta = [], {}
    for target in selections:
        for source in selections:
            if source["seed"] == target["seed"]:
                continue
            fits = {}
            for concept in cfg["concepts"]:
                row = dict(algorithm=algorithm, market=market, concept=concept, source_seed=source["seed"], target_seed=target["seed"],
                           source_run_id=source["run_id"], target_run_id=target["run_id"])
                src, tgt = source["concepts"][concept], target["concepts"][concept]
                if tgt["status"] != "completed":
                    row.update(status="target_ineligible"); records.append(row); continue
                if src["status"] != "completed":
                    row.update(status="source_ineligible"); records.append(row); continue
                if src["layer"] != tgt["layer"]:
                    row.update(status="layer_mismatch"); records.append(row); continue
                layer = src["layer"]
                if layer not in fits:
                    fits[layer] = FrozenMaps(activations[f"s{source['seed']}_{layer}"], activations[f"s{target['seed']}_{layer}"],
                                            signatures[(source["seed"], layer)], signatures[(target["seed"], layer)])
                    fit_meta[f"{source['seed']}_{target['seed']}_{layer}"] = fits[layer].metadata
                fitted = fits[layer]
                selected = sorted(src["final"]["indices"])
                width = fitted.left.shape[1]
                rng = np.random.default_rng(stable_seed(source["run_id"], target["run_id"], concept, "membership_control"))
                random_sets = [sorted(rng.choice(width, len(selected), replace=False).tolist()) for _ in range(cfg["random_controls"])]
                null_sets, shifts = fitted.nulls(selected, cfg["alignment_null_controls"], stable_seed(source["run_id"], target["run_id"], concept, "alignment_null"))
                mapping = {method: fitted.indices(method, selected) for method in cfg["alignment_methods"]}
                full_set = set(mapping["hungarian_abs"])
                block_sets = [set(m[selected].tolist()) for m in fitted.block_maps]
                row.update(status="completed", layer=layer, source_indices=selected, target_indices=sorted(tgt["final"]["indices"]),
                           mappings=mapping, random_sets=random_sets, alignment_null_sets=null_sets, alignment_null_shifts=shifts,
                           mapping_jaccard=float(np.mean([len(full_set & s)/len(full_set | s) for s in block_sets])),
                           mapping_exact=float(np.mean([full_set == s for s in block_sets])), **fitted.quality)
                records.append(row)
    path = out / "mappings.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(dict(records=records, metadata=fit_meta), f, separators=(",", ":"), allow_nan=False)
    statuses = {status: sum(r["status"] == status for r in records) for status in sorted({r["status"] for r in records})}
    complete = dict(stage="mapping", algorithm=algorithm, market=market, completed_at=now(), nominal_rows=len(records),
                    counts=statuses, mapping_sha256=sha256(path), protocol_sha256=sha256(root/"protocol.json"))
    write_json(out / "complete.json", complete)
    return complete


def freeze_pretest(cfg):
    root = Path(cfg["output_root"])
    files = {}
    for algorithm in cfg["algorithms"]:
        for market in cfg["markets"]:
            group = f"{algorithm}_{market}"
            for stage, names in (("selection", ["selection.json", "validation_activations.npz"]), ("mapping", ["mappings.json.gz"])):
                status = json.loads((root/stage/group/"complete.json").read_text(encoding="utf-8"))
                if status["protocol_sha256"] != sha256(root/"protocol.json"):
                    raise ValueError("pretest protocol hash changed")
                for name in names:
                    path = root/stage/group/name
                    files[str(path.relative_to(root))] = sha256(path)
    for name in ("protocol.json", "model_manifest.json", "data_manifest.json"):
        files[name] = sha256(root/name)
    payload = dict(frozen_at=now(), files=files, selected_policies=600, directed_policy_pairs=5400,
                   code_sha256=sha256(Path(__file__)), no_test_inference_during_selection=True)
    destination = root/"pretest_freeze.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous["files"] != files:
            raise ValueError("pretest inputs changed after freeze")
        return previous
    write_json(destination, payload)
    return payload


PROFILE_KEYS = ["signed_margin_correlation_to_native", "signed_margin_sign_agreement_to_native",
                "conditional_transition_agreement_to_native", "transition_jsd_to_native",
                "intervened_distribution_jsd_to_native", "probability_shift_cosine_to_native"]


def batch_metrics(original, candidates, native):
    """向量化既有compare_action_score_effect/compare_transfer_profiles的定义。"""
    original, candidates, native = [np.asarray(x, dtype=np.float64) for x in (original, candidates, native)]
    if candidates.ndim == 2:
        candidates = candidates[None]
    actions = original.argmax(axis=1)
    action_after = candidates.argmax(axis=2)
    native_after = native.argmax(axis=1)
    signed = margin(candidates, actions) - margin(original, actions)
    native_signed = (margin(native, actions) - margin(original, actions))[0]
    p = np.maximum(original, 0); p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    q = np.maximum(candidates, 0); q /= np.maximum(q.sum(axis=2, keepdims=True), 1e-12)
    n = np.maximum(native, 0); n /= np.maximum(n.sum(axis=1, keepdims=True), 1e-12)
    def jsd(a, b):
        mid = .5 * (a + b)
        return .5 * (a*np.log((a+1e-12)/(mid+1e-12)) + b*np.log((b+1e-12)/(mid+1e-12))).sum(axis=-1)
    centered = signed - signed.mean(axis=1, keepdims=True)
    centered_n = native_signed - native_signed.mean()
    denominators = np.linalg.norm(centered, axis=1) * np.linalg.norm(centered_n)
    corr = np.divide(centered @ centered_n, denominators, out=np.full(len(q), np.nan),
                     where=(signed.std(axis=1)>1e-12) & (native_signed.std()>1e-12))
    mask = np.maximum(np.abs(signed), np.abs(native_signed)[None]) > 1e-12
    sign = np.divide(((np.sign(signed)==np.sign(native_signed)[None]) & mask).sum(axis=1), mask.sum(axis=1),
                     out=np.full(len(q), np.nan), where=mask.sum(axis=1)>0)
    transition_mask = native_after != actions
    transition_agreement = ((action_after[:, transition_mask]==native_after[transition_mask]).mean(axis=1)
                             if transition_mask.any() else np.full(len(q), np.nan))
    counts = np.stack([np.bincount(actions*3+a, minlength=9) for a in action_after]).astype(float)
    native_counts = np.bincount(actions*3+native_after, minlength=9).astype(float)
    counts /= counts.sum(axis=1, keepdims=True); native_counts /= native_counts.sum()
    shift, native_shift = q-p[None], n-p
    shift_den = np.linalg.norm(shift.reshape(len(q), -1), axis=1)*np.linalg.norm(native_shift)
    cosine = np.divide((shift*native_shift[None]).sum(axis=(1,2)), shift_den,
                       out=np.full(len(q), np.nan), where=shift_den>1e-12)
    return dict(mean_absolute_margin_change=np.abs(signed).mean(axis=1), mean_signed_margin_change=signed.mean(axis=1),
                action_flip_rate=(action_after != actions[None]).mean(axis=1), mean_jsd=jsd(q,p[None]).mean(axis=1),
                signed_margin_correlation_to_native=corr, signed_margin_sign_agreement_to_native=sign,
                conditional_transition_agreement_to_native=transition_agreement, transition_jsd_to_native=jsd(counts,native_counts[None]),
                intervened_distribution_jsd_to_native=jsd(q,n[None]).mean(axis=1), probability_shift_cosine_to_native=cosine)
