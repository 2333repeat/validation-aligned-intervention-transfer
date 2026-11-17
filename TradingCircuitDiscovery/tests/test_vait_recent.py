"""针对批量加速、时序隔离与真实冻结网络的科学回归检查。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from TradingCircuitDiscovery.evaluation.causality_test import compare_action_score_effect, compare_transfer_profiles
from TradingCircuitDiscovery.evaluation.alignment_methods import fit_alignment_method, ALIGNMENT_METHODS
from TradingCircuitDiscovery.experiments import vait_recent_core as core
from TradingCircuitDiscovery.experiments.prepare_vait_recent import CONFIG


def test_batch_metrics_match_reference_and_degenerate_cases():
    rng = np.random.default_rng(13)
    base = rng.dirichlet([1,1,1], size=29)
    native = rng.dirichlet([1,1,1], size=29)
    candidates = rng.dirichlet([1,1,1], size=(12,29))
    candidates[0] = base
    for reference in (native, base):
        metrics = core.batch_metrics(base, candidates, reference)
        for i, candidate in enumerate(candidates):
            expected = {**compare_action_score_effect(base,candidate,True), **compare_transfer_profiles(base,candidate,reference,True)}
            for key, values in metrics.items():
                target = expected[key]
                if target is None:
                    assert np.isnan(values[i])
                else:
                    assert values[i] == pytest.approx(target, abs=1e-12)


def test_cached_maps_match_existing_eight_methods():
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    left = rng.normal(size=(80,16)); right = rng.normal(size=(80,16))
    ls, rs = rng.normal(size=(16,24)), rng.normal(size=(16,24))
    fit = core.FrozenMaps(left,right,ls,rs)
    for indices in ([0,1], [1,4,7,11]):
        for method in ALIGNMENT_METHODS:
            expected = fit_alignment_method(method,left,right,indices,reference_weight_signatures=ls,target_weight_signatures=rs)
            assert fit.indices(method, indices) == expected["target_indices"], method


@pytest.mark.parametrize("algorithm", ["A2C", "PPO", "DDQN", "SAC"])
def test_actual_checkpoint_batched_intervention_matches_hooks(algorithm):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    adapter, _ = core.load_adapter(cfg, algorithm, "AAPL", 1)
    states = core.load_period(cfg,"AAPL","validation")["states"][:31]
    discovery = core.load_period(cfg,"AAPL","discovery")["states"]
    for layer in core.layers(algorithm):
        means = adapter.get_hidden(discovery)[layer].numpy().mean(axis=0)
        engine = core.BatchedIntervention(adapter,states,layer,means)
        sets = [[0], [1,3], [0,2,4,6,8,10,12,14]]
        actual = engine.probabilities_many(sets, batch_size=2)
        np.testing.assert_allclose(engine.baseline, adapter.get_action_distribution(states).numpy(), atol=2e-6, rtol=2e-5)
        for indices, result in zip(sets, actual):
            with adapter.intervene_hidden(layer, indices, means[indices]):
                expected = adapter.get_action_distribution(states).numpy()
            np.testing.assert_allclose(result,expected,atol=2e-6,rtol=2e-5)
        np.testing.assert_array_equal(adapter.get_action_distribution(states).numpy(), engine.baseline)


def test_selection_never_opens_test_periods(tmp_path, monkeypatch):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    original_cfg = dict(cfg)
    cfg["output_root"] = str(tmp_path)
    cfg["seeds"] = [1]
    (tmp_path/"protocol.json").write_text(json.dumps(cfg),encoding="utf-8")
    original_load = core.load_period
    opened = []
    def guarded(_cfg, market, period):
        assert period in {"discovery", "validation"}, "test period accessed during selection"
        opened.append(period)
        return original_load(original_cfg,market,period)
    monkeypatch.setattr(core,"load_period",guarded)
    core.select_group(cfg,"PPO","AAPL")
    assert opened == ["discovery","validation"]
    output = json.loads((tmp_path/"selection/PPO_AAPL/selection.json").read_text(encoding="utf-8"))
    assert len(output) == 1
    assert any(v["status"]=="completed" for v in output[0]["concepts"].values())


def test_accounting_matches_original_environment_execution():
    import sys
    import pandas as pd
    from TradingCircuitDiscovery.experiments.evaluate_vait_recent import account_trace
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    sys.path.insert(0,str(Path(cfg["policy_root"]).resolve()))
    from TradingEnv import TradingEnv
    rng = np.random.default_rng(123)
    prices = 100*np.exp(np.cumsum(rng.normal(0,.01,100)))
    frame = pd.DataFrame(dict(Date=pd.date_range("2020-01-01",periods=100),Open=prices,High=prices*1.01,
                              Low=prices*.99,Close=prices,Volume=np.full(100,10000.)))
    actions = rng.integers(0,3,size=35)
    for cost in [0.,.003,.01]:
        env = TradingEnv(frame,"2020-01-01","2020-04-30")
        env.reset(); env.transitionCost=cost
        start=env.t
        expected=[]
        for action in actions:
            env.step(int(action));expected.append(float(env.account.loc[env.t-1,"Capitals"]))
        _,actual=account_trace(prices[start:start+len(actions)],actions,cost,execute_last=True)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-12)


def test_evaluation_refuses_unfrozen_inputs(tmp_path):
    from TradingCircuitDiscovery.experiments.evaluate_vait_recent import verify_pretest
    with pytest.raises(ValueError,match="must be frozen"):
        verify_pretest({"output_root":str(tmp_path)})


def test_assignment_ties_preserve_optimal_objective():
    from TradingCircuitDiscovery.evaluation.representation_stability import _linear_sum_assignment
    rng=np.random.default_rng(5)
    scores=rng.normal(size=(24,24));scores[8:,:]=0;scores[:,8:]=0
    actual=core.assignment(scores)
    rows,cols=_linear_sum_assignment(-np.abs(scores))
    assert len(set(actual.tolist()))==len(actual)
    assert np.abs(scores)[np.arange(24),actual].sum()==pytest.approx(np.abs(scores)[rows,cols].sum(),abs=1e-12)


def test_future_prices_cannot_change_discovery_observations():
    import pandas as pd
    from TradingCircuitDiscovery.experiments.prepare_vait_recent import build_observations
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    frame=pd.read_csv(Path(cfg["policy_root"])/"Data/AAPL.csv")
    provenance=json.loads((Path(cfg["output_root"])/"data/AAPL/provenance.json").read_text(encoding="utf-8"))
    expected=build_observations(frame,*cfg["periods"]["discovery"],provenance["thresholds"],cfg["policy_root"])
    changed=frame.copy()
    later=pd.to_datetime(changed.Date)>pd.Timestamp("2023-12-31")
    changed.loc[later,["Open","High","Low","Close","Volume"]]*=10000
    actual=build_observations(changed,*cfg["periods"]["discovery"],provenance["thresholds"],cfg["policy_root"])
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])


def test_evaluation_rejects_postfreeze_scientific_changes(tmp_path):
    from TradingCircuitDiscovery.experiments.evaluate_vait_recent import verify_pretest
    frozen={"output_root":str(tmp_path),"random_controls":50}
    (tmp_path/"protocol.json").write_text(json.dumps(frozen),encoding="utf-8")
    (tmp_path/"pretest_freeze.json").write_text(json.dumps({"files":{}}),encoding="utf-8")
    with pytest.raises(ValueError,match="scientific configuration"):
        verify_pretest({**frozen,"random_controls":10})


def test_evaluation_rejects_modified_observation_cache(tmp_path):
    from TradingCircuitDiscovery.experiments.evaluate_vait_recent import verify_pretest
    from TradingCircuitDiscovery.experiments.prepare_vait_recent import sha256
    cfg={"output_root":str(tmp_path)}
    (tmp_path/"protocol.json").write_text(json.dumps(cfg),encoding="utf-8")
    (tmp_path/"pretest_freeze.json").write_text(json.dumps({"files":{}}),encoding="utf-8")
    folder=tmp_path/"data/AAPL";folder.mkdir(parents=True)
    cache=folder/"test_2025.npz";cache.write_bytes(b"frozen")
    records=[{"market":"AAPL","periods":{"test_2025":{"sha256":sha256(cache)}}}]
    (tmp_path/"data_manifest.json").write_text(json.dumps(records),encoding="utf-8")
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError,match="observation data modified"):
        verify_pretest(cfg)
