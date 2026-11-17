import numpy as np
import pandas as pd


def _row(algorithm, market, concept, source, target, effect):
    return {
        "algorithm": algorithm,
        "market": market,
        "concept": concept,
        "source_run_id": f"{algorithm}-{market}-{source}",
        "target_run_id": f"{algorithm}-{market}-{target}",
        "effect": effect,
    }


def test_same_seed_block_gets_one_weight_not_squared():
    from TradingCircuitDiscovery.evaluation.dependence_bootstrap import _multiway_samples

    class FixedDraws:
        def exponential(self, size):
            return np.ones(1) if size == 1 else np.array([1.0, 3.0])

    frame = pd.DataFrame([
        _row('PPO', 'AAPL', 'trend', 's1', 's1', 0.0),
        _row('PPO', 'AAPL', 'trend', 's2', 's2', 4.0),
    ])
    np.testing.assert_allclose(_multiway_samples(frame, 'effect', 1, FixedDraws()), [3.0])


def test_hierarchical_estimate_equal_weights_concepts_markets_and_algorithms():
    from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
        hierarchical_transfer_estimate,
    )

    frame = pd.DataFrame(
        [
            _row("A2C", "AAPL", "trend", "s1", "s2", 1.0),
            _row("A2C", "AAPL", "trend", "s2", "s1", 3.0),
            _row("A2C", "AAPL", "sideway", "s1", "s2", 100.0),
            _row("A2C", "NVDA", "trend", "s1", "s2", 6.0),
            _row("PPO", "AAPL", "trend", "s1", "s2", 10.0),
            _row("PPO", "AAPL", "trend", "s2", "s1", 10.0),
            _row("PPO", "NVDA", "trend", "s1", "s2", 14.0),
        ]
    )

    estimate = hierarchical_transfer_estimate(
        frame,
        metric="effect",
        excluded_concepts=("sideway",),
    )

    # A2C: mean(market means 2, 6)=4; PPO: mean(10, 14)=12.
    assert np.isclose(estimate, 8.0)
    assert not np.isclose(estimate, frame.loc[frame["concept"] != "sideway", "effect"].mean())


def test_reciprocal_transfers_collapse_to_one_unordered_dyad():
    from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
        collapse_reciprocal_dyads,
    )

    frame = pd.DataFrame(
        [
            _row("PPO", "AAPL", "trend", "s1", "s2", 1.0),
            _row("PPO", "AAPL", "trend", "s2", "s1", 3.0),
            _row("PPO", "AAPL", "trend", "s1", "s3", 5.0),
        ]
    )

    collapsed = collapse_reciprocal_dyads(frame, metric="effect")

    assert len(collapsed) == 2
    assert set(collapsed["dyad_id"]) == {
        "PPO-AAPL-s1|PPO-AAPL-s2",
        "PPO-AAPL-s1|PPO-AAPL-s3",
    }
    reciprocal = collapsed.loc[
        collapsed["dyad_id"] == "PPO-AAPL-s1|PPO-AAPL-s2", "effect"
    ].iloc[0]
    assert np.isclose(reciprocal, 2.0)


def test_dependence_bootstraps_are_deterministic_finite_and_add_one_corrected():
    from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
        dependence_bootstrap_summary,
    )

    rows = []
    for algorithm in ("A2C", "PPO"):
        for market in ("AAPL", "NVDA", "SP500"):
            for concept in ("trend", "sideway"):
                for source, target in (("s1", "s2"), ("s2", "s1"), ("s1", "s3"), ("s3", "s1")):
                    rows.append(_row(algorithm, market, concept, source, target, 2.0))
    frame = pd.DataFrame(rows)

    first = dependence_bootstrap_summary(
        frame,
        metric="effect",
        repeats=20,
        random_state=9,
        excluded_concepts=("sideway",),
    )
    second = dependence_bootstrap_summary(
        frame,
        metric="effect",
        repeats=20,
        random_state=9,
        excluded_concepts=("sideway",),
    )

    assert first == second
    assert {row["bootstrap_method"] for row in first} == {
        "market_first",
        "dyadic",
        "multiway_pigeonhole",
    }
    for row in first:
        assert row["estimate"] == 2.0
        assert row["ci_low"] == 2.0
        assert row["ci_high"] == 2.0
        assert np.isclose(row["p_value"], 2.0 / 21.0)
        assert row["markets"] == 3
        assert row["repeats"] == 20
        assert row["excluded_concepts"] == "sideway"


def test_dyadic_bootstrap_resamples_each_dyad_jointly_across_concepts():
    from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
        _dyadic_samples,
    )

    frame = pd.DataFrame(
        [
            _row("PPO", "AAPL", "trend", "s1", "s2", 0.0),
            _row("PPO", "AAPL", "volatility", "s1", "s2", 100.0),
            _row("PPO", "AAPL", "trend", "s1", "s3", 100.0),
            _row("PPO", "AAPL", "volatility", "s1", "s3", 0.0),
        ]
    )

    samples = _dyadic_samples(
        frame,
        metric="effect",
        repeats=50,
        rng=np.random.default_rng(11),
    )

    np.testing.assert_allclose(samples, 50.0)


def test_complete_cell_filter_keeps_only_maximal_directed_transfer_cells():
    from TradingCircuitDiscovery.evaluation.dependence_bootstrap import (
        complete_cell_transfer_frame,
    )

    rows = [
        _row("PPO", "AAPL", "trend", source, target, 1.0)
        for source, target in (
            ("s1", "s2"),
            ("s2", "s1"),
            ("s1", "s3"),
            ("s3", "s1"),
            ("s2", "s3"),
            ("s3", "s2"),
        )
    ]
    rows.extend(
        [
            _row("PPO", "NVDA", "trend", "s1", "s2", 2.0),
            _row("PPO", "NVDA", "trend", "s2", "s1", 2.0),
        ]
    )

    selected = complete_cell_transfer_frame(pd.DataFrame(rows), metric="effect")

    assert set(selected["market"]) == {"AAPL"}
    assert len(selected) == 6
    assert selected.attrs["complete_cell_directions"] == 6
