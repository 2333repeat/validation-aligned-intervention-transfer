from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "algorithm",
    "market",
    "concept",
    "source_run_id",
    "target_run_id",
)


def _validated_frame(
    frame: pd.DataFrame,
    metric: str,
    excluded_concepts: Iterable[str] = (),
) -> pd.DataFrame:
    required = set(IDENTITY_COLUMNS) | {metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"transfer frame is missing columns: {sorted(missing)}")
    excluded = {str(value) for value in excluded_concepts}
    selected = frame.loc[~frame["concept"].astype(str).isin(excluded)].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.loc[np.isfinite(selected[metric])].copy()
    if selected.empty:
        raise ValueError("no finite transfer rows remain after filtering")
    return selected


def _cell_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.groupby(["algorithm", "market", "concept"], sort=True, as_index=False)[
            metric
        ]
        .mean()
    )


def _algorithm_market_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    cells = _cell_table(frame, metric)
    return (
        cells.groupby(["algorithm", "market"], sort=True, as_index=False)[metric]
        .mean()
    )


def hierarchical_transfer_estimate(
    frame: pd.DataFrame,
    metric: str,
    excluded_concepts: Iterable[str] = (),
) -> float:
    """Equal-weight concepts within markets, markets within algorithms, then algorithms."""
    selected = _validated_frame(frame, metric, excluded_concepts)
    algorithm_market = _algorithm_market_table(selected, metric)
    algorithm_means = algorithm_market.groupby("algorithm", sort=True)[metric].mean()
    return float(algorithm_means.mean())


def collapse_reciprocal_dyads(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = _validated_frame(frame, metric)
    endpoints = [
        sorted((str(source), str(target)))
        for source, target in zip(
            selected["source_run_id"], selected["target_run_id"]
        )
    ]
    selected["dyad_source"] = [values[0] for values in endpoints]
    selected["dyad_target"] = [values[1] for values in endpoints]
    selected["dyad_id"] = [f"{values[0]}|{values[1]}" for values in endpoints]
    return (
        selected.groupby(
            [
                "algorithm",
                "market",
                "concept",
                "dyad_source",
                "dyad_target",
                "dyad_id",
            ],
            sort=True,
            as_index=False,
        )[metric]
        .mean()
    )


def complete_cell_transfer_frame(
    frame: pd.DataFrame,
    metric: str,
    excluded_concepts: Iterable[str] = (),
) -> pd.DataFrame:
    """Keep cells attaining the maximum observed directed-transfer coverage."""
    selected = _validated_frame(frame, metric, excluded_concepts)
    cell_columns = ["algorithm", "market", "concept"]
    counts = (
        selected.groupby(cell_columns, sort=True)
        .size()
        .rename("directed_transfers")
        .reset_index()
    )
    expected = int(counts["directed_transfers"].max())
    complete = counts.loc[counts["directed_transfers"] == expected, cell_columns]
    output = selected.merge(complete, on=cell_columns, how="inner", validate="many_to_one")
    output.attrs["complete_cell_directions"] = expected
    return output


def _market_first_samples(
    frame: pd.DataFrame,
    metric: str,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    table = _algorithm_market_table(frame, metric)
    markets = sorted(table["market"].astype(str).unique().tolist())
    algorithms = sorted(table["algorithm"].astype(str).unique().tolist())
    values = (
        table.pivot(index="algorithm", columns="market", values=metric)
        .reindex(index=algorithms, columns=markets)
        .to_numpy(dtype=float)
    )
    chosen = rng.integers(
        0,
        len(markets),
        size=(int(repeats), len(markets)),
    )
    sampled = values[:, chosen]
    return np.nanmean(np.nanmean(sampled, axis=2), axis=0)


def _dyadic_samples(
    frame: pd.DataFrame,
    metric: str,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dyads = collapse_reciprocal_dyads(frame, metric)
    markets = sorted(dyads["market"].astype(str).unique().tolist())
    algorithms = sorted(dyads["algorithm"].astype(str).unique().tolist())
    algorithm_market_samples = {}
    for (algorithm, market), values in dyads.groupby(
        ["algorithm", "market"], sort=True
    ):
        matrix = values.pivot(
            index="dyad_id",
            columns="concept",
            values=metric,
        ).to_numpy(dtype=float)
        chosen = rng.integers(
            0,
            len(matrix),
            size=(int(repeats), len(matrix)),
        )
        sampled = matrix[chosen]
        finite = np.isfinite(sampled)
        concept_sums = np.where(finite, sampled, 0.0).sum(axis=1)
        concept_counts = finite.sum(axis=1)
        concept_means = np.divide(
            concept_sums,
            concept_counts,
            out=np.full_like(concept_sums, np.nan, dtype=float),
            where=concept_counts > 0,
        )
        algorithm_market_samples[(str(algorithm), str(market))] = np.nanmean(
            concept_means,
            axis=1,
        )
    chosen_markets = rng.integers(
        0,
        len(markets),
        size=(int(repeats), len(markets)),
    )
    repeat_indices = np.arange(int(repeats))[:, None]
    algorithm_samples = []
    for algorithm in algorithms:
        matrix = np.column_stack(
            [
                algorithm_market_samples.get(
                    (algorithm, market),
                    np.full(int(repeats), np.nan),
                )
                for market in markets
            ]
        )
        algorithm_samples.append(
            np.nanmean(matrix[repeat_indices, chosen_markets], axis=1)
        )
    return np.nanmean(np.vstack(algorithm_samples), axis=0)


def _multiway_samples(
    frame: pd.DataFrame,
    metric: str,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    markets = sorted(frame["market"].astype(str).unique().tolist())
    algorithms = sorted(frame["algorithm"].astype(str).unique().tolist())
    policies = sorted(
        set(frame["source_run_id"].astype(str))
        | set(frame["target_run_id"].astype(str))
    )
    policy_index = {value: index for index, value in enumerate(policies)}
    market_index = {value: index for index, value in enumerate(markets)}
    algorithm_index = {value: index for index, value in enumerate(algorithms)}
    source = np.asarray(
        [policy_index[value] for value in frame["source_run_id"].astype(str)],
        dtype=int,
    )
    target = np.asarray(
        [policy_index[value] for value in frame["target_run_id"].astype(str)],
        dtype=int,
    )
    metric_values = frame[metric].to_numpy(dtype=float)
    cell_keys = list(
        zip(
            frame["algorithm"].astype(str),
            frame["market"].astype(str),
            frame["concept"].astype(str),
        )
    )
    unique_cells = sorted(set(cell_keys))
    cell_index = {value: index for index, value in enumerate(unique_cells)}
    row_cell = np.asarray([cell_index[value] for value in cell_keys], dtype=int)
    algorithm_market_keys = [(algorithm, market) for algorithm, market, _ in unique_cells]
    unique_algorithm_markets = sorted(set(algorithm_market_keys))
    algorithm_market_index = {
        value: index for index, value in enumerate(unique_algorithm_markets)
    }
    cell_algorithm_market = np.asarray(
        [algorithm_market_index[value] for value in algorithm_market_keys], dtype=int
    )
    concepts_per_algorithm_market = np.bincount(
        cell_algorithm_market,
        minlength=len(unique_algorithm_markets),
    ).astype(float)
    algorithm_market_algorithm = np.asarray(
        [algorithm_index[algorithm] for algorithm, _market in unique_algorithm_markets],
        dtype=int,
    )
    algorithm_market_market = np.asarray(
        [market_index[market] for _algorithm, market in unique_algorithm_markets],
        dtype=int,
    )
    output = np.empty(int(repeats), dtype=float)
    for repeat in range(int(repeats)):
        market_draw = rng.exponential(size=len(markets))
        policy_draw = rng.exponential(size=len(policies))
        row_weights = policy_draw[source] * policy_draw[target]
        cell_weight = np.bincount(
            row_cell,
            weights=row_weights,
            minlength=len(unique_cells),
        )
        cell_weighted_sum = np.bincount(
            row_cell,
            weights=row_weights * metric_values,
            minlength=len(unique_cells),
        )
        cell_values = cell_weighted_sum / cell_weight
        algorithm_market_values = np.bincount(
            cell_algorithm_market,
            weights=cell_values,
            minlength=len(unique_algorithm_markets),
        ) / concepts_per_algorithm_market
        selected_market_weights = market_draw[algorithm_market_market]
        algorithm_weighted_sum = np.bincount(
            algorithm_market_algorithm,
            weights=algorithm_market_values * selected_market_weights,
            minlength=len(algorithms),
        )
        algorithm_weight = np.bincount(
            algorithm_market_algorithm,
            weights=selected_market_weights,
            minlength=len(algorithms),
        )
        output[repeat] = float(np.mean(algorithm_weighted_sum / algorithm_weight))
    return output


def _add_one_two_sided_p(samples: np.ndarray) -> float:
    lower = int(np.sum(samples <= 0.0))
    upper = int(np.sum(samples >= 0.0))
    return float(min(1.0, 2.0 * (1 + min(lower, upper)) / (len(samples) + 1)))


def dependence_bootstrap_summary(
    frame: pd.DataFrame,
    metric: str,
    repeats: int = 20_000,
    random_state: int = 1,
    excluded_concepts: Iterable[str] = (),
) -> list[dict]:
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    excluded = tuple(str(value) for value in excluded_concepts)
    selected = _validated_frame(frame, metric, excluded)
    estimate = hierarchical_transfer_estimate(selected, metric)
    rng = np.random.default_rng(random_state)
    generators = (
        ("market_first", _market_first_samples),
        ("dyadic", _dyadic_samples),
        ("multiway_pigeonhole", _multiway_samples),
    )
    dyads = collapse_reciprocal_dyads(selected, metric)
    policies = set(selected["source_run_id"].astype(str)) | set(
        selected["target_run_id"].astype(str)
    )
    rows = []
    for method, generator in generators:
        samples = generator(selected, metric, int(repeats), rng)
        rows.append(
            {
                "bootstrap_method": method,
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "p_value": _add_one_two_sided_p(samples),
                "repeats": int(repeats),
                "random_state": int(random_state),
                "markets": int(selected["market"].nunique()),
                "algorithms": int(selected["algorithm"].nunique()),
                "concepts": int(selected["concept"].nunique()),
                "policies": int(len(policies)),
                "dyads": int(len(dyads)),
                "transfer_rows": int(len(selected)),
                "excluded_concepts": ",".join(excluded),
            }
        )
    return rows
