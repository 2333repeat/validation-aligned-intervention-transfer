from __future__ import annotations

from pathlib import Path

import pandas as pd

from TradingCircuitDiscovery.concepts.market_regime import fit_regime_thresholds
from TradingCircuitDiscovery.integrations.policy_value import _add_policy_value_to_path


def temporal_partition(date) -> str:
    year = pd.Timestamp(date).year
    mapping = {2022: "discovery", 2023: "validation", 2024: "test"}
    return mapping.get(year, "excluded")


def fit_training_regime_thresholds(
    data: pd.DataFrame,
    train_end_date: str = "2021-12-31",
    lookback: int = 20,
    threshold_quantile: float = 0.7,
) -> dict[str, float]:
    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    train_indices = frame.index[frame["Date"] <= pd.Timestamp(train_end_date)]
    if len(train_indices) == 0:
        raise ValueError("No market rows are available in the training period")
    return fit_regime_thresholds(
        frame,
        train_end_index=int(train_indices[-1]),
        lookback=lookback,
        threshold_quantile=threshold_quantile,
    )


def load_market_data(project_root: str | Path, market: str) -> pd.DataFrame:
    path = Path(project_root).expanduser().resolve() / "Data" / f"{market}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Market data not found: {path}")
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def build_policy_value_environment(
    project_root: str | Path,
    market: str,
    start_date: str,
    end_date: str,
    init_money: float = 500000.0,
):
    _add_policy_value_to_path(project_root)
    from TradingEnv import TradingEnv

    data = load_market_data(project_root, market)
    return TradingEnv(data, start_date, end_date, init_money=init_money)
