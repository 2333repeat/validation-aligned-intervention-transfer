from __future__ import annotations

import numpy as np
import pandas as pd


def _past_returns(df: pd.DataFrame, end_index: int, lookback: int) -> np.ndarray:
    start = max(0, end_index + 1 - lookback)
    closes = df["Close"].iloc[start : end_index + 1].astype(float).to_numpy()
    if len(closes) < 2:
        return np.array([0.0], dtype=np.float64)
    return np.diff(closes) / np.maximum(closes[:-1], 1e-12)


def fit_regime_thresholds(
    data: pd.DataFrame,
    train_end_index: int,
    lookback: int = 20,
    threshold_quantile: float = 0.7,
) -> dict[str, float]:
    vols = []
    trends = []
    capped_end = min(train_end_index, len(data) - 1)
    for t in range(max(1, lookback - 1), capped_end + 1):
        returns = _past_returns(data, t, lookback)
        vols.append(float(np.std(returns)))
        trends.append(abs(float(np.sum(returns))))
    if not vols:
        vols = [0.0]
    if not trends:
        trends = [0.0]
    return {
        "volatility": float(np.quantile(vols, threshold_quantile)),
        "trend": float(np.quantile(trends, threshold_quantile)),
    }


def label_market_regime(
    data: pd.DataFrame,
    t: int,
    thresholds: dict[str, float],
    lookback: int = 20,
) -> dict[str, bool]:
    returns = _past_returns(data, min(t, len(data) - 1), lookback)
    volatility = float(np.std(returns))
    trend_strength = abs(float(np.sum(returns)))
    high_volatility = volatility > thresholds.get("volatility", 0.0)
    trend = trend_strength > thresholds.get("trend", 0.0)
    return {
        "high_volatility": bool(high_volatility),
        "trend": bool(trend),
        "sideway": bool(not trend),
    }
