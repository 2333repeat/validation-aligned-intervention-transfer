from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def resolve_device(requested: str | None = None) -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def choose_metric(estimator, metric_name: str) -> float:
    name = metric_name.lower()
    if name == "sharpe":
        return float(estimator.sharpeRatio)
    if name == "cr":
        return float(estimator.CR)
    return float(estimator.PnL)


def choose_start_index(env, random_reset: bool) -> Optional[int]:
    if not random_reset or not hasattr(env, "window_size") or not hasattr(env, "terminal_date"):
        return None
    lo = int(getattr(env, "window_size", 20)) - 1
    hi = int(getattr(env, "terminal_date", lo + 1)) - int(getattr(env, "window_size", 20))
    if hi <= lo:
        return None
    return int(np.random.randint(lo, hi + 1))


@dataclass
class EvalResult:
    metric: float
    pnl: float
    sharpe: float
    cr: float
