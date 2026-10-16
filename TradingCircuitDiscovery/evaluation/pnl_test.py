from __future__ import annotations

import numpy as np


def evaluate_account_returns(returns, actions=None) -> dict[str, float]:
    r = np.asarray(returns, dtype=np.float64)
    pnl = float(np.prod(1.0 + r) - 1.0)
    std = float(np.std(r))
    sharpe = float(np.mean(r) / (std + 1e-12) * np.sqrt(252)) if len(r) else 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    mdd = float(np.min(equity / np.maximum(peak, 1e-12) - 1.0)) if len(equity) else 0.0
    turnover = 0.0
    if actions is not None:
        a = np.asarray(actions)
        turnover = float(np.mean(a[1:] != a[:-1])) if len(a) > 1 else 0.0
    return {"pnl": pnl, "sharpe": sharpe, "mdd": mdd, "turnover": turnover}


def evaluate_account_frame(account) -> dict[str, float]:
    if "Returns" not in account:
        raise ValueError("Account frame must contain Returns")
    actions = account["Q_Action"].to_numpy() if "Q_Action" in account else None
    metrics = evaluate_account_returns(account["Returns"].to_numpy(), actions=actions)
    metrics["steps"] = int(len(account))
    if "Capitals" in account and len(account):
        metrics["final_capital"] = float(account["Capitals"].iloc[-1])
    return metrics
