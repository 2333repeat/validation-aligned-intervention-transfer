def risk_off_label(position: float, volatility: float, volatility_threshold: float) -> bool:
    return bool(position == 0 and volatility >= volatility_threshold)


def drawdown_risk_label(drawdown: float, max_drawdown_threshold: float) -> bool:
    return bool(drawdown <= -abs(max_drawdown_threshold))
