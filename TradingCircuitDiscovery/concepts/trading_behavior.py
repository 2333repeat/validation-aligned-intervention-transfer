LONG_MOMENTUM = "long_momentum"
SHORT_MOMENTUM = "short_momentum"
MEAN_REVERSION = "mean_reversion"
RISK_OFF = "risk_off"


def classify_trading_behavior(action: int, past_return: float, volatility: float = 0.0, volatility_threshold: float = 1.0) -> str:
    if volatility > volatility_threshold and action == 0:
        return RISK_OFF
    if action == 1:
        return LONG_MOMENTUM if past_return >= 0 else "long_mean_reversion"
    if action == 2:
        return SHORT_MOMENTUM if past_return < 0 else "short_mean_reversion"
    return "hold"
