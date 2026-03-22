"""Gold Multi-TF NYSE Scalper — TC baby V1.0 port to Python.

A rule-based gold scalping engine with:
- 5-timeframe EMA bias stack (D/4H/1H/15m/5m)
- RSI + engulfing candle entry confirmation
- 4-tier take-profit ladder with trailing stop
- Runner contract exits on bias flip
- NYSE session filter + daily loss circuit breaker
- Broker-agnostic design via adapter interface
"""

__version__ = "1.0.0"
