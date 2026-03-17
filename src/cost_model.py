"""
Per-symbol cost model: spread + slippage + fees.

Used by backtester for execution realism and by paper_trader for
cost_threshold validation. Provides stress-test multipliers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class SymbolCosts:
    """Estimated round-trip costs for a symbol."""
    half_spread_bps: float    # half-spread in basis points (one-way)
    slippage_bps: float       # market impact in basis points (one-way)
    fee_bps: float            # broker fee in basis points (one-way, 0 for Alpaca)
    fill_probability: float   # probability of fill at limit price (extended hours)

    @property
    def one_way_bps(self) -> float:
        return self.half_spread_bps + self.slippage_bps + self.fee_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.one_way_bps

    @property
    def round_trip_pct(self) -> float:
        return self.round_trip_bps / 10_000


# ---------------------------------------------------------------------------
# Default cost estimates by asset class
# ---------------------------------------------------------------------------

# Liquid large-cap ETFs (SPY, QQQ, IWM)
_LIQUID_ETF = SymbolCosts(
    half_spread_bps=0.5, slippage_bps=1.0, fee_bps=0.0, fill_probability=0.98
)

# Mid-liquidity ETFs (SMH, SOXX, IGV, XLK, GLD)
_MID_ETF = SymbolCosts(
    half_spread_bps=1.5, slippage_bps=2.0, fee_bps=0.0, fill_probability=0.95
)

# Low-liquidity / EM ETFs (EWT, EEM, MCHI, GDX, SLV)
_LOW_ETF = SymbolCosts(
    half_spread_bps=3.0, slippage_bps=3.0, fee_bps=0.0, fill_probability=0.90
)

# Crypto (Alpaca: no commission but wider spreads)
_CRYPTO_MAJOR = SymbolCosts(
    half_spread_bps=5.0, slippage_bps=5.0, fee_bps=0.0, fill_probability=0.99
)

_CRYPTO_ALT = SymbolCosts(
    half_spread_bps=10.0, slippage_bps=10.0, fee_bps=0.0, fill_probability=0.95
)

# Extended hours multiplier (wider spreads, thinner books)
_EXTENDED_HOURS_MULT = 3.0

# Per-symbol cost map
_SYMBOL_COSTS: Dict[str, SymbolCosts] = {
    # Liquid
    "SPY": _LIQUID_ETF, "QQQ": _LIQUID_ETF, "IWM": _LIQUID_ETF,
    # Mid
    "SMH": _MID_ETF, "SOXX": _MID_ETF, "IGV": _MID_ETF,
    "XLK": _MID_ETF, "GLD": _MID_ETF, "IBIT": _MID_ETF,
    # Low
    "EWT": _LOW_ETF, "EEM": _LOW_ETF, "MCHI": _LOW_ETF,
    "GDX": _LOW_ETF, "SLV": _LOW_ETF,
    # Crypto major
    "BTC/USD": _CRYPTO_MAJOR, "BTC-USD": _CRYPTO_MAJOR,
    "ETH/USD": _CRYPTO_MAJOR, "ETH-USD": _CRYPTO_MAJOR,
    "SOL/USD": _CRYPTO_MAJOR, "SOL-USD": _CRYPTO_MAJOR,
    # Crypto alt
    "AVAX/USD": _CRYPTO_ALT, "AVAX-USD": _CRYPTO_ALT,
    "LINK/USD": _CRYPTO_ALT, "LINK-USD": _CRYPTO_ALT,
    "DOGE/USD": _CRYPTO_ALT, "DOGE-USD": _CRYPTO_ALT,
    "DOT/USD": _CRYPTO_ALT, "DOT-USD": _CRYPTO_ALT,
    "SUSHI/USD": _CRYPTO_ALT, "SUSHI-USD": _CRYPTO_ALT,
    "ADA/USD": _CRYPTO_ALT, "ADA-USD": _CRYPTO_ALT,
    "CRV/USD": _CRYPTO_ALT, "CRV-USD": _CRYPTO_ALT,
    "AAVE/USD": _CRYPTO_ALT, "AAVE-USD": _CRYPTO_ALT,
    "RENDER/USD": _CRYPTO_ALT, "RENDER-USD": _CRYPTO_ALT,
    # Extended crypto universe (Layer 1 selector candidates)
    "NEAR/USD": _CRYPTO_ALT, "NEAR-USD": _CRYPTO_ALT,
    "UNI/USD": _CRYPTO_ALT, "UNI-USD": _CRYPTO_ALT,
    "LTC/USD": _CRYPTO_MAJOR, "LTC-USD": _CRYPTO_MAJOR,
    "ARB/USD": _CRYPTO_ALT, "ARB-USD": _CRYPTO_ALT,
    "OP/USD": _CRYPTO_ALT, "OP-USD": _CRYPTO_ALT,
    "FIL/USD": _CRYPTO_ALT, "FIL-USD": _CRYPTO_ALT,
    "APT/USD": _CRYPTO_ALT, "APT-USD": _CRYPTO_ALT,
    "INJ/USD": _CRYPTO_ALT, "INJ-USD": _CRYPTO_ALT,
}


def get_symbol_costs(
    symbol: str,
    session: str = "regular",
    stress_mult: float = 1.0,
) -> SymbolCosts:
    """Get cost estimates for a symbol, adjusted for session and stress.

    For crypto symbols not in the static map, tries the dynamic universe
    screener for per-coin cost data based on actual liquidity.

    Args:
        symbol: Trading symbol
        session: "regular", "extended", or "closed"
        stress_mult: Multiplier for stress testing (e.g., 2.0 = 2x costs)
    """
    if symbol in _SYMBOL_COSTS:
        base = _SYMBOL_COSTS[symbol]
    elif _is_crypto(symbol):
        base = _get_dynamic_crypto_costs(symbol)
    else:
        base = _MID_ETF

    session_mult = _EXTENDED_HOURS_MULT if session == "extended" else 1.0
    total_mult = session_mult * stress_mult

    return SymbolCosts(
        half_spread_bps=base.half_spread_bps * total_mult,
        slippage_bps=base.slippage_bps * total_mult,
        fee_bps=base.fee_bps,
        fill_probability=max(0.5, base.fill_probability ** total_mult),
    )


def _is_crypto(symbol: str) -> bool:
    """Check if symbol looks like a crypto pair."""
    return symbol.endswith("/USD") or symbol.endswith("-USD")


# Cache to avoid re-reading universe.json on every call
_dynamic_cost_cache: dict = {}


def _get_dynamic_crypto_costs(symbol: str) -> SymbolCosts:
    """Get crypto costs from the universe screener's per-coin data."""
    if symbol in _dynamic_cost_cache:
        return _dynamic_cost_cache[symbol]

    try:
        from universe_screener import get_coin_cost_config
        cfg = get_coin_cost_config(symbol)
        costs = SymbolCosts(
            half_spread_bps=cfg["spread_bps"],
            slippage_bps=cfg["slippage_bps"],
            fee_bps=cfg["fee_bps"],
            fill_probability=0.95 if cfg["liquidity_tier"] in ("mega", "large") else 0.90,
        )
    except Exception:
        costs = _CRYPTO_ALT  # fallback

    _dynamic_cost_cache[symbol] = costs
    return costs


def validate_cost_threshold(
    symbol: str,
    cost_threshold: float,
    safety_margin: float = 1.5,
    session: str = "regular",
) -> tuple:
    """Check that cost_threshold is safely above estimated round-trip costs.

    Returns (ok: bool, message: str, estimated_rt_cost: float).
    """
    costs = get_symbol_costs(symbol, session=session)
    rt_cost = costs.round_trip_pct
    min_threshold = rt_cost * safety_margin

    if cost_threshold < min_threshold:
        return (
            False,
            f"{symbol}: cost_threshold={cost_threshold:.4f} < "
            f"min={min_threshold:.4f} (RT cost={rt_cost:.4f} x {safety_margin}x margin)",
            rt_cost,
        )
    return True, "ok", rt_cost


def simulate_fill(
    symbol: str,
    price: float,
    side: str,
    session: str = "regular",
    stress_mult: float = 1.0,
) -> tuple:
    """Simulate realistic fill price including spread and slippage.

    Returns (fill_price: float, filled: bool).
    Used by backtester for execution realism.
    """
    import random

    costs = get_symbol_costs(symbol, session=session, stress_mult=stress_mult)

    # Check fill probability
    if random.random() > costs.fill_probability:
        return price, False

    # Apply spread + slippage
    total_impact_pct = (costs.half_spread_bps + costs.slippage_bps) / 10_000

    if side.upper() in ("BUY", "LONG"):
        fill_price = price * (1 + total_impact_pct)
    else:
        fill_price = price * (1 - total_impact_pct)

    return round(fill_price, 4), True


def adjust_return_for_costs(
    raw_return: float,
    symbol: str,
    session: str = "regular",
) -> float:
    """Adjust a realized return by subtracting estimated round-trip costs.

    Used for cost-aware labels in training.
    """
    costs = get_symbol_costs(symbol, session=session)
    return raw_return - costs.round_trip_pct
