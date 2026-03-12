"""
Crypto Derivatives Sentiment — Funding Rate Fetcher
====================================================
Fetches perpetual futures funding rates from OKX (primary), with Binance
and CoinGlass fallbacks. No API key required.

Usage:
    from crypto_sentiment import get_funding_rate
    rate = get_funding_rate("BTC/USD")  # returns float or None
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

log = logging.getLogger("crypto_sentiment")

# Alpaca symbol → base symbol for API lookups
_SYMBOL_MAP = {
    "BTC/USD": "BTC", "ETH/USD": "ETH", "SOL/USD": "SOL",
    "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL",
}

# Cache: {symbol: (rate, timestamp)}
_cache: dict[str, tuple[float, float]] = {}
_CACHE_TTL = 3600  # 1 hour — funding rates update every 8h


def _fetch_okx(base: str) -> Optional[float]:
    """OKX public API — works from US, no auth needed."""
    inst_id = f"{base}-USDT-SWAP"
    resp = requests.get(
        "https://www.okx.com/api/v5/public/funding-rate",
        params={"instId": inst_id},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("data") and len(data["data"]) > 0:
        return float(data["data"][0]["fundingRate"])
    return None


def _fetch_binance(base: str) -> Optional[float]:
    """Binance futures API — may be geo-blocked in US."""
    binance_sym = f"{base}USDT"
    resp = requests.get(
        "https://fapi.binance.com/fapi/v1/fundingRate",
        params={"symbol": binance_sym, "limit": 1},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data and len(data) > 0:
        return float(data[-1]["fundingRate"])
    return None


def get_funding_rate(symbol: str) -> Optional[float]:
    """Fetch the latest funding rate for a crypto symbol.

    Args:
        symbol: Alpaca or yfinance format (BTC/USD, BTC-USD, etc.)

    Returns:
        Funding rate as a float (e.g., 0.0001 = 0.01%), or None on error.
        Positive = longs pay shorts (bullish crowding).
        Negative = shorts pay longs (bearish crowding).
    """
    base = _SYMBOL_MAP.get(symbol)
    if base is None:
        log.warning("No symbol mapping for %s", symbol)
        return None

    # Check cache
    now = time.time()
    if base in _cache:
        rate, ts = _cache[base]
        if now - ts < _CACHE_TTL:
            return rate

    # Primary: OKX (works from US)
    try:
        rate = _fetch_okx(base)
        if rate is not None:
            _cache[base] = (rate, now)
            return rate
    except Exception as exc:
        log.debug("OKX funding fetch failed for %s: %s", base, exc)

    # Fallback: Binance (may be geo-blocked)
    try:
        rate = _fetch_binance(base)
        if rate is not None:
            _cache[base] = (rate, now)
            return rate
    except Exception as exc:
        log.debug("Binance funding fetch failed for %s: %s", base, exc)

    log.warning("All funding rate sources failed for %s", symbol)
    return None


def is_funding_crowded_long(symbol: str, threshold: float = 0.0003) -> bool:
    """True if funding rate > threshold (longs are crowded, avoid new longs)."""
    rate = get_funding_rate(symbol)
    if rate is None:
        return False  # fail open
    return rate > threshold


def is_funding_crowded_short(symbol: str, threshold: float = -0.0003) -> bool:
    """True if funding rate < threshold (shorts are crowded, avoid new shorts)."""
    rate = get_funding_rate(symbol)
    if rate is None:
        return False  # fail open
    return rate < threshold
