"""
Crypto Sentiment Features — external signals for crypto swing model.
====================================================================

Three free data sources, fetched lazily and cached to disk:

  1. OKX Long/Short Ratio   — contrarian positioning signal
  2. Fear & Greed Index      — broad crypto sentiment (Alternative.me)
  3. Polymarket probabilities — prediction market forward-looking sentiment

All features are daily granularity, aligned to the symbol's bar dates.
Missing/failed fetches gracefully fill with NaN (model handles via fillna).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("crypto_sentiment")

# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------
CRYPTO_SENTIMENT_FEATURES = [
    "okx_long_short_ratio",       # OKX BTC L/S ratio (>1 = net long)
    "okx_ls_zscore",              # L/S ratio z-score vs 30-day rolling
    "fear_greed_index",           # 0-100 (0=extreme fear, 100=extreme greed)
    "fear_greed_zscore",          # z-score vs 30-day rolling mean
    "polymarket_btc_bullish",     # Polymarket BTC price-target "Yes" probability
]

# Cache directory (alongside other data caches)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CACHE_TTL_HOURS = 12  # refresh external data every 12 hours


def get_crypto_sentiment_features() -> List[str]:
    """Return list of crypto sentiment feature names."""
    return list(CRYPTO_SENTIMENT_FEATURES)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"cache_{name}.json")


def _cache_fresh(name: str) -> bool:
    """True if cache file exists and is younger than TTL."""
    path = _cache_path(name)
    if not os.path.exists(path):
        return False
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours < _CACHE_TTL_HOURS


def _cache_read(name: str) -> Optional[dict]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _cache_write(name: str, data: dict) -> None:
    path = _cache_path(name)
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# 1. OKX Long/Short Ratio (public API, no auth, US-accessible)
# ---------------------------------------------------------------------------

def fetch_okx_long_short_ratio(
    currency: str = "BTC",
    period: str = "1D",
    limit: int = 90,
) -> pd.Series:
    """Fetch OKX long/short ratio history.

    Returns pd.Series indexed by date with L/S ratio values.
    L/S > 1 = net long positioning, < 1 = net short.
    """
    cache_name = f"okx_ls_{currency}_{period}"
    if _cache_fresh(cache_name):
        cached = _cache_read(cache_name)
        if cached:
            s = pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
            return s

    log.info("Fetching OKX L/S ratio for %s (period=%s)...", currency, period)
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
    params = {"ccy": currency, "period": period}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0" or not data.get("data"):
            log.warning("OKX L/S ratio: bad response code=%s", data.get("code"))
            return pd.Series(dtype=float)

        rows = data["data"]
        dates = []
        values = []
        for row in rows:
            # Response format: [timestamp_ms, ratio] as list
            ts_ms = int(row[0]) if isinstance(row, list) else int(row.get("ts", 0))
            ratio = float(row[1]) if isinstance(row, list) else float(row.get("longShortRatio", 1.0))
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            dates.append(dt.strftime("%Y-%m-%d"))
            values.append(ratio)

        _cache_write(cache_name, {"dates": dates, "values": values})
        s = pd.Series(values, index=pd.to_datetime(dates))
        log.info("OKX L/S ratio: %d data points fetched", len(s))
        return s

    except Exception as exc:
        log.warning("OKX L/S ratio fetch failed: %s", exc)
        # Try cache even if stale
        cached = _cache_read(cache_name)
        if cached:
            return pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 2. Fear & Greed Index (Alternative.me, public, no auth)
# ---------------------------------------------------------------------------

def fetch_fear_greed_index(limit: int = 365) -> pd.Series:
    """Fetch crypto Fear & Greed Index history.

    Returns pd.Series indexed by date, values 0-100.
    0 = Extreme Fear, 100 = Extreme Greed.
    """
    cache_name = "fear_greed"
    if _cache_fresh(cache_name):
        cached = _cache_read(cache_name)
        if cached:
            s = pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
            return s

    log.info("Fetching Fear & Greed Index (last %d days)...", limit)
    url = "https://api.alternative.me/fng/"
    params = {"limit": limit, "format": "json"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            log.warning("Fear & Greed: empty response")
            return pd.Series(dtype=float)

        dates = []
        values = []
        for item in data:
            ts = int(item.get("timestamp", 0))
            val = int(item.get("value", 50))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            dates.append(dt.strftime("%Y-%m-%d"))
            values.append(val)

        _cache_write(cache_name, {"dates": dates, "values": values})
        s = pd.Series(values, index=pd.to_datetime(dates))
        log.info("Fear & Greed: %d data points fetched", len(s))
        return s

    except Exception as exc:
        log.warning("Fear & Greed fetch failed: %s", exc)
        cached = _cache_read(cache_name)
        if cached:
            return pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 3. Polymarket BTC sentiment (Gamma API, public, no auth)
# ---------------------------------------------------------------------------

def fetch_polymarket_btc_probability() -> pd.Series:
    """Fetch Polymarket BTC price-target market probabilities.

    Searches for active crypto markets related to BTC price milestones
    and returns the "Yes" probability as a bullish sentiment signal.
    If multiple markets exist, uses the highest-volume one.

    Returns pd.Series with a single current-date value (point-in-time).
    Historical data not available — we cache snapshots over time.
    """
    cache_name = "polymarket_btc"

    # Load historical snapshots from cache
    cached = _cache_read(cache_name) or {"dates": [], "values": []}

    # Only fetch new snapshot if cache is stale
    if _cache_fresh(cache_name):
        if cached["dates"]:
            s = pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
            return s

    log.info("Fetching Polymarket BTC markets...")
    url = "https://gamma-api.polymarket.com/events"
    params = {"active": "true", "closed": "false", "tag": "Crypto", "limit": 50}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()

        # Find BTC price-target markets (highest volume)
        best_prob = 0.5  # default neutral
        best_volume = 0

        for event in events:
            for mkt in event.get("markets", []):
                question = (mkt.get("question") or "").lower()
                # Look for BTC price target markets
                if not any(kw in question for kw in ["bitcoin", "btc"]):
                    continue
                if not any(kw in question for kw in ["price", "hit", "reach", "above", "$"]):
                    continue

                volume = float(mkt.get("volume", 0) or 0)
                prices = json.loads(mkt["outcomePrices"]) if mkt.get("outcomePrices") else []
                if not prices:
                    continue

                yes_prob = float(prices[0])
                if volume > best_volume:
                    best_volume = volume
                    best_prob = yes_prob
                    log.info("  Polymarket: '%s' → Yes=%.1f%% (vol=$%.0f)",
                             mkt.get("question", "")[:60], yes_prob * 100, volume)

        # Append today's snapshot
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if cached["dates"] and cached["dates"][0] == today:
            # Update today's value
            cached["values"][0] = best_prob
        else:
            cached["dates"].insert(0, today)
            cached["values"].insert(0, best_prob)

        # Keep last 365 snapshots
        cached["dates"] = cached["dates"][:365]
        cached["values"] = cached["values"][:365]

        _cache_write(cache_name, cached)
        s = pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
        log.info("Polymarket BTC: prob=%.1f%% (vol=$%.0f)", best_prob * 100, best_volume)
        return s

    except Exception as exc:
        log.warning("Polymarket fetch failed: %s", exc)
        if cached["dates"]:
            return pd.Series(cached["values"], index=pd.to_datetime(cached["dates"]))
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Builder class (mirrors CrossAssetFeatureBuilder pattern)
# ---------------------------------------------------------------------------

class CryptoSentimentBuilder:
    """Build crypto sentiment features for swing models.

    Fetches from OKX, Alternative.me, and Polymarket.
    All data is cached to disk with 12-hour TTL.
    Gracefully degrades: if any source fails, that feature fills with NaN.
    """

    def __init__(self):
        self._fetched = False
        self._okx_ls: Optional[pd.Series] = None
        self._fear_greed: Optional[pd.Series] = None
        self._polymarket: Optional[pd.Series] = None

    def _ensure_data(self) -> None:
        """Fetch all data sources (cached, lazy)."""
        if self._fetched:
            return
        self._okx_ls = fetch_okx_long_short_ratio("BTC", "1D", 90)
        self._fear_greed = fetch_fear_greed_index(365)
        self._polymarket = fetch_polymarket_btc_probability()
        self._fetched = True

    def build_features(self, bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Build sentiment features aligned to bar dates.

        Args:
            bars_df: OHLCV DataFrame with 'ts' column
            symbol:  Symbol name (only builds for crypto symbols)

        Returns DataFrame with columns matching CRYPTO_SENTIMENT_FEATURES.
        """
        df = pd.DataFrame(index=bars_df.index)

        # Only for crypto symbols
        if not symbol or not symbol.upper().endswith("-USD"):
            for col in CRYPTO_SENTIMENT_FEATURES:
                df[col] = np.nan
            return df

        self._ensure_data()
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date

        # --- OKX Long/Short Ratio ---
        if self._okx_ls is not None and not self._okx_ls.empty:
            ls_map = dict(zip(self._okx_ls.index.date, self._okx_ls.values))
            ls_aligned = pd.Series(
                bar_dates.map(lambda d: ls_map.get(d, np.nan)).values,
                index=bars_df.index,
            ).astype(float).ffill()

            df["okx_long_short_ratio"] = ls_aligned
            # Z-score vs 30-day rolling
            mean30 = ls_aligned.rolling(30, min_periods=10).mean()
            std30 = ls_aligned.rolling(30, min_periods=10).std().replace(0, np.nan)
            df["okx_ls_zscore"] = (ls_aligned - mean30) / std30
        else:
            df["okx_long_short_ratio"] = np.nan
            df["okx_ls_zscore"] = np.nan

        # --- Fear & Greed Index ---
        if self._fear_greed is not None and not self._fear_greed.empty:
            fg_map = dict(zip(self._fear_greed.index.date, self._fear_greed.values))
            fg_aligned = pd.Series(
                bar_dates.map(lambda d: fg_map.get(d, np.nan)).values,
                index=bars_df.index,
            ).astype(float).ffill()

            # Normalize to 0-1 range
            df["fear_greed_index"] = fg_aligned / 100.0
            # Z-score vs 30-day rolling
            mean30 = fg_aligned.rolling(30, min_periods=10).mean()
            std30 = fg_aligned.rolling(30, min_periods=10).std().replace(0, np.nan)
            df["fear_greed_zscore"] = (fg_aligned - mean30) / std30
        else:
            df["fear_greed_index"] = np.nan
            df["fear_greed_zscore"] = np.nan

        # --- Polymarket BTC bullish probability ---
        if self._polymarket is not None and not self._polymarket.empty:
            pm_map = dict(zip(self._polymarket.index.date, self._polymarket.values))
            pm_aligned = pd.Series(
                bar_dates.map(lambda d: pm_map.get(d, np.nan)).values,
                index=bars_df.index,
            ).astype(float).ffill()
            df["polymarket_btc_bullish"] = pm_aligned
        else:
            df["polymarket_btc_bullish"] = np.nan

        return df
