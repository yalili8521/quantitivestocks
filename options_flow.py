#!/usr/bin/env python3
"""
Options Flow Signals
====================================
Put/call ratios, VIX term structure, and CBOE data as sentiment features.

Provides market-wide sentiment signals that can be used as additional features
in swing, expansion, and intraday models.

Data sources:
    - yfinance options chains (SPY, QQQ, IWM) for P/C volume and OI ratios
    - yfinance ^VIX / ^VIX3M for VIX term structure (contango/backwardation)
    - CBOE daily total put/call ratio (web scraper, graceful degradation)

Usage:
    from options_flow import OptionsFlowEngine
    engine = OptionsFlowEngine()
    features = engine.get_features()  # dict of sentiment features
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("options_flow")

# Feature names exported for use by model modules
OPTIONS_FLOW_FEATURES = [
    "pc_volume_ratio",    # aggregate put/call volume ratio (SPY+QQQ+IWM)
    "pc_oi_ratio",        # aggregate put/call open interest ratio
    "vix_term_ratio",     # VIX / VIX3M ratio (< 1 = contango, > 1 = backwardation)
    "vix_term_inverted",  # 1 if backwardation (term structure inverted), 0 otherwise
]

# Symbols used for aggregate P/C ratio
PC_RATIO_SYMBOLS = ["SPY", "QQQ", "IWM"]


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------
@dataclass
class _CacheEntry:
    value: object
    expires_at: float


class OptionsFlowCache:
    """Simple TTL-based in-memory cache."""

    def __init__(self, default_ttl: float = 300.0):
        self._ttl = default_ttl
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[object]:
        entry = self._store.get(key)
        if entry is None or time.time() > entry.expires_at:
            return None
        return entry.value

    def set(self, key: str, value: object, ttl: Optional[float] = None) -> None:
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.time() + (ttl if ttl is not None else self._ttl),
        )

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Put/Call Ratio Fetcher
# ---------------------------------------------------------------------------
class PutCallRatioFetcher:
    """Fetch aggregate P/C volume and OI ratios from yfinance options chains.

    Aggregates across SPY, QQQ, IWM for a broad market sentiment reading.
    Uses near-term expiration (closest expiry) for the most liquid options.
    """

    def __init__(self, symbols: list = None):
        self.symbols = symbols or PC_RATIO_SYMBOLS

    def fetch(self) -> Dict[str, float]:
        """Return dict with pc_volume_ratio and pc_oi_ratio.

        Returns NaN values if all fetches fail.
        """
        total_call_vol = 0.0
        total_put_vol = 0.0
        total_call_oi = 0.0
        total_put_oi = 0.0
        fetched_any = False

        for sym in self.symbols:
            try:
                ticker = yf.Ticker(sym)
                expirations = ticker.options
                if not expirations:
                    continue

                # Use nearest expiration for most liquid data
                nearest_exp = expirations[0]
                chain = ticker.option_chain(nearest_exp)

                calls = chain.calls
                puts = chain.puts

                if calls is not None and not calls.empty:
                    total_call_vol += float(calls["volume"].sum()) if "volume" in calls.columns else 0
                    total_call_oi += float(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0

                if puts is not None and not puts.empty:
                    total_put_vol += float(puts["volume"].sum()) if "volume" in puts.columns else 0
                    total_put_oi += float(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

                fetched_any = True
            except Exception as exc:
                log.debug("Options chain fetch failed for %s: %s", sym, exc)
                continue

        if not fetched_any or total_call_vol == 0:
            return {"pc_volume_ratio": np.nan, "pc_oi_ratio": np.nan}

        pc_volume = total_put_vol / max(total_call_vol, 1.0)
        pc_oi = total_put_oi / max(total_call_oi, 1.0) if total_call_oi > 0 else np.nan

        return {
            "pc_volume_ratio": round(pc_volume, 4),
            "pc_oi_ratio": round(pc_oi, 4),
        }


# ---------------------------------------------------------------------------
# VIX Term Structure Fetcher
# ---------------------------------------------------------------------------
class VIXTermStructureFetcher:
    """Compute VIX / VIX3M ratio for term structure analysis.

    VIX / VIX3M < 1.0 → contango (normal, low fear)
    VIX / VIX3M > 1.0 → backwardation (stressed, high near-term fear)
    """

    def fetch(self) -> Dict[str, float]:
        """Return dict with vix_term_ratio and vix_term_inverted."""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix3m_ticker = yf.Ticker("^VIX3M")

            vix_price = float(vix_ticker.fast_info.last_price)
            vix3m_price = float(vix3m_ticker.fast_info.last_price)

            if vix3m_price <= 0:
                return {"vix_term_ratio": np.nan, "vix_term_inverted": 0.0}

            ratio = vix_price / vix3m_price
            inverted = 1.0 if ratio > 1.0 else 0.0

            return {
                "vix_term_ratio": round(ratio, 4),
                "vix_term_inverted": inverted,
            }
        except Exception as exc:
            log.warning("VIX term structure fetch failed: %s", exc)
            return {"vix_term_ratio": np.nan, "vix_term_inverted": 0.0}


# ---------------------------------------------------------------------------
# CBOE Put/Call Ratio Scraper
# ---------------------------------------------------------------------------
class CBOEPutCallScraper:
    """Scrape daily total put/call ratio from CBOE public data.

    Graceful degradation: returns NaN if scraping fails (rate-limited,
    page structure changed, etc.).
    """

    CBOE_PC_URL = "https://www.cboe.com/us/options/market_statistics/daily/"

    def fetch(self) -> float:
        """Return CBOE total equity put/call ratio, or NaN on failure."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
            }
            resp = requests.get(self.CBOE_PC_URL, headers=headers, timeout=10)
            resp.raise_for_status()

            # Try to find the total P/C ratio in the page content
            text = resp.text

            # Look for common patterns in CBOE page
            # The page format may change — this is best-effort
            import re
            # Pattern: look for "Total" row with P/C ratio value
            patterns = [
                r'Total[^<]*Put/Call[^<]*Ratio[^<]*?(\d+\.\d+)',
                r'total.*?put.*?call.*?ratio.*?(\d+\.\d+)',
                r'"totalPCRatio"\s*:\s*(\d+\.?\d*)',
                r'"pcRatio"\s*:\s*(\d+\.?\d*)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    ratio = float(match.group(1))
                    if 0.1 < ratio < 5.0:  # sanity check
                        log.info("CBOE P/C ratio fetched: %.3f", ratio)
                        return ratio

            log.debug("Could not parse CBOE P/C ratio from page.")
            return np.nan

        except Exception as exc:
            log.debug("CBOE P/C scrape failed: %s", exc)
            return np.nan


# ---------------------------------------------------------------------------
# Unified Options Flow Engine
# ---------------------------------------------------------------------------
class OptionsFlowEngine:
    """Unified interface for all options flow sentiment signals.

    Provides cached access to:
        - Aggregate put/call volume and OI ratios
        - VIX term structure (contango/backwardation)
        - CBOE daily total put/call ratio

    Usage:
        engine = OptionsFlowEngine()
        features = engine.get_features()
        # {'pc_volume_ratio': 0.85, 'pc_oi_ratio': 0.92,
        #  'vix_term_ratio': 0.87, 'vix_term_inverted': 0.0,
        #  'cboe_pc_ratio': 0.76}
    """

    def __init__(self, cache_ttl: float = 300.0):
        self._cache = OptionsFlowCache(default_ttl=cache_ttl)
        self._pc_fetcher = PutCallRatioFetcher()
        self._vix_fetcher = VIXTermStructureFetcher()
        self._cboe_scraper = CBOEPutCallScraper()

    def get_features(self, force_refresh: bool = False) -> Dict[str, float]:
        """Return all options flow features as a dict.

        Results are cached with TTL. Use force_refresh=True to bypass cache.
        """
        cache_key = "options_flow_features"
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch all components
        pc_data = self._pc_fetcher.fetch()
        vix_data = self._vix_fetcher.fetch()
        cboe_ratio = self._cboe_scraper.fetch()

        features = {
            "pc_volume_ratio": pc_data.get("pc_volume_ratio", np.nan),
            "pc_oi_ratio": pc_data.get("pc_oi_ratio", np.nan),
            "vix_term_ratio": vix_data.get("vix_term_ratio", np.nan),
            "vix_term_inverted": vix_data.get("vix_term_inverted", 0.0),
            "cboe_pc_ratio": cboe_ratio,
        }

        self._cache.set(cache_key, features)
        return features

    def get_model_features(self, force_refresh: bool = False) -> Dict[str, float]:
        """Return only the features used by ML models (excludes cboe_pc_ratio).

        These 4 features are added to swing, expansion, and intraday models.
        """
        all_features = self.get_features(force_refresh=force_refresh)
        return {k: all_features.get(k, np.nan) for k in OPTIONS_FLOW_FEATURES}

    def get_historical_vix_term_structure(self, lookback_days: int = 252) -> pd.DataFrame:
        """Fetch historical VIX term structure ratio for backtesting features.

        Returns DataFrame with columns [date, vix_term_ratio, vix_term_inverted].
        """
        try:
            cal_days = int(lookback_days * 1.5) + 10
            vix_hist = yf.Ticker("^VIX").history(period=f"{cal_days}d", interval="1d")
            vix3m_hist = yf.Ticker("^VIX3M").history(period=f"{cal_days}d", interval="1d")

            if vix_hist.empty or vix3m_hist.empty:
                return pd.DataFrame(columns=["date", "vix_term_ratio", "vix_term_inverted"])

            # Align by date
            vix_close = vix_hist["Close"].rename("vix")
            vix3m_close = vix3m_hist["Close"].rename("vix3m")

            merged = pd.concat([vix_close, vix3m_close], axis=1).dropna()
            merged["vix_term_ratio"] = merged["vix"] / merged["vix3m"].replace(0, np.nan)
            merged["vix_term_inverted"] = (merged["vix_term_ratio"] > 1.0).astype(float)

            result = merged[["vix_term_ratio", "vix_term_inverted"]].reset_index()
            result.columns = ["date", "vix_term_ratio", "vix_term_inverted"]
            if result["date"].dt.tz is not None:
                result["date"] = result["date"].dt.tz_localize(None)
            result["date"] = result["date"].dt.date

            return result

        except Exception as exc:
            log.warning("Historical VIX term structure fetch failed: %s", exc)
            return pd.DataFrame(columns=["date", "vix_term_ratio", "vix_term_inverted"])

    def get_historical_pc_ratio(self, lookback_days: int = 60) -> pd.DataFrame:
        """Estimate historical P/C ratios from options volume data.

        Note: yfinance only provides current options chains, not historical.
        For backtesting, we use VIX term structure as the primary options flow
        feature and fill P/C ratios with NaN (will be handled by models).
        """
        # Historical P/C ratios are not available from yfinance
        # Models should handle NaN gracefully during backtesting
        log.info("Historical P/C ratios not available from yfinance. "
                 "Using VIX term structure as primary options flow signal for backtesting.")
        return pd.DataFrame(columns=["date", "pc_volume_ratio", "pc_oi_ratio"])
