#!/usr/bin/env python3
"""
Cross-Asset Signals
====================================
FRED macro data and cross-market tickers as additional ML features.

Provides per-symbol cross-asset features based on economic relationships:
    - GLD/SLV: real yield spread, gold/silver ratio, treasury slope
    - XLE: crude oil momentum, treasury slope
    - EWJ: USD/JPY momentum, treasury slope
    - EWT/EEM/EWS/INDA: USD strength (DXY) momentum, treasury slope
    - SPY/QQQ/IWM/SOXX: treasury slope only

Data sources:
    - FRED API: DGS10 (10Y yield), DFII10 (10Y TIPS), T10Y2Y (yield curve slope)
    - yfinance: CL=F (crude), JPY=X (USD/JPY), DX-Y.NYB (dollar index),
                GC=F (gold), SI=F (silver)

Usage:
    from cross_asset_signals import CrossAssetFeatureBuilder, CROSS_ASSET_MAP
    builder = CrossAssetFeatureBuilder(fred_key="your_key")
    features = builder.build_features(bars_df, symbol="GLD")
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cross_asset")

# ---------------------------------------------------------------------------
# Per-symbol cross-asset feature mapping
# ---------------------------------------------------------------------------
CROSS_ASSET_MAP: Dict[str, List[str]] = {
    "GLD":  ["real_yield_spread", "gold_silver_ratio", "treasury_slope"],
    "SLV":  ["real_yield_spread", "gold_silver_ratio", "treasury_slope"],
    "USO":  ["usd_strength_ret5", "usd_strength_ret20", "treasury_slope"],
    "XLE":  ["crude_ret5", "crude_ret20", "treasury_slope"],
    "EWJ":  ["usdjpy_ret5", "usdjpy_ret20", "treasury_slope"],
    "EWT":  ["usd_strength_ret5", "usd_strength_ret20", "treasury_slope"],
    "EEM":  ["usd_strength_ret5", "usd_strength_ret20", "treasury_slope"],
    "EWS":  ["usd_strength_ret5", "usd_strength_ret20", "treasury_slope"],
    "INDA": ["usd_strength_ret5", "usd_strength_ret20", "treasury_slope"],
    "SPY":  ["treasury_slope"],
    "QQQ":  ["treasury_slope"],
    "IWM":  ["treasury_slope"],
    "SOXX": ["treasury_slope"],
}

# All possible cross-asset feature names (union of all values)
ALL_CROSS_ASSET_FEATURES = sorted(set(
    feat for feats in CROSS_ASSET_MAP.values() for feat in feats
))


def get_cross_asset_features(symbol: str) -> List[str]:
    """Return list of cross-asset feature names for a given symbol."""
    return list(CROSS_ASSET_MAP.get(symbol, ["treasury_slope"]))


# ---------------------------------------------------------------------------
# FRED Macro Series Fetcher
# ---------------------------------------------------------------------------
class FREDMacroFetcher:
    """Fetch macro series from FRED API.

    Extends the FREDVixFetcher pattern from signals_engine.py.
    Supports multiple series: DGS10, DFII10, T10Y2Y.
    """

    BASE = "https://api.stlouisfed.org/fred/series/observations"

    # Series we need
    SERIES_IDS = {
        "DGS10":  "dgs10",     # 10-Year Treasury Constant Maturity Rate
        "DFII10": "dfii10",    # 10-Year TIPS (real yield)
        "T10Y2Y": "t10y2y",    # 10Y-2Y Treasury spread (yield curve slope)
    }

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.environ.get("FRED_API_KEY")
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 86400.0  # 24 hour cache

    def fetch_series(self, series_id: str, lookback_days: int = 500) -> pd.DataFrame:
        """Fetch a single FRED series. Returns DataFrame with [date, value]."""
        # Check cache
        cache_key = f"{series_id}_{lookback_days}"
        cached_time = self._cache_time.get(cache_key, 0)
        if time.time() - cached_time < self._cache_ttl and cache_key in self._cache:
            return self._cache[cache_key]

        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
            "file_type": "json",
            "sort_order": "asc",
        }
        if self._key:
            params["api_key"] = self._key
        else:
            params["api_key"] = "DEMO_KEY"
            log.warning("No FRED_API_KEY set; using DEMO_KEY (may be rate-limited).")

        try:
            resp = requests.get(self.BASE, params=params, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except Exception as exc:
            log.warning("FRED fetch failed for %s: %s", series_id, exc)
            return pd.DataFrame(columns=["date", "value"])

        rows = []
        for o in obs:
            val = o.get("value", ".")
            if val == ".":
                continue
            rows.append({"date": o["date"], "value": float(val)})

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").reset_index(drop=True)

        self._cache[cache_key] = df
        self._cache_time[cache_key] = time.time()
        return df

    def fetch_all(self, lookback_days: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch all macro series. Returns dict of series_id -> DataFrame."""
        result = {}
        for series_id in self.SERIES_IDS:
            result[series_id] = self.fetch_series(series_id, lookback_days)
        return result


# ---------------------------------------------------------------------------
# Cross-Market Ticker Fetcher (yfinance)
# ---------------------------------------------------------------------------
class CrossMarketFetcher:
    """Fetch cross-market tickers via yfinance with caching.

    Tickers: CL=F (crude oil), JPY=X (USD/JPY), DX-Y.NYB (dollar index),
             GC=F (gold futures), SI=F (silver futures)
    """

    TICKERS = {
        "crude":    "CL=F",
        "usdjpy":   "JPY=X",
        "dxy":      "DX-Y.NYB",
        "gold_fut": "GC=F",
        "silver_fut": "SI=F",
    }

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 86400.0  # 24 hour cache

    def fetch(self, ticker_key: str, lookback_days: int = 500) -> pd.DataFrame:
        """Fetch historical daily close for a cross-market ticker.

        Args:
            ticker_key: Key from TICKERS dict (e.g. 'crude', 'usdjpy')
            lookback_days: Number of trading days

        Returns:
            DataFrame with [date, close]
        """
        yf_symbol = self.TICKERS.get(ticker_key)
        if yf_symbol is None:
            log.warning("Unknown cross-market ticker key: %s", ticker_key)
            return pd.DataFrame(columns=["date", "close"])

        # Check cache
        cache_key = f"{ticker_key}_{lookback_days}"
        cached_time = self._cache_time.get(cache_key, 0)
        if time.time() - cached_time < self._cache_ttl and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            cal_days = int(lookback_days * 1.5) + 10
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=f"{cal_days}d", interval="1d")

            if hist.empty:
                log.warning("No data for %s (%s).", ticker_key, yf_symbol)
                return pd.DataFrame(columns=["date", "close"])

            df = pd.DataFrame({
                "date": hist.index,
                "close": hist["Close"].values,
            })
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["date"] = df["date"].dt.date
            df = df.sort_values("date").reset_index(drop=True)

            self._cache[cache_key] = df
            self._cache_time[cache_key] = time.time()
            return df

        except Exception as exc:
            log.warning("Cross-market fetch failed for %s (%s): %s",
                        ticker_key, yf_symbol, exc)
            return pd.DataFrame(columns=["date", "close"])

    def fetch_all(self, lookback_days: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch all cross-market tickers."""
        return {key: self.fetch(key, lookback_days) for key in self.TICKERS}


# ---------------------------------------------------------------------------
# Cross-Asset Feature Builder
# ---------------------------------------------------------------------------
class CrossAssetFeatureBuilder:
    """Build per-symbol cross-asset features for ML models.

    Aligns cross-asset data to the target symbol's bar dates and computes
    relevant features based on CROSS_ASSET_MAP.
    """

    def __init__(self, fred_key: Optional[str] = None):
        self._fred = FREDMacroFetcher(api_key=fred_key)
        self._cross = CrossMarketFetcher()
        self._macro_data: Optional[Dict[str, pd.DataFrame]] = None
        self._cross_data: Optional[Dict[str, pd.DataFrame]] = None

    def _ensure_data(self, lookback_days: int = 500) -> None:
        """Fetch all data sources if not already cached."""
        if self._macro_data is None:
            self._macro_data = self._fred.fetch_all(lookback_days)
        if self._cross_data is None:
            self._cross_data = self._cross.fetch_all(lookback_days)

    def _align_series(self, source_df: pd.DataFrame, bar_dates: pd.Series,
                      value_col: str = "value") -> pd.Series:
        """Align a source DataFrame to target bar dates via forward-fill."""
        if source_df.empty:
            return pd.Series(np.nan, index=bar_dates.index)

        date_map = dict(zip(source_df["date"].values,
                            source_df[value_col].values))
        aligned = bar_dates.map(lambda d: date_map.get(d, np.nan))
        aligned = pd.Series(aligned.values, index=bar_dates.index).ffill()
        return aligned

    def build_features(self, bars_df: pd.DataFrame, symbol: str,
                       lookback_days: int = 500) -> pd.DataFrame:
        """Build cross-asset features for a symbol.

        Args:
            bars_df: OHLCV bars for the target symbol
            symbol: Target symbol (determines which features to compute)
            lookback_days: Data lookback

        Returns:
            DataFrame with cross-asset feature columns, same index as bars_df.
            NaN values for warmup period (first ~20 rows).
        """
        self._ensure_data(lookback_days)
        feature_list = get_cross_asset_features(symbol)
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date

        df = pd.DataFrame(index=bars_df.index)

        # --- Treasury slope (T10Y2Y) — used by all symbols ---
        if "treasury_slope" in feature_list:
            t10y2y = self._macro_data.get("T10Y2Y", pd.DataFrame())
            df["treasury_slope"] = self._align_series(t10y2y, bar_dates)

        # --- Real yield spread — for GLD/SLV ---
        if "real_yield_spread" in feature_list:
            dgs10 = self._macro_data.get("DGS10", pd.DataFrame())
            dfii10 = self._macro_data.get("DFII10", pd.DataFrame())
            nominal = self._align_series(dgs10, bar_dates)
            real = self._align_series(dfii10, bar_dates)
            # Real yield spread: higher real yields → bearish for gold
            df["real_yield_spread"] = real  # TIPS yield level
            # Also compute change in real yields (more informative)
            df["real_yield_spread"] = real.diff(5)  # 5-day change in real yields

        # --- Gold/Silver ratio — for GLD/SLV ---
        if "gold_silver_ratio" in feature_list:
            gold_df = self._cross_data.get("gold_fut", pd.DataFrame())
            silver_df = self._cross_data.get("silver_fut", pd.DataFrame())
            gold = self._align_series(gold_df, bar_dates, "close")
            silver = self._align_series(silver_df, bar_dates, "close")
            df["gold_silver_ratio"] = gold / silver.replace(0, np.nan)

        # --- Crude oil momentum — for XLE ---
        if "crude_ret5" in feature_list or "crude_ret20" in feature_list:
            crude_df = self._cross_data.get("crude", pd.DataFrame())
            crude = self._align_series(crude_df, bar_dates, "close")
            if "crude_ret5" in feature_list:
                df["crude_ret5"] = crude.pct_change(5)
            if "crude_ret20" in feature_list:
                df["crude_ret20"] = crude.pct_change(20)

        # --- USD/JPY momentum — for EWJ ---
        if "usdjpy_ret5" in feature_list or "usdjpy_ret20" in feature_list:
            usdjpy_df = self._cross_data.get("usdjpy", pd.DataFrame())
            usdjpy = self._align_series(usdjpy_df, bar_dates, "close")
            if "usdjpy_ret5" in feature_list:
                df["usdjpy_ret5"] = usdjpy.pct_change(5)
            if "usdjpy_ret20" in feature_list:
                df["usdjpy_ret20"] = usdjpy.pct_change(20)

        # --- USD strength (DXY) momentum — for EM/Asia ETFs ---
        if "usd_strength_ret5" in feature_list or "usd_strength_ret20" in feature_list:
            dxy_df = self._cross_data.get("dxy", pd.DataFrame())
            dxy = self._align_series(dxy_df, bar_dates, "close")
            if "usd_strength_ret5" in feature_list:
                df["usd_strength_ret5"] = dxy.pct_change(5)
            if "usd_strength_ret20" in feature_list:
                df["usd_strength_ret20"] = dxy.pct_change(20)

        return df[feature_list]
