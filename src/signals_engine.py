#!/usr/bin/env python3
"""
ETF / Trust Sentiment Signal Engine
====================================
Read-only signal computation — NO orders, NO account access, NO trading.

Providers:
    yahoo   – Yahoo Finance via yfinance (free, no key needed)
    alpaca  – Alpaca Markets (free real-time data with paper/live account)
    hybrid  – Yahoo for daily bars + Alpaca for intraday bars

Usage (via main.py):
    python main.py signals --provider yahoo --intraday 5min
    python main.py signals --provider alpaca --intraday 1min --ml

Environment variables:
    ALPACA_API_KEY    – Alpaca API key    (required for alpaca/hybrid)
    ALPACA_API_SECRET – Alpaca API secret (required for alpaca/hybrid)
    ALPACA_PAPER      – Set to "1" to use paper trading endpoint (default: paper)
    FRED_API_KEY      – FRED API key (optional; falls back to keyless endpoint)

Get free Alpaca keys at https://app.alpaca.markets/signup
(paper account = instant, no funding needed)
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("signals_engine")

# ---------------------------------------------------------------------------
# Project root (one level up from src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Constants / default config
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE: List[str] = ["SPY", "QQQ", "IWM", "IGV", "SLV"]
DAILY_LOOKBACK = 120          # trading days
RSI_PERIOD = 14
RET5_DAYS = 5
RET10_DAYS = 10
WEEK_DAYS = 5                 # trading days in a week
MONTH_DAYS = 21               # trading days in a month
VOL_WINDOW = 20
ANNUALIZE = math.sqrt(252)

# Scoring weights — easy to tweak ------------------------------------------
PANIC_WEIGHTS = {
    "ret5_drop":     0.30,    # 5-day drawdown intensity
    "rsi_low":       0.25,    # RSI low-level degree
    "vix_rise":      0.25,    # VIX 1-day rise speed
    "rel_spy_under": 0.20,    # under-performance vs SPY
}
OVERHEAT_WEIGHTS = {
    "ret5_gain":  0.25,
    "ret10_gain": 0.25,
    "rsi_high":   0.30,
    "vix_low":    0.20,
}


# ===================================================================
# DataAdapter interface
# ===================================================================
class DataAdapter(abc.ABC):
    """Unified interface for market data providers."""

    @abc.abstractmethod
    def fetch_daily(self, symbol: str, lookback: int) -> pd.DataFrame:
        """Return daily OHLCV with columns:
        [symbol, ts, open, high, low, close, volume]
        *ts* must be UTC datetime.
        """

    @abc.abstractmethod
    def fetch_intraday(self, symbol: str, interval: str,
                       lookback_days: int = 1) -> pd.DataFrame:
        """Return intraday bars (same columns as daily).

        Parameters
        ----------
        lookback_days : number of calendar days of intraday history to fetch.
                        Default 1 = today only.
        """


# ===================================================================
# Yahoo Finance adapter
# ===================================================================
class YahooFinanceAdapter(DataAdapter):
    """Free data via the yfinance library (no API key needed)."""

    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert yfinance output to the unified schema."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        out = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]].copy()
        out["symbol"] = symbol
        out["ts"] = out.index
        # Ensure UTC
        if out["ts"].dt.tz is None:
            out["ts"] = out["ts"].dt.tz_localize("UTC")
        else:
            out["ts"] = out["ts"].dt.tz_convert("UTC")
        return out[["symbol", "ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def fetch_daily(self, symbol: str, lookback: int = DAILY_LOOKBACK) -> pd.DataFrame:
        # Request extra calendar days to cover ~lookback trading days
        cal_days = int(lookback * 1.5) + 10
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{cal_days}d", interval="1d", auto_adjust=True)
        out = self._normalize(df, symbol)
        return out.tail(lookback).reset_index(drop=True)

    def fetch_intraday(self, symbol: str, interval: str = "5min",
                       lookback_days: int = 1) -> pd.DataFrame:
        # yfinance interval format: "1m" or "5m"
        yf_interval = interval.replace("min", "m")
        ticker = yf.Ticker(symbol)
        # yfinance supports up to 60 days of intraday data
        period = f"{min(lookback_days, 60)}d"
        df = ticker.history(period=period, interval=yf_interval, auto_adjust=True)
        return self._normalize(df, symbol)


# ===================================================================
# Alpaca Markets adapter
# ===================================================================
_ALP_MAX_RETRIES = 3
_ALP_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class AlpacaAdapter(DataAdapter):
    """Alpaca Markets data adapter using the alpaca-py SDK.

    Free real-time data with a paper or live brokerage account.
    Supports daily and intraday (1min / 5min) bars.
    Rate limit: 200 req/min (generous).

    Required env vars: ALPACA_API_KEY, ALPACA_API_SECRET
    Get free keys at https://app.alpaca.markets/signup
    """

    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca provider requires ALPACA_API_KEY and ALPACA_API_SECRET "
                "environment variables. Get free keys (paper account) at "
                "https://app.alpaca.markets/signup"
            )
        from alpaca.data.historical import StockHistoricalDataClient
        self._client = StockHistoricalDataClient(api_key, api_secret)
        log.info("AlpacaAdapter initialized.")

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _normalize(bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert alpaca-py bars DataFrame to unified schema."""
        empty = pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        if bars_df is None or bars_df.empty:
            return empty

        # alpaca-py returns MultiIndex (symbol, timestamp) or single-index
        df = bars_df.reset_index()
        rename = {}
        if "timestamp" in df.columns:
            rename["timestamp"] = "ts"
        elif "Timestamp" in df.columns:
            rename["Timestamp"] = "ts"
        for col in df.columns:
            if col.lower() in ("open", "high", "low", "close", "volume"):
                rename[col] = col.lower()
        df = df.rename(columns=rename)

        df["symbol"] = symbol
        # Ensure UTC
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")
        else:
            df["ts"] = df["ts"].dt.tz_convert("UTC")

        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                df[c] = np.nan
        return df[["symbol", "ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)

    def _call_with_retry(self, func, *args, **kwargs):
        """Call with bounded exponential backoff on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(_ALP_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = _ALP_BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Alpaca API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _ALP_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        log.error("Alpaca API call failed after %d retries: %s", _ALP_MAX_RETRIES, last_exc)
        raise RuntimeError(f"Alpaca API failed after {_ALP_MAX_RETRIES} retries") from last_exc

    # -- interface ---------------------------------------------------------
    def fetch_daily(self, symbol: str, lookback: int = DAILY_LOOKBACK) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        cal_days = int(lookback * 1.5) + 10
        start = datetime.now(timezone.utc) - timedelta(days=cal_days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
        )
        bars = self._call_with_retry(self._client.get_stock_bars, request)
        df = self._normalize(bars.df, symbol)
        return df.tail(lookback).reset_index(drop=True)

    def fetch_intraday(self, symbol: str, interval: str = "5min",
                       lookback_days: int = 1) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = TimeFrame.Minute if interval == "1min" else TimeFrame(5, TimeFrameUnit.Minute)
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
        )
        bars = self._call_with_retry(self._client.get_stock_bars, request)
        return self._normalize(bars.df, symbol)


# ===================================================================
# Hybrid adapter: Yahoo daily + Alpaca intraday
# ===================================================================
class HybridAdapter(DataAdapter):
    """Uses Yahoo Finance for daily bars and Alpaca for intraday bars.

    Best of both: free reliable daily history from Yahoo,
    plus Alpaca's real-time intraday data.
    """

    def __init__(self, alpaca_adapter: AlpacaAdapter):
        self._yahoo = YahooFinanceAdapter()
        self._alpaca = alpaca_adapter

    def fetch_daily(self, symbol: str, lookback: int = DAILY_LOOKBACK) -> pd.DataFrame:
        return self._yahoo.fetch_daily(symbol, lookback)

    def fetch_intraday(self, symbol: str, interval: str = "5min",
                       lookback_days: int = 1) -> pd.DataFrame:
        return self._alpaca.fetch_intraday(symbol, interval, lookback_days)


# ===================================================================
# FRED VIX adapter
# ===================================================================
class FREDVixFetcher:
    """Fetch VIXCLS from FRED (daily)."""

    SERIES = "VIXCLS"
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key

    def fetch(self, lookback_days: int = 10) -> pd.DataFrame:
        """Return DataFrame with columns [date, vix]."""
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
        params = {
            "series_id": self.SERIES,
            "observation_start": start,
            "observation_end": end,
            "file_type": "json",
            "sort_order": "desc",
            "limit": lookback_days,
        }
        if self._key:
            params["api_key"] = self._key
        else:
            # FRED allows limited keyless access — may fail
            params["api_key"] = "DEMO_KEY"
            log.warning("No FRED_API_KEY set; using DEMO_KEY (may be rate-limited).")

        try:
            resp = requests.get(self.BASE, params=params, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except Exception as exc:
            log.error("FRED VIX fetch failed: %s", exc)
            return pd.DataFrame(columns=["date", "vix"])

        rows = []
        for o in obs:
            val = o.get("value", ".")
            if val == ".":
                continue
            rows.append({"date": o["date"], "vix": float(val)})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df


# ===================================================================
# Indicator calculations
# ===================================================================
def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_return(series: pd.Series, days: int) -> float:
    if len(series) < days + 1:
        return np.nan
    return (series.iloc[-1] / series.iloc[-1 - days]) - 1


def compute_realized_vol(series: pd.Series, window: int = VOL_WINDOW) -> float:
    if len(series) < window + 1:
        return np.nan
    rets = series.pct_change().dropna().tail(window)
    return float(rets.std() * ANNUALIZE)


def compute_dollar_volume(df_intra: pd.DataFrame) -> float:
    if df_intra.empty:
        return 0.0
    price = df_intra["close"].fillna(0)
    vol = df_intra["volume"].fillna(0)
    return float((price * vol).sum())


# ===================================================================
# Order-flow indicators
# ===================================================================
def compute_volume_imbalance(df: pd.DataFrame) -> float:
    """Estimate buy vs sell volume from bar data.

    Heuristic: for each bar, buy fraction = (close - low) / (high - low).
    Returns imbalance ratio in [-1, +1]:  +1 = all buying, -1 = all selling.
    """
    if df.empty:
        return 0.0
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    buy_frac = (df["close"] - df["low"]) / spread
    buy_frac = buy_frac.fillna(0.5)
    buy_vol = (buy_frac * df["volume"]).sum()
    sell_vol = ((1 - buy_frac) * df["volume"]).sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return float((buy_vol - sell_vol) / total)


def compute_vwap(df: pd.DataFrame) -> float:
    """Volume-weighted average price for the given bars."""
    if df.empty:
        return np.nan
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)
    total_vol = vol.sum()
    if total_vol == 0:
        return float(typical_price.mean())
    return float((typical_price * vol).sum() / total_vol)


def compute_dollar_vol_accel(daily_df: pd.DataFrame, window: int = 5) -> float:
    """Rate of change of dollar volume over `window` days.

    Returns (recent_avg - prior_avg) / prior_avg.
    Positive = accelerating volume; negative = decelerating.
    """
    if len(daily_df) < 2 * window:
        return np.nan
    dv = (daily_df["close"] * daily_df["volume"]).astype(float)
    recent = dv.iloc[-window:].mean()
    prior = dv.iloc[-2 * window:-window].mean()
    if prior == 0:
        return np.nan
    return float((recent - prior) / prior)


def compute_spread_proxy(df: pd.DataFrame) -> float:
    """(high - low) / close averaged across bars. Proxy for bid-ask spread."""
    if df.empty:
        return np.nan
    spread = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).dropna()
    if spread.empty:
        return np.nan
    return float(spread.mean())


# ===================================================================
# Scoring helpers (normalize to 0-1)
# ===================================================================
def _clip01(x: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def score_panic(
    ret5: float,
    rsi: float,
    vix_chg_pct: float,
    rel_spy: float,
) -> float:
    """Compute panic_score in [0, 1].  Higher -> more panic -> better buy signal."""
    # ret5 drop: map [-0.15, 0] -> [1, 0]
    s_ret5 = _clip01(-ret5 / 0.15) if not np.isnan(ret5) else 0.0

    # RSI low: map [20, 50] -> [1, 0]
    s_rsi = _clip01((50 - rsi) / 30) if not np.isnan(rsi) else 0.0

    # VIX rise: map [0%, +40%] -> [0, 1]
    s_vix = _clip01(vix_chg_pct / 40) if not np.isnan(vix_chg_pct) else 0.0

    # Relative underperformance vs SPY: map [-0.10, 0] -> [1, 0]
    s_rel = _clip01(-rel_spy / 0.10) if not np.isnan(rel_spy) else 0.0

    w = PANIC_WEIGHTS
    return float(
        w["ret5_drop"] * s_ret5
        + w["rsi_low"] * s_rsi
        + w["vix_rise"] * s_vix
        + w["rel_spy_under"] * s_rel
    )


def score_overheat(
    ret5: float,
    ret10: float,
    rsi: float,
    vix: float,
) -> float:
    """Compute overheat_score in [0, 1].  Higher -> more overheated."""
    # ret5 gain: map [0, +0.10] -> [0, 1]
    s_r5 = _clip01(ret5 / 0.10) if not np.isnan(ret5) else 0.0

    # ret10 gain: map [0, +0.15] -> [0, 1]
    s_r10 = _clip01(ret10 / 0.15) if not np.isnan(ret10) else 0.0

    # RSI high: map [50, 80] -> [0, 1]
    s_rsi = _clip01((rsi - 50) / 30) if not np.isnan(rsi) else 0.0

    # VIX low -> overheat: map [30, 12] -> [0, 1]
    s_vix = _clip01((30 - vix) / 18) if not np.isnan(vix) else 0.0

    w = OVERHEAT_WEIGHTS
    return float(
        w["ret5_gain"] * s_r5
        + w["ret10_gain"] * s_r10
        + w["rsi_high"] * s_rsi
        + w["vix_low"] * s_vix
    )


# ===================================================================
# Main engine
# ===================================================================
def _build_alpaca_adapter() -> AlpacaAdapter:
    """Create an AlpacaAdapter from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    return AlpacaAdapter(api_key, api_secret)


def build_adapter(provider: str) -> DataAdapter:
    provider = provider.lower()
    if provider == "yahoo":
        return YahooFinanceAdapter()
    if provider == "alpaca":
        return _build_alpaca_adapter()
    if provider == "hybrid":
        alp = _build_alpaca_adapter()
        return HybridAdapter(alp)
    raise ValueError(f"Unknown provider: {provider!r}. Choose 'yahoo', 'alpaca', or 'hybrid'.")


def run(symbols: List[str], provider: str, intraday_interval: str,
        use_ml: bool = False) -> None:
    adapter = build_adapter(provider)

    # --- VIX -----------------------------------------------------------
    fred_key = os.environ.get("FRED_API_KEY")
    vix_fetcher = FREDVixFetcher(api_key=fred_key)
    vix_df = vix_fetcher.fetch(lookback_days=10)
    vix_today: float = np.nan
    vix_chg_pct: float = np.nan
    if len(vix_df) >= 1:
        vix_today = float(vix_df["vix"].iloc[-1])
    if len(vix_df) >= 2:
        prev = float(vix_df["vix"].iloc[-2])
        if prev != 0:
            vix_chg_pct = (vix_today - prev) / prev * 100

    # --- per-symbol computation ----------------------------------------
    results: Dict[str, dict] = {}
    spy_ret5: float = np.nan  # needed for relative score

    for sym in symbols:
        log.info("Processing %s …", sym)
        rec: dict = {"symbol": sym}

        # daily
        try:
            daily = adapter.fetch_daily(sym, DAILY_LOOKBACK)
        except Exception as exc:
            log.error("Failed to fetch daily data for %s: %s", sym, exc)
            daily = pd.DataFrame()

        if daily.empty or len(daily) < RSI_PERIOD + 1:
            log.warning("Insufficient daily data for %s (%d rows). Skipping indicators.", sym, len(daily))
            rec.update({
                "rsi14": np.nan, "ret5": np.nan, "ret10": np.nan,
                "wk_ret": np.nan, "mo_ret": np.nan, "vol20": np.nan,
            })
        else:
            close = daily["close"].astype(float)
            rsi_series = compute_rsi(close, RSI_PERIOD)
            rec["rsi14"] = round(float(rsi_series.iloc[-1]), 2)
            rec["ret5"] = round(compute_return(close, RET5_DAYS), 6)
            rec["ret10"] = round(compute_return(close, RET10_DAYS), 6)
            rec["wk_ret"] = round(compute_return(close, WEEK_DAYS), 6)
            rec["mo_ret"] = round(compute_return(close, MONTH_DAYS), 6)
            rec["vol20"] = round(compute_realized_vol(close, VOL_WINDOW), 6)

        # cache SPY ret5 for relative calculation
        if sym == "SPY" and not np.isnan(rec.get("ret5", np.nan)):
            spy_ret5 = rec["ret5"]

        # intraday
        try:
            intra = adapter.fetch_intraday(sym, intraday_interval)
        except Exception as exc:
            log.error("Failed to fetch intraday data for %s: %s", sym, exc)
            intra = pd.DataFrame()

        rec["dollar_vol_today"] = round(compute_dollar_volume(intra), 2)
        rec["vol_imbalance"] = round(compute_volume_imbalance(intra), 4)
        rec["vwap"] = round(compute_vwap(intra), 4)
        rec["dv_accel"] = round(compute_dollar_vol_accel(daily, window=5), 6)
        rec["spread_proxy"] = round(compute_spread_proxy(daily.tail(5)), 6)

        results[sym] = rec

    # --- scoring -------------------------------------------------------
    for sym, rec in results.items():
        rel_spy = (rec.get("ret5", np.nan) or 0) - (spy_ret5 if not np.isnan(spy_ret5) else 0)
        rec["panic_score"] = round(
            score_panic(
                rec.get("ret5", np.nan),
                rec.get("rsi14", np.nan),
                vix_chg_pct,
                rel_spy,
            ), 4,
        )
        rec["overheat_score"] = round(
            score_overheat(
                rec.get("ret5", np.nan),
                rec.get("ret10", np.nan),
                rec.get("rsi14", np.nan),
                vix_today,
            ), 4,
        )

    # --- optional ML prediction ----------------------------------------
    if use_ml:
        try:
            from src.ml_model import Predictor, _fetch_vix_for_training
            model_dir = os.path.join(PROJECT_ROOT, "data", "models")
            ml_vix_df = _fetch_vix_for_training(fred_key, lookback_days=DAILY_LOOKBACK)
            for sym, rec in results.items():
                try:
                    predictor = Predictor(sym, model_dir=model_dir)
                    daily = adapter.fetch_daily(sym, DAILY_LOOKBACK)
                    pred = predictor.predict(daily, ml_vix_df)
                    rec["ml_direction"] = pred["direction"]
                    rec["ml_confidence"] = pred["confidence"]
                except FileNotFoundError:
                    rec["ml_direction"] = "N/A"
                    rec["ml_confidence"] = 0.0
        except ImportError:
            log.warning("ml_model not available; skipping ML predictions. "
                        "Run: python main.py train --symbol SPY")
            for rec in results.values():
                rec["ml_direction"] = "N/A"
                rec["ml_confidence"] = 0.0

    # --- console output ------------------------------------------------
    display_cols = [
        "symbol", "rsi14", "ret5", "ret10", "wk_ret", "mo_ret",
        "vol20", "dollar_vol_today", "panic_score", "overheat_score",
    ]
    if use_ml:
        display_cols += ["ml_direction", "ml_confidence"]
    rows = [rec for rec in results.values()]
    df_out = pd.DataFrame(rows)[display_cols]

    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "—"
        return v

    print("\n" + "=" * 100)
    print("  ETF/Trust Sentiment Signal Engine")
    print("=" * 100)
    print(f"  As-of (UTC) : {datetime.now(timezone.utc).isoformat()}")
    print(f"  VIX         : {vix_today if not np.isnan(vix_today) else '—'}  "
          f"(1d chg: {vix_chg_pct:.2f}%)" if not np.isnan(vix_chg_pct) else
          f"  VIX         : {vix_today if not np.isnan(vix_today) else '—'}  (1d chg: —)")
    print("-" * 100)
    header = f"{'Symbol':>6}  {'RSI14':>7}  {'Ret5':>9}  {'Ret10':>9}  {'WkRet':>9}  " \
             f"{'MoRet':>9}  {'Vol20':>8}  {'$VolToday':>14}  {'Panic':>6}  {'Overheat':>8}"
    if use_ml:
        header += f"  {'ML Dir':>6}  {'ML Conf':>7}"
    print(header)
    print("-" * 100)
    for _, row in df_out.iterrows():
        vals = []
        for c in display_cols:
            v = row[c]
            if isinstance(v, str):
                if c == "symbol":
                    vals.append(f"{v:>6}")
                elif c == "ml_direction":
                    vals.append(f"{v:>6}")
                else:
                    vals.append(v)
            elif isinstance(v, float) and np.isnan(v):
                width = 14 if c == "dollar_vol_today" else 9
                vals.append(f"{'—':>{width}}")
            elif c in ("ret5", "ret10", "wk_ret", "mo_ret"):
                vals.append(f"{v:>9.4%}")
            elif c == "vol20":
                vals.append(f"{v:>8.4f}")
            elif c == "dollar_vol_today":
                vals.append(f"{v:>14,.0f}")
            elif c == "rsi14":
                vals.append(f"{v:>7.2f}")
            elif c in ("panic_score", "overheat_score"):
                width = 6 if c == "panic_score" else 8
                vals.append(f"{v:>{width}.4f}")
            elif c == "ml_confidence":
                vals.append(f"{v:>7.4f}")
            else:
                vals.append(str(v))
        print("  ".join(vals))
    print("=" * 100 + "\n")

    # --- JSON output ---------------------------------------------------
    asof = datetime.now(timezone.utc).isoformat()
    output = {
        "asof": asof,
        "universe": symbols,
        "market": {
            "vix": None if np.isnan(vix_today) else vix_today,
            "vix_change_1d_pct": None if np.isnan(vix_chg_pct) else round(vix_chg_pct, 4),
        },
        "signals": {},
    }

    for sym, rec in results.items():
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        output["signals"][sym] = clean

    out_path = os.path.join(PROJECT_ROOT, "data", "output", "signals.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %s", out_path)


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETF/Trust Sentiment Signal Engine (read-only, no trading).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: SPY,QQQ,IWM,IGV,SLV)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="yahoo",
        choices=["yahoo", "alpaca", "hybrid"],
        help="Data provider: yahoo (free), alpaca (free w/ account), "
             "hybrid (Yahoo daily + Alpaca intraday). Default: yahoo",
    )
    parser.add_argument(
        "--intraday",
        type=str,
        default="5min",
        choices=["1min", "5min"],
        help="Intraday bar interval (default: 5min)",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        default=False,
        help="Include ML predictions (requires trained models in data/models/)",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else DEFAULT_UNIVERSE

    # Ensure SPY is present (needed for relative scoring)
    if "SPY" not in symbols:
        log.info("Adding SPY to universe (required for relative scoring).")
        symbols.insert(0, "SPY")

    run(symbols, args.provider, args.intraday, use_ml=args.ml)


if __name__ == "__main__":
    main()
