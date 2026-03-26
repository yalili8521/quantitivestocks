#!/usr/bin/env python3
"""
ETF Intraday Model (LightGBM 70% + GRU 30% Ensemble)
=====================================================
Rolling 5-min bar model for intraday ETF trading via Alpaca.

Architecture (mirrors crypto_intraday_model):
    - LightGBM: tabular features (momentum, volatility, microstructure, VIX)
    - GRU + Luong attention: temporal dependencies over 12-bar (1 hour) lookback
    - Ensemble: 0.70 * LGB + 0.30 * GRU (GRU discarded if val dir_acc < 50%)
    - Dynamic MAE-inverse weighting after 20 completed observations

Strategy:
    1. Every 5 minutes, compute features from last N bars
    2. Ensemble predicts expected return over next 12 bars (1 hour)
    3. If abs(E[r]) > cost_threshold: enter LONG/SHORT
    4. Exit via trailing stop, take-profit, or max hold time (6.5 hours)

Features (22 total):
    Momentum (3):      ret_6bar, ret_12bar, ret_24bar
    Volatility (3):    atr_pct, realized_vol_24bar, bb_pct_b
    Microstructure (4): close_position, vwap_deviation, cumulative_delta, spread_proxy
    Volume (3):        rvol, volume_imbalance_12, overnight_gap
    Macro (2):         vix, vix_chg
    Regime (3):        ret_autocorr_24, time_of_day, intraday_regime
    Relative (1):      sector_rel_strength
    Session (3):       opening_range_pct, gap_return, daily_adx

Label: forward 12-bar (1 hour) return, winsorized at 1st/99th percentile.

Usage (via main.py):
    python main.py train-intraday --symbols SMH,IWM,IGV,QQQ,SOXX --provider alpaca
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger("etf_intraday_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORWARD_BARS = 12           # predict 1 hour ahead (12 x 5min)
MAX_HOLD_BARS = 78          # max 6.5 hours (full trading day)
LOOKBACK_BARS = 60          # need 5 hours of history for features
COST_THRESHOLD = 0.001      # 10bps min signal (ETF costs ~3-12bps RT)
TARGET_RETURN = 0.005       # 0.5% for full position sizing (lower than crypto)

# GRU sequence length: 12 bars (1 hour of context)
GRU_SEQ_LEN = 12
GRU_HIDDEN = 32
GRU_LAYERS = 1
GRU_DROPOUT = 0.1

# Ensemble weights (initial defaults, adapted dynamically after 20 obs)
W_LGB = 0.70
W_GRU = 0.30

FEATURE_NAMES = [
    # Momentum (3) — dropped ret_48bar (weak at 1h horizon, Gao et al. 2018)
    "ret_6bar",              # 30-min return
    "ret_12bar",             # 1-hour return (matches forward target)
    "ret_24bar",             # 2-hour return
    # Volatility (3)
    "atr_pct",               # ATR(12) / close — intraday vol regime
    "realized_vol_24bar",    # realized vol over 24 bars
    "bb_pct_b",              # Bollinger %B — mean-reversion signal
    # Microstructure (4)
    "close_position",        # (close - low) / (high - low) over 12 bars
    "vwap_deviation",        # (close - vwap_12) / close
    "cumulative_delta",      # buy-sell volume imbalance over 12 bars
    "spread_proxy",          # avg((high-low)/close) over 12 bars — liquidity cost
    # Volume (2) — dropped dollar_volume_zscore (redundant w/ rvol)
    "rvol",                  # relative volume vs 48-bar avg
    "volume_imbalance_12",   # buy-sell volume ratio
    # New: overnight gap (Berkman et al. 2012, Lou et al. 2019)
    "overnight_gap",         # open / prev_close - 1 (gap reversal signal)
    # Macro (2) — VIX now fetched at 5-min intraday granularity
    "vix",                   # VIX level (5-min intraday, fallback to daily ffill)
    "vix_chg",               # VIX 12-bar (1-hour) pct change
    # Regime (3) — academic support for intraday regime separation
    "ret_autocorr_24",       # rolling 24-bar return autocorrelation (Lo & MacKinlay 1990)
    "time_of_day",           # sine encoding of minutes-since-open (Heston et al. 2010)
    "intraday_regime",       # 0=open/close momentum (9:30-10:30,15:00-16:00), 1=midday mean-reversion
    # Relative (1)
    "sector_rel_strength",   # ETF 12-bar ret - SPY 12-bar ret
    # Cross-asset lead-lag (2) — Lo & MacKinlay 1990: SPY leads sector ETFs by 2-6 bars
    "spy_lead_3bar",         # SPY 15-min return (propagation lag to sector ETFs)
    "spy_lead_6bar",         # SPY 30-min return
    # Session context (3) — v4 additions (Gao et al. 2018, Berkman et al. 2012)
    "opening_range_pct",     # first 30-min (H-L)/close — session volatility anchor
    "gap_return",            # open vs prev session close — mid-size gaps mean-revert 62-67%
    "daily_adx",             # daily ADX(14) — tells model if today is trending or mean-reverting
]

N_FEATURES = len(FEATURE_NAMES)  # 24

# -- Vol-normalization + cost-adjustment constants --
INTRADAY_VOL_FLOOR = 0.0005  # prevent division by zero in dead midday hours
INTRADAY_VOL_LOOKBACK = 24   # 2 hours of 5-min bars for trailing vol

# -- Intraday cost estimates (round-trip bps) keyed to cost_model.py tiers --
_INTRADAY_RT_COST_BPS: Dict[str, float] = {
    # Liquid: ~3bps RT
    "SPY": 3, "QQQ": 3, "IWM": 3, "IWF": 3, "IWB": 3,
    # Mid: ~7bps RT
    "SMH": 7, "SOXX": 7, "IGV": 7, "XLK": 7,
    # Low: ~12bps RT
    "EWT": 12, "EEM": 12, "MCHI": 12, "GDX": 12, "SLV": 12, "GDXJ": 12,
}
_DEFAULT_INTRADAY_RT_BPS = 7.0  # fallback

# -- Pooled intraday clusters --
INTRADAY_POOL_CLUSTERS: Dict[str, list] = {
    "us_tech": ["SMH", "IGV", "QQQ", "SOXX", "IWM"],
}
INTRADAY_SYMBOL_TO_POOL: Dict[str, str] = {}
for _pool_name, _pool_syms in INTRADAY_POOL_CLUSTERS.items():
    for _sym in _pool_syms:
        INTRADAY_SYMBOL_TO_POOL[_sym] = _pool_name
INTRADAY_CLUSTER_SYMBOL_IDS: Dict[str, Dict[str, int]] = {
    name: {sym: i for i, sym in enumerate(syms)}
    for name, syms in INTRADAY_POOL_CLUSTERS.items()
}

# Multi-horizon forward bars
FORWARD_BARS_6 = 6     # 30-min horizon
FORWARD_BARS_24 = 24   # 2-hour horizon


def get_etf_intraday_feature_cols(symbol: str | None = None) -> list:
    """Return the feature columns for a given symbol.

    SPY gets sector_rel_strength = 0 (self-referential), but keeps
    the column for model compatibility.
    """
    return list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _fetch_vix_daily() -> pd.DataFrame:
    """Fetch VIX daily close from yfinance (fallback for >60d lookback)."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1y")
        if hist.empty:
            return pd.DataFrame()
        df = hist[["Close"]].reset_index()
        df.columns = ["date", "vix_close"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception as exc:
        log.warning("VIX daily fetch failed: %s", exc)
        return pd.DataFrame()


def _fetch_vix_intraday(period: str = "60d") -> pd.DataFrame:
    """Fetch VIX at 5-min intervals from yfinance (max 60-day lookback).

    Returns DataFrame with columns [ts, vix_close] or empty DataFrame on failure.
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period=period, interval="5m")
        if hist.empty or len(hist) < 50:
            return pd.DataFrame()
        df = hist[["Close"]].reset_index()
        df.columns = ["ts", "vix_close"]
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is not None:
            df["ts"] = df["ts"].dt.tz_localize(None)
        log.info("Fetched %d intraday VIX bars (5-min)", len(df))
        return df
    except Exception as exc:
        log.warning("VIX intraday fetch failed: %s — falling back to daily", exc)
        return pd.DataFrame()


class EtfIntradayFeatureEngine:
    """Build feature vectors from 5-minute ETF OHLCV bars."""

    def __init__(self):
        self._vix_intraday_cache: Optional[pd.DataFrame] = None
        self._vix_daily_cache: Optional[pd.DataFrame] = None

    def _get_vix_intraday(self) -> pd.DataFrame:
        """Lazy-load and cache VIX 5-min intraday data."""
        if self._vix_intraday_cache is None:
            self._vix_intraday_cache = _fetch_vix_intraday()
        return self._vix_intraday_cache

    def _get_vix_daily(self) -> pd.DataFrame:
        """Lazy-load and cache VIX daily data (fallback)."""
        if self._vix_daily_cache is None:
            self._vix_daily_cache = _fetch_vix_daily()
        return self._vix_daily_cache

    def build_features(
        self,
        bars: pd.DataFrame,
        spy_bars: Optional[pd.DataFrame] = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Build feature matrix from 5-min bars.

        Args:
            bars: OHLCV DataFrame with columns [ts, open, high, low, close, volume]
            spy_bars: SPY OHLCV for sector-relative strength (None if symbol IS SPY)
            symbol: Symbol name for per-symbol feature routing

        Returns:
            DataFrame with feature columns, indexed by timestamp.
        """
        if len(bars) < LOOKBACK_BARS + 10:
            log.warning("Not enough bars for features: %d (need %d)",
                        len(bars), LOOKBACK_BARS + 10)
            return pd.DataFrame()

        df = bars.copy()
        df = df.sort_values("ts").reset_index(drop=True)

        c = df["close"].astype(float)
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        lo = df["low"].astype(float)
        v = df["volume"].astype(float)

        # --- Momentum (3) ---
        df["ret_6bar"] = c.pct_change(6)
        df["ret_12bar"] = c.pct_change(12)
        df["ret_24bar"] = c.pct_change(24)

        # --- Volatility (3) ---
        # ATR(12)
        tr = pd.concat([
            h - lo,
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_12 = tr.rolling(12).mean()
        df["atr_pct"] = atr_12 / c.clip(lower=1e-10)

        # Realized vol (24 bars, annualized for equity — 78 bars/day × 252 days)
        log_ret = np.log(c / c.shift(1))
        df["realized_vol_24bar"] = log_ret.rolling(24).std() * np.sqrt(78 * 252)

        # Bollinger %B (20, 2)
        sma_20 = c.rolling(20).mean()
        std_20 = c.rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        bb_range = (upper - lower).clip(lower=1e-10)
        df["bb_pct_b"] = (c - lower) / bb_range

        # --- Microstructure (4) ---
        # Close position in range over last 12 bars
        rolling_high_12 = h.rolling(12).max()
        rolling_low_12 = lo.rolling(12).min()
        range_12 = (rolling_high_12 - rolling_low_12).clip(lower=1e-10)
        df["close_position"] = (c - rolling_low_12) / range_12

        # VWAP deviation
        typical_price = (h + lo + c) / 3
        vwap_12 = (typical_price * v).rolling(12).sum() / v.rolling(12).sum().clip(lower=1)
        df["vwap_deviation"] = (c - vwap_12) / c.clip(lower=1e-10)

        # Cumulative delta (buy-sell imbalance over 12 bars)
        hl_range = (h - lo).replace(0, np.nan)
        buy_frac = (c - lo) / hl_range
        buy_frac = buy_frac.fillna(0.5)
        buy_vol = buy_frac * v
        sell_vol = (1 - buy_frac) * v
        delta = buy_vol - sell_vol
        cum_delta_12 = delta.rolling(12).sum()
        total_vol_12 = v.rolling(12).sum()
        df["cumulative_delta"] = cum_delta_12 / total_vol_12.clip(lower=1)

        # Spread proxy: avg((high-low)/close) over 12 bars
        bar_spread = (h - lo) / c.clip(lower=1e-10)
        df["spread_proxy"] = bar_spread.rolling(12).mean()

        # --- Volume (2) ---
        avg_vol_48 = v.rolling(48).mean()
        df["rvol"] = v / avg_vol_48.clip(lower=1)

        # Volume imbalance: net buy-sell ratio over 12 bars
        buy_vol_12 = buy_vol.rolling(12).sum()
        sell_vol_12 = sell_vol.rolling(12).sum()
        total_12 = (buy_vol_12 + sell_vol_12).clip(lower=1)
        df["volume_imbalance_12"] = (buy_vol_12 - sell_vol_12) / total_12

        # --- Overnight gap (Berkman et al. 2012, Lou et al. 2019) ---
        # Detect session boundaries: gap > 30 min between bars
        ts_col = pd.to_datetime(df["ts"])
        time_diff = ts_col.diff().dt.total_seconds().fillna(0)
        session_start = time_diff > 1800  # > 30 min = new session
        # prev_close is the last close of the previous session
        prev_session_close = c.shift(1).where(session_start)
        prev_session_close = prev_session_close.ffill()
        df["overnight_gap"] = (o / prev_session_close.clip(lower=1e-10)) - 1
        df["overnight_gap"] = df["overnight_gap"].fillna(0.0)

        # --- Macro: VIX (2) — hybrid intraday (5-min) + daily fallback ---
        bar_ts = pd.to_datetime(df["ts"])
        if bar_ts.dt.tz is not None:
            bar_ts = bar_ts.dt.tz_localize(None)

        vix_intraday = self._get_vix_intraday()
        if not vix_intraday.empty:
            # merge_asof: for each bar timestamp, find nearest prior VIX reading
            # Normalize both to ns precision to avoid merge dtype mismatch
            bar_frame = pd.DataFrame({"ts": bar_ts.astype("datetime64[ns]")})
            vix_intraday = vix_intraday.copy()
            vix_intraday["ts"] = vix_intraday["ts"].astype("datetime64[ns]")
            vix_sorted = vix_intraday.sort_values("ts")
            merged = pd.merge_asof(
                bar_frame, vix_sorted, on="ts", direction="backward"
            )
            df["vix"] = merged["vix_close"].values
            # Bars before the first intraday VIX reading: fill from daily
            n_missing = df["vix"].isna().sum()
            if n_missing > 0:
                vix_daily = self._get_vix_daily()
                if not vix_daily.empty:
                    bar_date = bar_ts.dt.normalize()
                    vix_series = vix_daily.set_index("date")["vix_close"].sort_index()
                    all_dates = pd.DatetimeIndex(bar_date.unique()).sort_values()
                    vix_reindexed = vix_series.reindex(
                        vix_series.index.union(all_dates)
                    ).sort_index().ffill().bfill()
                    daily_fill = bar_date.map(vix_reindexed).astype(float)
                    df["vix"] = df["vix"].fillna(daily_fill)
                log.info("VIX hybrid: %d bars from intraday, %d filled from daily",
                         len(df) - n_missing, n_missing)
        else:
            # Pure daily fallback (training with >60d lookback)
            vix_daily = self._get_vix_daily()
            if not vix_daily.empty:
                bar_date = bar_ts.dt.normalize()
                vix_series = vix_daily.set_index("date")["vix_close"].sort_index()
                all_dates = pd.DatetimeIndex(bar_date.unique()).sort_values()
                vix_reindexed = vix_series.reindex(
                    vix_series.index.union(all_dates)
                ).sort_index().ffill().bfill()
                df["vix"] = bar_date.map(vix_reindexed).astype(float)
                log.info("VIX: using daily forward-fill (intraday unavailable)")
            else:
                df["vix"] = np.nan

        df["vix"] = df["vix"].ffill().bfill().fillna(20.0)
        # VIX change: 12-bar (1-hour) pct change for intraday dynamics
        df["vix_chg"] = df["vix"].pct_change(12).fillna(0.0)

        # --- Regime (2) ---
        # Return autocorrelation: rolling 24-bar autocorrelation of 5-min returns
        # AC > 0 = momentum regime, AC < 0 = mean-reversion regime (Lo & MacKinlay 1990)
        ret_1bar = c.pct_change(1)
        df["ret_autocorr_24"] = ret_1bar.rolling(24).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 10 else 0.0,
            raw=False,
        ).fillna(0.0)

        # Time-of-day: sine encoding of minutes since market open (9:30 ET)
        # Captures intraday seasonality (Heston et al. 2010): U-shaped vol, L-shaped spread
        bar_ts_local = bar_ts
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        if bar_ts_local.dt.tz is None:
            bar_ts_local = bar_ts_local.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        else:
            bar_ts_local = bar_ts_local.dt.tz_convert("America/New_York")
        minutes_since_open = (bar_ts_local.dt.hour * 60 + bar_ts_local.dt.minute) - 570  # 9:30 = 570
        minutes_since_open = minutes_since_open.clip(lower=0, upper=390)  # 6.5h = 390 min
        df["time_of_day"] = np.sin(np.pi * minutes_since_open / 390)  # 0 at open/close, 1 at midday

        # Intraday regime: 0 = momentum window (open/close), 1 = mean-reversion (midday)
        # Open window: 9:30-10:30 (0-60 min), Close window: 15:00-16:00 (330-390 min)
        # Gao et al. 2018: momentum strongest at open/close, mean-reversion dominates midday
        df["intraday_regime"] = ((minutes_since_open > 60) & (minutes_since_open < 330)).astype(float)

        # --- Session context (3) — v4 additions ---

        # Opening range %: (first 30-min high - low) / close
        # Identifies session volatility early; Gao et al. 2018: Sharpe 1.33 on SPY
        bar_ts_for_session = pd.to_datetime(df["ts"])
        if bar_ts_for_session.dt.tz is not None:
            bar_ts_for_session_et = bar_ts_for_session.dt.tz_convert("America/New_York")
        else:
            bar_ts_for_session_et = bar_ts_for_session.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        bar_date = bar_ts_for_session_et.dt.date
        bar_min = bar_ts_for_session_et.dt.hour * 60 + bar_ts_for_session_et.dt.minute
        # First 30 min = 9:30-10:00 (570-600 minutes)
        is_opening = (bar_min >= 570) & (bar_min < 600)
        # Group by date, compute opening range
        opening_high = h.where(is_opening).groupby(bar_date).transform("max")
        opening_low = lo.where(is_opening).groupby(bar_date).transform("min")
        opening_range = (opening_high - opening_low) / c.clip(lower=1e-10)
        df["opening_range_pct"] = opening_range.ffill().fillna(0.0)

        # Gap return: today's open vs previous session close
        # Mid-size gaps mean-revert 62-67% (Berkman et al. 2012, Lou et al. 2019)
        # Reuse overnight_gap if already computed — they capture the same signal
        df["gap_return"] = df["overnight_gap"]  # already computed above

        # Daily ADX(14): daily-timeframe trend strength indicator
        # Tells intraday model whether the daily context is trending or ranging
        # Compute from daily OHLC aggregated from 5-min bars
        daily_high = h.groupby(bar_date).transform("max")
        daily_low = lo.groupby(bar_date).transform("min")
        daily_close = c.groupby(bar_date).transform("last")
        daily_prev_close = daily_close.shift(1)
        # True range on daily scale
        daily_tr = pd.concat([
            daily_high - daily_low,
            (daily_high - daily_prev_close).abs(),
            (daily_low - daily_prev_close).abs(),
        ], axis=1).max(axis=1)
        # +DM and -DM
        daily_high_diff = daily_high.diff()
        daily_low_diff = -daily_low.diff()
        plus_dm = daily_high_diff.where((daily_high_diff > daily_low_diff) & (daily_high_diff > 0), 0.0)
        minus_dm = daily_low_diff.where((daily_low_diff > daily_high_diff) & (daily_low_diff > 0), 0.0)
        # Smoothed over 14 periods (using EMA for efficiency on bar-level data)
        atr_14 = daily_tr.ewm(span=14 * 78, min_periods=14 * 20).mean()  # ~14 days × 78 bars/day
        plus_di = 100 * plus_dm.ewm(span=14 * 78, min_periods=14 * 20).mean() / atr_14.clip(lower=1e-10)
        minus_di = 100 * minus_dm.ewm(span=14 * 78, min_periods=14 * 20).mean() / atr_14.clip(lower=1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-10)
        df["daily_adx"] = dx.ewm(span=14 * 78, min_periods=14 * 20).mean().fillna(25.0)

        # --- Relative: sector relative strength (1) + lead-lag (2) ---
        is_spy = symbol and symbol.upper() == "SPY"
        if spy_bars is not None and not spy_bars.empty and not is_spy:
            spy = spy_bars.copy().sort_values("ts").reset_index(drop=True)
            spy_close = spy.set_index("ts")["close"].reindex(
                df["ts"], method="ffill"
            ).reset_index(drop=True).astype(float)
            spy_ret_12 = spy_close.pct_change(12)
            df["sector_rel_strength"] = df["ret_12bar"] - spy_ret_12
            # Cross-asset lead-lag: SPY short-term returns predict sector ETF moves
            df["spy_lead_3bar"] = spy_close.pct_change(3)
            df["spy_lead_6bar"] = spy_close.pct_change(6)
        else:
            df["sector_rel_strength"] = 0.0
            df["spy_lead_3bar"] = 0.0
            df["spy_lead_6bar"] = 0.0

        # Select feature columns
        use_cols = get_etf_intraday_feature_cols(symbol)
        result = df[["ts"] + use_cols].copy()

        # Drop rows with NaN in any feature
        valid_mask = result[use_cols].notna().all(axis=1)
        result = result[valid_mask].reset_index(drop=True)

        return result

    def build_training_data(
        self,
        bars: pd.DataFrame,
        spy_bars: Optional[pd.DataFrame] = None,
        symbol: str | None = None,
        vol_normalize: bool = True,
        cost_adjust: bool = True,
        forward_bars: int = FORWARD_BARS,
    ) -> pd.DataFrame:
        """Build training data with features and labels.

        Label: forward N-bar return, optionally vol-normalized and cost-adjusted.
        """
        features = self.build_features(bars, spy_bars, symbol=symbol)
        if features.empty:
            return pd.DataFrame()

        # Align bars to feature timestamps
        bars_indexed = bars.set_index("ts")
        close_aligned = bars_indexed["close"].reindex(
            features["ts"], method="ffill"
        ).astype(float)

        # Forward return
        fwd_ret = close_aligned.pct_change(forward_bars).shift(-forward_bars)

        # Trailing intraday vol for normalization (24-bar = 2h window)
        log_ret = np.log(close_aligned / close_aligned.shift(1))
        trailing_vol = log_ret.rolling(INTRADAY_VOL_LOOKBACK).std()
        trailing_vol = trailing_vol.clip(lower=INTRADAY_VOL_FLOOR)

        if vol_normalize:
            fwd_ret = fwd_ret / trailing_vol

        if cost_adjust and symbol:
            rt_bps = _INTRADAY_RT_COST_BPS.get(
                symbol.upper(), _DEFAULT_INTRADAY_RT_BPS
            )
            cost_frac = rt_bps / 10_000.0
            # Shrink label magnitude by cost (don't flip sign)
            if vol_normalize:
                # Cost in vol-normalized units: cost_frac / trailing_vol
                cost_norm = cost_frac / trailing_vol
                fwd_ret = np.sign(fwd_ret) * (np.abs(fwd_ret) - cost_norm).clip(lower=0)
            else:
                fwd_ret = np.sign(fwd_ret) * (np.abs(fwd_ret) - cost_frac).clip(lower=0)

        features["fwd_return"] = fwd_ret.values
        features["trailing_vol"] = trailing_vol.values

        # Drop rows without labels
        features = features.dropna(subset=["fwd_return"]).reset_index(drop=True)

        return features


# ---------------------------------------------------------------------------
# GRU Model (with Luong attention, shared module)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available — GRU ensemble disabled, LightGBM only")

if TORCH_AVAILABLE:
    from attention import GRUWithAttention as GRUReturnModel


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
class EtfIntradayTrainer:
    """Train LightGBM + GRU ensemble for ETF intraday prediction."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _train_lgb(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        feature_names: List[str],
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple:
        """Train LightGBM with Huber loss (robust to fat-tailed returns)."""
        train_set = lgb.Dataset(
            X_train, label=y_train, feature_name=feature_names,
            weight=sample_weights,
        )
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Auto-set huber_delta from training labels: P90 of |returns|
        # This treats the top 10% of absolute returns as outliers
        huber_delta = float(np.percentile(np.abs(y_train), 90))
        huber_delta = max(huber_delta, 1e-6)  # safety floor
        log.info("Huber delta (auto): %.6f (P90 of |y_train|)", huber_delta)

        params = {
            "objective": "huber",
            "huber_delta": huber_delta,
            "metric": "huber",
            "num_leaves": 15,
            "learning_rate": 0.01,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_data_in_leaf": 100,
            "verbose": -1,
            "seed": 42,
            "lambda_l1": 0.5,          # reduced from 1.0 — Huber handles outlier robustness
            "lambda_l2": 2.0,          # reduced from 5.0
            "max_depth": 4,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=callbacks,
        )

        val_pred = model.predict(X_val)
        return model, val_pred, huber_delta

    def _train_gru(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        seq_len: int = GRU_SEQ_LEN,
    ) -> Tuple:
        """Train GRU model on sequential data.

        Returns (model, scaler_mean, scaler_std, val_predictions) or (None, ...) if failed.
        """
        if not TORCH_AVAILABLE:
            return None, None, None, None

        # Normalize features (per-feature z-score from training set)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0

        X_train_n = (X_train - mean) / std
        X_val_n = (X_val - mean) / std

        # Build sequences
        def _make_sequences(X, y, seq_len):
            Xs, ys = [], []
            for i in range(seq_len, len(X)):
                Xs.append(X[i - seq_len:i])
                ys.append(y[i])
            return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

        X_train_seq, y_train_seq = _make_sequences(X_train_n, y_train, seq_len)
        X_val_seq, y_val_seq = _make_sequences(X_val_n, y_val, seq_len)

        if len(X_train_seq) < 200 or len(X_val_seq) < 50:
            log.warning("Not enough sequences for GRU (train=%d, val=%d)",
                        len(X_train_seq), len(X_val_seq))
            return None, mean, std, None

        device = torch.device("cpu")
        model = GRUReturnModel(n_features=X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train_seq).to(device)
        y_t = torch.from_numpy(y_train_seq).to(device)
        X_v = torch.from_numpy(X_val_seq).to(device)
        y_v = torch.from_numpy(y_val_seq).to(device)

        batch_size = 256
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(60):
            model.train()
            indices = torch.randperm(len(X_t))
            total_loss = 0
            n_batches = 0

            for start in range(0, len(X_t), batch_size):
                idx = indices[start:start + batch_size]
                xb, yb = X_t[idx], y_t[idx]

                pred = model(xb)
                loss = criterion(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = criterion(val_pred, y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info("GRU early stop at epoch %d (best val_loss=%.6f)", epoch, best_val_loss)
                    break

            if epoch % 10 == 0:
                log.info("GRU epoch %d: train_loss=%.6f, val_loss=%.6f",
                         epoch, total_loss / max(n_batches, 1), val_loss)

        # Load best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final val predictions
        model.eval()
        with torch.no_grad():
            val_pred_np = model(X_v).cpu().numpy()

        # Check GRU quality: dir_acc
        dir_correct = ((val_pred_np > 0) == (y_val_seq > 0)).sum()
        gru_dir_acc = dir_correct / len(y_val_seq)
        log.info("GRU val dir_acc=%.3f (threshold=0.52)", gru_dir_acc)

        if gru_dir_acc < 0.52:
            log.warning("GRU dir_acc < 0.52, discarding GRU (LGB-only fallback)")
            return None, mean, std, None

        return model, mean, std, val_pred_np

    def train(
        self,
        symbol: str,
        training_data: pd.DataFrame,
        val_frac: float = 0.2,
        walk_forward: bool = False,
        forward_bars: int = FORWARD_BARS,
    ) -> dict:
        """Train LightGBM + GRU ensemble.

        Returns dict with training metrics.
        """
        if len(training_data) < 500:
            log.warning("%s: only %d samples, need >= 500", symbol, len(training_data))
            return {"error": f"too few samples: {len(training_data)}"}

        feature_cols = get_etf_intraday_feature_cols(symbol)
        n_feat = len(feature_cols)

        if walk_forward:
            wf_split = int(len(training_data) * 0.75)
            training_data = training_data.iloc[:wf_split].copy()
            log.info("%s: walk-forward mode — training on first %d bars (75%%)", symbol, len(training_data))

        X = training_data[feature_cols].values.astype(np.float32)
        y = training_data["fwd_return"].values.astype(np.float32)
        timestamps = pd.to_datetime(training_data["ts"])

        # Time-based split with temporal gap to prevent label leakage.
        split_idx = int(len(X) * (1 - val_frac))
        gap = forward_bars
        train_end_idx = max(split_idx - gap, 0)
        X_train, X_val = X[:train_end_idx], X[split_idx:]
        y_train, y_val = y[:train_end_idx], y[split_idx:]

        # Record date boundaries for OOS verification
        train_start_ts = str(timestamps.iloc[0])
        train_end_ts = str(timestamps.iloc[train_end_idx - 1]) if train_end_idx > 0 else train_start_ts
        val_start_ts = str(timestamps.iloc[split_idx])
        val_end_ts = str(timestamps.iloc[-1])
        log.info("%s: train [%s → %s], gap=%d bars, val [%s → %s]",
                 symbol, train_start_ts[:10], train_end_ts[:10], gap,
                 val_start_ts[:10], val_end_ts[:10])

        # Winsorize labels at 1st/99th percentile — bounds from TRAIN only
        if len(y_train) > 100:
            lo_q = float(np.quantile(y_train, 0.01))
            hi_q = float(np.quantile(y_train, 0.99))
            y_train = np.clip(y_train, lo_q, hi_q)
            y_val = np.clip(y_val, lo_q, hi_q)
            log.info("Label winsorization (train-derived): [%.4f, %.4f]", lo_q, hi_q)

        # Compute label_std for deterministic pred_scale at inference.
        label_std = float(np.std(y_train))
        label_std = max(label_std, 1e-8)
        log.info("Training label std=%.6f (used for pred_scale calibration)", label_std)

        # --- Temporal sample weighting: exponential decay ---
        # Half-life of 20 trading days (~1560 bars at 78 bars/day)
        half_life_bars = 20 * 78
        sample_weights = np.exp(
            -0.693 * np.arange(len(y_train))[::-1] / half_life_bars
        ).astype(np.float32)
        log.info("%s: temporal weighting half-life=%d bars, weight range [%.4f, %.4f]",
                 symbol, half_life_bars, sample_weights.min(), sample_weights.max())

        log.info("%s: Train %d, Val %d, %d features",
                 symbol, len(X_train), len(X_val), n_feat)

        # --- Train LightGBM ---
        log.info("%s: Training LightGBM (fwd=%d bars)...", symbol, forward_bars)
        lgb_model, lgb_val_pred, huber_delta = self._train_lgb(
            X_train, y_train, X_val, y_val, feature_cols,
            sample_weights=sample_weights,
        )

        # --- Train GRU ---
        gru_model, gru_mean, gru_std, gru_val_pred = self._train_gru(
            X_train, y_train, X_val, y_val
        )

        # --- Ensemble ---
        use_gru = gru_model is not None and gru_val_pred is not None
        if use_gru:
            gru_offset = len(lgb_val_pred) - len(gru_val_pred)
            lgb_aligned = lgb_val_pred[gru_offset:]
            y_val_aligned = y_val[gru_offset:]
            ensemble_pred = W_LGB * lgb_aligned + W_GRU * gru_val_pred
            log.info("%s: Using LGB(%.0f%%) + GRU(%.0f%%) ensemble",
                     symbol, W_LGB * 100, W_GRU * 100)
        else:
            ensemble_pred = lgb_val_pred
            y_val_aligned = y_val
            log.info("%s: Using LGB-only (GRU disabled)", symbol)

        # Evaluate
        val_rmse = float(np.sqrt(np.mean((ensemble_pred - y_val_aligned) ** 2)))
        dir_correct = ((ensemble_pred > 0) == (y_val_aligned > 0)).sum()
        dir_acc = float(dir_correct / len(y_val_aligned))

        from scipy.stats import spearmanr
        ic, _ = spearmanr(ensemble_pred, y_val_aligned)
        ic = float(ic) if not np.isnan(ic) else 0.0

        # LGB-only comparison
        lgb_dir_correct = ((lgb_val_pred > 0) == (y_val > 0)).sum()
        lgb_dir_acc = float(lgb_dir_correct / len(y_val))
        lgb_ic, _ = spearmanr(lgb_val_pred, y_val)
        lgb_ic = float(lgb_ic) if not np.isnan(lgb_ic) else 0.0

        # Prediction calibration
        pred_std = float(np.std(ensemble_pred)) if len(ensemble_pred) > 0 else 1e-6
        ret_std = float(np.std(y_val_aligned)) if len(y_val_aligned) > 0 else 1e-6
        ratio_pred_scale = ret_std / max(pred_std, 1e-8)
        ratio_pred_scale = min(ratio_pred_scale, 50.0)
        pred_scale = ratio_pred_scale
        log.info("%s: pred_scale=%.2f (ret_std=%.6f, pred_std=%.6f, label_std=%.6f)",
                 symbol, pred_scale, ret_std, pred_std, label_std)

        # Feature importance
        importance = dict(zip(
            feature_cols,
            lgb_model.feature_importance(importance_type="gain").tolist(),
        ))
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

        # --- Save models ---
        sym_clean = symbol.replace("/", "-")

        # Determine file suffix based on forward horizon
        horizon_suffix = ""
        if forward_bars == FORWARD_BARS_6:
            horizon_suffix = "_6bar"
        elif forward_bars == FORWARD_BARS_24:
            horizon_suffix = "_24bar"

        lgb_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_etf{horizon_suffix}.joblib")
        joblib.dump(lgb_model, lgb_path)

        gru_active = False
        if use_gru and TORCH_AVAILABLE:
            gru_path = os.path.join(self.model_dir, f"{sym_clean}_gru_intraday_etf{horizon_suffix}.pt")
            torch.save({
                "model_state": gru_model.state_dict(),
                "n_features": n_feat,
                "hidden": GRU_HIDDEN,
                "n_layers": GRU_LAYERS,
                "seq_len": GRU_SEQ_LEN,
                "scaler_mean": gru_mean.tolist(),
                "scaler_std": gru_std.tolist(),
            }, gru_path)
            gru_active = True

        # RT cost for this symbol
        rt_bps = _INTRADAY_RT_COST_BPS.get(symbol.upper(), _DEFAULT_INTRADAY_RT_BPS)

        config = {
            "symbol": symbol,
            "feature_names": feature_cols,
            "n_features": n_feat,
            "forward_bars": forward_bars,
            "train_start": train_start_ts,
            "train_end": train_end_ts,
            "val_start": val_start_ts,
            "val_end": val_end_ts,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "val_rmse": round(val_rmse, 6),
            "val_dir_acc": round(dir_acc, 4),
            "val_ic": round(ic, 4),
            "lgb_dir_acc": round(lgb_dir_acc, 4),
            "lgb_ic": round(lgb_ic, 4),
            "lgb_best_iteration": lgb_model.best_iteration,
            "gru_active": gru_active,
            "ensemble_weights": {"lgb": W_LGB, "gru": W_GRU} if gru_active else {"lgb": 1.0},
            "feature_importance": dict(sorted_imp),
            "cost_threshold": COST_THRESHOLD,
            "target_return": TARGET_RETURN,
            "pred_scale": round(pred_scale, 6),
            "label_std": round(label_std, 6),
            "label_vol_normalized": True,
            "label_cost_adjust_bps": rt_bps,
            "temporal_weighting_halflife": half_life_bars,
            "objective": "huber",
            "huber_delta": round(huber_delta, 6),
        }
        config_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_etf{horizon_suffix}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        log.info("%s ENSEMBLE (fwd=%d): RMSE=%.6f, dir_acc=%.3f, IC=%.3f",
                 symbol, forward_bars, val_rmse, dir_acc, ic)
        log.info("%s LGB-only: dir_acc=%.3f, IC=%.3f, best_iter=%d",
                 symbol, lgb_dir_acc, lgb_ic, lgb_model.best_iteration)
        log.info("Top 5 features: %s",
                 ", ".join(f"{k}={v:.1f}" for k, v in sorted_imp[:5]))

        return config


# ---------------------------------------------------------------------------
# Pooled cross-sectional training
# ---------------------------------------------------------------------------
def train_pooled_intraday(
    trainer: EtfIntradayTrainer,
    all_training_data: Dict[str, pd.DataFrame],
    cluster_name: str = "us_tech",
    walk_forward: bool = False,
    forward_bars: int = FORWARD_BARS,
) -> dict:
    """Train a single pooled LightGBM across all symbols in a cluster.

    Adds a symbol_id integer feature so the tree can learn symbol-specific splits.
    """
    cluster_syms = INTRADAY_POOL_CLUSTERS.get(cluster_name, [])
    sym_ids = INTRADAY_CLUSTER_SYMBOL_IDS.get(cluster_name, {})
    if not cluster_syms:
        log.warning("Unknown cluster: %s", cluster_name)
        return {"error": f"unknown cluster {cluster_name}"}

    feature_cols = get_etf_intraday_feature_cols() + ["symbol_id"]
    dfs = []
    for sym in cluster_syms:
        td = all_training_data.get(sym)
        if td is None or td.empty:
            continue
        td = td.copy()
        td["symbol_id"] = sym_ids.get(sym, 0)
        dfs.append(td)

    if not dfs:
        return {"error": "no training data for cluster"}

    pooled = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    log.info("Pooled %s: %d samples from %d symbols", cluster_name, len(pooled), len(dfs))

    if walk_forward:
        wf_split = int(len(pooled) * 0.75)
        pooled = pooled.iloc[:wf_split].copy()

    X = pooled[feature_cols].values.astype(np.float32)
    y = pooled["fwd_return"].values.astype(np.float32)
    timestamps = pd.to_datetime(pooled["ts"])

    split_idx = int(len(X) * 0.8)
    gap = forward_bars
    train_end = max(split_idx - gap, 0)
    X_train, X_val = X[:train_end], X[split_idx:]
    y_train, y_val = y[:train_end], y[split_idx:]

    # Temporal weighting
    half_life = 20 * 78
    weights = np.exp(-0.693 * np.arange(len(y_train))[::-1] / half_life).astype(np.float32)

    # Train LGB
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, weight=weights)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    huber_delta = float(np.percentile(np.abs(y_train), 90))
    huber_delta = max(huber_delta, 1e-6)

    params = {
        "objective": "huber", "huber_delta": huber_delta, "metric": "huber",
        "num_leaves": 31, "learning_rate": 0.01, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5, "min_data_in_leaf": 100,
        "verbose": -1, "seed": 42, "lambda_l1": 0.5, "lambda_l2": 2.0, "max_depth": 5,
    }
    model = lgb.train(
        params, train_set, num_boost_round=1000, valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    val_pred = model.predict(X_val)
    from scipy.stats import spearmanr
    ic, _ = spearmanr(val_pred, y_val)
    ic = float(ic) if not np.isnan(ic) else 0.0
    dir_acc = float(((val_pred > 0) == (y_val > 0)).sum() / len(y_val))

    # Horizon suffix
    horizon_suffix = ""
    if forward_bars == FORWARD_BARS_6:
        horizon_suffix = "_6bar"
    elif forward_bars == FORWARD_BARS_24:
        horizon_suffix = "_24bar"

    # Save
    pool_path = os.path.join(trainer.model_dir, f"{cluster_name}_pool_lgb_intraday{horizon_suffix}.joblib")
    joblib.dump(model, pool_path)
    config = {
        "cluster": cluster_name, "symbols": cluster_syms, "forward_bars": forward_bars,
        "train_samples": int(len(X_train)), "val_samples": int(len(X_val)),
        "val_ic": round(ic, 4), "val_dir_acc": round(dir_acc, 4),
        "feature_names": feature_cols, "label_vol_normalized": True,
    }
    config_path = os.path.join(trainer.model_dir, f"{cluster_name}_pool_lgb_intraday{horizon_suffix}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Pooled %s (fwd=%d): IC=%.3f, dir_acc=%.3f, samples=%d",
             cluster_name, forward_bars, ic, dir_acc, len(X_train))
    return config


# ---------------------------------------------------------------------------
# Predictor (for live trading)
# ---------------------------------------------------------------------------
class EtfIntradayPredictor:
    """Load trained LGB+GRU ensemble and predict on live 5-min bars.

    Dynamic ensemble weighting via inverse-MAE (same approach as swing/crypto).
    """

    _ENSEMBLE_WINDOW = 50
    _MIN_WEIGHT = 0.20
    _MAX_WEIGHT = 0.80
    _MIN_OBS_FOR_DYNAMIC = 20

    def __init__(self, symbol: str, model_dir: str):
        self.symbol = symbol
        self.model_dir = model_dir
        self.lgb_model = None
        self.gru_model = None
        self.gru_mean = None
        self.gru_std = None
        self.gru_seq_len = GRU_SEQ_LEN
        self.config = None
        self.feature_engine = EtfIntradayFeatureEngine()

        # Multi-horizon models (Sprint 2)
        self._lgb_6bar = None
        self._lgb_24bar = None
        # Pooled model (Sprint 2)
        self._pooled_model = None
        self._pooled_feature_names: Optional[List[str]] = None

        # Vol-normalization flag
        self._vol_normalized = False

        # Dynamic ensemble weighting
        from collections import deque
        self._lgb_errors: deque = deque(maxlen=self._ENSEMBLE_WINDOW)
        self._gru_errors: deque = deque(maxlen=self._ENSEMBLE_WINDOW)
        self._pending_preds: deque = deque(maxlen=100)
        self._dynamic_lgb_weight = W_LGB

        self._load()

    def _load(self):
        sym_clean = self.symbol.replace("/", "-")
        lgb_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_etf.joblib")
        gru_path = os.path.join(self.model_dir, f"{sym_clean}_gru_intraday_etf.pt")
        config_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_etf_config.json")

        if not os.path.exists(lgb_path):
            raise FileNotFoundError(f"No ETF intraday model for {self.symbol} at {lgb_path}")

        self.lgb_model = joblib.load(lgb_path)

        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

        # Vol-normalization flag
        if self.config:
            self._vol_normalized = self.config.get("label_vol_normalized", False)

        # Load GRU if active
        if (self.config and self.config.get("gru_active", False)
                and os.path.exists(gru_path) and TORCH_AVAILABLE):
            checkpoint = torch.load(gru_path, map_location="cpu", weights_only=False)
            self.gru_model = GRUReturnModel(
                n_features=checkpoint["n_features"],
                hidden=checkpoint["hidden"],
                n_layers=checkpoint["n_layers"],
            )
            self.gru_model.load_state_dict(checkpoint["model_state"])
            self.gru_model.eval()
            self.gru_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
            self.gru_std = np.array(checkpoint["scaler_std"], dtype=np.float32)
            self.gru_seq_len = checkpoint.get("seq_len", GRU_SEQ_LEN)
            log.info("Loaded GRU ensemble for %s", self.symbol)

        # Load multi-horizon models (6-bar, 24-bar)
        lgb_6_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_etf_6bar.joblib")
        lgb_24_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_etf_24bar.joblib")
        if os.path.exists(lgb_6_path):
            self._lgb_6bar = joblib.load(lgb_6_path)
            log.info("Loaded 6-bar horizon model for %s", self.symbol)
        if os.path.exists(lgb_24_path):
            self._lgb_24bar = joblib.load(lgb_24_path)
            log.info("Loaded 24-bar horizon model for %s", self.symbol)

        # Load pooled model
        pool_name = INTRADAY_SYMBOL_TO_POOL.get(self.symbol.upper())
        if pool_name:
            pool_path = os.path.join(self.model_dir, f"{pool_name}_pool_lgb_intraday.joblib")
            pool_cfg_path = os.path.join(self.model_dir, f"{pool_name}_pool_lgb_intraday_config.json")
            if os.path.exists(pool_path):
                self._pooled_model = joblib.load(pool_path)
                if os.path.exists(pool_cfg_path):
                    with open(pool_cfg_path) as f:
                        pcfg = json.load(f)
                    self._pooled_feature_names = pcfg.get("feature_names")
                log.info("Loaded pooled %s model for %s", pool_name, self.symbol)

        ensemble_str = "LGB+GRU" if self.gru_model else "LGB-only"
        horizons = ["12bar"]
        if self._lgb_6bar:
            horizons.append("6bar")
        if self._lgb_24bar:
            horizons.append("24bar")
        log.info("Loaded ETF intraday %s model for %s (dir_acc=%.3f, IC=%.3f, horizons=%s, pooled=%s, vol_norm=%s)",
                 ensemble_str, self.symbol,
                 self.config.get("val_dir_acc", 0) if self.config else 0,
                 self.config.get("val_ic", 0) if self.config else 0,
                 "+".join(horizons),
                 "yes" if self._pooled_model else "no",
                 "yes" if self._vol_normalized else "no")

    def predict(
        self,
        bars: pd.DataFrame,
        spy_bars: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """Predict expected 1-hour return from latest bars.

        Returns:
            dict with {expected_return, direction, confidence, model_type} or None
        """
        if self.lgb_model is None:
            return None

        features = self.feature_engine.build_features(bars, spy_bars, symbol=self.symbol)
        if features.empty:
            return None

        feature_names = self.config.get("feature_names", FEATURE_NAMES) if self.config else FEATURE_NAMES

        # LGB prediction (last row) — 12-bar (primary)
        avail_cols = [c for c in feature_names if c in features.columns]
        x_lgb = features[avail_cols].iloc[-1:].values.astype(np.float32)
        lgb_pred = float(self.lgb_model.predict(x_lgb)[0])

        # GRU prediction (needs sequence)
        gru_pred = None
        if self.gru_model is not None and len(features) >= self.gru_seq_len:
            x_seq = features[avail_cols].iloc[-self.gru_seq_len:].values.astype(np.float32)
            x_seq_n = (x_seq - self.gru_mean) / np.clip(self.gru_std, 1e-8, None)
            x_tensor = torch.from_numpy(x_seq_n[np.newaxis])

            with torch.no_grad():
                gru_pred = float(self.gru_model(x_tensor).item())

        # --- Dynamic ensemble weighting ---
        if gru_pred is not None:
            if (len(self._lgb_errors) >= self._MIN_OBS_FOR_DYNAMIC and
                    len(self._gru_errors) >= self._MIN_OBS_FOR_DYNAMIC):
                lgb_mae = sum(self._lgb_errors) / len(self._lgb_errors)
                gru_mae = sum(self._gru_errors) / len(self._gru_errors)
                if lgb_mae > 0 and gru_mae > 0:
                    w_lgb = (1 / lgb_mae) / (1 / lgb_mae + 1 / gru_mae)
                    w_lgb = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, w_lgb))
                    self._dynamic_lgb_weight = w_lgb
            w_lgb = self._dynamic_lgb_weight
            w_gru = 1.0 - w_lgb
            expected_return = w_lgb * lgb_pred + w_gru * gru_pred
            model_type = f"lgb({w_lgb:.0%})+gru({w_gru:.0%})"
        else:
            expected_return = lgb_pred
            model_type = "lgb"

        # --- Multi-horizon consensus (Sprint 2) ---
        pred_6 = None
        pred_24 = None
        if self._lgb_6bar is not None:
            pred_6 = float(self._lgb_6bar.predict(x_lgb)[0])
        if self._lgb_24bar is not None:
            pred_24 = float(self._lgb_24bar.predict(x_lgb)[0])

        if pred_6 is not None and pred_24 is not None:
            signs = [np.sign(pred_6), np.sign(expected_return), np.sign(pred_24)]
            agreement = sum(1 for s in signs if s == signs[0])
            if agreement == 3:
                # All agree: blend 0.25/0.50/0.25
                expected_return = 0.25 * pred_6 + 0.50 * expected_return + 0.25 * pred_24
                model_type += "+mh3"
            elif agreement >= 2:
                # 2/3 agree: use only the majority direction at 50% conviction
                expected_return *= 0.5
                model_type += "+mh2"
            else:
                # No agreement: use primary only at reduced size
                expected_return *= 0.3
                model_type += "+mh0"
        elif pred_6 is not None:
            # Only 6+12 available
            if np.sign(pred_6) == np.sign(expected_return):
                expected_return = 0.40 * pred_6 + 0.60 * expected_return
                model_type += "+mh6"

        # --- Pooled model blend (Sprint 2) ---
        if self._pooled_model is not None and self._pooled_feature_names:
            pool_name = INTRADAY_SYMBOL_TO_POOL.get(self.symbol.upper(), "")
            sym_id = INTRADAY_CLUSTER_SYMBOL_IDS.get(pool_name, {}).get(self.symbol.upper(), 0)
            # Build pooled feature vector
            pool_row = features[avail_cols].iloc[-1:].copy()
            pool_row["symbol_id"] = sym_id
            pool_cols = [c for c in self._pooled_feature_names if c in pool_row.columns]
            x_pool = pool_row[pool_cols].values.astype(np.float32)
            pooled_pred = float(self._pooled_model.predict(x_pool)[0])
            # 70% per-symbol, 30% pooled
            expected_return = 0.70 * expected_return + 0.30 * pooled_pred
            model_type += "+pool"

        # --- Evaluate pending predictions ---
        current_ts = features["ts"].iloc[-1]
        current_price = float(bars["close"].iloc[-1])
        resolved = []
        for pp in self._pending_preds:
            pred_ts, lgb_p, gru_p, entry_px = pp
            try:
                elapsed = (pd.Timestamp(current_ts) - pd.Timestamp(pred_ts)).total_seconds()
            except (TypeError, ValueError):
                continue
            if elapsed >= FORWARD_BARS * 5 * 60 and entry_px > 0:
                realized = current_price / entry_px - 1
                self._lgb_errors.append(abs(lgb_p - realized))
                if gru_p is not None:
                    self._gru_errors.append(abs(gru_p - realized))
                resolved.append(pp)
        for pp in resolved:
            self._pending_preds.remove(pp)

        self._pending_preds.append((current_ts, lgb_pred, gru_pred, current_price))

        # Calibrate with pred_scale
        pred_scale = self.config.get("pred_scale", 1.0) if self.config else 1.0
        expected_return *= pred_scale

        # Vol-denormalization: multiply back by current trailing vol
        if self._vol_normalized:
            close_prices = bars["close"].astype(float)
            log_ret = np.log(close_prices / close_prices.shift(1))
            cur_vol = log_ret.rolling(INTRADAY_VOL_LOOKBACK).std().iloc[-1]
            if np.isnan(cur_vol) or cur_vol < INTRADAY_VOL_FLOOR:
                cur_vol = INTRADAY_VOL_FLOOR
            expected_return = expected_return * cur_vol

        direction = "UP" if expected_return > 0 else "DOWN"
        confidence = min(1.0, abs(expected_return) / TARGET_RETURN)

        return {
            "expected_return": expected_return,
            "direction": direction,
            "confidence": confidence,
            "model_type": model_type,
            "lgb_pred": lgb_pred,
            "gru_pred": gru_pred,
            "pred_6bar": pred_6,
            "pred_24bar": pred_24,
            "timestamp": str(features["ts"].iloc[-1]),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ETF intraday model training (LGB+GRU ensemble)")
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols (e.g. SMH,IWM,QQQ)")
    parser.add_argument("--days", type=int, default=180,
                        help="Days of history to fetch (default: 180)")
    parser.add_argument("--save-dir", default=None,
                        help="Model save directory")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Walk-forward split: train on first 75%%, OOS on last 25%%")
    parser.add_argument("--train-end", default=None,
                        help="Hard cutoff date (YYYY-MM-DD). Only train on data BEFORE this date.")
    parser.add_argument("--provider", default="alpaca",
                        help="Data provider (default: alpaca)")
    args = parser.parse_args()

    from signals_engine import PROJECT_ROOT, build_adapter

    # Default model dir
    model_dir = args.save_dir
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, "models", "intraday")
    os.makedirs(model_dir, exist_ok=True)

    adapter = build_adapter(args.provider)
    trainer = EtfIntradayTrainer(model_dir)
    symbols = [s.strip() for s in args.symbols.split(",")]

    # Fetch SPY bars once (for sector-relative strength)
    log.info("Fetching SPY reference bars...")
    spy_bars = adapter.fetch_intraday("SPY", "5min", lookback_days=args.days)
    if spy_bars is not None and not spy_bars.empty:
        # Standardize column names
        if "timestamp" in spy_bars.columns:
            spy_bars = spy_bars.rename(columns={"timestamp": "ts"})
        # Apply train-end cutoff to SPY reference bars too
        if args.train_end and "ts" in spy_bars.columns:
            cutoff_ts = pd.Timestamp(args.train_end)
            spy_bars = spy_bars[pd.to_datetime(spy_bars["ts"]) < cutoff_ts].copy()
        log.info("SPY reference bars: %d", len(spy_bars))
    else:
        log.warning("Could not fetch SPY bars — sector_rel_strength will be 0")
        spy_bars = None

    results = []
    all_training_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        log.info("=== Training ETF intraday for %s ===", sym)
        try:
            # Fetch bars
            bars = adapter.fetch_intraday(sym, "5min", lookback_days=args.days)
            if bars is None or len(bars) < 1000:
                log.warning("%s: only %d bars, skipping", sym, len(bars) if bars is not None else 0)
                continue

            # Standardize column names
            if "timestamp" in bars.columns:
                bars = bars.rename(columns={"timestamp": "ts"})

            # Apply train-end cutoff: only use data BEFORE cutoff date
            if args.train_end:
                cutoff_ts = pd.Timestamp(args.train_end)
                if "ts" in bars.columns:
                    bars = bars[pd.to_datetime(bars["ts"]) < cutoff_ts].copy()
                log.info("%s: train-end cutoff %s → %d bars", sym, args.train_end, len(bars))
                if len(bars) < 1000:
                    log.warning("%s: only %d bars after cutoff filter, skipping", sym, len(bars))
                    continue

            # Build training data (vol-normalized + cost-adjusted)
            engine = EtfIntradayFeatureEngine()
            is_spy = sym.upper() == "SPY"

            # Primary: 12-bar forward
            td_12 = engine.build_training_data(
                bars, spy_bars=None if is_spy else spy_bars,
                symbol=sym, vol_normalize=True, cost_adjust=True,
                forward_bars=FORWARD_BARS,
            )
            if len(td_12) < 500:
                log.warning("%s: only %d training samples, skipping", sym, len(td_12))
                continue

            all_training_data[sym] = td_12
            log.info("%s: %d training samples (12-bar)", sym, len(td_12))

            # Train primary 12-bar model
            metrics = trainer.train(sym, td_12, walk_forward=args.walk_forward, forward_bars=FORWARD_BARS)
            results.append(metrics)

            # Train 6-bar horizon model
            td_6 = engine.build_training_data(
                bars, spy_bars=None if is_spy else spy_bars,
                symbol=sym, vol_normalize=True, cost_adjust=True,
                forward_bars=FORWARD_BARS_6,
            )
            if len(td_6) >= 500:
                log.info("%s: Training 6-bar horizon...", sym)
                trainer.train(sym, td_6, walk_forward=args.walk_forward, forward_bars=FORWARD_BARS_6)

            # Train 24-bar horizon model
            td_24 = engine.build_training_data(
                bars, spy_bars=None if is_spy else spy_bars,
                symbol=sym, vol_normalize=True, cost_adjust=True,
                forward_bars=FORWARD_BARS_24,
            )
            if len(td_24) >= 500:
                log.info("%s: Training 24-bar horizon...", sym)
                trainer.train(sym, td_24, walk_forward=args.walk_forward, forward_bars=FORWARD_BARS_24)

        except Exception as exc:
            log.error("%s: training failed: %s", sym, exc, exc_info=True)
            continue

    # Train pooled cross-sectional models
    for cluster_name in INTRADAY_POOL_CLUSTERS:
        log.info("=== Training pooled intraday %s ===", cluster_name)
        try:
            for fwd in (FORWARD_BARS, FORWARD_BARS_6, FORWARD_BARS_24):
                train_pooled_intraday(
                    trainer, all_training_data, cluster_name,
                    walk_forward=args.walk_forward, forward_bars=fwd,
                )
        except Exception as exc:
            log.error("Pooled %s failed: %s", cluster_name, exc, exc_info=True)

    # Summary
    log.info("\n=== Training Summary ===")
    for r in results:
        if "error" in r:
            log.info("  %s: ERROR — %s", r.get("symbol", "?"), r["error"])
        else:
            log.info("  %s: dir_acc=%.3f, IC=%.3f, GRU=%s",
                     r["symbol"], r["val_dir_acc"], r["val_ic"],
                     "active" if r["gru_active"] else "off")


if __name__ == "__main__":
    main()
