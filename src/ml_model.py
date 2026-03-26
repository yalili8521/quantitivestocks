#!/usr/bin/env python3
"""
LSTM ML Model for ETF Forward Return Prediction (v2 Regression)
================================================================
Predicts forward 10-day expected return using an LSTM neural network.
Replaces v1 binary classification with continuous regression target.

Usage (via main.py):
    python main.py train   --symbol SPY --provider yahoo --epochs 50
    python main.py predict --symbol SPY --provider yahoo

Requires: torch (PyTorch), signals_engine.py in src/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader

# Import from signals_engine (same package)
from signals_engine import (
    DataAdapter,
    FREDVixFetcher,
    build_adapter,
    compute_rsi,
    compute_atr,
    compute_macd,
    compute_bollinger_bands,
    compute_adx,
    compute_momentum_quality,
    compute_hurst_series,
    RSI_PERIOD,
    DAILY_LOOKBACK,
    PROJECT_ROOT,
)
from alpha_signals import CBOEHistoricalFetcher, CBOE_FEATURES
from factor_signals import FactorFeatureBuilder, get_factor_features
from market_signals import MarketSignalBuilder, get_market_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN = 20  # 20-bar lookback window for LSTM input

# Regression constants (v2)
FORWARD_DAYS = 10           # predict 10-day forward return
COST_THRESHOLD = 0.001      # 0.1% minimum expected return to trade
TARGET_RETURN = 0.02        # 2% return = full position size

# Meta-labeling threshold (deprecated in v2 — kept for backward compat)
META_THRESHOLD = 0.55

# Full feature list (22) — used for building all indicators; subset by mode/symbol below
FEATURE_COLS = [
    "rsi14", "ret5", "ret10", "wk_ret", "mo_ret", "vol20",
    "log_dollar_vol", "vix", "vix_chg",
    "vol_imbalance", "vwap_ratio", "dv_accel", "spread_proxy",
    "atr_pct", "macd_hist_norm", "bb_pct_b", "bb_bandwidth",
    "adx", "momentum_quality", "vol_regime", "trend_strength",
    "frac_diff_close",   # FFD of log-price — stationarity-preserving momentum
]

# Mode-specific feature subsets (from MDA permutation importance analysis)
# Daily mode: mean-reversion + volatility structure dominate
FEATURE_COLS_DAILY = [
    "bb_bandwidth", "vol20", "ret5", "bb_pct_b", "wk_ret", "dv_accel",
    "rsi14", "ret10", "adx", "macd_hist_norm", "vwap_ratio", "vol_regime",
    "hurst",
]
# Intraday mode: trend + liquidity + macro context dominate
FEATURE_COLS_INTRADAY = [
    "trend_strength", "adx", "log_dollar_vol", "spread_proxy", "vol20",
    "atr_pct", "vix_chg", "vix", "ret5", "bb_pct_b", "wk_ret", "mo_ret",
    "momentum_quality",
]


# Symbol-specific feature overrides.
# IGV (software/tech ETF) and FXI (China large-cap) underperformed with the
# 12-feature daily set because they are more sensitive to macro regime (VIX) and
# momentum signals that were pruned by the cross-symbol MDA average.
# These symbols get 17 features: 12 base daily + 5 VIX/momentum columns.
SYMBOL_FEATURE_OVERRIDES: dict = {
    # IGV and FXI: extended with VIX + momentum (17 features)
    "IGV": FEATURE_COLS_DAILY + ["vix", "vix_chg", "trend_strength", "momentum_quality", "mo_ret"],
    "FXI": FEATURE_COLS_DAILY + ["vix", "vix_chg", "trend_strength", "momentum_quality", "mo_ret"],
    # GLD, SLV, TLT: non-stationary assets — add FFD of log-price (13 features)
    # frac_diff_close makes the LSTM see a stationary, memory-preserving price signal
    # rather than learning spurious trends from the raw random-walk price.
    "GLD": FEATURE_COLS_DAILY + ["frac_diff_close"],
    "SLV": FEATURE_COLS_DAILY + ["frac_diff_close"],
    "TLT": FEATURE_COLS_DAILY + ["frac_diff_close"],
}


def get_feature_cols(mode: str, symbol: str | None = None) -> list:
    """Return the feature list for a given mode (and optionally symbol).

    Order of precedence:
      1. Symbol-level override (e.g. IGV/FXI extended daily set)
      2. Mode-level default (FEATURE_COLS_DAILY or FEATURE_COLS_INTRADAY)

    Daily mode always appends CBOE alpha features (VIX term structure + SKEW).
    """
    if symbol and symbol in SYMBOL_FEATURE_OVERRIDES:
        cols = list(SYMBOL_FEATURE_OVERRIDES[symbol])
    else:
        cols = list(FEATURE_COLS_INTRADAY if mode == "intraday" else FEATURE_COLS_DAILY)
    # Add CBOE features for daily mode (VIX term structure + SKEW)
    if mode != "intraday":
        cols.extend(CBOE_FEATURES)
        cols.extend(get_factor_features(symbol or "SPY"))
        cols.extend(get_market_features(symbol or "SPY"))
    return cols

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ===================================================================
# LSTM Model
# ===================================================================
class TemporalAttention(nn.Module):
    """Soft attention over LSTM time steps — lets the model focus on key bars."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)
        return context


class ReturnLSTM(nn.Module):
    """LSTM with temporal attention for forward return regression (v2).

    Architecture:
        Input:      (batch, seq_len, n_features)
        LayerNorm:  normalize input features
        LSTM:       2-layer, hidden_size=96, dropout=0.25
        Attention:  soft attention over all time steps
        FC:         96 -> 48 -> ReLU -> Dropout -> 1 (linear, no sigmoid)

    Output is raw expected return (e.g., +0.012 = +1.2% over FORWARD_DAYS).
    """

    def __init__(self, n_features: int, hidden_size: int = 96,
                 num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
            # No activation — raw return output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.regressor(context).squeeze(-1)


# Backward compatibility alias
DirectionLSTM = ReturnLSTM


# ===================================================================
# Fractional Differentiation (López de Prado, AFML Chapter 5)
# ===================================================================
def frac_diff_ffd(series: pd.Series, d: float = 0.4, thres: float = 1e-4) -> pd.Series:
    """Fixed-width window fractional differentiation.

    Produces a stationary version of a price series while preserving maximum
    long-range memory.  Used for trend-following assets (GLD, SLV, TLT) where
    raw prices are non-stationary but standard differencing over-destroys memory.

    Parameters
    ----------
    series : log-price or price series (pd.Series, float values)
    d      : fractional order ∈ (0, 1).  d≈0.4 achieves stationarity while
             retaining ~60% of the original autocorrelation structure.
    thres  : weight cutoff — controls effective window width (default 1e-4 ≈ 30–50 bars)

    Returns
    -------
    pd.Series with same index; first W-1 values are NaN (warmup).
    """
    # Build convolution weights w_k = prod_{j=0}^{k-1} (d-j)/(j+1)  * (-1)^k
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1], dtype=np.float64)   # oldest weight first
    width = len(w)

    arr = series.values.astype(np.float64)
    out = np.full(len(arr), np.nan)
    for i in range(width - 1, len(arr)):
        window = arr[i - width + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(w, window))
    return pd.Series(out, index=series.index)


# ===================================================================
# Feature Engineering
# ===================================================================
class FeatureEngine:
    """Builds rolling feature matrix from OHLCV + VIX data (daily or intraday)."""

    def __init__(self):
        self._scaler_params: Optional[Dict] = None

    def build_features(self, bars_df: pd.DataFrame,
                       vix_df: pd.DataFrame,
                       mode: str = "daily",
                       symbol: str | None = None) -> pd.DataFrame:
        """Build 13-feature matrix from bars + VIX.

        Parameters
        ----------
        bars_df : DataFrame [symbol, ts, open, high, low, close, volume]
        vix_df  : DataFrame [date, vix] from FREDVixFetcher
        mode    : "daily" or "intraday"
                  For intraday, annualization uses sqrt(78*252) and VIX is forward-filled.

        Returns
        -------
        DataFrame with columns = FEATURE_COLS (warm-up rows dropped).
        """
        close = bars_df["close"].astype(float)
        high = bars_df["high"].astype(float)
        low = bars_df["low"].astype(float)
        volume = bars_df["volume"].astype(float)
        df = pd.DataFrame(index=bars_df.index)

        # Annualization factor: daily=sqrt(252), intraday=sqrt(78*252)
        annualize = np.sqrt(78 * 252) if mode == "intraday" else np.sqrt(252)

        # Existing indicators (rolling)
        df["rsi14"] = compute_rsi(close, RSI_PERIOD) / 100.0
        df["ret5"] = close.pct_change(5)
        df["ret10"] = close.pct_change(10)
        df["wk_ret"] = close.pct_change(5)
        df["mo_ret"] = close.pct_change(21)
        df["vol20"] = close.pct_change().rolling(20).std() * annualize

        # Dollar volume (log-scaled)
        dv = close * volume
        df["log_dollar_vol"] = np.log10(dv.replace(0, np.nan))

        # VIX — merge on date (forward-fill for intraday)
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        vix_map = {}
        if not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_map[d] = row["vix"]
        df["vix"] = bar_dates.map(lambda d: vix_map.get(d, np.nan)).values
        df["vix"] = df["vix"].ffill()
        if df["vix"].isna().all():
            log.warning("VIX column is all-NaN after forward-fill — features may be corrupted")
        df["vix_chg"] = df["vix"].pct_change(fill_method=None)

        # Order-flow features
        hl_spread = (high - low).replace(0, np.nan)
        buy_frac = (close - low) / hl_spread
        df["vol_imbalance"] = (2 * buy_frac - 1).fillna(0)

        typical = (high + low + close) / 3
        cum_tp_vol = (typical * volume).rolling(5).sum()
        cum_vol = volume.rolling(5).sum()
        rolling_vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        df["vwap_ratio"] = close / rolling_vwap

        dv_ma_5 = dv.rolling(5).mean()
        dv_ma_10 = dv.rolling(10).mean()
        df["dv_accel"] = (dv_ma_5 - dv_ma_10) / dv_ma_10.replace(0, np.nan)

        df["spread_proxy"] = (high - low) / close.replace(0, np.nan)

        # --- Enhanced features ---
        atr = compute_atr(high, low, close, period=14)
        df["atr_pct"] = atr / close.replace(0, np.nan)

        _, _, macd_hist = compute_macd(close)
        df["macd_hist_norm"] = macd_hist / close.replace(0, np.nan)

        _, _, _, pct_b, bandwidth = compute_bollinger_bands(close, window=20)
        df["bb_pct_b"] = pct_b
        df["bb_bandwidth"] = bandwidth

        df["adx"] = compute_adx(high, low, close, period=14) / 100.0

        df["momentum_quality"] = compute_momentum_quality(close, window=20)

        vol_short = close.pct_change().rolling(10).std() * annualize
        vol_long = close.pct_change().rolling(30).std() * annualize
        df["vol_regime"] = vol_short / vol_long.replace(0, np.nan)

        df["hurst"] = compute_hurst_series(close, window=252, step=5)

        ema_fast = close.ewm(span=10, adjust=False).mean()
        ema_slow = close.ewm(span=30, adjust=False).mean()
        df["trend_strength"] = (ema_fast - ema_slow) / close.replace(0, np.nan)

        # Fractional differentiation of log-price (d=0.4, FFD algorithm)
        # Makes price signal stationary while preserving ~60% of long-range memory.
        # Used by GLD, SLV, TLT overrides; computed for all (cheap, ~30 bar warmup).
        log_close = np.log(close.replace(0, np.nan))
        df["frac_diff_close"] = frac_diff_ffd(log_close, d=0.4, thres=1e-4)

        # --- CBOE alpha features (daily mode only) ---
        if mode != "intraday":
            try:
                if not hasattr(self, "_cboe_fetcher") or self._cboe_fetcher is None:
                    self._cboe_fetcher = CBOEHistoricalFetcher(
                        os.path.join(PROJECT_ROOT, "data"))
                bar_dates_alpha = pd.to_datetime(bars_df["ts"]).dt.date
                cboe_df = self._cboe_fetcher.build_features(bar_dates_alpha)
                for col in cboe_df.columns:
                    df[col] = cboe_df[col]
            except Exception as exc:
                log.warning("CBOE alpha features failed: %s", exc)
                for col in CBOE_FEATURES:
                    if col not in df.columns:
                        df[col] = np.nan

        # --- Factor signal features (daily mode only) ---
        if mode != "intraday" and symbol:
            try:
                if not hasattr(self, "_factor_builder") or self._factor_builder is None:
                    self._factor_builder = FactorFeatureBuilder()
                factor_df = self._factor_builder.build_features(bars_df, symbol)
                for col in factor_df.columns:
                    df[col] = factor_df[col]
            except Exception as exc:
                log.warning("Factor features failed for %s: %s", symbol, exc)
                for col in get_factor_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # --- Market signal features (daily mode only) ---
        if mode != "intraday" and symbol:
            try:
                if not hasattr(self, "_market_builder") or self._market_builder is None:
                    self._market_builder = MarketSignalBuilder()
                market_df = self._market_builder.build_features(bars_df, symbol)
                for col in market_df.columns:
                    df[col] = market_df[col]
            except Exception as exc:
                log.warning("Market features failed for %s: %s", symbol, exc)
                for col in get_market_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # Drop warm-up rows using only the columns actually needed for this symbol/mode,
        # then return that same subset.  Using the global FEATURE_COLS would drop all rows
        # for symbols that don't use frac_diff_close (window = 282 bars) even when
        # prediction is called with a short history window.
        relevant_cols = get_feature_cols(mode, symbol)
        df = df.dropna(subset=relevant_cols)
        return df[relevant_cols]

    def fit_scaler(self, features_df: pd.DataFrame) -> None:
        """Compute per-column mean and std from training data."""
        self._scaler_params = {
            "mean": features_df.mean(),
            "std": features_df.std().replace(0, 1),
        }

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize using stored scaler params."""
        if self._scaler_params is None:
            raise RuntimeError("Call fit_scaler() first.")
        return (features_df - self._scaler_params["mean"]) / self._scaler_params["std"]

    def save_scaler(self, path: str) -> None:
        """Persist scaler params alongside model weights."""
        data = {
            "mean": self._scaler_params["mean"].to_dict(),
            "std": self._scaler_params["std"].to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_scaler(self, path: str) -> None:
        """Load scaler params from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._scaler_params = {
            "mean": pd.Series(data["mean"]),
            "std": pd.Series(data["std"]),
        }
        n_mean = len(self._scaler_params["mean"])
        n_std = len(self._scaler_params["std"])
        if n_mean != n_std:
            raise ValueError(
                f"Scaler mean/std length mismatch in {path}: {n_mean} vs {n_std}"
            )


# ===================================================================
# Training data preparation
# ===================================================================
def prepare_sequences_triple_barrier(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    pt_pct: float = 0.015,
    sl_pct: float = 0.010,
    horizon: int = 5,
) -> tuple:
    """Triple-barrier labeling (López de Prado, *AFML*, Chapter 3).

    For each sample ending at bar t, look forward up to `horizon` bars:
      - Label 1 (UP):   price rises >= pt_pct before falling -sl_pct  (profit target)
      - Label 0 (DOWN): price falls >= sl_pct before rising +pt_pct   (stop loss)
      - Timeout:        neither barrier hit → label by final direction

    This is far more informative than next-bar binary prediction because:
      - Labels reflect economically meaningful price moves (not 1-bar noise)
      - Timeout labels still carry directional signal
      - Confidence of the training signal is higher (barrier = conviction)

    Uses full bars_df for look-forward close prices, avoiding boundary issues
    when features_df is a slice of the full dataset.
    """
    feature_values = features_df.values
    full_close = bars_df["close"].astype(float).values
    # Map features_df row positions → positions in bars_df
    bar_positions = bars_df.index.get_indexer(features_df.index)
    n = len(feature_values)

    X_list: list = []
    y_list: list = []

    for i in range(n - seq_len):
        entry_feat_pos = i + seq_len - 1          # last row of this window in features
        entry_bar_pos  = bar_positions[entry_feat_pos]  # position in full bars

        X_list.append(feature_values[i: i + seq_len])

        if entry_bar_pos < 0 or entry_bar_pos + horizon >= len(full_close):
            # Not enough forward data — fall back to next-bar direction
            if 0 <= entry_bar_pos + 1 < len(full_close):
                curr = full_close[entry_bar_pos]
                nxt  = full_close[entry_bar_pos + 1]
                y_list.append(1.0 if nxt > curr else 0.0)
            else:
                y_list.append(0.5)
            continue

        entry_price = full_close[entry_bar_pos]
        if entry_price <= 0:
            y_list.append(0.5)
            continue

        label = None
        for fwd in range(1, horizon + 1):
            fwd_pos = entry_bar_pos + fwd
            if fwd_pos >= len(full_close):
                break
            ret = (full_close[fwd_pos] - entry_price) / entry_price
            if ret >= pt_pct:
                label = 1.0   # profit target hit → UP
                break
            elif ret <= -sl_pct:
                label = 0.0   # stop loss hit → DOWN
                break

        if label is None:   # timeout → final direction
            final_pos = min(entry_bar_pos + horizon, len(full_close) - 1)
            final_ret = (full_close[final_pos] - entry_price) / entry_price
            label = 1.0 if final_ret > 0 else 0.0

        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def prepare_sequences_regression(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    forward_days: int = FORWARD_DAYS,
) -> tuple:
    """Forward N-day return regression labels (v2).

    For each sample ending at bar t:
        y = (close[t + forward_days] - close[t]) / close[t]

    This is the raw forward return — a continuous value, not binary.
    Winsorized at +/- 10% to reduce outlier influence during training.
    """
    feature_values = features_df.values
    full_close = bars_df["close"].astype(float).values
    bar_positions = bars_df.index.get_indexer(features_df.index)
    n = len(feature_values)

    X_list: list = []
    y_list: list = []

    for i in range(n - seq_len):
        entry_feat_pos = i + seq_len - 1
        entry_bar_pos = bar_positions[entry_feat_pos]

        if entry_bar_pos < 0 or entry_bar_pos + forward_days >= len(full_close):
            continue

        entry_price = full_close[entry_bar_pos]
        future_price = full_close[entry_bar_pos + forward_days]

        if entry_price <= 0:
            continue

        forward_return = (future_price - entry_price) / entry_price
        X_list.append(feature_values[i: i + seq_len])
        y_list.append(forward_return)

    y_arr = np.array(y_list, dtype=np.float32)

    # Percentile-based Winsorization (1st/99th) — more robust than fixed ±10%
    if len(y_arr) > 20:
        p1, p99 = np.percentile(y_arr, [1, 99])
        y_arr = np.clip(y_arr, p1, p99)
        log.info("Label Winsorization: clipped to [%.4f, %.4f] (1st/99th pctile)", p1, p99)

    return np.array(X_list, dtype=np.float32), y_arr


# ===================================================================
# VIX history helper (longer lookback for training)
# ===================================================================
def _fetch_vix_for_training(fred_key: Optional[str], lookback_days: int,
                            include_live: bool = True) -> pd.DataFrame:
    """Fetch VIX history. Try FRED first, fall back to yfinance ^VIX.

    Args:
        include_live: If True (default), append today's live VIX price for
            real-time trading. Set False during model training to prevent
            future VIX data from leaking into historical feature rows.
    """
    fetcher = FREDVixFetcher(api_key=fred_key)
    vix_df = fetcher.fetch(lookback_days=lookback_days)
    if len(vix_df) < 20:
        log.info("FRED VIX data sparse (%d rows); falling back to yfinance ^VIX.", len(vix_df))
        try:
            import yfinance as yf
            ticker = yf.Ticker("^VIX")
            cal_days = int(lookback_days * 1.5) + 10
            hist = ticker.history(period=f"{cal_days}d", interval="1d")
            if hist is not None and not hist.empty:
                vdf = pd.DataFrame({
                    "date": hist.index,
                    "vix": hist["Close"].values,
                })
                if vdf["date"].dt.tz is not None:
                    vdf["date"] = vdf["date"].dt.tz_localize(None)
                vdf = vdf.sort_values("date").reset_index(drop=True)
                log.info("Fetched %d VIX rows from yfinance.", len(vdf))
                vix_df = vdf
        except Exception as exc:
            log.warning("yfinance ^VIX fallback failed: %s", exc)

    # Append today's live VIX price so intraday spikes are captured.
    # IMPORTANT: Only for live trading. During training, this leaks future data.
    if include_live:
        try:
            import yfinance as yf
            _vix_ticker = yf.Ticker("^VIX")
            live_vix = float(_vix_ticker.fast_info.last_price)
            if live_vix > 0:
                today = pd.Timestamp.now().normalize()
                last_date = vix_df["date"].iloc[-1].normalize() if not vix_df.empty else pd.Timestamp("1900-01-01")
                if last_date < today:
                    today_row = pd.DataFrame({"date": [today], "vix": [live_vix]})
                    vix_df = pd.concat([vix_df, today_row], ignore_index=True)
                    log.info("Appended live intraday VIX=%.1f for today.", live_vix)
                else:
                    vix_df.loc[vix_df.index[-1], "vix"] = live_vix
                    log.info("Updated today's VIX row to live value=%.1f.", live_vix)
        except Exception as exc:
            log.debug("Live VIX fetch failed (using daily close): %s", exc)

    return vix_df


# ===================================================================
# Training pipeline
# ===================================================================
def train_model(
    symbol: str,
    adapter: DataAdapter,
    fred_key: Optional[str] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seq_len: int = SEQ_LEN,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
    mode: str = "daily",
    intraday_interval: str = "5min",
) -> tuple:
    """Full training pipeline for one symbol.

    1. Fetch historical data (daily or intraday)
    2. Build features + normalize
    3. Train LSTM with early stopping
    4. Save model weights + scaler (with suffix for intraday)
    """
    os.makedirs(save_dir, exist_ok=True)
    suffix = "" if mode == "daily" else f"_{intraday_interval}"

    # 1. Fetch data
    if mode == "daily":
        log.info("Fetching daily data for %s (lookback=%d)...", symbol, lookback)
        bars = adapter.fetch_daily(symbol, lookback)
    else:
        log.info("Fetching %s intraday data for %s (lookback=%d days)...",
                 intraday_interval, symbol, lookback)
        bars = adapter.fetch_intraday(symbol, intraday_interval,
                                      lookback_days=lookback)
    log.info("Got %d bars for %s.", len(bars), symbol)

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500),
                                     include_live=False)
    log.info("Got %d VIX rows.", len(vix_df))

    # 2. Build features
    engine = FeatureEngine()
    features = engine.build_features(bars, vix_df, mode=mode, symbol=symbol)
    log.info("Built %d feature rows (after warm-up).", len(features))

    if len(features) < seq_len + 10:
        log.error("Not enough data to train. Need at least %d rows, got %d.",
                  seq_len + 10, len(features))
        sys.exit(1)

    # 3. Scaler: fit on training portion only to prevent look-ahead bias
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    full_norm = engine.transform(features)   # transform all with training-only scaler

    # 4. Forward return regression labels (v2)
    if mode == "intraday":
        fwd_days = 2   # 2 days for intraday (shorter horizon)
    else:
        fwd_days = FORWARD_DAYS  # 10 days for daily
    log.info(
        "Regression labels: forward %d-day return (winsorized ±10%%)", fwd_days,
    )

    X_all, y_all = prepare_sequences_regression(
        full_norm, bars, seq_len, forward_days=fwd_days,
    )

    # 5. Purge + embargo split (López de Prado, AFML Chapter 7)
    # After the 80% split point, skip `seq_len` sequences so no validation
    # window shares bars with any training window (prevents leakage).
    seq_split  = int(len(X_all) * 0.8)
    embargo    = seq_len
    X_train, y_train = X_all[:seq_split], y_all[:seq_split]
    X_val,   y_val   = X_all[seq_split + embargo:], y_all[seq_split + embargo:]

    log.info("Training samples: %d, Validation samples: %d (embargo=%d seq)",
             len(X_train), len(X_val), embargo)
    log.info("Return stats — train mean: %+.4f%%, val mean: %+.4f%%",
             y_train.mean() * 100, y_val.mean() * 100 if len(y_val) > 0 else 0)

    # 4. Create model with regression architecture (v2)
    n_features = len(get_feature_cols(mode, symbol))
    model = ReturnLSTM(n_features=n_features)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Cosine annealing LR schedule — warm restarts improve convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01,
    )

    # 5. Training with early stopping, gradient clipping (v2 regression)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_run = 0
    patience_counter = 0
    PATIENCE = 10
    GRAD_CLIP = 1.0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation: MSE loss + direction accuracy (did sign match?)
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                correct += ((preds > 0).float() == (yb > 0).float()).sum().item()
        val_loss /= max(len(val_ds), 1)
        val_acc = correct / max(len(val_ds), 1)  # direction accuracy

        current_lr = optimizer.param_groups[0]["lr"]
        log.info("Epoch %2d/%d  train=%.4f  val=%.4f  acc=%.3f  lr=%.2e",
                 epoch + 1, epochs, train_loss, val_loss, val_acc, current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{symbol}_lstm{suffix}.pt"))
            engine.save_scaler(
                os.path.join(save_dir, f"{symbol}_scaler{suffix}.json"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d.", epoch + 1)
                break

    # --- Build calibration map on TRAINING set (not validation, to avoid leakage) ---
    cal_map = None
    try:
        from model_monitor import CalibrationMap
        model.eval()
        with torch.no_grad():
            train_preds = model(torch.FloatTensor(X_train)).numpy().flatten()
        cal_map = CalibrationMap(n_bins=10)
        cal_map.fit(train_preds, y_train)
        cal_path = os.path.join(save_dir, f"{symbol}_calibration{suffix}.json")
        cal_map.save(cal_path)
        log.info("Calibration map saved to %s (bins=%d, fit on training data)",
                 cal_path, cal_map.n_bins)
    except Exception as exc:
        log.warning("Calibration map build failed: %s", exc)

    # --- Horizon metadata (enforce model-mode separation) ---
    horizon = f"{fwd_days}d"

    metrics_path = os.path.join(save_dir, f"{symbol}_lstm{suffix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "symbol": symbol, "mode": mode,
            "horizon": horizon,
            "model_type": "lstm",
            "model_version": "v2_regression",
            "target": f"forward_{fwd_days}d_return",
            "best_val_loss_mse": round(best_val_loss, 8),
            "best_direction_accuracy": round(best_val_acc, 4),
            "epochs_run": epochs_run,
            "n_features": n_features,
            "has_calibration": cal_map is not None and cal_map.bin_edges is not None,
        }, f, indent=2)
    log.info("Training complete for %s (%s, horizon=%s). Best val_MSE=%.6f  dir_acc=%.3f",
             symbol, mode, horizon, best_val_loss, best_val_acc)
    return model, engine


# ===================================================================
# Meta-labeling (López de Prado, AFML Chapter 3)
# ===================================================================
def train_meta_model(
    symbol: str,
    adapter: DataAdapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
    mode: str = "daily",
    intraday_interval: str = "5min",
) -> None:
    """Train a Random Forest meta-model that predicts when the primary LSTM is correct.

    The meta-model answers: "Given current market conditions + the primary model's
    probability output, should we trust this signal?"

    Pipeline:
        1. Load trained primary LSTM for this symbol.
        2. Fetch historical data and build features (same as primary training).
        3. Run primary LSTM on every bar to get primary_prob.
        4. Label each bar: was_correct = 1 if (primary said UP and price went UP)
           or (primary said DOWN and price went DOWN), else 0.
           Uses same triple-barrier labeling as primary.
        5. Meta features = 21 normalized features of last bar + primary_prob (22-dim).
        6. Train RandomForestClassifier on meta features → was_correct.
        7. Save to {symbol}_meta_rf{suffix}.joblib.
    """
    os.makedirs(save_dir, exist_ok=True)
    suffix = "" if mode == "daily" else f"_{intraday_interval}"
    meta_path = os.path.join(save_dir, f"{symbol}_meta_rf{suffix}.joblib")

    # 1. Load primary model
    weights_path = os.path.join(save_dir, f"{symbol}_lstm{suffix}.pt")
    scaler_path  = os.path.join(save_dir, f"{symbol}_scaler{suffix}.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Primary model not found: {weights_path}. "
            f"Train it first: python main.py train --symbol {symbol}"
        )
    engine = FeatureEngine()
    engine.load_scaler(scaler_path)
    primary = DirectionLSTM(n_features=len(get_feature_cols(mode, symbol)))
    primary.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    primary.eval()
    log.info("Loaded primary LSTM for %s (%s).", symbol, mode)

    # 2. Fetch data (same lookback as primary training)
    if mode == "daily":
        bars = adapter.fetch_daily(symbol, lookback)
    else:
        bars = adapter.fetch_intraday(symbol, intraday_interval, lookback_days=lookback)
    log.info("Got %d bars for meta-training.", len(bars))

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500),
                                     include_live=False)
    features = engine.build_features(bars, vix_df, mode=mode, symbol=symbol)
    features_norm = engine.transform(features)

    if len(features_norm) < SEQ_LEN + 10:
        raise ValueError(f"Not enough data for meta-training: {len(features_norm)} rows.")

    # 3. Triple-barrier labels (same thresholds as primary)
    if mode == "intraday":
        pt_pct, sl_pct, horizon = 0.005, 0.003, 10
    else:
        pt_pct, sl_pct, horizon = 0.015, 0.010, 5

    _, y_barrier = prepare_sequences_triple_barrier(
        features_norm, bars, SEQ_LEN, pt_pct=pt_pct, sl_pct=sl_pct, horizon=horizon,
    )

    # 4. Run primary LSTM on every window to get primary_prob per sequence
    feat_arr = features_norm.values.astype(np.float32)
    n_seqs = len(feat_arr) - SEQ_LEN

    primary_probs = np.zeros(n_seqs, dtype=np.float32)
    with torch.no_grad():
        for i in range(n_seqs):
            window = feat_arr[i: i + SEQ_LEN]
            x = torch.FloatTensor(window).unsqueeze(0)
            primary_probs[i] = primary(x).item()

    # 5. Meta labels: was primary correct?
    primary_dirs = (primary_probs > 0.5).astype(np.float32)   # 1=UP, 0=DOWN
    was_correct  = (primary_dirs == y_barrier).astype(np.int32)

    # 6. Meta features: last bar of each window (21 dims) + primary_prob (1 dim)
    last_bar_features = feat_arr[SEQ_LEN - 1: SEQ_LEN - 1 + n_seqs]   # shape (n, 21)
    meta_X = np.hstack([last_bar_features, primary_probs.reshape(-1, 1)])  # shape (n, 22)
    meta_y = was_correct

    log.info(
        "Meta dataset: %d samples. Correct rate: %.1f%% (class balance).",
        len(meta_y), meta_y.mean() * 100,
    )

    # 7. Train RF with 80/20 temporal split (same as primary — no shuffle)
    split = int(len(meta_X) * 0.8)
    X_train, y_train = meta_X[:split], meta_y[:split]
    X_val,   y_val   = meta_X[split:], meta_y[split:]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    val_acc = rf.score(X_val, y_val)
    meta_val_probs = rf.predict_proba(X_val)[:, 1]

    # --- Per-symbol threshold calibration ---
    # Objective: maximise PRECISION LIFT above base rate, weighted by log(coverage).
    #   score = max(0, precision(t) - base_rate) * log1p(coverage(t) * 5)
    # This rewards thresholds that genuinely improve on the LSTM base rate while
    # maintaining enough coverage (>= 10% of val bars).
    # If no threshold beats the base rate, fall back to the global META_THRESHOLD.
    base_rate = float(y_val.mean())       # fraction of bars where primary was correct
    best_t, best_score = META_THRESHOLD, 0.0  # 0.0 so only positive-lift thresholds win
    best_prec, best_cov = 0.0, 0.0
    for t in np.arange(0.40, 0.92, 0.025):
        mask = meta_val_probs >= t
        cov = mask.mean()
        if cov < 0.10:
            break
        prec = float(y_val[mask].mean()) if mask.sum() > 0 else 0.0
        lift = max(0.0, prec - base_rate)
        score = lift * np.log1p(cov * 5)
        if score > best_score:
            best_score, best_t, best_prec, best_cov = score, round(float(t), 3), prec, cov

    n_tradeable_global = (meta_val_probs >= META_THRESHOLD).sum()
    n_tradeable_opt    = (meta_val_probs >= best_t).sum()
    log.info(
        "Meta RF val accuracy: %.3f | Global thresh %.2f → %d/%d (%.1f%%) | "
        "Optimal thresh %.3f → %d/%d (%.1f%%, prec %.1f%%)",
        val_acc,
        META_THRESHOLD, n_tradeable_global, len(y_val),
        n_tradeable_global / max(len(y_val), 1) * 100,
        best_t, n_tradeable_opt, len(y_val),
        n_tradeable_opt / max(len(y_val), 1) * 100,
        best_prec * 100,
    )

    # Feature importance summary
    feat_names = get_feature_cols(mode, symbol) + ["primary_prob"]
    top_idx = np.argsort(rf.feature_importances_)[::-1][:5]
    log.info("Top 5 meta features: %s",
             ", ".join(f"{feat_names[i]}={rf.feature_importances_[i]:.3f}"
                       for i in top_idx))

    joblib.dump(rf, meta_path)
    log.info("Saved meta RF model → %s", meta_path)

    # Save per-symbol threshold config alongside the model
    config_path = meta_path.replace(".joblib", "_config.json")
    config = {
        "symbol": symbol,
        "mode": mode,
        "threshold": best_t,
        "val_precision": round(best_prec, 4),
        "val_coverage": round(float(best_cov), 4),
        "val_accuracy": round(float(val_acc), 4),
        "global_threshold": META_THRESHOLD,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Saved threshold config (%.3f) → %s", best_t, config_path)


# ===================================================================
# Predictor (inference)
# ===================================================================
class Predictor:
    """Load a trained model and produce regression predictions (v2)."""

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR,
                 mode: str = "daily", intraday_interval: str = "5min"):
        self.symbol = symbol
        self.model_dir = model_dir
        self.mode = mode
        self.model_type = "lstm"
        self.intraday_interval = intraday_interval
        self.engine = FeatureEngine()
        self.model: Optional[ReturnLSTM] = None
        self.meta_model = None   # Deprecated in v2 — kept for file compat
        self._calibration_map = None
        self._load()

    def _load(self) -> None:
        suffix = "" if self.mode == "daily" else f"_{self.intraday_interval}"
        weights_path = os.path.join(self.model_dir, f"{self.symbol}_lstm{suffix}.pt")
        scaler_path  = os.path.join(self.model_dir, f"{self.symbol}_scaler{suffix}.json")
        if not os.path.exists(weights_path):
            mode_hint = f" --mode {self.mode}" if self.mode != "daily" else ""
            raise FileNotFoundError(
                f"No trained model for {self.symbol} ({self.mode}). "
                f"Run: python main.py train --symbol {self.symbol}{mode_hint}")
        self.engine.load_scaler(scaler_path)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Backward compat: v1 models used key prefix "classifier", v2 uses "regressor"
        state = {k.replace("classifier.", "regressor."): v for k, v in state.items()}
        # Detect n_features from saved LSTM input weight shape (4*hidden, n_features)
        n_features_saved = state["lstm.weight_ih_l0"].shape[1]
        self.model = ReturnLSTM(n_features=n_features_saved)
        self.model.load_state_dict(state)
        self.model.eval()
        # Store saved n_features so predict() can truncate the feature array if needed
        self._n_features_saved = n_features_saved

        # Meta RF (deprecated in v2 — load for backward compat but not used for gating)
        meta_path   = os.path.join(self.model_dir, f"{self.symbol}_meta_rf{suffix}.joblib")
        config_path = os.path.join(self.model_dir, f"{self.symbol}_meta_rf{suffix}_config.json")
        if os.path.exists(meta_path):
            self.meta_model = joblib.load(meta_path)
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                self.meta_threshold = float(cfg.get("threshold", META_THRESHOLD))
            else:
                self.meta_threshold = META_THRESHOLD
        else:
            self.meta_model = None
            self.meta_threshold = META_THRESHOLD

        # Load calibration map (optional — built during training)
        cal_path = os.path.join(self.model_dir, f"{self.symbol}_calibration{suffix}.json")
        if os.path.exists(cal_path):
            try:
                from model_monitor import CalibrationMap
                self._calibration_map = CalibrationMap.load(cal_path)
                log.info("Loaded calibration map for %s from %s", self.symbol, cal_path)
            except Exception as exc:
                log.debug("Calibration map load failed for %s: %s", self.symbol, exc)

        # Validate horizon: LSTM is daily-only (10d horizon)
        metrics_path = os.path.join(self.model_dir, f"{self.symbol}_lstm{suffix}_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                model_horizon = metrics.get("horizon", "10d")
                model_mode = metrics.get("mode", "daily")
                if self.mode == "intraday" and model_mode == "daily":
                    log.warning(
                        "LSTM model for %s was trained on daily data (horizon=%s) "
                        "but is being loaded in intraday mode. Use LightGBM for intraday.",
                        self.symbol, model_horizon,
                    )
            except Exception:
                pass

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = SEQ_LEN) -> dict:
        """Produce a regression prediction (v2).

        Returns:
            expected_return: continuous predicted forward return (e.g., +0.015 = +1.5%)
            direction:       "UP" if E[r] > cost_threshold, "DOWN" if < -cost_threshold, "FLAT" otherwise
            probability:     legacy compat: 0.5 + expected_return * 10, clipped to [0.05, 0.95]
            confidence:      abs(expected_return) / TARGET_RETURN, clipped to [0, 1]
            meta_confidence: 1.0 (meta model deprecated in v2)
            tradeable:       abs(expected_return) > COST_THRESHOLD
        """
        features = self.engine.build_features(bars_df, vix_df, mode=self.mode, symbol=self.symbol)
        features_norm = self.engine.transform(features)

        if len(features_norm) < seq_len:
            return {
                "expected_return": 0.0,
                "direction": "FLAT", "probability": 0.5, "confidence": 0.0,
                "meta_confidence": 1.0, "tradeable": False,
            }

        window = features_norm.iloc[-seq_len:].values
        # Backward compat: old models trained on fewer features — truncate to match
        n_saved = getattr(self, "_n_features_saved", window.shape[1])
        if window.shape[1] > n_saved:
            window = window[:, :n_saved]
        x = torch.FloatTensor(window).unsqueeze(0)

        with torch.no_grad():
            expected_return = self.model(x).item()

        if expected_return > COST_THRESHOLD:
            direction = "UP"
        elif expected_return < -COST_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Use calibrated confidence if calibration map is available
        if self._calibration_map is not None and self._calibration_map.bin_edges is not None:
            confidence = self._calibration_map.calibrated_confidence(
                expected_return, TARGET_RETURN
            )
            calibrated_return = self._calibration_map.calibrated_return(expected_return)
        else:
            confidence = min(1.0, abs(expected_return) / TARGET_RETURN)
            calibrated_return = expected_return

        # Legacy probability compat for options_trader and other consumers
        probability = max(0.05, min(0.95, 0.5 + expected_return * 10))

        return {
            "expected_return": round(expected_return, 6),
            "calibrated_return": round(calibrated_return, 6),
            "direction": direction,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "meta_confidence": 1.0,
            "tradeable": abs(expected_return) > COST_THRESHOLD,
        }


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSTM ML Model for ETF direction prediction.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- train --
    train_p = sub.add_parser("train", help="Train LSTM model on historical data")
    train_p.add_argument("--symbol", required=True, help="Symbol to train (e.g. SPY)")
    train_p.add_argument("--provider", default="yahoo",
                         choices=["yahoo", "alpaca", "hybrid"])
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--lookback", type=int, default=1000,
                         help="Bars to fetch for training (default: 1000)")
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                         help="Training mode (default: daily)")
    train_p.add_argument("--interval", default="5min", choices=["1min", "5min"],
                         help="Intraday bar interval (default: 5min)")

    # -- train-meta --
    meta_p = sub.add_parser("train-meta", help="Train meta RF model (requires trained primary LSTM)")
    meta_p.add_argument("--symbol", required=True,
                        help="Symbol to train meta-model for (e.g. SPY, or ALL for all active)")
    meta_p.add_argument("--provider", default="yahoo", choices=["yahoo", "alpaca", "hybrid"])
    meta_p.add_argument("--lookback", type=int, default=1000)
    meta_p.add_argument("--mode", default="daily", choices=["daily", "intraday"])
    meta_p.add_argument("--interval", default="5min", choices=["1min", "5min"])

    # -- predict --
    pred_p = sub.add_parser("predict", help="Run prediction for a symbol")
    pred_p.add_argument("--symbol", required=True, help="Symbol to predict")
    pred_p.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    pred_p.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                        help="Prediction mode (default: daily)")
    pred_p.add_argument("--interval", default="5min", choices=["1min", "5min"],
                        help="Intraday bar interval (default: 5min)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train-meta":
        from signals_engine import DEFAULT_UNIVERSE
        ACTIVE_SYMBOLS = [
            "SPY", "QQQ", "IWM", "IGV", "XLE", "SOXX",
            "GLD", "SLV", "EWJ", "EWT", "EEM",
        ]
        adapter  = build_adapter(args.provider)
        fred_key = os.environ.get("FRED_API_KEY")
        symbols  = ACTIVE_SYMBOLS if args.symbol.upper() == "ALL" else [args.symbol.upper()]
        lookback = args.lookback
        if args.mode == "intraday" and lookback == 1000:
            lookback = 60
        for sym in symbols:
            log.info("=== Meta-training %s (%s) ===", sym, args.mode)
            try:
                train_meta_model(
                    symbol=sym,
                    adapter=adapter,
                    fred_key=fred_key,
                    lookback=lookback,
                    mode=args.mode,
                    intraday_interval=args.interval,
                )
            except FileNotFoundError as e:
                log.error("%s — skipping: %s", sym, e)

    elif args.command == "train":
        adapter = build_adapter(args.provider)
        fred_key = os.environ.get("FRED_API_KEY")
        lookback = args.lookback
        if args.mode == "intraday" and lookback == 1000:
            lookback = 60  # default 60 days of intraday data
        train_model(
            symbol=args.symbol.upper(),
            adapter=adapter,
            fred_key=fred_key,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            lookback=lookback,
            mode=args.mode,
            intraday_interval=args.interval,
        )

    elif args.command == "predict":
        adapter = build_adapter(args.provider)
        fred_key = os.environ.get("FRED_API_KEY")
        symbol = args.symbol.upper()

        predictor = Predictor(symbol, mode=args.mode,
                              intraday_interval=args.interval)
        if args.mode == "intraday":
            bars = adapter.fetch_intraday(symbol, args.interval, lookback_days=2)
        else:
            bars = adapter.fetch_daily(symbol, DAILY_LOOKBACK)
        vix_df = _fetch_vix_for_training(fred_key, lookback_days=30)
        result = predictor.predict(bars, vix_df)

        print(f"\n  {symbol} ({args.mode}) -> {result['direction']}  "
              f"(confidence: {result['confidence']:.4f}, "
              f"probability: {result['probability']:.4f})\n")


if __name__ == "__main__":
    main()
