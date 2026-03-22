#!/usr/bin/env python3
"""
Cross-Sectional Coin Ranker (Layer 1)
========================================
LightGBM LambdaRank model that ranks crypto coins cross-sectionally.
Ranks the full universe; position limits are applied downstream by
paper_trader (max_positions in risk_config).

Architecture:
  - Features are z-scored cross-sectionally (relative positioning, not absolute)
  - Label: forward 5-day risk-adjusted return (return / realized vol)
  - Model: LightGBM LambdaRank (optimizes NDCG@3,6)
  - Output: ranked list of all coins ordered by score (no hard cutoff)

Usage (via main.py):
    python main.py train-selector              # train on full universe
    python main.py train-selector --top-k 4    # select top-4 instead of top-6
    python main.py rank-coins                  # print today's rankings
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from signals_engine import PROJECT_ROOT, compute_rsi, compute_adx, compute_atr
from utils import CRYPTO_MODEL_DIR
from universe_screener import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("coin_selector")

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Full candidate universe — filtered at runtime by liquidity
CRYPTO_UNIVERSE = [
    # Majors
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Large-cap alts
    "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "UNI-USD", "LTC-USD",
    # Mid-cap alts (higher vol, wider spread)
    "DOGE-USD", "RENDER-USD", "AAVE-USD", "CRV-USD",
    "SUSHI-USD", "ARB-USD", "FIL-USD",
    # Excluded (not tradeable on Alpaca):
    #   OP-USD — not listed, INJ-USD — not listed,
    #   NEAR-USD — inactive, APT-USD — inactive
]

# Minimum 20-day median daily dollar volume to qualify
MIN_DOLLAR_VOLUME = 5_000_000  # $5M


# Forward label horizon
LABEL_HORIZON = 5  # 5-day forward return for ranking label

# Model filename
SELECTOR_MODEL_FILE = "coin_selector_lgb.txt"
SELECTOR_CONFIG_FILE = "coin_selector_config.json"

# Cross-sectional feature names (z-scored across universe at each timestamp)
XS_FEATURES = [
    "momentum_7d",
    "momentum_30d",
    "reversal_1d",
    "vol_ratio",
    "volume_zscore",
    "btc_beta_resid",
    "rsi14_rank",
    "drawdown_from_ath",
    "spread_proxy",
    "ret63",
    "adx",
]

# ---------------------------------------------------------------------------
# Intraday selector constants
# ---------------------------------------------------------------------------

LABEL_HORIZON_INTRADAY = 12    # 12 bars × 5min = 1 hour
SNAPSHOT_INTERVAL_BARS = 48    # snapshot every 4 hours for cross-sectional ranking
INTRADAY_LOOKBACK_DAYS = 180

SELECTOR_MODEL_FILE_INTRADAY = "coin_selector_lgb_intraday.txt"
SELECTOR_CONFIG_FILE_INTRADAY = "coin_selector_intraday_config.json"

XS_FEATURES_INTRADAY = [
    "momentum_1h",             # pct_change(12 bars) — short-term relative momentum
    "momentum_4h",             # pct_change(48 bars) — medium-term momentum
    "reversal_30m",            # -pct_change(6 bars) — mean-reversion
    "vol_ratio_short_long",    # std(12-bar) / std(48-bar) — vol regime shift
    "dollar_vol_zscore",       # (current $vol - 48-bar mean) / std — liquidity surge
    "close_position_xs",       # (close - low12) / (high12 - low12) — microstructure quality
    "cumulative_delta_xs",     # rolling buy-sell imbalance — order flow
    "btc_beta_resid_hourly",   # hourly beta residual vs BTC — idiosyncratic alpha
    "spread_proxy_intraday",   # avg((high-low)/close) over 48 bars — liquidity cost
    "rvol_xs",                 # volume / 48-bar MA volume — relative demand
    "realized_vol_24bar",      # std(log returns) annualized — vol level
]

# data_quality is computed but used as a pre-filter, not a ranking feature.
# momentum_acceleration removed: noisy 2nd derivative, redundant with momentum_1h + momentum_4h.
DATA_QUALITY_THRESHOLD = 0.70  # drop coins with >30% stale/flat bars


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_coin_daily(symbol: str, lookback_days: int = 1500) -> pd.DataFrame:
    """Fetch daily OHLCV for a crypto symbol via yfinance."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    cal_days = int(lookback_days * 1.5) + 30
    hist = ticker.history(period=f"{cal_days}d", interval="1d")
    if hist is None or hist.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": hist.index,
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values,
    })
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_universe_data(
    universe: List[str],
    lookback_days: int = 1500,
) -> Dict[str, pd.DataFrame]:
    """Fetch daily data for the full universe. Skips symbols with insufficient data."""
    data = {}
    for sym in universe:
        try:
            df = _fetch_coin_daily(sym, lookback_days)
            if len(df) < 100:
                log.warning("Skipping %s: only %d bars (need ≥100)", sym, len(df))
                continue
            data[sym] = df
            log.info("Fetched %s: %d bars (%s → %s)",
                     sym, len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date())
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", sym, exc)
    return data


def fetch_universe_intraday_data(
    universe: List[str],
    lookback_days: int = INTRADAY_LOOKBACK_DAYS,
) -> Dict[str, pd.DataFrame]:
    """Fetch 5-min bars for the full universe via BinanceUS (for intraday selector).

    Returns {symbol: DataFrame} with columns [ts, open, high, low, close, volume].
    Minimum 2000 bars (~7 days) per coin to qualify for ranking.
    Training requires more data; ranking can work with less.
    """
    from crypto_intraday_data import CryptoIntradayData
    data_source = CryptoIntradayData()
    data: Dict[str, pd.DataFrame] = {}
    min_bars = 2000  # ~7 days of 5-min bars — enough for feature computation
    for sym in universe:
        try:
            df = data_source.fetch_training_bars(sym, days=lookback_days)
            if len(df) < min_bars:
                log.warning("Skipping %s: only %d 5m bars (need ≥%d)", sym, len(df), min_bars)
                continue
            data[sym] = df
            log.info("Fetched %s: %d 5m bars (%s → %s)",
                     sym, len(df), df["ts"].iloc[0], df["ts"].iloc[-1])
        except Exception as exc:
            log.warning("Failed to fetch %s intraday: %s", sym, exc)
    return data


# ---------------------------------------------------------------------------
# Cross-sectional feature computation
# ---------------------------------------------------------------------------

def compute_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-coin time-series features (before cross-sectional z-scoring)."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["date"] = df["date"]

    # Momentum signals
    feat["momentum_7d"] = close.pct_change(7)
    feat["momentum_30d"] = close.pct_change(30)
    feat["reversal_1d"] = -close.pct_change(1)  # sign-flipped for mean-reversion
    feat["ret63"] = close.pct_change(63)

    # Volatility
    ret_daily = close.pct_change()
    feat["vol_5d"] = ret_daily.rolling(5).std() * np.sqrt(365)
    feat["vol_30d"] = ret_daily.rolling(30).std() * np.sqrt(365)
    feat["vol_ratio"] = feat["vol_5d"] / feat["vol_30d"].replace(0, np.nan)

    # Volume z-score (20-day rolling)
    dollar_vol = close * volume
    dv_mean = dollar_vol.rolling(20).mean()
    dv_std = dollar_vol.rolling(20).std()
    feat["volume_zscore"] = (dollar_vol - dv_mean) / dv_std.replace(0, np.nan)

    # Dollar volume for liquidity filter
    feat["dollar_volume_20d"] = dollar_vol.rolling(20).median()

    # RSI
    feat["rsi14_rank"] = compute_rsi(close, 14)

    # ADX
    feat["adx"] = compute_adx(high, low, close, 14)

    # Drawdown from all-time high (rolling 365d max)
    rolling_high = close.rolling(365, min_periods=30).max()
    feat["drawdown_from_ath"] = (close - rolling_high) / rolling_high.replace(0, np.nan)

    # Spread proxy: (high - low) / close
    feat["spread_proxy"] = (high - low) / close.replace(0, np.nan)

    # BTC beta residual (placeholder — computed cross-sectionally in build_xs_panel)
    feat["btc_beta_resid"] = 0.0

    feat["close"] = close
    return feat


def compute_coin_features_intraday(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-coin intraday features from 5-min bars.

    Args:
        df: 5-min OHLCV with columns [ts, open, high, low, close, volume]
        btc_df: optional BTC 5-min bars for cross-market features

    Returns:
        DataFrame with intraday XS features + ts + close columns.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["ts"] = df["ts"]

    # Momentum (bar counts: 6=30m, 12=1h, 48=4h)
    feat["momentum_1h"] = close.pct_change(12)
    feat["momentum_4h"] = close.pct_change(48)
    feat["reversal_30m"] = -close.pct_change(6)

    # Volatility
    log_ret = np.log(close / close.shift(1))
    std_12 = log_ret.rolling(12).std()
    std_48 = log_ret.rolling(48).std()
    feat["vol_ratio_short_long"] = (std_12 / std_48.replace(0, np.nan)).fillna(1.0)
    feat["realized_vol_24bar"] = log_ret.rolling(24).std() * np.sqrt(288 * 365)

    # Volume / liquidity
    dollar_vol = close * volume
    dv_mean_48 = dollar_vol.rolling(48).mean()
    dv_std_48 = dollar_vol.rolling(48).std()
    feat["dollar_vol_zscore"] = ((dollar_vol - dv_mean_48) / dv_std_48.replace(0, np.nan)).fillna(0.0)
    vol_ma_48 = volume.rolling(48).mean()
    feat["rvol_xs"] = (volume / vol_ma_48.replace(0, np.nan)).fillna(1.0)

    # Microstructure: close position within 12-bar range
    rolling_high_12 = high.rolling(12).max()
    rolling_low_12 = low.rolling(12).min()
    rng = (rolling_high_12 - rolling_low_12).replace(0, np.nan)
    feat["close_position_xs"] = ((close - rolling_low_12) / rng).fillna(0.5)

    # Cumulative delta (order flow proxy from OHLC)
    # Approximation: if close > open → buying pressure, else selling
    bar_delta = np.where(close >= df["open"].astype(float), volume, -volume)
    feat["cumulative_delta_xs"] = pd.Series(bar_delta, index=df.index).rolling(12).sum()
    # Normalize by total volume to make it cross-sectionally comparable
    vol_sum_12 = volume.rolling(12).sum().replace(0, np.nan)
    feat["cumulative_delta_xs"] = (feat["cumulative_delta_xs"] / vol_sum_12).fillna(0.0)

    # Spread proxy: average (high-low)/close over 48 bars
    bar_spread = (high - low) / close.replace(0, np.nan)
    feat["spread_proxy_intraday"] = bar_spread.rolling(48).mean()

    # Momentum acceleration
    ret_6bar = close.pct_change(6)
    feat["momentum_acceleration"] = ret_6bar - ret_6bar.shift(6)

    # Data quality: fraction of valid bars (non-flat price AND non-zero volume) in 48-bar window.
    # Low values = illiquid/stale data → ranker should penalize instead of us masking with fillna.
    is_flat = (high == low)  # flat price bar = no real trading activity
    is_zero_vol = (volume == 0)
    is_bad = (is_flat | is_zero_vol).astype(float)
    feat["data_quality"] = 1.0 - is_bad.rolling(48, min_periods=1).mean()

    # BTC beta residual (hourly scale, 12-bar rolling)
    feat["btc_beta_resid_hourly"] = 0.0  # placeholder — computed in build_xs_panel_intraday
    if btc_df is not None and len(btc_df) >= 60:
        btc_close = btc_df["close"].astype(float)
        btc_ret = btc_close.pct_change().values
        sym_ret = close.pct_change().values
        resid = np.full(len(sym_ret), np.nan)
        window = 48  # 4-hour rolling window for beta estimation
        for i in range(window, len(sym_ret)):
            w_sym = sym_ret[i - window:i]
            w_btc = btc_ret[i - window:i]
            # Align lengths (btc_df may be shorter)
            min_len = min(len(w_sym), len(w_btc))
            if min_len < 12:
                continue
            w_sym = w_sym[:min_len]
            w_btc = w_btc[:min_len]
            btc_var = np.nanvar(w_btc)
            if btc_var > 1e-15:
                beta = np.nanmean((w_sym - np.nanmean(w_sym)) * (w_btc - np.nanmean(w_btc))) / btc_var
                resid[i] = sym_ret[i] - beta * btc_ret[i] if i < len(btc_ret) else np.nan
            else:
                resid[i] = sym_ret[i]
        feat["btc_beta_resid_hourly"] = resid

    feat["close"] = close
    return feat


def build_xs_panel(
    universe_data: Dict[str, pd.DataFrame],
    btc_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a cross-sectional panel: (date, symbol, features, label).

    Features are z-scored cross-sectionally at each date.
    """
    # Compute per-coin features
    coin_features = {}
    for sym, df in universe_data.items():
        feat = compute_coin_features(df)
        feat["symbol"] = sym
        coin_features[sym] = feat

    if not coin_features:
        return pd.DataFrame()

    # Stack into panel
    panel = pd.concat(coin_features.values(), ignore_index=True)

    # Compute BTC beta residual
    if btc_data is None and "BTC-USD" in universe_data:
        btc_data = universe_data["BTC-USD"]
    if btc_data is not None:
        btc_close = btc_data.set_index("date")["close"].astype(float)
        btc_ret = btc_close.pct_change()
        for sym in universe_data:
            mask = panel["symbol"] == sym
            sym_dates = panel.loc[mask, "date"]
            sym_close = panel.loc[mask, "close"]
            sym_ret = sym_close.pct_change()
            # Rolling 30-day beta residual
            aligned_btc_ret = btc_ret.reindex(sym_dates).fillna(0).values
            sym_ret_vals = sym_ret.fillna(0).values
            resid = np.full(len(sym_ret_vals), np.nan)
            for i in range(30, len(sym_ret_vals)):
                window_sym = sym_ret_vals[i-30:i]
                window_btc = aligned_btc_ret[i-30:i]
                if np.std(window_btc) > 1e-10:
                    beta = np.cov(window_sym, window_btc)[0, 1] / np.var(window_btc)
                    resid[i] = window_sym[-1] - beta * window_btc[-1]
                else:
                    resid[i] = window_sym[-1]
            panel.loc[mask, "btc_beta_resid"] = resid

    # Liquidity filter: drop rows where 20d median dollar volume < threshold
    panel = panel[panel["dollar_volume_20d"] >= MIN_DOLLAR_VOLUME].copy()

    # Cross-sectional z-scoring (per date)
    for feat_col in XS_FEATURES:
        if feat_col not in panel.columns:
            continue
        grouped = panel.groupby("date")[feat_col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1.0)
        panel[feat_col] = (panel[feat_col] - mean) / std

    # Compute forward label: 5-day risk-adjusted return
    panel["fwd_5d_ret"] = np.nan
    panel["fwd_5d_vol"] = np.nan
    panel["label"] = np.nan
    for sym in panel["symbol"].unique():
        mask = panel["symbol"] == sym
        sym_close = panel.loc[mask, "close"].values
        sym_ret = np.diff(sym_close, prepend=np.nan) / np.concatenate([[np.nan], sym_close[:-1]])
        n = len(sym_close)
        fwd_ret = np.full(n, np.nan)
        fwd_vol = np.full(n, np.nan)
        for i in range(n - LABEL_HORIZON):
            fwd_ret[i] = sym_close[i + LABEL_HORIZON] / sym_close[i] - 1
            fwd_vol[i] = np.std(sym_ret[i+1:i+1+LABEL_HORIZON]) if i + 1 + LABEL_HORIZON <= n else np.nan
        panel.loc[mask, "fwd_5d_ret"] = fwd_ret
        panel.loc[mask, "fwd_5d_vol"] = fwd_vol
        panel.loc[mask, "label"] = fwd_ret / np.maximum(fwd_vol, 0.001)

    # Drop rows with NaN features or labels
    feat_and_label = XS_FEATURES + ["label"]
    panel = panel.dropna(subset=feat_and_label).reset_index(drop=True)

    return panel


def build_xs_panel_intraday(
    universe_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a cross-sectional panel from 5-min bars for intraday ranking.

    Snapshots are taken every SNAPSHOT_INTERVAL_BARS (48 bars = 4 hours).
    Features are z-scored cross-sectionally per snapshot.
    Label: forward 12-bar (1 hour) risk-adjusted return.
    """
    # Find BTC data for cross-market features
    btc_data = None
    for sym in ("BTC-USD", "BTC/USD"):
        if sym in universe_data:
            btc_data = universe_data[sym]
            break

    # Compute per-coin features
    coin_features: Dict[str, pd.DataFrame] = {}
    for sym, df in universe_data.items():
        btc_for_sym = btc_data if sym not in ("BTC-USD", "BTC/USD") else None
        feat = compute_coin_features_intraday(df, btc_df=btc_for_sym)
        feat["symbol"] = sym
        coin_features[sym] = feat

    if not coin_features:
        return pd.DataFrame()

    # Sample every SNAPSHOT_INTERVAL_BARS per coin; pre-compute labels from
    # raw data using the known sampling indices (avoids tz-aware/naive mismatch).
    sampled_parts = []
    for sym, feat in coin_features.items():
        indices = list(range(SNAPSHOT_INTERVAL_BARS, len(feat), SNAPSHOT_INTERVAL_BARS))
        if not indices:
            continue
        sampled = feat.iloc[indices].copy()

        # Compute forward label directly from raw close using sample indices
        full_close = feat["close"].values
        full_ret = np.diff(np.log(full_close), prepend=np.nan)
        labels = np.full(len(indices), np.nan)
        for j, raw_i in enumerate(indices):
            if raw_i + LABEL_HORIZON_INTRADAY >= len(full_close):
                continue
            fwd_ret = full_close[raw_i + LABEL_HORIZON_INTRADAY] / full_close[raw_i] - 1
            fwd_vol = np.std(full_ret[raw_i + 1:raw_i + 1 + LABEL_HORIZON_INTRADAY])
            labels[j] = fwd_ret / max(fwd_vol, 0.0001)
        sampled["label"] = labels
        sampled_parts.append(sampled)

    if not sampled_parts:
        return pd.DataFrame()

    panel = pd.concat(sampled_parts, ignore_index=True)

    # Round timestamps to nearest 4h for cross-sectional alignment
    panel["snapshot"] = panel["ts"].dt.floor("4h")

    # Pre-filter: drop rows where data_quality is below threshold
    if "data_quality" in panel.columns:
        n_before = len(panel)
        panel = panel[panel["data_quality"] >= DATA_QUALITY_THRESHOLD].reset_index(drop=True)
        n_dropped = n_before - len(panel)
        if n_dropped > 0:
            log.info("Pre-filter: dropped %d/%d rows with data_quality < %.2f",
                     n_dropped, n_before, DATA_QUALITY_THRESHOLD)

    # Cross-sectional z-scoring per snapshot
    for feat_col in XS_FEATURES_INTRADAY:
        if feat_col not in panel.columns:
            continue
        grouped = panel.groupby("snapshot")[feat_col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1.0)
        panel[feat_col] = (panel[feat_col] - mean) / std
        # Clip at ±3σ to reduce outlier dominance
        panel[feat_col] = panel[feat_col].clip(-3, 3)

    # Drop NaN features or labels
    feat_and_label = XS_FEATURES_INTRADAY + ["label"]
    panel = panel.dropna(subset=feat_and_label).reset_index(drop=True)

    return panel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_selector(
    universe: Optional[List[str]] = None,
    train_end: str = "2025-01-01",
    save_dir: Optional[str] = None,
    lookback_days: int = 1500,
    mode: str = "swing",
) -> dict:
    """Train the cross-sectional LightGBM LambdaRank selector.

    Args:
        mode: "swing" for daily 5-day label, "intraday" for 5-min 1-hour label.

    Returns dict with training metrics.
    """
    import lightgbm as lgb

    is_intraday = (mode == "intraday")
    xs_features = XS_FEATURES_INTRADAY if is_intraday else XS_FEATURES
    model_file = SELECTOR_MODEL_FILE_INTRADAY if is_intraday else SELECTOR_MODEL_FILE
    config_file = SELECTOR_CONFIG_FILE_INTRADAY if is_intraday else SELECTOR_CONFIG_FILE

    if universe is None:
        dynamic = load_universe(save_dir or CRYPTO_MODEL_DIR)
        universe = dynamic if dynamic else list(CRYPTO_UNIVERSE)
    if save_dir is None:
        from utils import CRYPTO_INTRADAY_MODEL_DIR
        save_dir = CRYPTO_INTRADAY_MODEL_DIR if is_intraday else CRYPTO_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Fetch data
    if is_intraday:
        log.info("Fetching intraday (5m) data for %d coins...", len(universe))
        data = fetch_universe_intraday_data(universe, lookback_days=lookback_days)
    else:
        log.info("Fetching daily data for %d coins...", len(universe))
        data = fetch_universe_data(universe, lookback_days=lookback_days)
    if len(data) < 3:
        raise RuntimeError(f"Only {len(data)} coins have data — need ≥3 for cross-sectional ranking")

    # Build panel
    log.info("Building %s cross-sectional panel...", mode)
    if is_intraday:
        panel = build_xs_panel_intraday(data)
        time_col = "snapshot"
    else:
        panel = build_xs_panel(data)
        time_col = "date"
    log.info("Panel: %d rows, %d unique snapshots, %d coins",
             len(panel), panel[time_col].nunique(), panel["symbol"].nunique())

    # Train/val split
    if is_intraday:
        # Time-based 75/25 split for intraday
        unique_times = sorted(panel[time_col].unique())
        split_idx = int(len(unique_times) * 0.75)
        split_time = unique_times[split_idx]
        train_mask = panel[time_col] < split_time
        val_mask = panel[time_col] >= split_time
    else:
        train_end_dt = pd.Timestamp(train_end)
        train_mask = panel[time_col] < train_end_dt
        val_mask = panel[time_col] >= train_end_dt

    train_df = panel[train_mask].copy()
    val_df = panel[val_mask].copy()

    # Min sample gate
    if len(train_df) < 150:
        raise RuntimeError(f"Training set too small: {len(train_df)} rows (need ≥150)")
    n_train_snapshots = train_df[time_col].nunique()
    if n_train_snapshots < 30:
        raise RuntimeError(f"Training set has too few snapshots: {n_train_snapshots} (need ≥30)")
    if len(val_df) < 50:
        log.warning("Validation set small: %d rows", len(val_df))

    log.info("Train: %d rows (%s → %s), Val: %d rows (%s → %s)",
             len(train_df), train_df[time_col].min(), train_df[time_col].max(),
             len(val_df), val_df[time_col].min(), val_df[time_col].max())

    # Discretize continuous labels into integer grades 0-4 per snapshot
    def _discretize_labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label_int"] = 0
        for _, grp in df.groupby(time_col):
            vals = grp["label"].values
            if len(vals) < 5:
                ranks = pd.Series(vals).rank(method="min").astype(int) - 1
                ranks = (ranks * 4 / max(ranks.max(), 1)).astype(int).clip(0, 4)
                df.loc[grp.index, "label_int"] = ranks.values
            else:
                pcts = pd.Series(vals).rank(pct=True)
                grades = (pcts * 5).clip(0, 4.999).astype(int)
                df.loc[grp.index, "label_int"] = grades.values
        return df

    train_df = _discretize_labels(train_df)
    val_df = _discretize_labels(val_df)

    def build_lgb_data(df: pd.DataFrame):
        X = df[xs_features].values
        y = df["label_int"].values.astype(int)
        groups = df.groupby(time_col).size().values
        return X, y, groups

    X_train, y_train, groups_train = build_lgb_data(train_df)
    X_val, y_val, groups_val = build_lgb_data(val_df)

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train,
                            feature_name=xs_features, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, group=groups_val,
                          reference=train_set, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 6],
        "label_gain": [0, 1, 3, 7, 15],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "min_data_in_leaf": 50 if is_intraday else 20,
        "verbose": -1,
        "seed": 42,
    }

    log.info("Training LambdaRank model...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )

    # Evaluate: NDCG on validation set
    val_pred = model.predict(X_val)
    ndcg_scores = _compute_ndcg_by_group(val_pred, y_val, groups_val, k=6)
    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    # Feature importance
    importance = dict(zip(xs_features, model.feature_importance(importance_type="gain").tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

    # Compute top-K hit rate
    hit_rate = _compute_topk_hit_rate(val_pred, y_val, groups_val, k=6)

    # NDCG comparison gate: don't deploy a worse model
    config_path = os.path.join(save_dir, config_file)
    model_path = os.path.join(save_dir, model_file)
    _ndcg_tolerance = 0.02
    _deploy = True
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as _f:
                old_metrics = json.load(_f)
            old_ndcg = old_metrics.get("mean_ndcg", 0.0)
            if mean_ndcg < old_ndcg - _ndcg_tolerance:
                log.warning(
                    "Selector retrain REJECTED: NDCG %.4f < %.4f − %.2f (old kept)",
                    mean_ndcg, old_ndcg, _ndcg_tolerance,
                )
                _deploy = False
            else:
                log.info(
                    "Selector NDCG gate passed: %.4f → %.4f (old=%.4f, tol=%.2f)",
                    old_ndcg, mean_ndcg, old_ndcg, _ndcg_tolerance,
                )
    except Exception as exc:
        log.warning("Could not load old selector metrics for comparison: %s", exc)

    t_min = train_df[time_col].min()
    t_max = train_df[time_col].max()
    v_min = val_df[time_col].min()
    v_max = val_df[time_col].max()
    metrics = {
        "mode": mode,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_range": f"{t_min} → {t_max}",
        "val_range": f"{v_min} → {v_max}",
        "coins": sorted(panel["symbol"].unique().tolist()),
        "n_coins": int(panel["symbol"].nunique()),
        "ndcg_k": 6,
        "mean_ndcg": round(mean_ndcg, 4),
        "topk_hit_rate": round(hit_rate, 4),
        "best_iteration": model.best_iteration,
        "feature_importance": dict(sorted_imp),
        "deployed": _deploy,
    }

    if _deploy:
        model.save_model(model_path)
        log.info("Saved selector model to %s", model_path)
        with open(config_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Saved selector config to %s", config_path)
    else:
        log.info("Old selector model retained (new model NOT saved)")

    log.info("=== Selector Training Results ===")
    log.info("  NDCG@6 (val): %.4f%s", mean_ndcg,
             "" if _deploy else " [REJECTED — old model kept]")
    log.info("  Top-6 hit rate: %.1f%%", hit_rate * 100)
    log.info("  Best iteration: %d", model.best_iteration)
    log.info("  Feature importance:")
    for feat, imp in sorted_imp[:5]:
        log.info("    %s: %.1f", feat, imp)

    return metrics


def _compute_ndcg_by_group(
    pred: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int
) -> List[float]:
    """Compute NDCG@k for each query group."""
    ndcg_scores = []
    offset = 0
    for g_size in groups:
        g_pred = pred[offset:offset + g_size]
        g_labels = labels[offset:offset + g_size]
        offset += g_size
        if len(g_pred) < 2:
            continue
        # Sort by predicted score descending
        order = np.argsort(-g_pred)
        sorted_labels = g_labels[order]
        # DCG
        dcg = 0.0
        for i in range(min(k, len(sorted_labels))):
            dcg += (2 ** sorted_labels[i] - 1) / np.log2(i + 2)
        # Ideal DCG
        ideal_order = np.argsort(-g_labels)
        ideal_labels = g_labels[ideal_order]
        idcg = 0.0
        for i in range(min(k, len(ideal_labels))):
            idcg += (2 ** ideal_labels[i] - 1) / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return ndcg_scores


def _compute_topk_hit_rate(
    pred: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int
) -> float:
    """What fraction of true top-K coins are in predicted top-K?"""
    hits = []
    offset = 0
    for g_size in groups:
        g_pred = pred[offset:offset + g_size]
        g_labels = labels[offset:offset + g_size]
        offset += g_size
        actual_k = min(k, len(g_pred))
        if actual_k < 2:
            continue
        pred_topk = set(np.argsort(-g_pred)[:actual_k])
        true_topk = set(np.argsort(-g_labels)[:actual_k])
        overlap = len(pred_topk & true_topk)
        hits.append(overlap / actual_k)
    return float(np.mean(hits)) if hits else 0.0


# ---------------------------------------------------------------------------
# Inference (used by paper_trader)
# ---------------------------------------------------------------------------

@dataclass
class SelectorOutput:
    """Result of running the coin selector."""
    date: str
    rankings: List[Tuple[str, float]]  # [(coin, score), ...] sorted desc
    selected: List[str]                 # all coins ordered by score (best first)
    regime_ok: bool = True


class CoinSelector:
    """Load and run the trained cross-sectional selector."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        mode: str = "swing",
    ):
        import lightgbm as lgb

        self.mode = mode
        is_intraday = (mode == "intraday")
        self.xs_features = XS_FEATURES_INTRADAY if is_intraday else XS_FEATURES

        if model_dir is None:
            model_dir = CRYPTO_MODEL_DIR
        model_file = SELECTOR_MODEL_FILE_INTRADAY if is_intraday else SELECTOR_MODEL_FILE
        config_file = SELECTOR_CONFIG_FILE_INTRADAY if is_intraday else SELECTOR_CONFIG_FILE
        model_path = os.path.join(model_dir, model_file)
        config_path = os.path.join(model_dir, config_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Selector model not found: {model_path}")

        self.model = lgb.Booster(model_file=model_path)

        self.config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

        # Use feature list from config if available (matches what model was trained on),
        # otherwise fall back to the current constant. This ensures forward-compatibility
        # when new features are added but the model hasn't been retrained yet.
        if self.config.get("feature_importance"):
            trained_features = list(self.config["feature_importance"].keys())
            if set(trained_features) != set(self.xs_features):
                log.info("Selector model uses %d features (code has %d) — using model's feature list",
                         len(trained_features), len(self.xs_features))
            self.xs_features = trained_features

        log.info("Loaded coin selector (mode=%s, %d features) from %s",
                 mode, len(self.xs_features), model_path)

    def rank(
        self,
        universe_data: Dict[str, pd.DataFrame],
    ) -> SelectorOutput:
        """Rank all coins cross-sectionally. Returns ALL coins ordered by score.

        No hard cutoff — the paper trader applies its own position limits downstream.

        Args:
            universe_data: {symbol: bars_df} — daily bars (swing) or 5-min bars (intraday)

        Returns:
            SelectorOutput with rankings (all coins) and selected (all coins).
        """
        if self.mode == "intraday":
            panel = build_xs_panel_intraday(universe_data)
            time_col = "snapshot"
        else:
            panel = build_xs_panel(universe_data)
            time_col = "date"

        if panel.empty:
            log.warning("Empty panel — no coins to rank")
            return SelectorOutput(date="", rankings=[], selected=[])

        # Use only the latest snapshot
        latest_date = panel[time_col].max()
        latest = panel[panel[time_col] == latest_date].copy()

        date_str = str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)

        if len(latest) < 2:
            log.warning("Only %d coins on latest date — need ≥2", len(latest))
            return SelectorOutput(
                date=date_str,
                rankings=[], selected=[],
            )

        # Predict scores
        X = latest[self.xs_features].values
        scores = self.model.predict(X)
        latest = latest.copy()
        latest["score"] = scores

        # Sort by score descending — return ALL coins ranked
        latest = latest.sort_values("score", ascending=False)
        rankings = [(row["symbol"], float(row["score"]))
                    for _, row in latest.iterrows()]
        all_symbols = [sym for sym, _ in rankings]

        today = date_str
        log.info("Selector rankings (%s): %s",
                 today,
                 ", ".join(f"{s} ({sc:.2f})" for s, sc in rankings[:8]))

        return SelectorOutput(
            date=today,
            rankings=rankings,
            selected=all_symbols,
        )


def rank_coins_today(
    universe: Optional[List[str]] = None,
    model_dir: Optional[str] = None,
    mode: str = "swing",
) -> SelectorOutput:
    """Convenience: fetch data and rank all coins for today.

    Args:
        mode: "swing" for daily bars, "intraday" for 5-min bars.
    """
    if universe is None:
        dynamic = load_universe(model_dir or CRYPTO_MODEL_DIR)
        universe = dynamic if dynamic else list(CRYPTO_UNIVERSE)
    if model_dir is None:
        model_dir = CRYPTO_MODEL_DIR

    selector = CoinSelector(model_dir=model_dir, mode=mode)

    log.info("Fetching data for %d coins (mode=%s)...", len(universe), mode)
    if mode == "intraday":
        data = fetch_universe_intraday_data(universe, lookback_days=INTRADAY_LOOKBACK_DAYS)
    else:
        data = fetch_universe_data(universe, lookback_days=400)

    return selector.rank(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for training and ranking."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-sectional coin selector")
    sub = parser.add_subparsers(dest="action", help="Action to perform")

    # Train
    train_p = sub.add_parser("train", help="Train the selector model")
    train_p.add_argument("--train-end", default="2025-01-01",
                         help="OOS cutoff date (default: 2025-01-01)")
    train_p.add_argument("--save-dir", default=None,
                         help="Directory to save model (default: models/crypto/)")
    train_p.add_argument("--lookback", type=int, default=1500,
                         help="Days of historical data to fetch (default: 1500)")

    # Rank
    rank_p = sub.add_parser("rank", help="Rank all coins for today")
    rank_p.add_argument("--model-dir", default=None,
                         help="Directory with trained model (default: models/crypto/)")

    # Train-intraday
    train_i = sub.add_parser("train-intraday", help="Train the intraday selector model")
    train_i.add_argument("--train-end", default="2025-01-01",
                         help="OOS cutoff date (default: 2025-01-01)")
    train_i.add_argument("--save-dir", default=None,
                         help="Directory to save model (default: models/crypto_intraday/)")
    train_i.add_argument("--lookback", type=int, default=INTRADAY_LOOKBACK_DAYS,
                         help=f"Days of historical data (default: {INTRADAY_LOOKBACK_DAYS})")

    # Rank-intraday
    rank_i = sub.add_parser("rank-intraday", help="Rank all coins with intraday features")
    rank_i.add_argument("--model-dir", default=None,
                         help="Directory with trained model (default: models/crypto_intraday/)")

    args = parser.parse_args()

    if args.action == "train":
        metrics = train_selector(
            train_end=args.train_end,
            save_dir=args.save_dir,
            lookback_days=args.lookback,
        )
        print(f"\n  Selector trained: NDCG@6={metrics['mean_ndcg']:.4f}, "
              f"hit_rate={metrics['topk_hit_rate']:.1%}")

    elif args.action == "rank":
        result = rank_coins_today(model_dir=args.model_dir)
        print(f"\n  === Coin Rankings ({result.date}) ===\n")
        for i, (sym, score) in enumerate(result.rankings, 1):
            print(f"  {i:2d}. {sym:<12s} score={score:+.3f}")

    elif args.action == "train-intraday":
        from utils import get_model_dir
        save = args.save_dir or get_model_dir("crypto_intraday")
        metrics = train_selector(
            train_end=args.train_end,
            save_dir=save,
            lookback_days=args.lookback,
            mode="intraday",
        )
        print(f"\n  Intraday selector trained: NDCG@6={metrics['mean_ndcg']:.4f}, "
              f"hit_rate={metrics['topk_hit_rate']:.1%}")

    elif args.action == "rank-intraday":
        from utils import get_model_dir
        model_dir = args.model_dir or get_model_dir("crypto_intraday")
        result = rank_coins_today(model_dir=model_dir, mode="intraday")
        print(f"\n  === Intraday Coin Rankings ({result.date}) ===\n")
        for i, (sym, score) in enumerate(result.rankings, 1):
            print(f"  {i:2d}. {sym:<12s} score={score:+.3f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
