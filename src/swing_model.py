#!/usr/bin/env python3
"""
Swing XGBoost Factor Model (v3 — replaces PatchTST)
====================================
XGBRegressor for swing-trading commodity and EM ETFs (GLD, SLV, EEM, EWT).
Predicts forward 10-day expected return.

Why XGBoost instead of PatchTST:
  - PatchTST val direction accuracy was only 55.4% (near-random)
  - Root cause: transformer needs 10K+ samples; we have ~500 daily bars (2 years)
  - Grinsztajn et al. (NeurIPS 2022): tree models outperform deep learning on tabular data
    with N < 10K. XGBoost wins on the 58/79 benchmark datasets in this size range.
  - XGBoost's max_depth=4 regularisation handles 40-50 features at 500 samples cleanly.

Key features added for commodity/EM ETFs:
  - GLD/SLV: real_yield_spread + gold_silver_ratio (from cross_asset_signals)
  - EEM:     usd_strength + copper_ret5 (China demand proxy)
  - EWT:     usd_strength + sox_ret5 (TSMC/semiconductor supply chain)
  - All:     IBS, rel_momentum_vs_SPY, TSMOM (ret252)

Optional PatchTST ensemble:
  If a legacy {symbol}_patchtst_swing.pt file exists, SwingPredictor blends
  70% XGBoost + 30% PatchTST. Once XGBoost models are retrained and validated
  the old PatchTST files can be deleted to disable the blend.

Usage (via main.py):
    python main.py train-swing --symbols EWT,GLD,EEM,SLV --provider yahoo
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from signals_engine import (
    PROJECT_ROOT, build_adapter,
    compute_rsi, compute_atr, compute_macd, compute_bollinger_bands,
    compute_adx, compute_momentum_quality, compute_hurst_series, RSI_PERIOD,
)
from utils import _fetch_vix_for_training, DEFAULT_MODEL_DIR, SWING_MODEL_DIR, CRYPTO_MODEL_DIR, COST_THRESHOLD, TARGET_RETURN
from cross_asset_signals import CrossAssetFeatureBuilder, get_cross_asset_features
# Options flow features removed — CBOE does not publish P/C ratio as
# downloadable CSV and FRED does not carry it.  VIX term structure is
# already captured by cboe_vix_term_ratio in CBOE_FEATURES.
OPTIONS_FLOW_FEATURES: list[str] = []
from alpha_signals import AlphaFeatureBuilder, get_alpha_features
from factor_signals import FactorFeatureBuilder, get_factor_features
from market_signals import MarketSignalBuilder, get_market_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("swing_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base features — same as expansion group + IBS (good swing timing signal)
SWING_BASE_FEATURES = [
    "ret252",                   # TSMOM: 12-month trailing return (Moskowitz 2012)
    "ret63",                    # Quarterly momentum
    "ret21",                    # Monthly momentum
    "ret5",                     # Weekly (mean-reversion component)
    "rel_momentum_vs_spy_63",   # Cross-sectional alpha vs SPY (3m)
    "rel_momentum_vs_spy_252",  # Cross-sectional alpha vs SPY (12m)
    "ibs",                      # Internal Bar Strength — (close-low)/(high-low)
    "rsi14",
    "vol20",
    "adx",
    "bb_pct_b",
    "bb_bandwidth",
    "macd_hist_norm",
    "momentum_quality",
    "vol_regime",
    "vix_pctrank",
    "dv_accel",
    "hurst",
]

# Regime context features — baked into model so it learns regime-dependent behavior
# instead of relying solely on inference-time regime filters
REGIME_FEATURES = [
    "spy_above_sma200",       # 1 if SPY > 200-day MA, else 0 (bull/bear flag)
    "spy_trend_strength",     # SPY distance from SMA(200) as % (how bullish/bearish)
    "vix_regime",             # VIX level (raw, normalized by scaler)
    "vix_zscore",             # VIX z-score vs 60-day rolling mean/std
    "sector_etf_rel_spy_21",  # sector ETF 21d return minus SPY 21d return
]

# Crypto-specific regime features — only included for *-USD symbols
# BTC acts as the "SPY of crypto" — its trend dictates alt-coin regimes.
CRYPTO_REGIME_FEATURES = [
    "btc_sma200_flag",       # 1 if BTC > 200-day SMA (bull/bear market for all crypto)
    "btc_realized_vol_30",   # BTC 30-day realized volatility (annualized)
    "btc_drawdown",          # BTC drawdown from 90-day high (0 to -1 range)
    "eth_btc_ratio_zscore",  # ETH/BTC ratio z-score vs 60-day rolling mean/std
    # Short-term BTC trend features — the model needs to know if BTC
    # has been going UP or DOWN recently, not just whether it's above SMA200.
    "btc_ret5",              # BTC 5-day return (immediate trend)
    "btc_ret21",             # BTC 21-day return (monthly trend)
    "btc_trend_strength",    # BTC price / SMA20 ratio - 1 (>0 = above MA, <0 = below)
    "btc_momentum_accel",    # btc_ret5 - btc_ret5.shift(5): is BTC accelerating?
]

# Crypto sentiment features — OKX L/S ratio, Fear & Greed, Polymarket
try:
    from crypto_sentiment_features import get_crypto_sentiment_features, CryptoSentimentBuilder
except ImportError:
    get_crypto_sentiment_features = None
    CryptoSentimentBuilder = None

# Per-symbol supplement features (fetched from proxy ETFs)
SWING_SUPPLEMENT_FEATURES: Dict[str, List[str]] = {
    "EEM": ["copper_ret5"],   # Copper = China/EM demand proxy
    "EWT": ["sox_ret5"],      # SOX semiconductor = TSMC/Taiwan tech
}

# Proxy ETF tickers for supplement features
_SUPPLEMENT_TICKERS: Dict[str, str] = {
    "copper_ret5": "CPER",   # United States Copper Index Fund
    "sox_ret5":    "SOXX",   # iShares Semiconductor ETF
}

# XGBoost hyperparams
_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,          # shallower than options XGBoost (fewer relevant features)
    learning_rate=0.05,
    min_child_weight=15,  # conservative — ~3% of 500-sample dataset
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.2,
    eval_metric="rmse",
    random_state=42,
    verbosity=0,
    early_stopping_rounds=30,
)

# Triple-barrier thresholds for swing (wider — daily ETFs, longer holds)
TB_PT_PCT = 0.020
TB_SL_PCT = 0.012
TB_HORIZON = 7
FORWARD_DAYS = 10         # regression label: 10-day forward return
FORWARD_DAYS_SHORT = 5    # short-horizon model: 5-day forward return

# Commodity ETFs that can be pooled into a single model with symbol embedding
COMMODITY_POOL = ["GLD", "SLV", "GDX", "PDBC", "USO"]

# Cluster pools for cross-sectional training — each trains ONE shared XGBoost
# with symbol_id enabling symbol-specific tree splits.
POOL_CLUSTERS: Dict[str, list] = {
    "commodity":      ["GLD", "SLV", "GDX", "GDXJ", "PDBC", "USO", "IAU", "URNM", "XLB"],
    "fixed_income":   ["BND", "LQD", "HYG", "EMB", "TLT", "IEF", "SHY", "TIP"],
    "intl_developed": ["VGK", "EWG", "EWJ", "EWU", "EWA", "EWC", "EWH", "EWY"],
    "intl_emerging":  ["EEM", "VWO", "EWZ", "EWW", "EWT", "MCHI", "INDA"],
    "us_growth":      ["QQQ", "IWF", "IGV", "SMH", "SOXX", "XLK", "ARKK", "CIBR"],
    "us_value":       ["SPY", "IWM", "IWB", "VTV", "VBR", "MTUM", "QUAL", "USMV"],
    "us_sector":      ["XLE", "XLF", "XLI", "XLP", "XLRE", "XLU", "XLV", "XLY"],
    "crypto_etf":     ["IBIT", "FBTC", "ETHA"],
}

# Build flat lookup: symbol → pool name
SYMBOL_TO_POOL: Dict[str, str] = {}
for _pool_name, _pool_syms in POOL_CLUSTERS.items():
    for _sym in _pool_syms:
        SYMBOL_TO_POOL[_sym] = _pool_name

# Symbol → integer ID for pooled models (XGBoost learns symbol-specific splits)
POOL_SYMBOL_IDS: Dict[str, int] = {sym: i for i, sym in enumerate(COMMODITY_POOL)}

# Per-cluster symbol IDs
CLUSTER_SYMBOL_IDS: Dict[str, Dict[str, int]] = {
    name: {sym: i for i, sym in enumerate(syms)}
    for name, syms in POOL_CLUSTERS.items()
}

# TFT hyperparams — intentionally small to avoid overfitting on N~800 samples
# Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon
# Time Series Forecasting", International Journal of Forecasting.
# Adopted by Two Sigma, Goldman Sachs Marquee, and Google Ads team internally.
_TFT_SEQ_LEN = 20    # 20-day lookback window
_TFT_HIDDEN  = 32    # small — N~800 samples, regularisation-first
_TFT_HEADS   = 4     # must divide _TFT_HIDDEN
_TFT_DROPOUT = 0.25  # high dropout combats small-dataset overfitting
_TFT_WEIGHT  = 0.30  # TFT contribution in ensemble (XGBoost = 1 - this)


# ---------------------------------------------------------------------------
# Crypto detection helper
# ---------------------------------------------------------------------------

def _is_crypto_swing(symbol: str | None) -> bool:
    """True if symbol is a crypto pair in yahoo format (e.g. BTC-USD, ETH-USD)."""
    if symbol is None:
        return False
    return symbol.upper().endswith("-USD") and len(symbol) >= 6


# ---------------------------------------------------------------------------
# Config-driven feature selection (v2)
# ---------------------------------------------------------------------------

_FEATURE_CONFIG_CACHE: Dict[str, Optional[dict]] = {}


def load_feature_config(symbol: str) -> Optional[dict]:
    """Load feature config JSON for a symbol's asset class.

    Returns None if no config file exists (triggers fallback to hardcoded).
    """
    asset_class = "crypto_swing" if _is_crypto_swing(symbol) else "equity_swing"
    cache_key = asset_class

    if cache_key in _FEATURE_CONFIG_CACHE:
        return _FEATURE_CONFIG_CACHE[cache_key]

    config_path = os.path.join(PROJECT_ROOT, "config", f"features_{asset_class}.json")
    if not os.path.exists(config_path):
        _FEATURE_CONFIG_CACHE[cache_key] = None
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        _FEATURE_CONFIG_CACHE[cache_key] = config
        log.info("Loaded feature config from %s (%d core features)",
                 config_path, len(config.get("core_features", [])))
        return config
    except (json.JSONDecodeError, IOError) as e:
        log.warning("Failed to load feature config %s: %s", config_path, e)
        _FEATURE_CONFIG_CACHE[cache_key] = None
        return None


def resolve_features_from_config(symbol: str, config: dict) -> List[str]:
    """Resolve core + group + per_coin features for a symbol."""
    features = list(config.get("core_features", []))

    coin_groups = config.get("coin_groups", {})
    group_overrides = config.get("group_overrides", {})

    # Apply "crypto_all" override if present
    if "crypto_all" in group_overrides:
        override = group_overrides["crypto_all"]
        features.extend(override.get("add", []))
        for f in override.get("remove", []):
            if f in features:
                features.remove(f)

    # Determine which group(s) this symbol belongs to
    for group_name, members in coin_groups.items():
        if symbol in members and group_name in group_overrides:
            override = group_overrides[group_name]
            features.extend(override.get("add", []))
            for f in override.get("remove", []):
                if f in features:
                    features.remove(f)

    # Apply per-coin overrides
    per_coin = config.get("per_coin_overrides", {})
    if symbol in per_coin:
        override = per_coin[symbol]
        features.extend(override.get("add", []))
        for f in override.get("remove", []):
            if f in features:
                features.remove(f)

    return features


# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------

def get_swing_feature_cols(symbol: str | None = None) -> list:
    """Return full feature list for a swing symbol.

    If a config file exists at config/features_{asset_class}.json,
    loads features from config. Otherwise falls back to hardcoded constants.
    """
    # Config-driven selection (if config exists)
    if symbol:
        config = load_feature_config(symbol)
        if config is not None:
            cols = resolve_features_from_config(symbol, config)
            # Add symbol_id for pooled cluster models
            if symbol in SYMBOL_TO_POOL and "symbol_id" not in cols:
                cols.append("symbol_id")
            return cols

    # --- Fallback: hardcoded feature assembly (original behavior) ---
    cols: list = list(SWING_BASE_FEATURES)

    # Regime context features (SPY trend, VIX regime — baked into model, not just inference-time)
    cols.extend(REGIME_FEATURES)

    # Crypto regime features (BTC trend, vol, drawdown, ETH/BTC ratio)
    if _is_crypto_swing(symbol):
        cols.extend(CRYPTO_REGIME_FEATURES)
        if get_crypto_sentiment_features is not None:
            cols.extend(get_crypto_sentiment_features())

    # Symbol-specific supplement features
    if symbol and symbol in SWING_SUPPLEMENT_FEATURES:
        cols.extend(SWING_SUPPLEMENT_FEATURES[symbol])

    # Cross-asset features (real_yield_spread/gold_silver_ratio for GLD/SLV;
    # usd_strength for EWT/EEM; treasury_slope for all)
    if symbol:
        cols.extend(get_cross_asset_features(symbol))
    else:
        cols.append("treasury_slope")

    # Options flow (VIX term structure, P/C ratios)
    cols.extend(OPTIONS_FLOW_FEATURES)

    # Alpha, factor, market from existing builders
    if symbol:
        cols.extend(get_alpha_features(symbol))
    if symbol:
        cols.extend(get_factor_features(symbol))
    if symbol:
        cols.extend(get_market_features(symbol))

    # Symbol ID for pooled cluster models (ordinal int → XGBoost learns symbol-specific splits)
    if symbol and symbol in SYMBOL_TO_POOL:
        cols.append("symbol_id")

    return cols


# ---------------------------------------------------------------------------
# Feature Engineering (tabular — no sequences)
# ---------------------------------------------------------------------------

class SwingFeatureEngine:
    """Build tabular factor features for swing-group ETFs.

    Each bar produces one feature vector (point-in-time).
    No sequence windows — XGBoost is non-sequential.
    """

    def __init__(self):
        self._scaler_params: Optional[dict] = None
        self._cross_builder: Optional[CrossAssetFeatureBuilder] = None
        self._options_engine = None
        self._alpha_builder: Optional[AlphaFeatureBuilder] = None
        self._factor_builder: Optional[FactorFeatureBuilder] = None
        self._market_builder: Optional[MarketSignalBuilder] = None
        self._sentiment_builder: Optional[CryptoSentimentBuilder] = None
        self._supplement_cache: dict = {}
        self._crypto_cache: dict = {}  # cache BTC-USD / ETH-USD close series

    def build_features(
        self,
        bars_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        spy_bars: Optional[pd.DataFrame] = None,
        symbol: str | None = None,
        fred_key: str | None = None,
    ) -> pd.DataFrame:
        """Build tabular feature matrix.

        Args:
            bars_df:  OHLCV for the symbol (from adapter)
            vix_df:   VIX daily data
            spy_bars: SPY daily bars for relative momentum (optional, fetched
                      automatically when None during training)
            symbol:   Symbol name for per-symbol feature routing
        """
        close  = bars_df["close"].astype(float)
        high   = bars_df["high"].astype(float)
        low    = bars_df["low"].astype(float)
        volume = bars_df["volume"].astype(float)
        open_  = bars_df["open"].astype(float)
        df = pd.DataFrame(index=bars_df.index)
        annualize = np.sqrt(252)

        # --- Momentum signals ---
        df["ret252"] = close.pct_change(252)
        df["ret63"]  = close.pct_change(63)
        df["ret21"]  = close.pct_change(21)
        df["ret5"]   = close.pct_change(5)

        # Relative momentum vs SPY (cross-sectional alpha)
        if spy_bars is not None and not spy_bars.empty:
            bar_dates = pd.to_datetime(bars_df["ts"]).dt.date.values
            spy_dates = pd.to_datetime(spy_bars["ts"]).dt.date
            spy_close = spy_bars["close"].astype(float)
            spy_map = dict(zip(spy_dates.values, spy_close.values))
            spy_aligned = pd.Series(
                [spy_map.get(d, np.nan) for d in bar_dates],
                index=bars_df.index,
            ).ffill()
            df["rel_momentum_vs_spy_63"]  = df["ret63"]  - spy_aligned.pct_change(63)
            df["rel_momentum_vs_spy_252"] = df["ret252"] - spy_aligned.pct_change(252)
        else:
            df["rel_momentum_vs_spy_63"]  = 0.0
            df["rel_momentum_vs_spy_252"] = 0.0

        # --- IBS: Internal Bar Strength (Pagonidis 2014) ---
        df["ibs"] = (close - low) / (high - low).replace(0, np.nan)

        # --- Technical indicators ---
        df["rsi14"]  = compute_rsi(close, RSI_PERIOD) / 100.0
        df["vol20"]  = close.pct_change().rolling(20).std() * annualize
        df["adx"]    = compute_adx(high, low, close, period=14) / 100.0

        _, _, _, pct_b, bandwidth = compute_bollinger_bands(close, window=20)
        df["bb_pct_b"]    = pct_b
        df["bb_bandwidth"] = bandwidth

        _, _, macd_hist = compute_macd(close)
        df["macd_hist_norm"] = macd_hist / close.replace(0, np.nan)

        df["momentum_quality"] = compute_momentum_quality(close, window=20)

        vol_short = close.pct_change().rolling(10).std() * annualize
        vol_long  = close.pct_change().rolling(30).std() * annualize
        df["vol_regime"] = vol_short / vol_long.replace(0, np.nan)

        df["hurst"] = compute_hurst_series(close, window=252, step=5)

        # --- Volume ---
        dv = close * volume
        dv_ma5 = dv.rolling(5).mean()
        dv_ma10 = dv.rolling(10).mean()
        df["dv_accel"] = (dv_ma5 - dv_ma10) / dv_ma10.replace(0, np.nan)

        # --- VIX percentile rank ---
        bar_dates_d = pd.to_datetime(bars_df["ts"]).dt.date
        vix_map = {}
        if not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_map[d] = row["vix"]
        vix_series = bar_dates_d.map(lambda d: vix_map.get(d, np.nan)).astype(float)
        vix_series = pd.Series(vix_series.values, index=bars_df.index).ffill()
        df["vix_pctrank"] = vix_series.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # --- Regime context features (SPY trend, VIX regime, sector relative) ---
        if spy_bars is not None and not spy_bars.empty:
            spy_close_s = spy_bars["close"].astype(float)
            spy_sma200 = spy_close_s.rolling(200, min_periods=100).mean()
            # Build SPY regime as a time series, then align to symbol's dates
            spy_regime_raw = (spy_close_s > spy_sma200).astype(float)
            spy_strength_raw = (spy_close_s - spy_sma200) / spy_sma200.replace(0, np.nan)
            spy_ret21_raw = spy_close_s.pct_change(21)

            spy_dates_for_regime = pd.to_datetime(spy_bars["ts"]).dt.date
            regime_map = dict(zip(spy_dates_for_regime.values, spy_regime_raw.values))
            strength_map = dict(zip(spy_dates_for_regime.values, spy_strength_raw.values))
            spy_ret21_map = dict(zip(spy_dates_for_regime.values, spy_ret21_raw.values))

            df["spy_above_sma200"] = bar_dates_d.map(lambda d: regime_map.get(d, np.nan))
            df["spy_trend_strength"] = bar_dates_d.map(lambda d: strength_map.get(d, np.nan))
            # Sector relative strength: symbol's 21d return minus SPY's 21d return
            sym_ret21 = close.pct_change(21)
            spy_ret21_aligned = bar_dates_d.map(lambda d: spy_ret21_map.get(d, np.nan))
            spy_ret21_series = pd.Series(spy_ret21_aligned.values, index=bars_df.index).astype(float).ffill()
            df["sector_etf_rel_spy_21"] = sym_ret21 - spy_ret21_series
        else:
            df["spy_above_sma200"] = 0.0
            df["spy_trend_strength"] = 0.0
            df["sector_etf_rel_spy_21"] = 0.0

        # VIX regime features
        df["vix_regime"] = vix_series
        vix_mean60 = vix_series.rolling(60, min_periods=20).mean()
        vix_std60 = vix_series.rolling(60, min_periods=20).std().replace(0, 1)
        df["vix_zscore"] = (vix_series - vix_mean60) / vix_std60

        # Forward-fill regime features
        for col in REGIME_FEATURES:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0.0)

        # --- Crypto regime features (BTC trend, vol, drawdown, ETH/BTC ratio) ---
        if _is_crypto_swing(symbol):
            self._build_crypto_regime(df, bars_df)

        # --- Crypto sentiment features (OKX L/S, Fear & Greed, Polymarket) ---
        if _is_crypto_swing(symbol):
            try:
                if CryptoSentimentBuilder is None:
                    raise ImportError("crypto_sentiment_features not available")
                if self._sentiment_builder is None:
                    self._sentiment_builder = CryptoSentimentBuilder()
                sent_df = self._sentiment_builder.build_features(bars_df, symbol)
                for col in sent_df.columns:
                    df[col] = sent_df[col]
            except Exception as exc:
                log.warning("Crypto sentiment features failed for %s: %s", symbol, exc)
                if get_crypto_sentiment_features is not None:
                    for col in get_crypto_sentiment_features():
                        if col not in df.columns:
                            df[col] = np.nan

        # --- Supplement features (copper / SOX) ---
        if symbol and symbol in SWING_SUPPLEMENT_FEATURES:
            supp_df = self._fetch_supplement(bars_df, symbol)
            for col in supp_df.columns:
                df[col] = supp_df[col]

        # --- Cross-asset features (full superset — let config/IC filter decide) ---
        try:
            if self._cross_builder is None:
                self._cross_builder = CrossAssetFeatureBuilder(
                    fred_key=fred_key or os.environ.get("FRED_API_KEY"))
            cross_df = self._cross_builder.build_all_features(bars_df)
            for col in cross_df.columns:
                df[col] = cross_df[col]
        except Exception as exc:
            log.warning("Cross-asset features failed for %s: %s", symbol, exc)
            from cross_asset_signals import ALL_CROSS_ASSET_FEATURES as _ALL_CA
            for col in _ALL_CA:
                if col not in df.columns:
                    df[col] = np.nan

        # --- Options flow features (removed — data sources unavailable) ---

        # --- Alpha features ---
        if symbol:
            try:
                if self._alpha_builder is None:
                    self._alpha_builder = AlphaFeatureBuilder()
                alpha_df = self._alpha_builder.build_features(bars_df, symbol)
                for col in alpha_df.columns:
                    df[col] = alpha_df[col]
            except Exception as exc:
                log.warning("Alpha features failed for %s: %s", symbol, exc)
                for col in get_alpha_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # --- Factor features ---
        if symbol:
            try:
                if self._factor_builder is None:
                    self._factor_builder = FactorFeatureBuilder()
                factor_df = self._factor_builder.build_features(bars_df, symbol)
                for col in factor_df.columns:
                    df[col] = factor_df[col]
            except Exception as exc:
                log.warning("Factor features failed for %s: %s", symbol, exc)
                for col in get_factor_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # --- Market features ---
        if symbol:
            try:
                if self._market_builder is None:
                    self._market_builder = MarketSignalBuilder()
                market_df = self._market_builder.build_features(bars_df, symbol)
                for col in market_df.columns:
                    df[col] = market_df[col]
            except Exception as exc:
                log.warning("Market features failed for %s: %s", symbol, exc)
                for col in get_market_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # Drop NaN warmup on base features; fill optional features with 0
        # Symbol ID for pooled cluster models (e.g. commodity: GLD=0, SLV=1, ...)
        if symbol and symbol in SYMBOL_TO_POOL:
            pool_name = SYMBOL_TO_POOL[symbol]
            df["symbol_id"] = float(CLUSTER_SYMBOL_IDS[pool_name][symbol])

        all_cols = get_swing_feature_cols(symbol)
        df = df.dropna(subset=SWING_BASE_FEATURES)
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = df[col].ffill().fillna(0.0)

        return df[all_cols]

    def _fetch_supplement(self, bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Fetch proxy ETF data for EEM (copper) and EWT (SOX).

        Uses the last date in bars_df as cutoff to prevent train-set leakage.
        """
        import yfinance as yf

        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        # Use last bar date as cutoff to prevent future data leakage
        cutoff_date = bar_dates.iloc[-1]
        supplement_cols = SWING_SUPPLEMENT_FEATURES.get(symbol, [])
        df = pd.DataFrame(index=bars_df.index)

        for col in supplement_cols:
            ticker = _SUPPLEMENT_TICKERS.get(col)
            if ticker is None:
                df[col] = 0.0
                continue
            try:
                cache_key = f"{ticker}_{cutoff_date}"
                if cache_key not in self._supplement_cache:
                    data = yf.download(ticker, period="5y", progress=False,
                                       auto_adjust=True)
                    if data.empty:
                        self._supplement_cache[cache_key] = pd.Series(dtype=float)
                    else:
                        close = data["Close"]
                        if hasattr(close, "squeeze"):
                            close = close.squeeze()
                        # Truncate to cutoff date to prevent leakage
                        close = close[close.index.date <= cutoff_date]
                        self._supplement_cache[cache_key] = close

                proxy = self._supplement_cache[cache_key]
                if proxy.empty:
                    df[col] = 0.0
                    continue

                proxy_map = dict(zip(proxy.index.date, proxy.values))
                proxy_aligned = bar_dates.map(lambda d: proxy_map.get(d, np.nan))
                proxy_series = pd.Series(
                    proxy_aligned.values, index=bars_df.index
                ).astype(float).ffill()
                df[col] = proxy_series.pct_change(5)

            except Exception as exc:
                log.warning("Supplement feature %s (%s) failed: %s", col, ticker, exc)
                df[col] = 0.0

        return df

    # ------------------------------------------------------------------
    # Crypto regime helpers
    # ------------------------------------------------------------------

    def _fetch_crypto_close(self, ticker: str,
                            cutoff_date=None) -> pd.Series:
        """Fetch daily close for a crypto ticker (e.g. BTC-USD), cached.

        Args:
            cutoff_date: if set, truncate data to this date to prevent leakage.
        """
        cache_key = f"{ticker}_{cutoff_date}" if cutoff_date else ticker
        if cache_key in self._crypto_cache:
            return self._crypto_cache[cache_key]
        import yfinance as yf
        try:
            data = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if data.empty:
                self._crypto_cache[cache_key] = pd.Series(dtype=float)
            else:
                close = data["Close"]
                if hasattr(close, "squeeze"):
                    close = close.squeeze()
                if cutoff_date is not None:
                    close = close[close.index.date <= cutoff_date]
                self._crypto_cache[cache_key] = close
        except Exception as exc:
            log.warning("Failed to fetch %s for crypto regime: %s", ticker, exc)
            self._crypto_cache[cache_key] = pd.Series(dtype=float)
        return self._crypto_cache[cache_key]

    def _build_crypto_regime(self, df: pd.DataFrame, bars_df: pd.DataFrame) -> None:
        """Add CRYPTO_REGIME_FEATURES columns in-place.

        Fetches BTC-USD and ETH-USD daily close, aligns to bars_df dates,
        and computes the four crypto regime features.
        """
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        cutoff_date = bar_dates.iloc[-1]
        annualize = np.sqrt(365)  # crypto trades 365 days/year

        btc_close = self._fetch_crypto_close("BTC-USD", cutoff_date=cutoff_date)
        eth_close = self._fetch_crypto_close("ETH-USD", cutoff_date=cutoff_date)

        if btc_close.empty:
            for col in CRYPTO_REGIME_FEATURES:
                df[col] = 0.0
            return

        # Align BTC close to symbol's bar dates
        btc_map = dict(zip(btc_close.index.date, btc_close.values))
        btc_aligned = pd.Series(
            bar_dates.map(lambda d: btc_map.get(d, np.nan)).values,
            index=bars_df.index,
        ).astype(float).ffill()

        # 1. btc_sma200_flag: BTC > 200-day SMA
        btc_sma200 = btc_aligned.rolling(200, min_periods=100).mean()
        df["btc_sma200_flag"] = (btc_aligned > btc_sma200).astype(float)

        # 2. btc_realized_vol_30: 30-day realized vol (annualized for crypto = sqrt(365))
        btc_ret = btc_aligned.pct_change()
        df["btc_realized_vol_30"] = btc_ret.rolling(30, min_periods=10).std() * annualize

        # 3. btc_drawdown: drawdown from 90-day rolling high (0 to -1 range)
        btc_high90 = btc_aligned.rolling(90, min_periods=30).max()
        df["btc_drawdown"] = (btc_aligned - btc_high90) / btc_high90.replace(0, np.nan)

        # 4. eth_btc_ratio_zscore: ETH/BTC ratio z-score vs 60-day rolling stats
        if not eth_close.empty:
            eth_map = dict(zip(eth_close.index.date, eth_close.values))
            eth_aligned = pd.Series(
                bar_dates.map(lambda d: eth_map.get(d, np.nan)).values,
                index=bars_df.index,
            ).astype(float).ffill()

            eth_btc = eth_aligned / btc_aligned.replace(0, np.nan)
            eb_mean = eth_btc.rolling(60, min_periods=20).mean()
            eb_std = eth_btc.rolling(60, min_periods=20).std().replace(0, 1)
            df["eth_btc_ratio_zscore"] = (eth_btc - eb_mean) / eb_std
        else:
            df["eth_btc_ratio_zscore"] = 0.0

        # 5. btc_ret5: BTC 5-day return (immediate trend direction)
        df["btc_ret5"] = btc_aligned.pct_change(5)

        # 6. btc_ret21: BTC 21-day return (monthly trend)
        df["btc_ret21"] = btc_aligned.pct_change(21)

        # 7. btc_trend_strength: BTC price / SMA(20) - 1
        # >0 means BTC is above its 20-day moving average (short-term bullish)
        btc_sma20 = btc_aligned.rolling(20, min_periods=10).mean()
        df["btc_trend_strength"] = (btc_aligned / btc_sma20.replace(0, np.nan)) - 1.0

        # 8. btc_momentum_accel: is BTC momentum speeding up or slowing down?
        btc_ret5 = btc_aligned.pct_change(5)
        df["btc_momentum_accel"] = btc_ret5 - btc_ret5.shift(5)

        # Forward-fill and fill NaN warmup
        for col in CRYPTO_REGIME_FEATURES:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0.0)

    def fit_scaler(self, features_df: pd.DataFrame) -> None:
        self._scaler_params = {
            "mean": features_df.mean(),
            "std": features_df.std().replace(0, 1),
        }

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler_params is None:
            raise RuntimeError("Call fit_scaler() first.")
        return (features_df - self._scaler_params["mean"]) / self._scaler_params["std"]

    def save_scaler(self, path: str) -> None:
        data = {
            "mean": self._scaler_params["mean"].to_dict(),
            "std":  self._scaler_params["std"].to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_scaler(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._scaler_params = {
            "mean": pd.Series(data["mean"]),
            "std":  pd.Series(data["std"]),
        }


# ---------------------------------------------------------------------------
# Label preparation
# ---------------------------------------------------------------------------

def _prepare_tabular_regression(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    vol_normalize: bool = True,
    cost_adjust_bps: float = 0.0,
) -> tuple:
    """Forward N-day return regression labels for XGBoost.

    Args:
        vol_normalize: If True, divide returns by trailing 20-day realized vol.
            This makes the model learn risk-adjusted signal strength rather than
            raw returns that conflate alpha with volatility regime.
            At prediction time, multiply output by current vol to recover real units.
        cost_adjust_bps: If > 0, subtract this many bps from |fwd_ret| as round-trip
            cost. Teaches the model to ignore moves that won't survive trading costs.

    Returns (X, y, vol_mean) as float32 numpy arrays + mean vol for config.
    """
    full_close = bars_df["close"].astype(float).values
    bar_positions = bars_df.index.get_indexer(features_df.index)
    feature_values = features_df.values

    # Trailing 20-day realized vol (annualized) for vol normalization
    close_series = pd.Series(full_close)
    trailing_vol = close_series.pct_change().rolling(20).std().values
    # Floor vol to prevent division by near-zero in calm markets
    VOL_FLOOR = 0.005 / np.sqrt(252)  # ~0.03% daily = ~5% annualized floor

    # Validate alignment: all feature rows must map to a valid bar position
    n_valid = (bar_positions >= 0).sum()
    n_total = len(bar_positions)
    if n_valid == 0:
        raise ValueError("XGBoost label alignment: no feature dates found in bars_df index "
                         f"({n_total} features, 0 matched)")
    n_missing = n_total - n_valid
    if n_missing > 0:
        log.warning("XGBoost label alignment: %d/%d feature rows have no matching bar",
                    n_missing, n_total)

    cost_frac = cost_adjust_bps / 10000.0  # convert bps to decimal

    X_list, y_list, vol_list = [], [], []
    for i in range(len(feature_values)):
        bar_pos = bar_positions[i]
        if bar_pos < 0 or bar_pos + forward_days >= len(full_close):
            continue
        entry_price = full_close[bar_pos]
        if entry_price <= 0:
            continue
        fwd_ret = (full_close[bar_pos + forward_days] - entry_price) / entry_price

        # Cost-adjusted: subtract estimated round-trip cost from magnitude
        if cost_frac > 0:
            fwd_ret = fwd_ret - np.sign(fwd_ret) * cost_frac

        # Vol-normalized: divide by trailing vol so model learns risk-adjusted alpha
        raw_vol = trailing_vol[bar_pos] if bar_pos < len(trailing_vol) else np.nan
        bar_vol = raw_vol if not np.isnan(raw_vol) and raw_vol > VOL_FLOOR else VOL_FLOOR
        if vol_normalize:
            fwd_ret = fwd_ret / bar_vol
            vol_list.append(bar_vol)

        X_list.append(feature_values[i])
        y_list.append(fwd_ret)

    y_arr = np.array(y_list, dtype=np.float32)
    vol_mean = float(np.mean(vol_list)) if vol_list else 0.0

    # Percentile-based Winsorization (1st/99th) — more robust than fixed ±10%
    if len(y_arr) > 20:
        p1, p99 = np.percentile(y_arr, [1, 99])
        y_arr = np.clip(y_arr, p1, p99)
        log.info("Label Winsorization: clipped to [%.4f, %.4f] (1st/99th pctile)", p1, p99)

    if vol_normalize:
        log.info("Labels vol-normalized (mean_vol=%.4f, cost_adj=%.1fbps)", vol_mean, cost_adjust_bps)

    # Sample uniqueness weighting: each forward_days label overlaps with
    # up to (2 * forward_days - 1) neighbors. Weight inversely by overlap count.
    n_samples = len(y_arr)
    sample_weights = np.ones(n_samples, dtype=np.float32)
    if n_samples > forward_days:
        for i in range(n_samples):
            # Count how many neighbors share overlapping forward windows
            lo = max(0, i - forward_days + 1)
            hi = min(n_samples, i + forward_days)
            overlap = hi - lo  # includes self
            sample_weights[i] = 1.0 / overlap
        # Normalize so mean weight = 1.0 (preserves effective sample size)
        sample_weights *= n_samples / sample_weights.sum()
        log.info("Sample uniqueness weights: min=%.3f, max=%.3f (fwd=%dd)",
                 sample_weights.min(), sample_weights.max(), forward_days)

    return np.array(X_list, dtype=np.float32), y_arr, vol_mean, sample_weights


# ---------------------------------------------------------------------------
# TFT Architecture (Temporal Fusion Transformer — Lim et al., Google 2021)
# ---------------------------------------------------------------------------
# Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon
#             Time Series Forecasting", IJF 2021.
# Used by Two Sigma (multi-asset vol prediction), Goldman Sachs Marquee
# (regime detection), AQR (cross-sectional momentum scoring).
#
# Key advantage over PatchTST on small datasets:
#   - Variable Selection Networks learn WHICH features matter per timestep
#     (acts as learned feature importance, not random patches)
#   - GRN gating prevents gradient explosion on N~800 samples
#   - Interpretable attention shows WHICH past days drove the prediction
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Gated Residual Network — TFT building block (Lim et al. 2021, eq. 2-4).

    Architecture: input → Dense → GeLU → Dropout → Dense(×2) → GLU gate
                  + skip-projected residual → LayerNorm
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, output_size * 2)  # ×2 for GLU
        self.skip = (nn.Linear(input_size, output_size)
                     if input_size != output_size else nn.Identity())
        self.norm = nn.LayerNorm(output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        h1, h2 = h.chunk(2, dim=-1)    # Gated Linear Unit
        h = h1 * torch.sigmoid(h2)
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network — learns per-timestep feature importance.

    One small GRN per input feature → weighted sum controlled by a
    softmax over a separate selection GRN. Allows the model to ignore
    noisy features (e.g. cross-asset features irrelevant on a given day).
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        half_h = max(4, hidden_size // 2)
        # Per-feature GRNs: scalar input → hidden representation
        self.var_grns = nn.ModuleList([
            GRN(1, half_h, hidden_size, dropout) for _ in range(n_features)
        ])
        # Softmax weight selector
        self.selector = GRN(n_features, hidden_size, n_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_features)  →  (..., hidden_size)"""
        weights = torch.softmax(self.selector(x), dim=-1)   # (..., n_features)
        # Process each feature: stack → (..., n_features, hidden_size)
        processed = torch.stack(
            [self.var_grns[i](x[..., i:i+1]) for i in range(self.n_features)],
            dim=-2,
        )
        return (processed * weights.unsqueeze(-1)).sum(dim=-2)   # (..., hidden_size)


class TFTSwingModel(nn.Module):
    """Temporal Fusion Transformer for swing ETF return prediction.

    Calibrated for N~800 daily samples (lookback=1500):
        hidden=32, seq_len=20, dropout=0.25, n_heads=4.

    Pipeline:
        1. VSN — per-timestep variable selection  (batch, T, n_feat) → (batch, T, H)
        2. LSTM — temporal encoding               (batch, T, H) → (batch, T, H)
        3. Add & Norm skip connection
        4. Multi-head self-attention              (batch, T, H) → (batch, T, H)
        5. Add & Norm skip connection
        6. GRN on last timestep                   (batch, H) → (batch, H)
        7. Linear head                            (batch, H) → (batch,)
    """

    def __init__(self, n_features: int, seq_len: int = _TFT_SEQ_LEN,
                 hidden: int = _TFT_HIDDEN, n_heads: int = _TFT_HEADS,
                 dropout: float = _TFT_DROPOUT):
        super().__init__()
        self.seq_len   = seq_len
        self.hidden    = hidden
        self.vsn       = VariableSelectionNetwork(n_features, hidden, dropout)
        self.lstm      = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_norm = nn.LayerNorm(hidden)
        self.attn      = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                               batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.out_grn   = GRN(hidden, hidden, hidden, dropout)
        self.head      = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features)  →  (batch,)"""
        v = self.vsn(x)                                  # (B, T, H)
        h, _ = self.lstm(v)                              # (B, T, H)
        h = self.lstm_norm(h + v)                        # skip + norm
        a, _ = self.attn(h, h, h)                       # (B, T, H)
        a = self.attn_norm(a + h)                        # skip + norm
        out = self.out_grn(a[:, -1, :])                  # last timestep
        # Safety clamp — prevents extreme predictions from small-dataset
        # overfitting (N~800). Training labels are 1st/99th pctile winsorized
        # (typically ±3-8%), so ±15% is a loose safety net.
        return torch.clamp(self.head(out).squeeze(-1), -0.15, 0.15)


def _prepare_sequence_regression(
    features_norm: pd.DataFrame,
    bars_df: pd.DataFrame,
    seq_len: int = _TFT_SEQ_LEN,
    forward_days: int = FORWARD_DAYS,
    vol_normalize: bool = True,
    cost_adjust_bps: float = 0.0,
) -> tuple:
    """Sliding-window sequences for TFT training.

    For each valid bar i (i >= seq_len, i + forward_days < len(bars)):
        X[i] = features_norm[i-seq_len : i]   shape: (seq_len, n_features)
        y[i] = forward 10-day return at bar i (vol-normalized if enabled)

    Labels are winsorized to 1st/99th percentile (same as XGBoost) to ensure
    both models train on the same target scale.
    """
    full_close     = bars_df["close"].astype(float).values
    bar_positions  = bars_df.index.get_indexer(features_norm.index)
    feature_values = features_norm.values
    n = len(feature_values)

    # Trailing vol for normalization (same as XGBoost)
    close_series = pd.Series(full_close)
    trailing_vol = close_series.pct_change().rolling(20).std().values
    VOL_FLOOR = 0.005 / np.sqrt(252)
    cost_frac = cost_adjust_bps / 10000.0

    X_list, y_list = [], []
    for i in range(seq_len, n):
        bar_pos = bar_positions[i]
        if bar_pos < 0 or bar_pos + forward_days >= len(full_close):
            continue
        entry_price = full_close[bar_pos]
        if entry_price <= 0:
            continue
        fwd_ret = (full_close[bar_pos + forward_days] - entry_price) / entry_price

        if cost_frac > 0:
            fwd_ret = fwd_ret - np.sign(fwd_ret) * cost_frac

        if vol_normalize:
            raw_vol = trailing_vol[bar_pos] if bar_pos < len(trailing_vol) else np.nan
            bar_vol = raw_vol if not np.isnan(raw_vol) and raw_vol > VOL_FLOOR else VOL_FLOOR
            fwd_ret = fwd_ret / bar_vol

        X_list.append(feature_values[i - seq_len:i])
        y_list.append(fwd_ret)

    y_arr = np.array(y_list, dtype=np.float32)

    # Percentile-based Winsorization (1st/99th) — same as XGBoost labels
    if len(y_arr) > 20:
        p1, p99 = np.percentile(y_arr, [1, 99])
        y_arr = np.clip(y_arr, p1, p99)
        log.info("[TFT] Label Winsorization: clipped to [%.4f, %.4f] (1st/99th pctile)", p1, p99)

    return (np.array(X_list, dtype=np.float32), y_arr)


def train_tft_swing_model(
    symbol: str,
    features_norm: pd.DataFrame,
    bars_df: pd.DataFrame,
    save_dir: str = DEFAULT_MODEL_DIR,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seq_len: int = _TFT_SEQ_LEN,
) -> Optional["TFTSwingModel"]:
    """Train TFT swing model on pre-normalised features.

    Called from train_swing_model() after XGBoost has been saved.
    Shares the same scaler — features_norm is already z-scored.
    """
    X_all, y_all = _prepare_sequence_regression(
        features_norm, bars_df, seq_len,
        vol_normalize=True, cost_adjust_bps=0.0,  # cost already in XGB labels; TFT shares target scale
    )
    if len(X_all) < 50:
        log.warning("[TFT] %s: only %d sequences — skipping TFT.", symbol, len(X_all))
        return None

    split    = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val,   y_val   = X_all[split:], y_all[split:]
    n_features = X_all.shape[2]
    log.info("[TFT] %s: %d train / %d val seqs, %d features",
             symbol, len(X_train), len(X_val), n_features)

    model     = TFTSwingModel(n_features=n_features, seq_len=seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    train_dl = DataLoader(TensorDataset(X_t, y_t),
                          batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_mse    = float("inf")
    best_state      = None
    patience_count  = 0
    patience        = 15

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_mse  = criterion(val_pred, y_v).item()
            val_dir  = float(((val_pred > 0) == (y_v > 0)).float().mean())

        if val_mse < best_val_mse:
            best_val_mse   = val_mse
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info("[TFT] %s: early stop epoch %d  val_mse=%.6f  dir=%.3f",
                         symbol, epoch, best_val_mse, val_dir)
                break

        if epoch % 20 == 0:
            log.info("[TFT] %s ep%d: val_mse=%.6f  dir=%.3f", symbol, epoch, val_mse, val_dir)

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_pred = model(X_v)
        final_mse  = criterion(final_pred, y_v).item()
        final_dir  = float(((final_pred > 0) == (y_v > 0)).float().mean())

    log.info("[TFT] %s: final val_mse=%.6f  dir_acc=%.3f", symbol, final_mse, final_dir)

    tft_path     = os.path.join(save_dir, f"{symbol}_tft_swing.pt")
    tft_cfg_path = os.path.join(save_dir, f"{symbol}_tft_swing_config.json")

    torch.save(model.state_dict(), tft_path)
    with open(tft_cfg_path, "w") as f:
        json.dump({
            "symbol":                 symbol,
            "model_type":             "tft_swing",
            "n_features":             n_features,
            "seq_len":                seq_len,
            "hidden":                 _TFT_HIDDEN,
            "n_heads":                _TFT_HEADS,
            "val_mse":                round(final_mse, 8),
            "val_direction_accuracy": round(final_dir, 4),
            "n_train":                len(X_train),
            "n_val":                  len(X_val),
        }, f, indent=2)

    # Quality gate: only keep TFT if it beats random (>50% direction accuracy)
    # If it doesn't, XGBoost-only is better — delete the file so predictor falls back
    if final_dir < 0.50:
        log.warning("[TFT] %s: dir_acc=%.3f < 0.50 — discarding TFT, XGBoost-only for this symbol.",
                    symbol, final_dir)
        try:
            os.remove(tft_path)
            os.remove(tft_cfg_path)
        except OSError:
            pass
        return None

    log.info("[TFT] %s: saved → %s  (dir_acc=%.3f)", symbol, tft_path, final_dir)
    if final_dir >= 0.52:
        try:
            from model_monitor import ModelMonitor
            ModelMonitor().clear_model_pause(symbol, reason="retrained_swing_model")
        except Exception as exc:
            log.debug("Model pause clear skipped for %s: %s", symbol, exc)
    else:
        log.info("Pause state retained for %s: TFT dir_acc=%.3f < 0.52", symbol, final_dir)
    return model


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_swing_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    epochs: int = 60,         # ignored for XGBoost (kept for CLI compat)
    lr: float = 5e-4,         # ignored for XGBoost (kept for CLI compat)
    batch_size: int = 32,     # ignored for XGBoost (kept for CLI compat)
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
    train_end: Optional[str] = None,  # e.g. "2023-08-01" — hard cutoff for OOS separation
    train_recent: bool = False,       # walk-forward: train on first 75%, OOS on last 25%
) -> Optional[xgb.XGBRegressor]:
    """Train XGBoost swing model (v3 regression).

    1. Fetch daily data (symbol + SPY for relative momentum)
    2. Build tabular features (17 base + macro supplement + cross-asset + ...)
    3. Forward 10-day return regression labels
    4. Train XGBRegressor with walk-forward split
    5. Save model + scaler + config

    Args:
        train_end: If set (e.g. "2023-08-01"), all data on or after this date is
                   excluded from training and validation. This creates a clean OOS
                   boundary so backtests starting from train_end are truly OOS.
        train_recent: If True, walk-forward split — train on first 75% of data,
                      reserve last 25% as OOS. Sets train_end to the 75% date.
                      Overrides train_end.
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info("=== Training swing XGBoost for %s ===", symbol)
    if train_recent:
        log.info("Walk-forward mode: train on first 75%%, OOS on last 25%%.")
    elif train_end:
        log.info("Training cutoff: data before %s only (OOS boundary).", train_end)

    # 1. Fetch data
    log.info("Fetching %d daily bars for %s...", lookback, symbol)
    bars = adapter.fetch_daily(symbol, lookback)
    log.info("Got %d bars.", len(bars))

    log.info("Fetching SPY bars for relative momentum...")
    spy_bars = adapter.fetch_daily("SPY", lookback)
    log.info("Got %d SPY bars.", len(spy_bars))

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500),
                                     include_live=False)
    log.info("Got %d VIX rows.", len(vix_df))

    # 1b. Walk-forward split: train on first 75%, OOS on last 25%
    #     This keeps training data in the past and OOS in the future,
    #     matching production (train on history, deploy forward).
    #     With ~750 bars (3yr Yahoo), gives ~560 train vs 188 OOS.
    if train_recent:
        split_idx = int(len(bars) * 0.75)
        cutoff_date = None
        date_col = next((c for c in bars.columns if c.lower() in ("date", "timestamp", "time", "ts")), None)
        if date_col:
            cutoff_date = str(pd.to_datetime(bars[date_col].iloc[split_idx]).date())
        elif hasattr(bars.index, "dtype") and str(bars.index.dtype).startswith("datetime"):
            cutoff_date = str(bars.index[split_idx].date())
        # Keep only bars before cutoff for training; OOS starts at cutoff_date
        bars = bars.iloc[:split_idx].copy().reset_index(drop=True)
        spy_cut = int(len(spy_bars) * 0.75)
        spy_bars = spy_bars.iloc[:spy_cut].copy().reset_index(drop=True)
        if "date" in vix_df.columns and cutoff_date:
            vix_df = vix_df[pd.to_datetime(vix_df["date"]) < pd.Timestamp(cutoff_date)].copy()
        train_end = cutoff_date  # store for config — backtests starting here are OOS
        log.info("Walk-forward split at %s: training on %d bars (first 75%%), OOS after.", cutoff_date, len(bars))

    # 1c. Apply train_end cutoff — drop any rows on or after the OOS boundary
    elif train_end:
        cutoff = pd.Timestamp(train_end)
        # fetch_daily() returns a RangeIndex DataFrame; find the date column by name
        date_col = next((c for c in bars.columns if c.lower() in ("date", "timestamp", "time", "ts")), None)
        cutoff_applied = False
        if date_col:
            # Handle tz-aware vs tz-naive comparison
            ts_series = pd.to_datetime(bars[date_col])
            if ts_series.dt.tz is not None:
                cutoff = cutoff.tz_localize(ts_series.dt.tz)
            bars_cut = bars[ts_series < cutoff].copy().reset_index(drop=True)
            if len(bars_cut) == 0:
                log.warning("train_end cutoff left 0 bars for %s (symbol may have launched after %s). Training on all available data.", symbol, train_end)
            else:
                bars     = bars_cut
                spy_bars = spy_bars[pd.to_datetime(spy_bars[date_col]) < cutoff].copy().reset_index(drop=True)
                cutoff_applied = True
        elif hasattr(bars.index, "dtype") and str(bars.index.dtype).startswith("datetime"):
            # DatetimeIndex fallback
            bars_cut = bars[bars.index < cutoff].copy()
            if len(bars_cut) == 0:
                log.warning("train_end cutoff left 0 bars for %s (symbol may have launched after %s). Training on all available data.", symbol, train_end)
            else:
                bars     = bars_cut
                spy_bars = spy_bars[spy_bars.index < cutoff].copy()
                cutoff_applied = True
        else:
            log.warning("train_end cutoff: no date column or DatetimeIndex found — cutoff not applied! columns=%s", list(bars.columns[:5]))
        # Only filter vix_df when bars cutoff was actually applied (keeps date ranges consistent)
        if cutoff_applied and "date" in vix_df.columns:
            cutoff_naive = pd.Timestamp(train_end)  # VIX dates are tz-naive
            vix_df = vix_df[pd.to_datetime(vix_df["date"]) < cutoff_naive].copy()
        log.info("After cutoff: %d bars, %d SPY bars, %d VIX rows.", len(bars), len(spy_bars), len(vix_df))

    # 2. Build features
    engine = SwingFeatureEngine()
    features = engine.build_features(
        bars, vix_df, spy_bars=spy_bars, symbol=symbol, fred_key=fred_key
    )
    log.info("Built %d feature rows for %s.", len(features), symbol)

    n_features = len(get_swing_feature_cols(symbol))
    if len(features) < 50:
        log.error("Not enough data for %s (%d rows).", symbol, len(features))
        return None

    # 3. Scaler (fit on training portion only)
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    full_norm = engine.transform(features)

    # 4. Forward return labels (vol-normalized + cost-adjusted)
    try:
        from cost_model import get_symbol_costs
        rt_cost_bps = get_symbol_costs(symbol).round_trip_bps
    except Exception:
        rt_cost_bps = 7.0  # default mid-tier ETF cost
    X_all, y_all, vol_mean, w_all = _prepare_tabular_regression(
        full_norm, bars, forward_days=FORWARD_DAYS,
        vol_normalize=True, cost_adjust_bps=rt_cost_bps,
    )
    log.info("Labeled %d samples. Mean vol-norm return: %+.4f (cost_adj=%.1fbps)",
             len(y_all), y_all.mean(), rt_cost_bps)

    # 5. Walk-forward split (80/20 within the training-only window)
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val,   y_val   = X_all[split:], y_all[split:]
    w_train = w_all[:split]
    log.info("Train: %d, Val: %d.", len(X_train), len(X_val))

    if len(X_val) < 10:
        log.error("Validation set too small for %s. Need more data.", symbol)
        return None

    # 5b. Demean labels for small datasets
    #     When train_recent=True on short histories (e.g. crypto ~188 samples),
    #     the training window can be regime-biased (all bearish or all bullish).
    #     XGBoost's base_score auto-sets to the label mean, making ALL predictions
    #     carry the regime's sign regardless of features.
    #     Fix: center labels to zero, train on deviations, add mean back at prediction.
    #     The mean is saved in the config so the predictor can reconstruct.
    label_mean = 0.0
    if len(X_train) < 300:
        label_mean = float(y_train.mean())
        y_train = y_train - label_mean
        y_val = y_val - label_mean
        log.info("Demeaned labels: subtracted %.4f%% (regime bias removal)", label_mean * 100)

    # 6. Train XGBRegressor — adapt hyperparams to dataset size
    xgb_params = dict(_XGB_PARAMS)
    n_train = len(X_train)
    xgb_params["min_child_weight"] = max(3, n_train // 40)    # ~2.5% of data
    xgb_params["reg_lambda"] = max(1.0, 2.0 * n_train / 500)  # scale L2 with data
    xgb_params["reg_alpha"] = max(0.05, 0.2 * n_train / 500)  # scale L1 with data
    if n_train < 300:
        # Small dataset: early stopping on tiny val RMSE is unreliable.
        xgb_params["learning_rate"] = 0.01
        xgb_params["n_estimators"] = 80
        xgb_params["max_depth"] = 3
        xgb_params.pop("early_stopping_rounds", None)
    log.info("XGB adaptive params: mcw=%d, lambda=%.2f, alpha=%.2f, lr=%.3f, depth=%d, n_est=%d (n_train=%d)",
             xgb_params["min_child_weight"], xgb_params["reg_lambda"], xgb_params["reg_alpha"],
             xgb_params["learning_rate"], xgb_params["max_depth"], xgb_params["n_estimators"], n_train)
    model = xgb.XGBRegressor(**xgb_params)
    if xgb_params.get("early_stopping_rounds"):
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    # 7. Evaluate
    val_preds    = model.predict(X_val)
    val_rmse     = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    direction_acc = float(np.mean((val_preds > 0) == (y_val > 0)))
    log.info("Val RMSE: %.6f | Direction accuracy: %.3f", val_rmse, direction_acc)

    # Feature importance — top 10
    feature_cols = get_swing_feature_cols(symbol)
    importances  = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    log.info("Top 10 features: %s",
             ", ".join(f"{feature_cols[i]}={importances[i]:.3f}"
                       for i in top_idx if i < len(feature_cols)))

    # Save feature importances to JSON for stability tracking
    from datetime import date as _date
    imp_path = os.path.join(save_dir, f"{symbol}_xgb_swing_importance.json")
    _all_idx = np.argsort(importances)[::-1]
    _top10 = [[feature_cols[i], round(float(importances[i]), 6)]
              for i in _all_idx[:10] if i < len(feature_cols)]
    imp_data = {
        "symbol": symbol,
        "trained_at": str(_date.today()),
        "feature_cols": list(feature_cols),
        "importances": [round(float(v), 6) for v in importances],
        "top_10": _top10,
    }
    # Append to history for stability tracking
    imp_history_path = os.path.join(save_dir, f"{symbol}_xgb_swing_importance_history.json")
    try:
        with open(imp_history_path, "r", encoding="utf-8") as _fh:
            history = json.load(_fh)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(imp_data)
    with open(imp_history_path, "w", encoding="utf-8") as _fh:
        json.dump(history, _fh, indent=2)
    with open(imp_path, "w", encoding="utf-8") as _fh:
        json.dump(imp_data, _fh, indent=2)
    log.info("Feature importances saved → %s", imp_path)

    # Check stability against previous run
    if len(history) >= 2:
        try:
            from model_monitor import check_feature_stability
            stable, warnings = check_feature_stability(imp_data, history[-2])
            if not stable:
                for w in warnings:
                    log.warning("[Feature Stability] %s: %s", symbol, w)
            else:
                log.info("[Feature Stability] %s: stable (no large rank shifts).", symbol)
        except Exception as exc:
            log.debug("Feature stability check skipped: %s", exc)

    # 8. Save
    model_path  = os.path.join(save_dir, f"{symbol}_xgb_swing.joblib")
    scaler_path = os.path.join(save_dir, f"{symbol}_xgb_swing_scaler.json")
    config_path = os.path.join(save_dir, f"{symbol}_xgb_swing_config.json")

    joblib.dump(model, model_path)
    engine.save_scaler(scaler_path)

    config = {
        "symbol":               symbol,
        "model_type":           "xgb_swing",
        "horizon":              f"{FORWARD_DAYS}d",
        "model_version":        "v4_regression_regime",
        "target":               f"forward_{FORWARD_DAYS}d_return",
        "val_rmse":             round(val_rmse, 8),
        "val_direction_accuracy": round(direction_acc, 4),
        "n_train":              len(X_train),
        "n_val":                len(X_val),
        "n_features":           n_features,
        "feature_names":        feature_cols,
        "train_end":            train_end or "all_data",
        "train_mode":           "walk_forward_75_25" if train_recent else "standard",
        "label_mean_removed":   round(label_mean, 8),
        "label_vol_normalized": True,
        "label_cost_adjust_bps": rt_cost_bps,
        "training_vol_mean":    round(vol_mean, 8),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved swing XGBoost → %s (dir_acc=%.3f)", model_path, direction_acc)
    if direction_acc >= 0.52:
        try:
            from model_monitor import ModelMonitor
            ModelMonitor().clear_model_pause(symbol, reason="retrained_swing_model")
        except Exception as exc:
            log.debug("Model pause clear skipped for %s: %s", symbol, exc)
    else:
        log.info("Pause state retained for %s: XGBoost dir_acc=%.3f < 0.52", symbol, direction_acc)

    # Train quantile regression models for confidence bounds
    for qtag, qalpha in [("q25", 0.25), ("q75", 0.75)]:
        try:
            q_params = dict(xgb_params)
            q_params["objective"] = "reg:quantile"
            q_params["quantile_alpha"] = qalpha
            q_params.pop("eval_metric", None)
            q_model = xgb.XGBRegressor(**q_params)
            if q_params.get("early_stopping_rounds"):
                q_model.fit(X_train, y_train, sample_weight=w_train,
                            eval_set=[(X_val, y_val)], verbose=False)
            else:
                q_model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            q_path = os.path.join(save_dir, f"{symbol}_xgb_swing_{qtag}.joblib")
            joblib.dump(q_model, q_path)
            log.info("Saved %s quantile model → %s", qtag, q_path)
        except Exception as exc:
            log.warning("Quantile %s model failed for %s: %s", qtag, symbol, exc)

    # Train TFT ensemble component (60% weight in final blend)
    try:
        train_tft_swing_model(
            symbol=symbol,
            features_norm=full_norm,
            bars_df=bars,
            save_dir=save_dir,
        )
    except Exception as exc:
        log.warning("[TFT] %s: training failed (%s) — XGBoost model still usable.", symbol, exc)

    return model


# ---------------------------------------------------------------------------
# Pooled cluster model — trains single XGBoost on all symbols in a cluster
# with symbol_id feature enabling symbol-specific tree splits.
# ---------------------------------------------------------------------------

def train_pooled_cluster_model(
    cluster_name: str,
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = SWING_MODEL_DIR,
    train_recent: bool = False,
    forward_days: int = FORWARD_DAYS,
) -> Optional[xgb.XGBRegressor]:
    """Train a single XGBoost model on all symbols in a cluster.

    The symbol_id feature acts as a categorical embedding — XGBoost learns
    symbol-specific tree splits naturally, so one model captures shared
    dynamics while adapting to each symbol's idiosyncrasies.

    Benefits over per-symbol models:
      - N× more training data (e.g. 8 symbols × ~500 = ~4000 samples)
      - Shared regime features across correlated assets
      - Symbol-specific behavior via tree routing on symbol_id
    """
    if cluster_name not in POOL_CLUSTERS:
        log.error("Unknown cluster: %s. Available: %s", cluster_name, list(POOL_CLUSTERS.keys()))
        return None

    pool_symbols = POOL_CLUSTERS[cluster_name]
    sym_ids = CLUSTER_SYMBOL_IDS[cluster_name]
    os.makedirs(save_dir, exist_ok=True)
    horizon_tag = f"{forward_days}d"
    log.info("=== Training POOLED %s model (%s horizon, %d symbols: %s) ===",
             cluster_name, horizon_tag, len(pool_symbols), ", ".join(pool_symbols))

    all_X, all_y = [], []
    all_feature_cols = None
    engine = SwingFeatureEngine()

    for sym in pool_symbols:
        try:
            bars = adapter.fetch_daily(sym, lookback)
            if len(bars) < 100:
                log.warning("Skipping %s — only %d bars", sym, len(bars))
                continue
            spy_bars = adapter.fetch_daily("SPY", lookback)
            vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500),
                                             include_live=False)

            features = engine.build_features(bars, vix_df, spy_bars=spy_bars,
                                             symbol=sym, fred_key=fred_key)

            if len(features) < 50:
                log.warning("Skipping %s — only %d feature rows", sym, len(features))
                continue

            feature_cols = list(features.columns)
            if all_feature_cols is None:
                all_feature_cols = feature_cols
            else:
                if feature_cols != all_feature_cols:
                    # Align columns — some symbols may have extra/missing features
                    common = [c for c in all_feature_cols if c in feature_cols]
                    log.warning("Feature mismatch for %s: %d vs %d cols, using %d common",
                                sym, len(feature_cols), len(all_feature_cols), len(common))
                    all_feature_cols = common
                    features = features[common]

            # Scaler: fit on first symbol's training portion, transform all
            if len(all_X) == 0:
                split_idx = int(len(features) * 0.8)
                engine.fit_scaler(features.iloc[:split_idx])
            full_norm = engine.transform(features)

            try:
                from cost_model import get_symbol_costs
                rt_cost = get_symbol_costs(sym).round_trip_bps
            except Exception:
                rt_cost = 7.0
            X, y, _, _w = _prepare_tabular_regression(
                full_norm, bars, forward_days=forward_days,
                vol_normalize=True, cost_adjust_bps=rt_cost,
            )
            all_X.append(X)
            all_y.append(y)
            log.info("  %s: %d samples (mean fwd ret: %+.4f%%)", sym, len(y), y.mean() * 100)

        except Exception as exc:
            log.warning("Skipping %s in pooled training: %s", sym, exc)

    if not all_X:
        log.error("No data for pooled %s model", cluster_name)
        return None

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    log.info("Pooled %s dataset: %d total samples, %d features",
             cluster_name, len(X_all), X_all.shape[1])

    # Walk-forward split (75/25 matching per-symbol models)
    split = int(len(X_all) * 0.75)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Demean if small dataset
    label_mean = 0.0
    if len(X_train) < 300:
        label_mean = float(y_train.mean())
        y_train = y_train - label_mean
        y_val = y_val - label_mean

    # XGBoost with larger dataset params
    xgb_params = dict(_XGB_PARAMS)
    xgb_params["min_child_weight"] = max(5, len(X_train) // 50)
    xgb_params["n_estimators"] = min(500, max(200, len(X_train) // 5))
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_preds = model.predict(X_val)
    val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    direction_acc = float(np.mean((val_preds > 0) == (y_val > 0)))
    ic = float(np.corrcoef(val_preds, y_val)[0, 1]) if len(val_preds) > 5 else 0.0
    log.info("Pooled %s — RMSE: %.6f | Dir acc: %.3f | IC: %.3f",
             cluster_name, val_rmse, direction_acc, ic)

    # Save
    model_tag = f"{cluster_name}_pool_xgb_{horizon_tag}"
    model_path = os.path.join(save_dir, f"{model_tag}.joblib")
    scaler_path = os.path.join(save_dir, f"{model_tag}_scaler.json")
    config_path = os.path.join(save_dir, f"{model_tag}_config.json")

    joblib.dump(model, model_path)
    engine.save_scaler(scaler_path)

    config = {
        "model_type": "xgb_swing_pooled",
        "cluster": cluster_name,
        "symbols": pool_symbols,
        "symbol_ids": sym_ids,
        "horizon": horizon_tag,
        "forward_days": forward_days,
        "val_rmse": round(val_rmse, 8),
        "val_direction_accuracy": round(direction_acc, 4),
        "val_ic": round(ic, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(all_feature_cols) if all_feature_cols else 0,
        "feature_names": all_feature_cols,
        "label_mean_removed": round(label_mean, 8),
        "label_vol_normalized": True,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved pooled %s model → %s", cluster_name, model_path)
    return model


def train_pooled_commodity_model(
    adapter, fred_key=None, lookback=1000, save_dir=SWING_MODEL_DIR,
    train_recent=False, forward_days=FORWARD_DAYS,
):
    """Backward-compat wrapper → trains the 'commodity' cluster."""
    return train_pooled_cluster_model(
        "commodity", adapter, fred_key=fred_key, lookback=lookback,
        save_dir=save_dir, train_recent=train_recent, forward_days=forward_days,
    )


def train_all_pooled_clusters(
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = SWING_MODEL_DIR,
    train_recent: bool = False,
    forward_days: int = FORWARD_DAYS,
) -> Dict[str, Optional[xgb.XGBRegressor]]:
    """Train pooled models for ALL clusters defined in POOL_CLUSTERS."""
    results = {}
    for cluster_name in POOL_CLUSTERS:
        results[cluster_name] = train_pooled_cluster_model(
            cluster_name, adapter, fred_key=fred_key, lookback=lookback,
            save_dir=save_dir, train_recent=train_recent, forward_days=forward_days,
        )
    return results


# ---------------------------------------------------------------------------
# Short-horizon (5-day) swing model — trains alongside the standard 10-day
# ---------------------------------------------------------------------------

def train_short_horizon_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = SWING_MODEL_DIR,
    train_end: Optional[str] = None,
    train_recent: bool = False,
) -> Optional[xgb.XGBRegressor]:
    """Train a 5-day horizon XGBoost swing model.

    Identical pipeline to the standard 10-day model but with FORWARD_DAYS_SHORT=5.
    Captures faster mean-reversion dynamics that the 10-day model misses.
    The predictor can blend both horizons for stronger signals.
    """
    # Reuse train_swing_model but override forward_days via a temporary monkey-patch
    # This is cleaner than duplicating the entire training function
    global FORWARD_DAYS
    original_fwd = FORWARD_DAYS
    try:
        FORWARD_DAYS = FORWARD_DAYS_SHORT
        log.info("=== Training SHORT-HORIZON (5d) swing model for %s ===", symbol)
        model = train_swing_model(
            symbol=symbol,
            adapter=adapter,
            fred_key=fred_key,
            lookback=lookback,
            save_dir=save_dir,
            train_end=train_end,
            train_recent=train_recent,
        )
    finally:
        FORWARD_DAYS = original_fwd

    if model is not None:
        # Rename saved files to include _5d suffix
        for ext in [".joblib", "_scaler.json", "_config.json"]:
            src = os.path.join(save_dir, f"{symbol}_xgb_swing{ext}")
            dst = os.path.join(save_dir, f"{symbol}_xgb_swing_5d{ext}")
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                log.info("Copied %s → %s", os.path.basename(src), os.path.basename(dst))

        # Update the config to reflect 5d horizon
        config_path = os.path.join(save_dir, f"{symbol}_xgb_swing_5d_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            cfg["horizon"] = "5d"
            cfg["target"] = "forward_5d_return"
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    return model


# ---------------------------------------------------------------------------
# Predictor (inference — compatible with ml_model.Predictor interface)
# ---------------------------------------------------------------------------

class SwingPredictor:
    """TFT + XGBoost ensemble swing predictor (v5 — dynamic weighting).

    Compatible with ml_model.Predictor.predict() interface.

    Blends TFT + XGBoost using dynamic weights based on rolling prediction
    error (MAE). Falls back to XGBoost-only if no TFT model file is found.

    v5 changes (from v4):
    - Dynamic ensemble weighting via rolling MAE (was fixed 60/40)
    - Returns both sub-model predictions for transparency

    Architecture reference: Lim et al. (Google, 2021) — IJF.
    """

    model_type = "tft_xgb_swing"
    eod_exit   = False   # swing trades hold overnight

    _ENSEMBLE_WINDOW = 30   # rolling window for dynamic weighting
    _MIN_WEIGHT = 0.20      # floor: never go below 20% for either model
    _MAX_WEIGHT = 0.80      # cap: never exceed 80%

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol    = symbol
        self.model_dir = model_dir
        self.engine    = SwingFeatureEngine()
        self.model:  Optional[xgb.XGBRegressor] = None
        self._tft:   Optional[TFTSwingModel]     = None
        self._tft_seq_len    = _TFT_SEQ_LEN
        self._tft_n_features = 0
        self._xgb_feature_names: Optional[List[str]] = None  # from model config
        self._spy_adapter    = None   # lazy-loaded for relative momentum

        # Dynamic ensemble weighting: track rolling prediction errors.
        # Predictions are buffered for 10 days before comparing to realized
        # return, since the model predicts 10-day forward return.
        from collections import deque
        self._xgb_errors: deque = deque(maxlen=self._ENSEMBLE_WINDOW)
        self._tft_errors: deque = deque(maxlen=self._ENSEMBLE_WINDOW)
        self._pending_preds: deque = deque(maxlen=50)  # (date, xgb_pred, tft_pred, entry_price)
        self._tft_weight = _TFT_WEIGHT  # start at default, then adapt
        self._MIN_OBS_FOR_DYNAMIC = 20  # require 20 completed observations

        # Calibration map: maps raw predictions → calibrated expected returns
        self._calibration = None  # type: ignore[assignment]

        self._load()

    def _load(self) -> None:
        # --- Primary: XGBoost ---
        model_path  = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing.joblib")
        scaler_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_scaler.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained swing XGBoost for {self.symbol}. "
                f"Run: python main.py train-swing --symbols {self.symbol}"
            )
        self.model = joblib.load(model_path)
        self.engine.load_scaler(scaler_path)

        # Load feature names from XGBoost config (handles feature count mismatch
        # when new features are added but model hasn't been retrained yet)
        self._vol_normalized = False  # whether model was trained on vol-normalized labels
        xgb_cfg_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_config.json")
        if os.path.exists(xgb_cfg_path):
            try:
                with open(xgb_cfg_path) as f:
                    xgb_cfg = json.load(f)
                self._xgb_feature_names = xgb_cfg.get("feature_names")
                self._vol_normalized = xgb_cfg.get("label_vol_normalized", False)
            except Exception:
                pass

        log.info("Loaded swing XGBoost for %s.", self.symbol)

        # --- Optional: TFT ensemble (60%) ---
        tft_path     = os.path.join(self.model_dir, f"{self.symbol}_tft_swing.pt")
        tft_cfg_path = os.path.join(self.model_dir, f"{self.symbol}_tft_swing_config.json")
        if os.path.exists(tft_path) and os.path.exists(tft_cfg_path):
            try:
                with open(tft_cfg_path) as f:
                    cfg = json.load(f)
                n_feat  = cfg["n_features"]
                seq_len = cfg.get("seq_len", _TFT_SEQ_LEN)
                hidden  = cfg.get("hidden",  _TFT_HIDDEN)
                n_heads = cfg.get("n_heads", _TFT_HEADS)
                self._tft = TFTSwingModel(n_features=n_feat, seq_len=seq_len,
                                          hidden=hidden, n_heads=n_heads)
                state = torch.load(tft_path, map_location="cpu", weights_only=True)
                self._tft.load_state_dict(state)
                self._tft.eval()
                self._tft_seq_len    = seq_len
                self._tft_n_features = n_feat
                log.info("Loaded TFT for %s (%d feat, seq=%d) — %.0f%% ensemble blend.",
                         self.symbol, n_feat, seq_len, _TFT_WEIGHT * 100)
            except Exception as exc:
                log.warning("Could not load TFT for %s: %s — XGBoost only.", self.symbol, exc)
                self._tft = None

        # --- Optional: Short-horizon (5d) XGBoost for multi-horizon blend ---
        self._xgb_5d: Optional[xgb.XGBRegressor] = None
        self._xgb_5d_feature_names: Optional[List[str]] = None
        xgb_5d_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_5d.joblib")
        if os.path.exists(xgb_5d_path):
            try:
                self._xgb_5d = joblib.load(xgb_5d_path)
                cfg_5d_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_5d_config.json")
                if os.path.exists(cfg_5d_path):
                    with open(cfg_5d_path) as f:
                        cfg_5d = json.load(f)
                    self._xgb_5d_feature_names = cfg_5d.get("feature_names")
                log.info("Loaded 5d short-horizon XGBoost for %s.", self.symbol)
            except Exception as exc:
                log.debug("Could not load 5d model for %s: %s", self.symbol, exc)

        # --- Optional: Pooled cluster model (ensemble with per-symbol) ---
        self._pooled_model: Optional[xgb.XGBRegressor] = None
        self._pooled_feature_names: Optional[List[str]] = None
        if self.symbol in SYMBOL_TO_POOL:
            pool_name = SYMBOL_TO_POOL[self.symbol]
            pooled_path = os.path.join(self.model_dir, f"{pool_name}_pool_xgb_10d.joblib")
            pooled_cfg_path = os.path.join(self.model_dir, f"{pool_name}_pool_xgb_10d_config.json")
            if os.path.exists(pooled_path):
                try:
                    self._pooled_model = joblib.load(pooled_path)
                    if os.path.exists(pooled_cfg_path):
                        with open(pooled_cfg_path) as f:
                            pcfg = json.load(f)
                        self._pooled_feature_names = pcfg.get("feature_names")
                    log.info("Loaded pooled %s model for %s ensemble.", pool_name, self.symbol)
                except Exception as exc:
                    log.debug("Could not load pooled model for %s: %s", self.symbol, exc)

        # --- Optional: Quantile regression models (confidence bounds) ---
        self._xgb_q25: Optional[xgb.XGBRegressor] = None
        self._xgb_q75: Optional[xgb.XGBRegressor] = None
        for qtag, attr in [("q25", "_xgb_q25"), ("q75", "_xgb_q75")]:
            qpath = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_{qtag}.joblib")
            if os.path.exists(qpath):
                try:
                    setattr(self, attr, joblib.load(qpath))
                    log.info("Loaded %s quantile model for %s.", qtag, self.symbol)
                except Exception as exc:
                    log.debug("Could not load %s model for %s: %s", qtag, self.symbol, exc)

        # --- Optional: Calibration map ---
        cal_path = os.path.join(self.model_dir, f"{self.symbol}_calibration.json")
        if os.path.exists(cal_path):
            try:
                from model_monitor import CalibrationMap
                self._calibration = CalibrationMap.load(cal_path)
                if self._calibration.bin_edges is not None:
                    log.info("Loaded calibration map for %s (%d bins).",
                             self.symbol, self._calibration.n_bins)
                else:
                    self._calibration = None
            except Exception as exc:
                log.debug("Could not load calibration for %s: %s", self.symbol, exc)
                self._calibration = None

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = 60) -> dict:
        """Produce regression prediction (v5 — dynamic ensemble weighting).

        Returns standard prediction dict with expected_return field.
        Dynamically adjusts TFT/XGBoost weights based on rolling MAE.
        seq_len is accepted for interface compat but ignored by XGBoost.
        """
        # COST_THRESHOLD and TARGET_RETURN are now imported at module level from utils

        # Lazy-load SPY adapter for relative momentum
        if self._spy_adapter is None:
            from signals_engine import YahooFinanceAdapter
            self._spy_adapter = YahooFinanceAdapter()

        try:
            spy_bars = self._spy_adapter.fetch_daily("SPY", len(bars_df))
        except Exception:
            spy_bars = None

        features = self.engine.build_features(
            bars_df, vix_df, spy_bars=spy_bars, symbol=self.symbol
        )

        if features.empty:
            return {
                "expected_return": 0.0,
                "direction":       "FLAT",
                "probability":     0.5,
                "confidence":      0.0,
                "meta_confidence": 1.0,
                "tradeable":       False,
            }

        features_norm = self.engine.transform(features)

        # Select only features the XGBoost model was trained with
        if self._xgb_feature_names:
            available = [c for c in self._xgb_feature_names if c in features_norm.columns]
            x = features_norm[available].iloc[-1:].values.astype(np.float32)
        else:
            x = features_norm.iloc[-1:].values.astype(np.float32)
        xgb_ret = float(self.model.predict(x)[0])

        # --- Dynamic ensemble weighting ---
        # Evaluate pending predictions where 10 days have elapsed.
        current_price = float(bars_df["close"].iloc[-1])
        current_date = bars_df.index[-1] if hasattr(bars_df.index[-1], 'date') else None
        if current_date is not None:
            resolved = []
            for pp in self._pending_preds:
                pred_date, xgb_p, tft_p, entry_px = pp
                try:
                    days_elapsed = (current_date - pred_date).days
                except (TypeError, AttributeError):
                    continue
                if days_elapsed >= 10 and entry_px > 0:
                    realized_10d = current_price / entry_px - 1
                    self._xgb_errors.append(abs(xgb_p - realized_10d))
                    if tft_p is not None:
                        self._tft_errors.append(abs(tft_p - realized_10d))
                    resolved.append(pp)
            for pp in resolved:
                self._pending_preds.remove(pp)

        # Compute dynamic weights from rolling MAE (inverse error weighting)
        tft_ret = None
        expected_return = xgb_ret
        if self._tft is not None:
            try:
                seq = self._tft_seq_len
                n   = self._tft_n_features
                window = None  # set explicitly so we can guard below
                if len(features_norm) >= seq:
                    # Select TFT features by name (same as XGBoost training order)
                    if self._xgb_feature_names and len(self._xgb_feature_names) == n:
                        tft_cols = [c for c in self._xgb_feature_names
                                    if c in features_norm.columns]
                        # Guard: skip TFT if available features < expected (model trained
                        # on different feature set, e.g. 71-feat model vs 25-feat config)
                        if len(tft_cols) < n:
                            log.warning("[TFT] %s: feature mismatch — need %d, have %d. "
                                        "XGBoost only.", self.symbol, n, len(tft_cols))
                        else:
                            window = features_norm[tft_cols].iloc[-seq:].values.astype(np.float32)
                    else:
                        window = features_norm.iloc[-seq:].values[:, :n].astype(np.float32)

                    if window is not None:
                        xt = torch.FloatTensor(window).unsqueeze(0)   # (1, seq, n_feat)
                        with torch.no_grad():
                            tft_ret = self._tft(xt).item()

                        # NaN guard — fall back to XGBoost if TFT produced NaN
                        if np.isnan(tft_ret) or np.isinf(tft_ret):
                            log.warning("[TFT] %s: NaN/Inf prediction — XGBoost only.",
                                        self.symbol)
                            tft_ret = None

                    if tft_ret is not None:
                        # Dynamic weight calculation (require 20 completed obs)
                        if (len(self._xgb_errors) >= self._MIN_OBS_FOR_DYNAMIC and
                                len(self._tft_errors) >= self._MIN_OBS_FOR_DYNAMIC):
                            xgb_mae = sum(self._xgb_errors) / len(self._xgb_errors)
                            tft_mae = sum(self._tft_errors) / len(self._tft_errors)
                            # Inverse MAE weighting (lower error → higher weight)
                            if xgb_mae > 0 and tft_mae > 0:
                                w_tft = (1 / tft_mae) / (1 / tft_mae + 1 / xgb_mae)
                                w_tft = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, w_tft))
                                self._tft_weight = w_tft
                            # else keep previous weight
                        else:
                            # Pre-warmup: VIX-conditional default.
                            # High vol → XGBoost more robust on tabular features.
                            # Low vol → TFT temporal attention adds value.
                            try:
                                current_vix = float(vix_df["close"].iloc[-1])
                                self._tft_weight = 0.25 if current_vix > 25 else 0.40
                            except Exception:
                                self._tft_weight = _TFT_WEIGHT  # fallback to hardcoded

                        expected_return = self._tft_weight * tft_ret + (1 - self._tft_weight) * xgb_ret
                        log.debug("[TFT] %s: xgb=%.4f tft=%.4f w_tft=%.2f blend=%.4f",
                                  self.symbol, xgb_ret, tft_ret, self._tft_weight, expected_return)
            except Exception as exc:
                log.warning("[TFT] predict failed for %s: %s — XGB only.", self.symbol, exc)
                expected_return = xgb_ret

        # Multi-horizon blend: 5d + 10d (60/40 when agreeing, 10d-only when disagreeing)
        xgb_5d_ret = None
        if self._xgb_5d is not None:
            try:
                if self._xgb_5d_feature_names:
                    cols_5d = [c for c in self._xgb_5d_feature_names if c in features_norm.columns]
                    x5 = features_norm[cols_5d].iloc[-1:].values.astype(np.float32)
                else:
                    x5 = x  # same features
                xgb_5d_ret = float(self._xgb_5d.predict(x5)[0])
                # Blend when horizons agree on direction; dampen when they disagree
                if np.sign(xgb_5d_ret) == np.sign(expected_return):
                    expected_return = 0.60 * expected_return + 0.40 * xgb_5d_ret
                else:
                    # Horizons disagree → dampen signal (low conviction)
                    expected_return *= 0.3
            except Exception:
                pass

        # Blend with pooled cluster model if available (70/30 per-symbol/pooled)
        pooled_ret = None
        if self._pooled_model is not None:
            try:
                if self._pooled_feature_names:
                    pcols = [c for c in self._pooled_feature_names if c in features_norm.columns]
                    px = features_norm[pcols].iloc[-1:].values.astype(np.float32)
                else:
                    px = features_norm.iloc[-1:].values.astype(np.float32)
                pooled_ret = float(self._pooled_model.predict(px)[0])
                # Blend: 70% per-symbol, 30% pooled
                expected_return = 0.70 * expected_return + 0.30 * pooled_ret
            except Exception as exc:
                log.debug("Pooled prediction failed for %s: %s", self.symbol, exc)

        # Vol-denormalize: convert risk-adjusted prediction back to real return units
        # Model was trained on fwd_ret / vol, so multiply by current trailing vol
        if self._vol_normalized:
            close_prices = bars_df["close"].astype(float)
            current_vol = close_prices.pct_change().rolling(20).std().iloc[-1]
            vol_floor = 0.005 / np.sqrt(252)
            current_vol = max(current_vol if not np.isnan(current_vol) else vol_floor, vol_floor)
            expected_return = expected_return * current_vol
            if tft_ret is not None:
                tft_ret = tft_ret * current_vol
            xgb_ret = xgb_ret * current_vol

        # Queue this prediction for 10-day evaluation
        if current_date is not None:
            self._pending_preds.append(
                (current_date, xgb_ret, tft_ret, current_price)
            )

        # Apply calibration map if available: raw prediction → calibrated E[r]
        calibrated_return = expected_return
        if self._calibration is not None:
            calibrated_return = self._calibration.calibrated_return(expected_return)
            # Keep direction from raw prediction but use calibrated magnitude
            # for confidence and sizing (prevents oversized positions on
            # overconfident raw predictions)
            if np.sign(calibrated_return) != np.sign(expected_return) and abs(expected_return) > 0.001:
                # Calibration flipped direction — trust raw for direction,
                # but use lower confidence
                calibrated_return = np.sign(expected_return) * abs(calibrated_return)

        # Quantile bounds (confidence interval from q25/q75 models)
        lower_bound = None
        upper_bound = None
        if self._xgb_q25 is not None and self._xgb_q75 is not None:
            try:
                lower_bound = float(self._xgb_q25.predict(x)[0])
                upper_bound = float(self._xgb_q75.predict(x)[0])
                if self._vol_normalized:
                    lower_bound *= current_vol
                    upper_bound *= current_vol
            except Exception as exc:
                log.debug("Quantile prediction failed for %s: %s", self.symbol, exc)
                lower_bound = upper_bound = None

        if expected_return > COST_THRESHOLD:
            direction = "UP"
        elif expected_return < -COST_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        confidence = min(1.0, abs(calibrated_return) / TARGET_RETURN)
        # Derive display probability from confidence (regression model has no natural prob)
        prob_sign = 1 if expected_return >= 0 else -1
        probability = min(0.95, max(0.05, 0.5 + confidence * 0.45 * prob_sign))

        return {
            "expected_return": round(expected_return, 6),
            "calibrated_return": round(calibrated_return, 6),
            "direction":       direction,
            "probability":     round(probability, 4),
            "confidence":      round(confidence, 4),
            "meta_confidence": 1.0,
            "tradeable":       abs(expected_return) > COST_THRESHOLD,
            "xgb_return":      round(xgb_ret, 6),
            "tft_return":      round(tft_ret, 6) if tft_ret is not None else None,
            "tft_weight":      round(self._tft_weight, 4),
            "lower_bound":     round(lower_bound, 6) if lower_bound is not None else None,
            "upper_bound":     round(upper_bound, 6) if upper_bound is not None else None,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train XGBoost swing model (v3).",
    )
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols (e.g. EWT,GLD,EEM,SLV)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--epochs",   type=int,   default=60,
                        help="Ignored for XGBoost (kept for CLI compat)")
    parser.add_argument("--lookback", type=int,   default=1000,
                        help="Daily bars to fetch (default: 1000)")
    parser.add_argument("--lr",       type=float, default=5e-4,
                        help="Ignored for XGBoost (kept for CLI compat)")
    parser.add_argument("--train-end", default=None,
                        help="Hard cutoff date YYYY-MM-DD — data on/after this date excluded from training. "
                             "Use to create a clean OOS boundary (e.g. --train-end 2023-08-01).")
    parser.add_argument("--train-recent", action="store_true",
                        help="Walk-forward split: train on first 75%% of data, OOS on last 25%%. "
                             "Overrides --train-end.")
    parser.add_argument("--save-dir", default=SWING_MODEL_DIR,
                        help=f"Directory to save model files (default: {SWING_MODEL_DIR})")
    parser.add_argument("--pooled-commodity", action="store_true",
                        help="Train a single pooled model on all commodity ETFs")
    parser.add_argument("--pooled-cluster", default=None,
                        help="Train a pooled model for a specific cluster (e.g. commodity, fixed_income)")
    parser.add_argument("--pooled-all", action="store_true",
                        help="Train pooled models for ALL clusters")
    parser.add_argument("--short-horizon", action="store_true",
                        help="Also train a 5-day short-horizon model alongside the standard 10-day")
    args = parser.parse_args()

    adapter  = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols  = [s.strip().upper() for s in args.symbols.split(",")]

    # Pooled cluster models
    if args.pooled_all:
        train_all_pooled_clusters(
            adapter=adapter, fred_key=fred_key, lookback=args.lookback,
            save_dir=args.save_dir, train_recent=args.train_recent,
        )
    elif args.pooled_cluster:
        train_pooled_cluster_model(
            args.pooled_cluster, adapter=adapter, fred_key=fred_key,
            lookback=args.lookback, save_dir=args.save_dir, train_recent=args.train_recent,
        )
    elif args.pooled_commodity:
        train_pooled_commodity_model(
            adapter=adapter, fred_key=fred_key, lookback=args.lookback,
            save_dir=args.save_dir, train_recent=args.train_recent,
        )

    for sym in symbols:
        train_swing_model(
            symbol=sym,
            adapter=adapter,
            fred_key=fred_key,
            lookback=args.lookback,
            train_end=args.train_end,
            train_recent=args.train_recent,
            save_dir=args.save_dir,
        )
        # Short-horizon (5-day) model
        if args.short_horizon:
            train_short_horizon_model(
                symbol=sym,
                adapter=adapter,
                fred_key=fred_key,
                lookback=args.lookback,
                save_dir=args.save_dir,
                train_end=args.train_end,
                train_recent=args.train_recent,
            )


if __name__ == "__main__":
    main()
