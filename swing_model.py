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
    compute_adx, compute_momentum_quality, RSI_PERIOD,
)
from ml_model import _fetch_vix_for_training, DEFAULT_MODEL_DIR
from cross_asset_signals import CrossAssetFeatureBuilder, get_cross_asset_features
from options_flow import OptionsFlowEngine, OPTIONS_FLOW_FEATURES
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
]

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
    max_depth=4,          # shallower than expansion (fewer relevant features)
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

# TFT hyperparams — intentionally small to avoid overfitting on N~800 samples
# Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon
# Time Series Forecasting", International Journal of Forecasting.
# Adopted by Two Sigma, Goldman Sachs Marquee, and Google Ads team internally.
_TFT_SEQ_LEN = 20    # 20-day lookback window
_TFT_HIDDEN  = 32    # small — N~800 samples, regularisation-first
_TFT_HEADS   = 4     # must divide _TFT_HIDDEN
_TFT_DROPOUT = 0.25  # high dropout combats small-dataset overfitting
_TFT_WEIGHT  = 0.60  # TFT contribution in ensemble (XGBoost = 1 - this)


# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------

def get_swing_feature_cols(symbol: str | None = None) -> list:
    """Return full feature list for a swing symbol.

    Composition:
        17 base + N supplement (EEM/EWT) + cross-asset (3) + options flow (4)
        + alpha + factor + market (from existing builders)
    """
    cols: list = list(SWING_BASE_FEATURES)

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
        self._options_engine: Optional[OptionsFlowEngine] = None
        self._alpha_builder: Optional[AlphaFeatureBuilder] = None
        self._factor_builder: Optional[FactorFeatureBuilder] = None
        self._market_builder: Optional[MarketSignalBuilder] = None
        self._supplement_cache: dict = {}

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

        # --- Supplement features (copper / SOX) ---
        if symbol and symbol in SWING_SUPPLEMENT_FEATURES:
            supp_df = self._fetch_supplement(bars_df, symbol)
            for col in supp_df.columns:
                df[col] = supp_df[col]

        # --- Cross-asset features ---
        if symbol:
            try:
                if self._cross_builder is None:
                    self._cross_builder = CrossAssetFeatureBuilder(
                        fred_key=fred_key or os.environ.get("FRED_API_KEY"))
                cross_df = self._cross_builder.build_features(bars_df, symbol)
                for col in cross_df.columns:
                    df[col] = cross_df[col]
            except Exception as exc:
                log.warning("Cross-asset features failed for %s: %s", symbol, exc)
                for col in get_cross_asset_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # --- Options flow features ---
        try:
            if self._options_engine is None:
                self._options_engine = OptionsFlowEngine()
            vts_df = self._options_engine.get_historical_vix_term_structure(
                lookback_days=len(bars_df))
            if not vts_df.empty:
                bar_dates_dt = pd.to_datetime(bars_df["ts"]).dt.date
                vts_map_ratio = dict(zip(vts_df["date"].values, vts_df["vix_term_ratio"].values))
                vts_map_inv   = dict(zip(vts_df["date"].values, vts_df["vix_term_inverted"].values))
                df["vix_term_ratio"]   = bar_dates_dt.map(
                    lambda d: vts_map_ratio.get(d, np.nan)).values
                df["vix_term_ratio"]   = pd.Series(
                    df["vix_term_ratio"].values, index=df.index).ffill()
                df["vix_term_inverted"] = bar_dates_dt.map(
                    lambda d: vts_map_inv.get(d, np.nan)).values
                df["vix_term_inverted"] = pd.Series(
                    df["vix_term_inverted"].values, index=df.index).ffill().fillna(0)
            else:
                df["vix_term_ratio"]   = np.nan
                df["vix_term_inverted"] = 0.0
            df["pc_volume_ratio"] = np.nan
            df["pc_oi_ratio"]     = np.nan
        except Exception as exc:
            log.warning("Options flow features failed: %s", exc)
            for col in OPTIONS_FLOW_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan

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
        all_cols = get_swing_feature_cols(symbol)
        df = df.dropna(subset=SWING_BASE_FEATURES)
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = df[col].ffill().fillna(0.0)

        return df[all_cols]

    def _fetch_supplement(self, bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Fetch proxy ETF data for EEM (copper) and EWT (SOX)."""
        import yfinance as yf

        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        supplement_cols = SWING_SUPPLEMENT_FEATURES.get(symbol, [])
        df = pd.DataFrame(index=bars_df.index)

        for col in supplement_cols:
            ticker = _SUPPLEMENT_TICKERS.get(col)
            if ticker is None:
                df[col] = 0.0
                continue
            try:
                if ticker not in self._supplement_cache:
                    data = yf.download(ticker, period="5y", progress=False,
                                       auto_adjust=True)
                    if data.empty:
                        self._supplement_cache[ticker] = pd.Series(dtype=float)
                    else:
                        close = data["Close"]
                        if hasattr(close, "squeeze"):
                            close = close.squeeze()
                        self._supplement_cache[ticker] = close

                proxy = self._supplement_cache[ticker]
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
) -> tuple:
    """Forward N-day return regression labels for XGBoost.

    Returns (X, y) as float32 numpy arrays.
    """
    full_close = bars_df["close"].astype(float).values
    bar_positions = bars_df.index.get_indexer(features_df.index)
    feature_values = features_df.values

    X_list, y_list = [], []
    for i in range(len(feature_values)):
        bar_pos = bar_positions[i]
        if bar_pos < 0 or bar_pos + forward_days >= len(full_close):
            continue
        entry_price = full_close[bar_pos]
        if entry_price <= 0:
            continue
        fwd_ret = (full_close[bar_pos + forward_days] - entry_price) / entry_price
        fwd_ret = max(-0.10, min(0.10, fwd_ret))  # winsorize at ±10%
        X_list.append(feature_values[i])
        y_list.append(fwd_ret)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


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
        return self.head(out).squeeze(-1)                # (B,)


def _prepare_sequence_regression(
    features_norm: pd.DataFrame,
    bars_df: pd.DataFrame,
    seq_len: int = _TFT_SEQ_LEN,
    forward_days: int = FORWARD_DAYS,
) -> tuple:
    """Sliding-window sequences for TFT training.

    For each valid bar i (i >= seq_len, i + forward_days < len(bars)):
        X[i] = features_norm[i-seq_len : i]   shape: (seq_len, n_features)
        y[i] = forward 10-day return at bar i, winsorized ±10%
    """
    full_close     = bars_df["close"].astype(float).values
    bar_positions  = bars_df.index.get_indexer(features_norm.index)
    feature_values = features_norm.values
    n = len(feature_values)

    X_list, y_list = [], []
    for i in range(seq_len, n):
        bar_pos = bar_positions[i]
        if bar_pos < 0 or bar_pos + forward_days >= len(full_close):
            continue
        entry_price = full_close[bar_pos]
        if entry_price <= 0:
            continue
        fwd_ret = (full_close[bar_pos + forward_days] - entry_price) / entry_price
        fwd_ret = max(-0.10, min(0.10, fwd_ret))
        X_list.append(feature_values[i - seq_len:i])
        y_list.append(fwd_ret)

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


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
    X_all, y_all = _prepare_sequence_regression(features_norm, bars_df, seq_len)
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
) -> Optional[xgb.XGBRegressor]:
    """Train XGBoost swing model (v3 regression).

    1. Fetch daily data (symbol + SPY for relative momentum)
    2. Build tabular features (17 base + macro supplement + cross-asset + ...)
    3. Forward 10-day return regression labels
    4. Train XGBRegressor with walk-forward split
    5. Save model + scaler + config
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info("=== Training swing XGBoost for %s ===", symbol)

    # 1. Fetch data
    log.info("Fetching %d daily bars for %s...", lookback, symbol)
    bars = adapter.fetch_daily(symbol, lookback)
    log.info("Got %d bars.", len(bars))

    log.info("Fetching SPY bars for relative momentum...")
    spy_bars = adapter.fetch_daily("SPY", lookback)
    log.info("Got %d SPY bars.", len(spy_bars))

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))
    log.info("Got %d VIX rows.", len(vix_df))

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

    # 4. Forward return labels
    X_all, y_all = _prepare_tabular_regression(full_norm, bars, forward_days=FORWARD_DAYS)
    log.info("Labeled %d samples. Mean fwd return: %+.4f%%",
             len(y_all), y_all.mean() * 100)

    # 5. Walk-forward split
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val,   y_val   = X_all[split:], y_all[split:]
    log.info("Train: %d, Val: %d.", len(X_train), len(X_val))

    if len(X_val) < 10:
        log.error("Validation set too small for %s. Need more data.", symbol)
        return None

    # 6. Train XGBRegressor
    model = xgb.XGBRegressor(**_XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

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

    # 8. Save
    model_path  = os.path.join(save_dir, f"{symbol}_xgb_swing.joblib")
    scaler_path = os.path.join(save_dir, f"{symbol}_xgb_swing_scaler.json")
    config_path = os.path.join(save_dir, f"{symbol}_xgb_swing_config.json")

    joblib.dump(model, model_path)
    engine.save_scaler(scaler_path)

    config = {
        "symbol":               symbol,
        "model_type":           "xgboost_swing",
        "model_version":        "v3_regression",
        "target":               f"forward_{FORWARD_DAYS}d_return",
        "val_rmse":             round(val_rmse, 8),
        "val_direction_accuracy": round(direction_acc, 4),
        "n_train":              len(X_train),
        "n_val":                len(X_val),
        "n_features":           n_features,
        "feature_names":        feature_cols,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved swing XGBoost → %s (dir_acc=%.3f)", model_path, direction_acc)

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
# Predictor (inference — compatible with ml_model.Predictor interface)
# ---------------------------------------------------------------------------

class SwingPredictor:
    """TFT + XGBoost ensemble swing predictor (v4).

    Compatible with ml_model.Predictor.predict() interface.

    Blends 60% TFT (temporal self-attention over 20-day sequences) +
    40% XGBoost (point-in-time tabular). Falls back to XGBoost-only
    if no TFT model file is found.

    Architecture reference: Lim et al. (Google, 2021) — IJF.
    """

    model_type = "tft_xgb_swing"
    eod_exit   = False   # swing trades hold overnight

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol    = symbol
        self.model_dir = model_dir
        self.engine    = SwingFeatureEngine()
        self.model:  Optional[xgb.XGBRegressor] = None
        self._tft:   Optional[TFTSwingModel]     = None
        self._tft_seq_len    = _TFT_SEQ_LEN
        self._tft_n_features = 0
        self._spy_adapter    = None   # lazy-loaded for relative momentum
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

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = 60) -> dict:
        """Produce regression prediction (v3).

        Returns standard prediction dict with expected_return field.
        seq_len is accepted for interface compat but ignored by XGBoost.
        """
        from ml_model import COST_THRESHOLD, TARGET_RETURN

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
        x = features_norm.iloc[-1:].values.astype(np.float32)
        xgb_ret = float(self.model.predict(x)[0])

        # TFT ensemble blend: 60% TFT + 40% XGBoost
        expected_return = xgb_ret
        if self._tft is not None:
            try:
                seq = self._tft_seq_len
                n   = self._tft_n_features
                if len(features_norm) >= seq:
                    window = features_norm.iloc[-seq:].values[:, :n].astype(np.float32)
                    xt = torch.FloatTensor(window).unsqueeze(0)   # (1, seq, n_feat)
                    with torch.no_grad():
                        tft_ret = self._tft(xt).item()
                    expected_return = _TFT_WEIGHT * tft_ret + (1 - _TFT_WEIGHT) * xgb_ret
                    log.debug("[TFT] %s: xgb=%.4f tft=%.4f blend=%.4f",
                              self.symbol, xgb_ret, tft_ret, expected_return)
            except Exception as exc:
                log.warning("[TFT] predict failed for %s: %s — XGB only.", self.symbol, exc)
                expected_return = xgb_ret

        if expected_return > COST_THRESHOLD:
            direction = "UP"
        elif expected_return < -COST_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        confidence  = min(1.0, abs(expected_return) / TARGET_RETURN)
        probability = max(0.05, min(0.95, 0.5 + expected_return * 10))

        return {
            "expected_return": round(expected_return, 6),
            "direction":       direction,
            "probability":     round(probability, 4),
            "confidence":      round(confidence, 4),
            "meta_confidence": 1.0,
            "tradeable":       abs(expected_return) > COST_THRESHOLD,
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
    args = parser.parse_args()

    adapter  = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols  = [s.strip().upper() for s in args.symbols.split(",")]

    for sym in symbols:
        train_swing_model(
            symbol=sym,
            adapter=adapter,
            fred_key=fred_key,
            lookback=args.lookback,
        )


if __name__ == "__main__":
    main()
