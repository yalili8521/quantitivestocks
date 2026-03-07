#!/usr/bin/env python3
"""
Swing PatchTST Model
====================================
PatchTST (Patch Time Series Transformer) for swing-trading daily ETFs.

Based on Nie et al. (2023): PatchTST achieves Sharpe 2.31 in 2025 large-scale
financial forecasting benchmark (arxiv.org/abs/2603.01820), vs standard LSTM ~0.76.

Architecture:
    1. Split 60-day input sequence into 12 patches of 5 bars each (weekly patches)
    2. Linear embed each patch → d_model=64 dimensions
    3. Add learnable positional encoding + CLS token
    4. Process through 3-layer Transformer encoder (4 heads)
    5. Classify from CLS token → sigmoid probability

Extended features (17) vs base LSTM (12):
    + ret252 (TSMOM 12-month signal)
    + ibs (Internal Bar Strength)
    + overnight_gap
    + vix_pctrank (VIX percentile rank over 252 days)
    + obv_trend (OBV moving average slope)

Usage (via main.py):
    python main.py train-swing  --symbols EWT,GLD,EEM,SLV --provider yahoo
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from signals_engine import (
    PROJECT_ROOT, build_adapter,
    compute_rsi, compute_atr, compute_macd, compute_bollinger_bands,
    compute_adx, compute_momentum_quality, RSI_PERIOD,
)
from ml_model import (
    _fetch_vix_for_training, DEFAULT_MODEL_DIR,
    frac_diff_ffd, FeatureEngine,
)
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
SWING_SEQ_LEN = 60           # 60-day lookback (3 months)
PATCH_SIZE = 5               # 5-bar patches (weekly)
D_MODEL = 64                 # transformer embedding dimension
N_HEADS = 4                  # attention heads
N_LAYERS = 3                 # transformer layers
DROPOUT = 0.15               # dropout rate

# Base features from existing LSTM daily set
BASE_FEATURES = [
    "bb_bandwidth", "vol20", "ret5", "bb_pct_b", "wk_ret", "dv_accel",
    "rsi14", "ret10", "adx", "macd_hist_norm", "vwap_ratio", "vol_regime",
]

# New research-backed features
NEW_FEATURES = [
    "ret252",          # TSMOM: 12-month trailing return (Moskowitz et al. 2012)
    "ibs",             # Internal Bar Strength (Pagonidis 2014)
    "overnight_gap",   # (open - prev_close) / prev_close
    "vix_pctrank",     # VIX percentile rank over 252 days
    "obv_trend",       # OBV 20-day SMA slope (normalized by price)
]

SWING_FEATURE_COLS = BASE_FEATURES + NEW_FEATURES  # 17 total

# Symbol-specific: GLD/SLV add frac_diff_close (18 features)
SWING_FFD_SYMBOLS = {"GLD", "SLV", "TLT"}


def get_swing_feature_cols(symbol: str | None = None) -> list:
    """Return swing feature list, adding FFD, cross-asset, and options flow features."""
    cols = list(SWING_FEATURE_COLS)
    if symbol and symbol in SWING_FFD_SYMBOLS:
        cols.append("frac_diff_close")
    # Add cross-asset features (per-symbol)
    if symbol:
        cols.extend(get_cross_asset_features(symbol))
    else:
        cols.append("treasury_slope")  # default fallback
    # Add options flow features (market-wide)
    cols.extend(OPTIONS_FLOW_FEATURES)
    # Add alpha signal features (CBOE + insider)
    if symbol:
        cols.extend(get_alpha_features(symbol))
    # Add factor signal features (credit, risk appetite, yield curve, FF, etc.)
    if symbol:
        cols.extend(get_factor_features(symbol))
    # Add market signal features (volume, VRP, calendar, breadth)
    if symbol:
        cols.extend(get_market_features(symbol))
    return cols


# ---------------------------------------------------------------------------
# PatchTST Model
# ---------------------------------------------------------------------------
class PatchTST(nn.Module):
    """Patch Time Series Transformer for forward return regression (v2).

    Architecture from Nie et al. (2023), adapted for financial
    return prediction. Uses CLS token for regression output.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = SWING_SEQ_LEN,
        patch_size: int = PATCH_SIZE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        patch_dim = n_features * patch_size

        # Input normalization
        self.input_norm = nn.LayerNorm(n_features)

        # Patch embedding: flatten each patch → project to d_model
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Learnable positional encoding + CLS token
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Regression head (v2 — no sigmoid, raw return output)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            # No activation — raw return output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) — raw (normalized) daily bars

        Returns:
            (batch,) — expected forward return (e.g., +0.015 = +1.5%)
        """
        B, L, F = x.shape

        # Normalize input features
        x = self.input_norm(x)

        # Reshape into patches: (B, n_patches, patch_size * n_features)
        n_patches = L // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]
        x = x.reshape(B, n_patches, self.patch_size * F)

        # Embed patches
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, d_model)

        # Add positional encoding
        x = x + self.pos_embed[:, :n_patches + 1, :]

        # Transformer
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Regress from CLS token
        return self.regressor(x[:, 0]).squeeze(-1)


# ---------------------------------------------------------------------------
# Extended Feature Engineering
# ---------------------------------------------------------------------------
class SwingFeatureEngine(FeatureEngine):
    """Extension of FeatureEngine with additional research-backed features."""

    def __init__(self):
        super().__init__()
        self._cross_builder: Optional[CrossAssetFeatureBuilder] = None
        self._options_engine: Optional[OptionsFlowEngine] = None
        self._alpha_builder: Optional[AlphaFeatureBuilder] = None
        self._factor_builder: Optional[FactorFeatureBuilder] = None
        self._market_builder: Optional[MarketSignalBuilder] = None

    def build_features(self, bars_df: pd.DataFrame,
                       vix_df: pd.DataFrame,
                       mode: str = "daily",
                       symbol: str | None = None,
                       fred_key: str | None = None) -> pd.DataFrame:
        """Build extended feature matrix with 17+ base + cross-asset + options flow features.

        Adds to base LSTM features:
            ret252:        TSMOM 12-month trailing return
            ibs:           Internal Bar Strength
            overnight_gap: Open vs prev close
            vix_pctrank:   VIX percentile rank (252 days)
            obv_trend:     OBV 20-day slope
        Plus cross-asset and options flow features.
        """
        close = bars_df["close"].astype(float)
        high = bars_df["high"].astype(float)
        low = bars_df["low"].astype(float)
        volume = bars_df["volume"].astype(float)
        open_ = bars_df["open"].astype(float)
        df = pd.DataFrame(index=bars_df.index)

        annualize = np.sqrt(252)  # swing = daily only

        # === Existing base features (same calculation as FeatureEngine) ===
        df["rsi14"] = compute_rsi(close, RSI_PERIOD) / 100.0
        df["ret5"] = close.pct_change(5)
        df["ret10"] = close.pct_change(10)
        df["wk_ret"] = close.pct_change(5)
        df["mo_ret"] = close.pct_change(21)
        df["vol20"] = close.pct_change().rolling(20).std() * annualize

        dv = close * volume
        df["log_dollar_vol"] = np.log10(dv.replace(0, np.nan))

        # VIX
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
        df["vix_chg"] = df["vix"].pct_change(fill_method=None)

        # Order-flow
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

        # Enhanced features
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

        ema_fast = close.ewm(span=10, adjust=False).mean()
        ema_slow = close.ewm(span=30, adjust=False).mean()
        df["trend_strength"] = (ema_fast - ema_slow) / close.replace(0, np.nan)

        # FFD for precious metals
        log_close = np.log(close.replace(0, np.nan))
        df["frac_diff_close"] = frac_diff_ffd(log_close, d=0.4, thres=1e-4)

        # === NEW research-backed features ===

        # 1. TSMOM: 252-day trailing return (Moskowitz, Ooi, Pedersen 2012)
        df["ret252"] = close.pct_change(252)

        # 2. IBS: Internal Bar Strength (Pagonidis 2014)
        df["ibs"] = (close - low) / (high - low).replace(0, np.nan)

        # 3. Overnight gap: (today open - yesterday close) / yesterday close
        prev_close = close.shift(1)
        df["overnight_gap"] = (open_ - prev_close) / prev_close.replace(0, np.nan)

        # 4. VIX percentile rank over 252 days
        df["vix_pctrank"] = df["vix"].rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # 5. OBV trend: OBV 20-day SMA slope normalized by price
        obv_direction = np.sign(close.diff()).fillna(0)
        obv = (obv_direction * volume).cumsum()
        obv_sma20 = obv.rolling(20).mean()
        obv_slope = obv_sma20.diff(5) / 5  # 5-day slope
        df["obv_trend"] = obv_slope / close.replace(0, np.nan) / volume.rolling(20).mean().replace(0, np.nan)

        # === Cross-asset features (per-symbol macro/cross-market signals) ===
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

        # === Options flow features (market-wide sentiment) ===
        try:
            if self._options_engine is None:
                self._options_engine = OptionsFlowEngine()
            vts_df = self._options_engine.get_historical_vix_term_structure(
                lookback_days=len(bars_df))
            if not vts_df.empty:
                bar_dates_dt = pd.to_datetime(bars_df["ts"]).dt.date
                vts_map_ratio = dict(zip(vts_df["date"].values, vts_df["vix_term_ratio"].values))
                vts_map_inv = dict(zip(vts_df["date"].values, vts_df["vix_term_inverted"].values))
                df["vix_term_ratio"] = bar_dates_dt.map(
                    lambda d: vts_map_ratio.get(d, np.nan)).values
                df["vix_term_ratio"] = pd.Series(
                    df["vix_term_ratio"].values, index=df.index).ffill()
                df["vix_term_inverted"] = bar_dates_dt.map(
                    lambda d: vts_map_inv.get(d, np.nan)).values
                df["vix_term_inverted"] = pd.Series(
                    df["vix_term_inverted"].values, index=df.index).ffill().fillna(0)
            else:
                df["vix_term_ratio"] = np.nan
                df["vix_term_inverted"] = 0.0
            # P/C ratios not available historically — fill with NaN (handled by model)
            df["pc_volume_ratio"] = np.nan
            df["pc_oi_ratio"] = np.nan
        except Exception as exc:
            log.warning("Options flow features failed: %s", exc)
            for col in OPTIONS_FLOW_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan

        # === Alpha signal features (CBOE historical + insider) ===
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

        # === Factor signal features (credit, risk appetite, yield curve, FF, etc.) ===
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

        # === Market signal features (volume, VRP, calendar, breadth) ===
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

        # Select relevant columns and drop NaN warmup
        # Only require base features to be non-NaN; cross-asset/options flow can be NaN
        relevant_cols = get_swing_feature_cols(symbol)
        base_required = [c for c in SWING_FEATURE_COLS if c in df.columns]
        df = df.dropna(subset=base_required)
        # Fill remaining NaN in optional features
        for col in relevant_cols:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = df[col].ffill().fillna(0.0)
        return df[relevant_cols]


# ---------------------------------------------------------------------------
# Training data preparation (triple-barrier, same as ml_model)
# ---------------------------------------------------------------------------
def _prepare_sequences_triple_barrier(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    seq_len: int = SWING_SEQ_LEN,
    pt_pct: float = 0.015,
    sl_pct: float = 0.010,
    horizon: int = 5,
) -> tuple:
    """Triple-barrier labeling — identical to ml_model version."""
    feature_values = features_df.values
    full_close = bars_df["close"].astype(float).values
    bar_positions = bars_df.index.get_indexer(features_df.index)
    n = len(feature_values)

    X_list = []
    y_list = []

    for i in range(n - seq_len):
        entry_feat_pos = i + seq_len - 1
        entry_bar_pos = bar_positions[entry_feat_pos]
        X_list.append(feature_values[i: i + seq_len])

        if entry_bar_pos < 0 or entry_bar_pos + horizon >= len(full_close):
            if 0 <= entry_bar_pos + 1 < len(full_close):
                curr = full_close[entry_bar_pos]
                nxt = full_close[entry_bar_pos + 1]
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
                label = 1.0
                break
            elif ret <= -sl_pct:
                label = 0.0
                break
        if label is None:
            final_pos = min(entry_bar_pos + horizon, len(full_close) - 1)
            final_ret = (full_close[final_pos] - entry_price) / entry_price
            label = 1.0 if final_ret > 0 else 0.0

        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _prepare_sequences_regression(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    seq_len: int = SWING_SEQ_LEN,
    forward_days: int = 10,
) -> tuple:
    """Forward N-day return regression labels for PatchTST (v2)."""
    feature_values = features_df.values
    full_close = bars_df["close"].astype(float).values
    bar_positions = bars_df.index.get_indexer(features_df.index)
    n = len(feature_values)

    X_list, y_list = [], []
    for i in range(n - seq_len):
        entry_feat_pos = i + seq_len - 1
        entry_bar_pos = bar_positions[entry_feat_pos]

        if entry_bar_pos < 0 or entry_bar_pos + forward_days >= len(full_close):
            continue

        entry_price = full_close[entry_bar_pos]
        future_price = full_close[entry_bar_pos + forward_days]
        if entry_price <= 0:
            continue

        fwd_ret = (future_price - entry_price) / entry_price
        fwd_ret = max(-0.10, min(0.10, fwd_ret))

        X_list.append(feature_values[i: i + seq_len])
        y_list.append(fwd_ret)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_swing_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    epochs: int = 60,
    lr: float = 5e-4,
    batch_size: int = 32,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
) -> Optional[PatchTST]:
    """Train PatchTST model for swing trading (v2 regression).

    1. Fetch daily data (1000+ bars)
    2. Build extended features (17-18 for FFD symbols)
    3. Forward 10-day return regression labels
    4. Train PatchTST with MSE loss + early stopping
    5. Save model + scaler
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info("=== Training swing PatchTST for %s ===", symbol)

    # 1. Fetch data
    log.info("Fetching %d daily bars for %s...", lookback, symbol)
    bars = adapter.fetch_daily(symbol, lookback)
    log.info("Got %d bars.", len(bars))

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))
    log.info("Got %d VIX rows.", len(vix_df))

    # 2. Build features
    engine = SwingFeatureEngine()
    features = engine.build_features(bars, vix_df, mode="daily", symbol=symbol)
    log.info("Built %d feature rows.", len(features))

    n_features = len(get_swing_feature_cols(symbol))
    if len(features) < SWING_SEQ_LEN + 20:
        log.error("Not enough data for %s: %d rows (need at least %d).",
                  symbol, len(features), SWING_SEQ_LEN + 20)
        return None

    # 3. Scaler
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    full_norm = engine.transform(features)

    # 4. Forward 10-day return regression labels (v2)
    FORWARD_DAYS = 10
    X_all, y_all = _prepare_sequences_regression(
        full_norm, bars, SWING_SEQ_LEN, forward_days=FORWARD_DAYS,
    )

    # 5. Walk-forward split with purge + embargo
    seq_split = int(len(X_all) * 0.8)
    embargo = SWING_SEQ_LEN
    X_train, y_train = X_all[:seq_split], y_all[:seq_split]
    X_val, y_val = X_all[seq_split + embargo:], y_all[seq_split + embargo:]

    log.info("Train: %d, Val: %d (embargo=%d). Return — train mean: %+.4f%%, val mean: %+.4f%%",
             len(X_train), len(X_val), embargo,
             y_train.mean() * 100, y_val.mean() * 100 if len(y_val) > 0 else 0)

    if len(X_val) < 5:
        log.error("Validation set too small for %s. Need more data.", symbol)
        return None

    # 6. Create PatchTST model
    model = PatchTST(
        n_features=n_features,
        seq_len=SWING_SEQ_LEN,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )

    LABEL_SMOOTH = 0.05  # unused in v2 — kept for reference
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01,
    )

    # 7. Training with early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 15
    GRAD_CLIP = 1.0

    for epoch in range(epochs):
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info("Epoch %2d/%d  train=%.4f  val=%.4f  acc=%.3f  lr=%.2e",
                     epoch + 1, epochs, train_loss, val_loss, val_acc, current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{symbol}_patchtst_swing.pt"))
            engine.save_scaler(
                os.path.join(save_dir, f"{symbol}_patchtst_swing_scaler.json"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d.", epoch + 1)
                break

    # Save metrics
    metrics = {
        "symbol": symbol,
        "model_type": "patchtst_swing",
        "model_version": "v2_regression",
        "target": f"forward_{FORWARD_DAYS}d_return",
        "best_val_loss_mse": round(best_val_loss, 8),
        "best_direction_accuracy": round(best_val_acc, 4),
        "n_features": n_features,
        "seq_len": SWING_SEQ_LEN,
        "patch_size": PATCH_SIZE,
        "feature_cols": get_swing_feature_cols(symbol),
    }
    with open(os.path.join(save_dir, f"{symbol}_patchtst_swing_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Training complete for %s. Best val_MSE=%.6f  dir_acc=%.3f",
             symbol, best_val_loss, best_val_acc)
    return model


# ---------------------------------------------------------------------------
# Predictor (inference — compatible with ml_model.Predictor interface)
# ---------------------------------------------------------------------------
class SwingPredictor:
    """PatchTST predictor for swing trading (v2 regression).

    Compatible with ml_model.Predictor.predict() interface.
    """

    model_type = "patchtst_swing"
    eod_exit = False  # swing trades hold overnight

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol = symbol
        self.model_dir = model_dir
        self.engine = SwingFeatureEngine()
        self.model: Optional[PatchTST] = None
        self._load()

    def _load(self) -> None:
        weights_path = os.path.join(self.model_dir, f"{self.symbol}_patchtst_swing.pt")
        scaler_path = os.path.join(self.model_dir, f"{self.symbol}_patchtst_swing_scaler.json")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No trained swing PatchTST for {self.symbol}. "
                f"Run: python main.py train-swing --symbols {self.symbol}"
            )

        self.engine.load_scaler(scaler_path)
        n_features = len(get_swing_feature_cols(self.symbol))
        self.model = PatchTST(
            n_features=n_features,
            seq_len=SWING_SEQ_LEN,
            patch_size=PATCH_SIZE,
        )
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()
        log.info("Loaded swing PatchTST for %s (%d features).", self.symbol, n_features)

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = SWING_SEQ_LEN) -> dict:
        """Produce regression prediction (v2).

        Returns standard prediction dict with expected_return field.
        """
        from ml_model import COST_THRESHOLD, TARGET_RETURN

        features = self.engine.build_features(bars_df, vix_df, mode="daily", symbol=self.symbol)
        features_norm = self.engine.transform(features)

        if len(features_norm) < SWING_SEQ_LEN:
            return {
                "expected_return": 0.0,
                "direction": "FLAT",
                "probability": 0.5,
                "confidence": 0.0,
                "meta_confidence": 1.0,
                "tradeable": False,
            }

        window = features_norm.iloc[-SWING_SEQ_LEN:].values
        x = torch.FloatTensor(window).unsqueeze(0)

        with torch.no_grad():
            expected_return = self.model(x).item()

        if expected_return > COST_THRESHOLD:
            direction = "UP"
        elif expected_return < -COST_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        confidence = min(1.0, abs(expected_return) / TARGET_RETURN)
        probability = max(0.05, min(0.95, 0.5 + expected_return * 10))

        return {
            "expected_return": round(expected_return, 6),
            "direction": direction,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "meta_confidence": 1.0,
            "tradeable": abs(expected_return) > COST_THRESHOLD,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train PatchTST model for swing trading.",
    )
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols (e.g. EWT,GLD,EEM,SLV)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=1000,
                        help="Daily bars to fetch (default: 1000)")
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    for sym in symbols:
        train_swing_model(
            symbol=sym,
            adapter=adapter,
            fred_key=fred_key,
            epochs=args.epochs,
            lr=args.lr,
            lookback=args.lookback,
        )


if __name__ == "__main__":
    main()
