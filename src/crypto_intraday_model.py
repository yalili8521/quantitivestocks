#!/usr/bin/env python3
"""
Crypto Intraday Model (LightGBM 70% + GRU 30% Ensemble)
=========================================================
Independent intraday trading model for crypto, operating on 5-minute bars.

Architecture (based on research):
    - LightGBM: handles tabular features, strong on microstructure/volume signals
    - GRU: captures temporal dependencies, especially BTC lead-lag and momentum regime shifts
    - Ensemble: 0.70 * LGB + 0.30 * GRU (GRU discarded if val dir_acc < 50%)

Strategy:
    1. Every 5 minutes, compute features from last N bars
    2. Ensemble predicts expected return over next 12 bars (1 hour)
    3. If abs(E[r]) > cost_threshold: enter LONG/SHORT
    4. Exit via trailing stop, take-profit, or max hold time (4 hours)

Features (26 total):
    Momentum (5):  ret_6bar, ret_12bar, ret_24bar, ret_48bar, momentum_acceleration
    Volume (5):    rvol, volume_trend, cumulative_delta, dollar_volume_zscore, volume_imbalance_12
    Volatility (4): atr_pct, bb_pct_b, realized_vol_24bar, high_low_range
    Microstructure (3): close_position, bar_body_ratio, vwap_deviation
    Cross-market (4): btc_ret_12bar, btc_ret_12bar_lag1, btc_correlation, btc_vol_ratio
    Time (4):      hour_sin, hour_cos, dow_sin, dow_cos
    Regime (1):    mean_reversion_signal

Label: forward 12-bar (1 hour) return, winsorized at 1st/99th percentile.

Usage (via main.py):
    python main.py train-crypto-intraday --symbols BTC-USD,ETH-USD,SOL-USD
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

log = logging.getLogger("crypto_intraday_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORWARD_BARS = 12           # predict 1 hour ahead (12 x 5min)
MAX_HOLD_BARS = 48          # max 4 hours (48 x 5min)
LOOKBACK_BARS = 60          # need 5 hours of history for features (increased for lag features)
COST_THRESHOLD = 0.003      # 30bps min signal (crypto costs ~20-40bps RT)
TARGET_RETURN = 0.01        # 1% for full position sizing

# GRU sequence length: 12 bars (1 hour of context)
GRU_SEQ_LEN = 12
GRU_HIDDEN = 32
GRU_LAYERS = 1
GRU_DROPOUT = 0.1

# Ensemble weights
W_LGB = 0.70
W_GRU = 0.30

FEATURE_NAMES = [
    # Short momentum (5)
    "ret_6bar",              # 30-min return
    "ret_12bar",             # 1-hour return (matches forward target)
    "ret_24bar",             # 2-hour return
    "ret_48bar",             # 4-hour return (top-2 importance in BTC)
    "momentum_acceleration", # ret_6bar - ret_6bar.shift(6): is momentum increasing?
    # Volatility & range (3)
    "atr_pct",               # ATR(12) / close — intraday vol regime
    "realized_vol_24bar",    # realized vol over 24 bars (top-3 importance)
    "bb_pct_b",              # Bollinger %B — mean-reversion signal
    # Microstructure & liquidity (6)
    "close_position",        # (close - low) / (high - low) over 12 bars (#1 importance: 28%)
    "vwap_deviation",        # (close - vwap_12) / close: institutional reversion anchor
    "cumulative_delta",      # buy-sell volume imbalance over 12 bars
    "rvol",                  # relative volume vs 48-bar avg
    "dollar_volume_zscore",  # (current $ vol - mean) / std over 48 bars
    "volume_trend",          # linear slope of volume over 12 bars
    # BTC cross-market (2, altcoins only — removed for BTC-USD)
    "btc_ret_12bar",         # BTC 1-hour return (leader signal)
    "btc_ret_12bar_lag1",    # BTC 1-hour return, lagged 1 bar (5-15min lead-lag)
    # Time regime (2)
    "hour_sin",              # sin(2pi * hour / 24) — session transitions (#4-5 importance)
    "hour_cos",              # cos(2pi * hour / 24)
]
# v0→v1 dropped: volume_imbalance_12 (redundant w/ cumulative_delta),
# high_low_range (0.03% imp), bar_body_ratio (0.06% imp),
# mean_reversion_signal (linear transform of bb_pct_b),
# btc_correlation (0% imp), btc_vol_ratio (low intraday value),
# dow_sin/dow_cos (weak vs hour encoding for 24/7 crypto)

N_FEATURES = len(FEATURE_NAMES)  # 18

# BTC cross-market features (removed for BTC-USD: self-referential)
_BTC_CROSS_FEATURES = {"btc_ret_12bar", "btc_ret_12bar_lag1"}


def get_intraday_feature_cols(symbol: str | None = None) -> list:
    """Return the feature columns for a given symbol.

    BTC-USD drops self-referential btc_ret features (16 features).
    All other coins use the full 18-feature set.
    """
    if symbol and symbol.upper().startswith("BTC"):
        return [f for f in FEATURE_NAMES if f not in _BTC_CROSS_FEATURES]
    return list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
class CryptoIntradayFeatureEngine:
    """Build feature vectors from 5-minute crypto OHLCV bars."""

    def build_features(
        self,
        bars: pd.DataFrame,
        btc_bars: Optional[pd.DataFrame] = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Build feature matrix from 5-min bars.

        Args:
            bars: OHLCV DataFrame with columns [ts, open, high, low, close, volume]
            btc_bars: BTC OHLCV for cross-market features (None if symbol IS BTC)
            symbol: Symbol name for per-symbol feature routing (e.g., 'BTC-USD')

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

        # --- Momentum (5) ---
        df["ret_6bar"] = c.pct_change(6)
        df["ret_12bar"] = c.pct_change(12)
        df["ret_24bar"] = c.pct_change(24)
        df["ret_48bar"] = c.pct_change(48)
        # Momentum acceleration: is momentum speeding up or slowing down?
        ret6 = c.pct_change(6)
        df["momentum_acceleration"] = ret6 - ret6.shift(6)

        # --- Volume (5) ---
        avg_vol_48 = v.rolling(48).mean()
        df["rvol"] = v / avg_vol_48.clip(lower=1)

        # Volume trend: linear slope over 12 bars (normalized)
        def _vol_trend(series, window=12):
            result = pd.Series(np.nan, index=series.index)
            x = np.arange(window, dtype=float)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            for i in range(window - 1, len(series)):
                y = series.iloc[i - window + 1:i + 1].values.astype(float)
                if np.any(np.isnan(y)):
                    continue
                y_mean = y.mean()
                if y_mean <= 0:
                    continue
                slope = ((x - x_mean) * (y - y_mean)).sum() / max(x_var, 1e-10)
                result.iloc[i] = slope / y_mean
            return result

        df["volume_trend"] = _vol_trend(v)

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

        # Dollar volume z-score
        dollar_vol = c * v
        dv_mean = dollar_vol.rolling(48).mean()
        dv_std = dollar_vol.rolling(48).std()
        df["dollar_volume_zscore"] = (dollar_vol - dv_mean) / dv_std.clip(lower=1e-10)

        # Volume imbalance: net buy-sell ratio over 12 bars (enhanced version)
        buy_vol_12 = buy_vol.rolling(12).sum()
        sell_vol_12 = sell_vol.rolling(12).sum()
        total_12 = (buy_vol_12 + sell_vol_12).clip(lower=1)
        df["volume_imbalance_12"] = (buy_vol_12 - sell_vol_12) / total_12

        # --- Volatility (4) ---
        # ATR(12)
        tr = pd.concat([
            h - lo,
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_12 = tr.rolling(12).mean()
        df["atr_pct"] = atr_12 / c.clip(lower=1e-10)

        # Bollinger %B (20, 2)
        sma_20 = c.rolling(20).mean()
        std_20 = c.rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        bb_range = (upper - lower).clip(lower=1e-10)
        df["bb_pct_b"] = (c - lower) / bb_range

        # Realized vol (24 bars, annualized)
        log_ret = np.log(c / c.shift(1))
        df["realized_vol_24bar"] = log_ret.rolling(24).std() * np.sqrt(288 * 365)

        # High-low range of last bar
        df["high_low_range"] = (h - lo) / c.clip(lower=1e-10)

        # --- Microstructure (3) ---
        # Close position in range over last 12 bars
        rolling_high_12 = h.rolling(12).max()
        rolling_low_12 = lo.rolling(12).min()
        range_12 = (rolling_high_12 - rolling_low_12).clip(lower=1e-10)
        df["close_position"] = (c - rolling_low_12) / range_12

        # Bar body ratio: abs(close-open) / (high-low) averaged over 6 bars
        body = (c - o).abs()
        bar_range = (h - lo).clip(lower=1e-10)
        body_ratio = body / bar_range
        df["bar_body_ratio"] = body_ratio.rolling(6).mean()

        # VWAP deviation: (close - VWAP_12) / close
        typical_price = (h + lo + c) / 3
        vwap_12 = (typical_price * v).rolling(12).sum() / v.rolling(12).sum().clip(lower=1)
        df["vwap_deviation"] = (c - vwap_12) / c.clip(lower=1e-10)

        # --- Cross-market: BTC leadership (2, altcoins only) ---
        if btc_bars is not None and not btc_bars.empty:
            btc = btc_bars.copy().sort_values("ts").reset_index(drop=True)

            # Align BTC bars to same timestamps
            btc_aligned = btc.set_index("ts")["close"].reindex(
                df["ts"], method="ffill"
            ).reset_index(drop=True).astype(float)

            btc_ret_12 = btc_aligned.pct_change(12)
            df["btc_ret_12bar"] = btc_ret_12
            df["btc_ret_12bar_lag1"] = btc_ret_12.shift(1)  # 1-bar lagged BTC return
        else:
            # Symbol IS BTC or no BTC data — set to own returns (will be
            # excluded from feature list for BTC via get_intraday_feature_cols)
            df["btc_ret_12bar"] = df["ret_12bar"]
            df["btc_ret_12bar_lag1"] = df["ret_12bar"].shift(1)

        # --- Time features (cyclical encoding) (2) ---
        ts = pd.to_datetime(df["ts"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        hour = ts.dt.hour + ts.dt.minute / 60.0

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Select feature columns (BTC-USD drops btc_ret features)
        use_cols = get_intraday_feature_cols(symbol)
        result = df[["ts"] + use_cols].copy()

        # Drop rows with NaN in any feature
        valid_mask = result[use_cols].notna().all(axis=1)
        result = result[valid_mask].reset_index(drop=True)

        return result

    def build_training_data(
        self,
        bars: pd.DataFrame,
        btc_bars: Optional[pd.DataFrame] = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Build training data with features and labels.

        Label: forward FORWARD_BARS (12) bar return, winsorized.
        """
        features = self.build_features(bars, btc_bars, symbol=symbol)
        if features.empty:
            return pd.DataFrame()

        # Align bars to feature timestamps
        bars_indexed = bars.set_index("ts")
        close_aligned = bars_indexed["close"].reindex(
            features["ts"], method="ffill"
        ).astype(float)

        # Forward return
        fwd_ret = close_aligned.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
        features["fwd_return"] = fwd_ret.values

        # Drop rows without labels
        features = features.dropna(subset=["fwd_return"]).reset_index(drop=True)

        # NOTE: winsorization moved to train_model() to avoid leaking
        # validation data quantile bounds into training labels.

        return features


# ---------------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available — GRU ensemble disabled, LightGBM only")


if TORCH_AVAILABLE:
    class GRUReturnModel(nn.Module):
        """Lightweight GRU for temporal pattern capture in 5-min bars."""

        def __init__(self, n_features: int, hidden: int = GRU_HIDDEN,
                     n_layers: int = GRU_LAYERS, dropout: float = GRU_DROPOUT):
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, n_features)
            out, _ = self.gru(x)
            # Use last hidden state
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
class CryptoIntradayTrainer:
    """Train LightGBM + GRU ensemble for crypto intraday prediction."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _train_lgb(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        feature_names: List[str],
    ) -> Tuple:
        """Train LightGBM regression model."""
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 15,
            "learning_rate": 0.01,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_data_in_leaf": 200,
            "verbose": -1,
            "seed": 42,
            "lambda_l1": 1.0,
            "lambda_l2": 5.0,
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
        return model, val_pred

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
        std[std < 1e-8] = 1.0  # avoid div-by-zero

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

        # Convert to tensors
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
        log.info("GRU val dir_acc=%.3f (threshold=0.50)", gru_dir_acc)

        if gru_dir_acc < 0.50:
            log.warning("GRU dir_acc < 0.50, discarding GRU (LGB-only fallback)")
            return None, mean, std, None

        return model, mean, std, val_pred_np

    def train(
        self,
        symbol: str,
        training_data: pd.DataFrame,
        val_frac: float = 0.2,
        walk_forward: bool = False,
    ) -> dict:
        """Train LightGBM + GRU ensemble.

        Args:
            walk_forward: If True, use first 75% for training, last 25% for OOS.
                          The model only sees historical data, mimicking production.
        Returns dict with training metrics.
        """
        if len(training_data) < 500:
            log.warning("%s: only %d samples, need >= 500", symbol, len(training_data))
            return {"error": f"too few samples: {len(training_data)}"}

        feature_cols = get_intraday_feature_cols(symbol)
        n_feat = len(feature_cols)

        if walk_forward:
            # Walk-forward: train on first 75%, OOS on last 25%
            wf_split = int(len(training_data) * 0.75)
            training_data = training_data.iloc[:wf_split].copy()
            log.info("%s: walk-forward mode — training on first %d bars (75%%)", symbol, len(training_data))

        X = training_data[feature_cols].values.astype(np.float32)
        y = training_data["fwd_return"].values.astype(np.float32)

        # Time-based split (within training window)
        split_idx = int(len(X) * (1 - val_frac))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Winsorize labels at 1st/99th percentile — bounds from TRAIN only
        # to avoid leaking validation data distribution into training.
        if len(y_train) > 100:
            lo_q = float(np.quantile(y_train, 0.01))
            hi_q = float(np.quantile(y_train, 0.99))
            y_train = np.clip(y_train, lo_q, hi_q)
            y_val = np.clip(y_val, lo_q, hi_q)  # same bounds, not val-derived
            log.info("Label winsorization (train-derived): [%.4f, %.4f]", lo_q, hi_q)

        log.info("%s: Train %d, Val %d, %d features",
                 symbol, len(X_train), len(X_val), n_feat)

        # --- Train LightGBM ---
        log.info("%s: Training LightGBM...", symbol)
        lgb_model, lgb_val_pred = self._train_lgb(X_train, y_train, X_val, y_val, feature_cols)

        # --- Train GRU ---
        gru_model, gru_mean, gru_std, gru_val_pred = self._train_gru(
            X_train, y_train, X_val, y_val
        )

        # --- Ensemble ---
        use_gru = gru_model is not None and gru_val_pred is not None
        if use_gru:
            # GRU predictions are shorter (missing first seq_len rows of val set)
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

        # Evaluate ensemble
        val_rmse = float(np.sqrt(np.mean((ensemble_pred - y_val_aligned) ** 2)))
        dir_correct = ((ensemble_pred > 0) == (y_val_aligned > 0)).sum()
        dir_acc = float(dir_correct / len(y_val_aligned))

        from scipy.stats import spearmanr
        ic, _ = spearmanr(ensemble_pred, y_val_aligned)
        ic = float(ic) if not np.isnan(ic) else 0.0

        # Also evaluate LGB-only for comparison
        lgb_dir_correct = ((lgb_val_pred > 0) == (y_val > 0)).sum()
        lgb_dir_acc = float(lgb_dir_correct / len(y_val))
        lgb_ic, _ = spearmanr(lgb_val_pred, y_val)
        lgb_ic = float(lgb_ic) if not np.isnan(lgb_ic) else 0.0

        # Prediction calibration: LGB regression compresses predictions toward zero.
        # Compute a scaling factor from validation data so that predictions
        # have the same std as actual returns. This makes the cost_threshold meaningful.
        pred_std = float(np.std(ensemble_pred)) if len(ensemble_pred) > 0 else 1e-6
        ret_std = float(np.std(y_val_aligned)) if len(y_val_aligned) > 0 else 1e-6
        pred_scale = ret_std / max(pred_std, 1e-8)
        # Cap the scale factor to avoid amplifying noise for weak models
        pred_scale = min(pred_scale, 50.0)
        log.info("%s: prediction calibration scale=%.2f (pred_std=%.6f, ret_std=%.6f)",
                 symbol, pred_scale, pred_std, ret_std)

        # Feature importance (LGB)
        importance = dict(zip(
            feature_cols,
            lgb_model.feature_importance(importance_type="gain").tolist(),
        ))
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

        # --- Save models ---
        sym_clean = symbol.replace("/", "-")

        # Save LGB
        lgb_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_crypto.joblib")
        joblib.dump(lgb_model, lgb_path)

        # Save GRU
        gru_active = False
        if use_gru and TORCH_AVAILABLE:
            gru_path = os.path.join(self.model_dir, f"{sym_clean}_gru_intraday_crypto.pt")
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

        # Save config
        config = {
            "symbol": symbol,
            "feature_names": feature_cols,
            "n_features": n_feat,
            "forward_bars": FORWARD_BARS,
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
            "pred_scale": round(pred_scale, 4),
        }
        config_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_crypto_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        log.info("%s ENSEMBLE: RMSE=%.6f, dir_acc=%.3f, IC=%.3f",
                 symbol, val_rmse, dir_acc, ic)
        log.info("%s LGB-only: dir_acc=%.3f, IC=%.3f, best_iter=%d",
                 symbol, lgb_dir_acc, lgb_ic, lgb_model.best_iteration)
        log.info("Top 5 features: %s",
                 ", ".join(f"{k}={v:.1f}" for k, v in sorted_imp[:5]))

        return config


# ---------------------------------------------------------------------------
# Predictor (for live trading)
# ---------------------------------------------------------------------------
class CryptoIntradayPredictor:
    """Load trained LGB+GRU ensemble and predict on live bars.

    v2: Dynamic ensemble weighting via inverse-MAE (same approach as swing).
    Predictions are buffered for FORWARD_BARS (12) bars (~1 hour) before
    comparing to realized returns. Requires 20 completed observations.
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
        self.feature_engine = CryptoIntradayFeatureEngine()

        # Dynamic ensemble weighting
        from collections import deque
        self._lgb_errors: deque = deque(maxlen=self._ENSEMBLE_WINDOW)
        self._gru_errors: deque = deque(maxlen=self._ENSEMBLE_WINDOW)
        self._pending_preds: deque = deque(maxlen=100)  # (timestamp, lgb_pred, gru_pred, entry_price)
        self._dynamic_lgb_weight = W_LGB  # start at default 0.70

        self._load()

    def _load(self):
        sym_clean = self.symbol.replace("/", "-")
        lgb_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_crypto.joblib")
        gru_path = os.path.join(self.model_dir, f"{sym_clean}_gru_intraday_crypto.pt")
        config_path = os.path.join(self.model_dir, f"{sym_clean}_lgb_intraday_crypto_config.json")

        if not os.path.exists(lgb_path):
            log.warning("No intraday crypto model for %s at %s", self.symbol, lgb_path)
            return

        self.lgb_model = joblib.load(lgb_path)

        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

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

        ensemble_str = "LGB+GRU" if self.gru_model else "LGB-only"
        log.info("Loaded crypto intraday %s model for %s (dir_acc=%.3f, IC=%.3f)",
                 ensemble_str, self.symbol,
                 self.config.get("val_dir_acc", 0) if self.config else 0,
                 self.config.get("val_ic", 0) if self.config else 0)

    def predict(
        self,
        bars: pd.DataFrame,
        btc_bars: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """Predict expected 1-hour return from latest bars.

        Returns:
            dict with {expected_return, direction, confidence, model_type} or None
        """
        if self.lgb_model is None:
            return None

        features = self.feature_engine.build_features(bars, btc_bars, symbol=self.symbol)
        if features.empty:
            return None

        feature_names = self.config.get("feature_names", FEATURE_NAMES) if self.config else FEATURE_NAMES

        # LGB prediction (last row)
        x_lgb = features[feature_names].iloc[-1:].values.astype(np.float32)
        lgb_pred = float(self.lgb_model.predict(x_lgb)[0])

        # GRU prediction (needs sequence)
        gru_pred = None
        if self.gru_model is not None and len(features) >= self.gru_seq_len:
            x_seq = features[feature_names].iloc[-self.gru_seq_len:].values.astype(np.float32)
            x_seq_n = (x_seq - self.gru_mean) / np.clip(self.gru_std, 1e-8, None)
            x_tensor = torch.from_numpy(x_seq_n[np.newaxis])  # (1, seq_len, n_features)

            with torch.no_grad():
                gru_pred = float(self.gru_model(x_tensor).item())

        # --- Evaluate pending predictions (1-hour lag = FORWARD_BARS) ---
        current_ts = features["ts"].iloc[-1]
        current_price = float(bars["close"].iloc[-1])
        resolved = []
        for pp in self._pending_preds:
            pred_ts, lgb_p, gru_p, entry_px = pp
            try:
                elapsed = (pd.Timestamp(current_ts) - pd.Timestamp(pred_ts)).total_seconds()
            except (TypeError, ValueError):
                continue
            # 12 bars × 5 min = 3600s (1 hour)
            if elapsed >= FORWARD_BARS * 5 * 60 and entry_px > 0:
                realized = current_price / entry_px - 1
                self._lgb_errors.append(abs(lgb_p - realized))
                if gru_p is not None:
                    self._gru_errors.append(abs(gru_p - realized))
                resolved.append(pp)
        for pp in resolved:
            self._pending_preds.remove(pp)

        # --- Dynamic ensemble weighting (inverse-MAE) ---
        if gru_pred is not None:
            # Check if enough observations for dynamic weights
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

        # Queue current prediction for future evaluation
        self._pending_preds.append((current_ts, lgb_pred, gru_pred, current_price))

        # Calibrate: scale compressed predictions back to return magnitude
        pred_scale = self.config.get("pred_scale", 1.0) if self.config else 1.0
        expected_return *= pred_scale

        direction = "UP" if expected_return > 0 else "DOWN"
        confidence = min(1.0, abs(expected_return) / TARGET_RETURN)

        return {
            "expected_return": expected_return,
            "direction": direction,
            "confidence": confidence,
            "model_type": model_type,
            "lgb_pred": lgb_pred,
            "gru_pred": gru_pred,
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

    parser = argparse.ArgumentParser(description="Crypto intraday model training (LGB+GRU ensemble)")
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols (e.g. BTC-USD,ETH-USD)")
    parser.add_argument("--days", type=int, default=180,
                        help="Days of history to fetch (default: 180)")
    parser.add_argument("--save-dir", default=None,
                        help="Model save directory")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Walk-forward split: train on first 75%%, OOS on last 25%%")
    args = parser.parse_args()

    from crypto_intraday_data import CryptoIntradayData

    # Default model dir
    model_dir = args.save_dir
    if model_dir is None:
        from signals_engine import PROJECT_ROOT
        model_dir = os.path.join(PROJECT_ROOT, "models", "crypto_intraday")
    os.makedirs(model_dir, exist_ok=True)

    data_source = CryptoIntradayData()
    trainer = CryptoIntradayTrainer(model_dir)
    symbols = [s.strip() for s in args.symbols.split(",")]

    # Fetch BTC bars once (for cross-market features)
    btc_bars = data_source.fetch_training_bars("BTC/USD", days=args.days)
    log.info("BTC reference bars: %d", len(btc_bars))

    results = []
    for sym in symbols:
        log.info("=== Training crypto intraday for %s ===", sym)
        try:
            # Fetch bars
            bars = data_source.fetch_training_bars(sym, days=args.days)
            if len(bars) < 1000:
                log.warning("%s: only %d bars, skipping", sym, len(bars))
                continue

            # Build training data
            engine = CryptoIntradayFeatureEngine()
            is_btc = "BTC" in sym.upper()
            td = engine.build_training_data(
                bars,
                btc_bars=None if is_btc else btc_bars,
                symbol=sym,
            )

            if len(td) < 500:
                log.warning("%s: only %d training samples, skipping", sym, len(td))
                continue

            log.info("%s: %d training samples", sym, len(td))

            # Train
            metrics = trainer.train(sym.replace("/", "-"), td, walk_forward=args.walk_forward)
            results.append(metrics)

        except Exception as exc:
            log.error("%s: training failed — %s", sym, exc)
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("  CRYPTO INTRADAY TRAINING RESULTS (LGB+GRU Ensemble)")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"  {r.get('symbol', '?')}: FAILED — {r['error']}")
        else:
            gru_str = "LGB+GRU" if r.get("gru_active") else "LGB-only"
            print(f"  {r['symbol']:12s} [{gru_str:8s}]: "
                  f"RMSE={r['val_rmse']:.6f}  "
                  f"dir_acc={r['val_dir_acc']:.3f}  "
                  f"IC={r['val_ic']:.3f}  "
                  f"lgb_iter={r['lgb_best_iteration']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
