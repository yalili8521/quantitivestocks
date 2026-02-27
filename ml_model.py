#!/usr/bin/env python3
"""
LSTM ML Model for ETF Direction Prediction
============================================
Predicts next-bar direction (UP/DOWN) with confidence using an LSTM neural network.

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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    RSI_PERIOD,
    DAILY_LOOKBACK,
    PROJECT_ROOT,
)

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

FEATURE_COLS = [
    "rsi14", "ret5", "ret10", "wk_ret", "mo_ret", "vol20",
    "log_dollar_vol", "vix", "vix_chg",
    "vol_imbalance", "vwap_ratio", "dv_accel", "spread_proxy",
    # Enhanced features for strategy optimization
    "atr_pct", "macd_hist_norm", "bb_pct_b", "bb_bandwidth",
    "adx", "momentum_quality", "vol_regime", "trend_strength",
]

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


class DirectionLSTM(nn.Module):
    """LSTM with temporal attention for next-bar direction prediction.

    Architecture:
        Input:      (batch, seq_len, n_features)
        LayerNorm:  normalize input features
        LSTM:       2-layer, hidden_size=96, dropout=0.25
        Attention:  soft attention over all time steps
        FC:         96 -> 48 -> ReLU -> Dropout -> 1 -> Sigmoid
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.classifier(context).squeeze(-1)


# ===================================================================
# Feature Engineering
# ===================================================================
class FeatureEngine:
    """Builds rolling feature matrix from OHLCV + VIX data (daily or intraday)."""

    def __init__(self):
        self._scaler_params: Optional[Dict] = None

    def build_features(self, bars_df: pd.DataFrame,
                       vix_df: pd.DataFrame,
                       mode: str = "daily") -> pd.DataFrame:
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
        df["vix_chg"] = df["vix"].pct_change()

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

        ema_fast = close.ewm(span=10, adjust=False).mean()
        ema_slow = close.ewm(span=30, adjust=False).mean()
        df["trend_strength"] = (ema_fast - ema_slow) / close.replace(0, np.nan)

        # Drop warm-up rows
        df = df.dropna(subset=FEATURE_COLS)
        return df[FEATURE_COLS]

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


# ===================================================================
# VIX history helper (longer lookback for training)
# ===================================================================
def _fetch_vix_for_training(fred_key: Optional[str], lookback_days: int) -> pd.DataFrame:
    """Fetch VIX history. Try FRED first, fall back to yfinance ^VIX."""
    fetcher = FREDVixFetcher(api_key=fred_key)
    vix_df = fetcher.fetch(lookback_days=lookback_days)
    if len(vix_df) >= 20:
        return vix_df

    # FRED free tier only returns ~10 recent observations; fall back to yfinance
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
            return vdf
    except Exception as exc:
        log.warning("yfinance ^VIX fallback failed: %s", exc)

    return vix_df  # return whatever FRED gave us


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

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))
    log.info("Got %d VIX rows.", len(vix_df))

    # 2. Build features
    engine = FeatureEngine()
    features = engine.build_features(bars, vix_df, mode=mode)
    log.info("Built %d feature rows (after warm-up).", len(features))

    if len(features) < seq_len + 10:
        log.error("Not enough data to train. Need at least %d rows, got %d.",
                  seq_len + 10, len(features))
        sys.exit(1)

    # 3. Scaler: fit on training portion only to prevent look-ahead bias
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    full_norm = engine.transform(features)   # transform all with training-only scaler

    # 4. Triple-barrier labels (López de Prado, AFML Chapter 3)
    # Thresholds are tighter for intraday bars (smaller moves expected per bar).
    if mode == "intraday":
        pt_pct, sl_pct, horizon = 0.005, 0.003, 10   # 0.5% target / 0.3% stop / 10 bars
    else:
        pt_pct, sl_pct, horizon = 0.015, 0.010, 5    # 1.5% target / 1.0% stop / 5 bars
    log.info(
        "Triple-barrier labels: PT=%.1f%% SL=%.1f%% horizon=%d bars",
        pt_pct * 100, sl_pct * 100, horizon,
    )

    X_all, y_all = prepare_sequences_triple_barrier(
        full_norm, bars, seq_len, pt_pct=pt_pct, sl_pct=sl_pct, horizon=horizon,
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
    log.info("Class balance — train UP: %.1f%%, val UP: %.1f%%",
             y_train.mean() * 100, y_val.mean() * 100 if len(y_val) > 0 else 0)

    # 4. Create model with improved architecture
    n_features = len(FEATURE_COLS)
    model = DirectionLSTM(n_features=n_features)

    # Label smoothing: soft targets reduce overconfidence (0.95/0.05 instead of 1/0)
    LABEL_SMOOTH = 0.05
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Cosine annealing LR schedule — warm restarts improve convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01,
    )

    # 5. Training with early stopping, gradient clipping, label smoothing
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10
    GRAD_CLIP = 1.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            yb_smooth = yb * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            loss = criterion(preds, yb_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation (use hard labels for unbiased eval)
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                predicted = (preds > 0.5).float()
                correct += (predicted == yb).sum().item()
        val_loss /= max(len(val_ds), 1)
        val_acc = correct / max(len(val_ds), 1)

        current_lr = optimizer.param_groups[0]["lr"]
        log.info("Epoch %2d/%d  train=%.4f  val=%.4f  acc=%.3f  lr=%.2e",
                 epoch + 1, epochs, train_loss, val_loss, val_acc, current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

    log.info("Training complete for %s (%s). Best val_loss=%.4f",
             symbol, mode, best_val_loss)
    return model, engine


# ===================================================================
# Predictor (inference)
# ===================================================================
class Predictor:
    """Load a trained model and produce predictions."""

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR,
                 mode: str = "daily", intraday_interval: str = "5min"):
        self.symbol = symbol
        self.model_dir = model_dir
        self.mode = mode
        self.intraday_interval = intraday_interval
        self.engine = FeatureEngine()
        self.model: Optional[DirectionLSTM] = None
        self._load()

    def _load(self) -> None:
        suffix = "" if self.mode == "daily" else f"_{self.intraday_interval}"
        weights_path = os.path.join(self.model_dir, f"{self.symbol}_lstm{suffix}.pt")
        scaler_path = os.path.join(self.model_dir, f"{self.symbol}_scaler{suffix}.json")
        if not os.path.exists(weights_path):
            mode_hint = f" --mode {self.mode}" if self.mode != "daily" else ""
            raise FileNotFoundError(
                f"No trained model for {self.symbol} ({self.mode}). "
                f"Run: python main.py train --symbol {self.symbol}{mode_hint}")
        self.engine.load_scaler(scaler_path)
        self.model = DirectionLSTM(n_features=len(FEATURE_COLS))
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = SEQ_LEN) -> dict:
        """Produce a prediction from the most recent seq_len bars.

        Returns {"direction": "UP"/"DOWN", "probability": float, "confidence": float}
        """
        features = self.engine.build_features(bars_df, vix_df, mode=self.mode)
        features_norm = self.engine.transform(features)

        if len(features_norm) < seq_len:
            return {"direction": "UNKNOWN", "probability": 0.5, "confidence": 0.0}

        window = features_norm.iloc[-seq_len:].values
        x = torch.FloatTensor(window).unsqueeze(0)

        with torch.no_grad():
            prob = self.model(x).item()

        direction = "UP" if prob > 0.5 else "DOWN"
        confidence = abs(prob - 0.5) * 2

        return {
            "direction": direction,
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
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

    if args.command == "train":
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
