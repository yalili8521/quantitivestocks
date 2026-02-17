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
]

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ===================================================================
# LSTM Model
# ===================================================================
class DirectionLSTM(nn.Module):
    """LSTM that predicts next-bar direction (up/down).

    Architecture:
        Input:  (batch, seq_len, n_features)
        LSTM:   2-layer, hidden_size=64, dropout=0.2
        FC:     64 -> 32 -> ReLU -> 1 -> Sigmoid

    Output: probability that next bar closes higher than current bar.
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)


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
def prepare_sequences(features_df: pd.DataFrame,
                      daily_df: pd.DataFrame,
                      seq_len: int = SEQ_LEN):
    """Build (X, y) pairs for training.

    X[i] = features for days [i, i+seq_len)
    y[i] = 1 if close[i+seq_len] > close[i+seq_len-1], else 0
    """
    feature_values = features_df.values
    close_values = daily_df.loc[features_df.index, "close"].astype(float).values

    X_list, y_list = [], []
    for i in range(len(feature_values) - seq_len):
        X_list.append(feature_values[i: i + seq_len])
        next_close = close_values[i + seq_len]
        curr_close = close_values[i + seq_len - 1]
        y_list.append(1.0 if next_close > curr_close else 0.0)

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

    # 3. Temporal split: 80/20
    split_idx = int(len(features) * 0.8)
    train_feat = features.iloc[:split_idx]
    val_feat = features.iloc[split_idx:]

    engine.fit_scaler(train_feat)
    train_norm = engine.transform(train_feat)
    val_norm = engine.transform(val_feat)

    X_train, y_train = prepare_sequences(train_norm, bars, seq_len)
    X_val, y_val = prepare_sequences(val_norm, bars, seq_len)

    log.info("Training samples: %d, Validation samples: %d", len(X_train), len(X_val))
    log.info("Class balance — train UP: %.1f%%, val UP: %.1f%%",
             y_train.mean() * 100, y_val.mean() * 100 if len(y_val) > 0 else 0)

    # 4. Create model
    n_features = len(FEATURE_COLS)
    model = DirectionLSTM(n_features=n_features)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Training with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 7

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validation
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

        log.info("Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                 epoch + 1, epochs, train_loss, val_loss, val_acc)

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
