#!/usr/bin/env python3
"""
Vol Expansion LSTM — predicts whether realized volatility will expand ≥10% in next 5 bars.
Used to time straddle entries (buy cheap premium BEFORE vol spikes, not during them).

Usage (via main.py):
    python main.py train-vol --symbol SPY
    python main.py train-vol --symbol SPY --with-meta
    python main.py train-vol --symbol ALL

Architecture:
    Uses VolLSTM (same 2-layer LSTM + attention, classification head).
    15 vol-focused features: vol20, bb_bandwidth, vol_regime, bb_pct_b,
    ret5, rsi14, vix, vix_chg, momentum_quality, adx,
    iv_rv_spread, vov, ewma_vol, vol5, vol20_change
    Target: 1 if max(vol20[t+1..t+5]) > vol20[t] * 1.10 else 0

Model files (saved to models/ directory):
    {symbol}_vol_lstm.pt
    {symbol}_vol_scaler.json
    {symbol}_vol_meta_rf.joblib         (optional, --with-meta)
    {symbol}_vol_meta_rf_config.json    (optional, --with-meta)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader

from signals_engine import build_adapter, PROJECT_ROOT
from ml_model import (
    FeatureEngine, TemporalAttention, _fetch_vix_for_training,
    DEFAULT_MODEL_DIR, SEQ_LEN,
)


class VolLSTM(nn.Module):
    """Classification LSTM for vol expansion prediction (binary: expand yes/no).

    Identical architecture to the original DirectionLSTM (pre-v2 regression rename).
    Uses self.classifier with Sigmoid — NOT the ReturnLSTM regression head.
    This ensures backward compatibility with saved vol checkpoints.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("options_ml")

# ---------------------------------------------------------------------------
# Vol feature set (10 features)
# We use symbol="IGV" in build_features() to get the 17-feature daily set
# (FEATURE_COLS_DAILY + vix, vix_chg, trend_strength, momentum_quality, mo_ret)
# then select the 10 vol-relevant columns below.
# ---------------------------------------------------------------------------
_VOL_BUILD_SYMBOL = "IGV"   # gives a 17-feature superset that includes vix/vix_chg

VOL_FEATURE_COLS = [
    "vol20",           # 20-bar realized vol — primary predictor
    "bb_bandwidth",    # Bollinger band width — vol expansion indicator
    "vol_regime",      # short/long vol ratio — vol trend
    "bb_pct_b",        # Bollinger %B — price at extremes
    "ret5",            # 5-bar return — recent move magnitude
    "rsi14",           # RSI(14)/100 — momentum
    "vix",             # VIX level — implied vol environment
    "vix_chg",         # VIX % change — vol momentum
    "momentum_quality",# rolling up-day fraction
    "adx",             # ADX/100 — trend strength (trending = vol expands)
    # ── Sinclair (Volatility Trading) edge metrics ──────────────────
    "iv_rv_spread",    # IV premium: VIX/100 − vol20 (Ch.2 core edge signal)
    "vov",             # vol of vol: rolling std of VIX daily changes (Ch.4 timing)
    "ewma_vol",        # EWMA variance forecast λ=0.94/RiskMetrics (Ch.3 GARCH-lite)
    # ── Vol velocity & short-term regime ────────────────────────────
    "vol5",            # 5-bar realized vol — short-term volatility clustering
    "vol20_change",    # 5-bar % change in vol20 — vol acceleration signal
]

VOL_EXPAND_THRESHOLD = 1.10   # 10% increase in vol20 triggers label=1
VOL_FORWARD_BARS = 5          # look forward 5 bars for target
VOL_META_THRESHOLD = 0.55     # default meta RF threshold


# ---------------------------------------------------------------------------
# Sinclair-inspired derived vol features
# ---------------------------------------------------------------------------
def _add_derived_vol_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add IV/RV relationship features before scaler fitting / inference.

    These three features encode Euan Sinclair's core thesis from
    *Volatility Trading*: option edge comes from the spread between
    implied and realized vol, not from the level of either alone.

    Must be called AFTER build_features() (so vix, vol20, vix_chg exist)
    and BEFORE fit_scaler() / transform() so the scaler captures their
    distribution from the training split.

    Features added
    --------------
    iv_rv_spread : VIX/100 − vol20
        Positive  → implied vol exceeds realized → options expensive (selling edge).
        Negative  → realized vol exceeds implied → options cheap (buying / straddle edge).
        Sinclair Ch.2: the single most reliable predictor of vol-trade profitability.

    vov : rolling 20-bar std of vix_chg (vol of vol)
        High VVoV → IV is jumping around unpredictably → better timing for
        straddle entries when you expect a regime shift (Sinclair Ch.4).

    ewma_vol : EWMA annualized vol (λ≈0.94, RiskMetrics standard)
        More responsive than simple vol20 to recent variance clustering
        (Sinclair Ch.3 on GARCH-family forecasts). Gives the LSTM a
        forward-looking variance signal rather than a backward-looking average.
    """
    df = features_df.copy()

    # IV/RV spread: clipped to ±15% to prevent outlier dominance
    df["iv_rv_spread"] = (df["vix"] / 100.0 - df["vol20"]).clip(-0.15, 0.15)

    # Vol of vol: 20-bar rolling std of daily VIX % change
    vov = df["vix_chg"].rolling(20, min_periods=5).std()
    df["vov"] = vov.fillna(vov.median() if not vov.dropna().empty else 0.0)

    # EWMA variance (λ=0.94 ↔ span=32): annualised realised vol forecast
    daily_ret = df["ret5"] / 5.0
    ewma_var = (daily_ret ** 2).ewm(span=32, adjust=False).mean()
    df["ewma_vol"] = np.sqrt(ewma_var.clip(lower=0) * 252).fillna(df["vol20"])

    # Short-term realized vol (5-bar window) — captures recent clustering
    # Approximation: use ret5/5 as daily return proxy, rolling 5-bar std
    vol5 = (daily_ret ** 2).rolling(5, min_periods=3).mean()
    df["vol5"] = np.sqrt(vol5.clip(lower=0) * 252).fillna(df["vol20"])

    # Vol acceleration: 5-bar % change in vol20 — rising vol is more predictive
    # than level alone (Sinclair Ch.4: timing via vol momentum)
    vol20_chg = df["vol20"].pct_change(5)
    df["vol20_change"] = vol20_chg.fillna(0.0).clip(-1.0, 1.0)

    return df


# ---------------------------------------------------------------------------
# Target and sequence builders
# ---------------------------------------------------------------------------
def _build_vol_targets(features_raw: pd.DataFrame) -> np.ndarray:
    """Build per-bar vol expansion labels.

    label[t] = 1 if max(vol20[t+1..t+5]) > vol20[t] * 1.10, else 0.
    Last VOL_FORWARD_BARS entries cannot have a label → set to 0.
    Uses raw (unnormalized) vol20 for threshold comparison.
    """
    vol20 = features_raw["vol20"].values.astype(np.float64)
    n = len(vol20)
    labels = np.zeros(n, dtype=np.float32)
    for t in range(n - VOL_FORWARD_BARS):
        if vol20[t] <= 0:
            continue
        future_max = float(np.max(vol20[t + 1: t + VOL_FORWARD_BARS + 1]))
        labels[t] = 1.0 if future_max > vol20[t] * VOL_EXPAND_THRESHOLD else 0.0
    return labels


def _build_vol_sequences(
    features_norm: pd.DataFrame,
    vol_labels: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple:
    """Build (X, y) arrays for vol model training.

    X[i] = features_norm[VOL_FEATURE_COLS][i : i+seq_len]   shape (seq_len, 10)
    y[i] = vol_labels[i + seq_len - 1]                       label at last bar of window
    """
    # Select only vol features
    vol_feat = features_norm[VOL_FEATURE_COLS].values.astype(np.float32)
    n = len(vol_feat)
    X_list, y_list = [], []
    for i in range(n - seq_len):
        label_pos = i + seq_len - 1
        if label_pos >= len(vol_labels):
            break
        X_list.append(vol_feat[i: i + seq_len])
        y_list.append(vol_labels[label_pos])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_vol_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    epochs: int = 50,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
) -> None:
    """Train VolExpansion LSTM for one symbol.

    Architecture: reuses VolLSTM with hidden_size=96, 15 vol features.

    Saves:
        {symbol}_vol_lstm.pt
        {symbol}_vol_scaler.json
    """
    os.makedirs(save_dir, exist_ok=True)

    log.info("=== Vol LSTM: %s ===", symbol)
    log.info("Fetching daily data (lookback=%d)...", lookback)
    bars = adapter.fetch_daily(symbol, lookback)
    log.info("Got %d bars.", len(bars))

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))
    log.info("Got %d VIX rows.", len(vix_df))

    # Build features using IGV override to include vix/vix_chg/momentum_quality
    engine = FeatureEngine()
    features = engine.build_features(bars, vix_df, mode="daily", symbol=_VOL_BUILD_SYMBOL)
    log.info("Built %d feature rows.", len(features))

    if len(features) < SEQ_LEN + VOL_FORWARD_BARS + 20:
        log.error("Not enough data: %d rows (need ≥ %d).",
                  len(features), SEQ_LEN + VOL_FORWARD_BARS + 20)
        return

    # Add Sinclair-inspired iv_rv_spread / vov / ewma_vol before scaler
    features = _add_derived_vol_features(features)

    # Fit scaler on training split only (prevent look-ahead)
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    features_norm = engine.transform(features)

    # Vol expansion labels from raw (unnormalized) vol20
    vol_labels = _build_vol_targets(features)

    # Build sequences using normalized features
    X_all, y_all = _build_vol_sequences(features_norm, vol_labels, SEQ_LEN)
    log.info("Sequences: %d total. Vol expanding: %.1f%%", len(y_all), y_all.mean() * 100)

    # Purge + embargo split (same as primary LSTM training)
    seq_split = int(len(X_all) * 0.8)
    embargo = SEQ_LEN
    X_train, y_train = X_all[:seq_split], y_all[:seq_split]
    X_val,   y_val   = X_all[seq_split + embargo:], y_all[seq_split + embargo:]

    if len(X_val) == 0:
        log.warning("No validation samples — using last 10%% of training as validation.")
        val_cut = max(1, len(X_train) - int(len(X_train) * 0.1))
        X_val, y_val = X_train[val_cut:], y_train[val_cut:]
        X_train, y_train = X_train[:val_cut], y_train[:val_cut]

    log.info("Train: %d, Val: %d", len(X_train), len(X_val))

    # Model: reuse VolLSTM — hidden_size=96 matches direction model capacity
    model = VolLSTM(n_features=len(VOL_FEATURE_COLS), hidden_size=96)

    LABEL_SMOOTH = 0.05
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5,
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    weights_path = os.path.join(save_dir, f"{symbol}_vol_lstm.pt")
    scaler_path  = os.path.join(save_dir, f"{symbol}_vol_scaler.json")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_run = 0
    patience_counter = 0
    PATIENCE = 10

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            yb_smooth = yb * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            loss = criterion(preds, yb_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                correct += ((preds > 0.5).float() == yb).sum().item()
        val_loss /= max(len(val_ds), 1)
        val_acc = correct / max(len(val_ds), 1)

        log.info("Epoch %2d/%d  train=%.4f  val=%.4f  acc=%.3f",
                 epoch + 1, epochs, train_loss, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
            engine.save_scaler(scaler_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d.", epoch + 1)
                break

    metrics_path = os.path.join(save_dir, f"{symbol}_vol_lstm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_acc": round(best_val_acc, 4),
            "epochs_run": epochs_run,
        }, f, indent=2)
    log.info("Vol LSTM complete: %s  best_val_loss=%.4f  val_acc=%.3f", symbol, best_val_loss, best_val_acc)
    log.info("Saved → %s", weights_path)


def train_vol_meta_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
) -> None:
    """Train Random Forest meta-model on top of vol LSTM.

    Answers: 'When will the vol LSTM be correct?'
    Uses same calibration pattern as train_meta_model() in ml_model.py.

    Saves:
        {symbol}_vol_meta_rf.joblib
        {symbol}_vol_meta_rf_config.json
    """
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, f"{symbol}_vol_lstm.pt")
    scaler_path  = os.path.join(save_dir, f"{symbol}_vol_scaler.json")
    meta_path    = os.path.join(save_dir, f"{symbol}_vol_meta_rf.joblib")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Vol LSTM not found: {weights_path}. "
            f"Train first: python main.py train-vol --symbol {symbol}"
        )

    # Load vol LSTM
    engine = FeatureEngine()
    engine.load_scaler(scaler_path)
    vol_model = VolLSTM(n_features=len(VOL_FEATURE_COLS), hidden_size=96)
    vol_model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True))
    vol_model.eval()

    # Fetch data and build features (same as training)
    bars = adapter.fetch_daily(symbol, lookback)
    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))
    features = engine.build_features(bars, vix_df, mode="daily", symbol=_VOL_BUILD_SYMBOL)
    features = _add_derived_vol_features(features)
    features_norm = engine.transform(features)

    if len(features_norm) < SEQ_LEN + VOL_FORWARD_BARS + 10:
        log.error("Not enough data for meta-training: %d rows.", len(features_norm))
        return

    # Build vol expansion labels (raw features for threshold comparison)
    vol_labels = _build_vol_targets(features)

    # Run vol LSTM on all windows → get vol_prob per sequence
    vol_feat_arr = features_norm[VOL_FEATURE_COLS].values.astype(np.float32)
    n = len(vol_feat_arr)
    n_seqs = n - SEQ_LEN
    vol_probs = np.zeros(n_seqs, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_seqs):
            window = vol_feat_arr[i: i + SEQ_LEN]
            x = torch.FloatTensor(window).unsqueeze(0)
            vol_probs[i] = vol_model(x).item()

    # Meta labels: was vol LSTM correct?
    predicted_expanding = (vol_probs > 0.5).astype(np.float32)
    actual = vol_labels[SEQ_LEN - 1: SEQ_LEN - 1 + n_seqs]
    was_correct = (predicted_expanding == actual).astype(np.int32)

    # Meta features: last bar's 10 vol features + vol_prob → 11-dim
    last_bars = vol_feat_arr[SEQ_LEN - 1: SEQ_LEN - 1 + n_seqs]
    meta_X = np.hstack([last_bars, vol_probs.reshape(-1, 1)])
    meta_y = was_correct

    log.info("Vol meta dataset: %d samples. Correct rate: %.1f%%",
             len(meta_y), meta_y.mean() * 100)

    # Temporal 80/20 split
    split = int(len(meta_X) * 0.8)
    X_train, y_train = meta_X[:split], meta_y[:split]
    X_val,   y_val   = meta_X[split:], meta_y[split:]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    meta_val_probs = rf.predict_proba(X_val)[:, 1]

    # Per-symbol threshold calibration (maximize precision lift × log coverage)
    base_rate = float(y_val.mean())
    best_t, best_score = VOL_META_THRESHOLD, 0.0
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
            best_score, best_t = score, round(float(t), 3)
            best_prec, best_cov = prec, cov

    log.info("Vol meta RF — val_acc=%.3f  threshold=%.3f  prec=%.1f%%  cov=%.1f%%",
             val_acc, best_t, best_prec * 100, best_cov * 100)

    joblib.dump(rf, meta_path)
    config_path = meta_path.replace(".joblib", "_config.json")
    config = {
        "symbol": symbol, "threshold": best_t,
        "val_precision": round(best_prec, 4),
        "val_coverage": round(float(best_cov), 4),
        "val_accuracy": round(float(val_acc), 4),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Saved vol meta RF → %s (threshold %.3f)", meta_path, best_t)


# ---------------------------------------------------------------------------
# VolPredictor (inference)
# ---------------------------------------------------------------------------
class VolPredictor:
    """Load trained vol LSTM (+ optional meta RF) and predict vol expansion.

    Graceful fallback when model not found:
        predict() returns {"vol_expansion_prob": 0.5, "vol_expanding": False}
    """

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol    = symbol
        self.model_dir = model_dir
        self.engine    = FeatureEngine()
        self.model: Optional[VolLSTM] = None
        self.meta_model = None
        self.meta_threshold = VOL_META_THRESHOLD
        self._loaded = False
        self._load()

    def _load(self) -> None:
        weights_path = os.path.join(self.model_dir, f"{self.symbol}_vol_lstm.pt")
        scaler_path  = os.path.join(self.model_dir, f"{self.symbol}_vol_scaler.json")

        if not os.path.exists(weights_path):
            log.debug("No vol LSTM for %s — neutral fallback active.", self.symbol)
            return

        self.engine.load_scaler(scaler_path)
        self.model = VolLSTM(n_features=len(VOL_FEATURE_COLS), hidden_size=96)
        try:
            self.model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True))
        except RuntimeError as exc:
            if "size mismatch" in str(exc).lower() or "unexpected key" in str(exc).lower():
                log.warning(
                    "Vol LSTM architecture mismatch for %s (stale model — needs retrain). "
                    "Run: python main.py train-vol --symbol %s",
                    self.symbol, self.symbol,
                )
                self.model = None
                return
            raise
        self.model.eval()
        self._loaded = True

        # Optional meta RF
        meta_path   = os.path.join(self.model_dir, f"{self.symbol}_vol_meta_rf.joblib")
        config_path = meta_path.replace(".joblib", "_config.json")
        if os.path.exists(meta_path):
            self.meta_model = joblib.load(meta_path)
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                self.meta_threshold = float(cfg.get("threshold", VOL_META_THRESHOLD))
            log.debug("Loaded vol meta RF for %s (threshold=%.3f).",
                      self.symbol, self.meta_threshold)

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame) -> dict:
        """Predict vol expansion probability.

        Returns:
            vol_expansion_prob : float [0,1] — probability vol will expand ≥10% in 5 bars
            vol_expanding      : bool — True when model expects significant vol expansion
        """
        if not self._loaded:
            return {"vol_expansion_prob": 0.0, "vol_expanding": False}

        try:
            features = self.engine.build_features(
                bars_df, vix_df, mode="daily", symbol=_VOL_BUILD_SYMBOL)
            features = _add_derived_vol_features(features)
            features_norm = self.engine.transform(features)

            if len(features_norm) < SEQ_LEN:
                return {"vol_expansion_prob": 0.0, "vol_expanding": False}

            vol_feat = features_norm[VOL_FEATURE_COLS].iloc[-SEQ_LEN:].values.astype(np.float32)
            x = torch.FloatTensor(vol_feat).unsqueeze(0)
            with torch.no_grad():
                vol_prob = float(self.model(x).item())

            # Optional meta gate
            if self.meta_model is not None:
                last_bar = features_norm[VOL_FEATURE_COLS].iloc[-1].values.astype(np.float32)
                meta_feat = np.append(last_bar, vol_prob).reshape(1, -1)
                meta_conf = float(self.meta_model.predict_proba(meta_feat)[0][1])
                vol_expanding = vol_prob > 0.5 and meta_conf >= self.meta_threshold
            else:
                vol_expanding = vol_prob > 0.5

            return {
                "vol_expansion_prob": round(vol_prob, 4),
                "vol_expanding": vol_expanding,
            }
        except Exception as exc:
            log.warning("VolPredictor.predict failed for %s: %s", self.symbol, exc)
            return {"vol_expansion_prob": 0.0, "vol_expanding": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ACTIVE_SYMBOLS = [
    "SPY", "QQQ", "IWM", "SOXX",
    "EWT", "GLD", "EEM", "SLV",
    "EWJ", "EWS", "XLE", "INDA",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vol Expansion LSTM — train vol prediction models.",
    )
    parser.add_argument("--symbol", required=True,
                        help="Symbol (e.g. SPY) or ALL for all active symbols")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=1000)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--with-meta", action="store_true",
                        help="Also train meta RF on top of vol LSTM")
    args = parser.parse_args()

    adapter  = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols  = ACTIVE_SYMBOLS if args.symbol.upper() == "ALL" else [args.symbol.upper()]

    for sym in symbols:
        try:
            train_vol_model(
                symbol=sym, adapter=adapter, fred_key=fred_key,
                epochs=args.epochs, lookback=args.lookback, save_dir=args.model_dir,
            )
        except Exception as exc:
            log.error("Vol LSTM failed for %s: %s", sym, exc)
            continue

        if args.with_meta:
            try:
                train_vol_meta_model(
                    symbol=sym, adapter=adapter, fred_key=fred_key,
                    lookback=args.lookback, save_dir=args.model_dir,
                )
            except Exception as exc:
                log.error("Vol meta RF failed for %s: %s", sym, exc)


if __name__ == "__main__":
    main()
