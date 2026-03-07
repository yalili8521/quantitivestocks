#!/usr/bin/env python3
"""
Expansion XGBoost Factor Model (v2 Regression)
====================================
XGBoost regressor for expansion-group ETFs (international/sector).
Predicts forward 10-day expected return (replaces v1 binary classification).

Based on AQR factor research (Moskowitz, Ooi, Pedersen 2012; Asness et al. 2013):
- TSMOM (time-series momentum) positive for all 58 futures contracts tested
- Factor momentum Sharpe 0.84 across 65 factors
- NeurIPS 2022 (Grinsztajn et al.): tree models outperform deep learning on tabular data

Architecture:
    - Point-in-time tabular features (no sequences)
    - XGBClassifier with factor-based features
    - Relative momentum vs SPY as key cross-sectional signal
    - VIX regime percentile for macro context

Features (15):
    Factor signals: ret252 (TSMOM), ret63, rel_momentum_vs_spy_63, rel_momentum_vs_spy_252
    Technical: rsi14, vol20, adx, bb_pct_b, bb_bandwidth, macd_hist_norm
    Momentum quality: momentum_quality, vol_regime
    Macro: vix_pctrank
    Volume: dv_accel
    Mean reversion: ret5

Usage (via main.py):
    python main.py train-expansion --symbols EWJ,EWS,XLE,INDA --provider yahoo
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from signals_engine import (
    PROJECT_ROOT, build_adapter,
    compute_rsi, compute_atr, compute_macd, compute_bollinger_bands,
    compute_adx, compute_momentum_quality, RSI_PERIOD, DAILY_LOOKBACK,
)
from ml_model import _fetch_vix_for_training, DEFAULT_MODEL_DIR
from cross_asset_signals import CrossAssetFeatureBuilder, get_cross_asset_features
OPTIONS_FLOW_FEATURES = ["pc_volume_ratio", "pc_oi_ratio", "vix_term_ratio", "vix_term_inverted"]
from alpha_signals import AlphaFeatureBuilder, get_alpha_features
from factor_signals import FactorFeatureBuilder, get_factor_features
from market_signals import MarketSignalBuilder, get_market_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expansion_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.55  # XGBoost P > this to consider tradeable

EXPANSION_FEATURES = [
    # Factor signals (AQR research)
    "ret252",                 # TSMOM: 12-month trailing return
    "ret63",                  # Quarterly momentum
    "rel_momentum_vs_spy_63", # Cross-sectional: outperformance vs SPY (3m)
    "rel_momentum_vs_spy_252",# Cross-sectional: outperformance vs SPY (12m)
    # Technical
    "rsi14",
    "vol20",
    "adx",
    "bb_pct_b",
    "bb_bandwidth",
    "macd_hist_norm",
    # Momentum structure
    "momentum_quality",
    "vol_regime",
    # Macro
    "vix_pctrank",
    # Volume
    "dv_accel",
    # Mean reversion
    "ret5",
]

# Triple-barrier thresholds for expansion (wider — these are less liquid ETFs)
TB_PT_PCT = 0.020    # 2.0% profit target
TB_SL_PCT = 0.012    # 1.2% stop loss
TB_HORIZON = 7       # 7-bar look-forward


def get_expansion_feature_cols(symbol: str | None = None) -> list:
    """Return expansion feature list with cross-asset and options flow features."""
    cols = list(EXPANSION_FEATURES)
    # Add cross-asset features (per-symbol)
    if symbol:
        cols.extend(get_cross_asset_features(symbol))
    else:
        cols.append("treasury_slope")
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
# Feature Engineering
# ---------------------------------------------------------------------------
class ExpansionFeatureEngine:
    """Build tabular factor features for expansion-group ETFs.

    Unlike LSTM or PatchTST, XGBoost uses point-in-time features
    (no sequence window). Each bar produces one feature vector.
    """

    def __init__(self):
        self._scaler_params: Optional[dict] = None
        self._cross_builder: Optional[CrossAssetFeatureBuilder] = None
        self._options_engine = None
        self._alpha_builder: Optional[AlphaFeatureBuilder] = None
        self._factor_builder: Optional[FactorFeatureBuilder] = None
        self._market_builder: Optional[MarketSignalBuilder] = None

    def build_features(
        self,
        bars_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        spy_bars: Optional[pd.DataFrame] = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Build 15-feature tabular matrix.

        Args:
            bars_df: OHLCV for the symbol
            vix_df: VIX daily data
            spy_bars: SPY daily bars for relative momentum (optional)
        """
        close = bars_df["close"].astype(float)
        high = bars_df["high"].astype(float)
        low = bars_df["low"].astype(float)
        volume = bars_df["volume"].astype(float)
        df = pd.DataFrame(index=bars_df.index)
        annualize = np.sqrt(252)

        # --- Factor signals ---
        df["ret252"] = close.pct_change(252)
        df["ret63"] = close.pct_change(63)
        df["ret5"] = close.pct_change(5)

        # Relative momentum vs SPY (cross-sectional)
        if spy_bars is not None and not spy_bars.empty:
            spy_close = spy_bars["close"].astype(float)
            # Align by date
            bar_dates = pd.to_datetime(bars_df["ts"]).dt.date.values
            spy_dates = pd.to_datetime(spy_bars["ts"]).dt.date
            spy_map = dict(zip(spy_dates.values, spy_close.values))

            spy_aligned = pd.Series(
                [spy_map.get(d, np.nan) for d in bar_dates],
                index=bars_df.index,
            ).ffill()

            spy_ret63 = spy_aligned.pct_change(63)
            spy_ret252 = spy_aligned.pct_change(252)
            df["rel_momentum_vs_spy_63"] = df["ret63"] - spy_ret63
            df["rel_momentum_vs_spy_252"] = df["ret252"] - spy_ret252
        else:
            df["rel_momentum_vs_spy_63"] = 0.0
            df["rel_momentum_vs_spy_252"] = 0.0

        # --- Technical indicators ---
        df["rsi14"] = compute_rsi(close, RSI_PERIOD) / 100.0
        df["vol20"] = close.pct_change().rolling(20).std() * annualize
        df["adx"] = compute_adx(high, low, close, period=14) / 100.0

        _, _, _, pct_b, bandwidth = compute_bollinger_bands(close, window=20)
        df["bb_pct_b"] = pct_b
        df["bb_bandwidth"] = bandwidth

        _, _, macd_hist = compute_macd(close)
        df["macd_hist_norm"] = macd_hist / close.replace(0, np.nan)

        # --- Momentum structure ---
        df["momentum_quality"] = compute_momentum_quality(close, window=20)

        vol_short = close.pct_change().rolling(10).std() * annualize
        vol_long = close.pct_change().rolling(30).std() * annualize
        df["vol_regime"] = vol_short / vol_long.replace(0, np.nan)

        # --- Macro: VIX percentile rank ---
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        vix_map = {}
        if not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_map[d] = row["vix"]
        vix_series = bar_dates.map(lambda d: vix_map.get(d, np.nan)).astype(float)
        vix_series = pd.Series(vix_series.values, index=bars_df.index).ffill()

        df["vix_pctrank"] = vix_series.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # --- Volume ---
        dv = close * volume
        dv_ma_5 = dv.rolling(5).mean()
        dv_ma_10 = dv.rolling(10).mean()
        df["dv_accel"] = (dv_ma_5 - dv_ma_10) / dv_ma_10.replace(0, np.nan)

        # --- Cross-asset features (per-symbol macro/cross-market signals) ---
        if symbol:
            try:
                if self._cross_builder is None:
                    self._cross_builder = CrossAssetFeatureBuilder(
                        fred_key=os.environ.get("FRED_API_KEY"))
                cross_df = self._cross_builder.build_features(bars_df, symbol)
                for col in cross_df.columns:
                    df[col] = cross_df[col]
            except Exception as exc:
                log.warning("Cross-asset features failed for %s: %s", symbol, exc)
                for col in get_cross_asset_features(symbol):
                    if col not in df.columns:
                        df[col] = np.nan

        # --- Options flow features (stubbed — options_flow module removed) ---
        for col in OPTIONS_FLOW_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        # --- Alpha signal features (CBOE historical + insider) ---
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

        # --- Factor signal features (credit, risk appetite, yield curve, FF, etc.) ---
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

        # --- Market signal features (volume, VRP, calendar, breadth) ---
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

        # Drop NaN warmup (only on base features; optional features can be NaN)
        all_cols = get_expansion_feature_cols(symbol)
        df = df.dropna(subset=EXPANSION_FEATURES)
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = df[col].ffill().fillna(0.0)
        return df[all_cols]

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
            "std": self._scaler_params["std"].to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_scaler(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._scaler_params = {
            "mean": pd.Series(data["mean"]),
            "std": pd.Series(data["std"]),
        }


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------
def _prepare_tabular_triple_barrier(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    pt_pct: float = TB_PT_PCT,
    sl_pct: float = TB_SL_PCT,
    horizon: int = TB_HORIZON,
) -> tuple:
    """Triple-barrier labeling for tabular (non-sequential) model.

    Returns (X, y) where X is the feature matrix and y are binary labels.
    """
    full_close = bars_df["close"].astype(float).values
    bar_positions = bars_df.index.get_indexer(features_df.index)
    feature_values = features_df.values

    X_list = []
    y_list = []

    for i in range(len(feature_values)):
        bar_pos = bar_positions[i]
        if bar_pos < 0 or bar_pos + horizon >= len(full_close):
            continue

        entry_price = full_close[bar_pos]
        if entry_price <= 0:
            continue

        label = None
        for fwd in range(1, horizon + 1):
            fwd_pos = bar_pos + fwd
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
            final_pos = min(bar_pos + horizon, len(full_close) - 1)
            final_ret = (full_close[final_pos] - entry_price) / entry_price
            label = 1.0 if final_ret > 0 else 0.0

        X_list.append(feature_values[i])
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _prepare_tabular_regression(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    forward_days: int = 10,
) -> tuple:
    """Forward N-day return regression labels for tabular model (v2)."""
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
        fwd_ret = max(-0.10, min(0.10, fwd_ret))
        X_list.append(feature_values[i])
        y_list.append(fwd_ret)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_expansion_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
) -> Optional[xgb.XGBRegressor]:
    """Train XGBoost factor model for expansion-group ETFs (v2 regression).

    1. Fetch daily data (symbol + SPY for relative momentum)
    2. Build factor features (15 tabular features)
    3. Forward 10-day return regression labels
    4. Train XGBRegressor with walk-forward split
    5. Save model + scaler
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info("=== Training expansion XGBoost for %s ===", symbol)

    # 1. Fetch data (symbol + SPY for relative momentum)
    log.info("Fetching %d daily bars for %s...", lookback, symbol)
    bars = adapter.fetch_daily(symbol, lookback)
    log.info("Got %d bars.", len(bars))

    log.info("Fetching SPY bars for relative momentum...")
    spy_bars = adapter.fetch_daily("SPY", lookback)
    log.info("Got %d SPY bars.", len(spy_bars))

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))

    # 2. Build features
    engine = ExpansionFeatureEngine()
    features = engine.build_features(bars, vix_df, spy_bars=spy_bars, symbol=symbol)
    log.info("Built %d feature rows for %s.", len(features), symbol)

    if len(features) < 50:
        log.error("Not enough data for %s (%d rows). Need at least 50.", symbol, len(features))
        return None

    # 3. Scaler (fit on training portion only)
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    full_norm = engine.transform(features)

    # 4. Forward 10-day return regression labels (v2)
    FORWARD_DAYS = 10
    X_all, y_all = _prepare_tabular_regression(
        full_norm, bars, forward_days=FORWARD_DAYS,
    )
    log.info("Labeled %d samples. Mean return: %+.4f%%", len(y_all), y_all.mean() * 100)

    # 5. Walk-forward split
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    log.info("Train: %d, Val: %d. Train mean return: %+.4f%%",
             len(X_train), len(X_val), y_train.mean() * 100)

    # 6. Train XGBRegressor (v2)
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 7. Evaluate (regression metrics)
    val_preds = model.predict(X_val)
    val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    direction_acc = float(np.mean((val_preds > 0) == (y_val > 0)))
    log.info("Val RMSE: %.6f | Direction accuracy: %.3f", val_rmse, direction_acc)

    # Feature importance
    importances = model.feature_importances_
    feature_cols = get_expansion_feature_cols(symbol)
    top_idx = np.argsort(importances)[::-1][:5]
    log.info("Top 5 features: %s",
             ", ".join(f"{feature_cols[i]}={importances[i]:.3f}" for i in top_idx
                       if i < len(feature_cols)))

    # 8. Save model + config
    model_path = os.path.join(save_dir, f"{symbol}_xgb_expansion.joblib")
    config_path = os.path.join(save_dir, f"{symbol}_xgb_expansion_config.json")
    scaler_path = os.path.join(save_dir, f"{symbol}_xgb_expansion_scaler.json")

    joblib.dump(model, model_path)
    engine.save_scaler(scaler_path)

    config = {
        "symbol": symbol,
        "model_type": "xgboost_expansion",
        "model_version": "v2_regression",
        "target": f"forward_{FORWARD_DAYS}d_return",
        "val_rmse": round(val_rmse, 8),
        "val_direction_accuracy": round(direction_acc, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_names": get_expansion_feature_cols(symbol),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved expansion model → %s", model_path)
    return model


# ---------------------------------------------------------------------------
# Predictor (inference — compatible with ml_model.Predictor interface)
# ---------------------------------------------------------------------------
class ExpansionPredictor:
    """XGBoost factor-model predictor for expansion-group ETFs (v2 regression).

    Compatible with ml_model.Predictor.predict() interface.
    Requires SPY daily bars for relative momentum calculation.
    """

    model_type = "xgboost_expansion"
    eod_exit = False  # expansion holds for days/weeks

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol = symbol
        self.model_dir = model_dir
        self.engine = ExpansionFeatureEngine()
        self.model: Optional[xgb.XGBRegressor] = None
        self.threshold = CONFIDENCE_THRESHOLD
        self._spy_adapter = None  # lazy-loaded for SPY bars
        self._load()

    def _load(self) -> None:
        model_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_expansion.joblib")
        config_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_expansion_config.json")
        scaler_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_expansion_scaler.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained expansion model for {self.symbol}. "
                f"Run: python main.py train-expansion --symbols {self.symbol}"
            )

        self.model = joblib.load(model_path)
        self.engine.load_scaler(scaler_path)

        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.threshold = float(cfg.get("threshold", CONFIDENCE_THRESHOLD))

        log.info("Loaded expansion XGBoost for %s.", self.symbol)

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = 20) -> dict:
        """Produce regression prediction (v2).

        Note: seq_len is ignored (XGBoost uses point-in-time features).
        For relative momentum, fetches SPY bars via Yahoo (cached adapter).
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
                "direction": "FLAT",
                "probability": 0.5,
                "confidence": 0.0,
                "meta_confidence": 1.0,
                "tradeable": False,
            }

        features_norm = self.engine.transform(features)

        # Use last row (most recent bar)
        x = features_norm.iloc[-1:].values.astype(np.float32)
        expected_return = float(self.model.predict(x)[0])

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
        description="Train XGBoost factor model for expansion-group ETFs.",
    )
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols (e.g. EWJ,EWS,XLE,INDA)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--lookback", type=int, default=1000,
                        help="Daily bars to fetch (default: 1000)")
    args = parser.parse_args()

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    for sym in symbols:
        train_expansion_model(
            symbol=sym,
            adapter=adapter,
            fred_key=fred_key,
            lookback=args.lookback,
        )


if __name__ == "__main__":
    main()
