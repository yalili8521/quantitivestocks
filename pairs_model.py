#!/usr/bin/env python3
"""
Pairs / Cointegration Mean-Reversion Model
====================================
XGBoost classifier for mean-reverting pairs trades.

Activated when primary momentum models skip due to Hurst exponent < 0.45,
capturing returns in regimes where momentum strategies explicitly avoid trading.

Strategy overview:
    1. Define cointegrated ETF pairs (SPY-QQQ, GLD-SLV, etc.)
    2. Compute spread z-score using Engle-Granger cointegration
    3. XGBoost classifies whether spread will revert from current z-score
    4. Trade the spread: go long laggard / short leader when |z| > threshold

Pairs:
    SPY ↔ QQQ   (US large cap ↔ tech)
    GLD ↔ SLV   (gold ↔ silver)
    EWT ↔ EEM   (Taiwan ↔ EM)
    EWJ ↔ INDA  (Japan ↔ India)
    XLE ↔ SPY   (energy ↔ market)
    EWS ↔ EEM   (Singapore ↔ EM)

Usage (via main.py):
    python main.py train-pairs --symbols SPY,QQQ,GLD,SLV
    python main.py backtest --symbol SPY --start 2024-01-01 --model pairs
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from signals_engine import PROJECT_ROOT, build_adapter, compute_hurst_exponent
from ml_model import _fetch_vix_for_training, DEFAULT_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pairs_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAIRS_MAP: Dict[str, str] = {
    "SPY": "QQQ",
    "QQQ": "SPY",
    "GLD": "SLV",
    "SLV": "GLD",
    "EWT": "EEM",
    "EEM": "EWT",
    "EWJ": "INDA",
    "INDA": "EWJ",
    "XLE": "SPY",
    "EWS": "EEM",
}

PAIRS_FEATURES = [
    "z_score",            # spread z-score (current deviation)
    "z_score_velocity",   # z-score change over 5 bars
    "half_life",          # estimated mean-reversion half-life (Ornstein-Uhlenbeck)
    "coint_pvalue",       # cointegration p-value (lower = stronger relationship)
    "spread_vol",         # rolling 20-day spread volatility
    "spread_skew",        # rolling 60-day spread skewness
    "hurst_a",            # Hurst exponent of symbol A
    "hurst_b",            # Hurst exponent of symbol B
]

# Triple-barrier for spread mean-reversion
TB_PT_PCT = 0.015     # 1.5% profit target on spread
TB_SL_PCT = 0.010     # 1.0% stop loss on spread
TB_HORIZON = 5        # 5-bar look-forward

CONFIDENCE_THRESHOLD = 0.55
ZSCORE_ENTRY_THRESHOLD = 1.5   # enter when |z| > 1.5
ZSCORE_EXIT_THRESHOLD = 0.5    # exit when |z| < 0.5

ROLLING_COINT_WINDOW = 120     # rolling cointegration window (trading days)
ZSCORE_WINDOW = 20             # z-score lookback


# ---------------------------------------------------------------------------
# Statistical Functions
# ---------------------------------------------------------------------------
def engle_granger_coint(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Run Engle-Granger cointegration test.

    Returns (hedge_ratio, p_value).
    Lower p_value = stronger cointegration.
    """
    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tsa.stattools import adfuller
        import statsmodels.api as sm

        # Step 1: OLS regression y = a + b*x
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        hedge_ratio = float(model.params[1])

        # Step 2: ADF test on residuals
        residuals = model.resid
        adf_result = adfuller(residuals, maxlag=1, regression="c")
        p_value = float(adf_result[1])

        return hedge_ratio, p_value

    except Exception as exc:
        log.warning("Cointegration test failed: %s", exc)
        return 1.0, 1.0


def compute_spread(y: np.ndarray, x: np.ndarray, hedge_ratio: float) -> np.ndarray:
    """Compute cointegration spread: y - hedge_ratio * x."""
    return y - hedge_ratio * x


def compute_zscore(spread: np.ndarray, window: int = ZSCORE_WINDOW) -> np.ndarray:
    """Compute rolling z-score of spread."""
    s = pd.Series(spread)
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return ((s - mean) / std).values


def compute_half_life(spread: np.ndarray) -> float:
    """Estimate Ornstein-Uhlenbeck mean-reversion half-life.

    Uses AR(1) regression: spread_t = c + phi * spread_{t-1} + epsilon
    Half-life = -log(2) / log(phi)
    """
    try:
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm

        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        X = sm.add_constant(spread_lag)
        model = OLS(spread_diff, X).fit()
        phi = float(model.params[1])

        if phi >= 0:
            return 999.0  # not mean-reverting
        half_life = -np.log(2) / phi
        return float(np.clip(half_life, 1.0, 500.0))

    except Exception:
        return 999.0


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
class PairsFeatureEngine:
    """Build features for pairs/cointegration model.

    Computes rolling cointegration, spread z-score, half-life,
    and Hurst exponents for both legs.
    """

    def __init__(self):
        self._scaler_params: Optional[dict] = None

    def build_features(
        self,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
        symbol_a: str,
        symbol_b: str,
    ) -> pd.DataFrame:
        """Build pairs feature matrix.

        Args:
            bars_a: OHLCV for symbol A (the one being traded)
            bars_b: OHLCV for symbol B (the hedge leg)
            symbol_a, symbol_b: Symbol names for logging

        Returns:
            DataFrame with PAIRS_FEATURES columns.
        """
        close_a = bars_a["close"].astype(float).values
        close_b = bars_b["close"].astype(float).values

        # Align by date
        dates_a = pd.to_datetime(bars_a["ts"]).dt.date
        dates_b = pd.to_datetime(bars_b["ts"]).dt.date

        # Create date-aligned close series
        map_b = dict(zip(dates_b.values, close_b))
        aligned_b = np.array([map_b.get(d, np.nan) for d in dates_a.values])

        # Build aligned DataFrame
        n = len(close_a)
        df = pd.DataFrame(index=bars_a.index)

        # Rolling cointegration + z-score
        z_scores = np.full(n, np.nan)
        z_velocity = np.full(n, np.nan)
        half_lives = np.full(n, np.nan)
        coint_pvalues = np.full(n, np.nan)
        spread_vols = np.full(n, np.nan)
        spread_skews = np.full(n, np.nan)

        for i in range(ROLLING_COINT_WINDOW, n):
            window_a = close_a[i - ROLLING_COINT_WINDOW:i]
            window_b = aligned_b[i - ROLLING_COINT_WINDOW:i]

            # Skip if any NaN in window
            valid = ~(np.isnan(window_a) | np.isnan(window_b))
            if valid.sum() < ROLLING_COINT_WINDOW // 2:
                continue

            wa = window_a[valid]
            wb = window_b[valid]

            hedge_ratio, p_value = engle_granger_coint(wa, wb)
            spread = compute_spread(wa, wb, hedge_ratio)

            # Z-score of current point
            z = compute_zscore(spread, ZSCORE_WINDOW)
            z_scores[i] = z[-1] if len(z) > 0 and not np.isnan(z[-1]) else 0.0

            # Z-score velocity (5-bar change)
            if len(z) >= 6:
                z_velocity[i] = z[-1] - z[-6]

            # Half-life
            if len(spread) >= 20:
                half_lives[i] = compute_half_life(spread)

            coint_pvalues[i] = p_value

            # Spread volatility
            if len(spread) >= ZSCORE_WINDOW:
                spread_vols[i] = float(np.std(spread[-ZSCORE_WINDOW:]))

            # Spread skewness
            if len(spread) >= 60:
                from scipy.stats import skew
                spread_skews[i] = float(skew(spread[-60:]))

        df["z_score"] = z_scores
        df["z_score_velocity"] = z_velocity
        df["half_life"] = half_lives
        df["coint_pvalue"] = coint_pvalues
        df["spread_vol"] = spread_vols
        df["spread_skew"] = spread_skews

        # Hurst exponents (rolling 60-day)
        hurst_a = np.full(n, np.nan)
        hurst_b = np.full(n, np.nan)
        for i in range(60, n):
            try:
                hurst_a[i] = compute_hurst_exponent(
                    pd.Series(close_a[i - 60:i]))
            except Exception:
                pass
            try:
                b_window = aligned_b[i - 60:i]
                valid_b = b_window[~np.isnan(b_window)]
                if len(valid_b) >= 30:
                    hurst_b[i] = compute_hurst_exponent(
                        pd.Series(valid_b))
            except Exception:
                pass

        df["hurst_a"] = hurst_a
        df["hurst_b"] = hurst_b

        # Drop NaN warmup
        df = df.dropna(subset=PAIRS_FEATURES)
        return df[PAIRS_FEATURES]

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
def _prepare_pairs_labels(
    features_df: pd.DataFrame,
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_ratio: float,
    bar_positions: np.ndarray,
    pt_pct: float = TB_PT_PCT,
    sl_pct: float = TB_SL_PCT,
    horizon: int = TB_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triple-barrier labeling on spread for pairs trading.

    Label = 1 if spread reverts toward zero (mean-reversion profitable).
    """
    feature_values = features_df.values
    n_bars = len(close_a)

    X_list = []
    y_list = []

    for i in range(len(feature_values)):
        bar_pos = bar_positions[i]
        if bar_pos < 0 or bar_pos + horizon >= n_bars:
            continue

        # Current spread level
        entry_spread = close_a[bar_pos] - hedge_ratio * close_b[bar_pos]
        spread_sign = np.sign(entry_spread)  # which direction should revert

        label = None
        for fwd in range(1, horizon + 1):
            fwd_pos = bar_pos + fwd
            if fwd_pos >= n_bars:
                break
            fwd_spread = close_a[fwd_pos] - hedge_ratio * close_b[fwd_pos]
            spread_change = (entry_spread - fwd_spread)  # reversion = change toward zero

            # Normalize by entry spread magnitude
            if abs(entry_spread) < 1e-6:
                continue
            reversion_pct = spread_change / abs(entry_spread)

            if reversion_pct >= pt_pct:
                label = 1.0  # spread reverted (profitable)
                break
            elif reversion_pct <= -sl_pct:
                label = 0.0  # spread diverged (stop loss)
                break

        if label is None:
            # At end of horizon: did spread move toward zero?
            final_pos = min(bar_pos + horizon, n_bars - 1)
            final_spread = close_a[final_pos] - hedge_ratio * close_b[final_pos]
            label = 1.0 if abs(final_spread) < abs(entry_spread) else 0.0

        X_list.append(feature_values[i])
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_pairs_model(
    symbol_a: str,
    symbol_b: str,
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 1000,
    save_dir: str = DEFAULT_MODEL_DIR,
) -> Optional[xgb.XGBClassifier]:
    """Train XGBoost pairs/cointegration model.

    1. Fetch daily data for both legs
    2. Build pairs features (z-score, half-life, Hurst, etc.)
    3. Triple-barrier labeling on spread
    4. Train XGBoost with walk-forward split
    5. Save model + scaler
    """
    os.makedirs(save_dir, exist_ok=True)
    pair_name = f"{symbol_a}_{symbol_b}"
    log.info("=== Training pairs model for %s ↔ %s ===", symbol_a, symbol_b)

    # 1. Fetch data
    log.info("Fetching %d daily bars for %s and %s...", lookback, symbol_a, symbol_b)
    bars_a = adapter.fetch_daily(symbol_a, lookback)
    bars_b = adapter.fetch_daily(symbol_b, lookback)
    log.info("Got %d + %d bars.", len(bars_a), len(bars_b))

    # 2. Build features
    engine = PairsFeatureEngine()
    features = engine.build_features(bars_a, bars_b, symbol_a, symbol_b)
    log.info("Built %d feature rows for %s.", len(features), pair_name)

    if len(features) < 50:
        log.error("Not enough data for %s (%d rows).", pair_name, len(features))
        return None

    # 3. Scaler
    split_idx = int(len(features) * 0.8)
    engine.fit_scaler(features.iloc[:split_idx])
    full_norm = engine.transform(features)

    # Compute hedge ratio for labeling (from training window)
    close_a = bars_a["close"].astype(float).values
    dates_a = pd.to_datetime(bars_a["ts"]).dt.date
    dates_b = pd.to_datetime(bars_b["ts"]).dt.date
    map_b = dict(zip(dates_b.values, bars_b["close"].astype(float).values))
    close_b_aligned = np.array([map_b.get(d, np.nan) for d in dates_a.values])

    # Use full-sample hedge ratio for labeling
    valid = ~(np.isnan(close_a) | np.isnan(close_b_aligned))
    hedge_ratio, p_value = engle_granger_coint(close_a[valid], close_b_aligned[valid])
    log.info("Full-sample hedge ratio: %.4f, coint p-value: %.4f", hedge_ratio, p_value)

    # 4. Triple-barrier labels
    bar_positions = bars_a.index.get_indexer(full_norm.index)
    X_all, y_all = _prepare_pairs_labels(
        full_norm, close_a, close_b_aligned, hedge_ratio, bar_positions,
    )
    log.info("Labeled %d samples. Reversion rate: %.1f%%", len(y_all), y_all.mean() * 100)

    # 5. Walk-forward split
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    log.info("Train: %d, Val: %d.", len(X_train), len(X_val))

    if len(X_val) < 10:
        log.error("Validation set too small for %s.", pair_name)
        return None

    # 6. Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=25,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 7. Evaluate
    val_probs = model.predict_proba(X_val)[:, 1]
    val_acc = float(np.mean((val_probs >= 0.5).astype(int) == y_val))

    # Threshold calibration
    best_t = CONFIDENCE_THRESHOLD
    best_score = 0.0
    base_rate = float(y_val.mean())
    for t in np.arange(0.45, 0.75, 0.025):
        mask = val_probs >= t
        cov = mask.mean()
        if cov < 0.10:
            break
        prec = float(y_val[mask].mean()) if mask.sum() > 0 else 0.0
        lift = max(0.0, prec - base_rate)
        score = lift * np.log1p(cov * 5)
        if score > best_score:
            best_score = score
            best_t = round(float(t), 3)

    log.info("Val accuracy: %.3f | Optimal threshold: %.3f", val_acc, best_t)

    # Feature importance
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    log.info("Top 5 features: %s",
             ", ".join(f"{PAIRS_FEATURES[i]}={importances[i]:.3f}" for i in top_idx
                       if i < len(PAIRS_FEATURES)))

    # 8. Save
    model_path = os.path.join(save_dir, f"{pair_name}_xgb_pairs.joblib")
    config_path = os.path.join(save_dir, f"{pair_name}_xgb_pairs_config.json")
    scaler_path = os.path.join(save_dir, f"{pair_name}_xgb_pairs_scaler.json")

    joblib.dump(model, model_path)
    engine.save_scaler(scaler_path)

    config = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "pair_name": pair_name,
        "model_type": "xgboost_pairs",
        "hedge_ratio": round(hedge_ratio, 6),
        "coint_pvalue": round(p_value, 6),
        "threshold": best_t,
        "val_accuracy": round(val_acc, 4),
        "feature_names": PAIRS_FEATURES,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved pairs model → %s", model_path)
    return model


# ---------------------------------------------------------------------------
# Predictor (compatible with paper_trader Predictor interface)
# ---------------------------------------------------------------------------
class PairsPredictor:
    """XGBoost pairs predictor for mean-reverting pair trades.

    Compatible with paper_trader Predictor interface.
    Returns prediction for the primary symbol (symbol_a);
    the hedge leg (symbol_b) is used to compute the spread.
    """

    model_type = "xgboost_pairs"
    eod_exit = False  # pairs hold for days

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol = symbol
        self.pair_symbol = PAIRS_MAP.get(symbol)
        self.model_dir = model_dir
        self.engine = PairsFeatureEngine()
        self.model: Optional[xgb.XGBClassifier] = None
        self.threshold = CONFIDENCE_THRESHOLD
        self.hedge_ratio = 1.0
        self._pair_adapter = None  # lazy-loaded
        self._load()

    def _load(self) -> None:
        if not self.pair_symbol:
            raise ValueError(f"No pair defined for {self.symbol}. "
                             f"Available: {list(PAIRS_MAP.keys())}")

        pair_name = f"{self.symbol}_{self.pair_symbol}"
        model_path = os.path.join(self.model_dir, f"{pair_name}_xgb_pairs.joblib")
        config_path = os.path.join(self.model_dir, f"{pair_name}_xgb_pairs_config.json")
        scaler_path = os.path.join(self.model_dir, f"{pair_name}_xgb_pairs_scaler.json")

        # Try reverse pair order
        if not os.path.exists(model_path):
            pair_name = f"{self.pair_symbol}_{self.symbol}"
            model_path = os.path.join(self.model_dir, f"{pair_name}_xgb_pairs.joblib")
            config_path = os.path.join(self.model_dir, f"{pair_name}_xgb_pairs_config.json")
            scaler_path = os.path.join(self.model_dir, f"{pair_name}_xgb_pairs_scaler.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained pairs model for {self.symbol}↔{self.pair_symbol}. "
                f"Run: python main.py train-pairs --symbols {self.symbol},{self.pair_symbol}"
            )

        self.model = joblib.load(model_path)
        self.engine.load_scaler(scaler_path)

        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.threshold = float(cfg.get("threshold", CONFIDENCE_THRESHOLD))
            self.hedge_ratio = float(cfg.get("hedge_ratio", 1.0))

        log.info("Loaded pairs model for %s↔%s (threshold=%.3f, hedge=%.4f).",
                 self.symbol, self.pair_symbol, self.threshold, self.hedge_ratio)

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = 20) -> dict:
        """Produce prediction from latest daily bars.

        Fetches pair symbol bars internally.
        """
        # Lazy-load pair adapter
        if self._pair_adapter is None:
            from signals_engine import YahooFinanceAdapter
            self._pair_adapter = YahooFinanceAdapter()

        try:
            pair_bars = self._pair_adapter.fetch_daily(self.pair_symbol, len(bars_df))
        except Exception as exc:
            log.warning("Failed to fetch %s bars: %s", self.pair_symbol, exc)
            return self._unknown_prediction()

        features = self.engine.build_features(
            bars_df, pair_bars, self.symbol, self.pair_symbol)

        if features.empty:
            return self._unknown_prediction()

        features_norm = self.engine.transform(features)
        x = features_norm.iloc[-1:].values.astype(np.float32)

        prob = float(self.model.predict_proba(x)[0][1])

        # Determine direction from z-score
        z_score = float(features["z_score"].iloc[-1])
        if z_score > ZSCORE_ENTRY_THRESHOLD:
            # Spread is high → expect reversion down → SHORT symbol A
            direction = "DOWN"
        elif z_score < -ZSCORE_ENTRY_THRESHOLD:
            # Spread is low → expect reversion up → LONG symbol A
            direction = "UP"
        elif abs(z_score) < ZSCORE_EXIT_THRESHOLD:
            # Near zero → no signal
            direction = "FLAT"
        else:
            # Between entry and exit thresholds: use model probability
            direction = "UP" if prob > 0.5 else "DOWN"

        confidence = max(0.0, (prob - 0.5) * 2)

        return {
            "direction": direction,
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "meta_confidence": 1.0,
            "tradeable": prob >= self.threshold and abs(z_score) > ZSCORE_ENTRY_THRESHOLD,
            "z_score": round(z_score, 4),
            "half_life": round(float(features["half_life"].iloc[-1]), 1),
        }

    @staticmethod
    def _unknown_prediction() -> dict:
        return {
            "direction": "UNKNOWN",
            "probability": 0.5,
            "confidence": 0.0,
            "meta_confidence": 1.0,
            "tradeable": False,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train XGBoost pairs/cointegration model.",
    )
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbol pairs (e.g. SPY,QQQ,GLD,SLV). "
                             "Each symbol is paired via PAIRS_MAP.")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--lookback", type=int, default=1000,
                        help="Daily bars to fetch (default: 1000)")
    args = parser.parse_args()

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Train each unique pair (avoid duplicates)
    trained_pairs = set()
    for sym in symbols:
        pair = PAIRS_MAP.get(sym)
        if not pair:
            log.warning("No pair defined for %s. Skipping.", sym)
            continue
        pair_key = tuple(sorted([sym, pair]))
        if pair_key in trained_pairs:
            continue
        trained_pairs.add(pair_key)

        train_pairs_model(
            symbol_a=sym,
            symbol_b=pair,
            adapter=adapter,
            fred_key=fred_key,
            lookback=args.lookback,
        )


if __name__ == "__main__":
    main()
