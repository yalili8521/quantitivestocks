#!/usr/bin/env python3
"""
Intraday Momentum Model (LightGBM) v2
=======================================
Regime-aware intraday momentum signal with LightGBM confidence filter.

Based on Zarattini et al. (2024): first-30-minute return predicts
rest-of-day return direction. Sharpe 1.33, 19.6% annualized on SPY 2007-2024.

v2 adds 8 regime-switching features:
  - ORB close position (where in range did first-30m end?)
  - Gap/direction agreement
  - VWAP zscore (how extended from intraday anchor)
  - RVOL × |ret| conviction score
  - VIX momentum regime flag (VIX>=18 AND ORB>=0.5% → momentum day)
  - 2-day prior return (autocorrelation)
  - ORB vs prev-day range (volatility expansion)
  - Consecutive direction fraction (trend context)

Strategy:
    1. Compute first 30-minute return (9:30-10:00 ET)
    2. LightGBM (24 features) predicts P(following that direction is profitable today)
    3. If P > threshold: enter in direction of first-30m return at ~10:00
    4. Exit at 15:30 (EOD) or via trailing stop

Usage (via main.py):
    python main.py train-intraday   --symbols SPY,QQQ,IWM,SOXX --provider alpaca
    python main.py train-intraday   --symbols SPY --provider alpaca --lookback 500
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from signals_engine import PROJECT_ROOT, build_adapter
from utils import _fetch_vix_for_training, DEFAULT_MODEL_DIR, INTRADAY_MODEL_DIR
OPTIONS_FLOW_FEATURES = ["pc_volume_ratio", "pc_oi_ratio", "vix_term_ratio", "vix_term_inverted"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("intraday_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIRST_30M_BARS = 6          # 6 five-minute bars = 30 minutes (9:30-10:00)
CONFIDENCE_THRESHOLD = 0.55  # LightGBM P > this to consider tradeable

FEATURE_NAMES = [
    "first_30m_ret",         # first 30 min return (signed)
    "first_30m_ret_abs",     # magnitude of first 30 min return
    "overnight_gap",         # (today open - yesterday close) / yesterday close
    "rvol",                  # relative volume: first 30m vol / 20-day avg
    "cumulative_delta",      # buy-sell volume imbalance in first 30m
    "ibs_prev",              # IBS of previous day: (close-low)/(high-low)
    "vix",                   # VIX level
    "vix_chg",               # VIX 1-day pct change
    "day_of_week",           # 0=Mon, 4=Fri
    "first_30m_range_pct",   # (first 30m high - low) / open  [= ORB width]
    "prev_day_ret",          # previous day return
    "prev_day_range",        # previous day (high-low) / close
    # Options flow features (market-wide sentiment)
    "pc_volume_ratio",       # aggregate put/call volume ratio
    "pc_oi_ratio",           # aggregate put/call open interest ratio
    "vix_term_ratio",        # VIX / VIX3M ratio
    "vix_term_inverted",     # 1 if VIX term structure inverted
    # --- Zarattini / regime-switching features (v2) ---
    "orb_close_position",    # where in ORB did first-30m end: (close-low)/(high-low) [0=bottom,1=top]
    "gap_direction_agree",   # +1 gap confirms first_30m dir, -1 contradicts, 0 flat
    "vwap_zscore_first30",   # (first-30m close - VWAP) / std over first-30m bars
    "rvol_quality",          # rvol * first_30m_ret_abs — conviction score
    "vix_momentum_regime",   # 1 if VIX>=18 AND orb_width>=0.5% (Zarattini momentum flag)
    "prev_2d_ret",           # 2-day prior return (autocorrelation context)
    "orb_vs_prev_range",     # today ORB width / yesterday range (volatility expansion)
    "consecutive_dir",       # fraction of last 5 days moving same dir as first_30m
]


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------
def _to_et(ts_series: pd.Series) -> pd.Series:
    """Convert timestamp series to Eastern Time."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    ts = pd.to_datetime(ts_series)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(et)


def _filter_regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-hours bars (9:30-16:00 ET)."""
    et_times = _to_et(bars["ts"])
    mask = (
        (et_times.dt.time >= datetime.time(9, 30))
        & (et_times.dt.time < datetime.time(16, 0))
    )
    return bars[mask].copy()


def _get_current_et_time() -> datetime.time:
    """Return current Eastern Time as time object."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    return datetime.datetime.now(et).time()


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
class IntradayFeatureEngine:
    """Build per-day feature vectors from 5-min intraday bars."""

    def __init__(self):
        self._options_engine = None
        self._vts_cache: Optional[pd.DataFrame] = None

    def build_training_data(
        self,
        intraday_bars: pd.DataFrame,
        vix_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build training dataset: one row per trading day.

        Features computed from the first 30 minutes + daily context.
        Label: did following first_30m direction profit by EOD?
        """
        bars = _filter_regular_hours(intraday_bars)
        if bars.empty:
            return pd.DataFrame()

        # Add ET date column for grouping
        et_ts = _to_et(bars["ts"])
        bars = bars.copy()
        bars["et_date"] = et_ts.dt.date
        bars["et_time"] = et_ts.dt.time

        # Build VIX lookup
        vix_map = self._build_vix_map(vix_df)

        self._vts_cache = None  # options_flow module removed; features default to NaN

        # Group by trading day
        dates = sorted(bars["et_date"].unique())
        samples = []

        for i, day in enumerate(dates):
            day_bars = bars[bars["et_date"] == day].sort_values("ts")
            if len(day_bars) < FIRST_30M_BARS + 6:
                continue  # Need at least first 30m + some rest-of-day

            # First 30 minutes (first 6 five-min bars at 9:30-10:00)
            first_30m = day_bars.head(FIRST_30M_BARS)
            if len(first_30m) < FIRST_30M_BARS:
                continue

            first_open = float(first_30m["open"].iloc[0])
            first_close = float(first_30m["close"].iloc[-1])
            if first_open <= 0:
                continue
            first_30m_ret = (first_close - first_open) / first_open

            # Rest of day (after first 30 min)
            rest = day_bars.iloc[FIRST_30M_BARS:]
            if rest.empty:
                continue
            eod_close = float(day_bars["close"].iloc[-1])
            rest_ret = (eod_close - first_close) / first_close if first_close > 0 else 0

            # Label: did following first_30m direction produce profit by EOD (net of costs)?
            if abs(first_30m_ret) < 1e-6:
                continue  # near-zero signal, skip
            # Cost-adjusted label: subtract estimated round-trip cost before checking profitability
            try:
                from cost_model import get_symbol_costs
                symbol_name = str(intraday_bars.get("symbol", pd.Series(["UNKNOWN"])).iloc[0])
                rt_cost = get_symbol_costs(symbol_name).round_trip_pct
            except (ImportError, Exception):
                rt_cost = 0.0007  # ~7bps default for mid-liquidity ETFs
            net_rest_ret = abs(rest_ret) - rt_cost
            label = 1 if net_rest_ret > 0 and (
                (first_30m_ret > 0 and rest_ret > 0) or
                (first_30m_ret < 0 and rest_ret < 0)
            ) else 0

            # Build features
            features = self._compute_day_features(
                day_bars, first_30m, first_30m_ret, first_open,
                dates, i, bars, vix_map, day,
            )
            features["label"] = label
            features["date"] = day
            features["rest_ret"] = rest_ret  # for analysis, not used as feature
            features["entry_price"] = first_close   # ~10:00 entry price
            features["eod_close"] = eod_close       # ~15:30 exit price
            features["first_30m_ret_raw"] = first_30m_ret  # signed (for direction)
            samples.append(features)

        return pd.DataFrame(samples)

    def build_live_features(
        self,
        intraday_bars: pd.DataFrame,
        vix_df: pd.DataFrame,
    ) -> Optional[Dict]:
        """Build feature vector for today's prediction (after 10:00 AM ET).

        Returns None if first 30 min not yet complete.
        Returns dict with features + first_30m_ret (for direction signal).
        """
        bars = _filter_regular_hours(intraday_bars)
        if bars.empty:
            return None

        et_ts = _to_et(bars["ts"])
        bars = bars.copy()
        bars["et_date"] = et_ts.dt.date
        bars["et_time"] = et_ts.dt.time

        dates = sorted(bars["et_date"].unique())
        if not dates:
            return None

        today = dates[-1]
        today_bars = bars[bars["et_date"] == today].sort_values("ts")

        # Need at least 6 bars (first 30 min complete)
        if len(today_bars) < FIRST_30M_BARS:
            return None

        first_30m = today_bars.head(FIRST_30M_BARS)
        first_open = float(first_30m["open"].iloc[0])
        first_close = float(first_30m["close"].iloc[-1])
        if first_open <= 0:
            return None
        first_30m_ret = (first_close - first_open) / first_open

        if abs(first_30m_ret) < 1e-6:
            return None  # no signal

        vix_map = self._build_vix_map(vix_df)
        i = len(dates) - 1

        features = self._compute_day_features(
            today_bars, first_30m, first_30m_ret, first_open,
            dates, i, bars, vix_map, today,
        )

        features["first_30m_ret_direction"] = first_30m_ret
        return features

    def _compute_day_features(
        self, day_bars, first_30m, first_30m_ret, first_open,
        dates, day_idx, all_bars, vix_map, day,
    ) -> Dict:
        """Compute feature vector for a single day."""
        features = {}

        # Derive first_close from first_30m (used by multiple feature blocks)
        first_close = float(first_30m["close"].iloc[-1])

        # 1. First 30m return features
        features["first_30m_ret"] = first_30m_ret
        features["first_30m_ret_abs"] = abs(first_30m_ret)

        # 2. Overnight gap
        if day_idx > 0:
            prev_day = dates[day_idx - 1]
            prev_bars = all_bars[all_bars["et_date"] == prev_day]
            if not prev_bars.empty:
                prev_close = float(prev_bars["close"].iloc[-1])
                features["overnight_gap"] = (first_open - prev_close) / prev_close if prev_close > 0 else 0
            else:
                features["overnight_gap"] = 0
        else:
            features["overnight_gap"] = 0

        # 3. RVOL (relative volume in first 30m vs historical average)
        first_30m_vol = float(first_30m["volume"].sum())
        hist_vols = []
        for d in dates[max(0, day_idx - 20):day_idx]:
            d_bars = all_bars[all_bars["et_date"] == d].head(FIRST_30M_BARS)
            if len(d_bars) >= FIRST_30M_BARS:
                hist_vols.append(float(d_bars["volume"].sum()))
        avg_vol = np.mean(hist_vols) if hist_vols else first_30m_vol
        features["rvol"] = first_30m_vol / max(avg_vol, 1)

        # 4. Cumulative delta (buy-sell imbalance in first 30m)
        hl = (first_30m["high"].astype(float) - first_30m["low"].astype(float)).replace(0, np.nan)
        buy_frac = (first_30m["close"].astype(float) - first_30m["low"].astype(float)) / hl
        buy_frac = buy_frac.fillna(0.5)
        vol = first_30m["volume"].astype(float)
        buy_vol = float((buy_frac * vol).sum())
        sell_vol = float(((1 - buy_frac) * vol).sum())
        total_vol = buy_vol + sell_vol
        features["cumulative_delta"] = (buy_vol - sell_vol) / max(total_vol, 1)

        # 5. IBS of previous day
        if day_idx > 0:
            prev_day = dates[day_idx - 1]
            prev_bars = all_bars[all_bars["et_date"] == prev_day]
            if not prev_bars.empty:
                ph = float(prev_bars["high"].max())
                pl = float(prev_bars["low"].min())
                pc = float(prev_bars["close"].iloc[-1])
                features["ibs_prev"] = (pc - pl) / max(ph - pl, 0.01)
            else:
                features["ibs_prev"] = 0.5
        else:
            features["ibs_prev"] = 0.5

        # 6. VIX
        vix_val = vix_map.get(day)
        if vix_val is None:
            # Try recent days
            for d in reversed(dates[:day_idx]):
                vix_val = vix_map.get(d)
                if vix_val is not None:
                    break
        features["vix"] = vix_val if vix_val is not None else 20.0

        # 7. VIX change
        prev_vix = None
        if day_idx > 0:
            for d in reversed(dates[:day_idx]):
                prev_vix = vix_map.get(d)
                if prev_vix is not None:
                    break
        if prev_vix and prev_vix > 0:
            features["vix_chg"] = (features["vix"] - prev_vix) / prev_vix
        else:
            features["vix_chg"] = 0.0

        # 8. Day of week
        if hasattr(day, "weekday"):
            features["day_of_week"] = day.weekday()
        else:
            features["day_of_week"] = pd.Timestamp(day).weekday()

        # 9. First 30m range as pct of open
        f30_high = float(first_30m["high"].max())
        f30_low = float(first_30m["low"].min())
        features["first_30m_range_pct"] = (f30_high - f30_low) / first_open if first_open > 0 else 0

        # 10. Previous day return
        if day_idx > 0:
            prev_day = dates[day_idx - 1]
            prev_bars = all_bars[all_bars["et_date"] == prev_day]
            if len(prev_bars) > 1:
                po = float(prev_bars["open"].iloc[0])
                pc = float(prev_bars["close"].iloc[-1])
                features["prev_day_ret"] = (pc - po) / po if po > 0 else 0
            else:
                features["prev_day_ret"] = 0
        else:
            features["prev_day_ret"] = 0

        # 11. Previous day range
        if day_idx > 0:
            prev_day = dates[day_idx - 1]
            prev_bars = all_bars[all_bars["et_date"] == prev_day]
            if not prev_bars.empty:
                ph = float(prev_bars["high"].max())
                pl = float(prev_bars["low"].min())
                pc = float(prev_bars["close"].iloc[-1])
                features["prev_day_range"] = (ph - pl) / pc if pc > 0 else 0
            else:
                features["prev_day_range"] = 0
        else:
            features["prev_day_range"] = 0

        # 13-20. Zarattini / regime-switching features (v2)

        # 13. ORB close position: where in first-30m range did we end?
        orb_range = f30_high - f30_low
        if orb_range > 0:
            features["orb_close_position"] = (first_close - f30_low) / orb_range
        else:
            features["orb_close_position"] = 0.5

        # 14. Gap direction agree: does overnight gap confirm first-30m direction?
        gap = features["overnight_gap"]
        if abs(gap) < 0.0005:
            features["gap_direction_agree"] = 0.0
        elif (gap > 0 and first_30m_ret > 0) or (gap < 0 and first_30m_ret < 0):
            features["gap_direction_agree"] = 1.0
        else:
            features["gap_direction_agree"] = -1.0

        # 15. VWAP zscore over first 30m bars
        f30_closes = first_30m["close"].astype(float).values
        f30_vols = first_30m["volume"].astype(float).values
        total_vol_30m = float(f30_vols.sum())
        if total_vol_30m > 0:
            vwap_30m = float((f30_closes * f30_vols).sum()) / total_vol_30m
        else:
            vwap_30m = float(np.mean(f30_closes))
        std_30m = float(np.std(f30_closes)) if len(f30_closes) > 1 else 1.0
        features["vwap_zscore_first30"] = (first_close - vwap_30m) / max(std_30m, 1e-6)

        # 16. RVOL quality: conviction = volume surprise × price move
        features["rvol_quality"] = features["rvol"] * abs(first_30m_ret)

        # 17. VIX momentum regime: Zarattini flag (wide ORB + elevated VIX = momentum day)
        orb_width = features["first_30m_range_pct"]
        vix_val_now = features["vix"]
        features["vix_momentum_regime"] = float(vix_val_now >= 18.0 and orb_width >= 0.005)

        # 18. 2-day prior return
        if day_idx >= 2:
            two_back = dates[day_idx - 2]
            two_back_bars = all_bars[all_bars["et_date"] == two_back]
            if len(two_back_bars) > 1:
                po2 = float(two_back_bars["open"].iloc[0])
                pc2 = float(two_back_bars["close"].iloc[-1])
                features["prev_2d_ret"] = (pc2 - po2) / po2 if po2 > 0 else 0.0
            else:
                features["prev_2d_ret"] = 0.0
        else:
            features["prev_2d_ret"] = 0.0

        # 19. ORB width vs previous day range (volatility expansion)
        prev_rng = features["prev_day_range"]
        features["orb_vs_prev_range"] = orb_width / max(prev_rng, 1e-6)

        # 20. Consecutive direction: fraction of last 5 days moving in same dir as first_30m
        direction_sign = 1 if first_30m_ret > 0 else -1
        consecutive = 0
        n_check = 0
        for d in reversed(dates[max(0, day_idx - 5):day_idx]):
            d_bars = all_bars[all_bars["et_date"] == d]
            if len(d_bars) > 1:
                d_o = float(d_bars["open"].iloc[0])
                d_c = float(d_bars["close"].iloc[-1])
                d_ret = (d_c - d_o) / d_o if d_o > 0 else 0.0
                if (d_ret > 0) == (direction_sign > 0):
                    consecutive += 1
                n_check += 1
        features["consecutive_dir"] = consecutive / max(n_check, 1)

        # 12. Options flow features (market-wide sentiment)
        # For training: use historical VIX term structure, NaN for P/C (not available historically)
        # For live: OptionsFlowEngine provides real-time data
        if self._vts_cache is not None and not self._vts_cache.empty:
            vts_map = dict(zip(self._vts_cache["date"].values,
                              zip(self._vts_cache["vix_term_ratio"].values,
                                  self._vts_cache["vix_term_inverted"].values)))
            vts = vts_map.get(day, (np.nan, 0.0))
            features["vix_term_ratio"] = vts[0] if vts[0] == vts[0] else 1.0  # NaN check
            features["vix_term_inverted"] = vts[1]
        else:
            features["vix_term_ratio"] = 1.0   # default contango
            features["vix_term_inverted"] = 0.0
        features["pc_volume_ratio"] = 0.0   # not available historically; default neutral
        features["pc_oi_ratio"] = 0.0       # not available historically; default neutral

        return features

    @staticmethod
    def _build_vix_map(vix_df: pd.DataFrame) -> Dict:
        """Build date → vix value mapping."""
        vix_map = {}
        if vix_df is not None and not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_map[d] = float(row["vix"])
        return vix_map


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_intraday_model(
    symbol: str,
    adapter,
    fred_key: Optional[str] = None,
    lookback: int = 500,
    save_dir: str = INTRADAY_MODEL_DIR,
) -> Optional[lgb.LGBMClassifier]:
    """Train LightGBM model for intraday momentum filtering.

    1. Fetch lookback days of 5-min intraday bars
    2. Build per-day features (one sample per trading day)
    3. Label: did following first-30m direction profit by EOD?
    4. Train LightGBM with walk-forward split
    5. Save model + config
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info("=== Training intraday momentum model for %s ===", symbol)

    # 1. Fetch data
    log.info("Fetching %d days of 5-min intraday data for %s...", lookback, symbol)
    bars = adapter.fetch_intraday(symbol, "5min", lookback_days=lookback)
    log.info("Got %d bars for %s.", len(bars), symbol)

    vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500),
                                     include_live=False)
    log.info("Got %d VIX rows.", len(vix_df))

    # 2. Build training data
    engine = IntradayFeatureEngine()
    data = engine.build_training_data(bars, vix_df)
    log.info("Built %d daily samples for %s.", len(data), symbol)

    if len(data) < 30:
        log.error("Not enough training data for %s (%d days). Need at least 30.", symbol, len(data))
        return None

    # 3. Prepare X, y
    X = data[FEATURE_NAMES].values.astype(np.float32)
    y = data["label"].values.astype(np.int32)

    # Expanding-window walk-forward CV: estimate OOS performance across multiple folds
    n_folds = min(5, max(2, len(X) // 60))  # at least 60 samples per fold
    fold_size = len(X) // (n_folds + 1)     # reserve 1 fold-worth for initial train
    cv_accs = []
    for fold in range(n_folds):
        cv_train_end = fold_size * (fold + 1)
        cv_val_end = min(cv_train_end + fold_size, len(X))
        if cv_val_end <= cv_train_end:
            break
        cv_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, num_leaves=15,
            min_child_samples=15, learning_rate=0.04,
            reg_lambda=2.0, reg_alpha=0.2, subsample=0.8,
            colsample_bytree=0.7, random_state=42, verbose=-1,
        )
        cv_model.fit(X[:cv_train_end], y[:cv_train_end],
                     eval_set=[(X[cv_train_end:cv_val_end], y[cv_train_end:cv_val_end])],
                     callbacks=[lgb.early_stopping(stopping_rounds=20)])
        cv_preds = cv_model.predict(X[cv_train_end:cv_val_end])
        cv_acc = float(np.mean(cv_preds == y[cv_train_end:cv_val_end]))
        cv_accs.append(cv_acc)
    if cv_accs:
        log.info("Expanding-window CV (%d folds): acc=%.3f ± %.3f",
                 len(cv_accs), np.mean(cv_accs), np.std(cv_accs))

    # Final model: first 80% train, last 20% validate
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    log.info("Train: %d samples, Val: %d samples. Train label balance: %.1f%% UP.",
             len(X_train), len(X_val), y_train.mean() * 100)

    # 4. Train LightGBM (24 features; tighter regularization to prevent overfit)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        num_leaves=15,
        min_child_samples=15,
        learning_rate=0.04,
        reg_lambda=2.0,
        reg_alpha=0.2,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=50), lgb.early_stopping(stopping_rounds=30)],
    )

    # 5. Evaluate
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_acc = float(np.mean(val_preds == y_val))

    # Threshold calibration: find best threshold for precision-coverage balance
    best_t, best_score = CONFIDENCE_THRESHOLD, 0.0
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
    top_idx = np.argsort(importances)[::-1][:8]
    log.info("Top 8 features: %s",
             ", ".join(f"{FEATURE_NAMES[i]}={importances[i]}" for i in top_idx))

    # 6. Save model + config
    model_path = os.path.join(save_dir, f"{symbol}_lgb_intraday.joblib")
    config_path = os.path.join(save_dir, f"{symbol}_lgb_intraday_config.json")

    joblib.dump(model, model_path)
    config = {
        "symbol": symbol,
        "model_type": "lgb_intraday",
        "horizon": "1d",
        "threshold": best_t,
        "val_accuracy": round(val_acc, 4),
        "val_base_rate": round(base_rate, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_names": FEATURE_NAMES,
        "cost_adjusted_labels": True,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved intraday model → %s", model_path)
    log.info("Saved config → %s", config_path)
    if val_acc >= 0.52:
        try:
            from model_monitor import ModelMonitor
            ModelMonitor().clear_model_pause(symbol, reason="retrained_intraday_model")
        except Exception as exc:
            log.debug("Model pause clear skipped for %s: %s", symbol, exc)
    else:
        log.info("Pause state retained for %s: val_acc=%.3f < 0.52", symbol, val_acc)
    return model


# ---------------------------------------------------------------------------
# Predictor (inference — compatible with ml_model.Predictor interface)
# ---------------------------------------------------------------------------
class IntradayPredictor:
    """Intraday momentum predictor with LightGBM confidence filter.

    Compatible with ml_model.Predictor.predict() interface.
    Returns the same dict format: {direction, probability, confidence,
    meta_confidence, tradeable}.

    Attributes:
        model_type: "intraday_momentum" — used by paper_trader for EOD exit
        eod_exit: True — paper_trader forces position close at 15:30
    """

    model_type = "intraday_momentum"
    eod_exit = True

    def __init__(self, symbol: str, model_dir: str = DEFAULT_MODEL_DIR):
        self.symbol = symbol
        self.model_dir = model_dir
        self.engine = IntradayFeatureEngine()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.threshold = CONFIDENCE_THRESHOLD
        self._load()

    def _load(self) -> None:
        model_path = os.path.join(self.model_dir, f"{self.symbol}_lgb_intraday.joblib")
        config_path = os.path.join(self.model_dir, f"{self.symbol}_lgb_intraday_config.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained intraday model for {self.symbol}. "
                f"Run: python main.py train-intraday --symbols {self.symbol} --provider alpaca"
            )

        self.model = joblib.load(model_path)
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.threshold = float(cfg.get("threshold", CONFIDENCE_THRESHOLD))
        log.info("Loaded intraday LightGBM for %s (threshold=%.3f).", self.symbol, self.threshold)

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame,
                seq_len: int = 20) -> dict:
        """Produce prediction from latest intraday bars.

        Returns standard prediction dict:
            direction:       "UP" / "DOWN" based on first-30-min return
            probability:     LightGBM P(following direction works)
            confidence:      (probability - 0.5) * 2 if > 0.5, else 0
            meta_confidence: 1.0 (no meta model; LightGBM IS the filter)
            tradeable:       True if LightGBM prob > threshold and time is right
        """
        # Check if current time allows prediction (after first 30 min)
        current_time = _get_current_et_time()
        if current_time < datetime.time(10, 0):
            return {
                "direction": "UNKNOWN",
                "probability": 0.5,
                "confidence": 0.0,
                "meta_confidence": 1.0,
                "tradeable": False,
            }

        # Build features from today's bars
        features = self.engine.build_live_features(bars_df, vix_df)
        if features is None:
            return {
                "direction": "UNKNOWN",
                "probability": 0.5,
                "confidence": 0.0,
                "meta_confidence": 1.0,
                "tradeable": False,
            }

        # Direction from first-30-min momentum
        first_30m_ret = features.pop("first_30m_ret_direction")
        direction = "UP" if first_30m_ret > 0 else "DOWN"

        # LightGBM prediction
        x = pd.DataFrame([[features[f] for f in FEATURE_NAMES]],
                         columns=FEATURE_NAMES)
        prob = float(self.model.predict_proba(x)[0][1])

        # Confidence in LSTM-compatible format
        confidence = max(0.0, (prob - 0.5) * 2)

        # Bridge classification prob → expected_return for paper_trader entry gate
        # prob=0.5 → E[r]=0 (coin flip), prob=0.6 → |E[r]|=0.003, prob=0.7 → |E[r]|=0.006
        sign = 1 if direction == "UP" else -1
        expected_return = sign * (prob - 0.5) * 0.03

        return {
            "direction": direction,
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "expected_return": round(expected_return, 6),
            "meta_confidence": 1.0,
            "tradeable": prob >= self.threshold,
        }


# ---------------------------------------------------------------------------
# CLI (standalone training)
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train intraday momentum LightGBM model.",
    )
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols (e.g. SPY,QQQ,IWM,SOXX)")
    parser.add_argument("--provider", default="alpaca",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--lookback", type=int, default=500,
                        help="Calendar days of intraday data to fetch (default: 500)")
    args = parser.parse_args()

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    for sym in symbols:
        train_intraday_model(
            symbol=sym,
            adapter=adapter,
            fred_key=fred_key,
            lookback=args.lookback,
        )


if __name__ == "__main__":
    main()
