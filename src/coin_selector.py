#!/usr/bin/env python3
"""
Cross-Sectional Coin Selector (Layer 1)
========================================
LightGBM LambdaRank model that ranks crypto coins cross-sectionally.
Selects the top-K coins from a ~20-coin universe for Layer 2 per-coin
swing models to generate return forecasts.

Architecture:
  - Features are z-scored cross-sectionally (relative positioning, not absolute)
  - Label: forward 5-day risk-adjusted return (return / realized vol)
  - Model: LightGBM LambdaRank (optimizes NDCG@3,6)
  - Output: ranked list of coins + top-K selection

Usage (via main.py):
    python main.py train-selector              # train on full universe
    python main.py train-selector --top-k 4    # select top-4 instead of top-6
    python main.py rank-coins                  # print today's rankings
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from signals_engine import PROJECT_ROOT, compute_rsi, compute_adx, compute_atr
from utils import CRYPTO_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("coin_selector")

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Full candidate universe — filtered at runtime by liquidity
CRYPTO_UNIVERSE = [
    # Majors
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Large-cap alts
    "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "UNI-USD", "LTC-USD",
    # Mid-cap alts (higher vol, wider spread)
    "DOGE-USD", "RENDER-USD", "AAVE-USD", "CRV-USD",
    "SUSHI-USD", "ARB-USD", "FIL-USD",
    # Excluded (not tradeable on Alpaca):
    #   OP-USD — not listed, INJ-USD — not listed,
    #   NEAR-USD — inactive, APT-USD — inactive
]

# Minimum 20-day median daily dollar volume to qualify
MIN_DOLLAR_VOLUME = 5_000_000  # $5M

# Default top-K selection count
DEFAULT_TOP_K = 6

# Forward label horizon
LABEL_HORIZON = 5  # 5-day forward return for ranking label

# Model filename
SELECTOR_MODEL_FILE = "coin_selector_lgb.txt"
SELECTOR_CONFIG_FILE = "coin_selector_config.json"

# Cross-sectional feature names (z-scored across universe at each timestamp)
XS_FEATURES = [
    "momentum_7d",
    "momentum_30d",
    "reversal_1d",
    "vol_ratio",
    "volume_zscore",
    "btc_beta_resid",
    "rsi14_rank",
    "drawdown_from_ath",
    "spread_proxy",
    "ret63",
    "adx",
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_coin_daily(symbol: str, lookback_days: int = 1500) -> pd.DataFrame:
    """Fetch daily OHLCV for a crypto symbol via yfinance."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    cal_days = int(lookback_days * 1.5) + 30
    hist = ticker.history(period=f"{cal_days}d", interval="1d")
    if hist is None or hist.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": hist.index,
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values,
    })
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_universe_data(
    universe: List[str],
    lookback_days: int = 1500,
) -> Dict[str, pd.DataFrame]:
    """Fetch daily data for the full universe. Skips symbols with insufficient data."""
    data = {}
    for sym in universe:
        try:
            df = _fetch_coin_daily(sym, lookback_days)
            if len(df) < 100:
                log.warning("Skipping %s: only %d bars (need ≥100)", sym, len(df))
                continue
            data[sym] = df
            log.info("Fetched %s: %d bars (%s → %s)",
                     sym, len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date())
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", sym, exc)
    return data


# ---------------------------------------------------------------------------
# Cross-sectional feature computation
# ---------------------------------------------------------------------------

def compute_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-coin time-series features (before cross-sectional z-scoring)."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["date"] = df["date"]

    # Momentum signals
    feat["momentum_7d"] = close.pct_change(7)
    feat["momentum_30d"] = close.pct_change(30)
    feat["reversal_1d"] = -close.pct_change(1)  # sign-flipped for mean-reversion
    feat["ret63"] = close.pct_change(63)

    # Volatility
    ret_daily = close.pct_change()
    feat["vol_5d"] = ret_daily.rolling(5).std() * np.sqrt(365)
    feat["vol_30d"] = ret_daily.rolling(30).std() * np.sqrt(365)
    feat["vol_ratio"] = feat["vol_5d"] / feat["vol_30d"].replace(0, np.nan)

    # Volume z-score (20-day rolling)
    dollar_vol = close * volume
    dv_mean = dollar_vol.rolling(20).mean()
    dv_std = dollar_vol.rolling(20).std()
    feat["volume_zscore"] = (dollar_vol - dv_mean) / dv_std.replace(0, np.nan)

    # Dollar volume for liquidity filter
    feat["dollar_volume_20d"] = dollar_vol.rolling(20).median()

    # RSI
    feat["rsi14_rank"] = compute_rsi(close, 14)

    # ADX
    feat["adx"] = compute_adx(high, low, close, 14)

    # Drawdown from all-time high (rolling 365d max)
    rolling_high = close.rolling(365, min_periods=30).max()
    feat["drawdown_from_ath"] = (close - rolling_high) / rolling_high.replace(0, np.nan)

    # Spread proxy: (high - low) / close
    feat["spread_proxy"] = (high - low) / close.replace(0, np.nan)

    # BTC beta residual (placeholder — computed cross-sectionally in build_xs_panel)
    feat["btc_beta_resid"] = 0.0

    feat["close"] = close
    return feat


def build_xs_panel(
    universe_data: Dict[str, pd.DataFrame],
    btc_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a cross-sectional panel: (date, symbol, features, label).

    Features are z-scored cross-sectionally at each date.
    """
    # Compute per-coin features
    coin_features = {}
    for sym, df in universe_data.items():
        feat = compute_coin_features(df)
        feat["symbol"] = sym
        coin_features[sym] = feat

    if not coin_features:
        return pd.DataFrame()

    # Stack into panel
    panel = pd.concat(coin_features.values(), ignore_index=True)

    # Compute BTC beta residual
    if btc_data is None and "BTC-USD" in universe_data:
        btc_data = universe_data["BTC-USD"]
    if btc_data is not None:
        btc_close = btc_data.set_index("date")["close"].astype(float)
        btc_ret = btc_close.pct_change()
        for sym in universe_data:
            mask = panel["symbol"] == sym
            sym_dates = panel.loc[mask, "date"]
            sym_close = panel.loc[mask, "close"]
            sym_ret = sym_close.pct_change()
            # Rolling 30-day beta residual
            aligned_btc_ret = btc_ret.reindex(sym_dates).fillna(0).values
            sym_ret_vals = sym_ret.fillna(0).values
            resid = np.full(len(sym_ret_vals), np.nan)
            for i in range(30, len(sym_ret_vals)):
                window_sym = sym_ret_vals[i-30:i]
                window_btc = aligned_btc_ret[i-30:i]
                if np.std(window_btc) > 1e-10:
                    beta = np.cov(window_sym, window_btc)[0, 1] / np.var(window_btc)
                    resid[i] = window_sym[-1] - beta * window_btc[-1]
                else:
                    resid[i] = window_sym[-1]
            panel.loc[mask, "btc_beta_resid"] = resid

    # Liquidity filter: drop rows where 20d median dollar volume < threshold
    panel = panel[panel["dollar_volume_20d"] >= MIN_DOLLAR_VOLUME].copy()

    # Cross-sectional z-scoring (per date)
    for feat_col in XS_FEATURES:
        if feat_col not in panel.columns:
            continue
        grouped = panel.groupby("date")[feat_col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1.0)
        panel[feat_col] = (panel[feat_col] - mean) / std

    # Compute forward label: 5-day risk-adjusted return
    panel["fwd_5d_ret"] = np.nan
    panel["fwd_5d_vol"] = np.nan
    panel["label"] = np.nan
    for sym in panel["symbol"].unique():
        mask = panel["symbol"] == sym
        sym_close = panel.loc[mask, "close"].values
        sym_ret = np.diff(sym_close, prepend=np.nan) / np.concatenate([[np.nan], sym_close[:-1]])
        n = len(sym_close)
        fwd_ret = np.full(n, np.nan)
        fwd_vol = np.full(n, np.nan)
        for i in range(n - LABEL_HORIZON):
            fwd_ret[i] = sym_close[i + LABEL_HORIZON] / sym_close[i] - 1
            fwd_vol[i] = np.std(sym_ret[i+1:i+1+LABEL_HORIZON]) if i + 1 + LABEL_HORIZON <= n else np.nan
        panel.loc[mask, "fwd_5d_ret"] = fwd_ret
        panel.loc[mask, "fwd_5d_vol"] = fwd_vol
        panel.loc[mask, "label"] = fwd_ret / np.maximum(fwd_vol, 0.001)

    # Drop rows with NaN features or labels
    feat_and_label = XS_FEATURES + ["label"]
    panel = panel.dropna(subset=feat_and_label).reset_index(drop=True)

    return panel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_selector(
    universe: Optional[List[str]] = None,
    train_end: str = "2025-01-01",
    save_dir: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    lookback_days: int = 1500,
) -> dict:
    """Train the cross-sectional LightGBM LambdaRank selector.

    Returns dict with training metrics.
    """
    import lightgbm as lgb

    if universe is None:
        universe = list(CRYPTO_UNIVERSE)
    if save_dir is None:
        save_dir = CRYPTO_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    log.info("Fetching universe data for %d coins...", len(universe))
    data = fetch_universe_data(universe, lookback_days=lookback_days)
    if len(data) < 3:
        raise RuntimeError(f"Only {len(data)} coins have data — need ≥3 for cross-sectional ranking")

    log.info("Building cross-sectional panel...")
    panel = build_xs_panel(data)
    log.info("Panel: %d rows, %d unique dates, %d coins",
             len(panel), panel["date"].nunique(), panel["symbol"].nunique())

    # Train/val split by date
    train_end_dt = pd.Timestamp(train_end)
    train_mask = panel["date"] < train_end_dt
    val_mask = panel["date"] >= train_end_dt

    train_df = panel[train_mask].copy()
    val_df = panel[val_mask].copy()

    if len(train_df) < 100:
        raise RuntimeError(f"Training set too small: {len(train_df)} rows (need ≥100)")
    if len(val_df) < 50:
        log.warning("Validation set small: %d rows", len(val_df))

    log.info("Train: %d rows (%s → %s), Val: %d rows (%s → %s)",
             len(train_df), train_df["date"].min().date(), train_df["date"].max().date(),
             len(val_df), val_df["date"].min().date(), val_df["date"].max().date())

    # Build LightGBM datasets with group (query) structure
    # group = number of coins per date snapshot
    # LambdaRank requires integer relevance labels — discretize continuous
    # risk-adjusted returns into 5 grades (0=worst, 4=best) per date.
    def _discretize_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Convert continuous label to integer relevance grades 0-4 per date."""
        df = df.copy()
        df["label_int"] = 0
        for date, grp in df.groupby("date"):
            vals = grp["label"].values
            if len(vals) < 5:
                # Too few coins to quintile — use rank directly
                ranks = pd.Series(vals).rank(method="min").astype(int) - 1
                ranks = (ranks * 4 / max(ranks.max(), 1)).astype(int).clip(0, 4)
                df.loc[grp.index, "label_int"] = ranks.values
            else:
                pcts = pd.Series(vals).rank(pct=True)
                grades = (pcts * 5).clip(0, 4.999).astype(int)
                df.loc[grp.index, "label_int"] = grades.values
        return df

    train_df = _discretize_labels(train_df)
    val_df = _discretize_labels(val_df)

    def build_lgb_data(df: pd.DataFrame):
        X = df[XS_FEATURES].values
        y = df["label_int"].values.astype(int)
        groups = df.groupby("date").size().values
        return X, y, groups

    X_train, y_train, groups_train = build_lgb_data(train_df)
    X_val, y_val, groups_val = build_lgb_data(val_df)

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train,
                            feature_name=XS_FEATURES, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, group=groups_val,
                          reference=train_set, free_raw_data=False)

    # label_gain: relevance gain for each grade (0→0, 1→1, 2→3, 3→7, 4→15)
    # exponential gains emphasize correctly ranking the top coins
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, top_k],
        "label_gain": [0, 1, 3, 7, 15],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": 42,
    }

    log.info("Training LambdaRank model (top_k=%d)...", top_k)
    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )

    # Save model
    model_path = os.path.join(save_dir, SELECTOR_MODEL_FILE)
    model.save_model(model_path)
    log.info("Saved selector model to %s", model_path)

    # Evaluate: NDCG on validation set
    val_pred = model.predict(X_val)
    # Compute NDCG manually for logging
    ndcg_scores = _compute_ndcg_by_group(val_pred, y_val, groups_val, k=top_k)
    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    # Feature importance
    importance = dict(zip(XS_FEATURES, model.feature_importance(importance_type="gain").tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

    # Compute top-K hit rate: how often are the actual best coins in the predicted top-K?
    hit_rate = _compute_topk_hit_rate(val_pred, y_val, groups_val, k=top_k)

    metrics = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_dates": str(train_df["date"].min().date()) + " → " + str(train_df["date"].max().date()),
        "val_dates": str(val_df["date"].min().date()) + " → " + str(val_df["date"].max().date()),
        "coins": sorted(panel["symbol"].unique().tolist()),
        "n_coins": int(panel["symbol"].nunique()),
        "top_k": top_k,
        "mean_ndcg": round(mean_ndcg, 4),
        "topk_hit_rate": round(hit_rate, 4),
        "best_iteration": model.best_iteration,
        "feature_importance": dict(sorted_imp),
    }

    config_path = os.path.join(save_dir, SELECTOR_CONFIG_FILE)
    with open(config_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved selector config to %s", config_path)

    log.info("=== Selector Training Results ===")
    log.info("  NDCG@%d (val): %.4f", top_k, mean_ndcg)
    log.info("  Top-%d hit rate: %.1f%%", top_k, hit_rate * 100)
    log.info("  Best iteration: %d", model.best_iteration)
    log.info("  Feature importance:")
    for feat, imp in sorted_imp[:5]:
        log.info("    %s: %.1f", feat, imp)

    return metrics


def _compute_ndcg_by_group(
    pred: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int
) -> List[float]:
    """Compute NDCG@k for each query group."""
    ndcg_scores = []
    offset = 0
    for g_size in groups:
        g_pred = pred[offset:offset + g_size]
        g_labels = labels[offset:offset + g_size]
        offset += g_size
        if len(g_pred) < 2:
            continue
        # Sort by predicted score descending
        order = np.argsort(-g_pred)
        sorted_labels = g_labels[order]
        # DCG
        dcg = 0.0
        for i in range(min(k, len(sorted_labels))):
            dcg += (2 ** sorted_labels[i] - 1) / np.log2(i + 2)
        # Ideal DCG
        ideal_order = np.argsort(-g_labels)
        ideal_labels = g_labels[ideal_order]
        idcg = 0.0
        for i in range(min(k, len(ideal_labels))):
            idcg += (2 ** ideal_labels[i] - 1) / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return ndcg_scores


def _compute_topk_hit_rate(
    pred: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int
) -> float:
    """What fraction of true top-K coins are in predicted top-K?"""
    hits = []
    offset = 0
    for g_size in groups:
        g_pred = pred[offset:offset + g_size]
        g_labels = labels[offset:offset + g_size]
        offset += g_size
        actual_k = min(k, len(g_pred))
        if actual_k < 2:
            continue
        pred_topk = set(np.argsort(-g_pred)[:actual_k])
        true_topk = set(np.argsort(-g_labels)[:actual_k])
        overlap = len(pred_topk & true_topk)
        hits.append(overlap / actual_k)
    return float(np.mean(hits)) if hits else 0.0


# ---------------------------------------------------------------------------
# Inference (used by paper_trader)
# ---------------------------------------------------------------------------

@dataclass
class SelectorOutput:
    """Result of running the coin selector."""
    date: str
    rankings: List[Tuple[str, float]]  # [(coin, score), ...] sorted desc
    selected: List[str]                 # top-K coins
    regime_ok: bool = True


class CoinSelector:
    """Load and run the trained cross-sectional selector."""

    def __init__(self, model_dir: Optional[str] = None, top_k: int = DEFAULT_TOP_K):
        import lightgbm as lgb

        if model_dir is None:
            model_dir = CRYPTO_MODEL_DIR
        model_path = os.path.join(model_dir, SELECTOR_MODEL_FILE)
        config_path = os.path.join(model_dir, SELECTOR_CONFIG_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Selector model not found: {model_path}")

        self.model = lgb.Booster(model_file=model_path)
        self.top_k = top_k

        # Load config for coin universe
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

        log.info("Loaded coin selector (top_k=%d) from %s", top_k, model_path)

    def rank(
        self,
        universe_data: Dict[str, pd.DataFrame],
        min_score: float = -999.0,
    ) -> SelectorOutput:
        """Rank coins cross-sectionally and select top-K.

        Args:
            universe_data: {symbol: bars_df} with at least 60 bars each
            min_score: minimum predicted score to be eligible

        Returns:
            SelectorOutput with rankings and selected coins
        """
        # Build features for the latest date
        panel = build_xs_panel(universe_data)
        if panel.empty:
            log.warning("Empty panel — no coins to rank")
            return SelectorOutput(date="", rankings=[], selected=[])

        # Use only the latest date
        latest_date = panel["date"].max()
        latest = panel[panel["date"] == latest_date].copy()

        if len(latest) < 2:
            log.warning("Only %d coins on latest date — need ≥2", len(latest))
            return SelectorOutput(
                date=str(latest_date.date()),
                rankings=[], selected=[],
            )

        # Predict scores
        X = latest[XS_FEATURES].values
        scores = self.model.predict(X)
        latest = latest.copy()
        latest["score"] = scores

        # Sort by score descending
        latest = latest.sort_values("score", ascending=False)
        rankings = [(row["symbol"], float(row["score"]))
                    for _, row in latest.iterrows()]

        # Select top-K above minimum score
        selected = [sym for sym, sc in rankings[:self.top_k] if sc > min_score]

        today = str(latest_date.date())
        log.info("Selector rankings (%s): %s",
                 today,
                 ", ".join(f"{s} ({sc:.2f})" for s, sc in rankings[:8]))
        log.info("Selected top-%d: %s", self.top_k, ", ".join(selected))

        return SelectorOutput(
            date=today,
            rankings=rankings,
            selected=selected,
        )


def rank_coins_today(
    universe: Optional[List[str]] = None,
    model_dir: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> SelectorOutput:
    """Convenience: fetch data and rank coins for today."""
    if universe is None:
        universe = list(CRYPTO_UNIVERSE)
    if model_dir is None:
        model_dir = CRYPTO_MODEL_DIR

    selector = CoinSelector(model_dir=model_dir, top_k=top_k)

    log.info("Fetching data for %d coins...", len(universe))
    data = fetch_universe_data(universe, lookback_days=400)

    return selector.rank(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for training and ranking."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-sectional coin selector")
    sub = parser.add_subparsers(dest="action", help="Action to perform")

    # Train
    train_p = sub.add_parser("train", help="Train the selector model")
    train_p.add_argument("--train-end", default="2025-01-01",
                         help="OOS cutoff date (default: 2025-01-01)")
    train_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                         help=f"Number of coins to select (default: {DEFAULT_TOP_K})")
    train_p.add_argument("--save-dir", default=None,
                         help="Directory to save model (default: models/crypto/)")
    train_p.add_argument("--lookback", type=int, default=1500,
                         help="Days of historical data to fetch (default: 1500)")

    # Rank
    rank_p = sub.add_parser("rank", help="Rank coins for today")
    rank_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                         help=f"Number of coins to select (default: {DEFAULT_TOP_K})")
    rank_p.add_argument("--model-dir", default=None,
                         help="Directory with trained model (default: models/crypto/)")

    args = parser.parse_args()

    if args.action == "train":
        metrics = train_selector(
            train_end=args.train_end,
            top_k=args.top_k,
            save_dir=args.save_dir,
            lookback_days=args.lookback,
        )
        print(f"\n  Selector trained: NDCG@{args.top_k}={metrics['mean_ndcg']:.4f}, "
              f"hit_rate={metrics['topk_hit_rate']:.1%}")

    elif args.action == "rank":
        result = rank_coins_today(top_k=args.top_k, model_dir=args.model_dir)
        print(f"\n  === Coin Rankings ({result.date}) ===\n")
        for i, (sym, score) in enumerate(result.rankings, 1):
            marker = " <--" if sym in result.selected else ""
            print(f"  {i:2d}. {sym:<12s} score={score:+.3f}{marker}")
        print(f"\n  Selected ({len(result.selected)}): {', '.join(result.selected)}")

    else:
        parser.print_help()
