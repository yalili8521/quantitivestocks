#!/usr/bin/env python3
"""
ETF Selector (Layer 1 — Rules-Based)
======================================
Cross-sectional rules-based selector for the swing ETF trading group.
Ranks ETFs by momentum composite score and selects the top-K for trading.

Only activates when the promoted pool has > 15 symbols (otherwise trade all
promoted symbols directly). Uses momentum, relative strength, and vol regime
signals — no ML model, purely rules-based.

Pipeline:
  1. Fetch 252 days of daily bars for all candidates (yfinance)
  2. Compute cross-sectional features per ETF
  3. Score = weighted composite of momentum + relative strength + vol regime
  4. Return top-K symbols sorted by score

Usage (via paper_trader.py — automatic when promoted_symbols > 15):
    from etf_selector import ETFSelector
    selector = ETFSelector(top_k=10)
    selected = selector.select(symbols)  # returns top-K symbols

Standalone test:
    python src/etf_selector.py --symbols SPY,QQQ,IWM,GLD,SLV,SMH,...
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import SWING_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etf_selector")

# ---------------------------------------------------------------------------
# Selector parameters
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
MIN_POOL_SIZE = 15   # only activate selector when pool > this; else trade all

# Feature weights for composite score
WEIGHTS = {
    "momentum_63d":     0.25,   # 3-month momentum (z-scored)
    "momentum_21d":     0.15,   # 1-month momentum (z-scored)
    "rel_strength_spy": 0.20,   # relative strength vs SPY (z-scored)
    "vol_regime":       0.15,   # inverse volatility (prefer lower vol, z-scored)
    "trend_strength":   0.15,   # distance from 200-SMA (z-scored)
    "volume_surge":     0.10,   # recent volume surge (z-scored)
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def _compute_features(
    closes: Dict[str, pd.Series],
    volumes: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """Compute cross-sectional features for all symbols.

    Args:
        closes: dict of symbol → pd.Series of daily close prices (DatetimeIndex)
        volumes: dict of symbol → pd.Series of daily volume (optional)

    Returns:
        DataFrame with one row per symbol and feature columns.
    """
    spy_close = closes.get("SPY")
    if volumes is None:
        volumes = {}

    rows = []
    for sym, close in closes.items():
        if len(close) < 63:
            continue

        row = {"symbol": sym}

        # Momentum
        row["momentum_63d"] = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
        row["momentum_21d"] = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0

        # Relative strength vs SPY
        if spy_close is not None and len(spy_close) >= 63:
            spy_ret_63 = spy_close.iloc[-1] / spy_close.iloc[-63] - 1
            row["rel_strength_spy"] = row["momentum_63d"] - spy_ret_63
        else:
            row["rel_strength_spy"] = 0

        # Volatility regime: inverse of recent vol (lower vol = better rank)
        daily_ret = close.pct_change().dropna()
        vol_20d = daily_ret.tail(20).std() * np.sqrt(252) if len(daily_ret) >= 20 else 0.3
        row["vol_regime"] = -vol_20d  # negative so lower vol scores higher

        # Trend strength: distance above 200-SMA (positive = uptrend)
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            row["trend_strength"] = (close.iloc[-1] / sma200 - 1) if sma200 > 0 else 0
        else:
            row["trend_strength"] = 0

        # Volume surge: dollar-volume 5d avg / 60d avg
        vol = volumes.get(sym)
        if vol is not None and len(vol) >= 60:
            dollar_vol = vol * close.reindex(vol.index, method="ffill")
            avg_5d = dollar_vol.iloc[-5:].mean()
            avg_60d = dollar_vol.iloc[-60:].mean()
            row["volume_surge"] = (avg_5d / avg_60d - 1) if avg_60d > 0 else 0
        else:
            row["volume_surge"] = 0

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    # Z-score each feature cross-sectionally
    for col in WEIGHTS.keys():
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0

    return df


def _compute_composite_score(features: pd.DataFrame) -> pd.Series:
    """Compute weighted composite score from z-scored features."""
    score = pd.Series(0.0, index=features.index)
    for feat, weight in WEIGHTS.items():
        if feat in features.columns:
            score += weight * features[feat]
    return score


# ---------------------------------------------------------------------------
# ETFSelector class
# ---------------------------------------------------------------------------
class ETFSelector:
    """Rules-based cross-sectional ETF selector.

    Only activates when the candidate pool > MIN_POOL_SIZE. Otherwise,
    returns all candidates unchanged (no filtering needed).
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K, min_pool: int = MIN_POOL_SIZE):
        self.top_k = top_k
        self.min_pool = min_pool

    def select(
        self,
        symbols: List[str],
        closes: Optional[Dict[str, pd.Series]] = None,
    ) -> List[str]:
        """Select top-K symbols from candidate pool.

        Args:
            symbols: list of candidate ETF symbols
            closes: optional pre-fetched close price series (symbol → pd.Series).
                    If None, fetches via yfinance.

        Returns:
            List of selected symbols (top-K by composite score).
            If pool <= min_pool, returns all symbols unchanged.
        """
        if len(symbols) <= self.min_pool:
            log.info("ETFSelector: pool size %d <= %d — trading all symbols",
                     len(symbols), self.min_pool)
            return symbols

        # Fetch close + volume if not provided
        volumes: Optional[Dict[str, pd.Series]] = None
        if closes is None:
            closes, volumes = self._fetch_closes(symbols)

        if not closes:
            log.warning("ETFSelector: no data fetched — returning all symbols")
            return symbols

        # Compute features and score
        features = _compute_features(closes, volumes)
        if features.empty:
            log.warning("ETFSelector: no features computed — returning all symbols")
            return symbols

        scores = _compute_composite_score(features)
        scores = scores.sort_values(ascending=False)

        selected = scores.head(self.top_k).index.tolist()

        log.info("ETFSelector: selected %d/%d symbols (pool=%d)",
                 len(selected), len(symbols), len(symbols))
        for i, sym in enumerate(selected, 1):
            log.info("  %d. %s  score=%.3f", i, sym, scores[sym])

        return selected

    def rank(
        self,
        symbols: List[str],
        closes: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Rank all symbols with scores (for display/debug).

        Returns DataFrame with columns: symbol, score, rank, and all features.
        """
        volumes: Optional[Dict[str, pd.Series]] = None
        if closes is None:
            closes, volumes = self._fetch_closes(symbols)

        features = _compute_features(closes, volumes)
        if features.empty:
            return pd.DataFrame()

        scores = _compute_composite_score(features)
        features["score"] = scores
        features = features.sort_values("score", ascending=False)
        features["rank"] = range(1, len(features) + 1)

        return features.reset_index()

    @staticmethod
    def _fetch_closes(
        symbols: List[str],
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """Fetch 1 year of daily close prices + volume for all symbols via yfinance.

        Returns:
            (closes_dict, volumes_dict) — each maps symbol → pd.Series.
        """
        import yfinance as yf

        # Always include SPY for relative strength computation
        fetch_symbols = list(set(symbols + ["SPY"]))

        log.info("Fetching close+volume for %d symbols...", len(fetch_symbols))
        try:
            data = yf.download(fetch_symbols, period="1y", progress=False, auto_adjust=True)
            if data.empty:
                return {}, {}

            multi = hasattr(data.columns, "get_level_values")
            closes: Dict[str, pd.Series] = {}
            volumes: Dict[str, pd.Series] = {}

            if multi:
                close_df = data["Close"] if len(fetch_symbols) > 1 else data[["Close"]]
                vol_df = data["Volume"] if len(fetch_symbols) > 1 else data[["Volume"]]
            else:
                close_df = data
                vol_df = None

            if isinstance(close_df, pd.Series):
                closes[fetch_symbols[0]] = close_df.dropna()
            else:
                for sym in fetch_symbols:
                    if sym in close_df.columns:
                        series = close_df[sym].dropna()
                        if len(series) >= 63:
                            closes[sym] = series

            if vol_df is not None:
                if isinstance(vol_df, pd.Series):
                    volumes[fetch_symbols[0]] = vol_df.dropna()
                else:
                    for sym in fetch_symbols:
                        if sym in vol_df.columns:
                            series = vol_df[sym].dropna()
                            if len(series) >= 60:
                                volumes[sym] = series

            log.info("Fetched close data for %d/%d symbols (%d with volume)",
                     len(closes), len(fetch_symbols), len(volumes))
            return closes, volumes

        except Exception as e:
            log.warning("Failed to fetch close prices: %s", e)
            return {}, {}


# ---------------------------------------------------------------------------
# CLI (standalone test)
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ETF Selector (Layer 1 — Rules-Based)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (default: load from promoted_symbols.json)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of symbols to select (default: {DEFAULT_TOP_K})")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        from etf_screener import load_promoted_symbols, load_etf_universe
        symbols = load_promoted_symbols(SWING_MODEL_DIR)
        if not symbols:
            symbols = load_etf_universe(SWING_MODEL_DIR)
        if not symbols:
            print("No symbols found. Run: python main.py screen-etf-universe")
            return

    selector = ETFSelector(top_k=args.top_k, min_pool=0)  # min_pool=0 to force ranking
    ranking = selector.rank(symbols)

    if ranking.empty:
        print("No data available for ranking.")
        return

    print(f"\n{'=' * 90}")
    print(f"  ETF Selector Rankings ({len(ranking)} symbols)")
    print(f"{'=' * 90}")
    print(f"  {'Rank':<6} {'Symbol':<10} {'Score':>8} {'Mom63d':>8} {'Mom21d':>8} "
          f"{'RelStr':>8} {'VolReg':>8} {'Trend':>8} {'VolSrg':>8}")
    print(f"  {'-' * 90}")

    for _, row in ranking.iterrows():
        selected_marker = " *" if row["rank"] <= args.top_k else ""
        print(f"  {int(row['rank']):<6} {row['symbol']:<10} {row['score']:>+7.3f} "
              f"{row.get('momentum_63d', 0):>+7.3f} {row.get('momentum_21d', 0):>+7.3f} "
              f"{row.get('rel_strength_spy', 0):>+7.3f} {row.get('vol_regime', 0):>+7.3f} "
              f"{row.get('trend_strength', 0):>+7.3f} {row.get('volume_surge', 0):>+7.3f}"
              f"{selected_marker}")

    print(f"  {'-' * 90}")
    top_k_symbols = ranking[ranking["rank"] <= args.top_k]["symbol"].tolist()
    print(f"  Selected (top-{args.top_k}): {', '.join(top_k_symbols)}")
    print(f"{'=' * 98}\n")


if __name__ == "__main__":
    main()
