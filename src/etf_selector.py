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
# Intraday ETF Selector (5-min bar features)
# ---------------------------------------------------------------------------

# Intraday feature weights — prioritize tradability + intraday opportunity
INTRADAY_WEIGHTS = {
    "intraday_atr":         0.25,   # raw intraday opportunity (higher = better)
    "atr_to_spread":        0.20,   # profit per unit cost (key efficiency metric)
    "session_vol_ratio":    0.15,   # session vol / total vol (prefer session-driven vol)
    "relative_volume":      0.15,   # today vs 20d avg (elevated = catalyst/opportunity)
    "intraday_autocorr":    0.10,   # positive = trending (momentum model benefits)
    "gap_penalty":          0.10,   # penalize overnight-gap-dominated ETFs
    "dollar_volume":        0.05,   # raw liquidity (filter-like, but soft)
}

# Number of trading days of 5-min bars to fetch for feature computation
INTRADAY_LOOKBACK_SESSIONS = 20


def _fetch_intraday_bars(
    symbols: List[str],
    lookback_days: int = 30,
) -> Dict[str, pd.DataFrame]:
    """Fetch recent 5-min bars for all symbols via Alpaca data API.

    Returns {symbol: DataFrame} with columns [ts, open, high, low, close, volume].
    Requires ALPACA_API_KEY / ALPACA_API_SECRET env vars (data API, not trading).
    """
    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        log.warning("alpaca-py not installed — cannot fetch intraday bars")
        return {}

    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    # Fall back to intraday trading keys (can also access data API)
    if not api_key:
        api_key = os.environ.get("ALPACA_INTRADAY_KEY", "").strip()
    if not api_secret:
        api_secret = os.environ.get("ALPACA_INTRADAY_SECRET", "").strip()
    if not api_key or not api_secret:
        log.warning("No Alpaca API keys found — cannot fetch intraday bars")
        return {}

    client = StockHistoricalDataClient(api_key, api_secret)
    from datetime import datetime, timezone, timedelta
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    data: Dict[str, pd.DataFrame] = {}
    # Batch fetch to reduce API calls
    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
            )
            bars = client.get_stock_bars(request)
            bars_df = bars.df

            if bars_df.empty:
                continue

            for sym in batch:
                try:
                    if sym in bars_df.index.get_level_values(0):
                        sym_df = bars_df.loc[sym].reset_index()
                        sym_df = sym_df.rename(columns={
                            "timestamp": "ts", "open": "open", "high": "high",
                            "low": "low", "close": "close", "volume": "volume",
                        })
                        if len(sym_df) >= 200:  # need at least ~2.5 sessions
                            data[sym] = sym_df
                except Exception:
                    pass
        except Exception as exc:
            log.warning("Failed to fetch intraday batch %s: %s", batch, exc)

    log.info("Fetched 5-min bars for %d/%d symbols", len(data), len(symbols))
    return data


def _compute_intraday_features(
    bars_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute intraday-specific cross-sectional features from 5-min bars.

    Features:
      - intraday_atr:        20-session avg of intraday ATR (pct of price)
      - atr_to_spread:       intraday ATR / avg spread proxy (profitability ratio)
      - session_vol_ratio:   intraday session stdev / total daily stdev
      - relative_volume:     recent 3-session avg volume / 20-session avg volume
      - intraday_autocorr:   autocorrelation of 5-min returns (past 20 sessions)
      - gap_penalty:         negative of gap ratio (penalize gap-dominated ETFs)
      - dollar_volume:       log of average daily dollar volume
    """
    spy_bars = bars_data.get("SPY")

    rows = []
    for sym, df in bars_data.items():
        if len(df) < 200:
            continue

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        row = {"symbol": sym}

        # 5-min returns
        returns_5m = np.diff(close) / close[:-1]
        if len(returns_5m) < 100:
            continue

        # -- Intraday ATR (pct of price) --
        # True range on 5-min bars, averaged over recent data
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr_pct = np.mean(tr[-500:]) / np.mean(close[-500:]) if len(tr) >= 500 else \
                  np.mean(tr) / np.mean(close)
        row["intraday_atr"] = atr_pct

        # -- Spread proxy: avg (high - low) / close on 5-min bars --
        spread_proxy = np.mean((high[-500:] - low[-500:]) / close[-500:]) if len(close) >= 500 else \
                       np.mean((high - low) / close)
        row["atr_to_spread"] = atr_pct / max(spread_proxy, 1e-6)

        # -- Session volatility ratio --
        # Approximate: std of 5-min returns vs std of daily returns
        # Daily returns approximated from open-to-close every 78 bars (6.5h session)
        session_bars = 78  # ~one session
        n_sessions = len(returns_5m) // session_bars
        if n_sessions >= 3:
            intraday_stds = []
            daily_rets = []
            for s in range(n_sessions):
                start_idx = s * session_bars
                end_idx = start_idx + session_bars
                session_rets = returns_5m[start_idx:end_idx]
                intraday_stds.append(np.std(session_rets))
                # Open-to-close return for this "session"
                daily_rets.append(close[end_idx] / close[start_idx] - 1 if start_idx < len(close) and end_idx < len(close) else 0)

            avg_intraday_std = np.mean(intraday_stds[-20:]) if len(intraday_stds) >= 20 else np.mean(intraday_stds)
            daily_std = np.std(daily_rets[-20:]) if len(daily_rets) >= 20 else np.std(daily_rets)
            row["session_vol_ratio"] = avg_intraday_std / max(daily_std, 1e-6)
        else:
            row["session_vol_ratio"] = 1.0

        # -- Relative volume (recent 3 sessions vs 20 session avg) --
        if n_sessions >= 5:
            session_vols = []
            for s in range(n_sessions):
                start_idx = s * session_bars
                end_idx = min(start_idx + session_bars, len(volume))
                session_vols.append(np.sum(volume[start_idx:end_idx]))
            avg_recent_3 = np.mean(session_vols[-3:])
            avg_20 = np.mean(session_vols[-20:]) if len(session_vols) >= 20 else np.mean(session_vols)
            row["relative_volume"] = avg_recent_3 / max(avg_20, 1.0)
        else:
            row["relative_volume"] = 1.0

        # -- Intraday autocorrelation (lag-1 of 5-min returns) --
        recent_rets = returns_5m[-1000:]  # last ~13 sessions
        if len(recent_rets) >= 100:
            autocorr = np.corrcoef(recent_rets[:-1], recent_rets[1:])[0, 1]
            row["intraday_autocorr"] = autocorr if np.isfinite(autocorr) else 0.0
        else:
            row["intraday_autocorr"] = 0.0

        # -- Gap penalty --
        # Approximate overnight gaps: difference between session close and next session open
        if n_sessions >= 5:
            gaps = []
            for s in range(1, n_sessions):
                prev_close_idx = s * session_bars - 1
                curr_open_idx = s * session_bars
                if prev_close_idx < len(close) and curr_open_idx < len(close):
                    gap = abs(close[curr_open_idx] - close[prev_close_idx]) / close[prev_close_idx]
                    gaps.append(gap)
            avg_gap = np.mean(gaps[-20:]) if len(gaps) >= 20 else np.mean(gaps) if gaps else 0
            total_daily_range = np.mean(
                [(np.max(close[s*session_bars:(s+1)*session_bars]) -
                  np.min(close[s*session_bars:(s+1)*session_bars])) /
                 np.mean(close[s*session_bars:(s+1)*session_bars])
                 for s in range(max(0, n_sessions-20), n_sessions)
                 if (s+1)*session_bars <= len(close)]
            ) if n_sessions >= 1 else 0.01
            gap_ratio = avg_gap / max(total_daily_range, 1e-6)
            row["gap_penalty"] = -gap_ratio  # negative = penalize high gap ratio
        else:
            row["gap_penalty"] = 0.0

        # -- Dollar volume (log) --
        dollar_vol = np.mean(volume[-500:] * close[-500:]) if len(close) >= 500 else \
                     np.mean(volume * close)
        row["dollar_volume"] = np.log(max(dollar_vol, 1.0))

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    # Z-score each feature cross-sectionally
    for col in INTRADAY_WEIGHTS.keys():
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0

    return df


def _compute_intraday_composite_score(features: pd.DataFrame) -> pd.Series:
    """Compute weighted composite score from z-scored intraday features."""
    score = pd.Series(0.0, index=features.index)
    for feat, weight in INTRADAY_WEIGHTS.items():
        if feat in features.columns:
            score += weight * features[feat]
    return score


class ETFIntradaySelector:
    """Intraday-specific cross-sectional ETF selector.

    Uses 5-min bar features (intraday ATR, spread ratio, session vol,
    relative volume, autocorrelation, gap penalty) instead of daily
    momentum features. Designed for the ETF 5m trading group.
    """

    def __init__(self, top_k: int = 8, min_pool: int = MIN_POOL_SIZE):
        self.top_k = top_k
        self.min_pool = min_pool

    def rank(
        self,
        symbols: List[str],
        bars_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Rank all symbols with intraday-specific scores.

        Args:
            symbols: list of candidate ETF symbols
            bars_data: optional pre-fetched 5-min bar data {symbol: DataFrame}.
                       If None, fetches via Alpaca data API.

        Returns DataFrame with columns: symbol, score, rank, and all features.
        """
        if bars_data is None:
            # Always include SPY for reference
            fetch_symbols = list(set(symbols + ["SPY"]))
            bars_data = _fetch_intraday_bars(fetch_symbols, lookback_days=30)

        if not bars_data:
            log.warning("ETFIntradaySelector: no intraday data — returning empty")
            return pd.DataFrame()

        features = _compute_intraday_features(bars_data)
        if features.empty:
            return pd.DataFrame()

        scores = _compute_intraday_composite_score(features)
        features["score"] = scores
        features = features.sort_values("score", ascending=False)
        features["rank"] = range(1, len(features) + 1)

        return features.reset_index()

    def select(
        self,
        symbols: List[str],
        bars_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[str]:
        """Select top-K symbols by intraday composite score."""
        if len(symbols) <= self.min_pool:
            return symbols

        ranking = self.rank(symbols, bars_data)
        if ranking.empty:
            return symbols

        return ranking.head(self.top_k)["symbol"].tolist()


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
    parser.add_argument("--mode", default="swing", choices=["swing", "intraday"],
                        help="Selector mode: swing (daily features) or intraday (5-min features)")
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

    if args.mode == "intraday":
        selector = ETFIntradaySelector(top_k=args.top_k, min_pool=0)
        ranking = selector.rank(symbols)
        feature_cols = list(INTRADAY_WEIGHTS.keys())
        header_fmt = f"  {'Rank':<6} {'Symbol':<10} {'Score':>8} "
        header_fmt += " ".join(f"{c[:8]:>8}" for c in feature_cols)
    else:
        selector = ETFSelector(top_k=args.top_k, min_pool=0)
        ranking = selector.rank(symbols)
        feature_cols = list(WEIGHTS.keys())
        header_fmt = f"  {'Rank':<6} {'Symbol':<10} {'Score':>8} {'Mom63d':>8} {'Mom21d':>8} "
        header_fmt += f"{'RelStr':>8} {'VolReg':>8} {'Trend':>8} {'VolSrg':>8}"

    if ranking.empty:
        print("No data available for ranking.")
        return

    mode_label = "Intraday" if args.mode == "intraday" else "Swing"
    print(f"\n{'=' * 90}")
    print(f"  ETF {mode_label} Selector Rankings ({len(ranking)} symbols)")
    print(f"{'=' * 90}")
    print(header_fmt)
    print(f"  {'-' * 90}")

    for _, row in ranking.iterrows():
        selected_marker = " *" if row["rank"] <= args.top_k else ""
        line = f"  {int(row['rank']):<6} {row['symbol']:<10} {row['score']:>+7.3f} "
        for c in feature_cols:
            line += f"{row.get(c, 0):>+7.3f} "
        print(line + selected_marker)

    print(f"  {'-' * 90}")
    top_k_symbols = ranking[ranking["rank"] <= args.top_k]["symbol"].tolist()
    print(f"  Selected (top-{args.top_k}): {', '.join(top_k_symbols)}")
    print(f"{'=' * 98}\n")


if __name__ == "__main__":
    main()
