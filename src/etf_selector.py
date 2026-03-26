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

from utils import SWING_MODEL_DIR, INTRADAY_MODEL_DIR

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

# Feature weights for composite score (rules-based fallback)
WEIGHTS = {
    "momentum_63d":     0.20,   # 3-month momentum (z-scored)
    "momentum_21d":     0.10,   # 1-month momentum (z-scored)
    "rel_strength_spy": 0.15,   # relative strength vs SPY (z-scored)
    "vol_regime":       0.10,   # inverse volatility (prefer lower vol, z-scored)
    "trend_strength":   0.10,   # distance from 200-SMA (z-scored)
    "volume_surge":     0.05,   # recent volume surge (z-scored)
    "reversal_5d":      0.10,   # 5-day mean reversion signal (z-scored)
    "credit_spread_z":  0.05,   # HY credit spread z-score (macro risk)
    "xs_momentum_rank": 0.10,   # cross-sectional momentum rank (percentile)
    "sector_dispersion":0.05,   # sector return dispersion (opportunity)
}

# All features used by the ML selector (superset — rules-based uses same set)
ML_FEATURES = list(WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def _fetch_credit_spread_z() -> float:
    """Fetch ICE BofA HY OAS from FRED, return z-score over 1yr.

    Returns 0.0 if FRED key is missing or fetch fails. Higher z-score means
    wider spreads (risk-off environment) — the ML model can learn how to use it.
    """
    fred_key = os.environ.get("FRED_API_KEY", "").strip()
    if not fred_key:
        return 0.0
    try:
        import requests
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=BAMLH0A0HYM2&api_key={fred_key}"
            "&sort_order=desc&limit=260&file_type=json"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        if len(vals) < 20:
            return 0.0
        current = vals[0]
        mean = np.mean(vals)
        std = np.std(vals)
        return float((current - mean) / std) if std > 0 else 0.0
    except Exception:
        return 0.0


def _compute_features(
    closes: Dict[str, pd.Series],
    volumes: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """Compute cross-sectional features for all symbols.

    10 features: 6 original + 4 new (reversal_5d, credit_spread_z,
    xs_momentum_rank, sector_dispersion).

    Args:
        closes: dict of symbol → pd.Series of daily close prices (DatetimeIndex)
        volumes: dict of symbol → pd.Series of daily volume (optional)

    Returns:
        DataFrame with one row per symbol and feature columns.
    """
    spy_close = closes.get("SPY")
    if volumes is None:
        volumes = {}

    # Pre-compute macro features (shared across all symbols)
    credit_z = _fetch_credit_spread_z()

    # Collect raw momentum for cross-sectional ranking
    raw_mom_63: Dict[str, float] = {}

    rows = []
    for sym, close in closes.items():
        if len(close) < 63:
            continue

        row = {"symbol": sym}

        # Momentum
        row["momentum_63d"] = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
        row["momentum_21d"] = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
        raw_mom_63[sym] = row["momentum_63d"]

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

        # --- NEW: reversal_5d (5-day mean reversion) ---
        if len(close) >= 10:
            ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if close.iloc[-6] > 0 else 0
            row["reversal_5d"] = -ret_5d  # negative = reversal signal (winners revert)
        else:
            row["reversal_5d"] = 0

        # --- NEW: credit_spread_z (macro, same for all symbols) ---
        row["credit_spread_z"] = -credit_z  # negative = prefer tighter spreads (risk-on)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    # --- NEW: xs_momentum_rank (cross-sectional percentile) ---
    n_symbols = len(df)
    if n_symbols > 1:
        mom_rank = df["momentum_63d"].rank(pct=True)
        df["xs_momentum_rank"] = mom_rank
    else:
        df["xs_momentum_rank"] = 0.5

    # --- NEW: sector_dispersion (std of sector-level returns) ---
    # Approximate by std of all 63d returns — higher dispersion = more alpha opportunity
    if n_symbols > 2:
        dispersion = float(df["momentum_63d"].std())
        df["sector_dispersion"] = dispersion  # same for all — ML learns the level
    else:
        df["sector_dispersion"] = 0.0

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
# LambdaMART ETF Selector (Phase 1 — ML-based)
# ---------------------------------------------------------------------------

ETF_SELECTOR_MODEL_FILE = "etf_selector_lgb.txt"
ETF_SELECTOR_CONFIG_FILE = "etf_selector_config.json"


class ETFSelectorML:
    """ML-based ETF selector using LightGBM LambdaRank.

    Fallback chain: ML model → rules-based ETFSelector → return all.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        min_pool: int = MIN_POOL_SIZE,
    ):
        self.top_k = top_k
        self.min_pool = min_pool
        self._model = None
        self._feature_names: List[str] = ML_FEATURES
        self._rules_fallback = ETFSelector(top_k=top_k, min_pool=min_pool)

        model_dir = model_dir or SWING_MODEL_DIR
        model_path = os.path.join(model_dir, ETF_SELECTOR_MODEL_FILE)
        config_path = os.path.join(model_dir, ETF_SELECTOR_CONFIG_FILE)

        if os.path.exists(model_path):
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=model_path)
                if os.path.exists(config_path):
                    import json
                    with open(config_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    self._feature_names = cfg.get("feature_names", ML_FEATURES)
                log.info("ETFSelectorML: loaded LambdaRank model from %s", model_path)
            except Exception as exc:
                log.warning("ETFSelectorML: failed to load model: %s — using rules", exc)
                self._model = None
        else:
            log.info("ETFSelectorML: no model at %s — using rules fallback", model_path)

    def rank(
        self,
        symbols: List[str],
        closes: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Rank symbols using ML model, falling back to rules if unavailable."""
        if self._model is None:
            return self._rules_fallback.rank(symbols, closes)

        # Compute features (same as rules-based)
        volumes: Optional[Dict[str, pd.Series]] = None
        if closes is None:
            closes, volumes = ETFSelector._fetch_closes(symbols)

        features = _compute_features(closes, volumes)
        if features.empty:
            return self._rules_fallback.rank(symbols, closes)

        # Ensure feature alignment
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0.0

        X = features[self._feature_names].values
        scores = self._model.predict(X)
        features["score"] = scores
        features = features.sort_values("score", ascending=False)
        features["rank"] = range(1, len(features) + 1)

        return features.reset_index()

    def select(
        self,
        symbols: List[str],
        closes: Optional[Dict[str, pd.Series]] = None,
    ) -> List[str]:
        """Select top-K symbols by ML score."""
        if len(symbols) <= self.min_pool:
            return symbols
        ranking = self.rank(symbols, closes)
        if ranking.empty:
            return symbols
        return ranking.head(self.top_k)["symbol"].tolist()


# ---------------------------------------------------------------------------
# Training pipeline for LambdaMART ETF selector
# ---------------------------------------------------------------------------

def _build_etf_panel(
    symbols: List[str],
    lookback_days: int = 756,
) -> pd.DataFrame:
    """Build cross-sectional panel: (date, symbol, features, label).

    Label = 10-day forward risk-adjusted return, discretized to 5 bins.
    """
    import yfinance as yf

    log.info("Fetching %dd of daily data for %d ETFs + SPY...", lookback_days, len(symbols))
    fetch_syms = list(set(symbols + ["SPY"]))
    data = yf.download(
        fetch_syms,
        period=f"{lookback_days}d",
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise RuntimeError("No data from yfinance")

    multi = hasattr(data.columns, "get_level_values")
    if multi:
        close_df = data["Close"]
        vol_df = data["Volume"]
    else:
        close_df = data[["Close"]]
        vol_df = data[["Volume"]]

    # Build feature snapshots for each trading day
    rows = []
    dates = close_df.index[252:]  # need 252d history for features

    for dt in dates:
        # Slice closes up to this date
        closes_dict: Dict[str, pd.Series] = {}
        volumes_dict: Dict[str, pd.Series] = {}
        for sym in fetch_syms:
            if sym not in close_df.columns:
                continue
            c = close_df[sym].loc[:dt].dropna()
            if len(c) >= 63:
                closes_dict[sym] = c
            v = vol_df[sym].loc[:dt].dropna() if sym in vol_df.columns else None
            if v is not None and len(v) >= 60:
                volumes_dict[sym] = v

        if len(closes_dict) < 5:
            continue

        features = _compute_features(closes_dict, volumes_dict)
        if features.empty:
            continue

        # Forward 10-day return (label)
        future_idx = close_df.index.get_loc(dt)
        if future_idx + 10 >= len(close_df.index):
            continue

        for sym in features.index:
            if sym not in close_df.columns or sym == "SPY":
                continue
            try:
                price_now = close_df[sym].iloc[future_idx]
                price_fwd = close_df[sym].iloc[future_idx + 10]
                if pd.isna(price_now) or pd.isna(price_fwd) or price_now <= 0:
                    continue
                fwd_ret = float(price_fwd / price_now - 1)
                # Risk-adjust by 20d vol
                daily_ret = close_df[sym].pct_change().iloc[future_idx - 20:future_idx]
                vol = float(daily_ret.std())
                vol = max(vol, 0.001)
                risk_adj_ret = fwd_ret / vol
            except (IndexError, KeyError):
                continue

            row = {"date": dt, "symbol": sym, "label": risk_adj_ret}
            for col in ML_FEATURES:
                row[col] = float(features.loc[sym, col]) if col in features.columns else 0.0
            rows.append(row)

    panel = pd.DataFrame(rows)
    log.info("Panel: %d rows, %d dates, %d symbols",
             len(panel), panel["date"].nunique(), panel["symbol"].nunique())
    return panel


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
        order = np.argsort(-g_pred)
        sorted_labels = g_labels[order]
        dcg = sum((2 ** sorted_labels[i] - 1) / np.log2(i + 2)
                  for i in range(min(k, len(sorted_labels))))
        ideal_order = np.argsort(-g_labels)
        ideal_labels = g_labels[ideal_order]
        idcg = sum((2 ** ideal_labels[i] - 1) / np.log2(i + 2)
                   for i in range(min(k, len(ideal_labels))))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return ndcg_scores


def train_etf_selector(
    universe: Optional[List[str]] = None,
    train_end: str = "2025-01-01",
    save_dir: Optional[str] = None,
    lookback_days: int = 756,
) -> dict:
    """Train LightGBM LambdaRank ETF selector.

    Mirrors coin_selector.train_selector() pattern:
    - Panel: 3yr daily data x ETF universe, features z-scored per date
    - Labels: 10d forward risk-adjusted return, discretized to 5 bins
    - Model: LambdaRank, NDCG@[3,6], 10-day embargo between train/val

    Returns dict with training metrics.
    """
    import json
    import lightgbm as lgb

    if save_dir is None:
        save_dir = SWING_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    if universe is None:
        from etf_screener import load_promoted_symbols, load_etf_universe
        universe = load_promoted_symbols(save_dir)
        if not universe:
            universe = load_etf_universe(save_dir)
        if not universe:
            raise RuntimeError("No ETF universe available. Run screen-etf-universe first.")

    # Build panel
    panel = _build_etf_panel(universe, lookback_days=lookback_days)
    if len(panel) < 200:
        raise RuntimeError(f"Panel too small: {len(panel)} rows (need ≥200)")

    # Train/val split with 10-day embargo
    train_end_dt = pd.Timestamp(train_end)
    embargo = pd.Timedelta(days=10)
    train_df = panel[panel["date"] < train_end_dt].copy()
    val_df = panel[panel["date"] >= train_end_dt + embargo].copy()

    if len(train_df) < 150:
        raise RuntimeError(f"Training set too small: {len(train_df)} rows")
    if len(val_df) < 50:
        log.warning("Validation set small: %d rows", len(val_df))

    log.info("Train: %d rows (%s → %s), Val: %d rows (%s → %s)",
             len(train_df), train_df["date"].min().date(), train_df["date"].max().date(),
             len(val_df), val_df["date"].min().date(), val_df["date"].max().date())

    # Discretize labels into 0-4 grades per date
    def _discretize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label_int"] = 0
        for _, grp in df.groupby("date"):
            vals = grp["label"].values
            if len(vals) < 5:
                ranks = pd.Series(vals).rank(method="min").astype(int) - 1
                ranks = (ranks * 4 / max(ranks.max(), 1)).astype(int).clip(0, 4)
                df.loc[grp.index, "label_int"] = ranks.values
            else:
                pcts = pd.Series(vals).rank(pct=True)
                grades = (pcts * 5).clip(0, 4.999).astype(int)
                df.loc[grp.index, "label_int"] = grades.values
        return df

    train_df = _discretize(train_df)
    val_df = _discretize(val_df)

    def build_lgb_data(df: pd.DataFrame):
        X = df[ML_FEATURES].values
        y = df["label_int"].values.astype(int)
        groups = df.groupby("date").size().values
        return X, y, groups

    X_train, y_train, groups_train = build_lgb_data(train_df)
    X_val, y_val, groups_val = build_lgb_data(val_df)

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train,
                            feature_name=ML_FEATURES, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, group=groups_val,
                          reference=train_set, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 6],
        "label_gain": [0, 1, 3, 7, 15],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": 42,
    }

    log.info("Training LambdaRank ETF selector...")
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

    # Evaluate
    val_pred = model.predict(X_val)
    ndcg_scores = _compute_ndcg_by_group(val_pred, y_val, groups_val, k=6)
    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    importance = dict(zip(ML_FEATURES, model.feature_importance(importance_type="gain").tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

    # NDCG comparison gate
    config_path = os.path.join(save_dir, ETF_SELECTOR_CONFIG_FILE)
    model_path = os.path.join(save_dir, ETF_SELECTOR_MODEL_FILE)
    deploy = True
    try:
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                old = json.load(f)
            old_ndcg = old.get("mean_ndcg", 0.0)
            if mean_ndcg < old_ndcg - 0.02:
                log.warning("ETF selector retrain REJECTED: NDCG %.4f < %.4f − 0.02",
                            mean_ndcg, old_ndcg)
                deploy = False
    except Exception:
        pass

    from datetime import datetime, timezone
    metrics = {
        "feature_names": ML_FEATURES,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_range": f"{train_df['date'].min().date()} → {train_df['date'].max().date()}",
        "val_range": f"{val_df['date'].min().date()} → {val_df['date'].max().date()}",
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "n_symbols": int(panel["symbol"].nunique()),
        "mean_ndcg": round(mean_ndcg, 4),
        "best_iteration": model.best_iteration,
        "feature_importance": dict(sorted_imp),
        "deployed": deploy,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    if deploy:
        model.save_model(model_path)
        log.info("Saved ETF selector model to %s", model_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("Saved ETF selector config to %s", config_path)

    log.info("=== ETF Selector Training Results ===")
    log.info("  NDCG@6 (val): %.4f%s", mean_ndcg,
             "" if deploy else " [REJECTED]")
    log.info("  Best iteration: %d", model.best_iteration)
    log.info("  Feature importance:")
    for feat, imp in sorted_imp[:5]:
        log.info("    %s: %.1f", feat, imp)

    return metrics


# ---------------------------------------------------------------------------
# LambdaRank Intraday ETF Selector (Phase 3 — dedicated intraday ML ranking)
# ---------------------------------------------------------------------------

ETF_INTRADAY_SELECTOR_MODEL_FILE = "etf_intraday_selector_lgb.txt"
ETF_INTRADAY_SELECTOR_CONFIG_FILE = "etf_intraday_selector_config.json"
INTRADAY_ML_FEATURES = list(INTRADAY_WEIGHTS.keys())


class ETFIntradaySelectorML:
    """ML-based intraday ETF selector using LightGBM LambdaRank.

    Fallback chain: ML model → rules-based ETFIntradaySelector → return all.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        top_k: int = 8,
        min_pool: int = 0,
    ):
        self.top_k = top_k
        self.min_pool = min_pool
        self._model = None
        self._feature_names: List[str] = INTRADAY_ML_FEATURES
        self._rules_fallback = ETFIntradaySelector(top_k=top_k, min_pool=min_pool)

        model_dir = model_dir or INTRADAY_MODEL_DIR
        model_path = os.path.join(model_dir, ETF_INTRADAY_SELECTOR_MODEL_FILE)
        config_path = os.path.join(model_dir, ETF_INTRADAY_SELECTOR_CONFIG_FILE)

        if os.path.exists(model_path):
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=model_path)
                if os.path.exists(config_path):
                    import json as _json
                    with open(config_path, encoding="utf-8") as f:
                        cfg = _json.load(f)
                    self._feature_names = cfg.get("feature_names", INTRADAY_ML_FEATURES)
                log.info("ETFIntradaySelectorML: loaded LambdaRank from %s", model_path)
            except Exception as exc:
                log.warning("ETFIntradaySelectorML: model load failed: %s — using rules", exc)
                self._model = None
        else:
            log.info("ETFIntradaySelectorML: no model at %s — using rules fallback", model_path)

    def rank(
        self,
        symbols: List[str],
        bars_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Rank symbols using ML model, falling back to rules if unavailable."""
        if self._model is None:
            return self._rules_fallback.rank(symbols, bars_data)

        if bars_data is None:
            fetch_symbols = list(set(symbols + ["SPY"]))
            bars_data = _fetch_intraday_bars(fetch_symbols, lookback_days=30)

        if not bars_data:
            return self._rules_fallback.rank(symbols, bars_data)

        features = _compute_intraday_features(bars_data)
        if features.empty:
            return self._rules_fallback.rank(symbols, bars_data)

        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0.0

        X = features[self._feature_names].values
        scores = self._model.predict(X)
        features["score"] = scores
        features = features.sort_values("score", ascending=False)
        features["rank"] = range(1, len(features) + 1)
        return features.reset_index()

    def select(
        self,
        symbols: List[str],
        bars_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[str]:
        """Select top-K symbols by ML score."""
        if len(symbols) <= self.min_pool:
            return symbols
        ranking = self.rank(symbols, bars_data)
        if ranking.empty:
            return symbols
        return ranking.head(self.top_k)["symbol"].tolist()


def _build_intraday_etf_panel(
    symbols: List[str],
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Build cross-sectional panel for intraday LambdaRank training.

    Each "query group" is one trading session (1 day of 5-min bars).
    Features: cross-sectional intraday characteristics (ATR%, vol ratio, etc.)
    Label: forward 1-session risk-adjusted return (next day close / today close - 1) / vol.
    """
    import yfinance as yf

    fetch_syms = list(set(symbols + ["SPY"]))
    log.info("Building intraday panel: %d symbols x %dd...", len(fetch_syms), lookback_days)

    # Fetch daily data for labels (forward return)
    daily = yf.download(fetch_syms, period=f"{lookback_days + 30}d",
                        progress=False, auto_adjust=True)
    if daily.empty:
        raise RuntimeError("No daily data from yfinance")

    multi = hasattr(daily.columns, "get_level_values")
    if multi:
        close_df = daily["Close"]
        vol_df = daily["Volume"]
    else:
        close_df = daily[["Close"]]
        vol_df = daily[["Volume"]]

    # Build cross-sectional feature snapshots per day using daily close data.
    # Intraday features (ATR%, vol ratio, etc.) are computed from rolling windows
    # of daily returns, z-scored cross-sectionally per date — same approach as
    # the swing panel builder but with intraday-specific feature weights.
    rows = []
    dates = close_df.index[20:]  # need 20d history for vol features

    for dt in dates:
        # Forward 1-day return label
        idx = close_df.index.get_loc(dt)
        if idx + 1 >= len(close_df):
            break

        # Build per-symbol features cross-sectionally for this date
        closes_dict: Dict[str, pd.Series] = {}
        volumes_dict: Dict[str, pd.Series] = {}
        for sym in fetch_syms:
            if sym not in close_df.columns:
                continue
            c = close_df[sym].loc[:dt].dropna()
            if len(c) >= 20:
                closes_dict[sym] = c
            if sym in vol_df.columns:
                v = vol_df[sym].loc[:dt].dropna()
                if len(v) >= 20:
                    volumes_dict[sym] = v

        if len(closes_dict) < 5:
            continue

        # Compute cross-sectional features using daily data
        # (reuses the swing feature pipeline — same signals, different weights at inference)
        features = _compute_features(closes_dict, volumes_dict)
        if features.empty:
            continue

        for sym in features.index:
            if sym == "SPY" or sym not in close_df.columns:
                continue
            try:
                price_now = close_df[sym].iloc[idx]
                price_fwd = close_df[sym].iloc[idx + 1]
                if pd.isna(price_now) or pd.isna(price_fwd) or price_now <= 0:
                    continue
                fwd_ret = float(price_fwd / price_now - 1)
                daily_ret = close_df[sym].pct_change().iloc[max(0, idx - 20):idx]
                vol = float(daily_ret.std())
                vol = max(vol, 0.001)
                risk_adj_ret = fwd_ret / vol
            except (IndexError, KeyError):
                continue

            row = {"date": dt, "symbol": sym, "label": risk_adj_ret}
            for col in INTRADAY_ML_FEATURES:
                val = float(features.loc[sym, col]) if col in features.columns else 0.0
                row[col] = val
            rows.append(row)

    panel = pd.DataFrame(rows) if rows else pd.DataFrame()
    if panel.empty:
        raise RuntimeError("Intraday panel is empty — no valid data")

    log.info("Intraday panel: %d rows, %d symbols",
             len(panel), panel["symbol"].nunique() if not panel.empty else 0)
    return panel


def train_intraday_etf_selector(
    universe: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    lookback_days: int = 60,
) -> dict:
    """Train LightGBM LambdaRank intraday ETF selector.

    Mirrors train_etf_selector() but with intraday features and 1-day horizon.
    Saves to models/intraday/etf_intraday_selector_lgb.txt.
    """
    import json as _json
    import lightgbm as lgb

    if save_dir is None:
        save_dir = INTRADAY_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    if universe is None:
        from etf_screener import load_etf_universe, ALL_SEED_SYMBOLS
        universe = load_etf_universe(SWING_MODEL_DIR)
        if not universe:
            universe = list(ALL_SEED_SYMBOLS)

    # Build panel
    panel = _build_intraday_etf_panel(universe, lookback_days=lookback_days)
    if len(panel) < 50:
        raise RuntimeError(f"Panel too small: {len(panel)} rows (need ≥50)")

    # Train/val split: use last 20% of dates for validation
    dates = sorted(panel["date"].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])

    train_df = panel[panel["date"].isin(train_dates)].copy()
    val_df = panel[panel["date"].isin(val_dates)].copy()

    if len(train_df) < 20:
        raise RuntimeError(f"Training set too small: {len(train_df)} rows")

    log.info("Train: %d rows, Val: %d rows", len(train_df), len(val_df))

    # Discretize labels into 0-4 grades per date
    def _discretize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label_int"] = 0
        for _, grp in df.groupby("date"):
            vals = grp["label"].values
            if len(vals) < 3:
                df.loc[grp.index, "label_int"] = 0
            else:
                pcts = pd.Series(vals).rank(pct=True)
                bins = (pcts * 4).astype(int).clip(0, 4)
                df.loc[grp.index, "label_int"] = bins.values
        return df

    train_df = _discretize(train_df)
    val_df = _discretize(val_df) if len(val_df) > 0 else val_df

    # Build LightGBM datasets
    feature_cols = INTRADAY_ML_FEATURES
    X_train = train_df[feature_cols].values
    y_train = train_df["label_int"].values
    train_groups = train_df.groupby("date").size().values

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups,
                           feature_name=feature_cols)

    val_ds = None
    if len(val_df) > 0:
        X_val = val_df[feature_cols].values
        y_val = val_df["label_int"].values
        val_groups = val_df.groupby("date").size().values
        val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups,
                             feature_name=feature_cols, reference=train_ds)

    # Train
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)]
    valid_sets = [train_ds] + ([val_ds] if val_ds else [])
    valid_names = ["train"] + (["val"] if val_ds else [])

    model = lgb.train(
        params, train_ds,
        num_boost_round=300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    # Evaluate
    metrics: dict = {"feature_names": feature_cols, "n_train": len(train_df)}
    if val_ds is not None:
        pred_val = model.predict(X_val)
        val_groups_arr = val_df.groupby("date").size().values
        ndcg3 = _compute_ndcg_by_group(pred_val, y_val, val_groups_arr, k=3)
        ndcg5 = _compute_ndcg_by_group(pred_val, y_val, val_groups_arr, k=5)
        metrics["val_ndcg3"] = round(float(np.mean(ndcg3)), 4) if ndcg3 else 0.0
        metrics["val_ndcg5"] = round(float(np.mean(ndcg5)), 4) if ndcg5 else 0.0
        log.info("Val NDCG@3=%.4f  NDCG@5=%.4f", metrics["val_ndcg3"], metrics["val_ndcg5"])

    # NDCG gate: only deploy if better than existing
    existing_model_path = os.path.join(save_dir, ETF_INTRADAY_SELECTOR_MODEL_FILE)
    deploy = True
    if os.path.exists(existing_model_path) and val_ds is not None:
        try:
            old_model = lgb.Booster(model_file=existing_model_path)
            old_pred = old_model.predict(X_val)
            old_ndcg3 = _compute_ndcg_by_group(old_pred, y_val, val_groups_arr, k=3)
            old_mean = float(np.mean(old_ndcg3)) if old_ndcg3 else 0.0
            if metrics.get("val_ndcg3", 0) < old_mean:
                log.warning("New model NDCG@3=%.4f < old=%.4f — NOT deploying",
                            metrics.get("val_ndcg3", 0), old_mean)
                deploy = False
        except Exception:
            pass

    if deploy:
        model.save_model(os.path.join(save_dir, ETF_INTRADAY_SELECTOR_MODEL_FILE))
        with open(os.path.join(save_dir, ETF_INTRADAY_SELECTOR_CONFIG_FILE), "w") as f:
            _json.dump(metrics, f, indent=2)
        log.info("Intraday ETF selector saved → %s", save_dir)

    return metrics


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
