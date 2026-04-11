#!/usr/bin/env python3
"""
Market Signals
====================================
Volume anomalies, variance risk premium, calendar/seasonality,
and market breadth features for all 12 ETFs.

Data sources:
    G1 Abnormal Volume   — bars_df (zero API cost)
    G2 Variance Risk Prem — FRED VIX (existing) + CBOE VVIX CDN
    G3 Calendar/Season    — pure calendar logic (zero API cost)
    G4 Market Breadth     — yfinance 11 SPDR sector ETFs

Usage:
    from market_signals import MarketSignalBuilder, get_market_features
    builder = MarketSignalBuilder()
    features_df = builder.build_features(bars_df, symbol="SPY")
"""

from __future__ import annotations

import calendar as cal_mod
import io
import logging
import math
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from signals_engine import PROJECT_ROOT

# yfinance imported lazily (see signals_engine.py note).
_YF_MODULE = None

def _yf():
    global _YF_MODULE
    if _YF_MODULE is None:
        import yfinance as _yf_mod
        _YF_MODULE = _yf_mod
    return _YF_MODULE

log = logging.getLogger("market_signals")

# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------
VOLUME_FEATURES = [
    "volume_zscore",            # (vol - SMA20) / rolling_std20
    "volume_spike",             # 1 if vol > 2*SMA20 else 0
    "price_volume_divergence",  # +1 or -1
    "volume_trend",             # SMA5(vol) / SMA20(vol)
]

# Volume-profile support/resistance features (added for crypto swing)
VOLUME_PROFILE_FEATURES = [
    "vpoc_distance",            # (close - VPOC) / close: distance from volume point of control
    "vol_support_strength",     # volume concentration below price (higher = stronger support)
    "vol_resistance_strength",  # volume concentration above price (higher = stronger resistance)
    "vol_sr_imbalance",         # support_strength - resistance_strength (+ve = net support)
]

VRP_FEATURES = [
    "iv_rv_spread",             # VIX/100 - vol20 (annualized)
    "iv_rv_zscore",             # Z-score of iv_rv_spread over 60d
    "vvix_level",               # VVIX / 100
]

CALENDAR_FEATURES = [
    "day_of_week",              # 0-4 normalised to [0, 1]
    "turn_of_month",            # 1 if last 2 or first 3 calendar days
    "month_sin",                # sin(2*pi*month/12)
    "month_cos",                # cos(2*pi*month/12)
    "fomc_proximity",           # days to nearest FOMC / 30, capped 1.0
]

BREADTH_FEATURES = [
    "sector_breadth",           # sectors above SMA50 / 11
    "sector_dispersion",        # std of 20d returns across 11 sectors
    "breadth_momentum",         # sector_breadth.diff(5)
    "correlation_regime",       # mean pairwise 60d rolling corr across sectors
]

ALL_MARKET_FEATURES = (
    VOLUME_FEATURES + VOLUME_PROFILE_FEATURES + VRP_FEATURES
    + CALENDAR_FEATURES + BREADTH_FEATURES
)  # 20 total

ALL_SYMBOLS = [
    "SPY", "QQQ", "IWM", "SOXX",
    "EWT", "GLD", "EEM", "SLV",
    "EWJ", "EWS", "XLE", "INDA",
]

MARKET_FEATURE_MAP: Dict[str, List[str]] = {
    sym: list(ALL_MARKET_FEATURES) for sym in ALL_SYMBOLS
}


def get_market_features(symbol: str) -> List[str]:
    """Return market feature names for a given symbol."""
    return list(MARKET_FEATURE_MAP.get(symbol, ALL_MARKET_FEATURES))


# ---------------------------------------------------------------------------
# G2 helper: VVIX fetcher (CBOE CDN)
# ---------------------------------------------------------------------------
class VVIXFetcher:
    """Download and cache CBOE VVIX history CSV.

    VVIX = VIX-of-VIX.  VVIX > 30 signals high uncertainty about VIX itself.
    Pattern mirrors CBOEHistoricalFetcher from alpha_signals.py.
    """

    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv"
    CACHE_TTL = 86400.0  # 24 hours

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._memory: Optional[pd.DataFrame] = None

    def fetch(self) -> pd.DataFrame:
        """Return DataFrame[date, close] with daily VVIX values."""
        cache_path = os.path.join(self._cache_dir, "VVIX_history.csv")

        # 1. Disk cache check
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < self.CACHE_TTL:
                if self._memory is not None:
                    return self._memory
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"])
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    self._memory = df
                    return df
                except Exception:
                    pass

        # 2. Download from CDN
        try:
            log.info("Downloading CBOE VVIX history from CDN...")
            resp = requests.get(self.URL, timeout=30)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text))
            raw.columns = [c.strip().upper() for c in raw.columns]

            date_col = close_col = None
            for c in raw.columns:
                if "DATE" in c:
                    date_col = c
                if c == "CLOSE":
                    close_col = c
            if close_col is None:
                for c in raw.columns:
                    if c in ("VVIX",):
                        close_col = c
                        break

            if date_col is None or close_col is None:
                log.warning("VVIX CSV unexpected columns: %s", list(raw.columns))
                return pd.DataFrame(columns=["date", "close"])

            df = pd.DataFrame({
                "date": pd.to_datetime(raw[date_col], format="mixed", dayfirst=False),
                "close": pd.to_numeric(raw[close_col], errors="coerce"),
            })
            df = df.dropna(subset=["close"])
            df["date"] = df["date"].dt.date
            df = df.sort_values("date").reset_index(drop=True)

            df.to_csv(cache_path, index=False)
            self._memory = df
            log.info("Cached %d VVIX rows to %s", len(df), cache_path)
            return df

        except Exception as exc:
            log.warning("VVIX download failed: %s", exc)
            # Fallback to stale cache
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"])
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    return df
                except Exception:
                    pass
            return pd.DataFrame(columns=["date", "close"])


# ---------------------------------------------------------------------------
# G4 helper: Breadth fetcher (yfinance sector ETFs)
# ---------------------------------------------------------------------------
class BreadthFetcher:
    """Fetch 11 SPDR sector ETF close prices for breadth computation.

    Pattern mirrors CreditSpreadFetcher from factor_signals.py.
    """

    SECTORS = [
        "XLB", "XLC", "XLE", "XLF", "XLI",
        "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    ]

    def __init__(self):
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: float = 0
        self._cache_ttl = 86400.0

    def fetch(self, lookback_days: int = 500) -> pd.DataFrame:
        """Return DataFrame[date, XLB, XLC, ...] of daily closes."""
        if (time.time() - self._cache_time < self._cache_ttl
                and self._cache is not None):
            return self._cache

        cal_days = int(lookback_days * 1.5) + 30
        try:
            data = _yf().download(
                self.SECTORS,
                period=f"{cal_days}d",
                interval="1d",
                progress=False,
                threads=True,
            )
            if data.empty:
                return pd.DataFrame()

            df = pd.DataFrame({"date": data.index})
            if isinstance(data.columns, pd.MultiIndex):
                for s in self.SECTORS:
                    try:
                        df[s] = data[("Close", s)].values
                    except KeyError:
                        pass
            else:
                # Single-ticker edge case
                df["Close"] = data["Close"].values

            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["date"] = df["date"].dt.date
            df = df.sort_values("date").reset_index(drop=True)

            self._cache = df
            self._cache_time = time.time()
            log.info("Fetched %d rows of sector breadth data.", len(df))
            return df

        except Exception as exc:
            log.warning("Breadth fetch failed: %s", exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# G3 helper: FOMC calendar
# ---------------------------------------------------------------------------
class FOMCCalendar:
    """Hardcoded FOMC meeting announcement dates (2024-2026).

    Lucca & Moench (2015): pre-FOMC drift accounts for 80 pct of
    cumulative S&P 500 returns since 1994.
    """

    # End dates of each 2-day meeting (announcement day)
    DATES = [
        # 2024
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
        date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
        date(2024, 11, 7), date(2024, 12, 18),
        # 2025
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
        date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
        date(2025, 10, 29), date(2025, 12, 10),
        # 2026
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
        date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
        date(2026, 10, 28), date(2026, 12, 9),
    ]

    @staticmethod
    def days_to_nearest(d: date) -> int:
        """Return absolute days to nearest FOMC meeting date."""
        min_dist = 999
        for fomc_date in FOMCCalendar.DATES:
            dist = abs((fomc_date - d).days)
            if dist < min_dist:
                min_dist = dist
        return min_dist


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------
class MarketSignalBuilder:
    """Build market-signal features for ML models.

    Combines abnormal volume, variance risk premium, calendar/seasonality,
    and market breadth into a single DataFrame aligned to bars_df.

    Usage:
        builder = MarketSignalBuilder()
        df = builder.build_features(bars_df, "SPY")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "data")
        os.makedirs(cache_dir, exist_ok=True)
        self._vvix = VVIXFetcher(cache_dir)
        self._breadth = BreadthFetcher()
        # VIX for VRP — reuse existing FREDVixFetcher (self-contained)
        from signals_engine import FREDVixFetcher
        self._vix_fetcher = FREDVixFetcher(
            api_key=os.environ.get("FRED_API_KEY"))

    # --------------------------------------------------------------------- #
    def build_features(self, bars_df: pd.DataFrame,
                       symbol: str) -> pd.DataFrame:
        """Build 16 market-signal features aligned to bars_df."""
        feature_list = get_market_features(symbol)
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        close = bars_df["close"].astype(float)
        volume = bars_df["volume"].astype(float)
        n = len(bars_df)
        df = pd.DataFrame(index=bars_df.index)

        # === G1: Abnormal Volume ==========================================
        self._build_volume(df, close, volume)

        # === G1b: Volume Profile Support/Resistance =======================
        self._build_volume_profile(df, close, bars_df)

        # === G2: Variance Risk Premium ====================================
        self._build_vrp(df, close, bar_dates, bars_df.index, n)

        # === G3: Calendar / Seasonality ===================================
        self._build_calendar(df, bar_dates)

        # === G4: Market Breadth ===========================================
        self._build_breadth(df, bar_dates, bars_df.index, n)

        # Fill any missing columns with NaN
        for col in feature_list:
            if col not in df.columns:
                df[col] = np.nan

        return df[feature_list]

    # --------------------------------------------------------------------- #
    #  G1: Abnormal Volume
    # --------------------------------------------------------------------- #
    def _build_volume(self, df: pd.DataFrame,
                      close: pd.Series, volume: pd.Series) -> None:
        try:
            vol_sma20 = volume.rolling(20, min_periods=10).mean()
            vol_std20 = volume.rolling(20, min_periods=10).std()

            df["volume_zscore"] = (
                (volume - vol_sma20) / vol_std20.replace(0, np.nan)
            )
            df["volume_spike"] = (volume > 2 * vol_sma20).astype(float)

            price_dir = np.sign(close.pct_change())
            vol_dir = np.sign(volume.pct_change(5))
            df["price_volume_divergence"] = np.where(
                price_dir != vol_dir, -1.0, 1.0
            )

            vol_sma5 = volume.rolling(5, min_periods=3).mean()
            df["volume_trend"] = vol_sma5 / vol_sma20.replace(0, np.nan)
        except Exception as exc:
            log.warning("Volume features failed: %s", exc)

    # --------------------------------------------------------------------- #
    #  G1b: Volume Profile — Support/Resistance from volume concentration
    # --------------------------------------------------------------------- #
    def _build_volume_profile(self, df: pd.DataFrame,
                              close: pd.Series, bars_df: pd.DataFrame) -> None:
        """Build volume-weighted support/resistance features.

        Uses a rolling 30-day window. Divides the price range into bins,
        assigns volume to each bin, then measures:
        - VPOC (Volume Point of Control): price level with most volume
        - Support strength: volume below current price
        - Resistance strength: volume above current price
        """
        try:
            high = bars_df["high"].astype(float) if "high" in bars_df.columns else close
            low = bars_df["low"].astype(float) if "low" in bars_df.columns else close
            volume = bars_df["volume"].astype(float) if "volume" in bars_df.columns else pd.Series(1.0, index=close.index)

            window = 30  # 30-day lookback for volume profile
            n_bins = 20  # price bins

            vpoc_dist = pd.Series(np.nan, index=close.index)
            support_str = pd.Series(np.nan, index=close.index)
            resist_str = pd.Series(np.nan, index=close.index)

            for i in range(window, len(close)):
                sl = slice(i - window, i)
                c_window = close.iloc[sl].values
                h_window = high.iloc[sl].values
                l_window = low.iloc[sl].values
                v_window = volume.iloc[sl].values

                price_min = float(np.nanmin(l_window))
                price_max = float(np.nanmax(h_window))
                if price_max <= price_min or price_max == 0:
                    continue

                # Create price bins and assign volume
                bin_edges = np.linspace(price_min, price_max, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_volume = np.zeros(n_bins)

                # Distribute each bar's volume across bins it touches
                for j in range(len(v_window)):
                    bar_low = l_window[j]
                    bar_high = h_window[j]
                    bar_vol = v_window[j]
                    if np.isnan(bar_vol) or bar_vol <= 0:
                        continue
                    # Which bins does this bar span?
                    for b in range(n_bins):
                        if bin_edges[b + 1] >= bar_low and bin_edges[b] <= bar_high:
                            bin_volume[b] += bar_vol

                total_vol = bin_volume.sum()
                if total_vol <= 0:
                    continue

                # VPOC: bin with maximum volume
                vpoc_idx = int(np.argmax(bin_volume))
                vpoc_price = bin_centers[vpoc_idx]
                current_price = float(close.iloc[i])

                vpoc_dist.iloc[i] = (current_price - vpoc_price) / current_price

                # Support: fraction of volume below current price
                # Resistance: fraction of volume above current price
                below_mask = bin_centers < current_price
                above_mask = bin_centers > current_price

                vol_below = bin_volume[below_mask].sum() / total_vol
                vol_above = bin_volume[above_mask].sum() / total_vol

                support_str.iloc[i] = vol_below
                resist_str.iloc[i] = vol_above

            df["vpoc_distance"] = vpoc_dist
            df["vol_support_strength"] = support_str
            df["vol_resistance_strength"] = resist_str
            df["vol_sr_imbalance"] = support_str - resist_str

        except Exception as exc:
            log.warning("Volume profile features failed: %s", exc)

    # --------------------------------------------------------------------- #
    #  G2: Variance Risk Premium
    # --------------------------------------------------------------------- #
    def _build_vrp(self, df: pd.DataFrame, close: pd.Series,
                   bar_dates: pd.Series, idx: pd.Index, n: int) -> None:
        try:
            # Fetch VIX (self-contained via FRED)
            vix_df = self._vix_fetcher.fetch(lookback_days=n + 100)
            if not vix_df.empty:
                vix_map: Dict = {}
                for _, row in vix_df.iterrows():
                    d = row["date"]
                    if hasattr(d, "date"):
                        d = d.date()
                    vix_map[d] = row["vix"]
                vix_aligned = bar_dates.map(lambda d: vix_map.get(d, np.nan))
                vix_s = pd.Series(
                    vix_aligned.values, index=idx, dtype=float
                ).ffill()
            else:
                vix_s = pd.Series(np.nan, index=idx)

            # Realised vol (annualised)
            vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)

            # IV minus RV
            iv_rv = vix_s / 100.0 - vol20
            df["iv_rv_spread"] = iv_rv

            # Z-score over 60 days
            mean60 = iv_rv.rolling(60, min_periods=20).mean()
            std60 = iv_rv.rolling(60, min_periods=20).std()
            df["iv_rv_zscore"] = (iv_rv - mean60) / std60.replace(0, np.nan)

            # VVIX
            vvix_data = self._vvix.fetch()
            if not vvix_data.empty:
                vvix_map = dict(zip(vvix_data["date"].values,
                                    vvix_data["close"].values))
                vvix_aligned = bar_dates.map(
                    lambda d: vvix_map.get(d, np.nan))
                df["vvix_level"] = pd.Series(
                    vvix_aligned.values, index=idx, dtype=float
                ).ffill() / 100.0
            else:
                df["vvix_level"] = np.nan

        except Exception as exc:
            log.warning("VRP features failed: %s", exc)

    # --------------------------------------------------------------------- #
    #  G3: Calendar / Seasonality
    # --------------------------------------------------------------------- #
    def _build_calendar(self, df: pd.DataFrame,
                        bar_dates: pd.Series) -> None:
        try:
            dates_list = bar_dates.values
            n = len(dates_list)

            dow = np.zeros(n)
            tom = np.zeros(n)
            m_sin = np.zeros(n)
            m_cos = np.zeros(n)
            fomc_prox = np.zeros(n)

            for i, raw_d in enumerate(dates_list):
                d = (pd.Timestamp(raw_d).date()
                     if not isinstance(raw_d, date) else raw_d)
                dow[i] = d.weekday() / 4.0
                _, last_day = cal_mod.monthrange(d.year, d.month)
                tom[i] = 1.0 if (d.day >= last_day - 1 or d.day <= 3) else 0.0
                m_sin[i] = math.sin(2 * math.pi * d.month / 12.0)
                m_cos[i] = math.cos(2 * math.pi * d.month / 12.0)
                fomc_prox[i] = min(
                    FOMCCalendar.days_to_nearest(d) / 30.0, 1.0)

            df["day_of_week"] = dow
            df["turn_of_month"] = tom
            df["month_sin"] = m_sin
            df["month_cos"] = m_cos
            df["fomc_proximity"] = fomc_prox

        except Exception as exc:
            log.warning("Calendar features failed: %s", exc)

    # --------------------------------------------------------------------- #
    #  G4: Market Breadth
    # --------------------------------------------------------------------- #
    def _build_breadth(self, df: pd.DataFrame, bar_dates: pd.Series,
                       idx: pd.Index, n: int) -> None:
        try:
            sector_df = self._breadth.fetch(n + 100)
            if sector_df.empty:
                for col in BREADTH_FEATURES:
                    df[col] = np.nan
                return

            sectors = [s for s in BreadthFetcher.SECTORS
                       if s in sector_df.columns]
            if len(sectors) < 3:
                for col in BREADTH_FEATURES:
                    df[col] = np.nan
                return

            # Align each sector's close to bar_dates via map + ffill
            sector_maps: Dict[str, Dict] = {}
            for s in sectors:
                sector_maps[s] = dict(zip(
                    sector_df["date"].values, sector_df[s].values))

            aligned: Dict[str, pd.Series] = {}
            for s, smap in sector_maps.items():
                vals = pd.Series(
                    [smap.get(d, np.nan) for d in bar_dates.values],
                    index=idx,
                ).ffill()
                aligned[s] = vals

            n_sectors = len(aligned)

            # sector_breadth: fraction above 50-day SMA
            above_count = pd.Series(0.0, index=idx)
            for s, vals in aligned.items():
                sma50 = vals.rolling(50, min_periods=20).mean()
                above_count += (vals > sma50).astype(float)
            df["sector_breadth"] = above_count / n_sectors

            # sector_dispersion: std of 20-day returns
            ret20s = pd.DataFrame(
                {s: v.pct_change(20) for s, v in aligned.items()})
            df["sector_dispersion"] = ret20s.std(axis=1)

            # breadth_momentum: 5-day change
            df["breadth_momentum"] = df["sector_breadth"].diff(5)

            # correlation_regime: mean pairwise 60-day rolling corr
            daily_rets = pd.DataFrame(
                {s: v.pct_change() for s, v in aligned.items()})
            corr_vals = np.full(n, np.nan)
            for i in range(60, n):
                window = daily_rets.iloc[i - 60:i].dropna(axis=1, how="all")
                if window.shape[1] >= 3:
                    cm = window.corr()
                    mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
                    corr_vals[i] = cm.values[mask].mean()
            df["correlation_regime"] = corr_vals

        except Exception as exc:
            log.warning("Breadth features failed: %s", exc)
