#!/usr/bin/env python3
"""
Factor Signals
====================================
Academically-backed factor features for ML models.

Data sources (all FREE):
    1. Credit Spread: yfinance HYG/LQD (Gilchrist & Zakrajsek 2012)
    2. Risk Appetite: yfinance XLY/XLP/IWM/SPY sector rotation
    3. Enhanced Yield Curve: FRED API (DGS3MO, T5YIE, DFF, DFII10)
    4. Fama-French Factor Exposures: Kenneth French Data Library (5 factors + momentum)
    5. Commodity Term Structure: yfinance futures (GC=F, SI=F, CL=F)
    6. Carry Signal: yfinance dividends + FRED rates
    7. Earnings Season Calendar: Pure calendar logic

Usage:
    from factor_signals import FactorFeatureBuilder, get_factor_features
    builder = FactorFeatureBuilder()
    features = builder.build_features(bars_df, symbol="SPY")
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from signals_engine import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("factor_signals")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
CREDIT_FEATURES = [
    "credit_spread",           # -log(HYG / LQD) * 100
    "credit_spread_chg5",      # credit_spread.diff(5)
    "credit_spread_zscore",    # (cs - rolling_mean_60) / rolling_std_60
]

RISK_APPETITE_FEATURES = [
    "risk_appetite_ratio",     # XLY_close / XLP_close
    "risk_appetite_trend",     # ratio.pct_change(20)
    "smallcap_rotation",       # (IWM/SPY).pct_change(20)
]

YIELD_CURVE_FEATURES = [
    "yield_curve_3m10y",       # DGS10 - DGS3MO
    "real_yield_10y",          # DFII10 level
    "breakeven_inflation_5y",  # T5YIE level
    "fed_funds_rate",          # DFF level
]

FF_FACTOR_FEATURES = [
    "beta_mkt",                # 60d rolling beta to Mkt-RF
    "beta_smb",                # 60d rolling beta to SMB
    "beta_hml",                # 60d rolling beta to HML
    "beta_rmw",                # 60d rolling beta to RMW
    "beta_cma",                # 60d rolling beta to CMA
    "beta_mom",                # 60d rolling beta to Momentum
    "factor_momentum",         # aggregate factor trend signal
]

COMMODITY_FEATURES = [
    "commodity_contango",      # futures vs ETF divergence
    "commodity_roll_yield",    # -commodity_contango
    "commodity_curve_zscore",  # z-score of contango over 60d
]

CARRY_FEATURES = [
    "equity_carry",            # trailing_12m_div_yield - risk_free_rate
    "fx_carry_differential",   # -usd_strength_ret20 (proxy)
]

EARNINGS_FEATURES = [
    "earnings_season_flag",    # 1 during earnings season windows
    "days_to_earnings_season", # distance to nearest window edge
]

# Universal features applied to all 12 symbols
UNIVERSAL_FEATURES = (
    CREDIT_FEATURES + RISK_APPETITE_FEATURES + YIELD_CURVE_FEATURES +
    FF_FACTOR_FEATURES
)

# Per-symbol mapping
FACTOR_FEATURE_MAP: Dict[str, List[str]] = {
    # LSTM daily -- universal + equity_carry + earnings
    "SPY":  UNIVERSAL_FEATURES + ["equity_carry"] + EARNINGS_FEATURES,
    "QQQ":  UNIVERSAL_FEATURES + ["equity_carry"] + EARNINGS_FEATURES,
    "IWM":  UNIVERSAL_FEATURES + ["equity_carry"],
    "SOXX": UNIVERSAL_FEATURES + ["equity_carry"] + EARNINGS_FEATURES,
    # PatchTST swing -- universal + commodity/carry per symbol
    "EWT":  UNIVERSAL_FEATURES + CARRY_FEATURES,
    "GLD":  UNIVERSAL_FEATURES + COMMODITY_FEATURES + ["equity_carry"],
    "EEM":  UNIVERSAL_FEATURES + CARRY_FEATURES,
    "SLV":  UNIVERSAL_FEATURES + COMMODITY_FEATURES + ["equity_carry"],
    # XGBoost expansion -- universal + per-symbol extras
    "EWJ":  UNIVERSAL_FEATURES + CARRY_FEATURES,
    "EWS":  UNIVERSAL_FEATURES + CARRY_FEATURES,
    "XLE":  UNIVERSAL_FEATURES + COMMODITY_FEATURES + ["equity_carry"] + EARNINGS_FEATURES,
    "INDA": UNIVERSAL_FEATURES + CARRY_FEATURES,
}

# Symbols that get commodity term structure features
COMMODITY_SYMBOLS = {"GLD", "SLV", "XLE"}
COMMODITY_FUTURES_MAP = {
    "GLD": "GC=F",   # gold futures
    "SLV": "SI=F",   # silver futures
    "XLE": "CL=F",   # crude oil futures
}

# Symbols that get FX carry differential
FX_CARRY_SYMBOLS = {"EWJ", "EWT", "EEM", "EWS", "INDA"}

# Symbols that get earnings season features
EARNINGS_SYMBOLS = {"SPY", "QQQ", "SOXX", "XLE"}


def get_factor_features(symbol: str) -> List[str]:
    """Return factor feature names for a given symbol."""
    return list(FACTOR_FEATURE_MAP.get(symbol, UNIVERSAL_FEATURES + ["equity_carry"]))


# ---------------------------------------------------------------------------
# P1: Credit Spread Fetcher
# ---------------------------------------------------------------------------
class CreditSpreadFetcher:
    """Fetch HYG and LQD from yfinance to compute credit spread proxy.

    Academic basis: Gilchrist & Zakrajsek (2012) show credit spreads
    predict equity returns 1-12 months ahead.
    """

    def __init__(self):
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: float = 0
        self._cache_ttl = 86400.0

    def fetch(self, lookback_days: int = 500) -> pd.DataFrame:
        """Fetch HYG and LQD close prices.

        Returns DataFrame with [date, hyg_close, lqd_close].
        """
        if time.time() - self._cache_time < self._cache_ttl and self._cache is not None:
            return self._cache

        cal_days = int(lookback_days * 1.5) + 30
        try:
            data = yf.download(
                ["HYG", "LQD"], period=f"{cal_days}d",
                interval="1d", progress=False, threads=True,
            )
            if data.empty:
                log.warning("No credit spread data returned.")
                return pd.DataFrame(columns=["date", "hyg_close", "lqd_close"])

            # yfinance multi-ticker returns MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                hyg = data[("Close", "HYG")]
                lqd = data[("Close", "LQD")]
            else:
                hyg = data["Close"]
                lqd = data["Close"]

            df = pd.DataFrame({
                "date": data.index,
                "hyg_close": hyg.values,
                "lqd_close": lqd.values,
            })
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["date"] = df["date"].dt.date
            df = df.dropna().sort_values("date").reset_index(drop=True)

            self._cache = df
            self._cache_time = time.time()
            return df

        except Exception as exc:
            log.warning("Credit spread fetch failed: %s", exc)
            return pd.DataFrame(columns=["date", "hyg_close", "lqd_close"])

    def build_features(self, bar_dates: pd.Series) -> pd.DataFrame:
        """Compute credit spread features aligned to bar dates."""
        raw = self.fetch(len(bar_dates) + 100)
        if raw.empty:
            return pd.DataFrame({c: np.nan for c in CREDIT_FEATURES},
                                index=range(len(bar_dates)))

        raw_dates = raw["date"].values
        hyg = raw["hyg_close"].values
        lqd = raw["lqd_close"].values

        # Build date->value maps
        spread_series = -np.log(hyg / np.where(lqd > 0, lqd, np.nan)) * 100
        spread_map = dict(zip(raw_dates, spread_series))

        # Align to bar dates
        aligned = np.array([spread_map.get(d, np.nan) for d in bar_dates.values])
        s = pd.Series(aligned).ffill()

        df = pd.DataFrame(index=range(len(bar_dates)))
        df["credit_spread"] = s.values
        df["credit_spread_chg5"] = s.diff(5).values
        mean60 = s.rolling(60, min_periods=20).mean()
        std60 = s.rolling(60, min_periods=20).std()
        df["credit_spread_zscore"] = ((s - mean60) / std60.replace(0, np.nan)).values

        return df


# ---------------------------------------------------------------------------
# P2: Risk Appetite Fetcher
# ---------------------------------------------------------------------------
class RiskAppetiteFetcher:
    """Fetch XLY, XLP, IWM, SPY for risk appetite signals.

    XLY/XLP ratio is a canonical risk-on/risk-off indicator.
    IWM/SPY captures small-cap leadership (risk-on signal).
    """

    def __init__(self):
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: float = 0
        self._cache_ttl = 86400.0

    def fetch(self, lookback_days: int = 500) -> pd.DataFrame:
        """Fetch XLY, XLP, IWM, SPY daily close prices."""
        if time.time() - self._cache_time < self._cache_ttl and self._cache is not None:
            return self._cache

        cal_days = int(lookback_days * 1.5) + 30
        try:
            data = yf.download(
                ["XLY", "XLP", "IWM", "SPY"], period=f"{cal_days}d",
                interval="1d", progress=False, threads=True,
            )
            if data.empty:
                return pd.DataFrame(columns=["date", "xly", "xlp", "iwm", "spy"])

            if isinstance(data.columns, pd.MultiIndex):
                df = pd.DataFrame({
                    "date": data.index,
                    "xly": data[("Close", "XLY")].values,
                    "xlp": data[("Close", "XLP")].values,
                    "iwm": data[("Close", "IWM")].values,
                    "spy": data[("Close", "SPY")].values,
                })
            else:
                df = pd.DataFrame({
                    "date": data.index,
                    "xly": data["Close"].values,
                    "xlp": data["Close"].values,
                    "iwm": data["Close"].values,
                    "spy": data["Close"].values,
                })

            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["date"] = df["date"].dt.date
            df = df.dropna().sort_values("date").reset_index(drop=True)

            self._cache = df
            self._cache_time = time.time()
            return df

        except Exception as exc:
            log.warning("Risk appetite fetch failed: %s", exc)
            return pd.DataFrame(columns=["date", "xly", "xlp", "iwm", "spy"])

    def build_features(self, bar_dates: pd.Series) -> pd.DataFrame:
        """Compute risk appetite features aligned to bar dates."""
        raw = self.fetch(len(bar_dates) + 100)
        if raw.empty:
            return pd.DataFrame({c: np.nan for c in RISK_APPETITE_FEATURES},
                                index=range(len(bar_dates)))

        # Build date->value maps
        ratio_map = dict(zip(
            raw["date"].values,
            (raw["xly"] / raw["xlp"].replace(0, np.nan)).values
        ))
        iwm_spy_map = dict(zip(
            raw["date"].values,
            (raw["iwm"] / raw["spy"].replace(0, np.nan)).values
        ))

        # Align to bar dates
        ratio = pd.Series([ratio_map.get(d, np.nan) for d in bar_dates.values]).ffill()
        iwm_spy = pd.Series([iwm_spy_map.get(d, np.nan) for d in bar_dates.values]).ffill()

        df = pd.DataFrame(index=range(len(bar_dates)))
        df["risk_appetite_ratio"] = ratio.values
        df["risk_appetite_trend"] = ratio.pct_change(20).values
        df["smallcap_rotation"] = iwm_spy.pct_change(20).values

        return df


# ---------------------------------------------------------------------------
# P3: Enhanced Yield Curve Fetcher
# ---------------------------------------------------------------------------
class YieldCurveFetcher:
    """Fetch enhanced yield curve data from FRED API.

    Adds: DGS3MO (3m T-bill), T5YIE (breakeven inflation),
    DFF (fed funds rate) on top of existing DGS10/DFII10/T10Y2Y.
    """

    BASE = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "DGS10":  "yield_10y",
        "DGS3MO": "yield_3m",
        "DFII10": "real_yield_10y",
        "T5YIE":  "breakeven_inflation_5y",
        "DFF":    "fed_funds_rate",
    }

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.environ.get("FRED_API_KEY")
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 86400.0

    def _fetch_series(self, series_id: str, lookback_days: int = 600) -> pd.DataFrame:
        cache_key = f"{series_id}_{lookback_days}"
        cached_time = self._cache_time.get(cache_key, 0)
        if time.time() - cached_time < self._cache_ttl and cache_key in self._cache:
            return self._cache[cache_key]

        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
            "file_type": "json",
            "sort_order": "asc",
            "api_key": self._key if self._key else "DEMO_KEY",
        }

        try:
            resp = requests.get(self.BASE, params=params, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except Exception as exc:
            log.warning("FRED fetch failed for %s: %s", series_id, exc)
            return pd.DataFrame(columns=["date", "value"])

        rows = []
        for o in obs:
            val = o.get("value", ".")
            if val == ".":
                continue
            rows.append({"date": o["date"], "value": float(val)})

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").reset_index(drop=True)

        self._cache[cache_key] = df
        self._cache_time[cache_key] = time.time()
        return df

    def build_features(self, bar_dates: pd.Series) -> pd.DataFrame:
        """Compute enhanced yield curve features aligned to bar dates."""
        n = len(bar_dates)
        lookback = n + 100

        # Fetch all series
        series_data = {}
        for fred_id, col_name in self.SERIES.items():
            raw = self._fetch_series(fred_id, lookback)
            if not raw.empty:
                series_data[col_name] = dict(zip(raw["date"].values, raw["value"].values))

        df = pd.DataFrame(index=range(n))

        # Align each series to bar dates
        for col_name, date_map in series_data.items():
            aligned = pd.Series([date_map.get(d, np.nan) for d in bar_dates.values]).ffill()
            df[col_name] = aligned.values

        # Compute 3m-10y spread
        if "yield_10y" in df.columns and "yield_3m" in df.columns:
            df["yield_curve_3m10y"] = df["yield_10y"] - df["yield_3m"]
        else:
            df["yield_curve_3m10y"] = np.nan

        # Keep FRED series with standard names
        # real_yield_10y and breakeven_inflation_5y and fed_funds_rate already in df

        # Select output columns
        out = pd.DataFrame(index=range(n))
        out["yield_curve_3m10y"] = df.get("yield_curve_3m10y", pd.Series(np.nan, index=range(n)))
        out["real_yield_10y"] = df.get("real_yield_10y", pd.Series(np.nan, index=range(n)))
        out["breakeven_inflation_5y"] = df.get("breakeven_inflation_5y", pd.Series(np.nan, index=range(n)))
        out["fed_funds_rate"] = df.get("fed_funds_rate", pd.Series(np.nan, index=range(n)))

        return out


# ---------------------------------------------------------------------------
# P4: Fama-French Factor Fetcher
# ---------------------------------------------------------------------------
class FamaFrenchFetcher:
    """Download Fama-French 5 factors + momentum from Kenneth French Data Library.

    Academic basis: Fama & French (1993, 2015), Carhart (1997).
    Data is free CSV inside ZIP archives; updated monthly.
    """

    FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: float = 0
        self._cache_ttl = 7 * 86400.0  # 7-day TTL (data updates monthly)

    def _parse_ff_csv(self, content: str, expected_cols: int) -> pd.DataFrame:
        """Parse a Kenneth French CSV (values in percent, dates as YYYYMMDD)."""
        lines = content.strip().split("\n")
        header_idx = None
        data_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # Find the header line (contains 'Mkt-RF' or 'Mom')
            if "Mkt-RF" in stripped or "Mom" in stripped:
                header_idx = i
                continue
            if header_idx is not None:
                parts = stripped.split(",")
                # Daily data has YYYYMMDD format (8 digits)
                if len(parts) >= 2 and len(parts[0].strip()) == 8:
                    try:
                        int(parts[0].strip())
                        data_lines.append(stripped)
                    except ValueError:
                        # End of daily data section
                        break
                elif data_lines:
                    # We had data and hit a non-data line — stop
                    break

        if not data_lines:
            log.warning("No daily data found in Fama-French CSV.")
            return pd.DataFrame()

        csv_text = "\n".join(data_lines)
        df = pd.read_csv(io.StringIO(csv_text), header=None)

        # First column is date (YYYYMMDD), rest are factor returns in percent
        df.columns = ["date_int"] + [f"col_{i}" for i in range(len(df.columns) - 1)]
        df["date"] = pd.to_datetime(df["date_int"].astype(str), format="%Y%m%d").dt.date
        # Convert from percent to decimal
        for col in df.columns:
            if col.startswith("col_"):
                df[col] = df[col].astype(float) / 100.0

        return df

    def _download_zip_csv(self, url: str, cache_file: str) -> str:
        """Download ZIP, extract first CSV, return content string."""
        cache_path = os.path.join(self._cache_dir, cache_file)

        # Check disk cache
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if time.time() - mtime < self._cache_ttl:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()

        try:
            log.info("Downloading Fama-French data from %s ...", url.split("/")[-1])
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
                content = zf.read(csv_name).decode("utf-8", errors="replace")

            # Cache to disk
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)

            return content

        except Exception as exc:
            log.warning("Fama-French download failed: %s", exc)
            # Try stale cache
            if os.path.exists(cache_path):
                log.info("Using stale Fama-French cache.")
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()
            return ""

    def fetch(self) -> pd.DataFrame:
        """Fetch and combine FF5 + Momentum factor returns.

        Returns DataFrame with columns: [date, mkt_rf, smb, hml, rmw, cma, rf, mom]
        Values are daily decimal returns (e.g., 0.01 = 1%).
        """
        if time.time() - self._cache_time < self._cache_ttl and self._cache is not None:
            return self._cache

        # Fetch FF5
        ff5_content = self._download_zip_csv(self.FF5_URL, "ff5_daily.csv")
        if not ff5_content:
            return pd.DataFrame()

        ff5 = self._parse_ff_csv(ff5_content, 6)
        if ff5.empty:
            return pd.DataFrame()

        # Expected columns: date_int, Mkt-RF, SMB, HML, RMW, CMA, RF
        factor_cols = [c for c in ff5.columns if c.startswith("col_")]
        if len(factor_cols) >= 6:
            ff5 = ff5.rename(columns={
                factor_cols[0]: "mkt_rf",
                factor_cols[1]: "smb",
                factor_cols[2]: "hml",
                factor_cols[3]: "rmw",
                factor_cols[4]: "cma",
                factor_cols[5]: "rf",
            })
        elif len(factor_cols) >= 5:
            ff5 = ff5.rename(columns={
                factor_cols[0]: "mkt_rf",
                factor_cols[1]: "smb",
                factor_cols[2]: "hml",
                factor_cols[3]: "rmw",
                factor_cols[4]: "cma",
            })
            ff5["rf"] = 0.0

        # Fetch Momentum
        mom_content = self._download_zip_csv(self.MOM_URL, "ff_momentum_daily.csv")
        if mom_content:
            mom = self._parse_ff_csv(mom_content, 1)
            if not mom.empty:
                mom_cols = [c for c in mom.columns if c.startswith("col_")]
                if mom_cols:
                    mom = mom.rename(columns={mom_cols[0]: "mom"})
                    mom = mom[["date", "mom"]]
                    ff5 = ff5.merge(mom, on="date", how="left")

        if "mom" not in ff5.columns:
            ff5["mom"] = 0.0

        result_cols = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]
        for col in result_cols:
            if col not in ff5.columns:
                ff5[col] = 0.0

        result = ff5[result_cols].sort_values("date").reset_index(drop=True)
        self._cache = result
        self._cache_time = time.time()
        return result

    def build_features(self, bar_dates: pd.Series, etf_returns: pd.Series) -> pd.DataFrame:
        """Compute rolling factor betas and factor momentum.

        Args:
            bar_dates: Series of date objects aligned to bars
            etf_returns: Daily returns of the ETF (same length as bar_dates)

        Returns:
            DataFrame with FF_FACTOR_FEATURES columns.
        """
        n = len(bar_dates)
        ff_data = self.fetch()
        if ff_data.empty:
            return pd.DataFrame({c: np.nan for c in FF_FACTOR_FEATURES},
                                index=range(n))

        # Build factor date maps
        factor_names = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
        factor_maps = {}
        for fname in factor_names:
            factor_maps[fname] = dict(zip(ff_data["date"].values, ff_data[fname].values))

        rf_map = dict(zip(ff_data["date"].values, ff_data["rf"].values))

        # Align factors to bar dates
        aligned_factors = {}
        for fname in factor_names:
            fmap = factor_maps[fname]
            aligned_factors[fname] = pd.Series(
                [fmap.get(d, np.nan) for d in bar_dates.values]
            ).ffill().values

        rf_aligned = pd.Series(
            [rf_map.get(d, np.nan) for d in bar_dates.values]
        ).ffill().fillna(0).values

        # Excess ETF returns
        etf_excess = etf_returns.values - rf_aligned

        # Rolling 60-day betas: beta = cov(r_etf, r_factor) / var(r_factor)
        window = 60
        df = pd.DataFrame(index=range(n))
        beta_names = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "beta_mom"]

        for fname, bname in zip(factor_names, beta_names):
            factor_vals = aligned_factors[fname]
            betas = np.full(n, np.nan)
            for i in range(window, n):
                r_etf = etf_excess[i - window:i]
                r_fac = factor_vals[i - window:i]
                mask = ~(np.isnan(r_etf) | np.isnan(r_fac))
                if mask.sum() >= 30:
                    cov = np.cov(r_etf[mask], r_fac[mask])
                    var_f = cov[1, 1]
                    if var_f > 1e-12:
                        betas[i] = cov[0, 1] / var_f
            df[bname] = betas

        # Factor momentum: aggregate factor trend signal
        # Sum of 60-day cumulative returns across all 6 factors, weighted by sign
        fm = np.full(n, np.nan)
        for i in range(window, n):
            total = 0.0
            valid = 0
            for fname in factor_names:
                fvals = aligned_factors[fname]
                chunk = fvals[i - window:i]
                mask = ~np.isnan(chunk)
                if mask.sum() >= 30:
                    cum_ret = np.sum(chunk[mask])
                    total += np.sign(cum_ret) * abs(cum_ret)
                    valid += 1
            if valid >= 4:
                fm[i] = total
        df["factor_momentum"] = fm

        return df


# ---------------------------------------------------------------------------
# P5: Commodity Term Structure
# ---------------------------------------------------------------------------
class CommodityTermFetcher:
    """Compute commodity contango/backwardation from futures vs ETF divergence.

    Academic basis: Gorton & Rouwenhorst (2006) — backwardation
    predicts commodity returns.
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 86400.0

    def _fetch_futures(self, yf_symbol: str, lookback_days: int = 500) -> pd.DataFrame:
        cache_key = f"{yf_symbol}_{lookback_days}"
        cached_time = self._cache_time.get(cache_key, 0)
        if time.time() - cached_time < self._cache_ttl and cache_key in self._cache:
            return self._cache[cache_key]

        cal_days = int(lookback_days * 1.5) + 30
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=f"{cal_days}d", interval="1d")
            if hist.empty:
                return pd.DataFrame(columns=["date", "close"])

            df = pd.DataFrame({
                "date": hist.index,
                "close": hist["Close"].values,
            })
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["date"] = df["date"].dt.date
            df = df.dropna().sort_values("date").reset_index(drop=True)

            self._cache[cache_key] = df
            self._cache_time[cache_key] = time.time()
            return df

        except Exception as exc:
            log.warning("Futures fetch failed for %s: %s", yf_symbol, exc)
            return pd.DataFrame(columns=["date", "close"])

    def build_features(self, bar_dates: pd.Series, etf_close: pd.Series,
                       symbol: str) -> pd.DataFrame:
        """Compute commodity term structure features.

        Args:
            bar_dates: Series of date objects
            etf_close: ETF close prices (same length as bar_dates)
            symbol: ETF symbol (GLD, SLV, or XLE)
        """
        n = len(bar_dates)
        futures_symbol = COMMODITY_FUTURES_MAP.get(symbol)
        if futures_symbol is None:
            return pd.DataFrame({c: np.nan for c in COMMODITY_FEATURES},
                                index=range(n))

        raw = self._fetch_futures(futures_symbol, n + 100)
        if raw.empty:
            return pd.DataFrame({c: np.nan for c in COMMODITY_FEATURES},
                                index=range(n))

        # Align futures close to bar dates
        fut_map = dict(zip(raw["date"].values, raw["close"].values))
        fut_aligned = pd.Series(
            [fut_map.get(d, np.nan) for d in bar_dates.values]
        ).ffill()

        etf = etf_close.astype(float)

        # Contango proxy: divergence between futures and ETF 20-day returns
        fut_ret20 = fut_aligned.pct_change(20)
        etf_ret20 = etf.pct_change(20)
        contango = fut_ret20 - etf_ret20

        df = pd.DataFrame(index=range(n))
        df["commodity_contango"] = contango.values
        df["commodity_roll_yield"] = (-contango).values
        mean60 = contango.rolling(60, min_periods=20).mean()
        std60 = contango.rolling(60, min_periods=20).std()
        df["commodity_curve_zscore"] = ((contango - mean60) / std60.replace(0, np.nan)).values

        return df


# ---------------------------------------------------------------------------
# P6: Carry Signal
# ---------------------------------------------------------------------------
class CarrySignalFetcher:
    """Compute equity carry and FX carry signals.

    equity_carry = trailing 12-month dividend yield - risk-free rate
    fx_carry_differential = -USD strength (proxy for EM carry trade)
    """

    def __init__(self):
        self._div_cache: Dict[str, float] = {}
        self._div_cache_time: float = 0
        self._div_cache_ttl = 86400.0

    def _get_div_yield(self, symbol: str) -> float:
        """Get trailing annual dividend yield for an ETF."""
        if (time.time() - self._div_cache_time < self._div_cache_ttl and
                symbol in self._div_cache):
            return self._div_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            div_yield = info.get("trailingAnnualDividendYield", 0) or 0
            self._div_cache[symbol] = div_yield
            self._div_cache_time = time.time()
            return div_yield
        except Exception:
            return 0.0

    def build_features(self, bar_dates: pd.Series, symbol: str,
                       risk_free_rate: Optional[pd.Series] = None,
                       usd_ret20: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compute carry features.

        Args:
            bar_dates: Series of date objects
            symbol: ETF symbol
            risk_free_rate: Aligned fed funds / 3m T-bill rate (from YieldCurveFetcher)
            usd_ret20: USD index 20-day return (from RiskAppetiteFetcher)
        """
        n = len(bar_dates)
        feature_list = FACTOR_FEATURE_MAP.get(symbol, [])
        df = pd.DataFrame(index=range(n))

        # Equity carry: div yield - risk-free rate
        if "equity_carry" in feature_list:
            div_yield = self._get_div_yield(symbol)
            if risk_free_rate is not None:
                rf = risk_free_rate.fillna(0).values / 100.0  # FRED rates in percent
                df["equity_carry"] = div_yield - rf
            else:
                df["equity_carry"] = div_yield

        # FX carry differential: inverse USD strength as carry proxy
        if "fx_carry_differential" in feature_list and symbol in FX_CARRY_SYMBOLS:
            if usd_ret20 is not None:
                df["fx_carry_differential"] = (-usd_ret20).values
            else:
                df["fx_carry_differential"] = np.nan

        return df


# ---------------------------------------------------------------------------
# P7: Earnings Season Calendar
# ---------------------------------------------------------------------------
class EarningsSeasonCalendar:
    """Pure calendar logic for earnings season windows.

    Earnings seasons: Jan 10 - Feb 15, Apr 10 - May 15,
                      Jul 10 - Aug 15, Oct 10 - Nov 15.
    """

    # (month_start, day_start, month_end, day_end)
    WINDOWS = [
        (1, 10, 2, 15),   # Q4 earnings (Jan-Feb)
        (4, 10, 5, 15),   # Q1 earnings (Apr-May)
        (7, 10, 8, 15),   # Q2 earnings (Jul-Aug)
        (10, 10, 11, 15), # Q3 earnings (Oct-Nov)
    ]

    @staticmethod
    def build_features(bar_dates: pd.Series) -> pd.DataFrame:
        """Compute earnings season features."""
        from datetime import date

        n = len(bar_dates)
        flags = np.zeros(n)
        distances = np.zeros(n)

        for i, d in enumerate(bar_dates.values):
            if not isinstance(d, date):
                d = pd.Timestamp(d).date()

            year = d.year
            in_window = False
            min_dist = 999

            # Check all windows in current year and adjacent years
            for yr in [year - 1, year, year + 1]:
                for ms, ds, me, de in EarningsSeasonCalendar.WINDOWS:
                    try:
                        w_start = date(yr, ms, ds)
                        w_end = date(yr, me, de)
                    except ValueError:
                        continue

                    if w_start <= d <= w_end:
                        in_window = True
                        min_dist = 0
                        break

                    # Distance to nearest window edge
                    dist_to_start = (w_start - d).days
                    dist_from_end = (d - w_end).days

                    if dist_to_start > 0:
                        min_dist = min(min_dist, dist_to_start)
                    if dist_from_end > 0:
                        min_dist = min(min_dist, dist_from_end)

                if in_window:
                    break

            flags[i] = 1.0 if in_window else 0.0
            distances[i] = 0 if in_window else min_dist

        df = pd.DataFrame(index=range(n))
        df["earnings_season_flag"] = flags
        df["days_to_earnings_season"] = distances / 30.0  # Normalize to ~monthly scale

        return df


# ---------------------------------------------------------------------------
# Unified Builder
# ---------------------------------------------------------------------------
class FactorFeatureBuilder:
    """Build factor signal features for ML models.

    Combines all 7 factor groups. Follows the AlphaFeatureBuilder pattern.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "data")
        os.makedirs(cache_dir, exist_ok=True)
        self._credit = CreditSpreadFetcher()
        self._risk = RiskAppetiteFetcher()
        self._yield_curve = YieldCurveFetcher(
            api_key=os.environ.get("FRED_API_KEY"))
        self._ff = FamaFrenchFetcher(cache_dir)
        self._commodity = CommodityTermFetcher()
        self._carry = CarrySignalFetcher()

    def build_features(self, bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Build factor features for a symbol.

        Args:
            bars_df: OHLCV bars with 'ts' and 'close' columns
            symbol: Target ETF symbol

        Returns:
            DataFrame with factor feature columns, same index as bars_df.
        """
        feature_list = get_factor_features(symbol)
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date
        close = bars_df["close"].astype(float)
        n = len(bars_df)

        df = pd.DataFrame(index=bars_df.index)

        # --- P1: Credit spread features (all symbols) ---
        credit_cols = [c for c in CREDIT_FEATURES if c in feature_list]
        if credit_cols:
            try:
                credit_df = self._credit.build_features(bar_dates)
                for col in credit_cols:
                    if col in credit_df.columns:
                        df[col] = credit_df[col].values
            except Exception as exc:
                log.warning("Credit spread features failed: %s", exc)

        # --- P2: Risk appetite features (all symbols) ---
        risk_cols = [c for c in RISK_APPETITE_FEATURES if c in feature_list]
        if risk_cols:
            try:
                risk_df = self._risk.build_features(bar_dates)
                for col in risk_cols:
                    if col in risk_df.columns:
                        df[col] = risk_df[col].values
            except Exception as exc:
                log.warning("Risk appetite features failed: %s", exc)

        # --- P3: Enhanced yield curve features (all symbols) ---
        yc_cols = [c for c in YIELD_CURVE_FEATURES if c in feature_list]
        if yc_cols:
            try:
                yc_df = self._yield_curve.build_features(bar_dates)
                for col in yc_cols:
                    if col in yc_df.columns:
                        df[col] = yc_df[col].values
            except Exception as exc:
                log.warning("Yield curve features failed: %s", exc)

        # --- P4: Fama-French factor betas (all symbols) ---
        ff_cols = [c for c in FF_FACTOR_FEATURES if c in feature_list]
        if ff_cols:
            try:
                etf_returns = close.pct_change().fillna(0)
                ff_df = self._ff.build_features(bar_dates, etf_returns)
                for col in ff_cols:
                    if col in ff_df.columns:
                        df[col] = ff_df[col].values
            except Exception as exc:
                log.warning("Fama-French features failed: %s", exc)

        # --- P5: Commodity term structure (GLD, SLV, XLE only) ---
        commodity_cols = [c for c in COMMODITY_FEATURES if c in feature_list]
        if commodity_cols and symbol in COMMODITY_SYMBOLS:
            try:
                comm_df = self._commodity.build_features(bar_dates, close, symbol)
                for col in commodity_cols:
                    if col in comm_df.columns:
                        df[col] = comm_df[col].values
            except Exception as exc:
                log.warning("Commodity features failed for %s: %s", symbol, exc)

        # --- P6: Carry signal ---
        carry_cols = [c for c in CARRY_FEATURES if c in feature_list]
        if carry_cols:
            try:
                # Pass risk-free rate and USD strength from earlier fetches
                rf_series = df.get("fed_funds_rate")
                # USD strength: use risk_appetite_ratio trend as proxy
                usd_ret = df.get("smallcap_rotation")  # inverse correlation
                carry_df = self._carry.build_features(
                    bar_dates, symbol,
                    risk_free_rate=rf_series,
                    usd_ret20=usd_ret,
                )
                for col in carry_cols:
                    if col in carry_df.columns:
                        df[col] = carry_df[col].values
            except Exception as exc:
                log.warning("Carry features failed for %s: %s", symbol, exc)

        # --- P7: Earnings season calendar ---
        earnings_cols = [c for c in EARNINGS_FEATURES if c in feature_list]
        if earnings_cols and symbol in EARNINGS_SYMBOLS:
            try:
                earn_df = EarningsSeasonCalendar.build_features(bar_dates)
                for col in earnings_cols:
                    if col in earn_df.columns:
                        df[col] = earn_df[col].values
            except Exception as exc:
                log.warning("Earnings features failed: %s", exc)

        # Fill any missing columns with NaN
        for col in feature_list:
            if col not in df.columns:
                df[col] = np.nan

        return df[feature_list]
