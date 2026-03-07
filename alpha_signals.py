#!/usr/bin/env python3
"""
Alpha Signals
====================================
Alternative data sources as ML features for genuine alpha generation.

Data sources:
    1. CBOE Historical Indices (VIX, VIX3M, VIX9D, SKEW) — CSV downloads
    2. SEC EDGAR Form 4 Insider Trading — aggregate insider buying for ETF components

Usage:
    from alpha_signals import AlphaFeatureBuilder, get_alpha_features
    builder = AlphaFeatureBuilder()
    features = builder.build_features(bars_df, symbol="SPY")
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from signals_engine import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("alpha_signals")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
CBOE_FEATURES = [
    "cboe_vix_term_spread",   # VIX - VIX3M (backwardation spread in points)
    "cboe_vix_term_ratio",    # VIX / VIX3M (>1 = stressed/backwardation)
    "cboe_vix9d_ratio",       # VIX9D / VIX (>1 = extreme short-term fear spike)
    "cboe_skew",              # (SKEW - 100) / 50 (normalized tail risk pricing)
    "cboe_vix_percentile",    # VIX percentile rank over 252 days
]

INSIDER_FEATURES = [
    "insider_net_ratio",      # (buy_value - sell_value) / total_value, 30-day rolling
    "insider_buy_breadth",    # n_buyers / n_total_insiders, 30-day rolling
]

# Per-symbol feature mapping: US equity ETFs get insider features, all get CBOE
ALPHA_FEATURE_MAP: Dict[str, List[str]] = {
    "SPY":  CBOE_FEATURES + INSIDER_FEATURES,
    "QQQ":  CBOE_FEATURES + INSIDER_FEATURES,
    "SOXX": CBOE_FEATURES + INSIDER_FEATURES,
    "XLE":  CBOE_FEATURES + INSIDER_FEATURES,
    "IWM":  CBOE_FEATURES,    # 2000 stocks — impractical for insider aggregation
    "GLD":  CBOE_FEATURES,    # commodity ETF — no stock components
    "SLV":  CBOE_FEATURES,
    "EWJ":  CBOE_FEATURES,    # international — no EDGAR filings
    "EWT":  CBOE_FEATURES,
    "EEM":  CBOE_FEATURES,
    "EWS":  CBOE_FEATURES,
    "INDA": CBOE_FEATURES,
}

# Top 20 component stocks per ETF (for insider signal aggregation)
ETF_COMPONENTS: Dict[str, List[str]] = {
    "SPY": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B",
        "JPM", "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD",
        "AVGO", "COST", "PEP", "ABBV", "MRK",
    ],
    "QQQ": [
        "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "AVGO",
        "COST", "TSLA", "NFLX", "AMD", "PEP", "ADBE", "CSCO",
        "INTC", "QCOM", "TXN", "AMAT", "INTU", "AMGN",
    ],
    "SOXX": [
        "NVDA", "AMD", "AVGO", "INTC", "QCOM", "TXN", "AMAT",
        "LRCX", "MU", "ADI", "KLAC", "ON", "MRVL", "NXPI",
        "MPWR", "SWKS", "GFS", "MCHP", "ENTG", "ASML",
    ],
    "XLE": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX",
        "VLO", "PXD", "OXY", "WMB", "HES", "HAL", "DVN",
        "FANG", "CTRA", "MRO", "BKR", "OKE", "KMI",
    ],
}


def get_alpha_features(symbol: str) -> List[str]:
    """Return list of alpha feature names for a given symbol."""
    return list(ALPHA_FEATURE_MAP.get(symbol, CBOE_FEATURES))


# ---------------------------------------------------------------------------
# CBOE Historical Index Fetcher
# ---------------------------------------------------------------------------
class CBOEHistoricalFetcher:
    """Download and cache CBOE index CSVs (VIX, VIX3M, VIX9D, SKEW).

    CBOE publishes daily closing values as CSVs on their CDN.
    VIX history goes back to 1990 (9000+ rows).
    """

    INDICES = {
        "VIX":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
        "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
        "SKEW":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    }

    CACHE_TTL = 86400.0  # 24 hours

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._memory_cache: Dict[str, pd.DataFrame] = {}

    def _download_or_cache(self, index_name: str) -> pd.DataFrame:
        """Download CBOE CSV or return cached version if fresh."""
        cache_path = os.path.join(self._cache_dir, f"{index_name}_history.csv")

        # Check disk cache freshness
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < self.CACHE_TTL:
                if index_name in self._memory_cache:
                    return self._memory_cache[index_name]
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"])
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    self._memory_cache[index_name] = df
                    return df
                except Exception:
                    pass  # re-download

        url = self.INDICES.get(index_name)
        if not url:
            log.warning("Unknown CBOE index: %s", index_name)
            return pd.DataFrame(columns=["date", "close"])

        try:
            log.info("Downloading CBOE %s history from CDN...", index_name)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Parse CSV — CBOE uses "DATE", "OPEN", "HIGH", "LOW", "CLOSE"
            raw = pd.read_csv(io.StringIO(resp.text))

            # Normalize column names (CBOE sometimes uses different casing)
            raw.columns = [c.strip().upper() for c in raw.columns]

            # Find date and close columns
            # SKEW CSV uses "SKEW" instead of "CLOSE"
            date_col = None
            close_col = None
            for c in raw.columns:
                if "DATE" in c:
                    date_col = c
                if c == "CLOSE":
                    close_col = c
            # Fallback: if no CLOSE column, use the index name as value col
            if close_col is None:
                for c in raw.columns:
                    if c != date_col and c in ("SKEW", "VIX", "VIX3M", "VIX9D"):
                        close_col = c
                        break

            if date_col is None or close_col is None:
                log.warning("CBOE %s CSV unexpected columns: %s", index_name, list(raw.columns))
                return pd.DataFrame(columns=["date", "close"])

            df = pd.DataFrame({
                "date": pd.to_datetime(raw[date_col], format="mixed", dayfirst=False),
                "close": pd.to_numeric(raw[close_col], errors="coerce"),
            })
            df = df.dropna(subset=["close"])
            df["date"] = df["date"].dt.date
            df = df.sort_values("date").reset_index(drop=True)

            # Save to disk cache
            df.to_csv(cache_path, index=False)
            self._memory_cache[index_name] = df
            log.info("Cached %d %s rows to %s", len(df), index_name, cache_path)
            return df

        except Exception as exc:
            log.warning("CBOE %s download failed: %s", index_name, exc)
            # Try stale cache
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"])
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    log.info("Using stale %s cache (%d rows).", index_name, len(df))
                    return df
                except Exception:
                    pass
            return pd.DataFrame(columns=["date", "close"])

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Download all CBOE indices. Returns {name: DataFrame}."""
        return {name: self._download_or_cache(name) for name in self.INDICES}

    def build_features(self, bar_dates: pd.Series) -> pd.DataFrame:
        """Compute CBOE-derived features aligned to bar dates.

        Args:
            bar_dates: Series of datetime.date values (from bars_df["ts"].dt.date)

        Returns:
            DataFrame with CBOE_FEATURES columns, indexed like bar_dates.
        """
        data = self.fetch_all()
        df = pd.DataFrame(index=bar_dates.index)

        vix = self._align(data.get("VIX", pd.DataFrame()), bar_dates)
        vix3m = self._align(data.get("VIX3M", pd.DataFrame()), bar_dates)
        vix9d = self._align(data.get("VIX9D", pd.DataFrame()), bar_dates)
        skew = self._align(data.get("SKEW", pd.DataFrame()), bar_dates)

        # VIX term spread: VIX - VIX3M (positive = backwardation = fear)
        df["cboe_vix_term_spread"] = vix - vix3m

        # VIX term ratio: VIX / VIX3M (>1 = backwardation)
        df["cboe_vix_term_ratio"] = vix / vix3m.replace(0, np.nan)

        # VIX9D ratio: VIX9D / VIX (>1 = extreme short-term spike)
        df["cboe_vix9d_ratio"] = vix9d / vix.replace(0, np.nan)

        # SKEW normalized: (SKEW - 100) / 50 (center around baseline 100)
        df["cboe_skew"] = (skew - 100.0) / 50.0

        # VIX percentile rank over 252 trading days
        df["cboe_vix_percentile"] = vix.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        return df

    def _align(self, source_df: pd.DataFrame, bar_dates: pd.Series) -> pd.Series:
        """Align source DataFrame to bar dates via forward-fill."""
        if source_df.empty:
            return pd.Series(np.nan, index=bar_dates.index)

        date_map = dict(zip(source_df["date"].values, source_df["close"].values))
        aligned = bar_dates.map(lambda d: date_map.get(d, np.nan))
        return pd.Series(aligned.values, index=bar_dates.index, dtype=float).ffill()


# ---------------------------------------------------------------------------
# SEC EDGAR Insider Signal Fetcher
# ---------------------------------------------------------------------------
class EdgarInsiderFetcher:
    """Aggregate Form 4 insider trading across ETF component stocks.

    Uses SEC EDGAR public API (no authentication required).
    Rate limit: 10 requests/second (SEC Fair Access policy).
    """

    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    USER_AGENT = "QuantitativeStocksResearch admin@quantstocks.local"

    CACHE_TTL = 86400.0  # 24 hours

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._ticker_to_cik: Dict[str, str] = {}
        self._filings_cache: Dict[str, dict] = {}
        self._filings_cache_time: Dict[str, float] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.USER_AGENT})

    def _load_ticker_cik_map(self) -> None:
        """Download ticker-to-CIK mapping from SEC."""
        if self._ticker_to_cik:
            return

        cache_path = os.path.join(self._cache_dir, "sec_ticker_cik.json")

        # Check disk cache
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < self.CACHE_TTL * 7:  # weekly refresh for CIK map
                try:
                    with open(cache_path, "r") as f:
                        self._ticker_to_cik = json.load(f)
                    log.info("Loaded %d ticker-CIK mappings from cache.",
                             len(self._ticker_to_cik))
                    return
                except Exception:
                    pass

        try:
            log.info("Downloading SEC ticker-CIK map...")
            resp = self._session.get(self.SEC_TICKERS_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # SEC returns {index: {cik_str, ticker, title}}
            for entry in data.values():
                ticker = str(entry.get("ticker", "")).upper()
                cik = str(entry.get("cik_str", ""))
                if ticker and cik:
                    self._ticker_to_cik[ticker] = cik.zfill(10)

            # Save to disk
            with open(cache_path, "w") as f:
                json.dump(self._ticker_to_cik, f)
            log.info("Cached %d ticker-CIK mappings.", len(self._ticker_to_cik))

        except Exception as exc:
            log.warning("SEC ticker-CIK download failed: %s", exc)

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Look up CIK for a ticker."""
        self._load_ticker_cik_map()
        return self._ticker_to_cik.get(ticker.upper())

    def _fetch_form4_filings(self, ticker: str,
                              lookback_days: int = 90) -> List[dict]:
        """Fetch Form 4 filings for a ticker from SEC EDGAR.

        Returns list of {filing_date, is_buy, value} dicts.
        """
        cache_key = ticker.upper()
        cached_time = self._filings_cache_time.get(cache_key, 0)
        if (time.time() - cached_time < self.CACHE_TTL
                and cache_key in self._filings_cache):
            return self._filings_cache[cache_key].get("filings", [])

        cik = self._get_cik(ticker)
        if not cik:
            return []

        cache_path = os.path.join(self._cache_dir, f"insider_{ticker.upper()}.json")

        # Check disk cache
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < self.CACHE_TTL:
                try:
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                    self._filings_cache[cache_key] = cached
                    self._filings_cache_time[cache_key] = time.time()
                    return cached.get("filings", [])
                except Exception:
                    pass

        try:
            url = self.SEC_SUBMISSIONS_URL.format(cik=cik)
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            submissions = resp.json()

            recent = submissions.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            docs = recent.get("primaryDocument", [])

            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()

            filings = []
            parsed_count = 0
            max_parse = 10  # Limit XML parsing to 10 filings per stock
            for i, form in enumerate(forms):
                if form != "4" and form != "4/A":
                    continue
                filing_date = datetime.strptime(dates[i], "%Y-%m-%d").date()
                if filing_date < cutoff:
                    continue

                if parsed_count >= max_parse:
                    break

                # Parse the Form 4 XML for transaction details
                accession_clean = accessions[i].replace("-", "")
                doc = docs[i]
                # Strip XSLT prefix (e.g. "xslF345X05/") to get raw XML
                if "/" in doc:
                    doc = doc.split("/", 1)[-1]
                doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                           f"{cik.lstrip('0')}/{accession_clean}/{doc}")

                txns = self._parse_form4(doc_url)
                for txn in txns:
                    txn["filing_date"] = str(filing_date)
                    filings.append(txn)
                parsed_count += 1

                # Rate limiting: 10 req/sec
                time.sleep(0.12)

            # Cache to disk
            cached_data = {"ticker": ticker.upper(), "filings": filings,
                           "fetched_at": datetime.now(timezone.utc).isoformat()}
            with open(cache_path, "w") as f:
                json.dump(cached_data, f, indent=2)
            self._filings_cache[cache_key] = cached_data
            self._filings_cache_time[cache_key] = time.time()

            return filings

        except Exception as exc:
            log.warning("EDGAR filings fetch failed for %s: %s", ticker, exc)
            return []

    def _parse_form4(self, url: str) -> List[dict]:
        """Parse a Form 4 XML document for transaction details.

        Returns list of {is_buy: bool, value: float, insider_name: str}.
        """
        try:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()

            # Parse XML
            root = ET.fromstring(resp.text)

            # Namespace handling (SEC Form 4 XML uses default namespace)
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            # Get insider name
            reporter = root.find(f".//{ns}reportingOwner/{ns}reportingOwnerId/{ns}rptOwnerName")
            insider_name = reporter.text.strip() if reporter is not None else "Unknown"

            transactions = []

            # Non-derivative transactions
            for txn in root.findall(f".//{ns}nonDerivativeTransaction"):
                ad_code = txn.find(
                    f".//{ns}transactionAmounts/{ns}transactionAcquiredDisposedCode/{ns}value")
                shares_el = txn.find(
                    f".//{ns}transactionAmounts/{ns}transactionShares/{ns}value")
                price_el = txn.find(
                    f".//{ns}transactionAmounts/{ns}transactionPricePerShare/{ns}value")

                if ad_code is None or shares_el is None:
                    continue

                is_buy = ad_code.text.strip().upper() == "A"  # A=acquired, D=disposed
                shares = float(shares_el.text) if shares_el.text else 0
                price = float(price_el.text) if price_el is not None and price_el.text else 0

                # Filter: only include open-market purchases/sales (not options exercises)
                txn_code_el = txn.find(
                    f".//{ns}transactionCoding/{ns}transactionCode")
                txn_code = txn_code_el.text.strip().upper() if txn_code_el is not None else ""
                # Skip gifts (G) — keep P=purchase, S=sale, A=award, M=exercise
                if txn_code == "G":
                    continue

                value = shares * price
                if value > 0:
                    transactions.append({
                        "is_buy": is_buy,
                        "value": value,
                        "insider_name": insider_name,
                    })

            return transactions

        except Exception as exc:
            log.debug("Form 4 parse failed for %s: %s", url, exc)
            return []

    def build_features(self, bar_dates: pd.Series, symbol: str) -> pd.DataFrame:
        """Build insider signal features for an ETF.

        Aggregates Form 4 data from top component stocks.

        Args:
            bar_dates: Series of datetime.date values
            symbol: ETF symbol (must be in ETF_COMPONENTS)

        Returns:
            DataFrame with INSIDER_FEATURES columns.
        """
        components = ETF_COMPONENTS.get(symbol, [])
        if not components:
            df = pd.DataFrame(index=bar_dates.index)
            for col in INSIDER_FEATURES:
                df[col] = np.nan
            return df

        # Fetch all component insider data
        all_txns = []
        for ticker in components:
            filings = self._fetch_form4_filings(ticker, lookback_days=120)
            for f in filings:
                all_txns.append({
                    "date": datetime.strptime(f["filing_date"], "%Y-%m-%d").date(),
                    "is_buy": f.get("is_buy", False),
                    "value": f.get("value", 0),
                    "insider": f.get("insider_name", ""),
                })

        if not all_txns:
            df = pd.DataFrame(index=bar_dates.index)
            for col in INSIDER_FEATURES:
                df[col] = np.nan
            return df

        # Build daily aggregates
        txn_df = pd.DataFrame(all_txns)
        txn_df["buy_value"] = txn_df.apply(
            lambda r: r["value"] if r["is_buy"] else 0, axis=1)
        txn_df["sell_value"] = txn_df.apply(
            lambda r: r["value"] if not r["is_buy"] else 0, axis=1)
        txn_df["is_buyer"] = txn_df["is_buy"].astype(int)

        # Group by date
        daily = txn_df.groupby("date").agg(
            buy_total=("buy_value", "sum"),
            sell_total=("sell_value", "sum"),
            n_buyers=("is_buyer", "sum"),
            n_total=("insider", "nunique"),
        ).reset_index()

        # Create date-indexed series
        date_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in daily["date"]])
        buy_s = pd.Series(daily["buy_total"].values, index=date_idx)
        sell_s = pd.Series(daily["sell_total"].values, index=date_idx)
        buyers_s = pd.Series(daily["n_buyers"].values, index=date_idx)
        total_s = pd.Series(daily["n_total"].values, index=date_idx)

        # Resample to business days and compute rolling 30-day aggregates
        buy_30d = buy_s.resample("B").sum().rolling(30, min_periods=1).sum()
        sell_30d = sell_s.resample("B").sum().rolling(30, min_periods=1).sum()
        buyers_30d = buyers_s.resample("B").sum().rolling(30, min_periods=1).sum()
        total_30d = total_s.resample("B").sum().rolling(30, min_periods=1).sum()

        total_val_30d = buy_30d + sell_30d
        net_ratio = (buy_30d - sell_30d) / total_val_30d.replace(0, np.nan)
        breadth = buyers_30d / total_30d.replace(0, np.nan)

        # Align to bar dates
        net_map = dict(zip(net_ratio.index.date, net_ratio.values))
        breadth_map = dict(zip(breadth.index.date, breadth.values))

        df = pd.DataFrame(index=bar_dates.index)
        df["insider_net_ratio"] = bar_dates.map(
            lambda d: net_map.get(d, np.nan)).values
        df["insider_net_ratio"] = pd.Series(
            df["insider_net_ratio"].values, index=df.index, dtype=float).ffill()

        df["insider_buy_breadth"] = bar_dates.map(
            lambda d: breadth_map.get(d, np.nan)).values
        df["insider_buy_breadth"] = pd.Series(
            df["insider_buy_breadth"].values, index=df.index, dtype=float).ffill()

        return df


# ---------------------------------------------------------------------------
# Unified Alpha Feature Builder
# ---------------------------------------------------------------------------
class AlphaFeatureBuilder:
    """Build alpha signal features for ML models.

    Combines CBOE historical data and SEC EDGAR insider signals.
    Follows the same pattern as CrossAssetFeatureBuilder.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "data")
        os.makedirs(cache_dir, exist_ok=True)
        self._cboe = CBOEHistoricalFetcher(cache_dir)
        self._insider = EdgarInsiderFetcher(cache_dir)

    def build_features(self, bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Build alpha features for a symbol.

        Args:
            bars_df: OHLCV bars for the target symbol
            symbol: Target symbol (determines which features to compute)

        Returns:
            DataFrame with alpha feature columns, same index as bars_df.
        """
        feature_list = get_alpha_features(symbol)
        bar_dates = pd.to_datetime(bars_df["ts"]).dt.date

        df = pd.DataFrame(index=bars_df.index)

        # --- CBOE features (all symbols) ---
        cboe_cols = [c for c in CBOE_FEATURES if c in feature_list]
        if cboe_cols:
            try:
                cboe_df = self._cboe.build_features(bar_dates)
                for col in cboe_cols:
                    if col in cboe_df.columns:
                        df[col] = cboe_df[col].values
            except Exception as exc:
                log.warning("CBOE alpha features failed: %s", exc)

        # --- Insider features (US equity ETFs only) ---
        insider_cols = [c for c in INSIDER_FEATURES if c in feature_list]
        if insider_cols and symbol in ETF_COMPONENTS:
            try:
                insider_df = self._insider.build_features(bar_dates, symbol)
                for col in insider_cols:
                    if col in insider_df.columns:
                        df[col] = insider_df[col].values
            except Exception as exc:
                log.warning("Insider features failed for %s: %s", symbol, exc)

        # Fill any missing columns with NaN
        for col in feature_list:
            if col not in df.columns:
                df[col] = np.nan

        return df[feature_list]
