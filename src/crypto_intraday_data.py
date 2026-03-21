#!/usr/bin/env python3
"""
Crypto Intraday Data Adapter
=============================
Fetches 5-minute OHLCV bars from BinanceUS (training) and Kraken (live).
Falls back to MEXC for training when a coin isn't listed on BinanceUS.

BinanceUS: 4+ years of history, 1000 bars/request, no API key needed.
MEXC: full historical 5-min data, 500 bars/request, no API key needed.
Kraken: ~720 bars (~2.5 days), used for live prediction to match execution prices.

Usage:
    from crypto_intraday_data import CryptoIntradayData
    cid = CryptoIntradayData()

    # Training: fetch 6 months of 5min bars (BinanceUS, MEXC fallback)
    df = cid.fetch_training_bars("BTC/USD", days=180)

    # Live: fetch latest bars from Kraken (for prediction)
    df = cid.fetch_live_bars("BTC/USD", limit=100)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import ccxt
import pandas as pd

log = logging.getLogger("crypto_intraday_data")

# Exchange-specific quote currencies
_BINANCEUS_QUOTE = "USDT"
_KRAKEN_QUOTE = "USD"
_MEXC_QUOTE = "USDT"

# Max bars per request
_BINANCEUS_LIMIT = 1000
_KRAKEN_LIMIT = 720
_MEXC_LIMIT = 500


def _normalize_symbol(symbol: str, exchange: str) -> str:
    """Convert generic symbol to exchange-specific format.

    Input: BTC/USD, BTC-USD, BTCUSD
    Output: BTC/USDT (BinanceUS, MEXC) or BTC/USD (Kraken)
    """
    # Extract base
    base = symbol.upper().replace("-", "/").replace("USDT", "USD")
    if "/" in base:
        base = base.split("/")[0]

    if exchange in ("binanceus", "mexc"):
        return f"{base}/{_BINANCEUS_QUOTE}"
    return f"{base}/{_KRAKEN_QUOTE}"


def _bars_to_df(bars: list, symbol: str) -> pd.DataFrame:
    """Convert ccxt OHLCV list to DataFrame."""
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["symbol"] = symbol
    df = df.drop(columns=["ts_ms"])
    df = df.sort_values("ts").reset_index(drop=True)
    # Drop duplicates by timestamp
    df = df.drop_duplicates(subset=["ts"], keep="last")
    return df


class CryptoIntradayData:
    """Fetch 5-minute crypto OHLCV bars from BinanceUS, MEXC, and Kraken."""

    def __init__(self):
        self._binanceus: Optional[ccxt.Exchange] = None
        self._kraken: Optional[ccxt.Exchange] = None
        self._mexc: Optional[ccxt.Exchange] = None

    def _get_binanceus(self) -> ccxt.Exchange:
        if self._binanceus is None:
            self._binanceus = ccxt.binanceus({"enableRateLimit": True})
        return self._binanceus

    def _get_kraken(self) -> ccxt.Exchange:
        if self._kraken is None:
            self._kraken = ccxt.kraken({"enableRateLimit": True})
        return self._kraken

    def _get_mexc(self) -> ccxt.Exchange:
        if self._mexc is None:
            self._mexc = ccxt.mexc({"enableRateLimit": True})
        return self._mexc

    def _fetch_paginated(
        self,
        exchange: "ccxt.Exchange",
        pair: str,
        interval: str,
        start_ts: int,
        end_ts: int,
        limit: int,
        source_name: str,
    ) -> list:
        """Paginate through an exchange's OHLCV endpoint.

        Returns list of raw ccxt bars [[ts_ms, o, h, l, c, v], ...].
        """
        all_bars = []
        since = start_ts
        request_count = 0

        while since < end_ts:
            try:
                bars = exchange.fetch_ohlcv(pair, interval, since=since, limit=limit)
                request_count += 1
            except Exception as exc:
                log.error("%s fetch failed for %s at %s: %s",
                          source_name, pair,
                          datetime.fromtimestamp(since / 1000, tz=timezone.utc), exc)
                break

            if not bars:
                break

            all_bars.extend(bars)

            last_ts = bars[-1][0]
            if last_ts <= since:
                break  # no progress
            since = last_ts + 1

            if request_count % 50 == 0:
                log.info("  %d requests, %d bars so far...", request_count, len(all_bars))

        return all_bars

    def fetch_training_bars(
        self,
        symbol: str,
        days: int = 180,
        interval: str = "5m",
        source: str = "auto",
    ) -> pd.DataFrame:
        """Fetch historical 5min bars for training.

        source: "auto" tries BinanceUS then MEXC fallback,
                "mexc" uses MEXC directly (for coins not on BinanceUS),
                "binanceus" skips fallback.

        MEXC is used as fallback because it has 6800+ markets, supports full
        historical 5-min pagination, and is accessible from the US without auth.
        Kraken is NOT used for training — its OHLCV ignores the `since` param
        and only returns the most recent ~720 bars.

        Returns DataFrame with columns: ts, open, high, low, close, volume, symbol
        """
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        all_bars = []
        pair = ""

        # Try BinanceUS first (unless source specifies otherwise)
        if source in ("auto", "binanceus"):
            exchange = self._get_binanceus()
            pair = _normalize_symbol(symbol, "binanceus")
            log.info("Fetching %s 5min bars from BinanceUS (%d days)...", pair, days)

            all_bars = self._fetch_paginated(
                exchange, pair, interval, start_ts, end_ts,
                _BINANCEUS_LIMIT, "BinanceUS",
            )

        # Fallback to MEXC if BinanceUS returned nothing
        if not all_bars and source in ("auto", "mexc"):
            if source == "auto":
                log.info("BinanceUS has no data for %s — falling back to MEXC...", symbol)
            exchange = self._get_mexc()
            pair = _normalize_symbol(symbol, "mexc")
            log.info("Fetching %s 5min bars from MEXC (%d days)...", pair, days)
            all_bars = self._fetch_paginated(
                exchange, pair, interval, start_ts, end_ts,
                _MEXC_LIMIT, "MEXC",
            )

        if not all_bars:
            log.warning("No bars fetched for %s", symbol)
            return pd.DataFrame()

        # Normalize symbol back to /USD format
        base = pair.split("/")[0]
        df = _bars_to_df(all_bars, f"{base}/USD")

        src_name = "MEXC" if exchange is self._mexc else "BinanceUS"
        log.info("Fetched %s: %d bars (%s -> %s) from %s",
                 symbol, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                 src_name)

        return df

    def fetch_live_bars(
        self,
        symbol: str,
        limit: int = 200,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch latest 5min bars from Kraken for live prediction.

        Uses Kraken to match execution prices. Max ~720 bars (~2.5 days).
        """
        exchange = self._get_kraken()
        pair = _normalize_symbol(symbol, "kraken")

        try:
            bars = exchange.fetch_ohlcv(pair, interval, limit=min(limit, _KRAKEN_LIMIT))
        except Exception as exc:
            log.error("Kraken fetch failed for %s: %s", pair, exc)
            return pd.DataFrame()

        if not bars:
            return pd.DataFrame()

        base = pair.split("/")[0]
        df = _bars_to_df(bars, f"{base}/USD")

        log.info("Live bars %s: %d bars (%s -> %s)",
                 symbol, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M"))

        return df

    def fetch_btc_bars(
        self,
        source: str = "kraken",
        limit: int = 200,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch BTC bars for cross-market features (BTC leads altcoins)."""
        if source == "kraken":
            return self.fetch_live_bars("BTC/USD", limit=limit, interval=interval)
        return self.fetch_training_bars("BTC/USD", days=max(1, limit // 288), interval=interval)

    def check_symbol_available(self, symbol: str, exchange: str = "binanceus") -> bool:
        """Check if a symbol is available on the exchange."""
        try:
            if exchange == "binanceus":
                ex = self._get_binanceus()
            elif exchange == "mexc":
                ex = self._get_mexc()
            else:
                ex = self._get_kraken()
            ex.load_markets()
            pair = _normalize_symbol(symbol, exchange)
            return pair in ex.markets
        except Exception:
            return False


def validate_bar_quality(df: pd.DataFrame, interval_minutes: int = 5) -> dict:
    """Check bar quality: gaps, zero-volume bars, consistency.

    Returns dict with quality metrics.
    """
    if df.empty:
        return {"valid": False, "reason": "empty dataframe"}

    n = len(df)
    expected_interval = timedelta(minutes=interval_minutes)

    # Check time gaps
    diffs = df["ts"].diff().dropna()
    expected_ms = interval_minutes * 60 * 1000
    gap_mask = diffs > (expected_interval * 1.5)
    n_gaps = int(gap_mask.sum())
    gap_pct = n_gaps / max(n - 1, 1)

    # Zero-volume bars
    zero_vol = int((df["volume"] == 0).sum())
    zero_vol_pct = zero_vol / n

    # Price consistency (close should be between low and high)
    invalid_ohlc = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())

    return {
        "valid": True,
        "total_bars": n,
        "gaps": n_gaps,
        "gap_pct": round(gap_pct, 4),
        "zero_volume_bars": zero_vol,
        "zero_volume_pct": round(zero_vol_pct, 4),
        "invalid_ohlc": invalid_ohlc,
        "date_range": f"{df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}",
    }
