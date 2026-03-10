#!/usr/bin/env python3
"""
Options Spread + Directional Trader
=====================================
Activates when VIX >= 25 (equity regime-blocked). Two strategies run together:

1. Credit Spreads (premium selling)
   - Bull Put Spread: sell 0.25-delta put, buy 0.10-delta put (same expiry)
   - Bear Call Spread: sell 0.25-delta call, buy 0.10-delta call (RSI > 60)
   - Expiry: 30-46 DTE | Profit: 50% credit | Stop: 3× credit | Time: 21 DTE

2. Directional Options (trend following)
   - Activated when RSI > 65 (call) or RSI < 35 (put) + 5-day momentum confirms
   - Buys ITM call or put (~0.68 delta, 30-45 DTE)
   - Profit target: +60% premium | Stop loss: -30% | Time stop: 21 DTE

Usage (via main.py):
    python main.py spread-trade
    python main.py spread-trade --symbols QQQ,SPY,IWM --vix-threshold 25
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    ContractType, OrderClass, OrderSide, PositionIntent, TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest, LimitOrderRequest, MarketOrderRequest, OptionLegRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spread_trader")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE             = os.path.join(PROJECT_ROOT, "outputs", "open_spreads.json")
DIRECTIONAL_STATE_FILE = os.path.join(PROJECT_ROOT, "outputs", "open_directional.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS   = ["QQQ", "SPY", "IWM", "GLD", "SLV"]
SPREAD_WIDTHS     = {"QQQ": 5, "SPY": 5, "IWM": 5, "GLD": 5, "SLV": 2, "DEFAULT": 5}
VIX_THRESHOLD     = 25.0
VIX_SPIKE_SKIP    = 0.20   # skip entry if VIX jumped >20% today (wait for stabilization)
RSI_PERIOD        = 14
RSI_BULL_PUT_MAX  = 55.0   # underlying RSI must be below this for bull put entry
RSI_BEAR_CALL_MIN = 60.0   # underlying RSI must be above this for bear call entry
TARGET_SHORT_DELTA = 0.25
TARGET_LONG_DELTA  = 0.10
DTE_MIN           = 28
DTE_MAX           = 46
DTE_CLOSE         = 21
PROFIT_TARGET     = 0.50   # close at 50% of max credit
BREAKEVEN_TRIGGER = 0.25   # move stop to breakeven after 25% profit
STOP_MULT         = 3.0    # close if current debit = 3× entry credit (lost 2×)
MIN_CREDIT_PCT    = 0.20   # credit >= 20% of spread width required (raised from 15%)
MIN_LEG_BID       = 0.05   # both legs must have bid > this
MAX_RISK_PCT      = 0.01   # 1% account risk per trade
MAX_CONTRACTS     = 3      # hard cap per symbol
MAX_CONCURRENT    = 3      # max simultaneous open spreads across all symbols
LOOP_SLEEP_SEC    = 900    # check every 15 min during market hours
LCI_ATR_MIN_PCT   = 0.25   # LCI: last day's range must be >= 25% of ATR14 (real move)

# ---------------------------------------------------------------------------
# Directional (trend-following) options config
# ---------------------------------------------------------------------------
TREND_RSI_CALL_MIN   = 65.0   # RSI >= this → strong bullish → buy ITM call
TREND_RSI_PUT_MAX    = 35.0   # RSI <= this → strong bearish → buy ITM put
TREND_MOMENTUM_MIN   = 0.02   # 5-day return must be >= +/-2% to confirm trend
TREND_SMA_PERIOD     = 20     # price must be above/below SMA(20) for trend confirm
TARGET_TREND_DELTA   = 0.68   # target delta for ITM call/put (0.65-0.72 range)
TREND_PROFIT_TARGET  = 0.60   # close at +60% of entry premium
TREND_STOP_PCT       = 0.30   # close at -30% of entry premium
TREND_TIME_STOP_DTE  = 21     # close at 21 DTE (same as spread)
TREND_MAX_RISK_PCT   = 0.015  # 1.5% of equity per directional trade
TREND_MAX_CONTRACTS  = 2      # hard cap per position
TREND_MAX_CONCURRENT = 2      # max simultaneous open directional positions

# Scan/entry window — local computer time (PST):
#   6:25 AM → begin scanning (fetch data, compute signals, find legs) — no orders yet
#   6:35 AM → begin trading (submit spread orders)
#   8:00 AM → hard cutoff — no new entries after this (90-min window end = 11:00 ET)
SCAN_START        = (6, 25)   # (hour, minute) local time — scanning begins
ENTRY_START       = (6, 35)   # (hour, minute) local time — order submission begins
ENTRY_CUTOFF      = (8,  0)   # (hour, minute) local time — no new entries after this
# Market hours in local PST: 6:30 AM – 1:00 PM
MARKET_OPEN_LOCAL  = (6, 30)
MARKET_CLOSE_LOCAL = (13, 0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SpreadLeg:
    occ_symbol: str
    strike: float
    delta: float
    side: str  # "short" or "long"


@dataclass
class SpreadPosition:
    symbol: str
    direction: str          # "bull_put" or "bear_call"
    short_occ: str
    long_occ: str
    short_strike: float
    long_strike: float
    width: float
    entry_credit: float     # net credit per share (positive = received)
    num_contracts: int
    entry_date: str         # ISO date string
    expiry_date: str        # ISO date string
    entry_vix: float
    breakeven_stop_active: bool = False  # True after 25% profit — stop moves to entry credit


@dataclass
class DirectionalPosition:
    symbol: str
    direction: str      # "call" or "put"
    occ_symbol: str
    strike: float
    entry_price: float  # premium paid per share
    num_contracts: int
    entry_date: str     # ISO date string
    expiry_date: str    # ISO date string
    entry_vix: float
    entry_rsi: float


# ---------------------------------------------------------------------------
# Black-Scholes delta (fallback if Greeks missing from chain)
# ---------------------------------------------------------------------------
def _bs_put_delta(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Black-Scholes delta for a put option. Returns negative value."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) - 1.0  # put delta is negative


def _bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Black-Scholes delta for a call option. Returns positive value."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Return the latest RSI value."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _local_now() -> datetime:
    """Return current local time (computer's timezone — PST)."""
    return datetime.now()


def _is_market_open() -> bool:
    """Market open in local PST: Mon-Fri 6:30 AM – 1:00 PM."""
    now = _local_now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    open_min  = MARKET_OPEN_LOCAL[0]  * 60 + MARKET_OPEN_LOCAL[1]
    close_min = MARKET_CLOSE_LOCAL[0] * 60 + MARKET_CLOSE_LOCAL[1]
    return open_min <= t <= close_min


def _in_scan_window() -> bool:
    """True from 6:25 AM — scanning is active (data fetch + signal compute)."""
    now = _local_now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    scan_min   = SCAN_START[0]    * 60 + SCAN_START[1]
    cutoff_min = ENTRY_CUTOFF[0]  * 60 + ENTRY_CUTOFF[1]
    return scan_min <= t < cutoff_min


def _in_entry_window() -> bool:
    """True from 6:35 AM — order submission is active."""
    now = _local_now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    start_min  = ENTRY_START[0]   * 60 + ENTRY_START[1]
    cutoff_min = ENTRY_CUTOFF[0]  * 60 + ENTRY_CUTOFF[1]
    return start_min <= t < cutoff_min


def _days_to_expiry(expiry_str: str) -> int:
    expiry = date.fromisoformat(expiry_str)
    return (expiry - date.today()).days


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def _load_state() -> Dict[str, SpreadPosition]:
    """Load open spreads from disk. Returns {short_occ: SpreadPosition}."""
    try:
        with open(STATE_FILE) as f:
            raw = json.load(f)
        result = {}
        for k, v in raw.items():
            result[k] = SpreadPosition(**v)
        return result
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return {}


def _save_state(spreads: Dict[str, SpreadPosition]) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({k: asdict(v) for k, v in spreads.items()}, f, indent=2)


def _load_directional_state() -> Dict[str, DirectionalPosition]:
    try:
        with open(DIRECTIONAL_STATE_FILE) as f:
            raw = json.load(f)
        return {k: DirectionalPosition(**v) for k, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return {}


def _save_directional_state(positions: Dict[str, DirectionalPosition]) -> None:
    os.makedirs(os.path.dirname(DIRECTIONAL_STATE_FILE), exist_ok=True)
    with open(DIRECTIONAL_STATE_FILE, "w") as f:
        json.dump({k: asdict(v) for k, v in positions.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class SpreadTrader:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str] = None,
        vix_threshold: float = VIX_THRESHOLD,
    ):
        self.trading_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
        self.option_data = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        self.stock_data  = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.vix_threshold = vix_threshold
        self._running = True
        self._open_spreads: Dict[str, SpreadPosition] = _load_state()
        self._open_directional: Dict[str, DirectionalPosition] = _load_directional_state()
        self._profitable_closes: int = 0   # starter sizing: scale up after first win
        self._profitable_directional: int = 0
        self._last_lci: dict = {}          # populated by _get_underlying each call
        self._last_bars: Dict[str, pd.DataFrame] = {}  # cached daily bars per symbol
        log.info("SpreadTrader init: %d symbols, VIX threshold=%.1f", len(self.symbols), vix_threshold)
        log.info("Loaded %d open spread(s) from state file.", len(self._open_spreads))
        log.info("Loaded %d open directional position(s).", len(self._open_directional))

    # ------------------------------------------------------------------ #
    # Market data helpers
    # ------------------------------------------------------------------ #
    def _get_vix(self) -> Tuple[float, float]:
        """Return (current_vix, 1d_change_pct). Uses yfinance fallback."""
        try:
            import yfinance as yf
            vix = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
            # Handle multi-level columns (newer yfinance versions)
            close = vix["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) >= 2:
                cur  = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                chg  = (cur - prev) / prev if prev > 0 else 0.0
                return cur, chg
            elif len(close) == 1:
                return float(close.iloc[-1]), 0.0
        except Exception as exc:
            log.warning("VIX fetch failed: %s", exc)
        return 25.0, 0.0  # safe default: treat as elevated

    def _get_underlying(self, symbol: str) -> Tuple[float, float, bool]:
        """Return (current_price, rsi14, lci_confirmed).

        lci_confirmed: True if yesterday's candle range >= 25% of ATR14
        (Liquidity Candle Index — confirms a real move, not noise).
        For bull put: also requires the last day was a down day (close < open).
        For bear call: last day was an up day (close > open).
        """
        try:
            # Latest price
            q = self.stock_data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            price = float((q[symbol].ask_price + q[symbol].bid_price) / 2)

            # Daily bars for RSI + LCI
            bars = self.stock_data.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now(timezone.utc) - timedelta(days=45),
            ))
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")
            df = df.sort_index()

            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)
            open_ = df["open"].astype(float)

            rsi = _compute_rsi(close)

            # LCI: compute ATR14 and check last candle's range
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
            last_range = float(high.iloc[-1] - low.iloc[-1])
            last_down  = float(close.iloc[-1]) < float(open_.iloc[-1])
            last_up    = float(close.iloc[-1]) > float(open_.iloc[-1])

            lci_range_ok = atr14 > 0 and (last_range / atr14) >= LCI_ATR_MIN_PCT
            # Store raw flags for direction check in caller
            self._last_lci = {
                "range_ok": lci_range_ok,
                "last_down": last_down,
                "last_up":   last_up,
                "atr14":     atr14,
                "last_range": last_range,
            }
            # Cache daily bars for trend signal computation
            self._last_bars[symbol] = df
            return price, rsi, lci_range_ok
        except Exception as exc:
            log.warning("Underlying fetch failed for %s: %s", symbol, exc)
            self._last_lci = {"range_ok": True, "last_down": True, "last_up": True, "atr14": 0, "last_range": 0}
            return 0.0, 50.0, True  # fail open

    def _get_account_equity(self) -> float:
        try:
            return float(self.trading_client.get_account().equity)
        except Exception:
            return 97000.0

    # ------------------------------------------------------------------ #
    # Options chain: find best legs
    # ------------------------------------------------------------------ #
    def _find_spread_legs(
        self, symbol: str, current_price: float, direction: str, vix: float
    ) -> Optional[Tuple[SpreadLeg, SpreadLeg, float, str]]:
        """
        Find the short and long legs for a credit spread.
        Returns (short_leg, long_leg, net_credit_per_share, expiry_date) or None.
        """
        today = date.today()
        exp_min = today + timedelta(days=DTE_MIN)
        exp_max = today + timedelta(days=DTE_MAX)
        contract_type = ContractType.PUT if direction == "bull_put" else ContractType.CALL
        width = SPREAD_WIDTHS.get(symbol, SPREAD_WIDTHS["DEFAULT"])

        try:
            chain = self.option_data.get_option_chain(OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=exp_min,
                expiration_date_lte=exp_max,
                type=contract_type.value,
            ))
        except Exception as exc:
            log.warning("Chain fetch failed for %s: %s", symbol, exc)
            return None

        if not chain:
            log.info("No %s chain data for %s in %s-%s DTE window.", direction, symbol, DTE_MIN, DTE_MAX)
            return None

        # Use IV from VIX (proxy) for BS fallback
        sigma = vix / 100.0

        # Build candidate list: (occ_symbol, expiry, strike, delta, bid, ask)
        candidates = []
        for occ_sym, snap in chain.items():
            try:
                # Parse OCC symbol to get expiry + strike
                # Format: ROOT + YYMMDD + C/P + STRIKE*1000 (8 digits)
                root_len = len(symbol)
                date_str = occ_sym[root_len: root_len + 6]
                exp_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
                strike = int(occ_sym[-8:]) / 1000.0

                # Delta: prefer chain Greeks, fall back to BS
                if snap.greeks and snap.greeks.delta is not None:
                    delta = float(snap.greeks.delta)
                else:
                    T = (exp_date - today).days / 365.0
                    if direction == "bull_put":
                        delta = _bs_put_delta(current_price, strike, T, sigma)
                    else:
                        delta = _bs_call_delta(current_price, strike, T, sigma)

                bid = float(snap.latest_quote.bid_price) if snap.latest_quote else 0.0
                ask = float(snap.latest_quote.ask_price) if snap.latest_quote else 0.0

                # Filter: must have reasonable bid
                if bid < MIN_LEG_BID:
                    continue

                candidates.append({
                    "occ": occ_sym, "expiry": exp_date, "strike": strike,
                    "delta": delta, "bid": bid, "ask": ask,
                    "mid": (bid + ask) / 2,
                })
            except (ValueError, IndexError):
                continue

        if not candidates:
            log.info("No tradeable %s contracts for %s.", direction, symbol)
            return None

        # For bull put: want puts with delta closest to -TARGET_SHORT_DELTA (-0.25)
        # For bear call: want calls with delta closest to +TARGET_SHORT_DELTA (+0.25)
        target_short = -TARGET_SHORT_DELTA if direction == "bull_put" else TARGET_SHORT_DELTA
        target_long  = -TARGET_LONG_DELTA  if direction == "bull_put" else TARGET_LONG_DELTA

        # Sort candidates: group by expiry, pick the best DTE first
        # We want the expiry closest to 38 DTE (midpoint of 28-46)
        target_dte = (DTE_MIN + DTE_MAX) / 2
        by_expiry: Dict[date, list] = {}
        for c in candidates:
            by_expiry.setdefault(c["expiry"], []).append(c)

        # Try each expiry (closest to target first)
        sorted_expiries = sorted(by_expiry.keys(), key=lambda d: abs((d - today).days - target_dte))

        for expiry in sorted_expiries:
            exp_contracts = by_expiry[expiry]

            # Find short leg: closest to target_short delta
            short = min(exp_contracts, key=lambda c: abs(c["delta"] - target_short))

            # Find long leg: the contract at short_strike ± width (exact spread width)
            # Use strike proximity — not delta — to enforce the $5/$2 spread width
            target_long_strike = (
                short["strike"] - width if direction == "bull_put"
                else short["strike"] + width
            )
            long_candidates = [
                c for c in exp_contracts
                if abs(c["strike"] - target_long_strike) <= width * 0.6  # within 60% of width
            ]
            if not long_candidates:
                log.info(
                    "%s %s: no long leg near strike %.0f (short=%.0f, width=%.0f) for expiry %s",
                    symbol, direction, target_long_strike, short["strike"], width, expiry,
                )
                continue

            long = min(long_candidates, key=lambda c: abs(c["strike"] - target_long_strike))

            # Compute net credit (sell short mid, buy long mid)
            net_credit = short["mid"] - long["mid"]
            actual_width = abs(short["strike"] - long["strike"])

            if actual_width <= 0:
                continue

            # Enforce minimum credit
            if net_credit < MIN_CREDIT_PCT * actual_width:
                log.info(
                    "%s %s: credit $%.2f too low vs width $%.0f (need %.0f%%)",
                    symbol, direction, net_credit, actual_width, MIN_CREDIT_PCT * 100
                )
                continue

            # Enforce both legs have real bid
            if short["bid"] < MIN_LEG_BID or long["bid"] < MIN_LEG_BID:
                continue

            short_leg = SpreadLeg(short["occ"], short["strike"], short["delta"], "short")
            long_leg  = SpreadLeg(long["occ"],  long["strike"],  long["delta"],  "long")
            expiry_str = expiry.isoformat()

            log.info(
                "%s %s: short %s (d=%.3f) / long %s (d=%.3f) | credit=$%.2f width=$%.0f expiry=%s",
                symbol, direction,
                short_leg.occ_symbol, short_leg.delta,
                long_leg.occ_symbol,  long_leg.delta,
                net_credit, actual_width, expiry_str,
            )
            return short_leg, long_leg, net_credit, expiry_str

        log.info("No valid spread found for %s %s.", symbol, direction)
        return None

    # ------------------------------------------------------------------ #
    # Order execution
    # ------------------------------------------------------------------ #
    def _submit_spread(
        self,
        short_leg: SpreadLeg,
        long_leg: SpreadLeg,
        num_contracts: int,
        net_credit: float,
    ) -> bool:
        """Submit a 2-leg credit spread order. Returns True on success."""
        limit_price = round(net_credit, 2)
        try:
            order = self.trading_client.submit_order(
                LimitOrderRequest(
                    qty=num_contracts,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    legs=[
                        OptionLegRequest(
                            symbol=short_leg.occ_symbol,
                            ratio_qty=1,
                            position_intent=PositionIntent.SELL_TO_OPEN,
                        ),
                        OptionLegRequest(
                            symbol=long_leg.occ_symbol,
                            ratio_qty=1,
                            position_intent=PositionIntent.BUY_TO_OPEN,
                        ),
                    ],
                )
            )
            log.info("SPREAD OPEN order submitted: %s x%d @ $%.2f credit — order %s",
                     short_leg.occ_symbol, num_contracts, limit_price, order.id)
            return True
        except Exception as exc:
            log.error("Failed to submit spread: %s", exc)
            return False

    def _close_spread(self, spread: SpreadPosition, reason: str) -> bool:
        """Close both legs of an existing spread. Returns True if both legs closed."""
        log.info("Closing %s %s spread (reason: %s)...", spread.symbol, spread.direction, reason)
        success = True
        for occ_sym, intent, side_label in [
            (spread.short_occ, PositionIntent.BUY_TO_CLOSE, "short"),
            (spread.long_occ,  PositionIntent.SELL_TO_CLOSE, "long"),
        ]:
            try:
                self.trading_client.close_position(occ_sym)
                log.info("  Closed %s leg: %s", side_label, occ_sym)
            except Exception as exc:
                log.warning("  Could not close %s leg %s: %s", side_label, occ_sym, exc)
                success = False
        return success

    # ------------------------------------------------------------------ #
    # Position monitoring
    # ------------------------------------------------------------------ #
    def _get_account_option_positions(self) -> Dict[str, dict]:
        """Return {occ_symbol: {qty, side, entry_price, current_price}}."""
        result = {}
        try:
            for pos in self.trading_client.get_all_positions():
                sym = pos.symbol
                # OCC symbols: ROOT + 6-digit date + C/P + 8-digit strike
                import re
                if not re.match(r"^[A-Z]+\d{6}[CP]\d{8}$", sym):
                    continue
                result[sym] = {
                    "qty": float(pos.qty),
                    "side": "LONG" if float(pos.qty) > 0 else "SHORT",
                    "entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pnl": float(pos.unrealized_pl),
                }
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
        return result

    def _get_spread_mark(self, spread: SpreadPosition, positions: Dict[str, dict]) -> Optional[float]:
        """
        Estimate current spread debit (cost to close).
        For a bull put spread: short put is NEGATIVE qty, long put POSITIVE.
        Current debit = long_mid - short_mid  (what we'd pay to close).
        """
        try:
            # Get current mid prices from latest quotes
            from alpaca.data.requests import OptionSnapshotRequest
            snaps = self.option_data.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=[spread.short_occ, spread.long_occ])
            )
            short_mid = None
            long_mid  = None
            if spread.short_occ in snaps and snaps[spread.short_occ].latest_quote:
                q = snaps[spread.short_occ].latest_quote
                short_mid = (q.bid_price + q.ask_price) / 2
            if spread.long_occ in snaps and snaps[spread.long_occ].latest_quote:
                q = snaps[spread.long_occ].latest_quote
                long_mid = (q.bid_price + q.ask_price) / 2

            if short_mid is not None and long_mid is not None:
                # Current debit to close = buy back short (cost) - sell long (receive)
                return float(short_mid - long_mid)

        except Exception as exc:
            log.debug("Mark fetch failed: %s", exc)

        # Fallback: use position P&L from Alpaca
        pos_short = positions.get(spread.short_occ)
        pos_long  = positions.get(spread.long_occ)
        if pos_short and pos_long:
            short_cur = abs(pos_short["current_price"])
            long_cur  = abs(pos_long["current_price"])
            return float(short_cur - long_cur)

        return None

    def _check_exit(self, spread: SpreadPosition, current_debit: float) -> Optional[str]:
        """Return exit reason string, or None if should hold.

        Exit hierarchy:
        1. Time stop (21 DTE) — always wins
        2. Profit target (50% of credit)
        3. Breakeven stop — activates after 25% profit; stop = entry credit
        4. Hard stop loss (3× credit)
        """
        dte = _days_to_expiry(spread.expiry_date)

        # 1. Time stop
        if dte <= DTE_CLOSE:
            return f"time_stop ({dte} DTE <= {DTE_CLOSE})"

        # 2. Profit target: current debit <= 50% of entry credit
        profit_target_debit = spread.entry_credit * (1 - PROFIT_TARGET)
        if current_debit <= profit_target_debit:
            pnl = (spread.entry_credit - current_debit) * 100 * spread.num_contracts
            return f"profit_target (+${pnl:.0f}, {PROFIT_TARGET:.0%} of credit)"

        # 3. Breakeven stop: activate after 25% profit, stop at entry credit
        pnl_pct = (spread.entry_credit - current_debit) / spread.entry_credit
        if not spread.breakeven_stop_active and pnl_pct >= BREAKEVEN_TRIGGER:
            spread.breakeven_stop_active = True
            _save_state(self._open_spreads)
            log.info("  %s: breakeven stop ACTIVATED (profit=%.1f%% >= %.0f%%)",
                     spread.symbol, pnl_pct * 100, BREAKEVEN_TRIGGER * 100)

        if spread.breakeven_stop_active and current_debit >= spread.entry_credit:
            loss = (current_debit - spread.entry_credit) * 100 * spread.num_contracts
            return f"breakeven_stop (gave back profit, debit={current_debit:.2f} >= entry={spread.entry_credit:.2f}, -${loss:.0f})"

        # 4. Hard stop loss: current debit >= 3× entry credit
        stop_debit = spread.entry_credit * STOP_MULT
        if current_debit >= stop_debit:
            loss = (current_debit - spread.entry_credit) * 100 * spread.num_contracts
            return f"stop_loss (-${loss:.0f}, debit={current_debit:.2f} >= {stop_debit:.2f})"

        return None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        log.info("SpreadTrader running. Symbols: %s", ", ".join(self.symbols))

        def _handle_signal(sig, frame):
            print("\nShutting down spread trader...")
            self._running = False

        signal.signal(signal.SIGINT, _handle_signal)

        while self._running:
            try:
                self._cycle()
            except Exception as exc:
                log.error("Cycle error: %s", exc, exc_info=True)

            # Sleep until next check
            log.info("Sleeping %d min until next check...", LOOP_SLEEP_SEC // 60)
            for _ in range(LOOP_SLEEP_SEC):
                if not self._running:
                    break
                time.sleep(1)

    def _cycle(self) -> None:
        now = _local_now()
        log.info("=== SpreadTrader cycle @ %s local ===", now.strftime("%Y-%m-%d %H:%M"))

        if now.weekday() >= 5:
            log.info("Weekend — skipping.")
            return

        if not _is_market_open():
            log.info("Market closed (local time %s, market hours 6:30-13:00) — skipping.", now.strftime("%H:%M"))
            return

        # Sync open spreads + directional positions against actual account
        positions = self._get_account_option_positions()
        self._sync_state(positions)
        self._sync_directional_state(positions)

        # --- Exit management: directional positions ---
        dir_to_close = []
        for occ, pos in list(self._open_directional.items()):
            dte = _days_to_expiry(pos.expiry_date)
            current_mark = self._get_directional_mark(pos)
            if current_mark is None:
                log.info("  DIRECTIONAL %s %s: mark unavailable (DTE=%d)", pos.symbol, pos.direction, dte)
                continue
            pnl_pct = (current_mark - pos.entry_price) / pos.entry_price
            log.info(
                "  DIRECTIONAL %s %s: entry=$%.2f current=$%.2f P&L=%.1f%% DTE=%d",
                pos.symbol, pos.direction, pos.entry_price, current_mark, pnl_pct * 100, dte,
            )
            reason = self._check_directional_exit(pos, current_mark)
            if reason:
                dir_to_close.append((occ, pos, reason))

        for occ, pos, reason in dir_to_close:
            if self._close_directional_option(pos, reason):
                if "profit_target" in reason:
                    self._profitable_directional += 1
                del self._open_directional[occ]
                _save_directional_state(self._open_directional)

        # --- Exit management: check all open spreads ---
        to_close = []
        for short_occ, spread in list(self._open_spreads.items()):
            dte = _days_to_expiry(spread.expiry_date)
            current_debit = self._get_spread_mark(spread, positions)

            if current_debit is None:
                log.info("  %s %s: mark unavailable (DTE=%d)", spread.symbol, spread.direction, dte)
                continue

            pnl_pct = (spread.entry_credit - current_debit) / spread.entry_credit
            log.info(
                "  %s %s: entry_credit=%.2f current_debit=%.2f P&L=%.1f%% DTE=%d",
                spread.symbol, spread.direction,
                spread.entry_credit, current_debit, pnl_pct * 100, dte,
            )

            reason = self._check_exit(spread, current_debit)
            if reason:
                to_close.append((short_occ, spread, reason))

        for short_occ, spread, reason in to_close:
            if self._close_spread(spread, reason):
                # Track profitable closes for starter sizing
                if "profit_target" in reason:
                    self._profitable_closes += 1
                    log.info("Profitable close #%d — full sizing now unlocked.", self._profitable_closes)
                del self._open_spreads[short_occ]
                _save_state(self._open_spreads)

        # --- Entry: only within scan/entry windows ---
        if not _in_scan_window():
            log.info("Outside scan window (6:25–8:00 AM local) — %s. Monitoring open positions only.",
                     _local_now().strftime("%H:%M"))
            return

        if len(self._open_spreads) >= MAX_CONCURRENT:
            log.info("Max concurrent spreads (%d) reached — no new entries.", MAX_CONCURRENT)
            return

        # VIX gate
        vix, vix_1d_chg = self._get_vix()
        log.info("VIX=%.1f (1d chg=%.1f%%)", vix, vix_1d_chg * 100)

        if vix < self.vix_threshold:
            log.info("VIX %.1f < threshold %.1f — equity trader active, no spread entry.", vix, self.vix_threshold)
            return

        if vix_1d_chg > VIX_SPIKE_SKIP:
            log.info("VIX spiked %.1f%% today — waiting for stabilization before entering.", vix_1d_chg * 100)
            return

        equity = self._get_account_equity()
        log.info("Account equity: $%.2f | Profitable closes: %d (starter sizing: %s)",
                 equity, self._profitable_closes,
                 "FULL" if self._profitable_closes > 0 else "STARTER (1 contract)")

        symbols_with_spreads = {s.symbol for s in self._open_spreads.values()}

        for symbol in self.symbols:
            if not self._running:
                break
            if len(self._open_spreads) >= MAX_CONCURRENT:
                break
            if symbol in symbols_with_spreads:
                log.info("  %s: already has open spread — skipping.", symbol)
                continue

            price, rsi, lci_ok = self._get_underlying(symbol)
            if price <= 0:
                continue

            lci = self._last_lci
            log.info(
                "  %s: price=%.2f RSI=%.1f LCI=range%.1f%%ATR(%s) last_down=%s last_up=%s",
                symbol, price, rsi,
                (lci.get("last_range", 0) / lci.get("atr14", 1) * 100) if lci.get("atr14") else 0,
                "OK" if lci_ok else "FAIL",
                lci.get("last_down"), lci.get("last_up"),
            )

            # Determine direction
            if rsi <= RSI_BULL_PUT_MAX:
                direction = "bull_put"
                # LCI: confirm a real down move happened (last day was red + big range)
                if not (lci_ok and lci.get("last_down", False)):
                    log.info(
                        "  %s: LCI filter failed for bull_put (range_ok=%s, last_down=%s) — skip.",
                        symbol, lci_ok, lci.get("last_down")
                    )
                    continue
            elif rsi >= RSI_BEAR_CALL_MIN:
                direction = "bear_call"
                # LCI: confirm a real up move happened (last day was green + big range)
                if not (lci_ok and lci.get("last_up", False)):
                    log.info(
                        "  %s: LCI filter failed for bear_call (range_ok=%s, last_up=%s) — skip.",
                        symbol, lci_ok, lci.get("last_up")
                    )
                    continue
            else:
                log.info("  %s: RSI %.1f neutral (%.0f-%.0f) — skip.", symbol, rsi, RSI_BULL_PUT_MAX, RSI_BEAR_CALL_MIN)
                continue

            # Find legs
            result = self._find_spread_legs(symbol, price, direction, vix)
            if result is None:
                continue

            short_leg, long_leg, net_credit, expiry_str = result
            width = abs(short_leg.strike - long_leg.strike)

            # Sizing: starter (1 contract) until first profitable close, then full
            max_risk = equity * MAX_RISK_PCT
            max_loss_per_contract = (width - net_credit) * 100
            if max_loss_per_contract <= 0:
                log.warning("  %s: invalid spread (credit >= width) — skip.", symbol)
                continue

            full_contracts = max(1, min(MAX_CONTRACTS, int(max_risk / max_loss_per_contract)))
            num_contracts = full_contracts if self._profitable_closes > 0 else 1

            log.info(
                "  %s %s: entering %d contract(s) [%s] | credit=$%.2f width=$%.0f | risk=$%.0f",
                symbol, direction, num_contracts,
                "full" if self._profitable_closes > 0 else "starter",
                net_credit, width, max_loss_per_contract * num_contracts
            )

            if not _in_entry_window():
                log.info(
                    "  %s %s: spread ready but trade window not open yet (opens 6:35 AM) — scanning only.",
                    symbol, direction
                )
                continue

            if self._submit_spread(short_leg, long_leg, num_contracts, net_credit):
                spread = SpreadPosition(
                    symbol=symbol,
                    direction=direction,
                    short_occ=short_leg.occ_symbol,
                    long_occ=long_leg.occ_symbol,
                    short_strike=short_leg.strike,
                    long_strike=long_leg.strike,
                    width=width,
                    entry_credit=net_credit,
                    num_contracts=num_contracts,
                    entry_date=date.today().isoformat(),
                    expiry_date=expiry_str,
                    entry_vix=vix,
                    breakeven_stop_active=False,
                )
                self._open_spreads[short_leg.occ_symbol] = spread
                _save_state(self._open_spreads)
                symbols_with_spreads.add(symbol)

        # --- Directional entry: strong trend detected ---
        log.info("--- Directional scan (open=%d / max=%d) ---",
                 len(self._open_directional), TREND_MAX_CONCURRENT)

        if len(self._open_directional) >= TREND_MAX_CONCURRENT:
            log.info("Max directional positions (%d) reached — skipping.", TREND_MAX_CONCURRENT)
            return

        symbols_with_directional = {p.symbol for p in self._open_directional.values()}

        for symbol in self.symbols:
            if not self._running:
                break
            if len(self._open_directional) >= TREND_MAX_CONCURRENT:
                break
            if symbol in symbols_with_directional:
                log.info("  %s: already has directional position — skip.", symbol)
                continue

            # Re-use already-fetched data (price/rsi/bars from spread scan above)
            price, rsi, _ = self._get_underlying(symbol)
            if price <= 0:
                continue

            trend_dir, trend_reason = self._get_trend_signal(symbol, price, rsi)
            if trend_dir is None:
                log.info("  %s: no trend signal (%s)", symbol, trend_reason)
                continue

            log.info("  %s: TREND SIGNAL %s — %s", symbol, trend_dir.upper(), trend_reason)

            if not _in_entry_window():
                log.info("  %s: trend ready but entry window not open yet (opens 6:35 AM).", symbol)
                continue

            result = self._find_directional_option(symbol, trend_dir, price, vix)
            if result is None:
                continue

            occ_sym, strike, mid_price, expiry_str = result

            # Sizing: starter (1 contract) until first profitable close
            max_risk = equity * TREND_MAX_RISK_PCT
            cost_per_contract = mid_price * 100
            if cost_per_contract <= 0:
                continue

            full_contracts = max(1, min(TREND_MAX_CONTRACTS, int(max_risk / cost_per_contract)))
            num_contracts = full_contracts if self._profitable_directional > 0 else 1

            log.info(
                "  %s %s: entering %d contract(s) [%s] | strike=%.1f premium=$%.2f | risk=$%.0f",
                symbol, trend_dir, num_contracts,
                "full" if self._profitable_directional > 0 else "starter",
                strike, mid_price, cost_per_contract * num_contracts,
            )

            if self._submit_directional_option(occ_sym, num_contracts, mid_price):
                pos = DirectionalPosition(
                    symbol=symbol,
                    direction=trend_dir,
                    occ_symbol=occ_sym,
                    strike=strike,
                    entry_price=mid_price,
                    num_contracts=num_contracts,
                    entry_date=date.today().isoformat(),
                    expiry_date=expiry_str,
                    entry_vix=vix,
                    entry_rsi=rsi,
                )
                self._open_directional[occ_sym] = pos
                _save_directional_state(self._open_directional)
                symbols_with_directional.add(symbol)

    def _sync_state(self, positions: Dict[str, dict]) -> None:
        """Remove from state any spread whose short leg is no longer in account."""
        to_remove = []
        for short_occ, spread in self._open_spreads.items():
            if short_occ not in positions and spread.long_occ not in positions:
                log.info("Spread %s no longer in account — removing from state.", short_occ)
                to_remove.append(short_occ)
        for k in to_remove:
            del self._open_spreads[k]
        if to_remove:
            _save_state(self._open_spreads)

    # ------------------------------------------------------------------ #
    # Directional options: trend signal
    # ------------------------------------------------------------------ #
    def _get_trend_signal(
        self, symbol: str, price: float, rsi: float
    ) -> Tuple[Optional[str], str]:
        """
        Return (direction, reason) or (None, reason).
        direction: "call" | "put" | None
        Requires: RSI extreme + price above/below SMA20 + 5-day momentum >= 2%.
        """
        df = self._last_bars.get(symbol)
        if df is None or len(df) < TREND_SMA_PERIOD + 5:
            return None, "insufficient bar data"

        close = df["close"].astype(float)
        sma20 = float(close.rolling(TREND_SMA_PERIOD).mean().iloc[-1])
        ret5  = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) if len(close) >= 6 else 0.0

        if rsi >= TREND_RSI_CALL_MIN and price > sma20 and ret5 >= TREND_MOMENTUM_MIN:
            return "call", (
                f"RSI={rsi:.0f}>={TREND_RSI_CALL_MIN}, "
                f"price={price:.2f}>SMA20={sma20:.2f}, "
                f"ret5={ret5*100:.1f}%>={TREND_MOMENTUM_MIN*100:.0f}%"
            )
        if rsi <= TREND_RSI_PUT_MAX and price < sma20 and ret5 <= -TREND_MOMENTUM_MIN:
            return "put", (
                f"RSI={rsi:.0f}<={TREND_RSI_PUT_MAX}, "
                f"price={price:.2f}<SMA20={sma20:.2f}, "
                f"ret5={ret5*100:.1f}%<=-{TREND_MOMENTUM_MIN*100:.0f}%"
            )

        reasons = []
        if not (rsi >= TREND_RSI_CALL_MIN or rsi <= TREND_RSI_PUT_MAX):
            reasons.append(f"RSI={rsi:.0f} not extreme (need >={TREND_RSI_CALL_MIN} or <={TREND_RSI_PUT_MAX})")
        else:
            reasons.append(f"5d ret={ret5*100:.1f}% or price vs SMA20={sma20:.2f} not confirming")
        return None, "; ".join(reasons)

    # ------------------------------------------------------------------ #
    # Directional options: find ITM contract
    # ------------------------------------------------------------------ #
    def _find_directional_option(
        self, symbol: str, direction: str, price: float, vix: float
    ) -> Optional[Tuple[str, float, float, str]]:
        """
        Find the best ITM call or put for directional trade.
        Returns (occ_symbol, strike, mid_price, expiry_str) or None.
        Target delta: ~0.68 (ITM but not deep ITM).
        """
        today = date.today()
        exp_min = today + timedelta(days=DTE_MIN)
        exp_max = today + timedelta(days=DTE_MAX)
        contract_type = ContractType.CALL if direction == "call" else ContractType.PUT

        try:
            chain = self.option_data.get_option_chain(OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=exp_min,
                expiration_date_lte=exp_max,
                type=contract_type.value,
            ))
        except Exception as exc:
            log.warning("Directional chain fetch failed %s %s: %s", symbol, direction, exc)
            return None

        if not chain:
            log.info("No %s chain data for %s.", direction, symbol)
            return None

        sigma = vix / 100.0
        candidates = []
        for occ_sym, snap in chain.items():
            try:
                root_len = len(symbol)
                date_str = occ_sym[root_len: root_len + 6]
                exp_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
                strike = int(occ_sym[-8:]) / 1000.0

                if snap.greeks and snap.greeks.delta is not None:
                    delta = float(snap.greeks.delta)
                else:
                    T = (exp_date - today).days / 365.0
                    delta = (_bs_call_delta(price, strike, T, sigma) if direction == "call"
                             else _bs_put_delta(price, strike, T, sigma))

                bid = float(snap.latest_quote.bid_price) if snap.latest_quote else 0.0
                ask = float(snap.latest_quote.ask_price) if snap.latest_quote else 0.0
                mid = (bid + ask) / 2

                if bid < MIN_LEG_BID or mid <= 0:
                    continue

                candidates.append({
                    "occ": occ_sym, "expiry": exp_date, "strike": strike,
                    "delta": delta, "mid": mid,
                })
            except (ValueError, IndexError):
                continue

        if not candidates:
            return None

        # Target delta: +0.68 for call, -0.68 for put
        target = TARGET_TREND_DELTA if direction == "call" else -TARGET_TREND_DELTA
        target_dte = (DTE_MIN + DTE_MAX) / 2

        # Group by expiry, find best expiry first, then closest delta
        by_expiry: Dict[date, list] = {}
        for c in candidates:
            by_expiry.setdefault(c["expiry"], []).append(c)

        sorted_expiries = sorted(by_expiry.keys(), key=lambda d: abs((d - today).days - target_dte))
        for expiry in sorted_expiries:
            best = min(by_expiry[expiry], key=lambda c: abs(c["delta"] - target))
            log.info(
                "%s directional %s: %s strike=%.1f delta=%.3f mid=$%.2f expiry=%s",
                symbol, direction, best["occ"], best["strike"], best["delta"], best["mid"], expiry,
            )
            return best["occ"], best["strike"], best["mid"], expiry.isoformat()

        return None

    # ------------------------------------------------------------------ #
    # Directional options: order execution
    # ------------------------------------------------------------------ #
    def _submit_directional_option(
        self, occ_symbol: str, num_contracts: int, limit_price: float
    ) -> bool:
        """Buy a single-leg ITM option. Returns True on success."""
        try:
            order = self.trading_client.submit_order(
                LimitOrderRequest(
                    symbol=occ_symbol,
                    qty=num_contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                )
            )
            log.info("DIRECTIONAL BUY order: %s x%d @ $%.2f — order %s",
                     occ_symbol, num_contracts, limit_price, order.id)
            return True
        except Exception as exc:
            log.error("Failed to submit directional order: %s", exc)
            return False

    def _close_directional_option(self, pos: DirectionalPosition, reason: str) -> bool:
        """Market-close the directional option position."""
        log.info("Closing directional %s %s (reason: %s)...", pos.symbol, pos.direction, reason)
        try:
            self.trading_client.close_position(pos.occ_symbol)
            log.info("  Closed directional %s", pos.occ_symbol)
            return True
        except Exception as exc:
            log.warning("  Could not close directional %s: %s", pos.occ_symbol, exc)
            return False

    # ------------------------------------------------------------------ #
    # Directional options: exit checks
    # ------------------------------------------------------------------ #
    def _get_directional_mark(self, pos: DirectionalPosition) -> Optional[float]:
        """Return current mid-price per share for the directional option."""
        try:
            from alpaca.data.requests import OptionSnapshotRequest
            snaps = self.option_data.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=[pos.occ_symbol])
            )
            if pos.occ_symbol in snaps and snaps[pos.occ_symbol].latest_quote:
                q = snaps[pos.occ_symbol].latest_quote
                return float((q.bid_price + q.ask_price) / 2)
        except Exception as exc:
            log.debug("Directional mark fetch failed: %s", exc)
        return None

    def _check_directional_exit(self, pos: DirectionalPosition, current_price: float) -> Optional[str]:
        """
        Return exit reason or None.
        Exit hierarchy:
        1. Time stop (21 DTE)
        2. Profit target (+60% of entry premium)
        3. Stop loss (-30% of entry premium)
        """
        dte = _days_to_expiry(pos.expiry_date)
        if dte <= TREND_TIME_STOP_DTE:
            return f"time_stop ({dte} DTE <= {TREND_TIME_STOP_DTE})"

        pnl_pct = (current_price - pos.entry_price) / pos.entry_price

        if pnl_pct >= TREND_PROFIT_TARGET:
            pnl_usd = (current_price - pos.entry_price) * 100 * pos.num_contracts
            return f"profit_target (+{pnl_pct*100:.0f}%, +${pnl_usd:.0f})"

        if pnl_pct <= -TREND_STOP_PCT:
            loss_usd = (pos.entry_price - current_price) * 100 * pos.num_contracts
            return f"stop_loss ({pnl_pct*100:.0f}%, -${loss_usd:.0f})"

        return None

    def _sync_directional_state(self, positions: Dict[str, dict]) -> None:
        """Remove directional positions no longer held in the account."""
        to_remove = [occ for occ, pos in self._open_directional.items() if occ not in positions]
        for k in to_remove:
            log.info("Directional %s no longer in account — removing from state.", k)
            del self._open_directional[k]
        if to_remove:
            _save_directional_state(self._open_directional)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Options Credit Spread Trader")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated symbols (default: QQQ,SPY,IWM,GLD,SLV)")
    parser.add_argument("--vix-threshold", type=float, default=VIX_THRESHOLD,
                        help=f"VIX level to activate spread trading (default: {VIX_THRESHOLD})")
    args = parser.parse_args()

    # Group 3 (expansion account) credentials — fall back to main account if not set
    api_key    = (os.environ.get("ALPACA_EXPANSION_KEY")
                  or os.environ.get("ALPACA_API_KEY", "PKKWB6G2MJZZHYGEJXOAPEJTT7"))
    api_secret = (os.environ.get("ALPACA_EXPANSION_SECRET")
                  or os.environ.get("ALPACA_API_SECRET", "EGKKUKp73ZHcri7CHkpRezTEPWuWrPzyCm4UuE8P7Xg7"))

    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET env vars.")
        sys.exit(1)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    trader = SpreadTrader(api_key=api_key, api_secret=api_secret,
                         symbols=symbols, vix_threshold=args.vix_threshold)
    trader.run()


if __name__ == "__main__":
    main()
