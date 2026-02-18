#!/usr/bin/env python3
"""
Long ATM Straddle Options Trader — Real Alpaca Paper Trading
=============================================================
Profits from big moves in EITHER direction — no need to predict which way.

Strategy:
- Entry:  VIX daily change >= vix_spike_threshold (default 15%)
          Panic detected → big move likely, direction unknown → buy straddle
- Strike: ATM = nearest real contract to round(current_price)
- Expiry: real option contract ~30 calendar days out
- Exit 1: either leg reaches 1.8x its entry cost  (profit target +80%)
- Exit 2: total position value <= 0.6x total cost  (stop loss -40%)
- Exit 3: ML confidence >= 0.30 in one direction   (close losing leg, ride winner)

Usage (via main.py):
    python main.py trade-options --symbols SPY,QQQ --vix-spike-threshold 15
    python main.py trade-options --symbols SPY --max-risk 5000 --confidence 0.30

Required env vars: ALPACA_API_KEY, ALPACA_API_SECRET, FRED_API_KEY

Note: Paper trading accounts have automatic Level 3 options access.
      No personal info or options application required for paper trading.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest, MarketOrderRequest,
)
from alpaca.trading.enums import (
    ContractType, OrderSide, OrderType, TimeInForce, AssetClass,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.data.enums import OptionsFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from signals_engine import (
    DEFAULT_UNIVERSE, DAILY_LOOKBACK, build_adapter, FREDVixFetcher,
)
from ml_model import Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("options_trader")


# ===================================================================
# Module-level constants
# ===================================================================
VIX_SPIKE_THRESHOLD = 15.0   # % daily VIX change to trigger entry
PROFIT_TARGET_MULT  = 1.8    # close both legs when one leg reaches 1.8x entry cost
STOP_LOSS_MULT      = 0.6    # close both legs when total value drops to 0.6x total cost


# ===================================================================
# Math helpers (no scipy dependency)
# ===================================================================
def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ===================================================================
# IV Estimation from VIX history
# ===================================================================
class IVEstimator:
    """Estimate implied volatility environment using VIX as proxy."""

    def __init__(self, vix_df: pd.DataFrame):
        self._vix = vix_df

    def iv_rank(self) -> float:
        if self._vix.empty or len(self._vix) < 20:
            return 50.0
        vals = self._vix["vix"].tail(252)
        current = float(vals.iloc[-1])
        lo, hi = float(vals.min()), float(vals.max())
        if hi == lo:
            return 50.0
        return float(np.clip((current - lo) / (hi - lo) * 100, 0, 100))

    def current_vix(self) -> float:
        if self._vix.empty:
            return 20.0
        return float(self._vix["vix"].iloc[-1])

    def vix_change_pct(self) -> float:
        if len(self._vix) < 2:
            return 0.0
        cur  = float(self._vix["vix"].iloc[-1])
        prev = float(self._vix["vix"].iloc[-2])
        return ((cur - prev) / prev * 100) if prev != 0 else 0.0


# ===================================================================
# Black-Scholes Greeks (fallback pricing when live quote unavailable)
# ===================================================================
class GreeksEstimator:
    """Simple Black-Scholes approximations — used as fallback only."""

    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return GreeksEstimator._d1(S, K, T, r, sigma) - sigma * math.sqrt(T) if T > 0 else 0.0

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float = 0.05, sigma: float = 0.20) -> float:
        if T <= 0:
            return max(0.0, S - K)
        d1 = GreeksEstimator._d1(S, K, T, r, sigma)
        d2 = GreeksEstimator._d2(S, K, T, r, sigma)
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float = 0.05, sigma: float = 0.20) -> float:
        if T <= 0:
            return max(0.0, K - S)
        d1 = GreeksEstimator._d1(S, K, T, r, sigma)
        d2 = GreeksEstimator._d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ===================================================================
# Straddle Trade Logger (CSV persistence)
# ===================================================================
class OptionsTradeLogger:
    """Persist straddle trades to CSV for analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _path(self, symbol: str) -> str:
        return os.path.join(self.output_dir, f"straddle_trades_{symbol}.csv")

    def _base_row(self, symbol: str, straddle: dict) -> dict:
        return {
            "timestamp":         datetime.now().isoformat(),
            "symbol":            symbol,
            "action":            None,
            "strategy":          "LONG_STRADDLE",
            "strike":            straddle.get("strike"),
            "expiry":            straddle.get("expiry"),
            "call_contract":     straddle.get("call_contract"),
            "put_contract":      straddle.get("put_contract"),
            "entry_price":       straddle.get("entry_price"),
            "call_cost":         straddle.get("call_cost"),
            "put_cost":          straddle.get("put_cost"),
            "total_cost":        straddle.get("total_cost"),
            "entry_vix":         straddle.get("entry_vix"),
            "entry_vix_change":  straddle.get("entry_vix_change_pct"),
            "pnl":               None,
            "reason":            None,
        }

    def log_entry(self, symbol: str, straddle: dict) -> None:
        row = self._base_row(symbol, straddle)
        row["action"] = "OPEN"
        path = self._path(symbol)
        header = not os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

    def log_exit(self, symbol: str, straddle: dict, pnl: float, reason: str) -> None:
        row = self._base_row(symbol, straddle)
        row["action"] = f"CLOSE ({reason})"
        row["pnl"]    = round(pnl, 2)
        row["reason"] = reason
        pd.DataFrame([row]).to_csv(self._path(symbol), mode="a", header=False, index=False)

    def log_leg_close(self, symbol: str, straddle: dict, leg: str, reason: str) -> None:
        row = self._base_row(symbol, straddle)
        row["action"] = f"CLOSE_LEG_{leg.upper()} ({reason})"
        row["reason"] = reason
        pd.DataFrame([row]).to_csv(self._path(symbol), mode="a", header=False, index=False)


# ===================================================================
# Market hours helpers
# ===================================================================
def _is_market_open() -> bool:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _time_until_next_open() -> str:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now_et    = datetime.now(ZoneInfo("America/New_York"))
    next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et >= next_open:
        next_open += timedelta(days=1)
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)
    delta   = next_open - now_et
    hours   = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


# ===================================================================
# Long ATM Straddle Trader — Real Alpaca Paper Orders
# ===================================================================
class StraddleTrader:
    """Long ATM Straddle options trader triggered by VIX spikes.

    Submits real paper options orders to Alpaca.
    Paper trading accounts have automatic Level 3 options access.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        provider: str = "yahoo",
        confidence_threshold: float = 0.30,
        expiry_days: int = 30,
        max_risk_per_straddle: float = 2000.0,
        check_interval_min: int = 15,
        vix_spike_threshold: float = VIX_SPIKE_THRESHOLD,
        model_dir: str = DEFAULT_MODEL_DIR,
    ):
        # Paper trading only — hardcoded for safety
        self.trading_client = TradingClient(
            api_key=api_key, secret_key=api_secret, paper=True,
        )
        self.stock_data_client  = StockHistoricalDataClient(api_key, api_secret)
        self.option_data_client = OptionHistoricalDataClient(api_key, api_secret)

        self.symbols                = symbols
        self.confidence_threshold   = confidence_threshold
        self.expiry_days            = expiry_days
        self.max_risk_per_straddle  = max_risk_per_straddle
        self.check_interval         = check_interval_min * 60
        self.vix_spike_threshold    = vix_spike_threshold

        self.adapter  = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        # One straddle per symbol max: symbol → straddle dict
        self.active_straddles: Dict[str, dict] = {}

        # CSV trade logger
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        self.trade_logger = OptionsTradeLogger(output_dir)

        # IV estimator — refreshed each cycle
        self.iv_estimator: Optional[IVEstimator] = None

        # ML predictors (daily mode — VIX spike is a daily event)
        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            try:
                self.predictors[sym] = Predictor(sym, model_dir=model_dir, mode="daily")
                log.info("Loaded ML model for %s.", sym)
            except FileNotFoundError:
                log.warning("No ML model for %s — ML signal exit disabled.", sym)
                self.predictors[sym] = None

        self._running = True

    # ----------------------------------------------------------------
    # Account & price helpers
    # ----------------------------------------------------------------
    def get_account_summary(self) -> Dict[str, float]:
        try:
            acct = self.trading_client.get_account()
            return {
                "cash":         float(acct.cash),
                "equity":       float(acct.equity),
                "buying_power": float(acct.buying_power),
            }
        except Exception as exc:
            log.error("Failed to get account: %s", exc)
            return {"cash": 0.0, "equity": 0.0, "buying_power": 0.0}

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            req    = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_data_client.get_stock_latest_quote(req)
            if symbol in quotes:
                bid = float(quotes[symbol].bid_price)
                ask = float(quotes[symbol].ask_price)
                return (bid + ask) / 2
            return None
        except Exception as exc:
            log.error("Price fetch failed for %s: %s", symbol, exc)
            return None

    # ----------------------------------------------------------------
    # Options chain helpers
    # ----------------------------------------------------------------
    def _find_atm_contracts(
        self, symbol: str, current_price: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find real ATM call and put contracts ~expiry_days out.

        Returns (call_symbol, put_symbol) — e.g. ("SPY250321C00580000", "SPY250321P00580000").
        Returns (None, None) if no contracts found.
        """
        strike    = round(current_price)
        today     = date.today()
        expiry_min = today + timedelta(days=max(7, self.expiry_days - 9))
        expiry_max = today + timedelta(days=self.expiry_days + 15)

        call_symbol: Optional[str] = None
        put_symbol:  Optional[str] = None

        for contract_type in (ContractType.CALL, ContractType.PUT):
            try:
                result = self.trading_client.get_option_contracts(
                    GetOptionContractsRequest(
                        underlying_symbols=[symbol],
                        type=contract_type,
                        strike_price_gte=str(strike - 5),
                        strike_price_lte=str(strike + 5),
                        expiration_date_gte=expiry_min,
                        expiration_date_lte=expiry_max,
                        limit=20,
                    )
                )

                # alpaca-py returns GetOptionContractsResponse with .option_contracts
                contracts = getattr(result, "option_contracts", None)
                if contracts is None:
                    contracts = list(result) if result else []

                if not contracts:
                    log.warning(
                        "No %s contracts found for %s at strike ~%s (%s to %s)",
                        contract_type.value, symbol, strike, expiry_min, expiry_max,
                    )
                    continue

                # Pick contract with strike closest to current price,
                # then earliest expiry among ties
                best = min(
                    contracts,
                    key=lambda c: (
                        abs(float(c.strike_price) - current_price),
                        c.expiration_date,
                    ),
                )

                if contract_type == ContractType.CALL:
                    call_symbol = best.symbol
                    log.info(
                        "Found CALL %s | strike=%.0f | expiry=%s",
                        best.symbol, float(best.strike_price), best.expiration_date,
                    )
                else:
                    put_symbol = best.symbol
                    log.info(
                        "Found PUT  %s | strike=%.0f | expiry=%s",
                        best.symbol, float(best.strike_price), best.expiration_date,
                    )

            except Exception as exc:
                log.error("Contract search failed for %s %s: %s", symbol, contract_type.value, exc)

        return call_symbol, put_symbol

    def _get_quote_mid(self, contract_symbol: str) -> Optional[float]:
        """Get live mid-price (bid+ask)/2 for an option contract."""
        try:
            req    = OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol,
                                              feed=OptionsFeed.INDICATIVE)
            quotes = self.option_data_client.get_option_latest_quote(req)
            if contract_symbol in quotes:
                q   = quotes[contract_symbol]
                bid = float(q.bid_price) if q.bid_price else 0.0
                ask = float(q.ask_price) if q.ask_price else 0.0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                if ask > 0:
                    return ask
            return None
        except Exception as exc:
            log.warning("Quote fetch failed for %s: %s", contract_symbol, exc)
            return None

    def _bs_fallback_price(
        self, is_call: bool, current_price: float, strike: float,
        dte: int, vix: float,
    ) -> float:
        """Black-Scholes fallback when live quote is unavailable."""
        T     = max(0.001, dte / 365.0)
        sigma = max(0.05, vix / 100.0)
        if is_call:
            return GreeksEstimator.call_price(current_price, strike, T, sigma=sigma)
        return GreeksEstimator.put_price(current_price, strike, T, sigma=sigma)

    # ----------------------------------------------------------------
    # Real position data from Alpaca
    # ----------------------------------------------------------------
    def _get_positions_map(self) -> Dict[str, object]:
        """Return dict of {symbol: position} for all open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return {p.symbol: p for p in positions}
        except Exception as exc:
            log.error("Failed to get positions: %s", exc)
            return {}

    def _get_real_pnl(self, symbol: str, positions: Dict[str, object]) -> dict:
        """Compute P&L from real Alpaca position data.

        Falls back to Black-Scholes if a position isn't found yet
        (e.g., order just submitted, not filled yet).
        """
        if symbol not in self.active_straddles:
            return {}

        s         = self.active_straddles[symbol]
        vix       = self.iv_estimator.current_vix() if self.iv_estimator else 20.0
        call_sym  = s.get("call_contract")
        put_sym   = s.get("put_contract")
        expiry_dt = datetime.strptime(s["expiry"], "%Y-%m-%d")
        dte       = max(0, (expiry_dt - datetime.now()).days)

        def _leg_value(contract_sym: Optional[str], is_call: bool, leg_open: bool) -> float:
            if not leg_open or contract_sym is None:
                return 0.0
            pos = positions.get(contract_sym)
            if pos is not None:
                return abs(float(pos.market_value))
            # Fallback: live quote
            mid = self._get_quote_mid(contract_sym) if contract_sym else None
            if mid is not None:
                return mid * 100
            # Fallback: Black-Scholes
            current_price = self.get_current_price(symbol) or s["entry_price"]
            return self._bs_fallback_price(is_call, current_price, s["strike"], dte, vix) * 100

        call_value = _leg_value(call_sym, True,  s["call_open"])
        put_value  = _leg_value(put_sym,  False, s["put_open"])
        total_value = call_value + put_value

        call_entry  = s["call_cost"] * 100
        put_entry   = s["put_cost"]  * 100
        total_cost  = s["total_cost"]

        call_mult  = (call_value  / call_entry)  if call_entry  > 0 else 1.0
        put_mult   = (put_value   / put_entry)   if put_entry   > 0 else 1.0
        total_mult = (total_value / total_cost)  if total_cost  > 0 else 1.0

        return {
            "call_value":     round(call_value,  2),
            "put_value":      round(put_value,   2),
            "total_value":    round(total_value, 2),
            "call_pnl_mult":  round(call_mult,   4),
            "put_pnl_mult":   round(put_mult,    4),
            "total_pnl_mult": round(total_mult,  4),
            "pnl_dollars":    round(total_value - total_cost, 2),
            "dte":            dte,
        }

    # ----------------------------------------------------------------
    # Real order execution
    # ----------------------------------------------------------------
    def _submit_buy(self, contract_symbol: str) -> Optional[str]:
        """Submit a market buy order for 1 option contract. Returns order ID."""
        try:
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=contract_symbol,
                    qty=1,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("BUY order submitted: %s (id=%s)", contract_symbol, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("BUY order failed for %s: %s", contract_symbol, exc)
            return None

    def _submit_sell(self, contract_symbol: str) -> bool:
        """Submit a market sell order to close 1 option contract."""
        try:
            self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=contract_symbol,
                    qty=1,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("SELL order submitted: %s", contract_symbol)
            return True
        except Exception as exc:
            log.error("SELL order failed for %s: %s", contract_symbol, exc)
            return False

    # ----------------------------------------------------------------
    # Straddle lifecycle
    # ----------------------------------------------------------------
    def execute_straddle(
        self,
        symbol: str,
        current_price: float,
        vix_now: float,
        vix_change_pct: float,
    ) -> bool:
        """Find ATM contracts, check cost, submit real buy orders for both legs."""
        log.info("Looking for ATM contracts for %s @ %.2f...", symbol, current_price)
        call_symbol, put_symbol = self._find_atm_contracts(symbol, current_price)

        if not call_symbol or not put_symbol:
            log.warning("Could not find both contracts for %s — skipping.", symbol)
            return False

        # Get live quotes for cost estimation
        call_mid = self._get_quote_mid(call_symbol)
        put_mid  = self._get_quote_mid(put_symbol)

        # Fallback to BS if quotes unavailable
        strike = round(current_price)
        dte    = self.expiry_days
        if call_mid is None:
            call_mid = self._bs_fallback_price(True,  current_price, strike, dte, vix_now)
            log.info("Using BS fallback for call quote: %.2f", call_mid)
        if put_mid is None:
            put_mid = self._bs_fallback_price(False, current_price, strike, dte, vix_now)
            log.info("Using BS fallback for put quote: %.2f", put_mid)

        total_cost = (call_mid + put_mid) * 100  # per 100-share contract

        if total_cost > self.max_risk_per_straddle:
            log.info(
                "SKIP straddle %s — estimated cost $%.0f exceeds max_risk $%.0f",
                symbol, total_cost, self.max_risk_per_straddle,
            )
            return False

        # Submit real paper orders
        call_order_id = self._submit_buy(call_symbol)
        put_order_id  = self._submit_buy(put_symbol)

        if not call_order_id or not put_order_id:
            log.error("Failed to submit one or both legs for %s.", symbol)
            # Attempt to clean up any successful leg
            if call_order_id and not put_order_id:
                self._submit_sell(call_symbol)
            return False

        # Parse expiry from contract symbol (e.g., SPY250321C00580000 → 2025-03-21)
        expiry_str = _expiry_from_symbol(call_symbol)

        straddle = {
            "symbol":               symbol,
            "call_contract":        call_symbol,
            "put_contract":         put_symbol,
            "strike":               float(strike),
            "expiry":               expiry_str,
            "entry_price":          current_price,
            "call_cost":            round(call_mid, 4),
            "put_cost":             round(put_mid,  4),
            "total_cost":           round(total_cost, 2),
            "entry_date":           datetime.now(),
            "entry_vix":            vix_now,
            "entry_vix_change_pct": vix_change_pct,
            "call_open":            True,
            "put_open":             True,
            "call_order_id":        call_order_id,
            "put_order_id":         put_order_id,
        }

        self.active_straddles[symbol] = straddle
        self.trade_logger.log_entry(symbol, straddle)

        log.info(
            "OPEN STRADDLE %s | call=%s put=%s | total=$%.0f | VIX=%.1f (%+.1f%%)",
            symbol, call_symbol, put_symbol, total_cost, vix_now, vix_change_pct,
        )
        return True

    def close_straddle(self, symbol: str, reason: str, pnl: float = 0.0) -> None:
        """Close all open legs with real sell orders, log exit."""
        if symbol not in self.active_straddles:
            return

        s = self.active_straddles[symbol]

        if s["call_open"] and s.get("call_contract"):
            self._submit_sell(s["call_contract"])

        if s["put_open"] and s.get("put_contract"):
            self._submit_sell(s["put_contract"])

        self.trade_logger.log_exit(symbol, s, pnl, reason)
        del self.active_straddles[symbol]
        log.info("CLOSE STRADDLE %s | reason=%s | est P&L=$%.0f", symbol, reason, pnl)

    def close_leg(self, symbol: str, leg: str, reason: str = "") -> None:
        """Close one leg (call or put) with a real sell order."""
        if symbol not in self.active_straddles:
            return

        s   = self.active_straddles[symbol]
        key = f"{leg}_open"

        if not s.get(key, False):
            log.info("Leg %s for %s already closed.", leg, symbol)
            return

        contract_sym = s.get(f"{leg}_contract")
        if contract_sym:
            self._submit_sell(contract_sym)

        self.active_straddles[symbol][key] = False
        self.trade_logger.log_leg_close(symbol, s, leg, reason)

        remaining = "call" if leg == "put" else "put"
        log.info(
            "CLOSE LEG %s %s | reason=%s | %s leg still open",
            leg.upper(), symbol, reason, remaining,
        )

    # ----------------------------------------------------------------
    # Per-symbol decision engine
    # ----------------------------------------------------------------
    def process_symbol(self, symbol: str, positions: Dict[str, object]) -> str:
        """Evaluate entry/exit conditions and act. Returns console status string."""

        current_price = self.get_current_price(symbol)
        if current_price is None:
            return "SKIP  (price unavailable)"

        # ML prediction
        direction  = "UNKNOWN"
        confidence = 0.0
        predictor  = self.predictors.get(symbol)
        if predictor is not None:
            try:
                daily_bars = self.adapter.fetch_daily(symbol, DAILY_LOOKBACK)
                vix_df     = _fetch_vix_for_training(self.fred_key, lookback_days=DAILY_LOOKBACK)
                pred       = predictor.predict(daily_bars, vix_df)
                direction  = pred["direction"]
                confidence = pred["confidence"]
            except Exception as exc:
                log.warning("ML prediction failed for %s: %s", symbol, exc)

        vix_now        = self.iv_estimator.current_vix()    if self.iv_estimator else 20.0
        vix_change_pct = self.iv_estimator.vix_change_pct() if self.iv_estimator else 0.0

        # ============================================================
        # PATH A — Active straddle: manage exits
        # ============================================================
        if symbol in self.active_straddles:
            s = self.active_straddles[symbol]

            # Guard: both legs already individually closed
            if not s["call_open"] and not s["put_open"]:
                self.close_straddle(symbol, "both_legs_closed", pnl=0.0)
                return "CLOSED (both legs previously closed)"

            pnl         = self._get_real_pnl(symbol, positions)
            dte         = pnl.get("dte", 0)
            call_mult   = pnl.get("call_pnl_mult",  1.0)
            put_mult    = pnl.get("put_pnl_mult",   1.0)
            total_mult  = pnl.get("total_pnl_mult", 1.0)
            pnl_dollars = pnl.get("pnl_dollars",    0.0)

            # Exit: approaching expiry
            if dte <= 5:
                self.close_straddle(symbol, "approaching_expiry", pnl=pnl_dollars)
                return f"CLOSED (expiry in {dte}d) P&L=${pnl_dollars:+.0f}"

            # Exit 1: profit target — either leg >= 1.8x
            call_hit = s["call_open"] and call_mult >= PROFIT_TARGET_MULT
            put_hit  = s["put_open"]  and put_mult  >= PROFIT_TARGET_MULT
            if call_hit or put_hit:
                leg_hit  = "call" if call_hit else "put"
                mult_hit = call_mult if call_hit else put_mult
                self.close_straddle(symbol, "profit_target", pnl=pnl_dollars)
                return (
                    f"CLOSED profit_target "
                    f"({leg_hit} x{mult_hit:.2f}) P&L=${pnl_dollars:+.0f}"
                )

            # Exit 2: stop loss
            if total_mult <= STOP_LOSS_MULT:
                self.close_straddle(symbol, "stop_loss", pnl=pnl_dollars)
                return f"CLOSED stop_loss (total={total_mult:.2f}x) P&L=${pnl_dollars:+.0f}"

            # Exit 3: ML signal — close losing leg, let winner ride
            ml_acted = False
            if confidence >= self.confidence_threshold:
                if direction == "UP" and s["put_open"]:
                    self.close_leg(symbol, "put", "signal_exit_ml_up")
                    ml_acted = True
                elif direction == "DOWN" and s["call_open"]:
                    self.close_leg(symbol, "call", "signal_exit_ml_down")
                    ml_acted = True

            days_held   = (datetime.now() - s["entry_date"]).days
            legs_str    = ("C" if s["call_open"] else "-") + ("P" if s["put_open"] else "-")
            ml_note     = f"ML:{direction}({confidence:.2f})"
            action_note = " [LEG CLOSED]" if ml_acted else ""
            return (
                f"HOLD  [{legs_str}] {days_held}d | "
                f"call={call_mult:.2f}x put={put_mult:.2f}x total={total_mult:.2f}x "
                f"P&L=${pnl_dollars:+.0f} | VIX={vix_now:.1f} | {ml_note}{action_note}"
            )

        # ============================================================
        # PATH B — No position: check VIX spike for entry
        # ============================================================
        if vix_change_pct >= self.vix_spike_threshold:
            opened = self.execute_straddle(symbol, current_price, vix_now, vix_change_pct)
            if opened:
                s = self.active_straddles[symbol]
                return (
                    f"OPEN  straddle | strike={s['strike']:.0f} | "
                    f"expiry={s['expiry']} | total=${s['total_cost']:.0f} | "
                    f"VIX={vix_now:.1f} spike={vix_change_pct:+.1f}%"
                )
            return f"SKIP  (no contracts found or cost > ${self.max_risk_per_straddle:.0f})"

        return (
            f"WAIT  VIX spike needed | "
            f"VIX={vix_now:.1f} chg={vix_change_pct:+.1f}% "
            f"(need >={self.vix_spike_threshold:.0f}%) | "
            f"ML:{direction}({confidence:.2f})"
        )

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    def run_loop(self) -> None:
        log.info("Starting Long ATM Straddle trader (REAL paper orders)...")
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("VIX spike threshold: %.0f%%", self.vix_spike_threshold)
        log.info(
            "Exit: profit=%.1fx per leg | stop=%.1fx total | ML confidence>=%.2f",
            PROFIT_TARGET_MULT, STOP_LOSS_MULT, self.confidence_threshold,
        )
        print()

        def handle_signal(sig, frame):
            print("\n\n  Shutting down straddle trader...\n")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        while self._running:
            cycle += 1

            if not _is_market_open():
                next_open = _time_until_next_open()
                print(
                    f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Market closed. Next open in {next_open}. "
                    f"Checking again in {self.check_interval // 60} min..."
                )
                self._sleep(self.check_interval)
                continue

            try:
                # Refresh VIX each cycle
                vix_fetcher       = FREDVixFetcher(api_key=self.fred_key)
                vix_df            = vix_fetcher.fetch(lookback_days=365)
                self.iv_estimator = IVEstimator(vix_df)

                account = self.get_account_summary()
                vix_now = self.iv_estimator.current_vix()
                vix_chg = self.iv_estimator.vix_change_pct()
                iv_r    = self.iv_estimator.iv_rank()

                # Fetch all positions once per cycle (real Alpaca data)
                positions = self._get_positions_map()

                print(
                    f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"=== Straddle Cycle #{cycle} (REAL PAPER) ==="
                )
                print(
                    f"  Account: ${account['equity']:,.2f} | "
                    f"{len(self.active_straddles)} straddle(s) | "
                    f"VIX: {vix_now:.1f} ({vix_chg:+.1f}%) | IV Rank: {iv_r:.0f} | "
                    f"Spike trigger: >={self.vix_spike_threshold:.0f}%"
                )

                for symbol in self.symbols:
                    try:
                        result = self.process_symbol(symbol, positions)
                        print(f"    {symbol:>4}  {result}")
                    except Exception as exc:
                        print(f"    {symbol:>4}  ERROR: {exc}")
                        log.exception("Symbol processing failed for %s", symbol)

                print(f"  Next check in {self.check_interval // 60} min.")

            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"  Cycle error: {exc}")
                log.exception("Trading cycle failed")

            self._sleep(self.check_interval)

    def _sleep(self, seconds: int) -> None:
        start = time.time()
        while time.time() - start < seconds and self._running:
            time.sleep(min(1, seconds - (time.time() - start)))


# ===================================================================
# Helpers
# ===================================================================
def _expiry_from_symbol(option_symbol: str) -> str:
    """Parse expiry date from OCC option symbol.

    OCC format: ROOT + YYMMDD + C/P + 8-digit strike
    e.g. SPY250321C00580000 → 2025-03-21
    """
    try:
        # Strip root symbol (variable length) — find first digit
        i = 0
        while i < len(option_symbol) and not option_symbol[i].isdigit():
            i += 1
        date_str = option_symbol[i:i + 6]  # YYMMDD
        return f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    except Exception:
        # Fallback: estimate from today + expiry_days
        target = datetime.now() + timedelta(days=30)
        return target.strftime("%Y-%m-%d")


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long ATM Straddle options trader — VIX spike entry, real Alpaca paper orders.",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (default: SPY,QQQ,IWM)",
    )
    parser.add_argument(
        "--provider", default="yahoo", choices=["yahoo", "alpaca", "hybrid"],
        help="Data provider for price history (default: yahoo)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.30,
        help="ML confidence threshold for signal-exit leg close (default: 0.30)",
    )
    parser.add_argument(
        "--expiry-days", type=int, default=30,
        help="Target days to expiration (default: 30)",
    )
    parser.add_argument(
        "--max-risk", type=float, default=2000.0,
        help="Max estimated straddle cost in dollars (default: 2000). "
             "Set 5000+ for SPY at high VIX.",
    )
    parser.add_argument(
        "--check-interval", type=int, default=15,
        help="Check interval in minutes (default: 15)",
    )
    parser.add_argument(
        "--vix-spike-threshold", type=float, default=VIX_SPIKE_THRESHOLD,
        help="Min VIX daily %% change to trigger entry (default: 15.0). "
             "Use 1.0 to test immediately.",
    )

    args = parser.parse_args()

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        print("\n  ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        sys.exit(1)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols else ["SPY", "QQQ", "IWM"]
    )

    trader = StraddleTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        provider=args.provider,
        confidence_threshold=args.confidence,
        expiry_days=args.expiry_days,
        max_risk_per_straddle=args.max_risk,
        check_interval_min=args.check_interval,
        vix_spike_threshold=args.vix_spike_threshold,
    )

    try:
        account = trader.get_account_summary()
        print(f"\n  Connected to Alpaca Paper Trading  (Long ATM Straddle — REAL ORDERS)")
        print(f"  Account equity:    ${account['equity']:,.2f}")
        print(f"  Strategy:          Buy real ATM call + put contracts")
        print(f"  Entry trigger:     VIX daily change >= {args.vix_spike_threshold:.0f}%")
        print(f"  Profit target:     +{(PROFIT_TARGET_MULT - 1) * 100:.0f}% on either leg")
        print(f"  Stop loss:         -{(1 - STOP_LOSS_MULT) * 100:.0f}% on total position")
        print(f"  ML signal exit:    confidence >= {args.confidence:.2f}")
        print(f"  Max risk/straddle: ${args.max_risk:,.0f}")
        print()
        print("  NOTE: If you see 'forbidden' errors, complete your Alpaca account")
        print("        application to enable options on your paper account.")
        print()
        print("  Press Ctrl+C to stop.\n")
    except Exception as exc:
        print(f"  Connection error: {exc}")
        sys.exit(1)

    trader.run_loop()


if __name__ == "__main__":
    main()
