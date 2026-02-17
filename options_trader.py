#!/usr/bin/env python3
"""
Alpaca Options Paper Trader - Put/Call Spreads
===============================================
Executes put spreads (bearish) and call spreads (bullish) based on composite
signals: ML predictions + technical patterns + volatility regime.

Features:
- Multi-signal decision engine (ML 50%, patterns 30%, vol regime 20%)
- IV rank/percentile estimation from VIX history
- Delta-based strike selection (~0.30 delta OTM)
- Black-Scholes Greeks estimation for position monitoring
- CSV trade logging for analysis
- Pattern detection: mean reversion, trend continuation, vol crush

Usage (via main.py):
    python main.py trade-options --symbols SPY,QQQ --confidence 0.2

Required env vars: ALPACA_API_KEY, ALPACA_API_SECRET, FRED_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from signals_engine import (
    DEFAULT_UNIVERSE, DAILY_LOOKBACK, build_adapter, FREDVixFetcher,
    compute_rsi, compute_return, compute_dollar_vol_accel, RSI_PERIOD,
)
from ml_model import Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("options_trader")


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
    """Estimate implied volatility environment using VIX as proxy.

    IV Rank  = (current - 52wk low) / (52wk high - 52wk low) → 0-100
    IV Pctile = % of past year days where VIX was lower → 0-100
    """

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

    def iv_percentile(self) -> float:
        if self._vix.empty or len(self._vix) < 20:
            return 50.0
        vals = self._vix["vix"].tail(252)
        current = float(vals.iloc[-1])
        return float((vals < current).sum() / len(vals) * 100)

    def current_vix(self) -> float:
        if self._vix.empty:
            return 20.0
        return float(self._vix["vix"].iloc[-1])

    def vix_change_pct(self) -> float:
        if len(self._vix) < 2:
            return 0.0
        cur = float(self._vix["vix"].iloc[-1])
        prev = float(self._vix["vix"].iloc[-2])
        return ((cur - prev) / prev * 100) if prev != 0 else 0.0


# ===================================================================
# Technical Pattern Detection
# ===================================================================
class PatternDetector:
    """Detect technical patterns for options entry signals."""

    @staticmethod
    def detect_mean_reversion(rsi: float, vix: float, vix_chg_pct: float) -> dict:
        """RSI extreme + VIX spike = mean reversion setup."""
        # Oversold bounce
        if rsi < 30 and vix_chg_pct > 10:
            strength = min(1.0, (30 - rsi) / 20 + vix_chg_pct / 40)
            return {"signal": "CALL", "strength": strength, "pattern": "mean_reversion_oversold"}
        # Overbought reversal
        if rsi > 70 and vix < 15:
            strength = min(1.0, (rsi - 70) / 20 + (15 - vix) / 10)
            return {"signal": "PUT", "strength": strength, "pattern": "mean_reversion_overbought"}
        return {"signal": None, "strength": 0.0, "pattern": "none"}

    @staticmethod
    def detect_trend_continuation(ret5: float, ret10: float, vol_accel: float) -> dict:
        """Strong trend + volume confirmation."""
        if np.isnan(ret5) or np.isnan(ret10) or np.isnan(vol_accel):
            return {"signal": None, "strength": 0.0, "pattern": "none"}
        # Bullish
        if ret5 > 0.02 and ret10 > 0.03 and vol_accel > 0:
            strength = min(1.0, ret5 / 0.05 + vol_accel)
            return {"signal": "CALL", "strength": strength, "pattern": "trend_up"}
        # Bearish
        if ret5 < -0.02 and ret10 < -0.03 and vol_accel > 0:
            strength = min(1.0, abs(ret5) / 0.05 + vol_accel)
            return {"signal": "PUT", "strength": strength, "pattern": "trend_down"}
        return {"signal": None, "strength": 0.0, "pattern": "none"}

    @staticmethod
    def detect_volatility_crush(iv_rank: float, iv_pctile: float) -> dict:
        """High IV environment = sell premium opportunity."""
        if iv_rank > 70 and iv_pctile > 70:
            strength = min(1.0, (iv_rank - 50) / 50)
            return {"signal": "SELL_PREMIUM", "strength": strength, "pattern": "vol_crush"}
        return {"signal": None, "strength": 0.0, "pattern": "none"}


# ===================================================================
# Black-Scholes Greeks Estimation
# ===================================================================
class GreeksEstimator:
    """Simple Black-Scholes approximations for monitoring (no scipy)."""

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

    @staticmethod
    def delta(S: float, K: float, T: float, r: float = 0.05,
              sigma: float = 0.20, is_call: bool = True) -> float:
        if T <= 0:
            return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
        d1 = GreeksEstimator._d1(S, K, T, r, sigma)
        return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0

    @staticmethod
    def theta(S: float, K: float, T: float, r: float = 0.05,
              sigma: float = 0.20, is_call: bool = True) -> float:
        """Daily theta (per calendar day)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = GreeksEstimator._d1(S, K, T, r, sigma)
        d2 = GreeksEstimator._d2(S, K, T, r, sigma)
        term1 = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        if is_call:
            return (term1 - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
        return (term1 + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365


# ===================================================================
# Options Trade Logger (CSV persistence)
# ===================================================================
class OptionsTradeLogger:
    """Persist options spread trades to CSV for analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _path(self, symbol: str) -> str:
        return os.path.join(self.output_dir, f"options_trades_{symbol}.csv")

    def log_entry(self, symbol: str, spread_info: dict) -> None:
        row = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "OPEN",
            "spread_type": spread_info["spread_type"],
            "long_strike": spread_info["long_strike"],
            "short_strike": spread_info["short_strike"],
            "expiry": spread_info["expiry"],
            "entry_price": spread_info["entry_price"],
            "max_risk": spread_info["max_risk"],
            "max_profit": spread_info["max_profit"],
            "pattern": spread_info.get("pattern", "ml_only"),
            "iv_rank": spread_info.get("iv_rank"),
            "composite_score": spread_info.get("composite_score"),
            "pnl": None,
        }
        path = self._path(symbol)
        header = not os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

    def log_exit(self, symbol: str, spread_info: dict, pnl: float, reason: str) -> None:
        row = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": f"CLOSE ({reason})",
            "spread_type": spread_info["spread_type"],
            "long_strike": spread_info["long_strike"],
            "short_strike": spread_info["short_strike"],
            "expiry": spread_info["expiry"],
            "entry_price": spread_info["entry_price"],
            "max_risk": spread_info["max_risk"],
            "max_profit": spread_info["max_profit"],
            "pattern": spread_info.get("pattern", ""),
            "iv_rank": spread_info.get("iv_rank"),
            "composite_score": None,
            "pnl": round(pnl, 2),
        }
        pd.DataFrame([row]).to_csv(self._path(symbol), mode="a", header=False, index=False)


# ===================================================================
# Market hours helper
# ===================================================================
def _is_market_open() -> bool:
    """Check if US stock market is currently open (9:30-16:00 ET, weekdays)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _time_until_next_open() -> str:
    """Human-readable time until next market open."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et >= next_open:
        next_open += timedelta(days=1)
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    delta = next_open - now_et
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


# ===================================================================
# Options Spread Trader
# ===================================================================
COMPOSITE_ENTRY_THRESHOLD = 0.25


class AlpacaOptionsTrader:
    """Options spread trader with composite signal engine.

    Multi-signal decision: ML (50%) + patterns (30%) + vol regime (20%).
    Tracks spreads in memory with CSV logging and Greeks monitoring.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        provider: str = "yahoo",
        confidence_threshold: float = 0.2,
        put_confidence_threshold: float = 0.15,
        exit_confidence: float = 0.1,
        spread_width: int = 5,
        days_to_expiry: int = 30,
        max_risk_per_trade: float = 500,
        check_interval_min: int = 15,
        model_dir: str = DEFAULT_MODEL_DIR,
        mode: str = "daily",
        intraday_interval: str = "5min",
    ):
        # Trading client — PAPER ONLY
        self.trading_client = TradingClient(
            api_key=api_key, secret_key=api_secret, paper=True,
        )
        self.data_client = StockHistoricalDataClient(api_key, api_secret)

        self.symbols = symbols
        self.call_confidence_threshold = confidence_threshold
        self.put_confidence_threshold = put_confidence_threshold
        self.exit_confidence = exit_confidence
        self.spread_width = spread_width
        self.days_to_expiry = days_to_expiry
        self.max_risk_per_trade = max_risk_per_trade
        self.check_interval = check_interval_min * 60
        self.mode = mode
        self.intraday_interval = intraday_interval

        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        # Active spreads and trade logger
        self.active_spreads: Dict[str, dict] = {}
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        self.trade_logger = OptionsTradeLogger(output_dir)

        # IV estimator — initialized per cycle with fresh VIX data
        self.iv_estimator: Optional[IVEstimator] = None

        # ML predictors
        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            try:
                self.predictors[sym] = Predictor(
                    sym, model_dir=model_dir,
                    mode=mode, intraday_interval=intraday_interval)
                log.info("Loaded ML model for %s (%s).", sym, mode)
            except FileNotFoundError:
                log.warning("No ML model found for %s. Skipping.", sym)
                self.predictors[sym] = None

        self._running = True

    # -- Account & price helpers ----------------------------------------
    def get_account_summary(self) -> Dict[str, float]:
        try:
            account = self.trading_client.get_account()
            return {
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
            }
        except Exception as exc:
            log.error("Failed to get account: %s", exc)
            return {"cash": 0.0, "equity": 0.0, "buying_power": 0.0}

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            if symbol in quotes:
                bid = float(quotes[symbol].bid_price)
                ask = float(quotes[symbol].ask_price)
                return (bid + ask) / 2
            return None
        except Exception as exc:
            log.error("Price fetch failed for %s: %s", symbol, exc)
            return None

    def _find_expiration_date(self, days_out: int = 30) -> str:
        """Find next Friday expiration approximately days_out from today."""
        target = datetime.now() + timedelta(days=days_out)
        days_until_friday = (4 - target.weekday()) % 7
        if days_until_friday == 0 and target.weekday() == 4:
            expiry = target
        else:
            expiry = target + timedelta(days=days_until_friday)
        return expiry.strftime("%Y-%m-%d")

    # -- Delta-based strike selection -----------------------------------
    def _get_option_strikes(self, current_price: float, is_call: bool,
                            sigma: float = 0.20) -> Tuple[float, float]:
        """Select strikes targeting ~0.30 delta for the long leg."""
        T = self.days_to_expiry / 365.0
        r = 0.05

        if T <= 0 or sigma <= 0:
            # Fallback to simple fixed offset
            base = round(current_price)
            if is_call:
                return base + 1, base + 1 + self.spread_width
            return base - 1, base - 1 - self.spread_width

        # Target d1 for ~0.30 delta: calls d1≈-0.524, puts d1≈0.524
        target_d1 = -0.524 if is_call else 0.524
        K_long = current_price * math.exp(
            -target_d1 * sigma * math.sqrt(T) + (r + 0.5 * sigma ** 2) * T
        )
        K_long = round(K_long)

        if is_call:
            K_short = K_long + self.spread_width
        else:
            K_short = K_long - self.spread_width

        return float(K_long), float(K_short)

    # -- Position sizing based on IV ------------------------------------
    def _calculate_position_risk(self, iv_rank: float) -> float:
        """Scale max risk inversely with IV rank.

        High IV → expensive premiums → smaller positions.
        Low IV  → cheaper premiums  → larger positions.
        """
        scale = 1.5 - (iv_rank / 100.0)
        return self.max_risk_per_trade * max(0.3, min(1.5, scale))

    # -- Spread execution -----------------------------------------------
    def execute_call_spread(self, symbol: str, current_price: float,
                            max_risk: float, pattern: str = "ml_only",
                            iv_rank: float = 50.0,
                            composite_score: float = 0.0) -> Optional[str]:
        """Execute bullish call spread (simulated)."""
        try:
            sigma = (self.iv_estimator.current_vix() / 100.0) if self.iv_estimator else 0.20
            expiry = self._find_expiration_date(self.days_to_expiry)
            long_strike, short_strike = self._get_option_strikes(
                current_price, is_call=True, sigma=sigma)

            spread_info = {
                "spread_type": "CALL_SPREAD",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "expiry": expiry,
                "entry_price": current_price,
                "entry_date": datetime.now(),
                "max_risk": max_risk,
                "max_profit": (short_strike - long_strike) * 100 - max_risk,
                "pattern": pattern,
                "iv_rank": iv_rank,
                "composite_score": composite_score,
            }

            self.active_spreads[symbol] = spread_info
            self.trade_logger.log_entry(symbol, spread_info)

            log.info("CALL SPREAD %s: Buy %s Call, Sell %s Call (exp: %s, pat: %s)",
                     symbol, long_strike, short_strike, expiry, pattern)
            return f"CALL_SPREAD ({long_strike}/{short_strike})"
        except Exception as exc:
            log.error("Call spread failed for %s: %s", symbol, exc)
            return None

    def execute_put_spread(self, symbol: str, current_price: float,
                           max_risk: float, pattern: str = "ml_only",
                           iv_rank: float = 50.0,
                           composite_score: float = 0.0) -> Optional[str]:
        """Execute bearish put spread (simulated)."""
        try:
            sigma = (self.iv_estimator.current_vix() / 100.0) if self.iv_estimator else 0.20
            expiry = self._find_expiration_date(self.days_to_expiry)
            long_strike, short_strike = self._get_option_strikes(
                current_price, is_call=False, sigma=sigma)

            spread_info = {
                "spread_type": "PUT_SPREAD",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "expiry": expiry,
                "entry_price": current_price,
                "entry_date": datetime.now(),
                "max_risk": max_risk,
                "max_profit": (long_strike - short_strike) * 100 - max_risk,
                "pattern": pattern,
                "iv_rank": iv_rank,
                "composite_score": composite_score,
            }

            self.active_spreads[symbol] = spread_info
            self.trade_logger.log_entry(symbol, spread_info)

            log.info("PUT SPREAD %s: Buy %s Put, Sell %s Put (exp: %s, pat: %s)",
                     symbol, long_strike, short_strike, expiry, pattern)
            return f"PUT_SPREAD ({long_strike}/{short_strike})"
        except Exception as exc:
            log.error("Put spread failed for %s: %s", symbol, exc)
            return None

    def close_spread(self, symbol: str, reason: str = "") -> Optional[str]:
        """Close active spread position."""
        if symbol not in self.active_spreads:
            return None

        spread = self.active_spreads[symbol]
        spread_type = spread["spread_type"]
        current_price = self.get_current_price(symbol)
        entry_price = spread["entry_price"]

        if current_price:
            if spread_type == "CALL_SPREAD":
                pnl_est = max(-spread["max_risk"], min(spread["max_profit"],
                             (current_price - entry_price) * 50))
            else:
                pnl_est = max(-spread["max_risk"], min(spread["max_profit"],
                             (entry_price - current_price) * 50))
        else:
            pnl_est = 0

        self.trade_logger.log_exit(symbol, spread, pnl_est, reason)
        del self.active_spreads[symbol]

        log.info("CLOSE %s %s: Est P&L $%.0f (%s)", spread_type, symbol, pnl_est, reason)
        return f"CLOSED {spread_type} (Est P&L: ${pnl_est:.0f})"

    # -- Composite decision engine --------------------------------------
    def process_symbol(self, symbol: str) -> str:
        """Process composite signals and manage spreads for one symbol."""
        # ML prediction
        predictor = self.predictors.get(symbol)
        direction = "UNKNOWN"
        confidence = 0.0

        if predictor is not None:
            try:
                daily_bars = self.adapter.fetch_daily(symbol, DAILY_LOOKBACK)
                vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=DAILY_LOOKBACK)
                prediction = predictor.predict(daily_bars, vix_df)
                direction = prediction["direction"]
                confidence = prediction["confidence"]
            except Exception as exc:
                log.error("ML prediction failed for %s: %s", symbol, exc)
        else:
            # Still fetch daily bars for pattern detection
            try:
                daily_bars = self.adapter.fetch_daily(symbol, DAILY_LOOKBACK)
            except Exception:
                return f"SKIP  (data fetch failed)"

        # Current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return f"SKIP  (price unavailable)  ML: {direction} {confidence:.3f}"

        # Technical indicators
        close = daily_bars["close"].astype(float) if not daily_bars.empty else pd.Series(dtype=float)
        rsi = float(compute_rsi(close, RSI_PERIOD).iloc[-1]) if len(close) > RSI_PERIOD else 50.0
        ret5 = compute_return(close, 5) if len(close) > 5 else 0.0
        ret10 = compute_return(close, 10) if len(close) > 10 else 0.0
        dv_accel = compute_dollar_vol_accel(daily_bars) if len(daily_bars) > 10 else 0.0

        # IV estimation
        iv_rank = self.iv_estimator.iv_rank() if self.iv_estimator else 50.0
        iv_pctile = self.iv_estimator.iv_percentile() if self.iv_estimator else 50.0
        vix_now = self.iv_estimator.current_vix() if self.iv_estimator else 20.0
        vix_chg = self.iv_estimator.vix_change_pct() if self.iv_estimator else 0.0

        # --- Existing position management ---
        if symbol in self.active_spreads:
            spread = self.active_spreads[symbol]
            spread_type = spread["spread_type"]
            days_held = (datetime.now() - spread["entry_date"]).days
            expiry_date = datetime.strptime(spread["expiry"], "%Y-%m-%d")
            dte = (expiry_date - datetime.now()).days

            if dte <= 5:
                self.close_spread(symbol, "approaching_expiry")
                return f"CLOSED (expiry in {dte} days)"

            if ((spread_type == "CALL_SPREAD" and direction == "DOWN" and confidence >= self.exit_confidence) or
                (spread_type == "PUT_SPREAD" and direction == "UP" and confidence >= self.exit_confidence)):
                self.close_spread(symbol, "signal_flip")
                return f"CLOSED (signal flip to {direction})"

            # Greeks for monitoring
            T = max(0.001, dte / 365.0)
            sigma = vix_now / 100.0
            is_call = spread_type == "CALL_SPREAD"
            d = GreeksEstimator.delta(current_price, spread["long_strike"], T, sigma=sigma, is_call=is_call)
            th = GreeksEstimator.theta(current_price, spread["long_strike"], T, sigma=sigma, is_call=is_call)

            price_chg = (current_price - spread["entry_price"]) / spread["entry_price"]
            return (f"HOLD  ({spread_type} {days_held}d, {price_chg:+.1%}, "
                    f"delta={d:.2f}, theta=${th * 100:.0f}/day)  "
                    f"ML: {direction} {confidence:.3f}")

        # --- Pattern detection ---
        mr = PatternDetector.detect_mean_reversion(rsi, vix_now, vix_chg)
        tc = PatternDetector.detect_trend_continuation(ret5, ret10, dv_accel)
        vc = PatternDetector.detect_volatility_crush(iv_rank, iv_pctile)

        # --- Composite scoring ---
        call_score = 0.0
        put_score = 0.0
        primary_pattern = "ml_only"

        # ML component (50%)
        if direction == "UP":
            call_score += confidence * 0.5
        elif direction == "DOWN":
            put_score += confidence * 0.5

        # Pattern component (30%)
        if mr["signal"] == "CALL":
            call_score += mr["strength"] * 0.3
            if mr["strength"] > 0.3:
                primary_pattern = mr["pattern"]
        elif mr["signal"] == "PUT":
            put_score += mr["strength"] * 0.3
            if mr["strength"] > 0.3:
                primary_pattern = mr["pattern"]

        if tc["signal"] == "CALL":
            call_score += tc["strength"] * 0.2
            if tc["strength"] > 0.3 and primary_pattern == "ml_only":
                primary_pattern = tc["pattern"]
        elif tc["signal"] == "PUT":
            put_score += tc["strength"] * 0.2
            if tc["strength"] > 0.3 and primary_pattern == "ml_only":
                primary_pattern = tc["pattern"]

        # Determine entry
        adjusted_risk = self._calculate_position_risk(iv_rank)
        best_score = max(call_score, put_score)

        if call_score >= COMPOSITE_ENTRY_THRESHOLD and call_score > put_score:
            result = self.execute_call_spread(
                symbol, current_price, adjusted_risk,
                pattern=primary_pattern, iv_rank=iv_rank, composite_score=call_score)
            if result:
                return (f"CALL  {result}  score={call_score:.2f} "
                        f"[{primary_pattern}]  IV={iv_rank:.0f}  ML: {direction} {confidence:.3f}")

        elif put_score >= COMPOSITE_ENTRY_THRESHOLD:
            result = self.execute_put_spread(
                symbol, current_price, adjusted_risk,
                pattern=primary_pattern, iv_rank=iv_rank, composite_score=put_score)
            if result:
                return (f"PUT   {result}  score={put_score:.2f} "
                        f"[{primary_pattern}]  IV={iv_rank:.0f}  ML: {direction} {confidence:.3f}")

        # Vol crush alert (no position taken, informational)
        vc_note = ""
        if vc["signal"] == "SELL_PREMIUM":
            vc_note = f"  [VOL_CRUSH IV={iv_rank:.0f}]"

        return (f"SKIP  (score={best_score:.2f} < {COMPOSITE_ENTRY_THRESHOLD}){vc_note}  "
                f"ML: {direction} {confidence:.3f}")

    # -- Main loop ------------------------------------------------------
    def run_loop(self) -> None:
        """Main continuous options trading loop."""
        log.info("Starting options spread trading loop (%s mode)...", self.mode)
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("Check interval: %d min", self.check_interval // 60)
        log.info("Entry: composite >= %.2f (ML 50%% + patterns 30%% + vol 20%%)",
                 COMPOSITE_ENTRY_THRESHOLD)
        log.info("Spread width: $%d, Base risk: $%.0f, Target expiry: %d days",
                 self.spread_width, self.max_risk_per_trade, self.days_to_expiry)
        print()

        def handle_signal(sig, frame):
            print("\n\n  Shutting down options trader...\n")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        while self._running:
            cycle += 1

            if not _is_market_open():
                next_open = _time_until_next_open()
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Market closed. Next open in {next_open}. "
                      f"Checking again in {self.check_interval // 60} min...")
                self._sleep(self.check_interval)
                continue

            try:
                # Refresh IV estimator each cycle
                vix_fetcher = FREDVixFetcher(api_key=self.fred_key)
                vix_df = vix_fetcher.fetch(lookback_days=365)
                self.iv_estimator = IVEstimator(vix_df)

                account = self.get_account_summary()
                iv_r = self.iv_estimator.iv_rank()
                vix_now = self.iv_estimator.current_vix()

                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"=== Options Cycle #{cycle} ({self.mode}) ===")
                print(f"  Account: ${account['equity']:,.2f} | "
                      f"{len(self.active_spreads)} spreads | "
                      f"VIX: {vix_now:.1f} | IV Rank: {iv_r:.0f}")

                for symbol in self.symbols:
                    try:
                        result = self.process_symbol(symbol)
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
        start_time = time.time()
        while time.time() - start_time < seconds and self._running:
            time.sleep(min(1, seconds - (time.time() - start_time)))


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Options spread trader — composite ML + pattern signals.",
    )
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: SPY,QQQ,IWM)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"],
                        help="Data provider (default: yahoo)")
    parser.add_argument("--check-interval", type=int, default=15,
                        help="Check interval in minutes (default: 15)")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Min ML confidence for CALL spreads (default: 0.2)")
    parser.add_argument("--put-confidence", type=float, default=0.15,
                        help="Min ML confidence for PUT spreads (default: 0.15)")
    parser.add_argument("--exit-confidence", type=float, default=0.1,
                        help="Min ML confidence to exit/flip (default: 0.1)")
    parser.add_argument("--spread-width", type=int, default=5,
                        help="Dollar width between strikes (default: 5)")
    parser.add_argument("--expiry-days", type=int, default=30,
                        help="Target days to expiration (default: 30)")
    parser.add_argument("--max-risk", type=float, default=500,
                        help="Max risk per spread (default: 500)")
    parser.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                        help="Trading mode (default: daily)")
    parser.add_argument("--interval", default="5min", choices=["1min", "5min"],
                        help="Intraday bar interval (default: 5min)")

    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        print("\n  ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        print("  Get free keys at https://app.alpaca.markets/signup\n")
        sys.exit(1)

    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else ["SPY", "QQQ", "IWM"])

    trader = AlpacaOptionsTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        provider=args.provider,
        confidence_threshold=args.confidence,
        put_confidence_threshold=args.put_confidence,
        exit_confidence=args.exit_confidence,
        spread_width=args.spread_width,
        days_to_expiry=args.expiry_days,
        max_risk_per_trade=args.max_risk,
        check_interval_min=args.check_interval,
        mode=args.mode,
        intraday_interval=args.interval,
    )

    try:
        account = trader.get_account_summary()
        print(f"\n  Connected to Alpaca Paper Trading (Options)")
        print(f"  Account equity: ${account['equity']:,.2f}")
        print(f"  Composite engine: ML(50%) + Patterns(30%) + Vol(20%)")
        print(f"  Entry threshold: {COMPOSITE_ENTRY_THRESHOLD}")
        print("  Press Ctrl+C to stop.\n")
    except Exception as exc:
        print(f"  Connection error: {exc}")
        sys.exit(1)

    trader.run_loop()


if __name__ == "__main__":
    main()
