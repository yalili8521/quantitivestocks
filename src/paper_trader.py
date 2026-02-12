#!/usr/bin/env python3
"""
Alpaca Paper Trader — Continuous Loop
=======================================
Runs the LSTM ML model in a loop and executes trades on Alpaca paper trading.
Supports LONG and SHORT positions, trailing stop, take-profit, and intraday mode.

Usage (via main.py):
    python main.py trade
    python main.py trade --interval 5 --confidence 0.2 --trailing-stop 0.05
    python main.py trade --symbols SPY,QQQ --mode intraday --interval 1

Required env vars:
    ALPACA_API_KEY, ALPACA_API_SECRET, FRED_API_KEY

PAPER TRADING ONLY — paper=True is hardcoded for safety.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.signals_engine import (
    DEFAULT_UNIVERSE, DAILY_LOOKBACK, build_adapter, FREDVixFetcher,
)
from src.ml_model import Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paper_trader")


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

    # Weekend check
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

    # Skip weekends
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    delta = next_open - now_et
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


# ===================================================================
# Paper Trader
# ===================================================================
class AlpacaPaperTrader:
    """Continuous paper trading loop driven by ML predictions.

    Supports LONG and SHORT positions with trailing stop-loss,
    take-profit, and immediate position flipping.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        provider: str = "yahoo",
        confidence_threshold: float = 0.2,
        exit_confidence: float = 0.1,
        trailing_stop_pct: float = 0.05,
        take_profit_pct: float = 0.08,
        position_pct: float = 0.90,
        check_interval_min: int = 5,
        model_dir: str = DEFAULT_MODEL_DIR,
        mode: str = "daily",
        intraday_interval: str = "5min",
    ):
        # Trading client — PAPER ONLY
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,  # HARDCODED for safety
        )
        self.symbols = symbols
        self.confidence_threshold = confidence_threshold
        self.exit_confidence = exit_confidence
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.position_pct = position_pct
        self.check_interval = check_interval_min * 60  # seconds
        self.mode = mode
        self.intraday_interval = intraday_interval

        # Data adapter for fetching prices
        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        # Peak price tracking for trailing stop {symbol: peak_price}
        self._peak_prices: Dict[str, float] = {}

        # ML predictors — one per symbol
        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            try:
                self.predictors[sym] = Predictor(
                    sym, model_dir=model_dir,
                    mode=mode, intraday_interval=intraday_interval)
                log.info("Loaded ML model for %s (%s).", sym, mode)
            except FileNotFoundError:
                log.warning("No trained model for %s (%s) — will skip ML signals.", sym, mode)
                self.predictors[sym] = None

        self._running = True

    # -- Account info --------------------------------------------------
    def get_account_summary(self) -> dict:
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions as {symbol: {qty, side, entry, current, pnl, pnl_pct}}."""
        positions = self.trading_client.get_all_positions()
        result = {}
        for pos in positions:
            qty = float(pos.qty)
            side = "SHORT" if qty < 0 else "LONG"
            result[pos.symbol] = {
                "qty": abs(qty),
                "side": side,
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pnl": float(pos.unrealized_pl),
                "unrealized_pnl_pct": float(pos.unrealized_plpc),
            }
        return result

    # -- Order execution -----------------------------------------------
    def buy(self, symbol: str, qty: int) -> Optional[str]:
        """Submit a market buy order (open LONG). Returns order ID or None."""
        if qty <= 0:
            return None
        try:
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("BUY  %s x%d — order %s", symbol, qty, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("BUY order failed for %s: %s", symbol, exc)
            return None

    def sell(self, symbol: str, qty: int, reason: str = "") -> Optional[str]:
        """Submit a market sell order (close LONG). Returns order ID or None."""
        if qty <= 0:
            return None
        try:
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("SELL %s x%d (%s) — order %s", symbol, qty, reason, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("SELL order failed for %s: %s", symbol, exc)
            return None

    def sell_short(self, symbol: str, qty: int) -> Optional[str]:
        """Submit a market sell order to open a SHORT position."""
        if qty <= 0:
            return None
        try:
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("SHORT %s x%d — order %s", symbol, qty, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("SHORT order failed for %s: %s", symbol, exc)
            return None

    def buy_to_cover(self, symbol: str, qty: int, reason: str = "") -> Optional[str]:
        """Submit a market buy order to close a SHORT position."""
        if qty <= 0:
            return None
        try:
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("COVER %s x%d (%s) — order %s", symbol, qty, reason, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("COVER order failed for %s: %s", symbol, exc)
            return None

    # -- ML prediction -------------------------------------------------
    def get_prediction(self, symbol: str) -> dict:
        """Get ML prediction for a symbol."""
        predictor = self.predictors.get(symbol)
        if predictor is None:
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

        try:
            if self.mode == "intraday":
                bars = self.adapter.fetch_intraday(
                    symbol, self.intraday_interval, lookback_days=2)
            else:
                bars = self.adapter.fetch_daily(symbol, DAILY_LOOKBACK)
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=DAILY_LOOKBACK)
            return predictor.predict(bars, vix_df)
        except Exception as exc:
            log.error("Prediction failed for %s: %s", symbol, exc)
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest price for a symbol."""
        try:
            if self.mode == "intraday":
                bars = self.adapter.fetch_intraday(symbol, self.intraday_interval)
            else:
                bars = self.adapter.fetch_daily(symbol, 5)
            if bars.empty:
                return None
            return float(bars["close"].iloc[-1])
        except Exception:
            return None

    # -- Trading logic (one symbol) ------------------------------------
    def check_and_trade(self, symbol: str, positions: Dict[str, dict],
                        allocation: float) -> str:
        """Check ML signal and manage position for one symbol.

        Supports LONG/SHORT positions with trailing stop, take-profit,
        and immediate flip on signal change.

        Returns action string for display.
        """
        pred = self.get_prediction(symbol)
        direction = pred["direction"]
        confidence = pred["confidence"]

        pos = positions.get(symbol)
        has_position = pos is not None and pos["qty"] > 0
        flip_direction = None

        # --- Exit logic ---
        if has_position:
            qty = int(pos["qty"])
            side = pos["side"]
            current_price = pos["current_price"]
            entry_price = pos["entry_price"]
            pnl_pct = pos["unrealized_pnl_pct"]

            # Update peak tracking
            if symbol not in self._peak_prices:
                self._peak_prices[symbol] = current_price
            if side == "LONG":
                self._peak_prices[symbol] = max(self._peak_prices[symbol], current_price)
                drawdown_from_peak = ((self._peak_prices[symbol] - current_price)
                                      / self._peak_prices[symbol])
            else:  # SHORT
                self._peak_prices[symbol] = min(self._peak_prices[symbol], current_price)
                drawdown_from_peak = ((current_price - self._peak_prices[symbol])
                                      / self._peak_prices[symbol]
                                      if self._peak_prices[symbol] > 0 else 0)

            # Take profit
            if pnl_pct >= self.take_profit_pct:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"take_profit ({pnl_pct:.2%})")
                else:
                    self.buy_to_cover(symbol, qty, reason=f"take_profit ({pnl_pct:.2%})")
                self._peak_prices.pop(symbol, None)
                return (f"EXIT  (take profit {pnl_pct:+.2%}, {qty} sh {side})  "
                        f"ML: {direction} {confidence:.2f}")

            # Trailing stop
            if drawdown_from_peak >= self.trailing_stop_pct:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"trailing_stop ({drawdown_from_peak:.2%})")
                else:
                    self.buy_to_cover(symbol, qty, reason=f"trailing_stop ({drawdown_from_peak:.2%})")
                self._peak_prices.pop(symbol, None)
                return (f"EXIT  (trailing stop {drawdown_from_peak:.2%} from peak, "
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Signal flip (lower exit confidence)
            if (side == "LONG" and direction == "DOWN"
                    and confidence >= self.exit_confidence):
                self.sell(symbol, qty, reason="signal_flip")
                self._peak_prices.pop(symbol, None)
                flip_direction = "SHORT"
            elif (side == "SHORT" and direction == "UP"
                  and confidence >= self.exit_confidence):
                self.buy_to_cover(symbol, qty, reason="signal_flip")
                self._peak_prices.pop(symbol, None)
                flip_direction = "LONG"
            else:
                # Hold
                return (f"HOLD  ({side} {qty} sh @ ${entry_price:.2f}, "
                        f"P&L: {pnl_pct:+.2%})  ML: {direction} {confidence:.2f}")

        # --- Entry logic (includes immediate flip) ---
        enter_dir = None
        if flip_direction is not None and confidence >= self.exit_confidence:
            enter_dir = flip_direction
        elif confidence >= self.confidence_threshold:
            if direction == "UP":
                enter_dir = "LONG"
            elif direction == "DOWN":
                enter_dir = "SHORT"

        if enter_dir is not None:
            current_price = self._get_current_price(symbol)
            if current_price is None:
                action = "flip" if flip_direction else "entry"
                return (f"SKIP  (price fetch failed for {action})  "
                        f"ML: {direction} {confidence:.2f}")

            invest = allocation * self.position_pct
            qty = int(invest / current_price)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} {confidence:.2f}"

            if enter_dir == "LONG":
                self.buy(symbol, qty)
                self._peak_prices[symbol] = current_price
                verb = "FLIP->BUY" if flip_direction else "BUY"
            else:
                self.sell_short(symbol, qty)
                self._peak_prices[symbol] = current_price
                verb = "FLIP->SHORT" if flip_direction else "SHORT"

            return (f"{verb}  ({qty} sh @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f})  ML: {direction} {confidence:.2f}")

        if flip_direction:
            return (f"EXIT  (signal flip, no re-entry)  "
                    f"ML: {direction} {confidence:.2f}")

        # No signal
        return f"SKIP  (no signal)  ML: {direction} {confidence:.2f}"

    # -- Main loop -----------------------------------------------------
    def run_loop(self) -> None:
        """Main continuous trading loop."""
        log.info("Starting paper trading loop (%s mode)...", self.mode)
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("Check interval: %d min", self.check_interval // 60)
        log.info("Entry confidence: %.2f, Exit confidence: %.2f",
                 self.confidence_threshold, self.exit_confidence)
        log.info("Trailing stop: %.1f%%, Take profit: %.1f%%",
                 self.trailing_stop_pct * 100, self.take_profit_pct * 100)
        print()

        # Graceful shutdown
        def handle_signal(sig, frame):
            print("\n\n  Shutting down paper trader...\n")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        while self._running:
            cycle += 1

            # Market hours check
            if not _is_market_open():
                next_open = _time_until_next_open()
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Market closed. Next open in {next_open}. "
                      f"Checking again in {self.check_interval // 60} min...")
                self._sleep(self.check_interval)
                continue

            try:
                # Get account + positions
                account = self.get_account_summary()
                positions = self.get_positions()
                allocation_per_sym = account["equity"] / len(self.symbols)

                # Print header
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"=== Paper Trading Cycle #{cycle} ({self.mode}) ===")
                print(f"  Account: ${account['equity']:,.2f} equity | "
                      f"${account['cash']:,.2f} cash | "
                      f"${account['equity'] - account['cash']:,.2f} in positions")
                print(f"  Allocation per symbol: ${allocation_per_sym:,.2f}")
                print()

                # Check each symbol
                for sym in self.symbols:
                    action = self.check_and_trade(sym, positions, allocation_per_sym)
                    print(f"  {sym:>5}:  {action}")

                print(f"\n  Next check in {self.check_interval // 60} min...")

            except Exception as exc:
                log.error("Error in trading cycle: %s", exc)

            self._sleep(self.check_interval)

        print("  Paper trader stopped.\n")

    def _sleep(self, seconds: int) -> None:
        """Interruptible sleep."""
        end = time.time() + seconds
        while time.time() < end and self._running:
            time.sleep(1)


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpaca paper trader — continuous ML-driven trading loop.",
    )
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: SPY,QQQ,IWM,IGV,SLV)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"],
                        help="Data provider (default: yahoo)")
    parser.add_argument("--check-interval", type=int, default=None,
                        help="Check interval in minutes (default: 5 daily, 1 intraday)")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Min ML confidence to enter (default: 0.2)")
    parser.add_argument("--exit-confidence", type=float, default=0.1,
                        help="Min ML confidence to exit/flip (default: 0.1)")
    parser.add_argument("--trailing-stop", type=float, default=0.05,
                        help="Trailing stop from peak (default: 0.05 = 5%%)")
    parser.add_argument("--take-profit", type=float, default=0.08,
                        help="Take profit target (default: 0.08 = 8%%)")
    parser.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                        help="Trading mode (default: daily)")
    parser.add_argument("--interval", default="5min", choices=["1min", "5min"],
                        help="Intraday bar interval (default: 5min)")

    args = parser.parse_args()

    # Validate API keys
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        print("\n  ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        print("  Get free keys at https://app.alpaca.markets/signup\n")
        sys.exit(1)

    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else DEFAULT_UNIVERSE)

    # Default check interval: 1 min for intraday, 5 min for daily
    check_interval = args.check_interval
    if check_interval is None:
        check_interval = 1 if args.mode == "intraday" else 5

    trader = AlpacaPaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        provider=args.provider,
        confidence_threshold=args.confidence,
        exit_confidence=args.exit_confidence,
        trailing_stop_pct=args.trailing_stop,
        take_profit_pct=args.take_profit,
        check_interval_min=check_interval,
        mode=args.mode,
        intraday_interval=args.interval,
    )

    # Show account before starting
    try:
        account = trader.get_account_summary()
        print(f"\n  Connected to Alpaca Paper Trading")
        print(f"  Account equity: ${account['equity']:,.2f}")
        print(f"  Buying power:   ${account['buying_power']:,.2f}")
        print(f"  Mode: {args.mode}")
    except Exception as exc:
        print(f"\n  ERROR: Could not connect to Alpaca: {exc}")
        print("  Check your API keys and try again.\n")
        sys.exit(1)

    trader.run_loop()


if __name__ == "__main__":
    main()
