#!/usr/bin/env python3
"""
Alpaca Paper Trader — Continuous Loop
=======================================
Runs the LSTM ML model in a loop and executes trades on Alpaca paper trading.

Usage (via main.py):
    python main.py trade
    python main.py trade --interval 5 --confidence 0.2 --stop-loss 0.03
    python main.py trade --symbols SPY,QQQ --interval 10

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
    """Continuous paper trading loop driven by ML predictions."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        provider: str = "yahoo",
        confidence_threshold: float = 0.2,
        stop_loss_pct: float = 0.03,
        position_pct: float = 0.90,
        check_interval_min: int = 5,
        model_dir: str = DEFAULT_MODEL_DIR,
    ):
        # Trading client — PAPER ONLY
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,  # HARDCODED for safety
        )
        self.symbols = symbols
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.position_pct = position_pct
        self.check_interval = check_interval_min * 60  # seconds

        # Data adapter for fetching prices
        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        # ML predictors — one per symbol
        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            try:
                self.predictors[sym] = Predictor(sym, model_dir=model_dir)
                log.info("Loaded ML model for %s.", sym)
            except FileNotFoundError:
                log.warning("No trained model for %s — will skip ML signals.", sym)
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
        """Get current positions as {symbol: {qty, entry, current, pnl, pnl_pct}}."""
        positions = self.trading_client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pnl": float(pos.unrealized_pl),
                "unrealized_pnl_pct": float(pos.unrealized_plpc),
            }
        return result

    # -- Order execution -----------------------------------------------
    def buy(self, symbol: str, qty: int) -> Optional[str]:
        """Submit a market buy order. Returns order ID or None."""
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
        """Submit a market sell order. Returns order ID or None."""
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

    # -- ML prediction -------------------------------------------------
    def get_prediction(self, symbol: str) -> dict:
        """Get ML prediction for a symbol."""
        predictor = self.predictors.get(symbol)
        if predictor is None:
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

        try:
            daily = self.adapter.fetch_daily(symbol, DAILY_LOOKBACK)
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=DAILY_LOOKBACK)
            return predictor.predict(daily, vix_df)
        except Exception as exc:
            log.error("Prediction failed for %s: %s", symbol, exc)
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

    # -- Trading logic (one symbol) ------------------------------------
    def check_and_trade(self, symbol: str, positions: Dict[str, dict],
                        allocation: float) -> str:
        """Check ML signal and manage position for one symbol.

        Returns action string for display.
        """
        pred = self.get_prediction(symbol)
        direction = pred["direction"]
        confidence = pred["confidence"]

        pos = positions.get(symbol)
        has_position = pos is not None and pos["qty"] > 0

        # --- Exit logic ---
        if has_position:
            pnl_pct = pos["unrealized_pnl_pct"]

            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                qty = int(pos["qty"])
                self.sell(symbol, qty, reason=f"stop_loss ({pnl_pct:.2%})")
                return (f"SELL  (stop loss {pnl_pct:+.2%}, {qty} shares)  "
                        f"ML: {direction} {confidence:.2f}")

            # Signal flip
            if direction == "DOWN" and confidence >= self.confidence_threshold:
                qty = int(pos["qty"])
                self.sell(symbol, qty, reason="signal_flip")
                return (f"SELL  (signal flip, {qty} shares)  "
                        f"ML: {direction} {confidence:.2f}")

            # Hold
            return (f"HOLD  (pos: {int(pos['qty'])} shares @ ${pos['entry_price']:.2f}, "
                    f"P&L: ${pos['unrealized_pnl']:+,.2f})  "
                    f"ML: {direction} {confidence:.2f}")

        # --- Entry logic ---
        if direction == "UP" and confidence >= self.confidence_threshold:
            # Calculate shares to buy
            try:
                daily = self.adapter.fetch_daily(symbol, 5)
                if daily.empty:
                    return f"SKIP  (no price data)  ML: {direction} {confidence:.2f}"
                current_price = float(daily["close"].iloc[-1])
            except Exception:
                return f"SKIP  (price fetch failed)  ML: {direction} {confidence:.2f}"

            invest = allocation * self.position_pct
            qty = int(invest / current_price)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} {confidence:.2f}"

            self.buy(symbol, qty)
            return (f"BUY   ({qty} shares @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f})  "
                    f"ML: {direction} {confidence:.2f}")

        # No signal
        return f"SKIP  (no signal)  ML: {direction} {confidence:.2f}"

    # -- Main loop -----------------------------------------------------
    def run_loop(self) -> None:
        """Main continuous trading loop."""
        log.info("Starting paper trading loop...")
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("Check interval: %d min", self.check_interval // 60)
        log.info("Confidence threshold: %.2f", self.confidence_threshold)
        log.info("Stop loss: %.1f%%", self.stop_loss_pct * 100)
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
                      f"=== Paper Trading Cycle #{cycle} ===")
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
    parser.add_argument("--interval", type=int, default=5,
                        help="Check interval in minutes (default: 5)")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Min ML confidence to trade (default: 0.2)")
    parser.add_argument("--stop-loss", type=float, default=0.03,
                        help="Stop loss percentage (default: 0.03 = 3%%)")

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

    trader = AlpacaPaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        provider=args.provider,
        confidence_threshold=args.confidence,
        stop_loss_pct=args.stop_loss,
        check_interval_min=args.interval,
    )

    # Show account before starting
    try:
        account = trader.get_account_summary()
        print(f"\n  Connected to Alpaca Paper Trading")
        print(f"  Account equity: ${account['equity']:,.2f}")
        print(f"  Buying power:   ${account['buying_power']:,.2f}")
    except Exception as exc:
        print(f"\n  ERROR: Could not connect to Alpaca: {exc}")
        print("  Check your API keys and try again.\n")
        sys.exit(1)

    trader.run_loop()


if __name__ == "__main__":
    main()
