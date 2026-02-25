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

from signals_engine import (
    DEFAULT_UNIVERSE, DAILY_LOOKBACK, build_adapter, FREDVixFetcher,
    compute_atr, compute_adx,
)
from ml_model import Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paper_trader")

# Hold lock file handle for process lifetime
_INSTANCE_LOCK_FH = None


def _acquire_single_instance_lock() -> bool:
    """Return False if another paper trader instance is already running.

    Uses a PID file + byte-range lock. Stale locks (PID no longer running)
    are automatically cleared so a force-killed process never blocks restarts.
    """
    global _INSTANCE_LOCK_FH

    lock_path = os.path.join(
        os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__))),
        ".paper_trader.lock",
    )

    def _pid_alive(pid: int) -> bool:
        """Return True if a process with this PID is still running."""
        try:
            if os.name == "nt":
                import ctypes
                SYNCHRONIZE = 0x00100000
                handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
                if not handle:
                    return False
                result = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
                ctypes.windll.kernel32.CloseHandle(handle)
                return result != 0  # 0 = WAIT_OBJECT_0 means process exited
            else:
                os.kill(pid, 0)
                return True
        except (OSError, PermissionError):
            return False

    for attempt in range(2):
        try:
            lock_fh = open(lock_path, "w+")
        except OSError:
            log.warning("Could not open single-instance lock file; continuing without lock.")
            return True

        try:
            if os.name == "nt":
                import msvcrt
                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Lock acquired — write our PID so future starts can detect stale locks
            lock_fh.seek(0)
            lock_fh.write(str(os.getpid()))
            lock_fh.flush()
            _INSTANCE_LOCK_FH = lock_fh
            return True

        except (OSError, BlockingIOError):
            lock_fh.close()
            if attempt == 0:
                # Read the PID from the lock file to check if it's actually alive
                try:
                    with open(lock_path) as f:
                        owner_pid = int(f.read().strip())
                    if not _pid_alive(owner_pid):
                        log.warning(
                            "Stale lock file (PID %d no longer running) — removing and retrying.",
                            owner_pid,
                        )
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
                        time.sleep(0.5)
                        continue  # retry
                except (ValueError, OSError):
                    # Can't read PID; remove and retry once
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass
                    time.sleep(0.5)
                    continue
            return False

    return False


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

    Supports LONG and SHORT positions with:
    - ATR-based adaptive trailing stop
    - Volatility-adjusted, confidence-scaled position sizing
    - Regime filter (ADX-based trend detection)
    - Trailing take-profit that locks in gains progressively
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        provider: str = "yahoo",
        confidence_threshold: float = 0.2,
        short_confidence_threshold: float = 0.15,
        exit_confidence: float = 0.1,
        trailing_stop_pct: float = 0.05,
        take_profit_pct: float = 0.08,
        position_pct: float = 0.90,
        check_interval_min: int = 5,
        model_dir: str = DEFAULT_MODEL_DIR,
        mode: str = "daily",
        intraday_interval: str = "5min",
        # --- Optimization parameters ---
        use_atr_stop: bool = True,
        atr_stop_mult: float = 2.5,
        use_regime_filter: bool = True,
        adx_threshold: float = 20.0,
        use_vol_sizing: bool = True,
        target_vol: float = 0.15,
        use_confidence_sizing: bool = True,
        use_trailing_tp: bool = True,
        trailing_tp_activation: float = 0.04,
        trailing_tp_trail: float = 0.02,
    ):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,  # HARDCODED for safety
        )
        self.symbols = symbols
        self.long_confidence_threshold = confidence_threshold
        self.short_confidence_threshold = short_confidence_threshold
        self.exit_confidence = exit_confidence
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.position_pct = position_pct
        self.check_interval = check_interval_min * 60
        self.mode = mode
        self.intraday_interval = intraday_interval
        # Optimization params
        self.use_atr_stop = use_atr_stop
        self.atr_stop_mult = atr_stop_mult
        self.use_regime_filter = use_regime_filter
        self.adx_threshold = adx_threshold
        self.use_vol_sizing = use_vol_sizing
        self.target_vol = target_vol
        self.use_confidence_sizing = use_confidence_sizing
        self.use_trailing_tp = use_trailing_tp
        self.trailing_tp_activation = trailing_tp_activation
        self.trailing_tp_trail = trailing_tp_trail

        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        # Per-symbol tracking
        self._peak_prices: Dict[str, float] = {}
        self._entry_atrs: Dict[str, float] = {}
        self._tp_activated: Dict[str, bool] = {}
        self._tp_trail_peaks: Dict[str, float] = {}

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

    def _get_market_context(self, symbol: str) -> Dict[str, float]:
        """Fetch ATR, ADX, and realized vol for position sizing and stops."""
        try:
            bars = self.adapter.fetch_daily(symbol, 60)
            if len(bars) < 30:
                return {"atr": 0.0, "adx": 25.0, "vol20": 0.15}
            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            atr_s = compute_atr(high, low, close, period=14)
            adx_s = compute_adx(high, low, close, period=14)
            vol_s = close.pct_change().rolling(20).std() * np.sqrt(252)
            return {
                "atr": float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0,
                "adx": float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 25.0,
                "vol20": float(vol_s.iloc[-1]) if not np.isnan(vol_s.iloc[-1]) else 0.15,
            }
        except Exception as exc:
            log.warning("Market context fetch failed for %s: %s", symbol, exc)
            return {"atr": 0.0, "adx": 25.0, "vol20": 0.15}

    # -- Trading logic (one symbol) ------------------------------------
    def check_and_trade(self, symbol: str, positions: Dict[str, dict],
                        allocation: float) -> str:
        """Check ML signal and manage position for one symbol.

        Optimized strategy with ATR-based stops, vol sizing, regime filter,
        and trailing take-profit.

        Returns action string for display.
        """
        pred = self.get_prediction(symbol)
        direction = pred["direction"]
        confidence = pred["confidence"]

        ctx = self._get_market_context(symbol)
        bar_atr = ctx["atr"]
        bar_adx = ctx["adx"]
        bar_vol = ctx["vol20"]

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

            if symbol not in self._peak_prices:
                self._peak_prices[symbol] = current_price
            if side == "LONG":
                self._peak_prices[symbol] = max(self._peak_prices[symbol], current_price)
                drawdown_from_peak = ((self._peak_prices[symbol] - current_price)
                                      / self._peak_prices[symbol])
            else:
                self._peak_prices[symbol] = min(self._peak_prices[symbol], current_price)
                drawdown_from_peak = ((current_price - self._peak_prices[symbol])
                                      / self._peak_prices[symbol]
                                      if self._peak_prices[symbol] > 0 else 0)

            # ATR-based or fixed stop distance
            entry_atr = self._entry_atrs.get(symbol, bar_atr)
            if self.use_atr_stop and entry_atr > 0 and entry_price > 0:
                stop_distance = self.atr_stop_mult * entry_atr / entry_price
            else:
                stop_distance = self.trailing_stop_pct

            # Trailing take-profit: lock in gains once threshold is reached
            if self.use_trailing_tp and pnl_pct >= self.trailing_tp_activation:
                if not self._tp_activated.get(symbol, False):
                    self._tp_activated[symbol] = True
                    self._tp_trail_peaks[symbol] = pnl_pct
                    log.info("Trailing TP activated for %s at %+.2f%%", symbol, pnl_pct * 100)
                self._tp_trail_peaks[symbol] = max(
                    self._tp_trail_peaks.get(symbol, pnl_pct), pnl_pct)
                if pnl_pct < self._tp_trail_peaks[symbol] - self.trailing_tp_trail:
                    if side == "LONG":
                        self.sell(symbol, qty, reason=f"trailing_tp ({pnl_pct:+.2%})")
                    else:
                        self.buy_to_cover(symbol, qty, reason=f"trailing_tp ({pnl_pct:+.2%})")
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (trailing TP {pnl_pct:+.2%}, peak {self._tp_trail_peaks.get(symbol, 0):+.2%}, "
                            f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Trailing stop
            if drawdown_from_peak >= stop_distance:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"trailing_stop ({drawdown_from_peak:.2%})")
                else:
                    self.buy_to_cover(symbol, qty, reason=f"trailing_stop ({drawdown_from_peak:.2%})")
                self._clear_symbol_state(symbol)
                return (f"EXIT  (trailing stop {drawdown_from_peak:.2%} from peak, "
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Signal flip
            if (side == "LONG" and direction == "DOWN"
                    and confidence >= self.exit_confidence):
                self.sell(symbol, qty, reason="signal_flip")
                self._clear_symbol_state(symbol)
                flip_direction = "SHORT"
            elif (side == "SHORT" and direction == "UP"
                  and confidence >= self.exit_confidence):
                self.buy_to_cover(symbol, qty, reason="signal_flip")
                self._clear_symbol_state(symbol)
                flip_direction = "LONG"
            else:
                stop_info = f"ATR×{self.atr_stop_mult}" if self.use_atr_stop else f"{self.trailing_stop_pct:.0%}"
                return (f"HOLD  ({side} {qty} sh @ ${entry_price:.2f}, "
                        f"P&L: {pnl_pct:+.2%}, stop={stop_info})  "
                        f"ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}")

        # --- Entry logic ---
        # Regime filter: skip in trendless markets
        if self.use_regime_filter and bar_adx < self.adx_threshold:
            if flip_direction:
                return (f"EXIT  (signal flip, no re-entry: ADX={bar_adx:.0f}<{self.adx_threshold:.0f})  "
                        f"ML: {direction} {confidence:.2f}")
            return f"SKIP  (low trend ADX={bar_adx:.0f}<{self.adx_threshold:.0f})  ML: {direction} {confidence:.2f}"

        enter_dir = None
        if flip_direction is not None and confidence >= self.exit_confidence:
            enter_dir = flip_direction
        elif direction == "UP" and confidence >= self.long_confidence_threshold:
            enter_dir = "LONG"
        elif direction == "DOWN" and confidence >= self.short_confidence_threshold:
            enter_dir = "SHORT"

        if enter_dir is not None:
            current_price = self._get_current_price(symbol)
            if current_price is None:
                action = "flip" if flip_direction else "entry"
                return (f"SKIP  (price fetch failed for {action})  "
                        f"ML: {direction} {confidence:.2f}")

            # Volatility-adjusted + confidence-scaled sizing
            sizing_pct = self.position_pct
            if self.use_vol_sizing and bar_vol > 0:
                vol_scalar = min(2.0, max(0.3, self.target_vol / bar_vol))
                sizing_pct *= vol_scalar
            if self.use_confidence_sizing:
                conf_scalar = 0.5 + confidence
                sizing_pct *= conf_scalar
            sizing_pct = min(sizing_pct, 0.98)

            invest = allocation * sizing_pct
            qty = int(invest / current_price)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} {confidence:.2f}"

            if enter_dir == "LONG":
                self.buy(symbol, qty)
                verb = "FLIP->BUY" if flip_direction else "BUY"
            else:
                self.sell_short(symbol, qty)
                verb = "FLIP->SHORT" if flip_direction else "SHORT"

            self._peak_prices[symbol] = current_price
            self._entry_atrs[symbol] = bar_atr
            self._tp_activated[symbol] = False
            self._tp_trail_peaks[symbol] = 0.0

            return (f"{verb}  ({qty} sh @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f}, size={sizing_pct:.0%})  "
                    f"ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}")

        if flip_direction:
            return (f"EXIT  (signal flip, no re-entry)  "
                    f"ML: {direction} {confidence:.2f}")

        return f"SKIP  (no signal)  ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}"

    def _clear_symbol_state(self, symbol: str) -> None:
        """Reset tracking state for a symbol after exit."""
        self._peak_prices.pop(symbol, None)
        self._entry_atrs.pop(symbol, None)
        self._tp_activated.pop(symbol, None)
        self._tp_trail_peaks.pop(symbol, None)

    # -- Main loop -----------------------------------------------------
    def run_loop(self) -> None:
        """Main continuous trading loop."""
        log.info("Starting paper trading loop (%s mode)...", self.mode)
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("Check interval: %d min", self.check_interval // 60)
        log.info("Entry confidence: LONG %.2f, SHORT %.2f, Exit: %.2f",
                 self.long_confidence_threshold, self.short_confidence_threshold, self.exit_confidence)
        log.info("Stops: %s | Regime filter: %s (ADX>%.0f) | Vol sizing: %s (target=%.0f%%) | "
                 "Confidence sizing: %s | Trailing TP: %s",
                 f"ATR×{self.atr_stop_mult}" if self.use_atr_stop else f"Fixed {self.trailing_stop_pct:.0%}",
                 "ON" if self.use_regime_filter else "OFF", self.adx_threshold,
                 "ON" if self.use_vol_sizing else "OFF", self.target_vol * 100,
                 "ON" if self.use_confidence_sizing else "OFF",
                 "ON" if self.use_trailing_tp else "OFF")
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
    if not _acquire_single_instance_lock():
        msg = "Another paper trader instance is already running. Exiting duplicate process."
        log.warning(msg)
        print(f"\n  WARNING: {msg}\n")
        sys.exit(0)

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
                        help="Min ML confidence to enter LONG (default: 0.2)")
    parser.add_argument("--short-confidence", type=float, default=0.15,
                        help="Min ML confidence to enter SHORT (default: 0.15, more aggressive)")
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
    # Optimization flags
    parser.add_argument("--no-atr-stop", action="store_true",
                        help="Disable ATR-based adaptive trailing stop")
    parser.add_argument("--atr-stop-mult", type=float, default=2.5,
                        help="ATR multiplier for trailing stop (default: 2.5)")
    parser.add_argument("--no-regime-filter", action="store_true",
                        help="Disable ADX regime filter")
    parser.add_argument("--adx-threshold", type=float, default=20.0,
                        help="Min ADX to allow entries (default: 20.0)")
    parser.add_argument("--no-vol-sizing", action="store_true",
                        help="Disable volatility-adjusted position sizing")
    parser.add_argument("--target-vol", type=float, default=0.15,
                        help="Target annualized vol for sizing (default: 0.15)")
    parser.add_argument("--no-confidence-sizing", action="store_true",
                        help="Disable confidence-scaled position sizing")
    parser.add_argument("--no-trailing-tp", action="store_true",
                        help="Disable trailing take-profit")

    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        print("\n  ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        print("  Get free keys at https://app.alpaca.markets/signup\n")
        sys.exit(1)

    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else DEFAULT_UNIVERSE)

    check_interval = args.check_interval
    if check_interval is None:
        check_interval = 1 if args.mode == "intraday" else 5

    trader = AlpacaPaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        provider=args.provider,
        confidence_threshold=args.confidence,
        short_confidence_threshold=args.short_confidence,
        exit_confidence=args.exit_confidence,
        trailing_stop_pct=args.trailing_stop,
        take_profit_pct=args.take_profit,
        check_interval_min=check_interval,
        mode=args.mode,
        intraday_interval=args.interval,
        use_atr_stop=not args.no_atr_stop,
        atr_stop_mult=args.atr_stop_mult,
        use_regime_filter=not args.no_regime_filter,
        adx_threshold=args.adx_threshold,
        use_vol_sizing=not args.no_vol_sizing,
        target_vol=args.target_vol,
        use_confidence_sizing=not args.no_confidence_sizing,
        use_trailing_tp=not args.no_trailing_tp,
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
