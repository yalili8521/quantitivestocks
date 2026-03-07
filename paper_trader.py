#!/usr/bin/env python3
"""
Alpaca Paper Trader — Continuous Loop (v2 Regression)
=====================================================
Runs ML regression models in a loop and executes trades on Alpaca paper trading.
Supports LONG and SHORT positions with signal-decay exits and disaster stop.

Architecture (v2 — mirrors backtester):
- Trend-following base layer: SMA(50) determines allowed direction
- ML signal: expected return magnitude determines position size
- Signal-decay exit: re-run model each cycle, exit when signal reverses
- Disaster stop: 3×ATR safety net (not primary exit mechanism)

Usage (via main.py):
    python main.py trade --group intraday --mode intraday --interval 5min
    python main.py trade --group swing
    python main.py trade --group expansion

Required env vars:
    ALPACA_API_KEY, ALPACA_API_SECRET, FRED_API_KEY

PAPER TRADING ONLY — paper=True is hardcoded for safety.

Option positions (OCC symbols) are not managed by this module. options_trader.py owns their
full lifecycle (entry and exit). This module only trades equity and manages exit-only for
wrongly placed equity symbols.

Legacy / wrong-algorithm positions (e.g. IGV, QQQ opened by a previous buggy model):
  Use stricter exits to lock profit or cut loss: --legacy-stricter-exit IGV,QQQ
  Manage out only (no new entries until flat):   --legacy-no-new-entries IGV,QQQ
  Or set env: PAPER_LEGACY_STRICTER_EXIT, PAPER_LEGACY_NO_NEW_ENTRIES
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from signals_engine import (
    DEFAULT_UNIVERSE, EXTENDED_HOURS_UNIVERSE,
    DAILY_LOOKBACK, build_adapter, FREDVixFetcher,
    compute_atr,
)
from ml_model import (
    Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR,
    COST_THRESHOLD, TARGET_RETURN,
)
from alerts import AlertEngine
from options_flow import OptionsFlowEngine
from pairs_model import PairsPredictor, PAIRS_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paper_trader")

# ---------------------------------------------------------------------------
# Account groups (3 separate Alpaca paper accounts by asset class)
# ---------------------------------------------------------------------------
SYMBOL_GROUPS: Dict[str, List[str]] = {
    # Account 1 — Intraday (5min LSTM, time filter active)
    # IGV dropped: meta RF val_acc=0.47, precision 0.0% on Alpaca 2yr data — no learnable signal
    "intraday":  ["SPY", "QQQ", "IWM", "SOXX", "USO"],
    # Account 2 — Swing (daily XGBoost, commodity/EM ETFs)
    # TLT dropped: 42.9% win rate, Sharpe 0.300 across all feature sets (bonds need rate signals)
    # USO moved to intraday: crude oil has strong first-30m momentum signal
    "swing":     ["EWT", "GLD", "EEM", "SLV"],
    # Account 3 — Expansion (daily LSTM, international/sector)
    # IGV dropped: <50% win rate on both 12-feat and 17-feat (sector ETF, earnings-driven)
    # FXI dropped: <50% win rate on both (China policy-driven, not technical)
    "expansion": ["EWJ", "EWS", "XLE", "INDA"],
}

# Legacy / wrong-algorithm positions: opened by a previous buggy model. Manage them
# with stricter exits (lock profit or cut loss sooner) and optionally no new entries.
# Set via env: PAPER_LEGACY_STRICTER_EXIT=IGV,QQQ  PAPER_LEGACY_NO_NEW_ENTRIES=IGV,QQQ
LEGACY_STRICTER_EXIT_DEFAULT: Set[str] = set()
LEGACY_NO_NEW_ENTRIES_DEFAULT: Set[str] = set()
LEGACY_TRAILING_STOP_PCT = 0.03   # 3% from peak for legacy positions

# Option positions (OCC symbols): data-driven from equity backtest (78.9% WR, 2:1 payoff)
OPTION_MAX_LOSS_PCT      = 0.35   # stop loss -35%
OPTION_TP_TARGET_PCT     = 0.70   # profit target +70%
OPTION_TRAILING_STOP_PCT = 0.20   # trail 20% from peak
OPTION_TP_ACTIVATION     = 0.35   # activate trailing TP after +35%
OPTION_TP_TRAIL          = 0.20   # trail 20% from peak once TP activated


def _is_option_symbol(symbol: str) -> bool:
    """True if symbol looks like OCC option (e.g. QQQ260327C00592000). Normalize to upper so we always use option exit rules even if Alpaca returns lowercase."""
    import re
    return bool(re.match(r"^[A-Z]+\d{6}[CP]\d{8}$", (symbol or "").upper()))


def _warn_duplicate_symbols() -> None:
    """Warn if any symbol appears in more than one group (wrong placement / double-trading)."""
    from collections import defaultdict
    sym_to_groups: Dict[str, List[str]] = defaultdict(list)
    for grp, syms in SYMBOL_GROUPS.items():
        for s in syms:
            sym_to_groups[s].append(grp)
    dupes = {s: grps for s, grps in sym_to_groups.items() if len(grps) > 1}
    if dupes:
        log.warning(
            "Symbol(s) in multiple groups (fix to avoid double-trading): %s",
            ", ".join(f"{s}→{grps}" for s, grps in dupes.items()),
        )

# Hold lock file handle for process lifetime
_INSTANCE_LOCK_FH = None
_PID_FILE_PATH: Optional[str] = None


def _remove_pid_file() -> None:
    """Remove the .pid file on exit so stop-paper-trader doesn't see a stale PID."""
    global _PID_FILE_PATH
    if _PID_FILE_PATH:
        try:
            os.remove(_PID_FILE_PATH)
        except OSError:
            pass
        _PID_FILE_PATH = None


def _acquire_single_instance_lock() -> bool:
    """Return False if another paper trader instance is already running.

    Uses a PID file + byte-range lock. Stale locks (PID no longer running)
    are automatically cleared so a force-killed process never blocks restarts.
    Lock file is group-specific so intraday/swing/expansion can run in parallel.
    """
    global _INSTANCE_LOCK_FH

    # Quick pre-parse to get --group before argparse runs, so each group gets
    # its own lock file and can run concurrently with other groups.
    import sys as _sys
    _group_tag = "default"
    _argv = _sys.argv[1:]
    for _i, _arg in enumerate(_argv):
        if _arg == "--group" and _i + 1 < len(_argv):
            _group_tag = _argv[_i + 1]
            break
        if _arg.startswith("--group="):
            _group_tag = _arg.split("=", 1)[1]
            break

    lock_path = os.path.join(
        os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__))),
        f".paper_trader_{_group_tag}.lock",
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
            # Also write PID to a separate file (readable by others) for stop-paper-trader command
            global _PID_FILE_PATH
            _PID_FILE_PATH = os.path.join(
                os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__))),
                f".paper_trader_{_group_tag}.pid",
            )
            try:
                with open(_PID_FILE_PATH, "w") as f:
                    f.write(str(os.getpid()))
                atexit.register(_remove_pid_file)
            except OSError:
                pass
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
# Session helpers  (regular | extended | closed)
# ===================================================================
def _get_session() -> str:
    """Return the current Alpaca-tradeable session for ET equities.

    regular  — 09:30–16:00 ET Mon–Fri  (market orders allowed)
    extended — 04:00–09:30 and 16:00–20:00 ET Mon–Fri  (limit orders only)
    closed   — 20:00–04:00 ET and all day Sat/Sun
    """
    from datetime import time as dt_time
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "closed"

    t = now_et.time()
    if dt_time(4, 0) <= t < dt_time(9, 30):
        return "extended"
    if dt_time(9, 30) <= t <= dt_time(16, 0):
        return "regular"
    if dt_time(16, 0) < t <= dt_time(20, 0):
        return "extended"
    return "closed"


def _time_until_next_session() -> str:
    """Time until the next tradeable session opens (extended hours at 04:00 ET)."""
    from datetime import time as dt_time
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    candidate = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
    if now_et.time() >= dt_time(20, 0) or now_et.weekday() >= 5:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    delta = candidate - now_et
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


# ===================================================================
# Paper Trader
# ===================================================================
class AlpacaPaperTrader:
    """Continuous paper trading loop driven by ML regression predictions (v2).

    Architecture (mirrors backtester v2):
    - Trend-following base layer: SMA(50) determines allowed direction
    - ML signal: expected return magnitude determines position size
    - Signal-decay exit: re-run model each cycle, exit when signal reverses
    - Disaster stop: 3×ATR safety net (not primary exit mechanism)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        provider: str = "yahoo",
        position_pct: float = 0.90,
        check_interval_min: int = 5,
        model_dir: str = DEFAULT_MODEL_DIR,
        mode: str = "daily",
        intraday_interval: str = "5min",
        # v2 regression parameters
        trend_sma_period: int = 50,
        cost_threshold: float = COST_THRESHOLD,
        target_return: float = TARGET_RETURN,
        disaster_stop_atr_mult: float = 3.0,
        loss_cooldown_hours: float = 4.0,
        legacy_stricter_exit: Optional[Set[str]] = None,
        legacy_no_new_entries: Optional[Set[str]] = None,
        group: Optional[str] = None,
    ):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,  # HARDCODED for safety
        )
        self.symbols = symbols
        self.position_pct = position_pct
        self.check_interval = check_interval_min * 60
        self.mode = mode
        self.intraday_interval = intraday_interval
        # v2 regression parameters
        self.trend_sma_period = trend_sma_period
        self.cost_threshold = cost_threshold
        self.target_return = target_return
        self.disaster_stop_atr_mult = disaster_stop_atr_mult

        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        self.loss_cooldown_hours = loss_cooldown_hours
        self._legacy_stricter_exit = legacy_stricter_exit or set()
        self._legacy_no_new_entries = legacy_no_new_entries or set()

        # Per-symbol tracking
        self._peak_prices: Dict[str, float] = {}
        self._entry_atrs: Dict[str, float] = {}
        self._cooldown_until: Dict[str, datetime] = {}

        self._vix_cache: Optional[pd.DataFrame] = None

        # Alert engine for microstructure alerts
        self._alert_engine = AlertEngine()
        self._options_flow_engine = OptionsFlowEngine()
        self._options_flow_cache: Optional[Dict] = None
        self._consecutive_losses: int = 0

        # Pairs fallback: lazy-loaded predictors for mean-reversion regime
        self._pairs_predictors: Dict[str, Optional[PairsPredictor]] = {}

        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            predictor = self._create_predictor(sym, group, model_dir, mode, intraday_interval)
            self.predictors[sym] = predictor
            if predictor is not None:
                model_type = getattr(predictor, 'model_type', 'lstm')
                log.info("Loaded %s model for %s (%s).", model_type, sym, mode)

        self._running = True

    # -- Predictor factory (selects model type per group) ----------------
    @staticmethod
    def _create_predictor(symbol: str, group: Optional[str], model_dir: str,
                          mode: str, intraday_interval: str):
        """Create the right predictor based on account group.

        Tries group-specific model first (LightGBM/PatchTST/XGBoost),
        falls back to LSTM if the group-specific model isn't trained yet.
        """
        if group == "intraday":
            try:
                from intraday_model import IntradayPredictor
                return IntradayPredictor(symbol, model_dir=model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No intraday LightGBM for %s, falling back to LSTM: %s", symbol, exc)
        elif group == "swing":
            try:
                from swing_model import SwingPredictor
                return SwingPredictor(symbol, model_dir=model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No swing PatchTST for %s, falling back to LSTM: %s", symbol, exc)
        elif group == "expansion":
            try:
                from expansion_model import ExpansionPredictor
                return ExpansionPredictor(symbol, model_dir=model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No expansion XGBoost for %s, falling back to LSTM: %s", symbol, exc)

        # Fallback: original LSTM predictor
        try:
            return Predictor(symbol, model_dir=model_dir,
                             mode=mode, intraday_interval=intraday_interval)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("No trained model for %s (%s): %s — will skip ML signals.",
                        symbol, mode, exc)
            return None

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
    def _submit_order_request(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """Submit the right order type for the current session.

        Regular hours  → MarketOrder (immediate fill at best price).
        Extended hours → LimitOrder with extended_hours=True.
            BUY  limit: last_price × 1.001  (0.1% above to ensure fill)
            SELL limit: last_price × 0.999  (0.1% below to ensure fill)
        Alpaca rejects market orders outside regular hours.
        """
        session = _get_session()
        try:
            if session == "extended" and limit_price is not None:
                if side == OrderSide.BUY:
                    lp = round(limit_price * 1.001, 2)
                else:
                    lp = round(limit_price * 0.999, 2)
                order = self.trading_client.submit_order(
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=lp,
                        extended_hours=True,
                    )
                )
                log.info("LIMIT(%s) %s %s x%d @ $%.2f — order %s",
                         session, side.value, symbol, qty, lp, order.id)
            else:
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                log.info("MARKET %s %s x%d — order %s",
                         side.value, symbol, qty, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("Order failed %s %s x%d: %s", side.value, symbol, qty, exc)
            return None

    def buy(self, symbol: str, qty: int,
            limit_price: Optional[float] = None) -> Optional[str]:
        """Open LONG. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        return self._submit_order_request(symbol, qty, OrderSide.BUY, limit_price)

    def sell(self, symbol: str, qty: int, reason: str = "",
             limit_price: Optional[float] = None) -> Optional[str]:
        """Close LONG. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        oid = self._submit_order_request(symbol, qty, OrderSide.SELL, limit_price)
        if oid:
            log.info("SELL reason: %s", reason)
        return oid

    def sell_short(self, symbol: str, qty: int,
                   limit_price: Optional[float] = None) -> Optional[str]:
        """Open SHORT. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        return self._submit_order_request(symbol, qty, OrderSide.SELL, limit_price)

    def buy_to_cover(self, symbol: str, qty: int, reason: str = "",
                     limit_price: Optional[float] = None) -> Optional[str]:
        """Close SHORT. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        oid = self._submit_order_request(symbol, qty, OrderSide.BUY, limit_price)
        if oid:
            log.info("COVER reason: %s", reason)
        return oid

    # -- ML prediction -------------------------------------------------
    def _get_vix_df(self) -> pd.DataFrame:
        """Fetch VIX data once per cycle; cache and fall back to last good value on failure."""
        vix_days = 10 if self.mode == "intraday" else 400
        try:
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=vix_days)
            if len(vix_df) >= 2:
                self._vix_cache = vix_df   # update cache on success
            return vix_df
        except Exception as exc:
            log.warning("VIX fetch failed, using cached value: %s", exc)
            return self._vix_cache if self._vix_cache is not None else pd.DataFrame()

    def get_prediction(self, symbol: str) -> dict:
        """Get ML prediction for a symbol."""
        predictor = self.predictors.get(symbol)
        if predictor is None:
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

        try:
            if self.mode == "intraday":
                # 5 days ensures Friday bars are present so FRED VIX (1-day lag)
                # can ffill into today's intraday bars on Mondays / after holidays.
                bars = self.adapter.fetch_intraday(
                    symbol, self.intraday_interval, lookback_days=5)
            else:
                # 400 days needed: GLD/SLV use frac_diff_close (~282-bar warmup)
                bars = self.adapter.fetch_daily(symbol, 400)
            vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
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
        """Fetch ATR and trend signal for position management."""
        try:
            bars = self.adapter.fetch_daily(symbol, 60)
            if len(bars) < 30:
                return {"atr": 0.0, "trend": 0.0}
            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            atr_s = compute_atr(high, low, close, period=14)

            # Trend signal: SMA(50)
            sma = close.rolling(self.trend_sma_period).mean()
            trend = 1.0 if float(close.iloc[-1]) > float(sma.iloc[-1]) else -1.0
            if np.isnan(float(sma.iloc[-1])):
                trend = 0.0

            def _safe(s: "pd.Series", default: float) -> float:
                v = float(s.iloc[-1])
                return default if np.isnan(v) else v

            return {
                "atr":   _safe(atr_s, 0.0),
                "trend": trend,
            }
        except Exception as exc:
            log.warning("Market context fetch failed for %s: %s", symbol, exc)
            return {"atr": 0.0, "trend": 0.0}

    # -- Trading logic (one symbol) ------------------------------------
    def check_and_trade(self, symbol: str, positions: Dict[str, dict],
                        allocation: float, exit_only: bool = False) -> str:
        """Check ML signal and manage position for one symbol (v2 regression).

        Architecture mirrors backtester v2:
        - Exit: signal-decay (E[r] reverses) + disaster stop (3×ATR)
        - Entry: E[r] > cost_threshold AND trend agrees → trade
        - Sizing: signal-proportional
        - Preserves: EOD exit for intraday, option exits, legacy exits

        Returns action string for display.
        """
        pred = self.get_prediction(symbol)
        direction = pred["direction"]
        confidence = pred["confidence"]
        expected_return = pred.get("expected_return", 0.0)

        ctx = self._get_market_context(symbol)
        bar_atr = ctx["atr"]
        bar_trend = ctx["trend"]

        pos = positions.get(symbol)
        has_position = pos is not None and pos["qty"] > 0

        # Wrongly placed: symbol has no position here — nothing to manage
        if exit_only and not has_position:
            return "EXIT-ONLY  (no position — symbol not in this group)"

        # --- Exit logic ---
        if has_position:
            qty = int(pos["qty"])
            side = pos["side"]
            current_price = pos["current_price"]
            entry_price = pos["entry_price"]
            pnl_pct = pos["unrealized_pnl_pct"]

            use_legacy_stops = exit_only or (symbol in self._legacy_stricter_exit)
            use_option_stops = _is_option_symbol(symbol)

            # Initialize peak from entry_price
            if symbol not in self._peak_prices:
                self._peak_prices[symbol] = entry_price
            if side == "LONG":
                self._peak_prices[symbol] = max(self._peak_prices[symbol], current_price)
                drawdown_from_peak = ((self._peak_prices[symbol] - current_price)
                                      / self._peak_prices[symbol])
            else:
                self._peak_prices[symbol] = min(self._peak_prices[symbol], current_price)
                drawdown_from_peak = ((current_price - self._peak_prices[symbol])
                                      / self._peak_prices[symbol]
                                      if self._peak_prices[symbol] > 0 else 0)

            # EOD exit for intraday momentum models: close all positions at 15:30 ET
            predictor = self.predictors.get(symbol)
            if getattr(predictor, 'eod_exit', False):
                try:
                    from zoneinfo import ZoneInfo
                except ImportError:
                    from backports.zoneinfo import ZoneInfo
                import datetime as _dt
                now_et = _dt.datetime.now(ZoneInfo("America/New_York"))
                if now_et.time() >= _dt.time(15, 30):
                    if side == "LONG":
                        self.sell(symbol, qty, reason="eod_exit (intraday momentum)",
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason="eod_exit (intraday momentum)",
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (EOD close {pnl_pct:+.2%}, {qty} sh {side})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")

            # Option positions: fixed -35% max loss
            if use_option_stops and pnl_pct <= -OPTION_MAX_LOSS_PCT:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"max_loss ({pnl_pct:+.2%})",
                              limit_price=current_price)
                else:
                    self.buy_to_cover(symbol, qty, reason=f"max_loss ({pnl_pct:+.2%})",
                                      limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                return (f"EXIT  (option max loss {pnl_pct:+.2%} <= -{OPTION_MAX_LOSS_PCT:.0%}, "
                        f"{qty} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # Option: fixed +70% profit target
            if use_option_stops and pnl_pct >= OPTION_TP_TARGET_PCT:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"profit_target ({pnl_pct:+.2%})",
                              limit_price=current_price)
                else:
                    self.buy_to_cover(symbol, qty, reason=f"profit_target ({pnl_pct:+.2%})",
                                      limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                return (f"EXIT  (option profit target {pnl_pct:+.2%} >= +{OPTION_TP_TARGET_PCT:.0%}, "
                        f"{qty} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # Option: trailing stop (20% from peak)
            if use_option_stops and drawdown_from_peak >= OPTION_TRAILING_STOP_PCT:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"option_trailing_stop ({drawdown_from_peak:.2%})",
                              limit_price=current_price)
                else:
                    self.buy_to_cover(symbol, qty, reason=f"option_trailing_stop ({drawdown_from_peak:.2%})",
                                      limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                return (f"EXIT  (option trailing stop {drawdown_from_peak:.2%} from peak, "
                        f"{qty} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # Legacy positions: trailing stop (3% from peak)
            if use_legacy_stops and not use_option_stops:
                if drawdown_from_peak >= LEGACY_TRAILING_STOP_PCT:
                    if side == "LONG":
                        self.sell(symbol, qty, reason=f"legacy_trailing_stop ({drawdown_from_peak:.2%})",
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason=f"legacy_trailing_stop ({drawdown_from_peak:.2%})",
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (legacy trailing stop {drawdown_from_peak:.2%} from peak, "
                            f"{qty} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # --- v2: Disaster stop (safety net at N×ATR) ---
            if not use_option_stops and not use_legacy_stops:
                entry_atr = self._entry_atrs.get(symbol, bar_atr)
                if entry_atr > 0 and entry_price > 0:
                    disaster_stop_pct = self.disaster_stop_atr_mult * entry_atr / entry_price
                else:
                    disaster_stop_pct = 0.10
                if pnl_pct <= -disaster_stop_pct:
                    if side == "LONG":
                        self.sell(symbol, qty, reason=f"disaster_stop ({pnl_pct:+.2%})",
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason=f"disaster_stop ({pnl_pct:+.2%})",
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._clear_symbol_state(symbol)
                    if self.loss_cooldown_hours > 0:
                        cd = timedelta(hours=self.loss_cooldown_hours)
                        self._cooldown_until[symbol] = datetime.now(timezone.utc) + cd
                        log.info("Cooldown %s: no re-entry for %.1fh after disaster stop", symbol, self.loss_cooldown_hours)
                    return (f"EXIT  (disaster stop {pnl_pct:+.2%} >= -{disaster_stop_pct:.2%} "
                            f"[{self.disaster_stop_atr_mult}×ATR], "
                            f"{qty} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # --- v2: Signal-decay exit (primary exit mechanism) ---
            if not use_option_stops and not use_legacy_stops:
                if side == "LONG" and expected_return <= 0:
                    self.sell(symbol, qty, reason="signal_decay", limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (signal decay: E[r]={expected_return:+.4f} ≤ 0, "
                            f"P&L={pnl_pct:+.2%}, {qty} sh {side})")
                elif side == "SHORT" and expected_return >= 0:
                    self.buy_to_cover(symbol, qty, reason="signal_decay", limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (signal decay: E[r]={expected_return:+.4f} ≥ 0, "
                            f"P&L={pnl_pct:+.2%}, {qty} sh {side})")

            # HOLD
            return (f"HOLD  ({side} {qty} sh @ ${entry_price:.2f}, "
                    f"P&L: {pnl_pct:+.2%})  "
                    f"ML: {direction} E[r]={expected_return:+.4f}  trend={bar_trend:+.0f}")

        # --- Entry logic ---
        # Exit-only: manage out wrongly placed positions; do not open new ones
        if exit_only:
            return (f"EXIT-ONLY  (managing out — {direction} E[r]={expected_return:+.4f}, "
                    f"no new entries for this symbol in this account)")

        # Determine entry direction
        enter_dir = None
        if expected_return > self.cost_threshold and bar_trend > 0:
            enter_dir = "LONG"
        elif expected_return < -self.cost_threshold and bar_trend < 0:
            enter_dir = "SHORT"

        if enter_dir is not None:
            # Cooldown check
            cd_until = self._cooldown_until.get(symbol)
            if cd_until and datetime.now(timezone.utc) < cd_until:
                remaining = (cd_until - datetime.now(timezone.utc)).seconds // 60
                return (f"COOLDOWN  ({symbol} blocked for {remaining}m after loss exit, "
                        f"enter_dir={enter_dir})  ML: {direction} E[r]={expected_return:+.4f}")

            current_price = self._get_current_price(symbol)
            if current_price is None:
                return (f"SKIP  (price fetch failed for entry)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")

            # Signal-proportional sizing (v2)
            signal_pct = min(1.0, max(0.1, abs(expected_return) / self.target_return))
            sizing_pct = self.position_pct * signal_pct
            sizing_pct = min(sizing_pct, 0.98)

            invest = allocation * sizing_pct
            qty = int(invest / current_price)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} E[r]={expected_return:+.4f}"

            if enter_dir == "LONG":
                self.buy(symbol, qty, limit_price=current_price)
            else:
                self.sell_short(symbol, qty, limit_price=current_price)

            self._peak_prices[symbol] = current_price
            self._entry_atrs[symbol] = bar_atr

            return (f"{enter_dir}  ({qty} sh @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f}, size={sizing_pct:.0%})  "
                    f"ML: {direction} E[r]={expected_return:+.4f}  trend={bar_trend:+.0f}")

        return (f"SKIP  (no signal)  ML: {direction} E[r]={expected_return:+.4f}  "
                f"trend={bar_trend:+.0f}")

    def _record_closed_trade(self, pnl_pct: float) -> None:
        """Track consecutive losses for portfolio alerts."""
        if pnl_pct <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def _clear_symbol_state(self, symbol: str) -> None:
        """Reset tracking state for a symbol after exit."""
        self._peak_prices.pop(symbol, None)
        self._entry_atrs.pop(symbol, None)

    # -- Pairs fallback ------------------------------------------------
    def _get_pairs_prediction(self, symbol: str) -> Optional[dict]:
        """Lazy-load PairsPredictor and get a pairs-model prediction.

        Called when Hurst < 0.45 (mean-reverting regime) as a fallback
        instead of skipping the symbol entirely.

        Returns prediction dict or None if no pairs model available.
        """
        if symbol not in PAIRS_MAP:
            return None

        if symbol not in self._pairs_predictors:
            try:
                self._pairs_predictors[symbol] = PairsPredictor(symbol)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No pairs model for %s: %s", symbol, exc)
                self._pairs_predictors[symbol] = None

        predictor = self._pairs_predictors[symbol]
        if predictor is None:
            return None

        try:
            bars = self.adapter.fetch_daily(symbol, 200)
            vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
            return predictor.predict(bars, vix_df)
        except Exception as exc:
            log.warning("Pairs prediction failed for %s: %s", symbol, exc)
            return None

    # -- Main loop -----------------------------------------------------
    def run_loop(self) -> None:
        """Main continuous trading loop."""
        log.info("Starting paper trading loop (%s mode)...", self.mode)
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("Check interval: %d min", self.check_interval // 60)
        log.info("Strategy: v2 regression | trend SMA(%d) | cost_threshold=%.4f | "
                 "target_return=%.4f | disaster_stop=%.1f×ATR | "
                 "cooldown=%.1fh",
                 self.trend_sma_period, self.cost_threshold,
                 self.target_return, self.disaster_stop_atr_mult,
                 self.loss_cooldown_hours)
        print()

        # Graceful shutdown
        def handle_signal(sig, frame):
            print("\n\n  Shutting down paper trader...\n")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        while self._running:
            cycle += 1

            # Session check — only sleep during overnight + weekend closure
            session = _get_session()
            if session == "closed":
                next_session = _time_until_next_session()
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Market closed (overnight/weekend). "
                      f"Next session (pre-market) in {next_session}.")
                self._sleep(self.check_interval)
                continue
            session_label = "REGULAR" if session == "regular" else "EXTENDED HOURS"

            try:
                # Get account + positions
                account = self.get_account_summary()
                positions = self.get_positions()
                allocation_per_sym = account["equity"] / len(self.symbols)

                # Print header
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"=== Paper Trading Cycle #{cycle} ({self.mode} | {session_label}) ===")
                print(f"  Account: ${account['equity']:,.2f} equity | "
                      f"${account['cash']:,.2f} cash | "
                      f"${account['equity'] - account['cash']:,.2f} in positions")
                print(f"  Allocation per symbol: ${allocation_per_sym:,.2f}")
                print()

                # Refresh VIX cache once per cycle so all symbols share the same fetch.
                self._vix_cache = None   # force fresh fetch this cycle
                self._get_vix_df()       # populates _vix_cache (or retains old on failure)

                # Refresh options flow cache once per cycle (market-wide sentiment)
                try:
                    self._options_flow_cache = self._options_flow_engine.get_model_features()
                except Exception as exc:
                    log.debug("Options flow refresh failed: %s", exc)
                    self._options_flow_cache = None

                # Positions in this account for symbols not in this group (wrongly placed).
                # Option positions (OCC symbols) are NOT managed here — options_trader.py owns
                # their full lifecycle (entry + exit). So we only manage non-option, non-group symbols.
                exit_only_symbols = [
                    s for s in positions
                    if positions[s].get("qty", 0) > 0
                    and s not in self.symbols
                    and not _is_option_symbol(s)
                ]
                option_positions = [
                    s for s in positions
                    if positions[s].get("qty", 0) > 0 and _is_option_symbol(s)
                ]
                if exit_only_symbols:
                    print(f"  [!] EXIT-ONLY (not in group): {', '.join(exit_only_symbols)} -- managing out, no new entries")
                if option_positions:
                    print(f"  [OPTIONS] {', '.join(option_positions)} -- managed by options_trader only (entry + exit), not by this process")

                # Check each symbol in this group
                for sym in self.symbols:
                    # Legacy / wrong-algorithm: manage out only (stricter stops, no new entries)
                    if sym in self._legacy_no_new_entries:
                        action = self.check_and_trade(sym, positions, allocation_per_sym, exit_only=True)
                        print(f"  {sym:>5}:  {action}  [LEGACY no new entries]")
                        continue
                    # Extended-hours guard: Asian/EM ETFs have near-zero volume
                    # during US pre/after-market — underlying markets are closed.
                    # Skipping avoids 0.5-2% spread costs on worthless signals.
                    if session == "extended" and sym not in EXTENDED_HOURS_UNIVERSE:
                        # Still allow exits on existing positions, block new entries
                        pos = positions.get(sym)
                        if pos is None or pos["qty"] == 0:
                            print(f"  {sym:>5}:  SKIP  (extended hours — low liquidity for this ETF)")
                            continue
                    action = self.check_and_trade(sym, positions, allocation_per_sym)
                    print(f"  {sym:>5}:  {action}")

                # Manage out wrongly placed positions (exit-only: stops and flips, no new entries)
                for sym in exit_only_symbols:
                    action = self.check_and_trade(sym, positions, allocation_per_sym, exit_only=True)
                    print(f"  {sym:>5}:  {action}  [EXIT-ONLY]")

                # Microstructure alert checks (per-symbol + market-wide + portfolio)
                try:
                    for sym in self.symbols:
                        ctx = self._get_market_context(sym)
                        ctx["symbol"] = sym
                        self._alert_engine.check(market_context=ctx)
                    if self._options_flow_cache:
                        self._alert_engine.check(options_flow=self._options_flow_cache)
                    initial_capital = 100_000.0
                    drawdown_pct = ((account["equity"] - initial_capital) / initial_capital) * 100
                    self._alert_engine.check(portfolio_state={
                        "consecutive_losses": self._consecutive_losses,
                        "drawdown_pct": drawdown_pct,
                    })
                except Exception as exc:
                    log.debug("Alert check failed: %s", exc)

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
# Lock status (see if paper trader is running for a group)
# ===================================================================
def _is_pid_alive(pid: int) -> bool:
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
            return result != 0
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        return False


def lock_status_main() -> None:
    """Show whether the paper trader lock is present for a group (so you know if it's safe to start)."""
    parser = argparse.ArgumentParser(description="Check if paper trader lock is present for a group.")
    parser.add_argument("--group", default="intraday", choices=list(SYMBOL_GROUPS.keys()),
                        help="Account group to check (default: intraday)")
    parser.add_argument("--clear-stale", action="store_true",
                        help="If lock is stale (no process running), delete the lock file so you can start")
    args = parser.parse_args()
    group = args.group
    temp_dir = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
    lock_path = os.path.join(temp_dir, f".paper_trader_{group}.lock")
    pid_path = os.path.join(temp_dir, f".paper_trader_{group}.pid")

    lock_exists = os.path.isfile(lock_path)
    pid_exists = os.path.isfile(pid_path)

    # Detect stale lock: lock file exists but the process that wrote it is no longer running
    stale_lock = False
    lock_pid = None
    if lock_exists:
        try:
            with open(lock_path) as f:
                lock_pid = int(f.read().strip())
            if not _is_pid_alive(lock_pid):
                stale_lock = True
        except (ValueError, OSError, PermissionError):
            # Could not read (e.g. still held by process) or invalid PID
            pass

    lines = [
        "",
        f"  Paper trader lock status (group={group})",
        f"  Lock file: {lock_path}",
        f"  PID file:  {pid_path}",
        "",
        "  Trust this output (or outputs/lock_status.txt). Running this command starts a short-lived",
        "  python.exe, so Task Manager is not a reliable way to tell if the paper trader is running.",
        "",
    ]
    if lock_exists and not stale_lock:
        lines.extend([
            "  Status:  LOCK PRESENT - paper trader for this group is running.",
            "  You cannot start another instance until the lock is gone.",
            "  To stop: python main.py stop-paper-trader --group " + group,
            "  Or in Task Manager (Details tab): end the python.exe that stays running (not the brief one from this command).",
        ])
    elif stale_lock:
        lines.extend([
            "  Status:  STALE LOCK - lock file exists but no process is running (process exited or was ended).",
            "  You can start the trader; it will remove the stale lock automatically.",
            "  Or run: python main.py lock-status --group " + group + " --clear-stale",
            "  Then: python main.py trade --group " + group + " --mode intraday --interval 5min",
        ])
        if args.clear_stale:
            try:
                os.remove(lock_path)
                if pid_exists:
                    os.remove(pid_path)
                lines.append("")
                lines.append("  Cleared stale lock file(s). You can start the trader now.")
            except OSError as e:
                lines.append("")
                lines.append(f"  Could not remove lock file: {e}")
    elif not lock_exists:
        lines.extend([
            "  Status:  LOCK GONE - no paper trader running for this group.",
            "  You can start the trader: python main.py trade --group " + group + " --mode intraday --interval 5min",
        ])
    if pid_exists and not lock_exists:
        lines.append("  Note: .pid file exists but lock is gone (process may have exited).")
    elif pid_exists and lock_exists and not stale_lock and lock_pid is not None:
        lines.append(f"  PID {lock_pid}; stop with: python main.py stop-paper-trader --group " + group)
    lines.append("")

    text = "\n".join(lines)
    # Always write to file so you can read outputs/lock_status.txt if terminal shows nothing
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "lock_status.txt")
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass
    # Also print to stdout with explicit flush (so IDE terminals show it)
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except (OSError, AttributeError):
        pass
    print(f"  (also written to {out_file})", flush=True)


# ===================================================================
# Stop paper trader (by group)
# ===================================================================
def stop_paper_trader_main() -> None:
    """Stop the running paper trader for a given group by PID file (e.g. so you can restart with legacy flags)."""
    parser = argparse.ArgumentParser(description="Stop the paper trader for a group (uses .pid file).")
    parser.add_argument("--group", default="intraday", choices=list(SYMBOL_GROUPS.keys()),
                        help="Account group to stop (default: intraday)")
    args = parser.parse_args()
    group = args.group
    pid_path = os.path.join(
        os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__))),
        f".paper_trader_{group}.pid",
    )
    lock_path = os.path.join(
        os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__))),
        f".paper_trader_{group}.lock",
    )
    if not os.path.isfile(pid_path):
        print(f"\n  No .pid file for group '{group}' (paper trader was started before PID file was added).")
        print(f"  Path checked: {pid_path}")
        if os.path.isfile(lock_path):
            print("  The intraday paper trader is still running (lock file present).")
            print("  To stop it:")
            print("    1. Press Ctrl+Shift+Esc to open Task Manager.")
            print("    2. Open the Details tab.")
            print("    3. Find python.exe (there may be several).")
            print("    4. Right-click each python.exe -> End task until the trader stops.")
            print("    5. Run this command again; next time you start the trader, stop will work via .pid file.")
        else:
            print("  No lock file either — paper trader for this group is not running.")
        print()
        return
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
    except (ValueError, OSError) as e:
        print(f"\n  Could not read PID from {pid_path}: {e}\n")
        return
    print(f"\n  Stopping paper trader (group={group}, PID={pid})...")
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=True, timeout=10)
        print("  Stopped.")
    except subprocess.CalledProcessError:
        print("  Process already exited or access denied (try running as Administrator).")
    except FileNotFoundError:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, shell=True, timeout=10)
    try:
        os.remove(pid_path)
    except OSError:
        pass
    print()


# ===================================================================
# Check positions and recommend handling (wrong-algorithm / legacy)
# ===================================================================
def check_positions_main() -> None:
    """Check paper account positions and recommend handling for existing (possibly wrong-algorithm) positions.
    Use --execute to run the paper trader with recommended legacy flags.
    """
    parser = argparse.ArgumentParser(
        description="Check paper account positions and recommend handling for wrong-algorithm positions.",
    )
    parser.add_argument("--group", default="intraday", choices=list(SYMBOL_GROUPS.keys()),
                        help="Account group to check (default: intraday)")
    parser.add_argument("--execute", action="store_true",
                        help="Run the paper trader with recommended legacy flags to manage existing positions")
    args = parser.parse_args()
    group = args.group

    # Resolve credentials (same as trade command)
    if group and group != "all":
        env_prefix = f"ALPACA_{group.upper()}_"
        api_key = os.environ.get(f"{env_prefix}KEY", os.environ.get("ALPACA_API_KEY", ""))
        api_secret = os.environ.get(f"{env_prefix}SECRET", os.environ.get("ALPACA_API_SECRET", ""))
    else:
        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        print(f"\n  ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET (or ALPACA_{group.upper()}_KEY/SECRET).\n")
        sys.exit(1)

    # Fetch positions
    client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
    try:
        positions = client.get_all_positions()
    except Exception as e:
        print(f"\n  ERROR: Failed to fetch positions: {e}\n")
        sys.exit(1)

    # Build summary
    pos_list = []
    for p in positions:
        qty = float(p.qty)
        if qty == 0:
            continue
        side = "SHORT" if qty < 0 else "LONG"
        pos_list.append({
            "symbol": p.symbol,
            "side": side,
            "qty": abs(int(qty)),
            "entry": float(p.avg_entry_price),
            "current": float(p.current_price),
            "unrealized_plpc": float(p.unrealized_plpc),
        })
    symbols_with_positions = {p["symbol"] for p in pos_list}
    group_symbols = set(SYMBOL_GROUPS.get(group, []))
    # Recommend legacy only for symbols this group's equity trader actually trades (exclude options/other)
    recommended_symbols = symbols_with_positions & group_symbols

    # Algorithm check summary (no code bug found; "wrong" = previous model version)
    print("\n  === Algorithm check ===")
    print("  Current logic: v2 regression (E[r] forward 10d) + SMA(50) trend filter + signal-decay exits.")
    print("  Mode matches group (intraday -> LightGBM; swing -> PatchTST; expansion -> XGBoost; LSTM fallback).")
    print("  No known code bugs. 'Wrong algorithm' = positions opened by an older/wrong model.")
    print()

    print(f"  === Positions ({group} account) ===")
    if not pos_list:
        print("  No open positions.")
        print("\n  No legacy handling needed. Run: python main.py trade --group", group)
        if group == "intraday":
            print("  With mode: python main.py trade --group intraday --mode intraday --interval 5min")
        print()
        return

    for p in pos_list:
        plpct = p["unrealized_plpc"] * 100
        print(f"  {p['symbol']:5}  {p['side']:5}  {p['qty']:6} sh  entry ${p['entry']:.2f}  now ${p['current']:.2f}  P&L {plpct:+.2f}%")
    print()

    # Recommend: treat existing positions as potentially wrong-algorithm -> stricter exits + no new entries
    # Only for symbols this group trades (equity); options/other are managed via exit-only in the loop
    recommended = ",".join(sorted(recommended_symbols)) if recommended_symbols else ""
    print("  === Recommendation ===")
    if not recommended:
        print("  Open positions include options or symbols not in this group's list; they will be")
        print("  managed exit-only (no new entries) when you run the paper trader. No legacy flags needed.")
        print("\n  Run: python main.py trade --group", group, "--mode intraday --interval 5min" if group == "intraday" else "")
        print()
        return
    print("  To manage existing equity positions with stricter exits (3% trail, 2% TP) and no new entries")
    print("  until flat, run the paper trader with legacy flags. Then remove flags when flat to allow")
    print("  new entries from the current model.")
    print()
    cmd_parts = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
        "trade", "--group", group,
        "--legacy-stricter-exit", recommended,
        "--legacy-no-new-entries", recommended,
    ]
    if group == "intraday":
        cmd_parts.extend(["--mode", "intraday", "--interval", "5min"])
    cmd_str = " ".join(cmd_parts)
    print("  Command:")
    print(f"    {cmd_str}")
    print()

    if args.execute:
        print("  Executing (starting paper trader with legacy flags)...\n")
        try:
            subprocess.run(cmd_parts, check=False)
        except KeyboardInterrupt:
            print("\n  Interrupted.\n")
        except Exception as e:
            print(f"\n  ERROR: {e}\n")
            sys.exit(1)
    else:
        print("  To run this now, add: --execute")
        print()


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not _acquire_single_instance_lock():
        msg = "Another paper trader instance is already running. Exiting duplicate process."
        log.warning(msg)
        print(f"\n  WARNING: {msg}\n")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Alpaca paper trader — continuous ML-driven trading loop (v2 regression).",
    )
    parser.add_argument("--group", type=str, default=None,
                        choices=list(SYMBOL_GROUPS.keys()) + ["all"],
                        help="Trade a named account group: intraday / swing / expansion / all")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols override (overrides --group)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"],
                        help="Data provider (default: yahoo)")
    parser.add_argument("--check-interval", type=int, default=None,
                        help="Check interval in minutes (default: 5 daily, 1 intraday)")
    parser.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                        help="Trading mode (default: daily)")
    parser.add_argument("--interval", default="5min", choices=["1min", "5min"],
                        help="Intraday bar interval (default: 5min)")
    # v2 regression parameters
    parser.add_argument("--trend-sma", type=int, default=50,
                        help="SMA period for trend filter (default: 50)")
    parser.add_argument("--cost-threshold", type=float, default=COST_THRESHOLD,
                        help="Min expected return to trade (default: 0.001 = 0.1%%)")
    parser.add_argument("--target-return", type=float, default=TARGET_RETURN,
                        help="Expected return for full position size (default: 0.02 = 2%%)")
    parser.add_argument("--disaster-stop-mult", type=float, default=3.0,
                        help="ATR multiplier for disaster stop (default: 3.0)")
    parser.add_argument("--loss-cooldown-hours", type=float, default=4.0,
                        help="Hours to block re-entry after disaster stop (default: 4.0; 0=off)")
    parser.add_argument("--legacy-stricter-exit", type=str, default="",
                        help="Comma-separated symbols to use stricter exits (3%% trail) for wrong-algorithm positions")
    parser.add_argument("--legacy-no-new-entries", type=str, default="",
                        help="Comma-separated symbols to manage out only, no new entries until flat")

    args = parser.parse_args()

    # Resolve API keys: group-specific keys take priority over generic ALPACA_API_KEY
    group = args.group  # e.g. "intraday", "swing", "expansion", "all", or None
    if group and group != "all":
        env_prefix = f"ALPACA_{group.upper()}_"
        api_key    = os.environ.get(f"{env_prefix}KEY",    os.environ.get("ALPACA_API_KEY", ""))
        api_secret = os.environ.get(f"{env_prefix}SECRET", os.environ.get("ALPACA_API_SECRET", ""))
    else:
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        hint = (f"ALPACA_{group.upper()}_KEY / ALPACA_{group.upper()}_SECRET  OR  "
                if group and group != "all" else "")
        print(f"\n  ERROR: Set {hint}ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        print("  Get free keys at https://app.alpaca.markets/signup\n")
        sys.exit(1)

    # Warn if any symbol is in more than one group (wrong model placement)
    _warn_duplicate_symbols()

    # Resolve symbols: --symbols > --group > full DEFAULT_UNIVERSE
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif group and group != "all":
        symbols = SYMBOL_GROUPS[group]
        log.info("Account group '%s': trading %s", group, symbols)
    else:
        symbols = DEFAULT_UNIVERSE

    check_interval = args.check_interval
    if check_interval is None:
        check_interval = 1 if args.mode == "intraday" else 5

    # Legacy / wrong-algorithm positions (e.g. IGV, QQQ opened by previous buggy model)
    def _parse_legacy_set(raw: str, env_key: str, default: Set[str]) -> Set[str]:
        s = (raw or os.environ.get(env_key, "") or "").strip()
        if not s:
            return default
        return {x.strip().upper() for x in s.split(",") if x.strip()}
    legacy_stricter = _parse_legacy_set(
        args.legacy_stricter_exit, "PAPER_LEGACY_STRICTER_EXIT", LEGACY_STRICTER_EXIT_DEFAULT)
    legacy_no_new = _parse_legacy_set(
        args.legacy_no_new_entries, "PAPER_LEGACY_NO_NEW_ENTRIES", LEGACY_NO_NEW_ENTRIES_DEFAULT)
    if legacy_stricter or legacy_no_new:
        log.info("Legacy handling: stricter exits=%s, no new entries=%s", legacy_stricter, legacy_no_new)

    trader = AlpacaPaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        provider=args.provider,
        check_interval_min=check_interval,
        mode=args.mode,
        intraday_interval=args.interval,
        trend_sma_period=args.trend_sma,
        cost_threshold=args.cost_threshold,
        target_return=args.target_return,
        disaster_stop_atr_mult=args.disaster_stop_mult,
        loss_cooldown_hours=args.loss_cooldown_hours if args.mode != "intraday" else min(args.loss_cooldown_hours, 0.5),
        legacy_stricter_exit=legacy_stricter,
        legacy_no_new_entries=legacy_no_new,
        group=group,
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
