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
import collections
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
    compute_atr, compute_adx_full, compute_hurst_exponent,
)
from ml_model import Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR, META_THRESHOLD

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
    # IGV re-added: intraday backtest ~3mo showed 57.7% win rate, Sharpe 1.02, -0.55% max DD
    "intraday":  ["SPY", "QQQ", "IWM", "SOXX", "IGV"],
    # Account 2 — Swing (daily LSTM, broader macro ETFs)
    # TLT dropped: 42.9% win rate, Sharpe 0.300 across all feature sets (bonds need rate signals)
    "swing":     ["EWT", "GLD", "EEM", "SLV"],
    # Account 3 — Expansion (daily LSTM, international/sector)
    # IGV dropped: <50% win rate on both 12-feat and 17-feat (sector ETF, earnings-driven)
    # FXI dropped: <50% win rate on both (China policy-driven, not technical)
    "expansion": ["EWJ", "EWS", "XLE", "INDA"],
}

# Dynamic Kelly constants
KELLY_WINDOW       = 60   # rolling window: last N closed trades to estimate W and B
MIN_KELLY_TRADES   = 20   # minimum closed trades before switching from static to dynamic sizing

# Legacy / wrong-algorithm positions: opened by a previous buggy model. Manage them
# with stricter exits (lock profit or cut loss sooner) and optionally no new entries.
# Set via env: PAPER_LEGACY_STRICTER_EXIT=IGV,QQQ  PAPER_LEGACY_NO_NEW_ENTRIES=IGV,QQQ
LEGACY_STRICTER_EXIT_DEFAULT: Set[str] = set()   # e.g. {"IGV", "QQQ"} to use 3% trail, 2% TP
LEGACY_NO_NEW_ENTRIES_DEFAULT: Set[str] = set()  # e.g. {"IGV", "QQQ"} to manage out only
# Tighter params when symbol is in legacy_stricter_exit (or exit_only)
LEGACY_TRAILING_STOP_PCT = 0.03   # 3% from peak (vs default 5%)
LEGACY_TP_ACTIVATION     = 0.02   # lock profit after +2% (vs default 4%)
LEGACY_TP_TRAIL           = 0.01   # 1% trail after activation (vs default 2%)

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
        max_loss_atr_mult: float = 1.0,       # Hard cut: exit if loss > N×ATR from entry (0 = off)
        # --- Book-derived filters ---
        use_hurst_filter: bool = True,          # Chan: skip entries when H < hurst_threshold
        hurst_threshold: float = 0.55,          # H > 0.55 = confirmed trending regime
        kelly_fraction: float = 0.5,            # Chan: base half-Kelly multiplier (0 = off, 0.5 = half)
        use_dynamic_kelly: bool = True,         # Re-estimate Kelly fraction from recent 60 closed trades
        use_vix_halt: bool = True,              # Aldridge: halt new entries on VIX spike > 2σ
        vix_halt_sigma: float = 2.0,            # Aldridge: # of σ for VIX halt
        use_time_filter: bool = True,           # Aldridge: block opening/closing auction windows (intraday only)
        use_corr_sizing: bool = True,           # Scale down size when multiple same-direction positions open
        loss_cooldown_hours: float = 4.0,       # Hours to block re-entry after max_loss or regime_exit (0=off)
        legacy_stricter_exit: Optional[Set[str]] = None,   # Stricter stops (3% trail, 2% TP) for wrong-algorithm positions
        legacy_no_new_entries: Optional[Set[str]] = None, # Exit-only for these until flat (no new size)
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
        self.max_loss_atr_mult = max_loss_atr_mult
        self.use_hurst_filter = use_hurst_filter
        self.hurst_threshold  = hurst_threshold
        self.kelly_fraction    = kelly_fraction
        self.use_dynamic_kelly = use_dynamic_kelly
        self.use_vix_halt      = use_vix_halt
        self.vix_halt_sigma   = vix_halt_sigma
        self._vix_halted      = False   # set each cycle by run_loop
        self.use_time_filter  = use_time_filter
        self.use_corr_sizing  = use_corr_sizing

        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")

        self.loss_cooldown_hours = loss_cooldown_hours
        self._legacy_stricter_exit = legacy_stricter_exit or set()
        self._legacy_no_new_entries = legacy_no_new_entries or set()

        # Per-symbol tracking
        self._peak_prices: Dict[str, float] = {}
        self._entry_atrs: Dict[str, float] = {}
        self._tp_activated: Dict[str, bool] = {}
        self._tp_trail_peaks: Dict[str, float] = {}
        self._cooldown_until: Dict[str, datetime] = {}  # symbol → earliest re-entry time

        # Dynamic Kelly: rolling closed-trade P&L history (cross-symbol, shared)
        self._closed_trades: collections.deque = collections.deque(maxlen=KELLY_WINDOW)

        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            try:
                self.predictors[sym] = Predictor(
                    sym, model_dir=model_dir,
                    mode=mode, intraday_interval=intraday_interval)
                log.info("Loaded ML model for %s (%s).", sym, mode)
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning("No trained model for %s (%s): %s — will skip ML signals.", sym, mode, exc)
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
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=400)
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

    def _get_hurst(self, symbol: str) -> float:
        """Compute Hurst exponent from ~100 daily bars.

        H > 0.55 → trending (momentum strategy valid).
        H < 0.45 → mean-reverting (momentum signals unreliable).
        Per Ernest Chan, *Algorithmic Trading*, Chapter 2.
        """
        try:
            bars = self.adapter.fetch_daily(symbol, 100)
            if len(bars) < 30:
                return 0.5
            return compute_hurst_exponent(bars["close"].astype(float))
        except Exception as exc:
            log.warning("Hurst fetch failed for %s: %s", symbol, exc)
            return 0.5

    def _check_vix_halt(self) -> bool:
        """Return True if today's VIX move is anomalously large (> vix_halt_sigma × σ).

        When True, halt ALL new entries for this cycle (exits still allowed).
        Per Irene Aldridge, *High-Frequency Trading* — cease new entries during
        structural regime changes flagged by abnormal VIX spikes.
        """
        try:
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=60)
            if len(vix_df) < 22:
                return False
            vix_chg = vix_df["vix"].pct_change().dropna()
            rolling_std = float(vix_chg.rolling(20).std().iloc[-1])
            today_chg   = float(vix_chg.iloc[-1])
            if rolling_std > 0 and abs(today_chg) > self.vix_halt_sigma * rolling_std:
                log.warning(
                    "VIX HALT: today_chg=%.2f%% > %.1fσ (σ=%.2f%%) — no new entries this cycle.",
                    today_chg * 100, self.vix_halt_sigma, rolling_std * 100,
                )
                return True
            return False
        except Exception as exc:
            log.warning("VIX halt check failed: %s", exc)
            return False

    def _intraday_time_window(self) -> str:
        """Return the current intraday time-of-day window (ET).

        Aldridge, High-Frequency Trading Ch. 5-7:
          open_noise   9:30-10:00  — opening auction, wide spreads, fake momentum → block entries
          momentum_1  10:00-11:30  — primary momentum window → full size
          midday      11:30-14:00  — low volume, mean-reverting → half size
          momentum_2  14:00-15:30  — secondary momentum window → full size
          close_noise 15:30-16:00  — closing auction risk → block entries, exits only

        Only meaningful in intraday mode.
        """
        from datetime import time as dt_time
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        t = datetime.now(ZoneInfo("America/New_York")).time()
        if   dt_time(9,  30) <= t < dt_time(10,  0): return "open_noise"
        elif dt_time(10,  0) <= t < dt_time(11, 30): return "momentum_1"
        elif dt_time(11, 30) <= t < dt_time(14,  0): return "midday"
        elif dt_time(14,  0) <= t < dt_time(15, 30): return "momentum_2"
        elif dt_time(15, 30) <= t <= dt_time(16, 0): return "close_noise"
        return "n/a"

    def _get_market_context(self, symbol: str) -> Dict[str, float]:
        """Fetch ATR, ADX, +DI, -DI, and realized vol for position sizing, stops, and DI filter."""
        try:
            bars = self.adapter.fetch_daily(symbol, 60)
            if len(bars) < 30:
                return {"atr": 0.0, "adx": 25.0, "plus_di": 25.0, "minus_di": 25.0, "vol20": 0.15}
            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            atr_s = compute_atr(high, low, close, period=14)
            adx_s, plus_di_s, minus_di_s = compute_adx_full(high, low, close, period=14)
            vol_s = close.pct_change().rolling(20).std() * np.sqrt(252)

            def _safe(s: "pd.Series", default: float) -> float:
                v = float(s.iloc[-1])
                return default if np.isnan(v) else v

            return {
                "atr":      _safe(atr_s,      0.0),
                "adx":      _safe(adx_s,      25.0),
                "plus_di":  _safe(plus_di_s,  25.0),
                "minus_di": _safe(minus_di_s, 25.0),
                "vol20":    _safe(vol_s,       0.15),
            }
        except Exception as exc:
            log.warning("Market context fetch failed for %s: %s", symbol, exc)
            return {"atr": 0.0, "adx": 25.0, "plus_di": 25.0, "minus_di": 25.0, "vol20": 0.15}

    # -- Trading logic (one symbol) ------------------------------------
    def check_and_trade(self, symbol: str, positions: Dict[str, dict],
                        allocation: float, exit_only: bool = False) -> str:
        """Check ML signal and manage position for one symbol.

        Optimized strategy with ATR-based stops, vol sizing, regime filter,
        and trailing take-profit.

        When exit_only=True (symbol not in this group's list), only exit logic
        runs: trailing stop, take-profit, signal flip. No new entries. Use this
        to manage out positions that were opened under a previous/wrong model.

        Returns action string for display.
        """
        pred = self.get_prediction(symbol)
        direction = pred["direction"]
        confidence = pred["confidence"]
        meta_confidence = pred.get("meta_confidence", 1.0)
        meta_tradeable  = pred.get("tradeable", True)

        ctx = self._get_market_context(symbol)
        bar_atr      = ctx["atr"]
        bar_adx      = ctx["adx"]
        bar_plus_di  = ctx["plus_di"]
        bar_minus_di = ctx["minus_di"]
        bar_vol      = ctx["vol20"]
        # DI-implied direction: True when +DI > -DI (uptrend dominant)
        di_bullish = bar_plus_di > bar_minus_di

        pos = positions.get(symbol)
        has_position = pos is not None and pos["qty"] > 0
        flip_direction = None

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

            # Legacy / wrong-algorithm positions: use tighter stops to lock profit or cut loss sooner
            use_legacy_stops = exit_only or (symbol in self._legacy_stricter_exit)
            use_option_stops = _is_option_symbol(symbol)

            # Initialize peak from entry_price when we first see a position so inherited/down positions
            # trigger the trailing stop immediately (e.g. short entered at 80, now 84 -> peak=80, drawdown=5%).
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

            # Option positions: fixed -35% max loss (data-driven from backtest)
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
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Hard max-loss cut: exit if loss exceeds N×ATR from entry price (ATR-adaptive floor)
            entry_atr_for_floor = self._entry_atrs.get(symbol, bar_atr)
            if (not use_option_stops
                    and self.max_loss_atr_mult > 0 and entry_atr_for_floor > 0
                    and entry_price > 0
                    and pnl_pct <= -(self.max_loss_atr_mult * entry_atr_for_floor / entry_price)):
                floor_pct = self.max_loss_atr_mult * entry_atr_for_floor / entry_price
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"max_loss ({pnl_pct:+.2%})",
                              limit_price=current_price)
                else:
                    self.buy_to_cover(symbol, qty, reason=f"max_loss ({pnl_pct:+.2%})",
                                      limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                if self.loss_cooldown_hours > 0:
                    cd = timedelta(hours=self.loss_cooldown_hours)
                    self._cooldown_until[symbol] = datetime.now(timezone.utc) + cd
                    log.info("Cooldown %s: no re-entry for %.1fh after max_loss exit", symbol, self.loss_cooldown_hours)
                return (f"EXIT  (max loss {pnl_pct:+.2%} >= -{floor_pct:.2%} floor [{self.max_loss_atr_mult}×ATR], "
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Regime breakdown exit: trend that justified entry has collapsed (equity only; skip for options)
            # ADX < 15 = no trend at all; Hurst < 0.50 = mean-reverting/random
            regime_exit_reason: str | None = None
            if not use_option_stops and self.use_regime_filter and bar_adx < 15.0:
                regime_exit_reason = f"ADX={bar_adx:.1f}<15 (no trend)"
            if regime_exit_reason is None and not use_option_stops and self.use_hurst_filter:
                hurst = self._get_hurst(symbol)
                if hurst < 0.50:
                    label = "mean-reverting" if hurst < 0.45 else "random-walk"
                    regime_exit_reason = f"H={hurst:.2f}<0.50 ({label})"
            if regime_exit_reason:
                if side == "LONG":
                    self.sell(symbol, qty, reason="regime_exit", limit_price=current_price)
                else:
                    self.buy_to_cover(symbol, qty, reason="regime_exit", limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                if self.loss_cooldown_hours > 0:
                    cd = timedelta(hours=self.loss_cooldown_hours)
                    self._cooldown_until[symbol] = datetime.now(timezone.utc) + cd
                    log.info("Cooldown %s: no re-entry for %.1fh after regime exit", symbol, self.loss_cooldown_hours)
                return (f"EXIT  (regime breakdown: {regime_exit_reason}, P&L={pnl_pct:+.2%}, "
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # ATR-based or fixed stop distance (option vs legacy vs normal)
            entry_atr = self._entry_atrs.get(symbol, bar_atr)
            if use_option_stops:
                stop_distance = OPTION_TRAILING_STOP_PCT
                tp_activation, tp_trail = OPTION_TP_ACTIVATION, OPTION_TP_TRAIL
            elif use_legacy_stops:
                stop_distance = LEGACY_TRAILING_STOP_PCT
                tp_activation, tp_trail = LEGACY_TP_ACTIVATION, LEGACY_TP_TRAIL
            else:
                if self.use_atr_stop and entry_atr > 0 and entry_price > 0:
                    stop_distance = self.atr_stop_mult * entry_atr / entry_price
                else:
                    stop_distance = self.trailing_stop_pct
                tp_activation, tp_trail = self.trailing_tp_activation, self.trailing_tp_trail

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
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Trailing take-profit: lock in gains once threshold is reached
            if self.use_trailing_tp and pnl_pct >= tp_activation:
                if not self._tp_activated.get(symbol, False):
                    self._tp_activated[symbol] = True
                    self._tp_trail_peaks[symbol] = pnl_pct
                    log.info("Trailing TP activated for %s at %+.2f%%", symbol, pnl_pct * 100)
                self._tp_trail_peaks[symbol] = max(
                    self._tp_trail_peaks.get(symbol, pnl_pct), pnl_pct)
                if pnl_pct < self._tp_trail_peaks[symbol] - tp_trail:
                    if side == "LONG":
                        self.sell(symbol, qty, reason=f"trailing_tp ({pnl_pct:+.2%})",
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason=f"trailing_tp ({pnl_pct:+.2%})",
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (trailing TP {pnl_pct:+.2%}, peak {self._tp_trail_peaks.get(symbol, 0):+.2%}, "
                            f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Trailing stop
            if drawdown_from_peak >= stop_distance:
                if side == "LONG":
                    self.sell(symbol, qty, reason=f"trailing_stop ({drawdown_from_peak:.2%})",
                              limit_price=current_price)
                else:
                    self.buy_to_cover(symbol, qty, reason=f"trailing_stop ({drawdown_from_peak:.2%})",
                                      limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                return (f"EXIT  (trailing stop {drawdown_from_peak:.2%} from peak, "
                        f"{qty} sh {side})  ML: {direction} {confidence:.2f}")

            # Signal flip
            if (side == "LONG" and direction == "DOWN"
                    and confidence >= self.exit_confidence):
                self.sell(symbol, qty, reason="signal_flip", limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                flip_direction = "SHORT"
            elif (side == "SHORT" and direction == "UP"
                  and confidence >= self.exit_confidence):
                self.buy_to_cover(symbol, qty, reason="signal_flip", limit_price=current_price)
                self._record_closed_trade(pnl_pct)
                self._clear_symbol_state(symbol)
                flip_direction = "LONG"
            else:
                if use_option_stops:
                    stop_info = f"{OPTION_TRAILING_STOP_PCT:.0%} trail"
                else:
                    stop_info = f"ATR×{self.atr_stop_mult}" if self.use_atr_stop else f"{self.trailing_stop_pct:.0%}"
                return (f"HOLD  ({side} {qty} sh @ ${entry_price:.2f}, "
                        f"P&L: {pnl_pct:+.2%}, stop={stop_info})  "
                        f"ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}")

        # --- Entry logic ---
        # Exit-only: manage out wrongly placed positions; do not open new ones
        if exit_only:
            return (f"EXIT-ONLY  (managing out — {direction} {confidence:.2f}, "
                    f"no new entries for this symbol in this account)")
        # Aldridge: intraday time-of-day filter — block opening/closing auction windows
        _midday_sizing_mult = 1.0
        if self.use_time_filter and self.mode == "intraday" and not flip_direction:
            _tw = self._intraday_time_window()
            if _tw == "open_noise":
                return (f"TIME BLOCK  (9:30-10:00 ET opening noise — wide spreads, false momentum)  "
                        f"ML: {direction} {confidence:.2f}")
            if _tw == "close_noise":
                return (f"TIME BLOCK  (15:30-16:00 ET closing risk — exits only)  "
                        f"ML: {direction} {confidence:.2f}")
            if _tw == "midday":
                _midday_sizing_mult = 0.5   # low volume doldrums: half size

        # Aldridge: VIX halt — block ALL new entries on extreme VIX spikes
        if self._vix_halted and not flip_direction:
            return (f"SKIP  (VIX halt active — no new entries this cycle)  "
                    f"ML: {direction} {confidence:.2f}")

        # Regime filter: skip in trendless markets (ADX)
        if self.use_regime_filter and bar_adx < self.adx_threshold:
            if flip_direction:
                return (f"EXIT  (signal flip, no re-entry: ADX={bar_adx:.0f}<{self.adx_threshold:.0f})  "
                        f"ML: {direction} {confidence:.2f}")
            return f"SKIP  (low trend ADX={bar_adx:.0f}<{self.adx_threshold:.0f})  ML: {direction} {confidence:.2f}"

        # Chan: Hurst regime filter — skip momentum entries in mean-reverting markets
        if self.use_hurst_filter and not flip_direction:
            hurst = self._get_hurst(symbol)
            if hurst < self.hurst_threshold:
                regime = "mean-reverting" if hurst < 0.45 else "random-walk"
                return (f"SKIP  (H={hurst:.2f}<{self.hurst_threshold} {regime}, "
                        f"momentum invalid)  ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}")

        # De Prado: meta-labeling gate — only enter when meta RF trusts the primary signal
        if not meta_tradeable and not flip_direction:
            return (f"META BLOCK  (meta_conf={meta_confidence:.2f}<{META_THRESHOLD:.2f}, "
                    f"primary signal not trusted)  ML: {direction} {confidence:.2f}")

        enter_dir = None
        if flip_direction is not None and confidence >= self.exit_confidence:
            enter_dir = flip_direction
        elif direction == "UP" and confidence >= self.long_confidence_threshold:
            if not di_bullish:
                return (f"SKIP  (ML=UP but -DI={bar_minus_di:.0f} > +DI={bar_plus_di:.0f}, "
                        f"DI conflict)  ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}")
            enter_dir = "LONG"
        elif direction == "DOWN" and confidence >= self.short_confidence_threshold:
            if di_bullish:
                return (f"SKIP  (ML=DOWN but +DI={bar_plus_di:.0f} > -DI={bar_minus_di:.0f}, "
                        f"DI conflict)  ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}")
            enter_dir = "SHORT"

        if enter_dir is not None:
            # Cooldown check: don't re-enter after a max_loss or regime_exit
            cd_until = self._cooldown_until.get(symbol)
            if cd_until and datetime.now(timezone.utc) < cd_until:
                remaining = (cd_until - datetime.now(timezone.utc)).seconds // 60
                return (f"COOLDOWN  ({symbol} blocked for {remaining}m after loss exit, "
                        f"enter_dir={enter_dir})  ML: {direction} {confidence:.2f}")

            current_price = self._get_current_price(symbol)
            if current_price is None:
                action = "flip" if flip_direction else "entry"
                return (f"SKIP  (price fetch failed for {action})  "
                        f"ML: {direction} {confidence:.2f}")

            # Chan: volatility-adjusted + confidence-scaled sizing with half-Kelly multiplier
            sizing_pct = self.position_pct
            if self.use_vol_sizing and bar_vol > 0:
                vol_scalar = min(2.0, max(0.3, self.target_vol / bar_vol))
                sizing_pct *= vol_scalar
            if self.use_confidence_sizing:
                conf_scalar = 0.5 + confidence
                sizing_pct *= conf_scalar
            dyn_kelly = self._compute_dynamic_kelly()
            if dyn_kelly > 0:
                sizing_pct *= dyn_kelly   # dynamic half-Kelly (falls back to static if < MIN_KELLY_TRADES)
            # Aldridge: midday doldrums — reduce size during 11:30-14:00 ET
            if _midday_sizing_mult < 1.0:
                sizing_pct *= _midday_sizing_mult
            # Correlated position sizing: 1/sqrt(N) when opening additional same-direction positions
            if self.use_corr_sizing:
                n_same = sum(1 for p in positions.values() if p["side"] == enter_dir) + 1
                if n_same > 1:
                    corr_scalar = 1.0 / np.sqrt(n_same)
                    sizing_pct *= corr_scalar
                    log.debug("Corr sizing: %.2fx (%d same-direction open)", corr_scalar, n_same)
            sizing_pct = min(sizing_pct, 0.98)

            invest = allocation * sizing_pct
            qty = int(invest / current_price)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} {confidence:.2f}"

            if enter_dir == "LONG":
                self.buy(symbol, qty, limit_price=current_price)
                verb = "FLIP->BUY" if flip_direction else "BUY"
            else:
                self.sell_short(symbol, qty, limit_price=current_price)
                verb = "FLIP->SHORT" if flip_direction else "SHORT"

            self._peak_prices[symbol] = current_price
            self._entry_atrs[symbol] = bar_atr
            self._tp_activated[symbol] = False
            self._tp_trail_peaks[symbol] = 0.0

            return (f"{verb}  ({qty} sh @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f}, size={sizing_pct:.0%})  "
                    f"ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}  "
                    f"+DI={bar_plus_di:.0f}/-DI={bar_minus_di:.0f}")

        if flip_direction:
            return (f"EXIT  (signal flip, no re-entry)  "
                    f"ML: {direction} {confidence:.2f}")

        return f"SKIP  (no signal)  ML: {direction} {confidence:.2f}  ADX={bar_adx:.0f}"

    def _record_closed_trade(self, pnl_pct: float) -> None:
        """Push realized P&L fraction into the rolling Kelly window."""
        self._closed_trades.append(pnl_pct)

    def _compute_dynamic_kelly(self) -> float:
        """Return the dynamic half-Kelly fraction estimated from recent closed trades.

        Formula (Chan, *Machine Trading*, ch. 6):
            full-Kelly f* = (W*(B+1) - 1) / B
            half-Kelly   = f* * 0.5
        where W = win rate, B = avg_win / avg_loss over the last KELLY_WINDOW trades.

        Falls back to the static ``self.kelly_fraction`` when:
        - dynamic Kelly is disabled (use_dynamic_kelly=False)
        - fewer than MIN_KELLY_TRADES are recorded
        - computed f* is non-positive (no edge detected)

        The result is clamped to [kelly_fraction*0.5, kelly_fraction*2.0] so it never
        deviate more than 2x from the configured base fraction.
        """
        if self.kelly_fraction <= 0 or not self.use_dynamic_kelly:
            return self.kelly_fraction

        trades = list(self._closed_trades)
        if len(trades) < MIN_KELLY_TRADES:
            return self.kelly_fraction  # not enough history yet

        wins   = [p for p in trades if p > 0]
        losses = [abs(p) for p in trades if p <= 0]

        if not losses:
            # Never lost in the window — cap at 2x base
            return self.kelly_fraction * 2.0
        if not wins:
            return 0.0  # no wins in window — skip new trades

        W = len(wins) / len(trades)
        B = np.mean(wins) / np.mean(losses)  # profit ratio

        kelly_full = (W * (B + 1) - 1) / B
        if kelly_full <= 0:
            return 0.0  # negative edge — do not trade

        half_kelly = kelly_full * 0.5
        kelly_min  = self.kelly_fraction * 0.5
        kelly_max  = self.kelly_fraction * 2.0
        result = float(np.clip(half_kelly, kelly_min, kelly_max))
        log.debug(
            "Dynamic Kelly: W=%.2f B=%.2f f*=%.3f half=%.3f clamped=%.3f (n=%d trades)",
            W, B, kelly_full, half_kelly, result, len(trades),
        )
        return result

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
        kelly_desc = (f"Dynamic (base={self.kelly_fraction}, window={KELLY_WINDOW}, min={MIN_KELLY_TRADES})"
                      if self.use_dynamic_kelly else f"Static {self.kelly_fraction}")
        log.info("Stops: %s + MaxLoss=%.1f×ATR + RegimeExit(ADX<15,H<0.50) | "
                 "ADX filter: %s (>%.0f) | Vol sizing: %s (target=%.0f%%) | "
                 "Conf sizing: %s | Trailing TP: %s | Kelly: %s | "
                 "Hurst filter: %s (>%.2f) | VIX halt: %s (>%.1fσ) | "
                 "Time filter: %s | Corr sizing: %s",
                 f"ATR×{self.atr_stop_mult}" if self.use_atr_stop else f"Fixed {self.trailing_stop_pct:.0%}",
                 self.max_loss_atr_mult,
                 "ON" if self.use_regime_filter else "OFF", self.adx_threshold,
                 "ON" if self.use_vol_sizing else "OFF", self.target_vol * 100,
                 "ON" if self.use_confidence_sizing else "OFF",
                 "ON" if self.use_trailing_tp else "OFF",
                 kelly_desc,
                 "ON" if self.use_hurst_filter else "OFF", self.hurst_threshold,
                 "ON" if self.use_vix_halt else "OFF", self.vix_halt_sigma,
                 "ON (intraday)" if self.use_time_filter else "OFF",
                 "ON (1/√N)" if self.use_corr_sizing else "OFF")
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

                # Aldridge: compute VIX halt once per cycle (not per symbol)
                if self.use_vix_halt:
                    self._vix_halted = self._check_vix_halt()
                    if self._vix_halted:
                        print(f"  [!] VIX HALT ACTIVE -- exits allowed, no new entries this cycle")

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
    print("  Current logic: LSTM direction + meta gate, ATR/ADX stops, trailing TP, DI filter.")
    print("  Mode matches group (intraday -> 5min LSTM; daily -> daily LSTM).")
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
        description="Alpaca paper trader — continuous ML-driven trading loop.",
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
    parser.add_argument("--confidence", type=float, default=0.01,
                        help="Min ML confidence to enter LONG (default: 0.01; attention-LSTM "
                             "outputs compressed probs, ~0.01-0.12 range — meta gate filters quality)")
    parser.add_argument("--short-confidence", type=float, default=0.01,
                        help="Min ML confidence to enter SHORT (default: 0.01)")
    parser.add_argument("--exit-confidence", type=float, default=0.005,
                        help="Min ML confidence to exit/flip (default: 0.005)")
    parser.add_argument("--trailing-stop", type=float, default=0.05,
                        help="Trailing stop from peak (default: 0.05 = 5%%)")
    parser.add_argument("--max-loss-atr", type=float, default=1.0,
                        help="Hard max-loss cut: exit if loss > N×ATR from entry (default: 1.0; 0 = off)")
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
    # Book-derived filters
    parser.add_argument("--no-hurst-filter", action="store_true",
                        help="Disable Hurst regime filter (Chan)")
    parser.add_argument("--hurst-threshold", type=float, default=0.55,
                        help="Min Hurst exponent to allow momentum entries (default: 0.55)")
    parser.add_argument("--kelly-fraction", type=float, default=0.5,
                        help="Base half-Kelly multiplier: 0.5=half-Kelly, 0=off (Chan, default: 0.5)")
    parser.add_argument("--no-dynamic-kelly", action="store_true",
                        help="Use static kelly-fraction instead of rolling Kelly estimate")
    parser.add_argument("--no-vix-halt", action="store_true",
                        help="Disable VIX spike halt rule (Aldridge)")
    parser.add_argument("--vix-halt-sigma", type=float, default=2.0,
                        help="VIX change σ threshold for halt (Aldridge, default: 2.0)")
    parser.add_argument("--no-time-filter", action="store_true",
                        help="Disable intraday time-of-day filter (Aldridge: blocks 9:30-10:00 and 15:30-16:00 ET)")
    parser.add_argument("--no-corr-sizing", action="store_true",
                        help="Disable correlated position sizing (1/sqrt(N) when multiple same-direction positions)")
    parser.add_argument("--loss-cooldown-hours", type=float, default=4.0,
                        help="Hours to block re-entry after max_loss or regime_exit (default: 4.0; 0=off)")
    parser.add_argument("--legacy-stricter-exit", type=str, default="",
                        help="Comma-separated symbols to use stricter exits (3%% trail, 2%% TP) for wrong-algorithm positions (e.g. IGV,QQQ)")
    parser.add_argument("--legacy-no-new-entries", type=str, default="",
                        help="Comma-separated symbols to manage out only, no new entries until flat (e.g. IGV,QQQ)")

    args = parser.parse_args()

    # Resolve API keys: group-specific keys take priority over generic ALPACA_API_KEY
    group = args.group  # e.g. "equities", "commodities", "intl", "all", or None
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
        confidence_threshold=args.confidence,
        short_confidence_threshold=args.short_confidence,
        exit_confidence=args.exit_confidence,
        trailing_stop_pct=args.trailing_stop,
        max_loss_atr_mult=args.max_loss_atr,
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
        use_hurst_filter=not args.no_hurst_filter,
        hurst_threshold=args.hurst_threshold,
        kelly_fraction=args.kelly_fraction,
        use_dynamic_kelly=not args.no_dynamic_kelly,
        use_vix_halt=not args.no_vix_halt,
        vix_halt_sigma=args.vix_halt_sigma,
        use_time_filter=not args.no_time_filter,
        use_corr_sizing=not args.no_corr_sizing,
        loss_cooldown_hours=args.loss_cooldown_hours if args.mode != "intraday" else min(args.loss_cooldown_hours, 0.5),
        legacy_stricter_exit=legacy_stricter,
        legacy_no_new_entries=legacy_no_new,
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
