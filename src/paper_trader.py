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
    python main.py trade --group crypto

Required env vars:
    ALPACA_API_KEY, ALPACA_API_SECRET, FRED_API_KEY

PAPER TRADING ONLY — paper=True is hardcoded for safety.

Crypto group trades 24/7 via Alpaca crypto endpoints (BTC/USD, ETH/USD, SOL/USD).
Uses BTC tiered SMA(50)/SMA(100) + momentum as regime filter. Fractional qty supported.

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
from ml_model import Predictor  # LSTM fallback (deprecated)
from utils import _fetch_vix_for_training, DEFAULT_MODEL_DIR, CRYPTO_MODEL_DIR, COST_THRESHOLD, TARGET_RETURN
from alerts import AlertEngine

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
    # Account 1 — Intraday LightGBM v2 (first-30m momentum, EOD exit ~15:30)
    # Selection rule: >60% in-sample (2024-2026) AND positive OOS (2022-2023)
    #                 AND regime cooldown fired <5 times OOS
    # In-sample: SMH +369%, IWM +198%, IGV +165%, SOXX +101%, QQQ +83%, EWT +137%
    # OOS 2022-2023: SMH +30.5%, IGV +34.7%, IWM +28.7%, QQQ +20.9%, SPY+15.9%, EWT +12.4%, SOXX +11.5%
    # Dropped (failed OOS or regime-cooldown >16×): XLE -11%, USO -10.5%, SLV +1%,
    #          GDX +3.3%, EEM -0.5%, SPY (cooldown >16×), XLK (cooldown >16×)
    "intraday":  ["SMH", "IWM", "IGV", "QQQ", "EWT", "SOXX"],
    # Account 2 — Swing XGBoost (daily regression, selective signal-decay exits)
    # Selection rule: swing backtest 2024-2026 total return > 60%
    # GDX +188%, SLV +118%, IGV +100%, QQQ +91%, GLD +88%, SMH +88%, XLK +73%, IBIT +64%
    # Dropped: MCHI (OOS 2022-2023: -7.2%, TFT discarded at 41.5% dir_acc)
    # Note: GDX/SLV/IGV/QQQ/SMH/XLK shared with intraday (separate account, separate model)
    "swing":     ["GDX", "SLV", "IGV", "QQQ", "GLD", "SMH", "XLK", "IBIT"],
    # Account 3 — Crypto: individual crypto pairs via Alpaca crypto trading.
    # 24/7 trading, no market hours restriction. Uses BTC tiered SMA(50/100) + momentum regime.
    # Symbols use Alpaca format (BTC/USD); mapped to yfinance format (BTC-USD) for data.
    # Low-cap / high-IV additions: AVAX (~$10B, very high IV), LINK (~$10B, high IV),
    #   DOGE (meme-driven, extreme IV) — OOS-validated after train-crypto run.
    # Screened 2026-03: SHIB (135% IV), DOT (130%), NEAR (110%), SUSHI (72%),
    #   ADA (69%), CRV (63%), AAVE (63%), RENDER (59%) — all Alpaca tradable.
    "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD",
               "SHIB/USD", "DOT/USD", "NEAR/USD", "SUSHI/USD", "ADA/USD",
               "CRV/USD", "AAVE/USD", "RENDER/USD"],
}

# Legacy / wrong-algorithm positions: opened by a previous buggy model. Manage them
# with stricter exits (lock profit or cut loss sooner) and optionally no new entries.
# Set via env: PAPER_LEGACY_STRICTER_EXIT=IGV,QQQ  PAPER_LEGACY_NO_NEW_ENTRIES=IGV,QQQ
LEGACY_STRICTER_EXIT_DEFAULT: Set[str] = set()
LEGACY_NO_NEW_ENTRIES_DEFAULT: Set[str] = set()
LEGACY_TRAILING_STOP_PCT = 0.03   # 3% from peak for legacy positions

def _is_option_symbol(symbol: str) -> bool:
    """True if symbol looks like an OCC option (e.g. QQQ260327C00592000). Used to skip
    option positions that may exist in the account but are not managed by this trader."""
    import re
    return bool(re.match(r"^[A-Z]+\d{6}[CP]\d{8}$", (symbol or "").upper()))


def _is_crypto_symbol(symbol: str) -> bool:
    """True if symbol is a crypto pair (e.g. BTC/USD, BTCUSD)."""
    return "/" in symbol or symbol.upper().endswith("USD") and len(symbol) >= 6


def _crypto_to_yfinance(symbol: str) -> str:
    """Convert Alpaca crypto symbol to yfinance format: BTC/USD → BTC-USD, DOTUSD → DOT-USD."""
    if "/" in symbol:
        return symbol.replace("/", "-")
    # Alpaca can return crypto positions as DOTUSD (no slash); Yahoo expects DOT-USD
    if symbol.upper().endswith("USD") and len(symbol) > 3:
        return symbol[:-3] + "-" + symbol[-3:]
    return symbol


def _warn_duplicate_symbols() -> None:
    """Log info if any symbol appears in more than one group (intentional cross-group allowed)."""
    from collections import defaultdict
    sym_to_groups: Dict[str, List[str]] = defaultdict(list)
    for grp, syms in SYMBOL_GROUPS.items():
        for s in syms:
            sym_to_groups[s].append(grp)
    dupes = {s: grps for s, grps in sym_to_groups.items() if len(grps) > 1}
    if dupes:
        log.info(
            "Symbol(s) shared across groups (intentional — separate accounts, separate models): %s",
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
    Lock file is group-specific so intraday/swing/crypto can run in parallel.
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
        max_loss_cooldown_hours: Optional[float] = None,
        post_loss_size_mult: float = 0.5,
        post_loss_size_hours: float = 4.0,
        same_dir_confidence_mult: float = 1.5,
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
        self.group = group

        self.loss_cooldown_hours = loss_cooldown_hours
        # Opt #1: separate (longer) cooldown after disaster-stop / max-loss
        self.max_loss_cooldown_hours = (
            max_loss_cooldown_hours if max_loss_cooldown_hours is not None
            else (2.0 if mode == "intraday" else loss_cooldown_hours)
        )
        # Opt #5: reduced sizing after max-loss (0.5x for N hours)
        self.post_loss_size_mult = post_loss_size_mult
        self.post_loss_size_hours = post_loss_size_hours
        # Opt #2: higher confidence multiplier for same-direction re-entry after max-loss
        self.same_dir_confidence_mult = same_dir_confidence_mult

        self._legacy_stricter_exit = legacy_stricter_exit or set()
        self._legacy_no_new_entries = legacy_no_new_entries or set()

        # Per-symbol tracking
        self._peak_prices: Dict[str, float] = {}
        self._entry_atrs: Dict[str, float] = {}
        self._cooldown_until: Dict[str, datetime] = {}
        self._decay_cooldown_until: Dict[str, datetime] = {}  # re-entry block after signal-decay exit
        # Opt #1/#2/#5: track max-loss exits per symbol (direction + time)
        self._max_loss_exits: Dict[str, dict] = {}

        self._vix_cache: Optional[pd.DataFrame] = None

        # Regime filter state
        from collections import deque
        self._recent_trade_wins: deque = deque(maxlen=20)
        self._regime_cooldown_until: Optional[datetime] = None

        # Alert engine for microstructure alerts
        self._alert_engine = AlertEngine()
        self._consecutive_losses: int = 0
        self._warned_shared_crypto: bool = False  # one-time warn if intraday account has crypto positions

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
        elif group == "crypto":
            # Crypto uses swing model with yfinance-mapped symbols (BTC/USD → BTC-USD)
            # Models stored separately in models/crypto/ to isolate from stock models
            yf_sym = _crypto_to_yfinance(symbol)
            crypto_dir = CRYPTO_MODEL_DIR
            try:
                from swing_model import SwingPredictor
                return SwingPredictor(yf_sym, model_dir=crypto_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No crypto swing model for %s (%s), falling back to LSTM: %s", symbol, yf_sym, exc)
            try:
                return Predictor(yf_sym, model_dir=crypto_dir,
                                 mode=mode, intraday_interval=intraday_interval)
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning("No trained model for %s (%s): %s — will skip ML signals.",
                            symbol, yf_sym, exc)
                return None
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
        qty: float,
        side: OrderSide,
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """Submit the right order type for the current session.

        Regular hours  → MarketOrder (immediate fill at best price).
        Extended hours → LimitOrder with extended_hours=True.
            BUY  limit: last_price × 1.001  (0.1% above to ensure fill)
            SELL limit: last_price × 0.999  (0.1% below to ensure fill)
        Alpaca rejects market orders outside regular hours.

        Crypto → MarketOrder with GTC time-in-force (24/7 market, fractional qty).
        """
        is_crypto = _is_crypto_symbol(symbol)
        session = _get_session()
        try:
            if is_crypto:
                # Crypto: always market order, GTC, fractional qty allowed
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.GTC,
                    )
                )
                log.info("CRYPTO MARKET %s %s x%.6f — order %s",
                         side.value, symbol, qty, order.id)
            elif session == "extended" and limit_price is not None:
                if side == OrderSide.BUY:
                    lp = round(limit_price * 1.001, 2)
                else:
                    lp = round(limit_price * 0.999, 2)
                order = self.trading_client.submit_order(
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=int(qty),
                        side=side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=lp,
                        extended_hours=True,
                    )
                )
                log.info("LIMIT(%s) %s %s x%d @ $%.2f — order %s",
                         session, side.value, symbol, int(qty), lp, order.id)
            else:
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=int(qty),
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                log.info("MARKET %s %s x%d — order %s",
                         side.value, symbol, int(qty), order.id)
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

    # -- Regime filter --------------------------------------------------
    def _check_regime(self) -> tuple:
        """Return (regime_ok: bool, reason: str).

        Equities:
          Gate 1 — SPY SMA(200): only trade in bull market (SPY close > 200-day MA).
          Gate 2 — VIX threshold: swing < 30 (skip for intraday — model handles vol).
        Crypto:
          Gate 1 — BTC tiered SMA: pass if BTC > SMA(20), or BTC > SMA(50)
                   with positive 14-day momentum, or 14d bounce > 5%.
                   Crypto is far more volatile than equities — 200/100-day
                   SMAs are too slow and miss valid recovery entries.
          Gate 2 — Skipped (VIX doesn't apply to crypto).
        All groups:
          Gate 3 — Rolling win rate: pause 7 days if last 20 trades win rate < 50%.

        Returns True only when ALL gates pass.
        """
        if self.group == "crypto":
            # Crypto regime: tiered SMA with momentum recovery
            try:
                btc_bars = self.adapter.fetch_daily("BTC-USD", 210)
                btc_close = float(btc_bars["close"].iloc[-1])
                btc_sma20 = float(btc_bars["close"].rolling(20).mean().iloc[-1])
                btc_sma50 = float(btc_bars["close"].rolling(50).mean().iloc[-1])
                btc_ret14 = float(btc_bars["close"].pct_change(14).iloc[-1])

                if btc_close > btc_sma20:
                    pass  # Short-term uptrend — fully open
                elif btc_close > btc_sma50 and btc_ret14 > 0:
                    pass  # Above SMA(50) with positive 14d momentum — allow
                elif btc_close > btc_sma50:
                    return False, (f"BTC between SMA(20)/SMA(50) but negative momentum: "
                                   f"{btc_close:,.0f}, SMA20={btc_sma20:,.0f}, "
                                   f"SMA50={btc_sma50:,.0f}, ret14={btc_ret14:+.2%}")
                else:
                    # Below SMA(50) but allow if strong positive momentum (bounce entry)
                    if btc_ret14 > 0.05:
                        pass  # 5%+ bounce in 14 days — allow cautious entries
                    else:
                        return False, (f"BTC below SMA(50): {btc_close:,.0f} <= "
                                       f"{btc_sma50:,.0f}, ret14={btc_ret14:+.2%}")
            except Exception as exc:
                log.warning("Regime BTC check failed (defaulting open): %s", exc)
        else:
            # Gate 1: SPY SMA(200)
            try:
                spy_bars = self.adapter.fetch_daily("SPY", 210)
                spy_close = float(spy_bars["close"].iloc[-1])
                spy_sma200 = float(spy_bars["close"].rolling(200).mean().iloc[-1])
                if spy_close <= spy_sma200:
                    return False, f"SPY below SMA(200): {spy_close:.2f} <= {spy_sma200:.2f}"
            except Exception as exc:
                log.warning("Regime SPY check failed (defaulting open): %s", exc)

            # Gate 2: VIX — skip for intraday (short holds benefit from vol),
            #               block swing only above 30 (real panic, not routine vol)
            if self.mode != "intraday":
                try:
                    vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
                    if not vix_df.empty:
                        vix_now = float(vix_df.iloc[-1].iloc[0] if vix_df.shape[1] == 1
                                        else vix_df["vix"].iloc[-1])
                        if vix_now >= 30.0:
                            return False, f"VIX elevated: {vix_now:.1f} >= 30"
                except Exception as exc:
                    log.warning("Regime VIX check failed (defaulting open): %s", exc)

        # Gate 3: rolling 20-trade win rate cooldown
        import datetime as _dt
        now = _dt.datetime.now(timezone.utc)
        if self._regime_cooldown_until is not None:
            if now < self._regime_cooldown_until:
                remaining = (self._regime_cooldown_until - now).seconds // 3600
                return False, f"Win-rate cooldown active ({remaining}h remaining)"
            else:
                self._regime_cooldown_until = None  # expired

        return True, "ok"

    def _data_symbol(self, symbol: str) -> str:
        """Map trading symbol to data-fetch symbol (crypto: BTC/USD → BTC-USD for yfinance)."""
        if _is_crypto_symbol(symbol):
            return _crypto_to_yfinance(symbol)
        return symbol

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
            ds = self._data_symbol(symbol)
            if self.mode == "intraday":
                # 5 days ensures Friday bars are present so FRED VIX (1-day lag)
                # can ffill into today's intraday bars on Mondays / after holidays.
                bars = self.adapter.fetch_intraday(
                    ds, self.intraday_interval, lookback_days=5)
            else:
                # 400 days needed: GLD/SLV use frac_diff_close (~282-bar warmup)
                bars = self.adapter.fetch_daily(ds, 400)
            vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
            return predictor.predict(bars, vix_df)
        except Exception as exc:
            log.error("Prediction failed for %s: %s", symbol, exc)
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest price for a symbol."""
        try:
            ds = self._data_symbol(symbol)
            if self.mode == "intraday":
                bars = self.adapter.fetch_intraday(ds, self.intraday_interval)
            else:
                bars = self.adapter.fetch_daily(ds, 5)
            if bars.empty:
                return None
            return float(bars["close"].iloc[-1])
        except Exception:
            return None

    def _get_market_context(self, symbol: str) -> Dict[str, float]:
        """Fetch ATR, trend signal, realized vol, and price for position management."""
        try:
            ds = self._data_symbol(symbol)
            bars = self.adapter.fetch_daily(ds, 60)
            if len(bars) < 30:
                return {"atr": 0.0, "trend": 0.0, "rv_30d": 0.0, "price": 0.0}
            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            atr_s = compute_atr(high, low, close, period=14)

            # Trend signal: SMA(50)
            sma = close.rolling(self.trend_sma_period).mean()
            trend = 1.0 if float(close.iloc[-1]) > float(sma.iloc[-1]) else -1.0
            if np.isnan(float(sma.iloc[-1])):
                trend = 0.0

            # Realized vol (annualized, 30-day)
            rv_30d = float(close.pct_change().rolling(30).std().iloc[-1]) * np.sqrt(365)

            def _safe(s: "pd.Series", default: float) -> float:
                v = float(s.iloc[-1])
                return default if np.isnan(v) else v

            return {
                "atr":    _safe(atr_s, 0.0),
                "trend":  trend,
                "rv_30d": rv_30d if not np.isnan(rv_30d) else 0.0,
                "price":  float(close.iloc[-1]),
            }
        except Exception as exc:
            log.warning("Market context fetch failed for %s: %s", symbol, exc)
            return {"atr": 0.0, "trend": 0.0, "rv_30d": 0.0, "price": 0.0}

    # -- Trading logic (one symbol) ------------------------------------
    def check_and_trade(self, symbol: str, positions: Dict[str, dict],
                        allocation: float, exit_only: bool = False) -> str:
        """Check ML signal and manage position for one symbol.

        Architecture:
        - Exit priority: disaster stop > signal decay
        - Entry: fixed threshold + trend filter (BTC SMA(200) regime for crypto)
        - Sizing: signal-proportional
        - Preserves: EOD exit for intraday, legacy exits

        Returns action string for display.
        """
        pred = self.get_prediction(symbol)
        direction = pred["direction"]
        confidence = pred["confidence"]
        expected_return = pred.get("expected_return", 0.0)

        ctx = self._get_market_context(symbol)
        bar_atr = ctx["atr"]
        bar_trend = ctx["trend"]

        is_crypto = _is_crypto_symbol(symbol)

        pos = positions.get(symbol)
        has_position = pos is not None and pos["qty"] > 0

        # Wrongly placed: symbol has no position here — nothing to manage
        if exit_only and not has_position:
            return "EXIT-ONLY  (no position — symbol not in this group)"

        # --- Exit logic ---
        if has_position:
            qty = pos["qty"]
            # For equities qty is int; for crypto it can be float
            qty_display = f"{qty:.6f}" if is_crypto and isinstance(qty, float) else str(int(qty))
            side = pos["side"]
            current_price = pos["current_price"]
            entry_price = pos["entry_price"]
            pnl_pct = pos["unrealized_pnl_pct"]

            use_legacy_stops = exit_only or (symbol in self._legacy_stricter_exit)

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
                    exit_reason = "eod_exit (intraday momentum)"
                    if side == "LONG":
                        self.sell(symbol, qty, reason=exit_reason,
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason=exit_reason,
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._log_daily_trade(symbol, side, qty, current_price, exit_reason, pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (EOD close {pnl_pct:+.2%}, {qty_display} sh {side})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")

            # Legacy positions: trailing stop (3% from peak)
            if use_legacy_stops:
                if drawdown_from_peak >= LEGACY_TRAILING_STOP_PCT:
                    exit_reason = f"legacy_trailing_stop ({drawdown_from_peak:.2%})"
                    if side == "LONG":
                        self.sell(symbol, qty, reason=exit_reason,
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason=exit_reason,
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._log_daily_trade(symbol, side, qty, current_price, exit_reason, pnl_pct)
                    self._clear_symbol_state(symbol)
                    return (f"EXIT  (legacy trailing stop {drawdown_from_peak:.2%} from peak, "
                            f"{qty_display} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # --- Priority 1: Disaster stop (safety net at N*ATR) ---
            if not use_legacy_stops:
                entry_atr = self._entry_atrs.get(symbol, bar_atr)
                if entry_atr > 0 and entry_price > 0:
                    disaster_stop_pct = self.disaster_stop_atr_mult * entry_atr / entry_price
                else:
                    disaster_stop_pct = 0.10
                if pnl_pct <= -disaster_stop_pct:
                    exit_reason = f"disaster_stop ({pnl_pct:+.2%})"
                    if side == "LONG":
                        self.sell(symbol, qty, reason=exit_reason,
                                  limit_price=current_price)
                    else:
                        self.buy_to_cover(symbol, qty, reason=exit_reason,
                                          limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._log_daily_trade(symbol, side, qty, current_price, exit_reason, pnl_pct)
                    self._clear_symbol_state(symbol)
                    # Opt #1: use the longer max-loss cooldown (2h intraday vs 0.5h normal)
                    cd_hours = self.max_loss_cooldown_hours
                    if cd_hours > 0:
                        cd = timedelta(hours=cd_hours)
                        self._cooldown_until[symbol] = datetime.now(timezone.utc) + cd
                        log.info("Max-loss cooldown %s: no re-entry for %.1fh after disaster stop", symbol, cd_hours)
                    # Opt #2/#5: record the exit direction and time for same-dir penalty + reduced sizing
                    self._max_loss_exits[symbol] = {
                        "direction": side,
                        "time": datetime.now(timezone.utc),
                    }
                    return (f"EXIT  (disaster stop {pnl_pct:+.2%} >= -{disaster_stop_pct:.2%} "
                            f"[{self.disaster_stop_atr_mult}xATR], "
                            f"{qty_display} sh {side})  ML: {direction} E[r]={expected_return:+.4f}")

            # --- Priority 2: Signal-decay exit ---
            if not use_legacy_stops:
                if side == "LONG" and expected_return <= 0:
                    exit_reason = f"signal_decay (E[r]={expected_return:+.4f})"
                    self.sell(symbol, qty, reason=exit_reason, limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._log_daily_trade(symbol, side, qty, current_price, exit_reason, pnl_pct)
                    self._clear_symbol_state(symbol)
                    self._decay_cooldown_until[symbol] = (
                        datetime.now(timezone.utc) + timedelta(seconds=2 * self.check_interval)
                    )
                    return (f"EXIT  (signal decay: E[r]={expected_return:+.4f} <= 0, "
                            f"P&L={pnl_pct:+.2%}, {qty_display} sh {side})")
                elif side == "SHORT" and expected_return >= 0:
                    exit_reason = f"signal_decay (E[r]={expected_return:+.4f})"
                    self.buy_to_cover(symbol, qty, reason=exit_reason, limit_price=current_price)
                    self._record_closed_trade(pnl_pct)
                    self._log_daily_trade(symbol, side, qty, current_price, exit_reason, pnl_pct)
                    self._clear_symbol_state(symbol)
                    self._decay_cooldown_until[symbol] = (
                        datetime.now(timezone.utc) + timedelta(seconds=2 * self.check_interval)
                    )
                    return (f"EXIT  (signal decay: E[r]={expected_return:+.4f} >= 0, "
                            f"P&L={pnl_pct:+.2%}, {qty_display} sh {side})")

            # HOLD
            return (f"HOLD  ({side} {qty_display} sh @ ${entry_price:.2f}, "
                    f"P&L: {pnl_pct:+.2%})  "
                    f"ML: {direction} E[r]={expected_return:+.4f}  trend={bar_trend:+.0f}")

        # --- Entry logic ---
        if exit_only:
            return (f"EXIT-ONLY  (managing out -- {direction} E[r]={expected_return:+.4f}, "
                    f"no new entries for this symbol in this account)")

        # Regime gate
        regime_ok, regime_reason = self._check_regime()
        if not regime_ok:
            return f"REGIME-BLOCK  ({regime_reason})  ML: {direction} E[r]={expected_return:+.4f}"

        # Determine entry direction
        enter_dir = None
        if expected_return > self.cost_threshold and bar_trend > 0:
            enter_dir = "LONG"
        elif expected_return < -self.cost_threshold and bar_trend < 0:
            enter_dir = "SHORT"

        if enter_dir is not None:
            # Cooldown check (disaster stop)
            cd_until = self._cooldown_until.get(symbol)
            if cd_until and datetime.now(timezone.utc) < cd_until:
                remaining = (cd_until - datetime.now(timezone.utc)).seconds // 60
                return (f"COOLDOWN  ({symbol} blocked for {remaining}m after loss exit, "
                        f"enter_dir={enter_dir})  ML: {direction} E[r]={expected_return:+.4f}")

            # Decay cooldown check
            dc_until = self._decay_cooldown_until.get(symbol)
            if dc_until and datetime.now(timezone.utc) < dc_until:
                remaining = max(1, int((dc_until - datetime.now(timezone.utc)).total_seconds() // 60))
                return (f"DECAY-COOLDOWN  ({symbol} blocked for {remaining}m after signal-decay exit, "
                        f"enter_dir={enter_dir})  ML: {direction} E[r]={expected_return:+.4f}")

            # Opt #2: after a max-loss, require higher confidence to re-enter in the SAME direction
            last_loss = self._max_loss_exits.get(symbol)
            same_dir_penalty = False
            if last_loss:
                loss_age_h = (datetime.now(timezone.utc) - last_loss["time"]).total_seconds() / 3600
                if loss_age_h < 24 and enter_dir == last_loss["direction"]:
                    effective_threshold = self.cost_threshold * self.same_dir_confidence_mult
                    if abs(expected_return) < effective_threshold:
                        return (f"SAME-DIR-BLOCK  ({symbol} {enter_dir} after max-loss {enter_dir}; "
                                f"E[r]={expected_return:+.4f} < {effective_threshold:.4f} "
                                f"[{self.same_dir_confidence_mult:.1f}x threshold])  "
                                f"ML: {direction}")
                    same_dir_penalty = True

            current_price = self._get_current_price(symbol)
            if current_price is None:
                return (f"SKIP  (price fetch failed for entry)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")

            # Signal-proportional sizing (V1 for all asset types)
            signal_pct = min(1.0, max(0.1, abs(expected_return) / self.target_return))
            sizing_pct = self.position_pct * signal_pct
            sizing_pct = min(sizing_pct, 0.98)

            # Opt #5: reduce sizing after a recent max-loss in this symbol
            size_note = ""
            if last_loss:
                loss_age_h = (datetime.now(timezone.utc) - last_loss["time"]).total_seconds() / 3600
                if loss_age_h < self.post_loss_size_hours:
                    sizing_pct *= self.post_loss_size_mult
                    size_note = f" [post-loss {self.post_loss_size_mult:.0%} sizing]"

            invest = allocation * sizing_pct
            qty = invest / current_price
            if is_crypto:
                qty = round(qty, 6)
            else:
                qty = int(qty)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} E[r]={expected_return:+.4f}"

            if enter_dir == "LONG":
                self.buy(symbol, qty, limit_price=current_price)
            else:
                self.sell_short(symbol, qty, limit_price=current_price)

            self._peak_prices[symbol] = current_price
            self._entry_atrs[symbol] = bar_atr

            qty_display = f"{qty:.6f}" if is_crypto else str(qty)
            self._log_daily_trade(symbol, enter_dir, qty, current_price,
                                  f"entry ({enter_dir})", 0.0)
            return (f"{enter_dir}  ({qty_display} sh @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f}, size={sizing_pct:.0%}{size_note})  "
                    f"ML: {direction} E[r]={expected_return:+.4f}  trend={bar_trend:+.0f}")

        return (f"SKIP  (no signal)  ML: {direction} E[r]={expected_return:+.4f}  "
                f"trend={bar_trend:+.0f}")

    def _record_closed_trade(self, pnl_pct: float) -> None:
        """Track consecutive losses and rolling win rate for regime cooldown."""
        if pnl_pct <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        # Rolling 20-trade win rate cooldown
        self._recent_trade_wins.append(1 if pnl_pct > 0 else 0)
        if len(self._recent_trade_wins) == 20 and self._regime_cooldown_until is None:
            wr = sum(self._recent_trade_wins) / 20
            if wr < 0.5:
                import datetime as _dt
                self._regime_cooldown_until = (
                    _dt.datetime.now(timezone.utc) + _dt.timedelta(days=7)
                )
                log.info("Rolling WR %.0f%% < 50%% → regime cooldown for 7 days (until %s)",
                         wr * 100, self._regime_cooldown_until.date())

    def _clear_symbol_state(self, symbol: str) -> None:
        """Reset tracking state for a symbol after exit."""
        self._peak_prices.pop(symbol, None)
        self._entry_atrs.pop(symbol, None)
        # Note: _decay_cooldown_until is intentionally NOT cleared here —
        # signal-decay exits set it after calling this method.

    def _log_daily_trade(self, symbol: str, side: str, qty: float,
                         price: float, reason: str, pnl_pct: float) -> None:
        """Append one row to outputs/daily_trades_YYYYMMDD.csv for reproducible trade history."""
        import csv
        try:
            from signals_engine import PROJECT_ROOT
            out_dir = os.path.join(PROJECT_ROOT, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            today_str = datetime.now().strftime("%Y%m%d")
            csv_path = os.path.join(out_dir, f"daily_trades_{today_str}.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "group", "symbol", "side", "qty",
                                     "price", "reason", "pnl_pct"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.group or "default",
                    symbol, side, qty, f"{price:.2f}", reason,
                    f"{pnl_pct:+.4f}",
                ])
        except Exception as exc:
            log.debug("Failed to write daily trade log: %s", exc)

    # -- Main loop -----------------------------------------------------
    def run_loop(self) -> None:
        """Main continuous trading loop."""
        if not self.symbols:
            log.info("No symbols configured for this group — exiting.")
            return
        log.info("Starting paper trading loop (%s mode)...", self.mode)
        log.info("Symbols: %s", ", ".join(self.symbols))
        log.info("Check interval: %d min", self.check_interval // 60)
        log.info("Strategy: v2 regression | trend SMA(%d) | cost_threshold=%.4f | "
                 "target_return=%.4f | disaster_stop=%.1f×ATR | "
                 "cooldown=%.1fh | max_loss_cooldown=%.1fh | "
                 "post_loss_size=%.0f%% for %.0fh | same_dir_mult=%.1fx",
                 self.trend_sma_period, self.cost_threshold,
                 self.target_return, self.disaster_stop_atr_mult,
                 self.loss_cooldown_hours, self.max_loss_cooldown_hours,
                 self.post_loss_size_mult * 100, self.post_loss_size_hours,
                 self.same_dir_confidence_mult)
        print()

        # Graceful shutdown
        def handle_signal(sig, frame):
            print("\n\n  Shutting down paper trader...\n")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        while self._running:
            cycle += 1

            # Session check — intraday: only trade during market/open hours; crypto & swing: run 24/7
            # Crypto and swing run 24/7 (orders fill when Alpaca allows; swing/crypto can use extended hours).
            if self.group == "crypto":
                session = "regular"
                session_label = "24/7 CRYPTO"
            elif self.group == "swing":
                session = "regular"
                session_label = "24/7 SWING"
            else:
                # Intraday: sleep when market closed, trade only during 04:00–20:00 ET weekdays
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

                # Positions in this account for symbols not in this group (wrongly placed).
                # Option positions (OCC symbols) are NOT managed here — they are legacy and
                # should be closed manually. So we only manage non-option, non-group symbols.
                exit_only_symbols = [
                    s for s in positions
                    if positions[s].get("qty", 0) > 0
                    and s not in self.symbols
                    and not _is_option_symbol(s)
                ]
                # One-time warn: intraday should use a dedicated account; crypto positions here = shared account
                if (self.group == "intraday" and exit_only_symbols and not self._warned_shared_crypto
                        and any(_is_crypto_symbol(s) for s in exit_only_symbols)):
                    self._warned_shared_crypto = True
                    log.warning(
                        "Intraday account has crypto positions (%s). Use a dedicated Alpaca paper account "
                        "for intraday (ALPACA_INTRADAY_KEY) so only ETF positions appear here.",
                        ", ".join(exit_only_symbols),
                    )
                    print("  [WARN] This account holds crypto positions; intraday group should use its own "
                          "Alpaca paper account (ALPACA_INTRADAY_KEY) to avoid mixing.")
                option_positions = [
                    s for s in positions
                    if positions[s].get("qty", 0) > 0 and _is_option_symbol(s)
                ]
                if exit_only_symbols:
                    print(f"  [!] EXIT-ONLY (not in group): {', '.join(exit_only_symbols)} -- managing out, no new entries")
                if option_positions:
                    print(f"  [OPTIONS] {', '.join(option_positions)} -- legacy option positions, not managed by this process")

                # Check each symbol in this group
                for sym in self.symbols:
                    sym_alloc = allocation_per_sym
                    # Legacy / wrong-algorithm: manage out only (stricter stops, no new entries)
                    if sym in self._legacy_no_new_entries:
                        action = self.check_and_trade(sym, positions, sym_alloc, exit_only=True)
                        print(f"  {sym:>5}:  {action}  [LEGACY no new entries]")
                        continue
                    # Extended-hours guard: Asian/EM ETFs have near-zero volume
                    # during US pre/after-market — underlying markets are closed.
                    # Skipping avoids 0.5-2% spread costs on worthless signals.
                    if session == "extended" and sym not in EXTENDED_HOURS_UNIVERSE:
                        # Still allow exits on existing positions, block new entries
                        pos = positions.get(sym)
                        if pos is None or pos["qty"] == 0:
                            print(f"  {sym:>5}:  SKIP  (extended hours -- low liquidity for this ETF)")
                            continue
                    action = self.check_and_trade(sym, positions, sym_alloc)
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
    print("  Mode matches group (intraday -> LightGBM; swing -> PatchTST; crypto -> swing model; LSTM fallback).")
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
    parser.add_argument("--config", type=str, default=None,
                        help="Path to trading config JSON (default: config/trading.json if it exists)")
    parser.add_argument("--group", type=str, default=None,
                        choices=list(SYMBOL_GROUPS.keys()) + ["all"],
                        help="Trade a named account group: intraday / swing / crypto / all")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols override (overrides --group)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"],
                        help="Data provider (default: yahoo)")
    parser.add_argument("--check-interval", type=int, default=None,
                        help="Check interval in minutes (default: 5 daily, 1 intraday)")
    parser.add_argument("--mode", default=None,
                        help="Trading mode (default: daily, or from config)")
    parser.add_argument("--interval", default=None,
                        help="Intraday bar interval (default: 5min, or from config)")
    # v2 regression parameters
    parser.add_argument("--trend-sma", type=int, default=None,
                        help="SMA period for trend filter (default: 50)")
    parser.add_argument("--cost-threshold", type=float, default=None,
                        help="Min expected return to trade (default: 0.001 = 0.1%%)")
    parser.add_argument("--target-return", type=float, default=None,
                        help="Expected return for full position size (default: 0.02 = 2%%)")
    parser.add_argument("--disaster-stop-mult", type=float, default=None,
                        help="ATR multiplier for disaster stop (default: 3.0)")
    parser.add_argument("--loss-cooldown-hours", type=float, default=None,
                        help="Hours to block re-entry after disaster stop (default: 4.0; 0=off)")
    parser.add_argument("--max-loss-cooldown-hours", type=float, default=None,
                        help="Hours to block re-entry after max-loss specifically (default: 2.0 intraday, same as --loss-cooldown-hours otherwise)")
    parser.add_argument("--post-loss-size-mult", type=float, default=None,
                        help="Position size multiplier after max-loss (default: 0.5 = half size)")
    parser.add_argument("--post-loss-size-hours", type=float, default=None,
                        help="Hours to apply reduced sizing after max-loss (default: 4.0)")
    parser.add_argument("--same-dir-confidence-mult", type=float, default=None,
                        help="Confidence multiplier for same-direction re-entry after max-loss (default: 1.5)")
    parser.add_argument("--legacy-stricter-exit", type=str, default="",
                        help="Comma-separated symbols to use stricter exits (3%% trail) for wrong-algorithm positions")
    parser.add_argument("--legacy-no-new-entries", type=str, default="",
                        help="Comma-separated symbols to manage out only, no new entries until flat")

    args = parser.parse_args()

    # Load config file: explicit --config > auto-detect config/trading.json
    import json as _json
    from signals_engine import PROJECT_ROOT as _PR
    cfg = {}
    config_path = args.config
    if config_path is None:
        auto_path = os.path.join(_PR, "config", "trading.json")
        if os.path.isfile(auto_path):
            config_path = auto_path
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                all_cfg = _json.load(f)
            # Use group-specific section if available, else top-level
            group_key = args.group or "swing"
            cfg = all_cfg.get(group_key, {})
            log.info("Loaded trading config from %s [%s]", config_path, group_key)
        except Exception as exc:
            log.warning("Failed to load config %s: %s", config_path, exc)

    # Apply config defaults (CLI args override config which overrides hardcoded defaults)
    def _resolve(cli_val, cfg_key, default):
        if cli_val is not None:
            return cli_val
        return cfg.get(cfg_key, default)

    args.mode = _resolve(args.mode, "mode", "daily")
    args.interval = _resolve(args.interval, "interval", "5min")
    args.trend_sma = _resolve(args.trend_sma, "trend_sma", 50)
    args.cost_threshold = _resolve(args.cost_threshold, "cost_threshold", COST_THRESHOLD)
    args.target_return = _resolve(args.target_return, "target_return", TARGET_RETURN)
    args.disaster_stop_mult = _resolve(args.disaster_stop_mult, "disaster_stop_mult", 3.0)
    args.loss_cooldown_hours = _resolve(args.loss_cooldown_hours, "loss_cooldown_hours", 4.0)
    args.max_loss_cooldown_hours = _resolve(args.max_loss_cooldown_hours, "max_loss_cooldown_hours", None)
    args.post_loss_size_mult = _resolve(args.post_loss_size_mult, "post_loss_size_mult", 0.5)
    args.post_loss_size_hours = _resolve(args.post_loss_size_hours, "post_loss_size_hours", 4.0)
    args.same_dir_confidence_mult = _resolve(args.same_dir_confidence_mult, "same_dir_confidence_mult", 1.5)

    # Resolve API keys: group-specific keys take priority over generic ALPACA_API_KEY
    group = args.group  # e.g. "intraday", "swing", "crypto", "all", or None
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

    # Guard: intraday-only symbols must not be traded on non-intraday accounts
    INTRADAY_ONLY = set(SYMBOL_GROUPS.get("intraday", [])) - set(SYMBOL_GROUPS.get("swing", [])) - set(SYMBOL_GROUPS.get("crypto", []))
    if group and group not in ("intraday", "all") and args.mode == "intraday":
        print(f"\n  ERROR: --mode intraday requires --group intraday (got --group {group}).")
        print("  Intraday mode must use the intraday account to avoid cross-account contamination.\n")
        sys.exit(1)
    if group and group not in ("intraday", "all") and args.symbols:
        bad = [s for s in args.symbols.split(",") if s.strip().upper() in INTRADAY_ONLY]
        if bad:
            print(f"\n  ERROR: Symbols {bad} are intraday-only and cannot be traded on the '{group}' account.")
            print(f"  Use --group intraday instead.\n")
            sys.exit(1)

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
        max_loss_cooldown_hours=args.max_loss_cooldown_hours,
        post_loss_size_mult=args.post_loss_size_mult,
        post_loss_size_hours=args.post_loss_size_hours,
        same_dir_confidence_mult=args.same_dir_confidence_mult,
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
