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
import json
import logging
import math
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
    PROJECT_ROOT, DEFAULT_UNIVERSE, EXTENDED_HOURS_UNIVERSE,
    DAILY_LOOKBACK, build_adapter, FREDVixFetcher,
    compute_atr,
)
from ml_model import Predictor  # LSTM fallback (deprecated)
from utils import (
    _fetch_vix_for_training, DEFAULT_MODEL_DIR, SWING_MODEL_DIR, INTRADAY_MODEL_DIR,
    CRYPTO_MODEL_DIR, COST_THRESHOLD, TARGET_RETURN, TRADES_DIR, MONITOR_DIR,
    PAPER_STATE_DIR, get_model_dir,
)
from alerts import AlertEngine
from risk_config import (
    get_risk_config, check_position_allowed, check_theme_cap,
    check_sleeve_budget,
    validate_model_mode, get_symbol_cap, is_symbol_disabled,
    DeRiskState, evaluate_derisk,
    get_effective_min_hold, drawdown_size_mult,
)
from cost_model import validate_cost_threshold
from model_monitor import ModelMonitor
from oos_feedback import (
    load_promoted_oos, compute_composite_scores as _composite_scores_shared,
    blended_sharpe as _blended_sharpe_shared,
)
from coin_selector import (CoinSelector, CRYPTO_UNIVERSE, fetch_universe_data,
                           fetch_universe_intraday_data, INTRADAY_LOOKBACK_DAYS)
from universe_screener import load_universe, get_coin_cost_config
from kraken_executor import KrakenExecutor

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
    # Full trained universe as fallback pools. Dynamic selectors pick top-K each
    # cycle via composite scoring (James-Stein + OOS Sharpe + correlation penalty
    # + fee-aware). Symbols with negative OOS Sharpe rank low and get zero capital.
    #
    # Account 1 — Intraday LGB+GRU (5-min bars, 1-hour horizon)
    # Selector picks top-8 from ranked file; all 59 trained models as fallback
    "intraday": [
        "ARKK", "BND", "CIBR", "EEM", "EMB", "ETHA", "EWA", "EWC", "EWG",
        "EWH", "EWJ", "EWT", "EWU", "EWW", "EWY", "EWZ", "FBTC", "GDX",
        "GDXJ", "GLD", "HYG", "IAU", "IBIT", "IEF", "IGV", "INDA", "IWB",
        "IWF", "IWM", "LQD", "MCHI", "MTUM", "PDBC", "QQQ", "QUAL", "SHY",
        "SLV", "SMH", "SOXX", "SPY", "TIP", "TLT", "URNM", "USMV", "USO",
        "VBR", "VGK", "VTV", "VWO", "XLB", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY",
    ],
    # Account 2 — Swing XGBoost+TFT (daily, 10-day horizon)
    # Selector picks top-10 from ranked file; all 59 trained models as fallback
    "swing": [
        "ARKK", "BND", "CIBR", "EEM", "EMB", "ETHA", "EWA", "EWC", "EWG",
        "EWH", "EWJ", "EWT", "EWU", "EWW", "EWY", "EWZ", "FBTC", "GDX",
        "GDXJ", "GLD", "HYG", "IAU", "IBIT", "IEF", "IGV", "INDA", "IWB",
        "IWF", "IWM", "LQD", "MCHI", "MTUM", "PDBC", "QQQ", "QUAL", "SHY",
        "SLV", "SMH", "SOXX", "SPY", "TIP", "TLT", "URNM", "USMV", "USO",
        "VBR", "VGK", "VTV", "VWO", "XLB", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY",
    ],
    # Account 3 — Crypto swing (daily, 10-day horizon)
    # Coin selector ranks full 65-coin universe; these are fallback if selector fails
    "crypto": ["CRV/USD", "AVAX/USD", "ADA/USD", "LINK/USD",
               "BTC/USD", "ETH/USD", "SOL/USD", "DOT/USD", "ATOM/USD",
               "QNT/USD", "AR/USD", "BAT/USD", "FARTCOIN/USD", "AKT/USD",
               "HNT/USD", "SAND/USD", "SNX/USD", "EGLD/USD", "FIL/USD",
               "SYRUP/USD", "AXS/USD", "ONDO/USD", "GOMINING/USD", "RUNE/USD"],
    # Account 4 — Crypto Intraday LGB+GRU (5-min bars, 1-hour horizon)
    # Coin selector ranks full universe; these are fallback
    "crypto_intraday": ["ATOM/USD", "QNT/USD", "XLM/USD", "LPT/USD",
                        "FIL/USD", "LTC/USD", "FET/USD", "DOT/USD",
                        "SEI/USD", "VET/USD", "BCH/USD", "ONDO/USD",
                        "NEAR/USD", "MANA/USD", "AXS/USD", "ICP/USD",
                        "WLD/USD", "AAVE/USD", "ALGO/USD"],
}

# Legacy / wrong-algorithm positions: opened by a previous buggy model. Manage them
# with stricter exits (lock profit or cut loss sooner) and optionally no new entries.
# Set via env: PAPER_LEGACY_STRICTER_EXIT=IGV,QQQ  PAPER_LEGACY_NO_NEW_ENTRIES=IGV,QQQ
LEGACY_STRICTER_EXIT_DEFAULT: Set[str] = set()
LEGACY_NO_NEW_ENTRIES_DEFAULT: Set[str] = set()
LEGACY_TRAILING_STOP_PCT = 0.03   # 3% from peak for legacy positions

def _resolve_swing_symbols() -> List[str]:
    """Resolve swing group symbols dynamically.

    Fallback chain:
      1. etf_candidates_ranked.json (pipeline-ranked, filtered to trained models, top-K)
      2. promoted_symbols.json (OOS-validated symbols from batch-backtest)
      3. etf_universe.json (full screened universe)
      4. Hardcoded fallback (SYMBOL_GROUPS["swing"])

    Only symbols with a trained swing model on disk are included.
    """
    try:
        ranked = _load_ranked_symbols("swing")
        if ranked:
            return ranked
    except Exception as e:
        log.warning("Ranked swing symbol load failed: %s", e)

    # Legacy fallback chain
    try:
        from etf_screener import load_promoted_symbols, load_etf_universe
        promoted = load_promoted_symbols(SWING_MODEL_DIR)
        if promoted:
            if len(promoted) > 15:
                try:
                    from etf_selector import ETFSelector
                    selector = ETFSelector(top_k=10)
                    selected = selector.select(promoted)
                    log.info("Swing symbols: ETFSelector picked %d/%d promoted",
                             len(selected), len(promoted))
                    return selected
                except Exception as e:
                    log.warning("ETFSelector failed: %s — using all promoted", e)
            log.info("Swing symbols from promoted_symbols.json: %s", promoted)
            return promoted
        universe = load_etf_universe(SWING_MODEL_DIR)
        if universe:
            log.info("Swing symbols from etf_universe.json (%d symbols)", len(universe))
            return universe
    except Exception as e:
        log.warning("Dynamic swing symbol load failed: %s — using hardcoded", e)
    return SYMBOL_GROUPS["swing"]


# Maximum symbols the trader will trade per group (safety cap)
MAX_SWING_SYMBOLS = 10
MAX_INTRADAY_SYMBOLS = 7


def _load_ranked_symbols(group: str) -> List[str]:
    """Load top-K symbols from pipeline-ranked JSON, gated on trained models.

    For swing:  reads etf_candidates_ranked.json, checks *_xgb_swing.joblib
    For intraday: reads etf_candidates_intraday_ranked.json, checks *_lgb_intraday_etf.joblib

    Returns ranked list of symbols (best first), or empty list if no ranked file.
    """
    import glob as _glob

    if group == "swing":
        ranked_file = os.path.join(SWING_MODEL_DIR, "etf_candidates_ranked.json")
        model_pattern = os.path.join(SWING_MODEL_DIR, "*_xgb_swing.joblib")
        max_k = MAX_SWING_SYMBOLS
        score_key = "final_score"
    elif group == "intraday":
        ranked_file = os.path.join(SWING_MODEL_DIR, "etf_candidates_intraday_ranked.json")
        # Check both intraday model patterns (etf-specific and generic)
        model_pattern_etf = os.path.join(INTRADAY_MODEL_DIR, "*_lgb_intraday_etf.joblib")
        model_pattern_gen = os.path.join(INTRADAY_MODEL_DIR, "*_lgb_intraday.joblib")
        max_k = MAX_INTRADAY_SYMBOLS
        score_key = "intraday_rank_score"
    else:
        return []

    if not os.path.exists(ranked_file):
        return []

    try:
        with open(ranked_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, KeyError):
        log.warning("Corrupt ranked file: %s", ranked_file)
        return []

    candidates = data.get("candidates", [])
    if not candidates:
        return []

    # Build set of symbols that have trained models on disk
    if group == "intraday":
        model_files = set(_glob.glob(model_pattern_etf)) | set(_glob.glob(model_pattern_gen))
    else:
        model_files = set(_glob.glob(model_pattern))

    trained: Set[str] = set()
    for path in model_files:
        fname = os.path.basename(path)
        # Strip suffix to get symbol: SMH_xgb_swing.joblib → SMH
        sym = fname.split("_")[0]
        trained.add(sym)

    # Filter to trained models, preserve ranking order
    ranked_symbols = []
    for c in candidates:
        sym = c.get("symbol", "")
        if sym in trained and sym not in ranked_symbols:
            ranked_symbols.append(sym)
        if len(ranked_symbols) >= max_k:
            break

    if ranked_symbols:
        log.info("%s fallback symbols from ranked file (top-%d of %d with models): %s",
                 group.capitalize(), len(ranked_symbols), len(trained), ranked_symbols)
    return ranked_symbols


def _resolve_intraday_symbols() -> List[str]:
    """Resolve intraday group symbols dynamically.

    Fallback chain:
      1. etf_candidates_intraday_ranked.json (pipeline-ranked, top-K with models)
      2. Hardcoded fallback (SYMBOL_GROUPS["intraday"])

    Only symbols with a trained intraday model on disk are included.
    """
    try:
        ranked = _load_ranked_symbols("intraday")
        if ranked:
            return ranked
    except Exception as e:
        log.warning("Ranked intraday symbol load failed: %s", e)
    return SYMBOL_GROUPS["intraday"]


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
                            "Stale lock file (PID %d no longer running) — breaking stale lock.",
                            owner_pid,
                        )
                        # On Windows, can't delete a file with an OS-level lock held
                        # by a zombie/service process. Try to remove, but if that fails
                        # just proceed without lock protection (safe since owner is dead).
                        try:
                            os.remove(lock_path)
                        except OSError:
                            log.warning(
                                "Cannot delete stale lock file (held by zombie PID %d) — "
                                "proceeding without single-instance lock.", owner_pid,
                            )
                            return True  # safe: owner is confirmed dead
                        time.sleep(0.5)
                        continue  # retry
                except (ValueError, OSError):
                    # Can't read PID; try remove and retry once
                    try:
                        os.remove(lock_path)
                    except OSError:
                        log.warning("Cannot read or remove lock file — proceeding without lock.")
                        return True
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


def _parse_window_list(windows: List[str]) -> List[tuple]:
    """Parse ["HH:MM-HH:MM", ...] into list of (time, time) tuples (ET)."""
    from datetime import time as dt_time
    parsed = []
    for w in windows:
        try:
            start_s, end_s = w.split("-")
            sh, sm = start_s.strip().split(":")
            eh, em = end_s.strip().split(":")
            parsed.append((dt_time(int(sh), int(sm)), dt_time(int(eh), int(em))))
        except (ValueError, AttributeError):
            log.warning("Invalid time window format: %r — skipping", w)
    return parsed


def _in_time_window(windows: List[tuple]) -> bool:
    """Check if current ET time falls within any of the parsed windows."""
    if not windows:
        return False
    from datetime import time as dt_time
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York")).time()
    return any(start <= now_et <= end for start, end in windows)


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
# Selector refresh gate — date-change, not wall-clock
# ===================================================================
# ETFs: daily bars settle at 16:00 ET; refresh after 06:30 ET next day
#        (pre-market data available, yesterday's bar confirmed)
# Crypto: 24/7 market; daily bars settle at 00:00 UTC; refresh after 00:30 UTC
_SELECTOR_REFRESH_HOUR_ET = 6   # ETF: refresh allowed after 6 AM ET
_SELECTOR_REFRESH_MIN_ET = 30
_SELECTOR_REFRESH_HOUR_UTC = 0  # Crypto: refresh allowed after 00:00 UTC
_SELECTOR_REFRESH_MIN_UTC = 30

# Intraday groups re-rank more frequently (within the same day)
# Swing groups only refresh once per day (no fast re-rank needed)
_FAST_REFRESH_SECS: Dict[str, int] = {
    "intraday": 3600,         # 1 hour — intraday opportunities rotate
    "crypto_intraday": 1800,  # 30 min — crypto moves faster
}


def _should_refresh_selector(
    group: str,
    last_run_date: Optional["date"],  # noqa: F821
) -> bool:
    """Return True if the selector should refresh (new trading day, past settlement).

    ETFs: refreshes once per trading day (Mon-Fri) after 06:30 ET.
          Skips weekends — no new daily bar settles on Sat/Sun.
    Crypto: refreshes once per calendar day after 00:30 UTC (24/7 market).
    """
    if last_run_date is None:
        return True  # first run

    if group in ("crypto", "crypto_intraday"):
        now = datetime.now(timezone.utc)
        today = now.date()
        if today > last_run_date and (
            now.hour > _SELECTOR_REFRESH_HOUR_UTC
            or (now.hour == _SELECTOR_REFRESH_HOUR_UTC
                and now.minute >= _SELECTOR_REFRESH_MIN_UTC)
        ):
            return True
    else:
        # ETF groups: use Eastern time, skip weekends
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today = now_et.date()
        # weekday(): Mon=0 … Sun=6; skip Sat(5) and Sun(6)
        if now_et.weekday() >= 5:
            return False
        if today > last_run_date and (
            now_et.hour > _SELECTOR_REFRESH_HOUR_ET
            or (now_et.hour == _SELECTOR_REFRESH_HOUR_ET
                and now_et.minute >= _SELECTOR_REFRESH_MIN_ET)
        ):
            return True
    return False


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
        disaster_stop_max_pct: float = 0.20,
        profit_lock_atr_mult: float = 2.0,
        profit_lock_trail_atr_mult: float = 1.5,
        max_underwater_days: int = 90,
        loss_cooldown_hours: float = 4.0,
        max_loss_cooldown_hours: Optional[float] = None,
        post_loss_size_mult: float = 0.5,
        post_loss_size_hours: float = 4.0,
        same_dir_confidence_mult: float = 1.5,
        legacy_stricter_exit: Optional[Set[str]] = None,
        legacy_no_new_entries: Optional[Set[str]] = None,
        group: Optional[str] = None,
        blocked_windows: Optional[List[str]] = None,
        half_size_windows: Optional[List[str]] = None,
        target_vol: float = 0.20,
    ):
        # Alpaca client is optional when using Kraken for crypto-only group
        if api_key and api_secret:
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=True,  # HARDCODED for safety
            )
        else:
            self.trading_client = None
            log.info("No Alpaca keys provided — Kraken executor will handle all orders")
        self.symbols = symbols
        self.position_pct = position_pct
        self.check_interval = check_interval_min * 60
        self.mode = mode
        self.intraday_interval = intraday_interval
        # v2 regression parameters
        self.trend_sma_period = trend_sma_period
        self.cost_threshold = cost_threshold
        self.target_return = target_return
        self.target_vol = target_vol
        self.disaster_stop_atr_mult = disaster_stop_atr_mult
        self.disaster_stop_max_pct = disaster_stop_max_pct
        self.profit_lock_atr_mult = profit_lock_atr_mult
        self.profit_lock_trail_atr_mult = profit_lock_trail_atr_mult
        self.max_underwater_days = max_underwater_days

        self.adapter = build_adapter(provider)
        self.fred_key = os.environ.get("FRED_API_KEY")
        self.group = group

        # Time-of-day windows: block new entries or reduce size during specified ET windows
        self._blocked_windows = _parse_window_list(blocked_windows or [])
        self._half_size_windows = _parse_window_list(half_size_windows or [])

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
        self._entry_times: Dict[str, datetime] = {}
        self._profit_lock_active: Dict[str, bool] = {}
        self._cooldown_until: Dict[str, datetime] = {}
        self._decay_cooldown_until: Dict[str, datetime] = {}  # re-entry block after signal-decay exit
        # Opt #1/#2/#5: track max-loss exits per symbol (direction + time)
        self._max_loss_exits: Dict[str, dict] = {}

        # Regime detector (Phase 4): HMM-based adaptive stops
        self._regime_detector = None
        self._regime_mult = 1.0  # current regime stop multiplier
        try:
            from regime_detector import RegimeDetector
            self._regime_detector = RegimeDetector()
            log.info("Regime detector initialized (will fit on first cycle)")
        except ImportError:
            log.info("Regime detector unavailable (hmmlearn not installed)")

        self._vix_cache: Optional[pd.DataFrame] = None

        # Crypto intraday data source (lazy init)
        self._crypto_intraday_data = None

        # Regime filter state
        from collections import deque
        self._recent_trade_wins: deque = deque(maxlen=50)
        self._regime_cooldown_until: Optional[datetime] = None

        # Alert engine for microstructure alerts
        self._alert_engine = AlertEngine()
        self._consecutive_losses: int = 0
        self._warned_shared_crypto: bool = False  # one-time warn if intraday account has crypto positions
        self._peak_equity: float = 0.0  # high-water mark for drawdown throttle (set below after account init)
        self.initial_capital: float = 100_000.0  # fallback if account fetch fails on first cycle

        # Load group-specific risk config for portfolio constraints
        self._risk_config = get_risk_config(group)
        log.info("Risk config: position_pct=%.0f%%, max_pos_pct=%.0f%%, "
                 "max_sector=%.0f%%, max_exposure=%.0f%%, max_positions=%d",
                 self._risk_config.position_pct * 100,
                 self._risk_config.max_position_pct * 100,
                 self._risk_config.max_sector_pct * 100,
                 self._risk_config.max_total_exposure * 100,
                 self._risk_config.max_positions)

        # Layer 1 coin selector (crypto group only)
        self._coin_selector: Optional[CoinSelector] = None
        # Use group-specific model dir for universe + selector
        self._crypto_model_dir = get_model_dir(group) if group in ("crypto", "crypto_intraday") else CRYPTO_MODEL_DIR
        # Prefer dynamic universe from Layer 0 screener; fall back to shared crypto universe
        dynamic_universe = load_universe(self._crypto_model_dir)
        if not dynamic_universe and self._crypto_model_dir != CRYPTO_MODEL_DIR:
            dynamic_universe = load_universe(CRYPTO_MODEL_DIR)
        self._selector_universe = dynamic_universe if dynamic_universe else list(CRYPTO_UNIVERSE)
        self._universe_last_check: Optional[datetime] = None  # throttle reload checks
        self._selector_active_symbols: List[str] = []    # dynamically selected coins
        self._selector_rank_scores: Dict[str, float] = {}  # symbol → rank score
        self._selector_last_run_date: Optional["date"] = None  # date-change gate
        self._selector_last_fast_refresh: Optional[datetime] = None  # intraday fast re-rank
        self._btc_correlations: Dict[str, float] = {}  # cached BTC correlations for sizing
        self._selector_mode = "intraday" if group == "crypto_intraday" else "swing"
        if group in ("crypto", "crypto_intraday"):
            # Defer selector loading to first cycle — loading at init hangs when
            # all 5 groups launch simultaneously under memory pressure.
            log.info("Coin selector deferred — will load on first trading cycle")

        # Layer 1 ETF selector (swing + intraday groups)
        self._etf_selector = None
        self._etf_selector_last_run_date: Optional["date"] = None
        self._etf_selector_last_fast_refresh: Optional[datetime] = None  # intraday fast re-rank
        self._etf_active_symbols: List[str] = []
        self._etf_universe: List[str] = []
        self._etf_train_failures: Dict[str, datetime] = {}  # symbol → last failure time
        self._etf_train_cooldown_hours = 24  # retry failed training after 24h
        # Background subprocess auto-training (Phase 4: non-blocking)
        self._bg_train_procs: Dict[str, "subprocess.Popen"] = {}  # symbol → Popen
        self._bg_train_started: Dict[str, datetime] = {}  # symbol → start time
        _BG_TRAIN_TIMEOUT = 300  # 5 min max per background train
        _BG_MAX_CONCURRENT = 5
        self._bg_train_timeout = _BG_TRAIN_TIMEOUT
        self._bg_max_concurrent = _BG_MAX_CONCURRENT
        # ETF selector rotation: enabled for swing only.
        # Intraday rotation disabled — trades only OOS-validated symbols (IC > 0.05).
        # Rotating to untested symbols is out-of-distribution inference with no proven edge.
        if group == "swing":
            try:
                from etf_screener import load_etf_universe
                from etf_selector import ETFSelectorML
                top_k = 10
                self._etf_selector = ETFSelectorML(
                    model_dir=SWING_MODEL_DIR, top_k=top_k, min_pool=0,
                )
                self._etf_universe = load_etf_universe(SWING_MODEL_DIR)
                if not self._etf_universe:
                    from etf_screener import ALL_SEED_SYMBOLS
                    self._etf_universe = list(ALL_SEED_SYMBOLS)
                log.info("ETF selector loaded for %s — %d universe symbols, top_k=%d",
                         group, len(self._etf_universe), top_k)
            except Exception as exc:
                log.warning("ETF selector init failed: %s — using static symbol list", exc)
        elif group == "intraday":
            try:
                from etf_screener import load_etf_universe, ALL_SEED_SYMBOLS
                from etf_selector import ETFIntradaySelectorML
                top_k = 8
                self._etf_selector = ETFIntradaySelectorML(
                    model_dir=INTRADAY_MODEL_DIR, top_k=top_k, min_pool=0,
                )
                self._etf_universe = load_etf_universe(SWING_MODEL_DIR)
                if not self._etf_universe:
                    self._etf_universe = list(ALL_SEED_SYMBOLS)
                log.info("ETF intraday selector loaded — %d universe symbols, top_k=%d",
                         len(self._etf_universe), top_k)
            except Exception as exc:
                log.warning("Intraday ETF selector init failed: %s — using static list", exc)
                log.info("Intraday ETF symbols (static): %s", symbols)

        # Kraken executor for crypto groups (supports long + short)
        # Paper mode doesn't need real API keys — uses public price API + local state
        self._kraken: Optional[KrakenExecutor] = None
        if group in ("crypto", "crypto_intraday"):
            kraken_key = os.environ.get("KRAKEN_API_KEY", "")
            kraken_secret = os.environ.get("KRAKEN_API_SECRET", "")
            use_kraken = os.environ.get("KRAKEN_EXECUTOR", "1")  # enabled by default for crypto
            if use_kraken != "0":
                state_dir = PAPER_STATE_DIR
                # Separate state file for crypto_intraday to avoid conflicts
                state_file = ("kraken_intraday_paper_state.json" if group == "crypto_intraday"
                              else "kraken_paper_state.json")
                self._kraken = KrakenExecutor(
                    api_key=kraken_key,
                    api_secret=kraken_secret,
                    paper=not (kraken_key and kraken_secret),  # live only with real keys
                    leverage=2,
                    state_dir=state_dir,
                    state_file=state_file,
                    initial_balance=float(os.environ.get("KRAKEN_PAPER_BALANCE", "10000")),
                )
                mode_label = "LIVE" if (kraken_key and kraken_secret) else "PAPER"
                log.info("Kraken executor enabled for %s group (%s mode)", group, mode_label)
                # Initialize peak equity from initial_balance so drawdown halt
                # works correctly even on first cycle after restart.
                # Without this, _peak_equity starts at 0 and gets set to
                # current (already-low) equity, masking the true drawdown.
                self._peak_equity = max(self._peak_equity, self._kraken._paper_initial)
                log.info("Peak equity initialized to %.0f (initial_balance)", self._peak_equity)
            else:
                log.info("KRAKEN_EXECUTOR=0 — crypto will use Alpaca (long-only)")

        # Validate cost threshold against estimated costs
        for sym in symbols:
            ok, msg, _ = validate_cost_threshold(sym, cost_threshold, safety_margin=1.5)
            if not ok:
                log.warning("Cost threshold validation: %s", msg)

        self.predictors: Dict[str, Optional[Predictor]] = {}
        for sym in symbols:
            predictor = self._create_predictor(sym, group, model_dir, mode, intraday_interval)
            self.predictors[sym] = predictor
            if predictor is not None:
                model_type = getattr(predictor, 'model_type', 'lstm')
                log.info("Loaded %s model for %s (%s).", model_type, sym, mode)

        self._running = True

        # Initialize peak equity from account if not already set (Alpaca accounts)
        if self._peak_equity <= 0.0:
            try:
                acct = self.get_account_summary()
                self._peak_equity = acct["equity"]
                log.info("Peak equity initialized to %.0f (current account equity)", self._peak_equity)
            except Exception:
                pass  # will be set on first trade cycle

        # Model monitor: tracks predicted vs realized returns
        os.makedirs(MONITOR_DIR, exist_ok=True)
        self._monitor = ModelMonitor(output_dir=MONITOR_DIR, window=60)
        self._monitor.log_pause_summary()
        self._entry_predictions: Dict[str, float] = {}  # symbol → predicted return at entry
        self._derisk_states: Dict[str, DeRiskState] = {}  # symbol → rolling perf tracker
        self._rebuild_derisk_states()
        # Crypto swing guardrails: concentration limiter + win-rate auto-pause
        self._symbol_pnl_tracker: Dict[str, float] = {}  # symbol → cumulative P&L over recent trades
        self._symbol_pnl_count: int = 0  # trades tracked in current window
        self._crypto_wr_pause_until: Optional[datetime] = None  # win-rate auto-pause expiry

        self._pending_reconcile: Dict[str, dict] = {}  # symbols that failed to close during reconciliation
        self._crypto_reconciled: bool = False  # True after first post-selector reconciliation
        self._etf_reconciled: bool = False     # True after first ETF selector reconciliation
        self._stale_since: Dict[str, datetime] = {}   # symbol → first time seen as exit-only
        self._stale_max_hours: float = 8.0             # auto-close stale positions after 8 hours

        # Volatility-tier exit params (classified at startup, recomputed weekly)
        from src.risk_config import (VolTier, ExitParams, classify_vol_tier,
                                     compute_vol_metrics, get_exit_params)
        self._vol_tiers: Dict[str, VolTier] = {}
        self._exit_params: Dict[str, ExitParams] = {}
        self._vol_tier_last_update: Optional[datetime] = None
        self._breakeven_floor: Dict[str, float] = {}  # symbol → floor price (entry or higher)
        self._signal_flip_counts: Dict[str, int] = {}  # consecutive signal flips
        self._bars_held: Dict[str, int] = {}  # checks since entry (for min_hold_bars)
        self._ci_bars_held: Dict[str, int] = {}  # crypto_intraday: 5-min bars since entry
        self._ci_last_bar_ts: Dict[str, datetime] = {}  # last bar timestamp per symbol (for dedup)
        self._classify_vol_tiers()

    # -- Lazy coin selector loading ----------------------------------------

    def _try_load_coin_selector(self) -> Optional["CoinSelector"]:
        """Load CoinSelector with timeout to avoid hanging under memory pressure.

        When all 5 groups launch simultaneously the LightGBM Booster can hang
        or OOM.  Runs the load in a thread with a 30s timeout.
        """
        import concurrent.futures

        dirs_to_try = [self._crypto_model_dir]
        if self._crypto_model_dir != CRYPTO_MODEL_DIR:
            dirs_to_try.append(CRYPTO_MODEL_DIR)

        def _load(d: str) -> "CoinSelector":
            return CoinSelector(model_dir=d, mode=self._selector_mode)

        for d in dirs_to_try:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_load, d)
                    sel = future.result(timeout=30)
                log.info("Coin selector loaded (mode=%s) from %s",
                         self._selector_mode, d)
                return sel
            except FileNotFoundError:
                continue
            except concurrent.futures.TimeoutError:
                log.warning("Coin selector load timed out (30s) — will retry next cycle")
                return None
            except Exception as exc:
                log.warning("Coin selector load failed (%s) — will retry next cycle: %s",
                            type(exc).__name__, exc)
                return None
        log.info("No trained coin selector — using static symbol list")
        return None

    def _ensure_coin_selector(self) -> Optional["CoinSelector"]:
        """Lazy retry: if selector is None, try loading once more."""
        if self._coin_selector is not None:
            return self._coin_selector
        if self.group not in ("crypto", "crypto_intraday"):
            return None
        self._coin_selector = self._try_load_coin_selector()
        return self._coin_selector

    # -- Derisk state persistence -----------------------------------------

    def _rebuild_derisk_states(self) -> None:
        """Rebuild derisk states from CSV trade logs so Kelly survives restarts."""
        import csv, glob
        group = self.group or "default"
        pattern = os.path.join(TRADES_DIR, "daily_trades_*.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            return
        count = 0
        for csv_path in csv_files:
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("group") != group:
                            continue
                        reason = row.get("reason", "")
                        pnl_str = row.get("pnl_pct", "+0.0000")
                        pnl = float(pnl_str)
                        # Skip entries (pnl == 0 and reason contains "entry")
                        if "entry" in reason.lower() and abs(pnl) < 1e-8:
                            continue
                        sym = row.get("symbol", "")
                        if not sym:
                            continue
                        derisk_key = sym.replace("/", "-")
                        if derisk_key not in self._derisk_states:
                            self._derisk_states[derisk_key] = DeRiskState()
                        self._derisk_states[derisk_key].record_trade(pnl)
                        count += 1
            except Exception as exc:
                log.debug("Skipping %s in derisk rebuild: %s", csv_path, exc)
        if count:
            kelly_ready = [k for k, v in self._derisk_states.items()
                           if v.half_kelly() is not None]
            log.info("Rebuilt derisk states from %d closed trades across %d symbols; "
                     "%d symbols have Kelly: %s",
                     count, len(self._derisk_states),
                     len(kelly_ready), kelly_ready or "none")

    # -- Crypto swing guardrails ------------------------------------------

    def _check_symbol_concentration(self, symbol: str) -> Optional[str]:
        """Block entry if a single symbol dominates recent P&L (>50% of group total).

        Returns reason string if blocked, None if OK.
        Only applies to crypto/crypto_intraday groups.
        """
        if self.group not in ("crypto", "crypto_intraday"):
            return None
        if self._symbol_pnl_count < 20:
            return None  # need 20+ trades for meaningful concentration check
        total_abs = sum(abs(v) for v in self._symbol_pnl_tracker.values())
        if total_abs < 1e-8:
            return None
        sym_key = symbol.replace("/", "-")
        sym_pnl = abs(self._symbol_pnl_tracker.get(sym_key, 0.0))
        concentration = sym_pnl / total_abs
        if concentration > 0.50:
            return (f"concentration={concentration:.0%} for {symbol} "
                    f"(>{50}% of group P&L over last {self._symbol_pnl_count} trades)")
        return None

    def _check_crypto_winrate_pause(self) -> Optional[str]:
        """Block entries if crypto win rate has dropped below 52% over last 50 trades.

        Returns reason string if paused, None if OK.
        """
        if self.group not in ("crypto", "crypto_intraday"):
            return None
        if self._crypto_wr_pause_until is not None:
            if datetime.now(timezone.utc) < self._crypto_wr_pause_until:
                remaining_h = (self._crypto_wr_pause_until - datetime.now(timezone.utc)).total_seconds() / 3600
                return f"win-rate auto-pause ({remaining_h:.1f}h remaining)"
            else:
                self._crypto_wr_pause_until = None
        return None

    def _update_crypto_guardrails(self, symbol: str, pnl_pct: float) -> None:
        """Update concentration tracker and win-rate pause after a crypto trade closes."""
        if self.group not in ("crypto", "crypto_intraday"):
            return
        sym_key = symbol.replace("/", "-")
        self._symbol_pnl_tracker[sym_key] = self._symbol_pnl_tracker.get(sym_key, 0.0) + pnl_pct
        self._symbol_pnl_count += 1
        # Decay: reset window after 50 trades
        if self._symbol_pnl_count >= 50:
            self._symbol_pnl_tracker.clear()
            self._symbol_pnl_count = 0

        # Win-rate check over last 50 trades using _recent_trade_wins deque
        if len(self._recent_trade_wins) >= 50:
            recent_50 = list(self._recent_trade_wins)[-50:]
            wr = sum(recent_50) / 50
            if wr < 0.52 and self._crypto_wr_pause_until is None:
                self._crypto_wr_pause_until = datetime.now(timezone.utc) + timedelta(hours=48)
                log.warning("Crypto win rate %.1f%% < 52%% over last 50 trades → "
                            "auto-pausing entries for 48h (until %s)",
                            wr * 100, self._crypto_wr_pause_until)

    # -- Volatility tier classification -----------------------------------

    def _classify_vol_tiers(self) -> None:
        """Classify all symbols into volatility tiers and cache exit params.

        Called at startup and recomputed weekly.
        """
        from src.risk_config import (classify_vol_tier, compute_vol_metrics,
                                     get_exit_params, VolTier)
        import yfinance as yf

        yf_symbols = []
        sym_map = {}  # yf_symbol → original symbol
        # Include both static symbols and dynamically selected symbols
        all_syms = set(self.symbols)
        if self._etf_active_symbols:
            all_syms.update(self._etf_active_symbols)
        if self._selector_active_symbols:
            all_syms.update(self._selector_active_symbols)
        for sym in all_syms:
            yf_sym = _crypto_to_yfinance(sym) if _is_crypto_symbol(sym) else sym
            yf_symbols.append(yf_sym)
            sym_map[yf_sym] = sym

        try:
            data = yf.download(yf_symbols, period="60d", progress=False, threads=True)
            for yf_sym, orig_sym in sym_map.items():
                try:
                    if len(yf_symbols) == 1:
                        df = data
                    else:
                        df = data.xs(yf_sym, axis=1, level=1) if hasattr(data.columns, 'levels') else data
                    if len(df) < 15:
                        raise ValueError(f"insufficient data ({len(df)} rows)")
                    atr_ratio, vol20 = compute_vol_metrics(df)
                    tier = classify_vol_tier(atr_ratio, vol20)
                    self._vol_tiers[orig_sym] = tier
                    self._exit_params[orig_sym] = get_exit_params(self.group or "swing", tier)
                    log.info("Vol tier %s: %s (ATR/P=%.2f%%, Vol20=%.1f%%)",
                             orig_sym, tier.value, atr_ratio * 100, vol20 * 100)
                except Exception as exc:
                    log.warning("Vol tier classification failed for %s: %s — defaulting to HIGH",
                                orig_sym, exc)
                    self._vol_tiers[orig_sym] = VolTier.HIGH
                    self._exit_params[orig_sym] = get_exit_params(self.group or "swing", VolTier.HIGH)
        except Exception as exc:
            log.warning("Vol tier bulk download failed: %s — defaulting all to HIGH", exc)
            for sym in self.symbols:
                self._vol_tiers[sym] = VolTier.HIGH
                self._exit_params[sym] = get_exit_params(self.group or "swing", VolTier.HIGH)

        self._vol_tier_last_update = datetime.now(timezone.utc)

    def _maybe_recompute_vol_tiers(self) -> None:
        """Recompute vol tiers daily (was weekly — too slow for regime shifts)."""
        if self._vol_tier_last_update is None:
            self._classify_vol_tiers()
            return
        elapsed = (datetime.now(timezone.utc) - self._vol_tier_last_update).total_seconds()
        if elapsed > 24 * 3600:  # daily
            log.info("Daily vol tier recomputation triggered")
            self._classify_vol_tiers()

    def _classify_vol_tiers_for(self, symbols: List[str]) -> None:
        """Classify vol tiers for a specific set of new symbols (on-the-fly).

        Symbols should be in the format used as dict keys (plain tickers for ETFs).
        For crypto symbols, use _classify_vol_tiers_for_crypto instead.
        """
        from src.risk_config import (classify_vol_tier, compute_vol_metrics,
                                     get_exit_params, VolTier)
        import yfinance as yf

        try:
            data = yf.download(symbols, period="60d", progress=False, threads=True)
            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = data
                    else:
                        df = data.xs(sym, axis=1, level=1) if hasattr(data.columns, 'levels') else data
                    if len(df) < 15:
                        raise ValueError(f"insufficient data ({len(df)} rows)")
                    atr_ratio, vol20 = compute_vol_metrics(df)
                    tier = classify_vol_tier(atr_ratio, vol20)
                    self._vol_tiers[sym] = tier
                    self._exit_params[sym] = get_exit_params(self.group or "swing", tier)
                    log.info("Vol tier %s: %s (ATR/P=%.2f%%, Vol20=%.1f%%)",
                             sym, tier.value, atr_ratio * 100, vol20 * 100)
                except Exception as exc:
                    log.warning("Vol tier failed for %s: %s — defaulting to HIGH", sym, exc)
                    self._vol_tiers[sym] = VolTier.HIGH
                    self._exit_params[sym] = get_exit_params(self.group or "swing", VolTier.HIGH)
        except Exception as exc:
            log.warning("Vol tier download failed for %s: %s — defaulting to HIGH",
                        symbols, exc)
            for sym in symbols:
                self._vol_tiers[sym] = VolTier.HIGH
                self._exit_params[sym] = get_exit_params(self.group or "swing", VolTier.HIGH)

    def _classify_vol_tiers_for_crypto(self, symbols: List[str]) -> None:
        """Classify vol tiers for crypto symbols (Alpaca format: BTC/USD).

        Converts to yfinance format for download, stores results under
        the original Alpaca-format key.
        """
        from src.risk_config import (classify_vol_tier, compute_vol_metrics,
                                     get_exit_params, VolTier)
        import yfinance as yf

        yf_syms = []
        sym_map = {}  # yf_sym → orig_sym
        for sym in symbols:
            yf_sym = _crypto_to_yfinance(sym) if _is_crypto_symbol(sym) else sym
            yf_syms.append(yf_sym)
            sym_map[yf_sym] = sym

        try:
            data = yf.download(yf_syms, period="60d", progress=False, threads=True)
            for yf_sym, orig_sym in sym_map.items():
                try:
                    if len(yf_syms) == 1:
                        df = data
                    else:
                        df = data.xs(yf_sym, axis=1, level=1) if hasattr(data.columns, 'levels') else data
                    if len(df) < 15:
                        raise ValueError(f"insufficient data ({len(df)} rows)")
                    atr_ratio, vol20 = compute_vol_metrics(df)
                    tier = classify_vol_tier(atr_ratio, vol20)
                    self._vol_tiers[orig_sym] = tier
                    self._exit_params[orig_sym] = get_exit_params(self.group or "crypto", tier)
                    log.info("Vol tier %s: %s (ATR/P=%.2f%%, Vol20=%.1f%%)",
                             orig_sym, tier.value, atr_ratio * 100, vol20 * 100)
                except Exception as exc:
                    log.warning("Vol tier failed for %s: %s — defaulting to HIGH", orig_sym, exc)
                    self._vol_tiers[orig_sym] = VolTier.HIGH
                    self._exit_params[orig_sym] = get_exit_params(self.group or "crypto", VolTier.HIGH)
        except Exception as exc:
            log.warning("Vol tier download failed for crypto: %s — defaulting to HIGH", exc)
            for sym in symbols:
                self._vol_tiers[sym] = VolTier.HIGH
                self._exit_params[sym] = get_exit_params(self.group or "crypto", VolTier.HIGH)

    # -- Predictor factory (selects model type per group) ----------------
    @staticmethod
    def _create_predictor(symbol: str, group: Optional[str], model_dir: str,
                          mode: str, intraday_interval: str):
        """Create the right predictor based on account group.

        Tries group-specific model first (LightGBM/PatchTST/XGBoost),
        falls back to LSTM if the group-specific model isn't trained yet.
        """
        # Resolve model directory per group
        group_model_dir = get_model_dir(group) if group else model_dir

        if group == "intraday":
            validate_model_mode("lgb_intraday_etf", mode)
            try:
                from etf_intraday_model import EtfIntradayPredictor
                return EtfIntradayPredictor(symbol, model_dir=group_model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No ETF intraday model for %s, falling back to LSTM: %s", symbol, exc)
        elif group == "swing":
            validate_model_mode("tft_swing", mode)
            try:
                from swing_model import SwingPredictor
                return SwingPredictor(symbol, model_dir=group_model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No swing model for %s, falling back to LSTM: %s", symbol, exc)
        elif group == "crypto":
            validate_model_mode("xgb_swing", mode)
            # Crypto uses swing model with yfinance-mapped symbols (BTC/USD → BTC-USD)
            yf_sym = _crypto_to_yfinance(symbol)
            crypto_dir = group_model_dir
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
        elif group == "crypto_intraday":
            validate_model_mode("lgb_intraday_crypto", mode)
            # Crypto intraday uses LGB+GRU ensemble on 5-min bars
            yf_sym = _crypto_to_yfinance(symbol)
            try:
                from crypto_intraday_model import CryptoIntradayPredictor
                return CryptoIntradayPredictor(yf_sym, model_dir=group_model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.warning("No crypto intraday model for %s (%s): %s — will skip.",
                            symbol, yf_sym, exc)
                return None
        # Fallback: original LSTM predictor
        validate_model_mode("lstm", mode)
        try:
            return Predictor(symbol, model_dir=model_dir,
                             mode=mode, intraday_interval=intraday_interval)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("No trained model for %s (%s): %s — will skip ML signals.",
                        symbol, mode, exc)
            return None

    def _get_model_universe(self) -> Set[str]:
        """Return yfinance-format symbols that have a trained model on disk."""
        import glob as _glob
        if self.group == "crypto_intraday":
            pattern = os.path.join(self._crypto_model_dir, "*_lgb_intraday_crypto_config.json")
            suffix = "_lgb_intraday_crypto_config.json"
        else:
            pattern = os.path.join(self._crypto_model_dir, "*_xgb_swing_config.json")
            suffix = "_xgb_swing_config.json"
        model_syms: Set[str] = set()
        for path in _glob.glob(pattern):
            fname = os.path.basename(path)
            sym = fname.replace(suffix, "")
            model_syms.add(sym)
        return model_syms

    def _load_oos_registries(self) -> tuple:
        """Load OOS Sharpe and performance registries from config/trading.json.

        Crypto intraday uses separate registries (oos_sharpe_registry_intraday,
        oos_performance_registry_intraday) since metrics come from 1-hour holds
        on 5-min bars, not 5-day holds on daily bars.
        Returns (sharpe_dict, performance_dict).
        """
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "trading.json",
            )
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                if self.group == "crypto_intraday":
                    sharpe_reg = cfg.get("oos_sharpe_registry_intraday", {})
                    perf_reg = cfg.get("oos_performance_registry_intraday", {})
                    # No fallback to swing registries — horizon mismatch
                    # (swing = 5-10d holds vs intraday = 1h holds).
                    # Empty registries → cold-start weights (30/70) in composite scoring.
                    if not sharpe_reg:
                        log.info("crypto_intraday: no intraday OOS registry — cold start weights apply")
                else:
                    sharpe_reg = cfg.get("oos_sharpe_registry", {})
                    perf_reg = cfg.get("oos_performance_registry", {})
                return sharpe_reg, perf_reg
        except Exception as exc:
            log.warning("Failed to load OOS registries: %s", exc)
        return {}, {}

    def _compute_btc_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Compute 30-day rolling correlation of each coin's returns with BTC.

        Returns {symbol: correlation} where correlation is in [0, 1].
        BTC-USD itself and ETH-USD (anchor coins) return 0.0 (no penalty).
        """
        btc_df = data.get("BTC-USD")
        if btc_df is None or len(btc_df) < 35:
            return {}

        btc_ret = btc_df["close"].astype(float).pct_change().rename("btc")
        correlations: Dict[str, float] = {}
        for sym, df in data.items():
            if sym in ("BTC-USD", "ETH-USD"):
                correlations[sym] = 0.0  # anchor coins — no penalty
                continue
            try:
                sym_ret = df["close"].astype(float).pct_change().rename("alt")
                # Align on common dates
                aligned = pd.concat([btc_ret, sym_ret], axis=1, join="inner")
                if len(aligned) < 30:
                    correlations[sym] = 0.5  # insufficient data → moderate default
                    continue
                corr = aligned.iloc[-30:]["btc"].corr(aligned.iloc[-30:]["alt"])
                correlations[sym] = max(0.0, corr) if not np.isnan(corr) else 0.5
            except Exception:
                correlations[sym] = 0.5
        return correlations

    def _compute_spy_correlations(self, symbols: List[str]) -> Dict[str, float]:
        """Compute 30-day rolling correlation of each ETF's returns with SPY.

        Returns {symbol: correlation} where correlation is in [0, 1].
        SPY itself returns 0.0 (no self-penalty).
        """
        import yfinance as yf

        try:
            fetch_syms = list(set(symbols + ["SPY"]))
            data = yf.download(fetch_syms, period="60d", progress=False, auto_adjust=True)
            if data.empty:
                return {}

            multi = hasattr(data.columns, "get_level_values")
            if multi:
                close_df = data["Close"]
            else:
                close_df = data[["Close"]]

            spy_ret = close_df["SPY"].pct_change().dropna() if "SPY" in close_df.columns else None
            if spy_ret is None or len(spy_ret) < 35:
                return {}

            correlations: Dict[str, float] = {}
            for sym in symbols:
                if sym == "SPY":
                    correlations[sym] = 0.0
                    continue
                try:
                    sym_close = close_df[sym] if sym in close_df.columns else None
                    if sym_close is None:
                        correlations[sym] = 0.5
                        continue
                    sym_ret = sym_close.pct_change().dropna()
                    aligned = pd.concat([spy_ret.rename("spy"), sym_ret.rename("sym")],
                                        axis=1, join="inner")
                    if len(aligned) < 30:
                        correlations[sym] = 0.5
                        continue
                    corr = aligned.iloc[-30:]["spy"].corr(aligned.iloc[-30:]["sym"])
                    correlations[sym] = max(0.0, corr) if not np.isnan(corr) else 0.5
                except Exception:
                    correlations[sym] = 0.5
            return correlations
        except Exception as exc:
            log.warning("SPY correlation computation failed: %s", exc)
            return {}

    @staticmethod
    def _compute_etf_fee_costs(symbols: List[str]) -> Dict[str, float]:
        """Get round-trip cost (as fraction) for each ETF from cost_model."""
        from cost_model import get_symbol_costs
        costs: Dict[str, float] = {}
        for sym in symbols:
            try:
                sc = get_symbol_costs(sym, session="regular")
                costs[sym] = sc.round_trip_pct
            except Exception:
                costs[sym] = 0.001  # conservative default
        return costs

    @staticmethod
    def _compute_quick_scores(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Vol-adjusted momentum × liquidity score for pre-screening / training priority.

        quick_score = (ret_5 / realized_vol_5d) × log(1 + avg_daily_volume_usd)

        Higher = stronger signal quality on a liquid coin. Used for:
        - On-the-fly training priority (cold-start: no model yet)
        - Pre-screen ranking when composite score unavailable
        """
        scores: Dict[str, float] = {}
        for sym, df in data.items():
            try:
                close = df["close"].astype(float)
                volume = df["volume"].astype(float)
                if len(close) < 10:
                    continue
                ret_5 = float(close.iloc[-1] / close.iloc[-6] - 1) if (len(close) >= 6 and close.iloc[-6] > 0) else 0.0
                vol_5d = float(close.pct_change().iloc[-5:].std()) if len(close) >= 6 else 1.0
                vol_5d = max(vol_5d, 0.001)  # floor to avoid div/0
                avg_vol_usd = float((close * volume).iloc[-20:].mean()) if len(close) >= 20 else 1.0
                scores[sym] = (ret_5 / vol_5d) * np.log1p(avg_vol_usd)
            except Exception:
                continue
        return scores

    def _compute_composite_scores(
        self,
        rankings: list,
        oos_sharpe: dict,
        oos_perf: dict,
        btc_correlations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Composite score = w1*selector + w2*blended_sharpe (rank-normalized).

        Delegates to shared ``oos_feedback.compute_composite_scores()``
        with a derisk_lookup closure that reads from self._derisk_states.
        """
        return _composite_scores_shared(
            rankings=rankings,
            oos_sharpe=oos_sharpe,
            derisk_lookup=self._derisk_lookup,
            btc_correlations=btc_correlations,
        )

    def _derisk_lookup(self, symbol: str) -> Optional[tuple]:
        """Return (n_trades, live_rolling_sharpe) for a symbol, or None."""
        alpaca_sym = symbol.replace("-", "/")
        derisk = self._derisk_states.get(alpaca_sym) or self._derisk_states.get(symbol)
        if derisk is None:
            return None
        n_trades = len(derisk.returns)
        live_sharpe = derisk.rolling_sharpe(window=60)
        return (n_trades, live_sharpe)

    # -- Coin ranker (Layer 1 → rank ALL, Layer 2 → ML decides) -----------
    def _run_coin_selector(self) -> List[str]:
        """Rank-first crypto selection: ranking decides who trades, not model existence.

        Pipeline:
          1. Rank the full universe via LambdaRank + composite scoring
          2. Take top-N ranked coins (N = max_positions)
          3. Only those top-N with trained models become active
          4. check_and_trade() + risk limits decide which actually open

        Falls back to static self.symbols if ranker unavailable or fails.
        """
        if self._coin_selector is None:
            self._ensure_coin_selector()
        if self._coin_selector is None:
            return list(self.symbols)

        try:
            # Data freshness check: BTC canary
            # Crypto markets trade 24/7 — yfinance daily bars may lag 1-2 days
            # on weekends (no equity market close to trigger bar). Only skip
            # ranking if data is truly stale (>3 days for crypto, >1 day for equities).
            if self._selector_mode != "intraday":
                try:
                    import yfinance as yf
                    _btc = yf.download("BTC-USD", period="5d", progress=False, auto_adjust=True)
                    if not _btc.empty:
                        _latest = pd.Timestamp(_btc.index[-1]).date()
                        _days_stale = (datetime.now(timezone.utc).date() - _latest).days
                        # Crypto 24/7: tolerate up to 3 days stale (weekend lag)
                        _stale_threshold = 3 if self.group in ("crypto", "crypto_intraday") else 1
                        if _days_stale > _stale_threshold:
                            log.warning(
                                "Coin selector skipped — bars stale (latest BTC bar: %s, %dd old, threshold=%dd)",
                                _latest, _days_stale, _stale_threshold,
                            )
                            if self._selector_active_symbols:
                                return self._selector_active_symbols
                            return list(self.symbols)
                except Exception as exc:
                    log.warning("Crypto freshness check failed: %s — proceeding anyway", exc)

            # Step 1: Rank the FULL universe — ranking is king
            rank_input = list(self._selector_universe) if self._selector_universe else []
            if len(rank_input) < 3:
                log.warning("Selector universe too small (%d coins) — using static list",
                            len(rank_input))
                self._composite_scores = {}
                return list(self.symbols)

            log.info("Ranker: scoring %d universe coins (mode=%s)",
                     len(rank_input), self._selector_mode)

            # Fetch data for ranking
            if self._selector_mode == "intraday":
                data = fetch_universe_intraday_data(
                    rank_input, lookback_days=INTRADAY_LOOKBACK_DAYS)
            else:
                data = fetch_universe_data(rank_input, lookback_days=400)
            if len(data) < 3:
                log.warning("Ranker: only %d coins with data — falling back to static list", len(data))
                self._composite_scores = {}
                return list(self.symbols)

            result = self._coin_selector.rank(data)
            if not result.rankings:
                log.warning("Ranker returned empty rankings — using static list")
                self._composite_scores = {}
                return list(self.symbols)

            # BTC correlation penalty
            btc_correlations = self._compute_btc_correlations(data)
            self._btc_correlations = {
                s.replace("-", "/"): c for s, c in btc_correlations.items()
            } if btc_correlations else {}
            if btc_correlations:
                top_corr = sorted(btc_correlations.items(), key=lambda x: -x[1])[:5]
                log.info("BTC correlations (top-5): %s",
                         ", ".join(f"{s}={c:.2f}" for s, c in top_corr))

            # Composite scoring
            oos_sharpe, oos_perf = self._load_oos_registries()
            composite = self._compute_composite_scores(
                result.rankings, oos_sharpe, oos_perf,
                btc_correlations=btc_correlations,
            )
            if not composite:
                composite = {sym: score for sym, score in result.rankings}

            # TFT confidence boost
            _TFT_CONF_WEIGHT = 0.10
            _vix = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
            for sym in list(composite.keys()):
                alpaca_sym = sym.replace("-", "/")
                predictor = self.predictors.get(alpaca_sym)
                if predictor is None:
                    continue
                try:
                    if sym in data:
                        pred = predictor.predict(data[sym], _vix)
                        if pred and "confidence" in pred:
                            conf = min(1.0, abs(float(pred["confidence"])))
                            composite[sym] += _TFT_CONF_WEIGHT * conf
                except Exception:
                    pass

            # Sort by composite score, best-first
            composite = dict(sorted(composite.items(), key=lambda x: -x[1]))
            all_ranked = list(composite.keys())

            # Log full ranking
            for i, sym in enumerate(all_ranked, 1):
                sel_score = dict(result.rankings).get(sym, 0)
                oos_s = oos_sharpe.get(sym, 0)
                blended_s = self._blended_sharpe(sym, oos_s)
                log.info("  Rank #%d %s: composite=%.3f (sel=%.2f, oos_sharpe=%.2f, blended=%.2f)",
                         i, sym, composite[sym], sel_score, oos_s, blended_s)

            # Step 2: Take top-K ranked as CANDIDATES (K ≥ 12)
            # The candidate pool is larger than max_positions so that
            # the gate/sizing stage has enough symbols to choose from.
            # Actual open positions are still limited by max_positions.
            _MIN_CANDIDATE_K = 12
            top_k = max(_MIN_CANDIDATE_K, self._risk_config.max_positions)
            top_n = all_ranked[:top_k]
            log.info("Ranker: top-%d candidates: %s", top_k, ", ".join(top_n))

            # Store composite scores for rank-weighted sizing
            self._composite_scores = composite

            # Step 3: Filter to top-K that have models + working predictors.
            # If a top-K symbol has no model, trigger on-the-fly training.
            model_universe = self._get_model_universe()
            top_n_alpaca = [s.replace("-", "/") for s in top_n]
            _selector_model_dir = get_model_dir(self.group)
            ready = []
            _train_triggered = 0
            _MAX_TRAIN_PER_CYCLE = 2  # limit on-the-fly training to avoid stalling
            for sym_yf, sym in zip(top_n, top_n_alpaca):
                has_model = (sym_yf in model_universe
                             or sym_yf.replace("/", "-") in model_universe)

                if not has_model:
                    # Trigger on-the-fly training if under limit
                    if _train_triggered < _MAX_TRAIN_PER_CYCLE:
                        log.info("Ranker: %s ranked top-%d but no model — triggering training",
                                 sym, top_k)
                        trained = self._train_on_the_fly(sym)
                        _train_triggered += 1
                        if trained:
                            has_model = True
                        else:
                            log.info("Ranker: %s training failed or cooldown — skipped", sym)
                            continue
                    else:
                        log.info("Ranker: %s no model, training limit reached — skipped", sym)
                        continue

                if sym not in self.predictors or self.predictors[sym] is None:
                    predictor = self._create_predictor(
                        sym, self.group, _selector_model_dir,
                        self.mode, self.intraday_interval,
                    )
                    self.predictors[sym] = predictor
                    if predictor is not None:
                        log.info("Loaded predictor for %s", sym)
                    else:
                        log.warning("No predictor for %s — skipping", sym)
                        continue
                ready.append(sym)

            # Classify vol tiers for any new symbols
            new_crypto = [s for s in ready if s not in self._vol_tiers]
            if new_crypto:
                self._classify_vol_tiers_for_crypto(new_crypto)

            log.info("Ranker: %d coins active (top-%d candidates, %d trained on-the-fly): %s",
                     len(ready), top_k, _train_triggered, ", ".join(ready))
            self._selector_active_symbols = ready

            # Sync full ranked list (not just ready) to Gist for dashboard
            self._sync_rankings_to_gist(all_ranked)

            return self._selector_active_symbols

        except Exception as exc:
            log.error("Coin ranker failed: %s — using static list", exc)
            self._composite_scores = {}
            return list(self.symbols)

    # -- ETF selector (Layer 1 → dynamic symbol selection for swing/intraday) --

    def _get_etf_model_universe(self, group: str) -> Set[str]:
        """Return symbols that have a trained model on disk for the given group."""
        import glob as _glob
        if group == "swing":
            pattern = os.path.join(SWING_MODEL_DIR, "*_xgb_swing_config.json")
            suffix = "_xgb_swing_config.json"
        elif group == "intraday":
            pattern = os.path.join(INTRADAY_MODEL_DIR, "*_lgb_intraday_config.json")
            suffix = "_lgb_intraday_config.json"
        else:
            return set()
        model_syms: Set[str] = set()
        for path in _glob.glob(pattern):
            fname = os.path.basename(path)
            sym = fname.replace(suffix, "")
            model_syms.add(sym)
        return model_syms

    def _train_on_the_fly(self, symbol: str) -> bool:
        """Train a model on-the-fly for a newly selected ETF.

        Returns True if training succeeded and predictor is ready.
        Rate-limited: skips symbols that failed training in last 24h.
        """
        # Check cooldown for failed training attempts
        last_fail = self._etf_train_failures.get(symbol)
        if last_fail:
            elapsed_h = (datetime.now(timezone.utc) - last_fail).total_seconds() / 3600
            if elapsed_h < self._etf_train_cooldown_hours:
                log.info("Skipping on-the-fly training for %s (failed %.1fh ago, cooldown %dh)",
                         symbol, elapsed_h, self._etf_train_cooldown_hours)
                return False

        group = self.group
        log.info("=== On-the-fly training: %s for %s group ===", symbol, group)

        try:
            adapter = build_adapter("yahoo")
            fred_key = os.environ.get("FRED_API_KEY")

            if group == "swing":
                from swing_model import train_swing_model
                model = train_swing_model(
                    symbol=symbol,
                    adapter=adapter,
                    fred_key=fred_key,
                    lookback=1000,
                    save_dir=SWING_MODEL_DIR,
                    train_recent=True,  # walk-forward: 75% train, 25% OOS
                )
                if model is None:
                    raise RuntimeError(f"train_swing_model returned None for {symbol}")

            elif group == "intraday":
                from intraday_model import train_intraday_model
                model = train_intraday_model(
                    symbol=symbol,
                    adapter=adapter,
                    fred_key=fred_key,
                    lookback=500,
                    save_dir=INTRADAY_MODEL_DIR,
                )
                if model is None:
                    raise RuntimeError(f"train_intraday_model returned None for {symbol}")

            elif group == "crypto":
                from swing_model import train_swing_model
                # Crypto daily uses same XGBoost swing pipeline with crypto model dir
                model = train_swing_model(
                    symbol=symbol,
                    adapter=adapter,
                    fred_key=fred_key,
                    lookback=1000,
                    save_dir=CRYPTO_MODEL_DIR,
                    train_recent=True,
                )
                if model is None:
                    raise RuntimeError(f"train_swing_model (crypto) returned None for {symbol}")
            else:
                return False

            # Load the predictor for this newly trained model
            group_model_dir = get_model_dir(group)
            predictor = self._create_predictor(
                symbol, group, group_model_dir, self.mode, self.intraday_interval,
            )
            if predictor is not None:
                self.predictors[symbol] = predictor
                log.info("On-the-fly training succeeded for %s — predictor loaded", symbol)
                # Clear any previous failure
                self._etf_train_failures.pop(symbol, None)
                return True
            else:
                raise RuntimeError(f"Predictor creation failed after training {symbol}")

        except Exception as exc:
            log.error("On-the-fly training failed for %s: %s", symbol, exc)
            self._etf_train_failures[symbol] = datetime.now(timezone.utc)
            return False

    def _spawn_background_train(self, symbol: str, group: str) -> bool:
        """Spawn a background subprocess to train a model. Non-blocking.

        Returns True if subprocess was spawned (or already running).
        The symbol becomes available on the NEXT selector cycle after
        the subprocess completes and writes model files to disk.
        """
        import subprocess as _sp

        # Already running?
        if symbol in self._bg_train_procs:
            proc = self._bg_train_procs[symbol]
            if proc.poll() is None:
                return True  # still running
            # Completed — reap it
            self._bg_train_procs.pop(symbol)
            self._bg_train_started.pop(symbol, None)

        # Check cooldown
        last_fail = self._etf_train_failures.get(symbol)
        if last_fail:
            elapsed_h = (datetime.now(timezone.utc) - last_fail).total_seconds() / 3600
            if elapsed_h < self._etf_train_cooldown_hours:
                return False

        # Concurrent limit
        active = sum(1 for p in self._bg_train_procs.values() if p.poll() is None)
        if active >= self._bg_max_concurrent:
            log.info("Background train cap reached (%d running) — deferring %s",
                     active, symbol)
            return False

        # Build command
        python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(PROJECT_ROOT, "main.py")
        if group == "swing":
            cmd = [python_exe, main_py, "train-swing", "--symbols", symbol,
                   "--provider", "yahoo"]
        elif group == "intraday":
            cmd = [python_exe, main_py, "train-intraday", "--symbols", symbol,
                   "--provider", "yahoo"]
        elif group == "crypto":
            cmd = [python_exe, main_py, "train-swing", "--symbols", symbol,
                   "--provider", "yahoo", "--save-dir", CRYPTO_MODEL_DIR]
        elif group == "crypto_intraday":
            cmd = [python_exe, main_py, "train-intraday", "--symbols", symbol,
                   "--provider", "yahoo", "--save-dir", CRYPTO_MODEL_DIR]
        else:
            return False

        log.info("Spawning background train for %s (%s): PID pending...", symbol, group)
        try:
            proc = _sp.Popen(
                cmd,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
                cwd=PROJECT_ROOT,
            )
            self._bg_train_procs[symbol] = proc
            self._bg_train_started[symbol] = datetime.now(timezone.utc)
            log.info("Background train for %s started (PID %d)", symbol, proc.pid)
            return True
        except Exception as exc:
            log.error("Failed to spawn background train for %s: %s", symbol, exc)
            self._etf_train_failures[symbol] = datetime.now(timezone.utc)
            return False

    def _reap_background_trains(self) -> set:
        """Check all background training subprocesses. Return newly trained symbols."""
        newly_trained: set = set()
        for symbol in list(self._bg_train_procs.keys()):
            proc = self._bg_train_procs[symbol]
            if proc.poll() is None:
                # Check timeout
                started = self._bg_train_started.get(symbol)
                if (started and
                        (datetime.now(timezone.utc) - started).total_seconds()
                        > self._bg_train_timeout):
                    log.warning("Background train for %s timed out — killing PID %d",
                                symbol, proc.pid)
                    proc.kill()
                    self._bg_train_procs.pop(symbol)
                    self._bg_train_started.pop(symbol, None)
                    self._etf_train_failures[symbol] = datetime.now(timezone.utc)
                continue
            # Process finished
            if proc.returncode == 0:
                log.info("Background train for %s completed (PID %d)", symbol, proc.pid)
                newly_trained.add(symbol)
                self._etf_train_failures.pop(symbol, None)
            else:
                log.warning("Background train for %s failed (PID %d, rc=%d)",
                            symbol, proc.pid, proc.returncode)
                self._etf_train_failures[symbol] = datetime.now(timezone.utc)
            self._bg_train_procs.pop(symbol)
            self._bg_train_started.pop(symbol, None)
        return newly_trained

    def _run_etf_selector(self) -> List[str]:
        """Rank-first ETF selection: ranking decides who trades, not model existence.

        Pipeline:
          1. Rank the full universe → determines priority
          2. Take top-N ranked symbols (N = max_positions)
          3. If top-N symbols lack trained models, spawn background training
          4. Return only top-N that have models (pre-trained or just-trained)
          5. check_and_trade() + risk limits decide which actually open

        Falls back to static SYMBOL_GROUPS if selector unavailable or fails.
        """
        if self._etf_selector is None:
            return list(self.symbols)

        group = self.group or "swing"

        try:
            # Data freshness check: verify daily bars are current before ranking.
            try:
                import yfinance as yf
                _spy = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
                if not _spy.empty:
                    _latest_bar = pd.Timestamp(_spy.index[-1]).date()
                    _today = datetime.now(timezone.utc).date()
                    _days_stale = (_today - _latest_bar).days
                    if _days_stale > 2:
                        log.warning(
                            "ETF selector skipped — bars stale (latest SPY bar: %s, %dd old)",
                            _latest_bar, _days_stale,
                        )
                        if self._etf_active_symbols:
                            return self._etf_active_symbols
                        return list(self.symbols)
            except Exception as exc:
                log.warning("ETF freshness check failed: %s — proceeding anyway", exc)

            universe = self._etf_universe
            if not universe:
                log.warning("ETF universe empty — using static list")
                return list(self.symbols)

            # Step 1: Rank the full universe FIRST — ranking is king
            log.info("ETF ranker: scoring %d universe symbols for %s...", len(universe), group)
            ranking = self._etf_selector.rank(universe)

            if ranking.empty:
                log.warning("ETF ranker: no ranking data — using static list")
                self._composite_scores = {}
                return list(self.symbols)

            ranked_symbols = ranking["symbol"].tolist()

            # Step 1b: Blend selector scores with OOS Sharpe (composite scoring)
            # Load OOS metrics from promoted_symbols.json for this group
            _group_model_dir = get_model_dir(group)
            oos_metrics = load_promoted_oos(_group_model_dir)
            oos_sharpe_map = {sym: d.get("sharpe_ratio", 0.0) for sym, d in oos_metrics.items()}

            if oos_sharpe_map:
                # Build (symbol, selector_score) tuples for composite scoring
                sel_rankings = list(zip(ranking["symbol"], ranking["score"]))

                # SPY correlation penalty (matches BTC penalty for crypto)
                _top30 = [s for s, _ in sel_rankings[:30]]
                spy_correlations = self._compute_spy_correlations(_top30)
                if spy_correlations:
                    _top_corr = sorted(spy_correlations.items(), key=lambda x: -x[1])[:5]
                    log.info("SPY correlations (top-5): %s",
                             ", ".join(f"{s}={c:.2f}" for s, c in _top_corr))

                # Fee-aware cost adjustment
                fee_costs = self._compute_etf_fee_costs(_top30)

                self._composite_scores = _composite_scores_shared(
                    rankings=sel_rankings,
                    oos_sharpe=oos_sharpe_map,
                    derisk_lookup=self._derisk_lookup,
                    btc_correlations=spy_correlations,
                    fee_costs=fee_costs,
                )
                # Re-sort by composite score
                ranked_symbols = sorted(
                    self._composite_scores.keys(),
                    key=lambda s: -self._composite_scores[s],
                )
                log.info("ETF ranker: composite scoring applied (%d OOS, SPY penalty, fee-aware)",
                         len(oos_sharpe_map))
            else:
                # No OOS data — use raw selector scores
                self._composite_scores = dict(
                    zip(ranking["symbol"], ranking["score"])
                )
                log.info("ETF ranker: no OOS data — using raw selector scores")

            # Step 2: Take top-N ranked symbols — evaluate 2x max_positions for a
            # deep bench (risk limits decide which actually open, not the candidate count)
            max_active = self._risk_config.max_positions * 2
            top_n = ranked_symbols[:max_active]
            log.info("ETF ranker: top-%d candidates (max_pos=%d): %s",
                     max_active, self._risk_config.max_positions, ", ".join(top_n))

            # Step 3: Reap completed background trains, then spawn new ones
            newly_trained = self._reap_background_trains()
            model_universe = self._get_etf_model_universe(group)
            if newly_trained:
                log.info("ETF ranker: %d symbols trained in background: %s",
                         len(newly_trained), ", ".join(newly_trained))
            log.info("ETF ranker: %d trained models on disk", len(model_universe))

            untrained_top = [s for s in top_n if s not in model_universe]
            if untrained_top:
                log.info("ETF ranker: %d top-ranked symbols need training: %s",
                         len(untrained_top), ", ".join(untrained_top))
                for sym in untrained_top:
                    self._spawn_background_train(sym, group)

            # Step 4: Build final list — only top-N that have models
            group_model_dir = get_model_dir(group)
            final = []
            for sym in top_n:
                if sym not in model_universe:
                    log.info("ETF ranker: %s ranked #%d but no model — skipped",
                             sym, top_n.index(sym) + 1)
                    continue
                if sym not in self.predictors or self.predictors[sym] is None:
                    predictor = self._create_predictor(
                        sym, group, group_model_dir, self.mode, self.intraday_interval,
                    )
                    self.predictors[sym] = predictor
                    if predictor:
                        log.info("Loaded predictor for %s", sym)
                    else:
                        log.warning("Predictor failed for %s — skipping", sym)
                        continue
                final.append(sym)

            if not final:
                log.warning("No ready symbols after ranking — using static list")
                return list(self.symbols)

            # Classify vol tiers for any new symbols
            new_syms = [s for s in final if s not in self._vol_tiers]
            if new_syms:
                self._classify_vol_tiers_for(new_syms)

            log.info("ETF ranker: %d active symbols for %s: %s",
                     len(final), group, ", ".join(final))
            self._etf_active_symbols = final

            # Sync full ranked list (not just final) to Gist for dashboard
            self._sync_rankings_to_gist(ranked_symbols)

            return final

        except Exception as exc:
            log.error("ETF ranker failed: %s — using static list", exc)
            return list(self.symbols)

    # -- Sync selector rankings to Gist for dashboard consumption --

    # Map paper_trader group names → dashboard group keys
    _DASHBOARD_GROUP_MAP = {
        "intraday": "etf_intraday",
        "swing": "etf_swing",
        "crypto": "crypto",
        "crypto_intraday": "crypto_intraday",
    }

    def _sync_rankings_to_gist(self, ranked_symbols: List[str]) -> None:
        """Write current group's ranked symbols to selector_rankings.json in the Gist.

        Read-modify-write: reads the existing file from Gist, updates this
        group's entry, and writes back.  All 4 groups share one JSON file.
        """
        gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        if not gist_id or not gh_token:
            gist_id, gh_token = KrakenExecutor._load_gist_env_fallback(gist_id, gh_token)
        if not gist_id or not gh_token:
            return

        dashboard_key = self._DASHBOARD_GROUP_MAP.get(self.group, self.group)

        # Normalize symbols for dashboard display (strip /USD, -USD suffixes)
        display_symbols = []
        for s in ranked_symbols:
            clean = s.replace("/USD", "").replace("-USD", "").replace("/", "").replace("-", "")
            display_symbols.append(clean)

        try:
            import requests
            headers = {
                "Authorization": f"token {gh_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            # Read existing rankings from Gist
            existing = {}
            resp = requests.get(
                f"https://api.github.com/gists/{gist_id}",
                headers=headers,
                timeout=10,
            )
            if resp.ok:
                content = resp.json().get("files", {}).get(
                    "selector_rankings.json", {}
                ).get("content", "{}")
                existing = json.loads(content) if content else {}

            # Update this group's entry
            existing[dashboard_key] = display_symbols

            # Write back
            resp = requests.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers=headers,
                json={"files": {
                    "selector_rankings.json": {
                        "content": json.dumps(existing, indent=2),
                    }
                }},
                timeout=10,
            )
            if resp.ok:
                log.info("Synced %s rankings to Gist (%d symbols)",
                         dashboard_key, len(display_symbols))
            else:
                log.warning("Rankings Gist sync failed (%d): %s",
                            resp.status_code, resp.text[:200])
        except Exception as exc:
            log.warning("Rankings Gist sync error: %s", exc)



    # -- Account info --------------------------------------------------
    def get_account_summary(self) -> dict:
        if self._kraken is not None:
            return self._kraken.get_account_summary()
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions as {symbol: {qty, side, entry, current, pnl, pnl_pct}}."""
        if self._kraken is not None:
            return self._kraken.get_positions()
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

    # -- Order helpers --------------------------------------------------
    def _cancel_open_orders_for(self, symbol: str) -> int:
        """Cancel all open orders for *symbol*. Returns count cancelled."""
        if self._kraken is not None and _is_crypto_symbol(symbol):
            return 0  # Kraken paper fills instantly, no pending orders
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=50)
            )
            for order in open_orders:
                try:
                    self.trading_client.cancel_order_by_id(str(order.id))
                except Exception:
                    pass
            if open_orders:
                log.info("Cancelled %d open order(s) for %s", len(open_orders), symbol)
            return len(open_orders)
        except Exception as exc:
            log.debug("Failed to cancel orders for %s: %s", symbol, exc)
            return 0

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
        Closed → equity orders are deferred (logged but not submitted) to prevent
                 stuck orders that block future attempts.
        """
        is_crypto = _is_crypto_symbol(symbol)
        session = _get_session()

        # Block equity orders when market is closed (weekends, overnight).
        # Submitting during closed hours creates orders that sit in ACCEPTED
        # state, hold the qty, and block all future sell attempts until cancelled.
        if not is_crypto and session == "closed":
            log.info("DEFERRED %s %s x%s — market closed, will retry next session",
                     side.value, symbol, qty)
            return None
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
                    lp = math.ceil(limit_price * 1.001 * 100) / 100
                else:
                    lp = math.floor(limit_price * 0.999 * 100) / 100
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

    def buy(self, symbol: str, qty: float,
            limit_price: Optional[float] = None) -> Optional[str]:
        """Open LONG. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        if self._kraken is not None and _is_crypto_symbol(symbol):
            return self._kraken.buy(symbol, qty)
        return self._submit_order_request(symbol, qty, OrderSide.BUY, limit_price)

    def sell(self, symbol: str, qty: float, reason: str = "",
             limit_price: Optional[float] = None) -> Optional[str]:
        """Close LONG. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        if self._kraken is not None and _is_crypto_symbol(symbol):
            return self._kraken.sell(symbol, qty, reason)
        oid = self._submit_order_request(symbol, qty, OrderSide.SELL, limit_price)
        if oid:
            log.info("SELL reason: %s", reason)
        return oid

    def sell_short(self, symbol: str, qty: float,
                   limit_price: Optional[float] = None) -> Optional[str]:
        """Open SHORT. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        if self._kraken is not None and _is_crypto_symbol(symbol):
            return self._kraken.sell_short(symbol, qty)
        return self._submit_order_request(symbol, qty, OrderSide.SELL, limit_price)

    def buy_to_cover(self, symbol: str, qty: float, reason: str = "",
                     limit_price: Optional[float] = None) -> Optional[str]:
        """Close SHORT. Uses limit order during extended hours."""
        if qty <= 0:
            return None
        if self._kraken is not None and _is_crypto_symbol(symbol):
            return self._kraken.buy_to_cover(symbol, qty, reason)
        oid = self._submit_order_request(symbol, qty, OrderSide.BUY, limit_price)
        if oid:
            log.info("COVER reason: %s", reason)
        return oid

    # -- Regime filter --------------------------------------------------
    def _check_regime(self) -> tuple:
        """Return (regime_ok: bool, reason: str).

        By default, hardcoded SMA/VIX gates are DISABLED — the model's own
        regime features (btc_sma200_flag, spy_trend_strength, vix_regime, etc.)
        handle regime-dependent behavior.  Set ``use_hardcoded_regime_gates``
        in RiskConfig to re-enable the old gates for A/B testing.

        Gate that always remains active:
          Rolling win rate: pause 7 days if last 20 trades win rate < 50%.
        """
        risk = self._risk_config

        if risk.use_hardcoded_regime_gates:
            # --- Legacy hardcoded gates (disabled by default) ---
            if self.group == "crypto":
                try:
                    btc_bars = self.adapter.fetch_daily("BTC-USD", 210)
                    btc_close = float(btc_bars["close"].iloc[-1])
                    btc_sma20 = float(btc_bars["close"].rolling(20).mean().iloc[-1])
                    btc_sma50 = float(btc_bars["close"].rolling(50).mean().iloc[-1])
                    btc_ret14 = float(btc_bars["close"].pct_change(14).iloc[-1])

                    if btc_close > btc_sma20:
                        pass
                    elif btc_close > btc_sma50 and btc_ret14 > 0:
                        pass
                    elif btc_close > btc_sma50:
                        return False, (f"BTC between SMA(20)/SMA(50) but negative momentum: "
                                       f"{btc_close:,.0f}, SMA20={btc_sma20:,.0f}, "
                                       f"SMA50={btc_sma50:,.0f}, ret14={btc_ret14:+.2%}")
                    else:
                        if btc_ret14 > 0.05:
                            pass
                        else:
                            return False, (f"BTC below SMA(50): {btc_close:,.0f} <= "
                                           f"{btc_sma50:,.0f}, ret14={btc_ret14:+.2%}")
                except Exception as exc:
                    log.warning("Regime BTC check failed (defaulting open): %s", exc)
            else:
                try:
                    spy_bars = self.adapter.fetch_daily("SPY", 210)
                    spy_close = float(spy_bars["close"].iloc[-1])
                    spy_sma200 = float(spy_bars["close"].rolling(200).mean().iloc[-1])
                    if spy_close <= spy_sma200:
                        return False, f"SPY below SMA(200): {spy_close:.2f} <= {spy_sma200:.2f}"
                except Exception as exc:
                    log.warning("Regime SPY check failed (defaulting open): %s", exc)

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
        else:
            log.debug("Hardcoded regime gates disabled — model features drive entry decisions.")

        # Rolling 20-trade win rate cooldown (always active)
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
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=vix_days,
                                                include_live=False)
            if len(vix_df) >= 2:
                self._vix_cache = vix_df   # update cache on success
            return vix_df
        except Exception as exc:
            log.warning("VIX fetch failed, using cached value: %s", exc)
            return self._vix_cache if self._vix_cache is not None else pd.DataFrame()

    def _get_current_vix(self) -> Optional[float]:
        """Get latest VIX value from cache."""
        vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
        if vix_df is None or vix_df.empty:
            return None
        try:
            if vix_df.shape[1] == 1:
                return float(vix_df.iloc[-1].iloc[0])
            # Try common column names
            for col in ("vix_close", "vix", "Close"):
                if col in vix_df.columns:
                    return float(vix_df[col].iloc[-1])
            return float(vix_df.iloc[-1].iloc[0])
        except Exception:
            return None

    def get_prediction(self, symbol: str) -> dict:
        """Get ML prediction for a symbol."""
        predictor = self.predictors.get(symbol)
        if predictor is None:
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

        try:
            # Crypto intraday: fetch 5-min bars from Kraken (not Yahoo)
            if self.group == "crypto_intraday":
                return self._get_crypto_intraday_prediction(symbol, predictor)

            ds = self._data_symbol(symbol)
            if self.mode == "intraday":
                # 5 days ensures Friday bars are present so FRED VIX (1-day lag)
                # can ffill into today's intraday bars on Mondays / after holidays.
                bars = self.adapter.fetch_intraday(
                    ds, self.intraday_interval, lookback_days=5)
            else:
                # 400 days needed: GLD/SLV use frac_diff_close (~282-bar warmup)
                bars = self.adapter.fetch_daily(ds, 400)

            # EtfIntradayPredictor expects (bars, spy_bars) — it fetches VIX internally.
            # Other predictors (LSTM, SwingPredictor) expect (bars, vix_df).
            from etf_intraday_model import EtfIntradayPredictor
            if isinstance(predictor, EtfIntradayPredictor):
                spy_bars = self._get_spy_intraday_bars()
                result = predictor.predict(bars, spy_bars=spy_bars)
                if result is None:
                    return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}
                return result
            else:
                vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
                return predictor.predict(bars, vix_df)
        except Exception as exc:
            log.error("Prediction failed for %s: %s", symbol, exc)
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

    def _get_spy_intraday_bars(self) -> Optional[pd.DataFrame]:
        """Fetch SPY 5-min bars for EtfIntradayPredictor's sector relative strength feature."""
        if hasattr(self, '_spy_intraday_cache'):
            cache_ts, cache_bars = self._spy_intraday_cache
            # Reuse cache if less than 5 minutes old
            if (datetime.now(timezone.utc) - cache_ts).total_seconds() < 300:
                return cache_bars
        try:
            spy_bars = self.adapter.fetch_intraday("SPY", self.intraday_interval, lookback_days=5)
            if spy_bars is not None and not spy_bars.empty:
                self._spy_intraday_cache = (datetime.now(timezone.utc), spy_bars)
                return spy_bars
        except Exception as exc:
            log.warning("SPY intraday bars fetch failed: %s", exc)
        return None

    def _get_crypto_intraday_prediction(self, symbol: str, predictor) -> dict:
        """Fetch 5-min bars from Kraken and run CryptoIntradayPredictor.

        Increments the bar counter only when a genuinely new 5-min bar arrives
        (timestamp-based dedup, not call-based — check_interval is 1 min).
        """
        # Timestamp-based bar counting: only increment when a new 5-min bar settles
        _now = datetime.now(timezone.utc)
        _last = self._ci_last_bar_ts.get(symbol)
        _bar_secs = 300  # 5-min bar
        if _last is None or (_now - _last).total_seconds() >= _bar_secs:
            self._ci_bars_held[symbol] = self._ci_bars_held.get(symbol, 0) + 1
            self._ci_last_bar_ts[symbol] = _now

        if self._crypto_intraday_data is None:
            from crypto_intraday_data import CryptoIntradayData
            self._crypto_intraday_data = CryptoIntradayData()

        # Fetch live bars (200 = ~16h of 5-min bars, enough for features)
        ccxt_sym = symbol.replace("-", "/").replace("USD", "/USD") if "/" not in symbol else symbol
        # Normalize: BTC/USD stays BTC/USD
        if not ccxt_sym.endswith("/USD"):
            ccxt_sym = ccxt_sym.split("/")[0] + "/USD"
        bars = self._crypto_intraday_data.fetch_live_bars(ccxt_sym, limit=200)
        if bars.empty or len(bars) < 60:
            log.warning("Insufficient live bars for %s (%d)", symbol, len(bars))
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

        # Fetch BTC bars for cross-market features (altcoins only)
        btc_bars = None
        is_btc = "BTC" in symbol.upper()
        if not is_btc:
            btc_bars = self._crypto_intraday_data.fetch_btc_bars(source="kraken", limit=200)

        result = predictor.predict(bars, btc_bars)
        if result is None:
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}
        return result

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
        """Fetch ATR, trend signal, realized vol, and price for position management.

        For crypto_intraday: uses 5-min bars with intraday-native trend (VWAP + 4h momentum).
        For all other groups: uses daily bars with SMA(50) trend filter.
        """
        # --- Crypto intraday: intraday-native trend from 5-min bars ---
        if self.group == "crypto_intraday":
            return self._get_intraday_market_context(symbol)

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

    def _get_intraday_market_context(self, symbol: str) -> Dict[str, float]:
        """Intraday-native trend filter for crypto_intraday group.

        Uses 5-min bars from Kraken instead of daily SMA-50.
        Trend = majority vote of 3 intraday signals:
          1. close > VWAP(48) — short-term institutional anchor
          2. ret_48bar > 0    — 4-hour momentum positive
          3. close > EMA(20)  — session-level trend on 5-min bars
        trend = +1 if >=2 bullish, -1 if >=2 bearish
        """
        try:
            if self._crypto_intraday_data is None:
                from crypto_intraday_data import CryptoIntradayData
                self._crypto_intraday_data = CryptoIntradayData()

            ccxt_sym = symbol.replace("-", "/").replace("USD", "/USD") if "/" not in symbol else symbol
            if not ccxt_sym.endswith("/USD"):
                ccxt_sym = ccxt_sym.split("/")[0] + "/USD"
            bars = self._crypto_intraday_data.fetch_live_bars(ccxt_sym, limit=100)

            if bars.empty or len(bars) < 50:
                return {"atr": 0.0, "trend": 0.0, "rv_30d": 0.0, "price": 0.0}

            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            vol = bars["volume"].astype(float)
            price = float(close.iloc[-1])

            # ATR(14) on 5-min bars
            atr_s = compute_atr(high, low, close, period=14)
            atr = float(atr_s.iloc[-1]) if not np.isnan(float(atr_s.iloc[-1])) else 0.0

            # Signal 1: close > VWAP(48) — ~4h VWAP
            typical = (high + low + close) / 3
            vwap_48 = (typical * vol).rolling(48).sum() / vol.rolling(48).sum().clip(lower=1)
            vwap_bull = 1 if price > float(vwap_48.iloc[-1]) else 0

            # Signal 2: 4-hour momentum (ret_48bar > 0)
            ret_48 = float(close.iloc[-1] / close.iloc[-48] - 1) if len(close) >= 48 else 0.0
            mom_bull = 1 if ret_48 > 0 else 0

            # Signal 3: close > EMA(20) on 5-min bars
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_bull = 1 if price > float(ema_20.iloc[-1]) else 0

            # Majority vote: >=2 bullish → trend = +1, else -1
            bull_votes = vwap_bull + mom_bull + ema_bull
            trend = 1.0 if bull_votes >= 2 else -1.0

            # Realized vol (annualized from 5-min bars)
            rv = float(close.pct_change().rolling(48).std().iloc[-1]) * np.sqrt(288 * 365)

            return {
                "atr":    atr,
                "trend":  trend,
                "rv_30d": rv if not np.isnan(rv) else 0.0,
                "price":  price,
            }
        except Exception as exc:
            log.warning("Intraday market context failed for %s: %s", symbol, exc)
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

        # Clear stale tracker when symbol is back in the active set
        if not exit_only:
            self._stale_since.pop(symbol, None)

        # Wrongly placed: symbol has no position here — nothing to manage
        if exit_only and not has_position:
            return "EXIT-ONLY  (no position — symbol not in this group)"

        # Exit-only with no model signal → close immediately, no reason to hold
        if exit_only and has_position:
            pred = self.get_prediction(symbol)
            er = pred.get("expected_return", 0.0)
            if er == 0.0 and pred.get("direction", "UNKNOWN") == "UNKNOWN":
                pos_info = positions[symbol]
                qty = pos_info["qty"]
                side = pos_info["side"]
                pnl_pct = pos_info["unrealized_pnl_pct"]
                qty_display = f"{qty:.6f}" if is_crypto and isinstance(qty, float) else str(int(qty))
                reason = "exit_only_no_model (no predictor loaded)"
                if side == "LONG":
                    self.sell(symbol, qty, reason)
                else:
                    self.buy_to_cover(symbol, qty, reason)
                return (f"EXIT  ({reason}, P&L={pnl_pct:+.2%}, "
                        f"{qty_display} sh {side})")

        # --- Exit logic (tier-based) ---
        if has_position:
            qty = pos["qty"]
            qty_display = f"{qty:.6f}" if is_crypto and isinstance(qty, float) else str(int(qty))
            side = pos["side"]
            current_price = pos["current_price"]
            entry_price = pos["entry_price"]
            pnl_pct = pos["unrealized_pnl_pct"]

            use_legacy_stops = exit_only or (symbol in self._legacy_stricter_exit)

            # Peak tracking (best price for this position's direction)
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

            # Get tier-based exit params for this symbol
            ep = self._exit_params.get(symbol)
            tier = self._vol_tiers.get(symbol)
            if ep is None:
                from src.risk_config import get_exit_params, VolTier
                ep = get_exit_params(self.group or "swing", VolTier.HIGH)
                tier = VolTier.HIGH

            # Helper: execute exit order and return action string
            def _do_exit(reason: str, layer: str) -> Optional[str]:
                if side == "LONG":
                    oid = self.sell(symbol, qty, reason=reason, limit_price=current_price)
                else:
                    oid = self.buy_to_cover(symbol, qty, reason=reason, limit_price=current_price)
                if oid is None:
                    return None  # market closed
                self._record_closed_trade(pnl_pct, symbol)
                self._log_daily_trade(symbol, side, qty, current_price, reason, pnl_pct)
                log.info("EXIT layer=%s symbol=%s tier=%s pnl=%.2f%% reason=%s",
                         layer, symbol, tier.value if tier else "?", pnl_pct * 100, reason)
                self._clear_symbol_state(symbol)
                return (f"EXIT  ({reason}, P&L={pnl_pct:+.2%}, {qty_display} sh {side})  "
                        f"ML: {direction} E[r]={expected_return:+.4f}  [layer={layer}, tier={tier.value if tier else '?'}]")

            # --- EOD exit for intraday momentum models ---
            predictor = self.predictors.get(symbol)
            if getattr(predictor, 'eod_exit', False):
                try:
                    from zoneinfo import ZoneInfo
                except ImportError:
                    from backports.zoneinfo import ZoneInfo
                import datetime as _dt
                now_et = _dt.datetime.now(ZoneInfo("America/New_York"))
                if now_et.time() >= _dt.time(15, 30):
                    result = _do_exit("eod_exit (intraday momentum)", "eod")
                    if result:
                        return result
                    return f"EXIT-DEFERRED  (EOD close, market closed)"

            # Legacy positions: trailing stop (3% from peak)
            if use_legacy_stops:
                if drawdown_from_peak >= LEGACY_TRAILING_STOP_PCT:
                    result = _do_exit(f"legacy_trailing_stop ({drawdown_from_peak:.2%})", "legacy")
                    if result:
                        return result
                    return f"EXIT-DEFERRED  (legacy trailing stop, market closed)"

            # Stale position auto-close: if exit-only for > N hours, close unconditionally
            if exit_only:
                if symbol not in self._stale_since:
                    self._stale_since[symbol] = datetime.now(timezone.utc)
                stale_hours = (datetime.now(timezone.utc) - self._stale_since[symbol]).total_seconds() / 3600
                if stale_hours >= self._stale_max_hours:
                    reason = f"stale_auto_close ({stale_hours:.1f}h exit-only, P&L={pnl_pct:+.2%})"
                    result = _do_exit(reason, "stale")
                    if result:
                        self._stale_since.pop(symbol, None)
                        return result
                    return f"EXIT-DEFERRED  (stale auto-close, market closed)"

            # ===== LAYER 1: Hard safety (disaster stop) =====
            # Regime-adaptive: widen stops in high-vol, tighten in low-vol
            regime_mult = getattr(self, '_regime_mult', 1.0)
            if not use_legacy_stops:
                if ep.use_atr:
                    entry_atr = self._entry_atrs.get(symbol, bar_atr)
                    if entry_atr > 0 and entry_price > 0:
                        disaster_stop_pct = min(ep.disaster_stop_pct,
                                                ep.disaster_atr_mult * entry_atr / entry_price)
                    else:
                        disaster_stop_pct = ep.disaster_stop_pct
                else:
                    disaster_stop_pct = ep.disaster_stop_pct
                disaster_stop_pct *= regime_mult

                if pnl_pct <= -disaster_stop_pct:
                    reason = f"disaster_stop ({pnl_pct:+.2%} <= -{disaster_stop_pct:.2%})"
                    result = _do_exit(reason, "safety")
                    if result is None:
                        return f"EXIT-DEFERRED  (disaster stop {pnl_pct:+.2%}, market closed)"
                    cd_hours = self.max_loss_cooldown_hours
                    if cd_hours > 0:
                        self._cooldown_until[symbol] = datetime.now(timezone.utc) + timedelta(hours=cd_hours)
                        log.info("Max-loss cooldown %s: %.1fh after disaster stop", symbol, cd_hours)
                    self._max_loss_exits[symbol] = {
                        "direction": side, "time": datetime.now(timezone.utc),
                    }
                    return result

            # ===== Breakeven ratchet: set floor once profitable past threshold =====
            if not use_legacy_stops and pnl_pct >= ep.breakeven_ratchet_pct:
                if symbol not in self._breakeven_floor:
                    self._breakeven_floor[symbol] = entry_price
                    log.info("Breakeven ratchet armed for %s at entry=%.4f (PnL=%.2f%%)",
                             symbol, entry_price, pnl_pct * 100)
            # Check breakeven floor violation
            if not use_legacy_stops and symbol in self._breakeven_floor:
                floor = self._breakeven_floor[symbol]
                floor_breached = (current_price <= floor if side == "LONG"
                                  else current_price >= floor)
                if floor_breached and pnl_pct <= 0:
                    reason = (f"breakeven_ratchet (floor=${floor:.4f}, "
                              f"current=${current_price:.4f})")
                    result = _do_exit(reason, "breakeven")
                    if result is None:
                        return f"EXIT-DEFERRED  (breakeven ratchet, market closed)"
                    return result

            # ===== LAYER 2: Profit protection (profit-lock arm + trail) =====
            if not use_legacy_stops:
                if ep.use_atr:
                    entry_atr = self._entry_atrs.get(symbol, bar_atr)
                    if entry_atr > 0 and entry_price > 0:
                        arm_pct = min(ep.profit_lock_arm_pct,
                                      ep.arm_atr_mult * entry_atr / entry_price)
                        trail_pct = min(ep.profit_lock_trail_pct,
                                        ep.trail_atr_mult * entry_atr / entry_price)
                    else:
                        arm_pct = ep.profit_lock_arm_pct
                        trail_pct = ep.profit_lock_trail_pct
                else:
                    arm_pct = ep.profit_lock_arm_pct
                    trail_pct = ep.profit_lock_trail_pct

                # Regime-adaptive trail width + time-decay tightening
                trail_pct *= regime_mult
                try:
                    from regime_detector import time_decay_tightening
                    _bars = self._bars_held.get(symbol, 0)
                    _max = ep.max_hold_bars if hasattr(ep, 'max_hold_bars') else 100
                    trail_pct *= time_decay_tightening(_bars, _max)
                except ImportError:
                    pass

                # Arm profit-lock when PnL exceeds threshold
                if pnl_pct >= arm_pct:
                    if not self._profit_lock_active.get(symbol, False):
                        log.info("Profit-lock ARMED for %s at PnL=%.2f%% (arm=%.2f%%)",
                                 symbol, pnl_pct * 100, arm_pct * 100)
                    self._profit_lock_active[symbol] = True

                profit_lock_armed = self._profit_lock_active.get(symbol, False)

                if profit_lock_armed:
                    # 2a: Profit-lock armed + model flips + still profitable -> exit,
                    # but signal-decay exits still respect the group's min hold.
                    signal_min_hold_ok = True
                    if self.group == "crypto_intraday":
                        from src.risk_config import CRYPTO_INTRADAY_MIN_HOLD_BARS
                        # _ci_bars_held already incremented earlier in the cycle
                        signal_min_hold_ok = (
                            self._ci_bars_held.get(symbol, 0) >= CRYPTO_INTRADAY_MIN_HOLD_BARS
                        )
                    else:
                        # Use bars_held + 1 to match signal-decay path (line ~2279)
                        # where counter is incremented before the min-hold check.
                        _held = self._bars_held.get(symbol, 0) + 1
                        signal_min_hold_ok = (
                            _held >= get_effective_min_hold(
                                getattr(ep, "min_hold_bars", 0),
                                self._entry_atrs.get(symbol, 0),
                                bar_atr,
                            )
                        )
                    signal_flipped = ((side == "LONG" and expected_return <= 0)
                                      or (side == "SHORT" and expected_return >= 0))
                    if signal_flipped and pnl_pct > 0 and signal_min_hold_ok:
                        reason = (f"profit_lock+signal_decay (armed, E[r]={expected_return:+.4f}, "
                                  f"PnL={pnl_pct:+.2%})")
                        result = _do_exit(reason, "profit_lock+signal")
                        if result is None:
                            return (f"EXIT-DEFERRED  (profit-lock+signal decay, market closed)")
                        self._decay_cooldown_until[symbol] = (
                            datetime.now(timezone.utc) + timedelta(seconds=2 * self.check_interval))
                        return result

                    # 2b: Price hits trail → exit (price invalidates leg even if model is slow)
                    if drawdown_from_peak >= trail_pct:
                        reason = (f"profit_lock_trail ({drawdown_from_peak:.2%} from peak, "
                                  f"PnL={pnl_pct:+.2%}, trail={trail_pct:.2%})")
                        result = _do_exit(reason, "profit_lock")
                        if result is None:
                            return f"EXIT-DEFERRED  (profit-lock trail, market closed)"
                        return result

            # ===== LAYER 3: Model-state exits =====
            if not use_legacy_stops:
                if self.group == "crypto_intraday":
                    # --- Crypto intraday: horizon-aware hold ---
                    # Bar counter incremented in _get_crypto_intraday_prediction()
                    from src.risk_config import (CRYPTO_INTRADAY_MIN_HOLD_BARS,
                                                 CRYPTO_INTRADAY_MAX_HOLD_BARS,
                                                 CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS,
                                                 CRYPTO_INTRADAY_BAR_SECONDS)
                    ci_bars = self._ci_bars_held.get(symbol, 0)

                    # Max hold: force exit at 48 bars (4 hours)
                    if ci_bars >= CRYPTO_INTRADAY_MAX_HOLD_BARS:
                        reason = (f"max_hold_expired ({ci_bars} bars = "
                                  f"{ci_bars * 5 / 60:.1f}h, PnL={pnl_pct:+.2%})")
                        result = _do_exit(reason, "time")
                        if result is None:
                            return f"EXIT-DEFERRED  (max hold {ci_bars} bars)"
                        log.info("[CryptoIntraday] %s: exit after %d bars, reason=%s, pnl=%.4f%%",
                                 symbol, ci_bars, "max_hold_expired", pnl_pct * 100)
                        return result

                    # Before 12 bars (1 hour): hold — no model reassessment
                    if ci_bars < CRYPTO_INTRADAY_MIN_HOLD_BARS:
                        pass  # only disaster stop and profit lock can exit (checked above)
                    else:
                        # After 12 bars: reassess — exit on signal reversal
                        signal_flipped = ((side == "LONG" and expected_return <= 0)
                                          or (side == "SHORT" and expected_return >= 0))
                        if signal_flipped:
                            reason = (f"signal_reversal_post_horizon (E[r]={expected_return:+.4f}, "
                                      f"bars={ci_bars}, PnL={pnl_pct:+.2%})")
                            result = _do_exit(reason, "signal_reversal")
                            if result is None:
                                return (f"EXIT-DEFERRED  (signal reversal E[r]={expected_return:+.4f}, "
                                        f"market closed)")
                            log.info("[CryptoIntraday] %s: exit after %d bars, reason=%s, pnl=%.4f%%",
                                     symbol, ci_bars, "signal_reversal_post_horizon", pnl_pct * 100)
                            # 3-bar (15-min) cooldown after reversal exits
                            cooldown_secs = CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS * CRYPTO_INTRADAY_BAR_SECONDS
                            self._decay_cooldown_until[symbol] = (
                                datetime.now(timezone.utc) + timedelta(seconds=cooldown_secs))
                            return result
                else:
                    # --- Non-crypto-intraday: original signal decay logic ---
                    bars_held = self._bars_held.get(symbol, 0) + 1
                    self._bars_held[symbol] = bars_held

                    signal_flipped = ((side == "LONG" and expected_return <= 0)
                                      or (side == "SHORT" and expected_return >= 0))
                    if signal_flipped:
                        self._signal_flip_counts[symbol] = self._signal_flip_counts.get(symbol, 0) + 1
                    else:
                        self._signal_flip_counts[symbol] = 0

                    flip_count = self._signal_flip_counts.get(symbol, 0)
                    required_flips = ep.signal_flip_consecutive
                    base_min_hold = getattr(ep, "min_hold_bars", 0)
                    # ATR-adaptive min hold: scale by vol ratio.
                    # High vol → hold longer (signal needs more time to play out).
                    # Low vol → exit sooner (opportunity cost).
                    # Clamped to [base/2, base*2] to prevent extremes.
                    min_hold = get_effective_min_hold(
                        base_min_hold,
                        self._entry_atrs.get(symbol, 0),
                        bar_atr,
                    )

                    if bars_held >= min_hold and flip_count >= required_flips:
                        reason = (f"signal_decay (E[r]={expected_return:+.4f}, "
                                  f"flips={flip_count}/{required_flips}, held={bars_held})")
                        result = _do_exit(reason, "signal_decay")
                        if result is None:
                            return (f"EXIT-DEFERRED  (signal decay E[r]={expected_return:+.4f}, "
                                    f"P&L={pnl_pct:+.2%}, market closed)")
                        self._decay_cooldown_until[symbol] = (
                            datetime.now(timezone.utc) + timedelta(seconds=2 * self.check_interval))
                        return result

            # ===== LAYER 4: Time exits =====
            # 4a: Max underwater duration (non-crypto-intraday only; CI uses bar-based max hold above)
            if not use_legacy_stops and ep.max_underwater_days > 0 and self.group != "crypto_intraday":
                entry_time = self._entry_times.get(symbol)
                if entry_time and pnl_pct < 0:
                    days_held = (datetime.now(timezone.utc) - entry_time).days
                    if days_held >= ep.max_underwater_days:
                        reason = (f"max_underwater_{ep.max_underwater_days}d "
                                  f"(held {days_held}d, PnL={pnl_pct:+.2%})")
                        result = _do_exit(reason, "time")
                        if result is None:
                            return f"EXIT-DEFERRED  (max underwater {days_held}d, market closed)"
                        return result

            # HOLD
            hold_extras = []
            if self._profit_lock_active.get(symbol, False):
                hold_extras.append("PL=armed")
            if symbol in self._breakeven_floor:
                hold_extras.append(f"BE=${self._breakeven_floor[symbol]:.2f}")
            if self.group == "crypto_intraday":
                ci_bars = self._ci_bars_held.get(symbol, 0)
                hold_extras.append(f"bars={ci_bars}/48")
            extras_str = f"  [{', '.join(hold_extras)}]" if hold_extras else ""
            return (f"HOLD  ({side} {qty_display} sh @ ${entry_price:.2f}, "
                    f"P&L: {pnl_pct:+.2%})  "
                    f"ML: {direction} E[r]={expected_return:+.4f}  "
                    f"tier={tier.value if tier else '?'}{extras_str}")

        # --- Entry logic ---
        if exit_only:
            return (f"EXIT-ONLY  (managing out -- {direction} E[r]={expected_return:+.4f}, "
                    f"no new entries for this symbol in this account)")

        # Regime gate
        regime_ok, regime_reason = self._check_regime()
        if not regime_ok:
            return f"REGIME-BLOCK  ({regime_reason})  ML: {direction} E[r]={expected_return:+.4f}"

        # Model health gate: pause entries if model is degraded
        should_pause, pause_reason = self._monitor.should_pause_model(symbol)
        if should_pause:
            log.warning("Model paused for %s: %s", symbol, pause_reason)
            self._alert_engine.notify_model_paused(symbol, pause_reason, group=self.group or "")
            return f"MODEL-PAUSED  ({pause_reason})  ML: {direction} E[r]={expected_return:+.4f}"

        # Crypto guardrails: concentration limiter + win-rate auto-pause
        conc_block = self._check_symbol_concentration(symbol)
        if conc_block:
            return f"CONC-BLOCK  ({conc_block})  ML: {direction} E[r]={expected_return:+.4f}"
        wr_block = self._check_crypto_winrate_pause()
        if wr_block:
            return f"WR-PAUSE  ({wr_block})  ML: {direction} E[r]={expected_return:+.4f}"

        # Determine entry direction.
        # Respect the model's own quality gate; cost threshold is the entry gate.
        # For crypto, use per-symbol cost threshold from universe.json if higher
        # than the global config (illiquid coins have wider spreads).
        effective_cost_threshold = self.cost_threshold
        if self.group in ("crypto", "crypto_intraday"):
            yf_sym = _crypto_to_yfinance(symbol) if "/" in symbol else symbol
            coin_cfg = get_coin_cost_config(yf_sym)
            per_sym_threshold = coin_cfg.get("cost_threshold", 0.0)
            if per_sym_threshold > effective_cost_threshold:
                effective_cost_threshold = per_sym_threshold
            # Per-symbol cost floor: threshold must be >= 1.5x actual RT costs
            try:
                from cost_model import get_symbol_costs
                sym_costs = get_symbol_costs(symbol)
                cost_floor = sym_costs.round_trip_pct * 1.5
                if cost_floor > effective_cost_threshold:
                    log.debug("Cost floor raised threshold for %s: %.4f → %.4f (RT=%.1fbps)",
                              symbol, effective_cost_threshold, cost_floor,
                              sym_costs.round_trip_bps)
                    effective_cost_threshold = cost_floor
            except Exception:
                pass  # fall back to config threshold

        enter_dir = None
        if not pred.get("tradeable", True):
            return (f"SKIP  (model gate: not tradeable)  ML: {direction} "
                    f"E[r]={expected_return:+.4f}")

        # Confidence-gated entry: use quantile bounds when available.
        # LONG only if pessimistic (q25) scenario still clears cost threshold.
        # SHORT only if optimistic (q75) scenario still below -cost_threshold.
        lower_bound = pred.get("lower_bound")
        upper_bound = pred.get("upper_bound")
        if lower_bound is not None and upper_bound is not None:
            if lower_bound > effective_cost_threshold:
                enter_dir = "LONG"
            elif upper_bound < -effective_cost_threshold:
                enter_dir = "SHORT"
            elif expected_return > effective_cost_threshold or expected_return < -effective_cost_threshold:
                log.info("Quantile gate filtered %s: E[r]=%.4f but bounds=[%.4f, %.4f] vs threshold=%.4f",
                         symbol, expected_return, lower_bound, upper_bound, effective_cost_threshold)
        else:
            # Fallback: no quantile models, use raw expected return
            if expected_return > effective_cost_threshold:
                enter_dir = "LONG"
            elif expected_return < -effective_cost_threshold:
                enter_dir = "SHORT"

        if enter_dir is not None:
            # Pending order check: skip entry if there are unfilled orders for this symbol
            # (prevents duplicate entries when extended-hours limit orders haven't filled yet)
            if self._kraken is None:  # Alpaca only (Kraken paper fills instantly)
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    open_req = GetOrdersRequest(
                        status=QueryOrderStatus.OPEN, symbols=[symbol], limit=5,
                    )
                    open_orders = self.trading_client.get_orders(open_req)
                    if open_orders:
                        return (f"PENDING-ORDER  ({len(open_orders)} unfilled order(s) for {symbol}, "
                                f"skipping entry)  ML: {direction} E[r]={expected_return:+.4f}")
                except Exception as exc:
                    log.debug("Open order check failed for %s: %s", symbol, exc)

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
                    effective_threshold = effective_cost_threshold * self.same_dir_confidence_mult
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

            # --- Position sizing: half-Kelly × signal strength, capped by risk limits ---
            # Size from TOTAL equity (not per-symbol sub-allocation).
            risk = self._risk_config
            try:
                account = self.get_account_summary()
                equity = account["equity"]
                self._last_known_equity = equity  # cache for fallback
                if equity > self._peak_equity:
                    self._peak_equity = equity
            except Exception:
                equity = getattr(self, '_last_known_equity', self.initial_capital)

            signal_pct = min(1.0, max(risk.min_signal_scale,
                                      abs(expected_return) / self.target_return))

            # Half-Kelly from rolling trade history (falls back to position_pct)
            derisk_key = symbol.replace("/", "-")
            derisk_state = self._derisk_states.get(derisk_key)
            kelly_f = None
            if derisk_state is not None:
                kelly_f = derisk_state.half_kelly(
                    window=risk.kelly_window, min_trades=risk.kelly_min_trades
                )
            if kelly_f is not None:
                # Cross-group discount: data-driven overlay replaces static discount
                from risk_overlay import load_scaling_factor
                overlay_factor = load_scaling_factor(self.group or "swing")
                effective_discount = min(risk.cross_group_kelly_discount, overlay_factor)
                base_frac = min(kelly_f * effective_discount, risk.kelly_cap)
                size_source = f"kelly={kelly_f:.3f}×{effective_discount:.2f}"
                if overlay_factor < risk.cross_group_kelly_discount:
                    size_source += f"[overlay={overlay_factor:.2f}]"
            else:
                base_frac = risk.position_pct
                size_source = "fixed"

            sizing_pct = base_frac * signal_pct

            # --- Vol-targeting layer: normalize size by inverse volatility ---
            # Ensures each position contributes ~target_vol risk regardless of
            # the asset's own volatility. High-vol assets get smaller dollar size.
            rv_30d = ctx.get("rv_30d", 0.0)
            size_note_parts = []
            if rv_30d > 0.01:
                target_vol = getattr(self, 'target_vol', 0.20)
                vol_scalar = min(2.0, max(0.25, target_vol / rv_30d))
                sizing_pct *= vol_scalar
                if abs(vol_scalar - 1.0) > 0.05:
                    size_note_parts.append(f"vol={vol_scalar:.2f}")

            # --- VIX-based continuous regime scaling ---
            # Smoothly reduces size as VIX rises: at VIX=40 size is halved.
            # Intraday skips this (VIX is already a model feature).
            if self.mode != "intraday":
                vix_now = self._get_current_vix()
                if vix_now is not None and vix_now > 0:
                    vix_scalar = min(1.0, 20.0 / vix_now)
                    sizing_pct *= vix_scalar
                    if vix_scalar < 0.95:
                        size_note_parts.append(f"vix={vix_scalar:.2f}")

            # --- Vol-adjusted position cap ---
            # Tighter cap for high-vol assets, looser for low-vol (up to 1.5x base)
            if rv_30d > 0.01:
                target_vol = getattr(self, 'target_vol', 0.20)
                vol_adj_cap = min(
                    risk.max_position_pct * 1.5,
                    risk.max_position_pct * (target_vol / rv_30d)
                )
                vol_adj_cap = max(0.03, vol_adj_cap)
            else:
                vol_adj_cap = risk.max_position_pct
            sizing_pct = min(sizing_pct, vol_adj_cap)

            if is_crypto:
                log.debug("Crypto sizing %s: base=%.3f signal=%.3f max_pos=%.2f "
                          "max_exp=%.2f sizing=%.3f",
                          symbol, base_frac, signal_pct,
                          risk.max_position_pct, risk.max_total_exposure,
                          sizing_pct)

            # --- Per-symbol cap (OOS Sharpe-tiered) ---
            # This is the SINGLE source of OOS-based sizing. Replaces the old
            # conf_mult (redundant with sym_cap) and rank_scalar (redundant with
            # sym_cap). Both mapped OOS Sharpe → multiplier, stacking 3 layers
            # that all expressed the same information.
            sym_cap = get_symbol_cap(symbol, self.group or "swing")
            if sym_cap <= 0.0:
                return (f"DISABLED  ({symbol} cap=0% by OOS performance)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")
            sizing_pct = min(sizing_pct, sym_cap)

            # --- BTC correlation penalty (crypto only) ---
            # Always penalize highly BTC-correlated alts, regardless of whether
            # a BTC position is open. In crypto, ALL alts are implicitly BTC-exposed.
            # Anchor coins (BTC, ETH) are exempt (their btc_corr is set to 0).
            _BTC_KEYS = ("BTC/USD", "BTCUSD", "ETH/USD", "ETHUSD")
            if is_crypto and self._btc_correlations and symbol not in _BTC_KEYS:
                btc_corr = self._btc_correlations.get(symbol, 0.5)
                corr_penalty = 1 - 0.5 * btc_corr
                sizing_pct *= corr_penalty
                if btc_corr > 0.7:
                    log.info("BTC corr penalty for %s: corr=%.2f → size ×%.2f",
                             symbol, btc_corr, corr_penalty)

            # --- Auto de-risking check (rolling performance gate) ---
            size_note = f" [{size_source}]"
            if size_note_parts:
                size_note += f" [{', '.join(size_note_parts)}]"
            if derisk_state is not None:
                derisk_action, derisk_reason = evaluate_derisk(
                    derisk_state,
                    window=50,
                )
                if derisk_action == "disable":
                    return (f"DERISK-DISABLED  ({derisk_reason})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")
                elif derisk_action == "halfsize":
                    sizing_pct *= 0.5
                    size_note += f" [derisk-half: {derisk_reason}]"

            # --- Drawdown throttle (equity-curve-based) ---
            dd_mult = drawdown_size_mult(equity, self._peak_equity)
            if dd_mult <= 0.0:
                dd_pct = (self._peak_equity - equity) / self._peak_equity * 100
                return (f"DRAWDOWN-HALT  (equity {equity:.0f} is {dd_pct:.1f}% below peak "
                        f"{self._peak_equity:.0f})  ML: {direction} E[r]={expected_return:+.4f}")
            if dd_mult < 1.0:
                sizing_pct *= dd_mult
                size_note += f" [dd-throttle {dd_mult:.0%}]"

            # --- Post-loss size reduction (Opt #5) ---
            # After a disaster-stop / max-loss exit, reduce size for N hours
            if last_loss:
                loss_age_h = (datetime.now(timezone.utc) - last_loss["time"]).total_seconds() / 3600
                if loss_age_h < self.post_loss_size_hours:
                    sizing_pct *= self.post_loss_size_mult
                    size_note += (f" [post-loss {self.post_loss_size_mult:.0%} "
                                  f"for {self.post_loss_size_hours - loss_age_h:.1f}h more]")

            # --- Minimum size floor ---
            # Below ~1% of equity, round-trip costs eat most expected return.
            # Skip rather than open a position that can't cover its costs.
            _MIN_SIZE_PCT = 0.01
            if sizing_pct < _MIN_SIZE_PCT:
                return (f"SKIP  (sizing {sizing_pct:.3%} < {_MIN_SIZE_PCT:.0%} floor)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")

            invest = equity * sizing_pct
            qty = invest / current_price
            if is_crypto:
                qty = round(qty, 6)
            else:
                qty = int(qty)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} E[r]={expected_return:+.4f}"

            # Portfolio constraint check (equity already fetched above)
            # Exclude pending-reconcile positions — they're queued for close
            # and should not block new entries.
            _constrained_positions = {
                s: p for s, p in positions.items()
                if s not in self._pending_reconcile
            }
            try:
                allowed, constraint_reason = check_position_allowed(
                    symbol, invest, equity, _constrained_positions, risk
                )
                if not allowed:
                    return (f"CONSTRAINT-BLOCK  ({constraint_reason})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")
                # Theme cap check (includes cross-group positions)
                try:
                    from risk_overlay import get_all_group_positions
                    _cross = get_all_group_positions(exclude_group=self.group)
                    _all_positions = {**positions, **_cross}
                except Exception:
                    _all_positions = positions
                theme_ok, theme_reason = check_theme_cap(
                    symbol, invest, equity, _all_positions
                )
                if not theme_ok:
                    return (f"THEME-CAP  ({theme_reason})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")
                # Sleeve budget check (cap per group as % of total portfolio)
                # Estimate total equity: ~$95K per group × 4 ML groups
                _total_eq = equity * 4
                sleeve_ok, sleeve_reason = check_sleeve_budget(
                    self.group or "default", invest, equity, _total_eq, positions
                )
                if not sleeve_ok:
                    return (f"SLEEVE-CAP  ({sleeve_reason})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")
            except Exception as exc:
                log.debug("Portfolio constraint check failed: %s", exc)

            # Hard safety cap: re-count positions as final guard.
            # Even if check_position_allowed passed (stale dict), refuse entry
            # if we're already at max_positions. Belt-and-suspenders.
            # Exclude _pending_reconcile (being closed) to avoid blocking entries.
            _active_count = sum(1 for s, p in positions.items()
                                if p.get("qty", 0) != 0
                                and s not in self._pending_reconcile)
            if _active_count >= self._risk_config.max_positions:
                return (f"HARD-CAP  ({_active_count} >= {self._risk_config.max_positions} positions)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")

            if enter_dir == "LONG":
                self.buy(symbol, qty, limit_price=current_price)
            else:
                self.sell_short(symbol, qty, limit_price=current_price)

            self._peak_prices[symbol] = current_price
            self._entry_atrs[symbol] = bar_atr
            self._entry_times[symbol] = datetime.now(timezone.utc)
            self._profit_lock_active[symbol] = False
            if self.group == "crypto_intraday":
                self._ci_bars_held[symbol] = 0
                self._ci_last_bar_ts[symbol] = datetime.now(timezone.utc)

            # Monitor: record prediction at entry
            model_type = getattr(self.predictors.get(symbol), 'model_type', 'lstm')
            self._monitor.record_prediction(symbol, expected_return, model_type=model_type)
            self._entry_predictions[symbol] = expected_return

            notional = qty * current_price
            self._alert_engine.notify_entry(
                symbol, enter_dir, qty, current_price, notional,
                group=self.group or "", expected_return=expected_return,
            )

            qty_display = f"{qty:.6f}" if is_crypto else str(qty)
            self._log_daily_trade(symbol, enter_dir, qty, current_price,
                                  f"entry ({enter_dir})", 0.0)
            return (f"{enter_dir}  ({qty_display} sh @ ~${current_price:.2f}, "
                    f"${qty * current_price:,.0f}, size={sizing_pct:.0%}{size_note})  "
                    f"ML: {direction} E[r]={expected_return:+.4f}  trend={bar_trend:+.0f}")

        return (f"SKIP  (no signal)  ML: {direction} E[r]={expected_return:+.4f}  "
                f"trend={bar_trend:+.0f}")

    def _record_closed_trade(self, pnl_pct: float, symbol: str = "") -> None:
        """Track consecutive losses, rolling win rate, model monitor, and de-risk state."""
        # Monitor: record realized return
        if symbol:
            self._monitor.record_realized(symbol, pnl_pct)
            self._entry_predictions.pop(symbol, None)
            # Feed de-risk state
            derisk_key = symbol.replace("/", "-")
            if derisk_key not in self._derisk_states:
                self._derisk_states[derisk_key] = DeRiskState()
            self._derisk_states[derisk_key].record_trade(pnl_pct)
            # Update crypto concentration + win-rate guardrails
            self._update_crypto_guardrails(symbol, pnl_pct)
        if pnl_pct <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        # Rolling 20-trade win rate cooldown
        self._recent_trade_wins.append(1 if pnl_pct > 0 else 0)
        if len(self._recent_trade_wins) >= 20 and self._regime_cooldown_until is None:
            recent_20 = list(self._recent_trade_wins)[-20:]
            wr = sum(recent_20) / 20
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
        self._entry_times.pop(symbol, None)
        self._profit_lock_active.pop(symbol, None)
        self._breakeven_floor.pop(symbol, None)
        self._signal_flip_counts.pop(symbol, None)
        self._bars_held.pop(symbol, None)
        self._ci_bars_held.pop(symbol, None)
        self._ci_last_bar_ts.pop(symbol, None)
        # Note: _decay_cooldown_until is intentionally NOT cleared here —
        # signal-decay exits set it after calling this method.

    def _log_daily_trade(self, symbol: str, side: str, qty: float,
                         price: float, reason: str, pnl_pct: float) -> None:
        """Append one row to outputs/daily_trades_YYYYMMDD.csv for reproducible trade history."""
        import csv
        try:
            os.makedirs(TRADES_DIR, exist_ok=True)
            today_str = datetime.now().strftime("%Y%m%d")
            csv_path = os.path.join(TRADES_DIR, f"daily_trades_{today_str}.csv")
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
    # -- Startup reconciliation ----------------------------------------
    def _reconcile_positions(self) -> None:
        """Close positions that don't belong to this group's symbol set.

        Runs once at startup to prevent stale/misaligned positions from
        accumulating across restarts.  Covers two cases:
        1. Symbols removed from SYMBOL_GROUPS or the coin selector universe.
        2. Cross-account contamination (e.g. crypto on Alpaca intraday).
        """
        try:
            positions = self.get_positions()
        except Exception as exc:
            log.warning("Reconciliation skipped — could not fetch positions: %s", exc)
            return

        if not positions:
            return

        # Build the valid symbol set for this group.
        # Includes both static group symbols AND dynamic selector picks
        # (ETF selector for swing, coin selector for crypto).
        valid_symbols: set = set(self.symbols)
        # Swing/equity: include ETF selector's active symbols.
        # Defer reconciliation until after the first ETF selector run
        # so we don't close positions the selector is about to pick.
        if self._etf_selector is not None and not self._etf_active_symbols:
            log.info("Swing reconciliation deferred — waiting for first ETF selector run")
            return
        if self._etf_active_symbols:
            valid_symbols.update(self._etf_active_symbols)
        if self.group in ("crypto", "crypto_intraday"):
            # If selector hasn't run yet (startup), defer crypto reconciliation
            # until after the first selector run establishes valid picks.
            if self._coin_selector is not None and not self._selector_active_symbols:
                log.info("Crypto reconciliation deferred — waiting for first selector run")
                return
            # Add currently selected coins (both dash and slash forms)
            for s in (self._selector_active_symbols or []):
                valid_symbols.add(s)
                valid_symbols.add(s.replace("-", "/"))
                valid_symbols.add(s.replace("/", "-"))
            # Also keep symbols that are in self.symbols (static fallback list)
            for s in self.symbols:
                valid_symbols.add(s.replace("-", "/"))
                valid_symbols.add(s.replace("/", "-"))

        stale: list = []
        for sym, pos in positions.items():
            if pos.get("qty", 0) == 0:
                continue
            if _is_option_symbol(sym):
                continue  # don't touch option positions
            if sym not in valid_symbols:
                stale.append((sym, pos))

        if not stale:
            log.info("Reconciliation OK — all %d positions belong to this group",
                     len(positions))
            return

        print(f"\n  [RECONCILE] Found {len(stale)} position(s) not in "
              f"'{self.group}' symbol set — closing:")
        for sym, pos in stale:
            qty = pos["qty"]
            side = pos["side"]
            entry = pos.get("entry_price", 0)
            pnl = pos.get("unrealized_pnl", 0)
            print(f"    {sym:>10s}  {side:5s}  qty={qty}  "
                  f"entry=${entry:.4f}  pnl=${pnl:+.2f}")
            try:
                # Cancel any existing open orders first — stale orders from
                # previous processes hold the qty and block close attempts.
                self._cancel_open_orders_for(sym)
                # Pass current_price as limit so orders fill during extended hours
                # (market orders with DAY TIF don't fill outside regular session).
                lp = pos.get("current_price") or None
                if side == "LONG":
                    oid = self.sell(sym, qty, reason="reconcile_stale",
                                   limit_price=lp)
                else:
                    oid = self.buy_to_cover(sym, qty, reason="reconcile_stale",
                                            limit_price=lp)
                if oid:
                    log.info("Reconciled %s %s %.6f (stale position closed)", side, sym, qty)
                    self._stale_since.pop(sym, None)
                else:
                    # Order was deferred (market closed) — queue for retry
                    self._pending_reconcile[sym] = pos
                    # Start stale timer so auto-close kicks in after _stale_max_days
                    if sym not in self._stale_since:
                        self._stale_since[sym] = datetime.now(timezone.utc)
                    log.warning("Reconcile deferred for %s (market closed) — will retry each cycle", sym)
            except Exception as exc:
                self._pending_reconcile[sym] = pos
                if sym not in self._stale_since:
                    self._stale_since[sym] = datetime.now(timezone.utc)
                log.error("Failed to close stale position %s: %s — will retry", sym, exc)
        print()

    def _maybe_reload_universe(self) -> None:
        """Reload crypto universe from disk if Layer 0 updated it.

        Checks every 30 minutes. If the universe changed (coins added/removed),
        re-runs reconciliation to close positions from removed coins.
        """
        if self.group not in ("crypto", "crypto_intraday"):
            return
        now = datetime.now(timezone.utc)
        if (self._universe_last_check is not None
                and (now - self._universe_last_check).total_seconds() < 1800):
            return
        self._universe_last_check = now

        fresh = load_universe(self._crypto_model_dir)
        if not fresh and self._crypto_model_dir != CRYPTO_MODEL_DIR:
            fresh = load_universe(CRYPTO_MODEL_DIR)
        if not fresh:
            return
        old_set = set(self._selector_universe)
        new_set = set(fresh)
        if old_set == new_set:
            return

        removed = old_set - new_set
        added = new_set - old_set
        self._selector_universe = fresh
        log.info("Universe reloaded: %d coins (+%d, -%d)",
                 len(fresh), len(added), len(removed))
        if removed:
            log.info("Removed from universe: %s", ", ".join(sorted(removed)))
            # Re-run reconciliation to close positions from removed coins
            self._reconcile_positions()

    def _retry_pending_reconcile(self) -> None:
        """Retry closing positions that failed during startup reconciliation.

        Called every cycle. Positions deferred because market was closed will
        be retried until they succeed. This prevents stale positions from
        permanently occupying max_positions slots.
        """
        if not self._pending_reconcile:
            return

        # Refresh actual positions to confirm they still exist
        try:
            current_positions = self.get_positions()
        except Exception:
            return

        closed = []
        for sym, saved_pos in list(self._pending_reconcile.items()):
            pos = current_positions.get(sym)
            if pos is None or pos.get("qty", 0) == 0:
                # Already closed (manually or by another process)
                closed.append(sym)
                continue

            qty = pos["qty"]
            side = pos["side"]
            lp = pos.get("current_price") or None
            try:
                self._cancel_open_orders_for(sym)
                if side == "LONG":
                    oid = self.sell(sym, qty, reason="reconcile_stale_retry",
                                   limit_price=lp)
                else:
                    oid = self.buy_to_cover(sym, qty, reason="reconcile_stale_retry",
                                            limit_price=lp)
                if oid:
                    closed.append(sym)
                    pnl = pos.get("unrealized_pnl", 0)
                    log.info("[RECONCILE-RETRY] Closed stale %s %s %.4f (P&L: $%.2f)",
                             side, sym, qty, pnl)
                    print(f"  [RECONCILE-RETRY] Closed stale {sym} {side} qty={qty} P&L=${pnl:+.2f}")
                # else: still deferred, will retry next cycle
            except Exception as exc:
                log.debug("Reconcile retry failed for %s: %s", sym, exc)

        for sym in closed:
            self._pending_reconcile.pop(sym, None)

        if self._pending_reconcile:
            pending_list = ", ".join(self._pending_reconcile.keys())
            log.debug("Pending reconciliation (will retry): %s", pending_list)

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

        # Reconcile stale positions before entering the main loop
        self._reconcile_positions()

        # Initialize bar counters for existing positions (survive restarts)
        try:
            existing = self.get_positions()
            for sym, pos in existing.items():
                if pos.get("qty", 0) <= 0:
                    continue
                entry_str = pos.get("entry_time", "")
                if entry_str:
                    try:
                        entry_dt = datetime.fromisoformat(entry_str)
                        elapsed = (datetime.now(timezone.utc) - entry_dt).total_seconds()
                        # Use actual check interval for bar estimation (60s for crypto_intraday, 300s otherwise)
                        bar_secs = 60 if self.group == "crypto_intraday" else 300
                        bars = max(0, int(elapsed / bar_secs))
                    except (ValueError, TypeError):
                        bars = 36  # safe default: past min hold
                else:
                    bars = 36
                # Restore both counters so signal decay and crypto intraday
                # horizon logic work correctly after restarts
                if sym not in self._bars_held:
                    self._bars_held[sym] = bars
                    log.info("Restored bars_held for %s: %d bars (%.1fh)",
                             sym, bars, elapsed / 3600 if entry_str else 0)
                if self.group == "crypto_intraday" and sym not in self._ci_bars_held:
                    self._ci_bars_held[sym] = bars
                    log.info("Restored ci_bars_held for %s: %d bars (%.1fh)",
                             sym, bars, elapsed / 3600 if entry_str else 0)
        except Exception as exc:
            log.warning("Could not restore bar counters: %s", exc)

        # Graceful shutdown
        def handle_signal(sig, frame):
            print("\n\n  Shutting down paper trader...\n")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)

        cycle = 0
        while self._running:
            cycle += 1

            # Weekly vol tier recomputation
            self._maybe_recompute_vol_tiers()

            # Session check — intraday: only trade during market/open hours; crypto & swing: run 24/7
            # Crypto and swing run 24/7 (orders fill when Alpaca allows; swing/crypto can use extended hours).
            # All groups run 24/7; session label is informational only
            session = _get_session()
            if self.group == "crypto":
                session_label = "24/7 CRYPTO"
            elif self.group == "crypto_intraday":
                session_label = "24/7 CRYPTO INTRADAY"
            elif self.group == "swing":
                session_label = f"SWING ({session.upper()})"
            else:
                session_label = f"INTRADAY ({session.upper()})"

            try:
                # Retry any stale positions that failed to close during startup reconciliation
                self._retry_pending_reconcile()
                # Check if Layer 0 updated the universe (every 30min)
                self._maybe_reload_universe()

                # Cross-group risk overlay refresh (every 30 min)
                _overlay_now = datetime.now(timezone.utc)
                if (not hasattr(self, '_overlay_last_refresh')
                        or self._overlay_last_refresh is None
                        or (_overlay_now - self._overlay_last_refresh).total_seconds() >= 1800):
                    try:
                        from risk_overlay import update_overlay
                        _factors = update_overlay()
                        log.info("Risk overlay refreshed: %s", _factors)
                        self._overlay_last_refresh = _overlay_now
                    except Exception as _oe:
                        log.debug("Risk overlay refresh failed: %s", _oe)

                # Get account + positions
                account = self.get_account_summary()
                positions = self.get_positions()

                # Publish positions for cross-group exposure checks
                try:
                    from risk_overlay import publish_positions
                    publish_positions(self.group or "default", positions)
                except Exception as _pe:
                    log.debug("Failed to publish positions: %s", _pe)

                # Regime detection: refit daily, update stop multiplier
                if self._regime_detector is not None:
                    _today = datetime.now(timezone.utc).date()
                    _last_fit = getattr(self, '_regime_last_fit_date', None)
                    if _last_fit != _today:
                        try:
                            import yfinance as yf
                            spy = yf.download("SPY", period="2y", progress=False, auto_adjust=True)
                            if not spy.empty:
                                spy_ret = spy["Close"].pct_change().dropna()
                                if self._regime_detector.fit(spy_ret, window=252):
                                    regime = self._regime_detector.predict_regime(spy_ret)
                                    self._regime_mult = self._regime_detector.get_stop_multiplier(regime)
                                    log.info("Regime: %s (mult=%.2f)",
                                             "HIGH-VOL" if regime == 1 else "LOW-VOL", self._regime_mult)
                                self._regime_last_fit_date = _today
                        except Exception as exc:
                            log.warning("Regime fit failed: %s", exc)

                # Layer 1: dynamic symbol selection (date-change gate)
                # Full refresh once per trading day AFTER bar settlement:
                #   ETF: after 06:30 ET (previous day's daily bar confirmed)
                #   Crypto: after 00:30 UTC (daily bar settled at midnight UTC)
                # Intraday fast re-rank: re-score every 1h (ETF) / 30min (crypto)
                #   using the same ranker with fresh short-term data. This catches
                #   intraday opportunity rotation without retraining any ML model.
                # Fallback: static self.symbols
                now_utc = datetime.now(timezone.utc)

                if self.group in ("crypto", "crypto_intraday") and self._coin_selector is not None:
                    _fast_interval = _FAST_REFRESH_SECS.get(self.group)
                    _do_full = (_should_refresh_selector(self.group, self._selector_last_run_date)
                                or not self._selector_active_symbols)
                    _do_fast = (
                        not _do_full
                        and _fast_interval is not None
                        and (self._selector_last_fast_refresh is None
                             or (now_utc - self._selector_last_fast_refresh).total_seconds()
                             >= _fast_interval)
                    )
                    if _do_full or _do_fast:
                        cycle_symbols = self._run_coin_selector()
                        if _do_full:
                            self._selector_last_run_date = now_utc.date()
                        self._selector_last_fast_refresh = now_utc
                        # After first successful selector run, reconcile orphaned crypto positions
                        if not self._crypto_reconciled and self._selector_active_symbols:
                            self._crypto_reconciled = True
                            self._reconcile_positions()
                            # Re-fetch positions after reconciliation closed stale ones
                            positions = self.get_positions()
                    else:
                        cycle_symbols = self._selector_active_symbols
                    # exit_only_symbols (below) handles positions outside cycle_symbols

                elif self.group in ("swing", "intraday") and self._etf_selector is not None:
                    _fast_interval = _FAST_REFRESH_SECS.get(self.group)
                    _do_full = (_should_refresh_selector(self.group, self._etf_selector_last_run_date)
                                or not self._etf_active_symbols)
                    _do_fast = (
                        not _do_full
                        and _fast_interval is not None
                        and (self._etf_selector_last_fast_refresh is None
                             or (now_utc - self._etf_selector_last_fast_refresh).total_seconds()
                             >= _fast_interval)
                    )
                    if _do_full or _do_fast:
                        cycle_symbols = self._run_etf_selector()
                        if _do_full:
                            self._etf_selector_last_run_date = now_utc.date()
                        self._etf_selector_last_fast_refresh = now_utc
                        # After first successful ETF selector run, reconcile stale positions
                        if not self._etf_reconciled and self._etf_active_symbols:
                            self._etf_reconciled = True
                            self._reconcile_positions()
                            positions = self.get_positions()
                    else:
                        cycle_symbols = self._etf_active_symbols
                else:
                    cycle_symbols = list(self.symbols)

                # For crypto ranker: max_positions drives actual sizing, not candidate count.
                # allocation_per_sym is only used for crypto liquidity cap.
                n_sizing = min(len(cycle_symbols), self._risk_config.max_positions)
                allocation_per_sym = account["equity"] / max(n_sizing, 1)

                # Print header
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"=== Paper Trading Cycle #{cycle} ({self.mode} | {session_label}) ===")
                print(f"  Account: ${account['equity']:,.2f} equity | "
                      f"${account['cash']:,.2f} cash | "
                      f"${account['equity'] - account['cash']:,.2f} in positions")
                print(f"  Candidates: {len(cycle_symbols)} | Max positions: {self._risk_config.max_positions}")
                print()

                # Refresh VIX cache once per cycle so all symbols share the same fetch.
                self._vix_cache = None   # force fresh fetch this cycle
                self._get_vix_df()       # populates _vix_cache (or retains old on failure)

                # VIX > 35 hard gate for intraday: model trained on Sep 2025-Mar 2026
                # (VIX 15-30 range). Extreme vol environments are out-of-distribution.
                if self.group == "intraday" and self._vix_cache is not None and not self._vix_cache.empty:
                    try:
                        latest_vix = float(self._vix_cache["vix_close"].iloc[-1])
                        if latest_vix > 35.0:
                            print(f"  [VIX GATE] VIX={latest_vix:.1f} > 35 — "
                                  f"intraday model untrained on extreme vol. Skipping new entries.")
                            log.warning("VIX %.1f > 35: intraday hard gate active, exit-only mode", latest_vix)
                            # Only allow exits on existing positions, no new entries
                            for sym in cycle_symbols:
                                pos = positions.get(sym)
                                if pos and pos.get("qty", 0) != 0:
                                    action = self.check_and_trade(sym, positions, allocation_per_sym, exit_only=True)
                                    print(f"  {sym:>5}:  {action}  [VIX gate exit-only]")
                            continue  # skip to next cycle

                    except (IndexError, KeyError, TypeError):
                        pass  # VIX fetch issue — proceed normally

                # Positions in this account for symbols not in this cycle's active set.
                # Option positions (OCC symbols) are NOT managed here — they are legacy.
                # For crypto with selector: positions from deselected coins are exit-only
                # (they may have been selected last cycle but dropped this cycle).
                active_set = set(cycle_symbols)
                exit_only_symbols = [
                    s for s in positions
                    if positions[s].get("qty", 0) > 0
                    and s not in active_set
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

                # HRP weight computation: correlation-aware allocation (Phase 3)
                _hrp_weights: Dict[str, float] = {}
                try:
                    from hrp_sizer import HRPSizer
                    if len(cycle_symbols) >= 2:
                        _hrp_returns: Dict[str, pd.Series] = {}
                        for _s in cycle_symbols:
                            ctx = self._get_market_context(_s)
                            _p = ctx.get("price", 0)
                            if _p > 0:
                                _hrp_returns[_s] = pd.Series(dtype=float)  # placeholder
                        # Try to get actual returns from bars data
                        import yfinance as yf
                        _yf_syms = [s.replace("/", "-") for s in cycle_symbols[:20]]
                        _bars = yf.download(_yf_syms, period="90d", progress=False, auto_adjust=True)
                        if not _bars.empty and hasattr(_bars.columns, "get_level_values"):
                            _close_df = _bars["Close"]
                            for _s_orig, _s_yf in zip(cycle_symbols[:20], _yf_syms):
                                if _s_yf in _close_df.columns:
                                    _ret = _close_df[_s_yf].pct_change().dropna()
                                    if len(_ret) >= 20:
                                        _hrp_returns[_s_orig] = _ret
                        if len(_hrp_returns) >= 2:
                            _sizer = HRPSizer(lookback=60)
                            _hrp_weights = _sizer.compute_weights(_hrp_returns)
                except Exception as _hrp_exc:
                    log.debug("HRP sizing skipped: %s", _hrp_exc)

                # Check each symbol in this group (cycle_symbols for crypto with selector)
                _entries_this_cycle = 0
                for sym in cycle_symbols:
                    # HRP-adjusted allocation: scale by relative weight
                    if _hrp_weights and sym in _hrp_weights:
                        n_syms = len(cycle_symbols)
                        # HRP weight relative to equal weight (1/n)
                        hrp_scale = _hrp_weights[sym] * n_syms
                        hrp_scale = max(0.3, min(2.0, hrp_scale))  # clamp
                        sym_alloc = allocation_per_sym * hrp_scale
                    else:
                        sym_alloc = allocation_per_sym
                    # Cap allocation by per-coin max order size (liquidity-based)
                    if self.group == "crypto":
                        yf_sym = _crypto_to_yfinance(sym) if "/" in sym else sym
                        coin_cfg = get_coin_cost_config(yf_sym)
                        max_order = coin_cfg.get("max_order_usd", sym_alloc)
                        if sym_alloc > max_order > 0:
                            log.info("Capping %s alloc $%.0f -> $%.0f (liquidity limit)",
                                     sym, sym_alloc, max_order)
                            sym_alloc = max_order
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
                    # Blocked windows: skip new entries, still allow exits
                    if _in_time_window(self._blocked_windows):
                        pos = positions.get(sym)
                        if pos is None or pos.get("qty", 0) == 0:
                            print(f"  {sym:>5}:  SKIP  (blocked window -- low-alpha period)")
                            continue
                    # Half-size windows: reduce allocation during volatile open/close
                    if _in_time_window(self._half_size_windows):
                        sym_alloc = sym_alloc * 0.5
                    had_position = sym in positions and positions[sym].get("qty", 0) != 0
                    action = self.check_and_trade(sym, positions, sym_alloc)
                    print(f"  {sym:>5}:  {action}")
                    # Track new entries so subsequent symbols see updated position count.
                    # Insert a placeholder into positions dict so check_position_allowed
                    # counts it towards max_positions for the rest of this cycle.
                    if not had_position and (action.startswith("LONG") or action.startswith("SHORT")):
                        _entries_this_cycle += 1
                        positions[sym] = {"qty": 1, "current_price": 0, "side": action.split()[0]}

                # Manage out wrongly placed positions (exit-only: stops and flips, no new entries)
                for sym in exit_only_symbols:
                    action = self.check_and_trade(sym, positions, allocation_per_sym, exit_only=True)
                    print(f"  {sym:>5}:  {action}  [EXIT-ONLY]")

                # Microstructure alert checks (per-symbol + market-wide + portfolio)
                try:
                    for sym in cycle_symbols:
                        ctx = self._get_market_context(sym)
                        ctx["symbol"] = sym
                        self._alert_engine.check(market_context=ctx)
                    initial_capital = (self._kraken._paper_initial
                                       if self._kraken is not None
                                       else self._peak_equity or 100_000.0)
                    drawdown_pct = ((account["equity"] - initial_capital) / initial_capital) * 100
                    self._alert_engine.check(portfolio_state={
                        "consecutive_losses": self._consecutive_losses,
                        "drawdown_pct": drawdown_pct,
                    })
                except Exception as exc:
                    log.debug("Alert check failed: %s", exc)

                # Periodic model health report (every ~1h = 12 cycles at 5min)
                if cycle % 12 == 0:
                    for sym in cycle_symbols:
                        health = self._monitor.compute_health(sym)
                        if health.n_trades >= 10:
                            log.info("Model health %s: IC=%.3f  hit_rate=%.1f%%  n=%d",
                                     sym, health.rolling_ic, health.rolling_hit_rate * 100,
                                     health.n_trades)

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
    args.disaster_stop_max_pct = _resolve(getattr(args, 'disaster_stop_max_pct', None), "disaster_stop_max_pct", 0.20)
    args.profit_lock_atr_mult = _resolve(getattr(args, 'profit_lock_atr_mult', None), "profit_lock_atr_mult", 2.0)
    args.profit_lock_trail_atr_mult = _resolve(getattr(args, 'profit_lock_trail_atr_mult', None), "profit_lock_trail_atr_mult", 1.5)
    args.max_underwater_days = _resolve(getattr(args, 'max_underwater_days', None), "max_underwater_days", 90)
    args.loss_cooldown_hours = _resolve(args.loss_cooldown_hours, "loss_cooldown_hours", 4.0)
    args.max_loss_cooldown_hours = _resolve(args.max_loss_cooldown_hours, "max_loss_cooldown_hours", None)
    args.post_loss_size_mult = _resolve(args.post_loss_size_mult, "post_loss_size_mult", 0.5)
    args.post_loss_size_hours = _resolve(args.post_loss_size_hours, "post_loss_size_hours", 4.0)
    args.same_dir_confidence_mult = _resolve(args.same_dir_confidence_mult, "same_dir_confidence_mult", 1.5)
    args.target_vol = cfg.get("target_vol", 0.20)

    # --- Log rotation: RotatingFileHandler (10 MB, 5 backups) per group ---
    from logging.handlers import RotatingFileHandler
    group = args.group  # e.g. "intraday", "swing", "crypto", "all", or None
    _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.path.join(_log_dir, f"paper_trader_{group or 'default'}.log")
    _rfh = RotatingFileHandler(_log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
                               encoding="utf-8")
    _rfh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_rfh)
    log.info("Log rotation enabled: %s (10MB x 5 backups)", _log_file)

    # Resolve API keys: group-specific keys take priority over generic ALPACA_API_KEY
    if group and group != "all":
        env_prefix = f"ALPACA_{group.upper()}_"
        api_key    = os.environ.get(f"{env_prefix}KEY",    os.environ.get("ALPACA_API_KEY", ""))
        api_secret = os.environ.get(f"{env_prefix}SECRET", os.environ.get("ALPACA_API_SECRET", ""))
    else:
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")

    # Crypto groups can run without Alpaca keys (use Kraken executor)
    is_crypto_group = group in ("crypto", "crypto_intraday")
    has_kraken = bool(os.environ.get("KRAKEN_API_KEY")) or is_crypto_group
    if (not api_key or not api_secret) and not (is_crypto_group and has_kraken):
        hint = (f"ALPACA_{group.upper()}_KEY / ALPACA_{group.upper()}_SECRET  OR  "
                if group and group != "all" else "")
        print(f"\n  ERROR: Set {hint}ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        print("  Get free keys at https://app.alpaca.markets/signup\n")
        sys.exit(1)

    # Warn if any symbol is in more than one group (wrong model placement)
    _warn_duplicate_symbols()

    # Guard: intraday-only symbols must not be traded on non-intraday accounts
    INTRADAY_ONLY = set(SYMBOL_GROUPS.get("intraday", [])) - set(SYMBOL_GROUPS.get("swing", [])) - set(SYMBOL_GROUPS.get("crypto", []))
    if group and group not in ("intraday", "crypto_intraday", "all") and args.mode == "intraday":
        print(f"\n  ERROR: --mode intraday requires --group intraday or crypto_intraday (got --group {group}).")
        print("  Intraday mode must use the correct account to avoid cross-account contamination.\n")
        sys.exit(1)
    if group and group not in ("intraday", "all") and args.symbols:
        bad = [s for s in args.symbols.split(",") if s.strip().upper() in INTRADAY_ONLY]
        if bad:
            print(f"\n  ERROR: Symbols {bad} are intraday-only and cannot be traded on the '{group}' account.")
            print(f"  Use --group intraday instead.\n")
            sys.exit(1)

    # Resolve symbols: --symbols > --group (dynamic for swing) > full DEFAULT_UNIVERSE
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif group and group != "all":
        if group == "swing":
            symbols = _resolve_swing_symbols()
        elif group == "intraday":
            symbols = _resolve_intraday_symbols()
        else:
            symbols = SYMBOL_GROUPS[group]
        log.info("Account group '%s': initial/fallback symbols %s (dynamic ranker overrides each cycle)", group, symbols)
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
        disaster_stop_max_pct=args.disaster_stop_max_pct,
        profit_lock_atr_mult=args.profit_lock_atr_mult,
        profit_lock_trail_atr_mult=args.profit_lock_trail_atr_mult,
        max_underwater_days=int(args.max_underwater_days),
        loss_cooldown_hours=args.loss_cooldown_hours if args.mode != "intraday" else min(args.loss_cooldown_hours, 0.5),
        max_loss_cooldown_hours=args.max_loss_cooldown_hours,
        post_loss_size_mult=args.post_loss_size_mult,
        post_loss_size_hours=args.post_loss_size_hours,
        same_dir_confidence_mult=args.same_dir_confidence_mult,
        target_vol=args.target_vol,
        legacy_stricter_exit=legacy_stricter,
        legacy_no_new_entries=legacy_no_new,
        group=group,
        blocked_windows=cfg.get("blocked_windows", []),
        half_size_windows=cfg.get("half_size_windows", []),
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
