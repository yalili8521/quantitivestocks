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
    validate_model_mode, get_symbol_cap, is_symbol_disabled,
    get_confidence_multiplier, DeRiskState, evaluate_derisk,
)
from cost_model import validate_cost_threshold
from model_monitor import ModelMonitor
from coin_selector import CoinSelector, CRYPTO_UNIVERSE, fetch_universe_data
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
    # Account 1 — Intraday LightGBM v2
    # OOS Sharpe (90d): SMH 3.80, SOXX 2.24, IWM 2.05, QQQ 1.60, IGV 1.44
    # EWT disabled (0 trades in OOS)
    "intraday":  ["SMH", "IWM", "IGV", "QQQ", "SOXX"],
    # Account 2 — Swing XGBoost+TFT (daily regression)
    # OOS Sharpe (2024→): GLD 1.55, IBIT 2.20, SLV 1.11, SMH 0.72, QQQ 0.65, GDX 0.59, IGV 0.54
    # XLK disabled (Sharpe 0.19, near-zero edge)
    "swing":     ["GDX", "SLV", "IGV", "QQQ", "GLD", "SMH", "IBIT"],
    # Account 3 — Crypto (heavily pruned by OOS)
    # OOS Sharpe (2025→): CRV 1.07, AVAX 0.54, ADA 0.49, LINK 0.48
    # Disabled (negative Sharpe): BTC, ETH, SOL, DOGE, DOT, SUSHI, AAVE, RENDER
    "crypto": ["CRV/USD", "AVAX/USD", "ADA/USD", "LINK/USD"],
    # Account 4 — Crypto Intraday LGB+GRU (5-min bars, 1-hour horizon)
    # OOS Sharpe (45d): ATOM 2.63, BCH 2.13, EIGEN 1.75, SAND 1.30, VET 1.30, ETC 1.29
    # Uses coin selector (top-6) + CryptoIntradayPredictor
    "crypto_intraday": ["ATOM/USD", "BCH/USD", "LTC/USD", "VET/USD"],
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
      1. promoted_symbols.json (OOS-validated symbols from batch-backtest)
         If > 15 promoted, run ETFSelector to pick top-10
      2. etf_universe.json (full screened universe)
      3. Hardcoded fallback (SYMBOL_GROUPS["swing"])
    """
    try:
        from etf_screener import load_promoted_symbols, load_etf_universe
        promoted = load_promoted_symbols(SWING_MODEL_DIR)
        if promoted:
            # If pool is large, apply Layer 1 selector to pick top-K
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
        self.disaster_stop_atr_mult = disaster_stop_atr_mult
        self.disaster_stop_max_pct = disaster_stop_max_pct
        self.profit_lock_atr_mult = profit_lock_atr_mult
        self.profit_lock_trail_atr_mult = profit_lock_trail_atr_mult
        self.max_underwater_days = max_underwater_days

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
        self._entry_times: Dict[str, datetime] = {}
        self._profit_lock_active: Dict[str, bool] = {}
        self._cooldown_until: Dict[str, datetime] = {}
        self._decay_cooldown_until: Dict[str, datetime] = {}  # re-entry block after signal-decay exit
        # Opt #1/#2/#5: track max-loss exits per symbol (direction + time)
        self._max_loss_exits: Dict[str, dict] = {}

        self._vix_cache: Optional[pd.DataFrame] = None

        # Crypto intraday data source (lazy init)
        self._crypto_intraday_data = None

        # Regime filter state
        from collections import deque
        self._recent_trade_wins: deque = deque(maxlen=20)
        self._regime_cooldown_until: Optional[datetime] = None

        # Alert engine for microstructure alerts
        self._alert_engine = AlertEngine()
        self._consecutive_losses: int = 0
        self._warned_shared_crypto: bool = False  # one-time warn if intraday account has crypto positions

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
        # Prefer dynamic universe from Layer 0 screener; fall back to hardcoded
        dynamic_universe = load_universe(CRYPTO_MODEL_DIR)
        self._selector_universe = dynamic_universe if dynamic_universe else list(CRYPTO_UNIVERSE)
        self._selector_active_symbols: List[str] = []    # dynamically selected coins
        self._selector_rank_scores: Dict[str, float] = {}  # symbol → rank score
        self._selector_last_run: Optional[datetime] = None  # gate selector to once/day
        if group in ("crypto", "crypto_intraday"):
            try:
                self._coin_selector = CoinSelector(model_dir=CRYPTO_MODEL_DIR)
                log.info("Coin selector loaded — dynamic symbol selection enabled")
            except FileNotFoundError:
                log.info("No trained coin selector found — using static symbol list")

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

        # Model monitor: tracks predicted vs realized returns
        os.makedirs(MONITOR_DIR, exist_ok=True)
        self._monitor = ModelMonitor(output_dir=MONITOR_DIR, window=60)
        self._entry_predictions: Dict[str, float] = {}  # symbol → predicted return at entry
        self._derisk_states: Dict[str, DeRiskState] = {}  # symbol → rolling perf tracker

        # Volatility-tier exit params (classified at startup, recomputed weekly)
        from src.risk_config import (VolTier, ExitParams, classify_vol_tier,
                                     compute_vol_metrics, get_exit_params)
        self._vol_tiers: Dict[str, VolTier] = {}
        self._exit_params: Dict[str, ExitParams] = {}
        self._vol_tier_last_update: Optional[datetime] = None
        self._breakeven_floor: Dict[str, float] = {}  # symbol → floor price (entry or higher)
        self._signal_flip_counts: Dict[str, int] = {}  # consecutive signal flips
        self._classify_vol_tiers()

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
        for sym in self.symbols:
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
            try:
                from intraday_model import IntradayPredictor
                return IntradayPredictor(symbol, model_dir=group_model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No intraday LightGBM for %s, falling back to LSTM: %s", symbol, exc)
        elif group == "swing":
            try:
                from swing_model import SwingPredictor
                return SwingPredictor(symbol, model_dir=group_model_dir)
            except (FileNotFoundError, ImportError) as exc:
                log.info("No swing model for %s, falling back to LSTM: %s", symbol, exc)
        elif group == "crypto":
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
        try:
            return Predictor(symbol, model_dir=model_dir,
                             mode=mode, intraday_interval=intraday_interval)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("No trained model for %s (%s): %s — will skip ML signals.",
                        symbol, mode, exc)
            return None

    def _get_model_universe(self) -> Set[str]:
        """Return yfinance-format symbols that have a trained XGBoost model on disk."""
        import glob as _glob
        pattern = os.path.join(CRYPTO_MODEL_DIR, "*_xgb_swing_config.json")
        model_syms: Set[str] = set()
        for path in _glob.glob(pattern):
            fname = os.path.basename(path)
            # e.g. "BTC-USD_xgb_swing_config.json" → "BTC-USD"
            sym = fname.replace("_xgb_swing_config.json", "")
            model_syms.add(sym)
        return model_syms

    def _load_oos_registries(self) -> tuple:
        """Load OOS Sharpe and performance registries from config/trading.json.
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
                sharpe_reg = cfg.get("oos_sharpe_registry", {})
                perf_reg = cfg.get("oos_performance_registry", {})
                return sharpe_reg, perf_reg
        except Exception as exc:
            log.warning("Failed to load OOS registries: %s", exc)
        return {}, {}

    def _compute_composite_scores(
        self,
        rankings: list,
        oos_sharpe: dict,
        oos_perf: dict,
        w_selector: float = 0.3,
        w_sharpe: float = 0.4,
        w_return: float = 0.3,
    ) -> Dict[str, float]:
        """Compute composite score = w1*selector + w2*sharpe + w3*return (all normalized).

        Only includes coins with OOS Sharpe > 0.
        Returns {symbol: composite_score} sorted descending.
        """
        # Collect raw values for coins with positive OOS Sharpe
        candidates = []
        for sym, sel_score in rankings:
            sharpe = oos_sharpe.get(sym, -999)
            if sharpe <= 0:
                continue
            perf = oos_perf.get(sym, {})
            ret = perf.get("return_pct", 0.0) if isinstance(perf, dict) else 0.0
            candidates.append((sym, sel_score, sharpe, ret))

        if not candidates:
            return {}

        # Normalize each dimension to [0, 1] using rank-percentile (robust to outliers)
        def _rank_normalize(values):
            n = len(values)
            if n <= 1:
                return [1.0] * n
            order = sorted(range(n), key=lambda i: values[i])
            ranks = [0.0] * n
            for rank_pos, idx in enumerate(order):
                ranks[idx] = rank_pos / (n - 1)
            return ranks

        sel_scores = [c[1] for c in candidates]
        sharpe_vals = [c[2] for c in candidates]
        return_vals = [c[3] for c in candidates]

        sel_norm = _rank_normalize(sel_scores)
        sharpe_norm = _rank_normalize(sharpe_vals)
        return_norm = _rank_normalize(return_vals)

        result = {}
        for i, (sym, sel, sharpe, ret) in enumerate(candidates):
            composite = (w_selector * sel_norm[i]
                         + w_sharpe * sharpe_norm[i]
                         + w_return * return_norm[i])
            result[sym] = round(composite, 4)

        # Sort descending
        result = dict(sorted(result.items(), key=lambda x: -x[1]))
        return result

    # -- Coin selector (Layer 1 → dynamic symbol selection) -------------
    def _run_coin_selector(self) -> List[str]:
        """Run Layer 1 cross-sectional selector to pick today's top-K crypto coins.

        Only selects from model_universe (coins with trained models).
        Returns Alpaca-format symbols (BTC/USD) for selected coins.
        Falls back to static self.symbols if selector unavailable or fails.
        """
        if self._coin_selector is None:
            return list(self.symbols)

        try:
            # Restrict selector input to coins that have trained models
            model_universe = self._get_model_universe()
            selector_input = [s for s in self._selector_universe if s in model_universe]
            if len(selector_input) < 3:
                log.warning("Model universe too small (%d coins) — using static list",
                            len(selector_input))
                self._composite_scores = {}  # clear stale scores
                return list(self.symbols)

            log.info("Selector input: %d coins (model_universe) out of %d (screen_universe)",
                     len(selector_input), len(self._selector_universe))

            # Fetch data for model_universe only
            data = fetch_universe_data(selector_input, lookback_days=400)
            if len(data) < 3:
                log.warning("Selector: only %d coins with data — falling back to static list", len(data))
                self._composite_scores = {}
                return list(self.symbols)

            result = self._coin_selector.rank(data)
            if not result.selected:
                log.warning("Selector returned empty selection — using static list")
                self._composite_scores = {}
                return list(self.symbols)

            # Layer 2: composite scoring (selector + OOS Sharpe + OOS return)
            oos_sharpe, oos_perf = self._load_oos_registries()
            composite = self._compute_composite_scores(
                result.rankings, oos_sharpe, oos_perf,
            )

            if not composite:
                log.warning("No coins passed composite scoring — using static list")
                self._composite_scores = {}
                return list(self.symbols)

            # Select top-K by composite score
            top_k = self._coin_selector.top_k
            selected = list(composite.keys())[:top_k]

            # Log composite breakdown
            for sym in selected:
                sel_score = dict(result.rankings).get(sym, 0)
                sharpe = oos_sharpe.get(sym, 0)
                ret = oos_perf.get(sym, {}).get("return_pct", 0) if isinstance(oos_perf.get(sym), dict) else 0
                log.info("  Composite %s: %.3f (sel=%.2f, sharpe=%.2f, ret=%.1f%%)",
                         sym, composite[sym], sel_score, sharpe, ret)

            # Convert yfinance format (BTC-USD) to Alpaca format (BTC/USD)
            selected_alpaca = [s.replace("-", "/") for s in selected]

            # Store composite scores for rank-weighted sizing
            self._composite_scores = composite

            # Ensure predictors exist for newly selected coins
            _selector_model_dir = get_model_dir(self.group)
            for sym in selected_alpaca:
                if sym not in self.predictors:
                    predictor = self._create_predictor(
                        sym, self.group, _selector_model_dir,
                        self.mode, self.intraday_interval,
                    )
                    self.predictors[sym] = predictor
                    if predictor is not None:
                        log.info("Loaded predictor for newly selected %s", sym)

            log.info("Selector chose %d coins: %s",
                     len(selected_alpaca), ", ".join(selected_alpaca))
            self._selector_active_symbols = selected_alpaca
            return selected_alpaca

        except Exception as exc:
            log.error("Coin selector failed: %s — using static list", exc)
            self._composite_scores = {}  # clear stale scores
            return list(self.symbols)

    def _get_rank_scalar(self, symbol: str) -> float:
        """Get composite-score-based sizing scalar.

        Maps composite score to position weight:
        - Highest composite → 1.0 (full size)
        - Lowest in selected set → 0.5 (half size)
        - Not in set → 0.3 (minimum probe)
        """
        if not hasattr(self, "_composite_scores") or not self._composite_scores:
            return 1.0
        yf_sym = _crypto_to_yfinance(symbol)
        scores = self._composite_scores
        if yf_sym not in scores:
            return 0.3

        # Normalize within selected set: top → 1.0, bottom → 0.5
        vals = list(scores.values())
        max_score = max(vals) if vals else 1.0
        min_score = min(vals) if vals else 0.0
        score_range = max_score - min_score

        if score_range < 1e-6:
            return 0.75  # all equal

        normalized = (scores[yf_sym] - min_score) / score_range  # 0 to 1
        return 0.5 + 0.5 * normalized  # 0.5 to 1.0

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
            vix_df = self._vix_cache if self._vix_cache is not None else self._get_vix_df()
            return predictor.predict(bars, vix_df)
        except Exception as exc:
            log.error("Prediction failed for %s: %s", symbol, exc)
            return {"direction": "UNKNOWN", "confidence": 0.0, "probability": 0.5}

    def _get_crypto_intraday_prediction(self, symbol: str, predictor) -> dict:
        """Fetch 5-min bars from Kraken and run CryptoIntradayPredictor."""
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

        # Wrongly placed: symbol has no position here — nothing to manage
        if exit_only and not has_position:
            return "EXIT-ONLY  (no position — symbol not in this group)"

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

            # ===== LAYER 1: Hard safety (disaster stop) =====
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

                # Arm profit-lock when PnL exceeds threshold
                if pnl_pct >= arm_pct:
                    if not self._profit_lock_active.get(symbol, False):
                        log.info("Profit-lock ARMED for %s at PnL=%.2f%% (arm=%.2f%%)",
                                 symbol, pnl_pct * 100, arm_pct * 100)
                    self._profit_lock_active[symbol] = True

                profit_lock_armed = self._profit_lock_active.get(symbol, False)

                if profit_lock_armed:
                    # 2a: Profit-lock armed + model flips + still profitable → exit immediately
                    signal_flipped = ((side == "LONG" and expected_return <= 0)
                                      or (side == "SHORT" and expected_return >= 0))
                    if signal_flipped and pnl_pct > 0:
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

            # ===== LAYER 3: Model-state exits (signal decay) =====
            if not use_legacy_stops:
                signal_flipped = ((side == "LONG" and expected_return <= 0)
                                  or (side == "SHORT" and expected_return >= 0))
                if signal_flipped:
                    self._signal_flip_counts[symbol] = self._signal_flip_counts.get(symbol, 0) + 1
                else:
                    self._signal_flip_counts[symbol] = 0

                flip_count = self._signal_flip_counts.get(symbol, 0)
                required_flips = ep.signal_flip_consecutive

                if flip_count >= required_flips:
                    reason = (f"signal_decay (E[r]={expected_return:+.4f}, "
                              f"flips={flip_count}/{required_flips})")
                    result = _do_exit(reason, "signal_decay")
                    if result is None:
                        return (f"EXIT-DEFERRED  (signal decay E[r]={expected_return:+.4f}, "
                                f"P&L={pnl_pct:+.2%}, market closed)")
                    self._decay_cooldown_until[symbol] = (
                        datetime.now(timezone.utc) + timedelta(seconds=2 * self.check_interval))
                    return result

            # ===== LAYER 4: Time exits =====
            # 4a: Max underwater duration
            if not use_legacy_stops and ep.max_underwater_days > 0:
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

            # 4b: Max hold hours (crypto_intraday: 4h max hold)
            if self.group == "crypto_intraday":
                entry_time = self._entry_times.get(symbol)
                max_hold_hours = 4.0
                if entry_time:
                    hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                    if hours_held >= max_hold_hours:
                        reason = f"max_hold_{max_hold_hours}h (held {hours_held:.1f}h, PnL={pnl_pct:+.2%})"
                        result = _do_exit(reason, "time")
                        if result is None:
                            return f"EXIT-DEFERRED  (max hold {hours_held:.1f}h)"
                        return result

            # HOLD
            hold_extras = []
            if self._profit_lock_active.get(symbol, False):
                hold_extras.append("PL=armed")
            if symbol in self._breakeven_floor:
                hold_extras.append(f"BE=${self._breakeven_floor[symbol]:.2f}")
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

        # Determine entry direction
        enter_dir = None
        if expected_return > self.cost_threshold and bar_trend > 0:
            enter_dir = "LONG"
        elif expected_return < -self.cost_threshold and bar_trend < 0:
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

            # --- Position sizing: half-Kelly × signal strength, capped by risk limits ---
            # Size from TOTAL equity (not per-symbol sub-allocation).
            risk = self._risk_config
            try:
                account = self.get_account_summary()
                equity = account["equity"]
                self._last_known_equity = equity  # cache for fallback
            except Exception:
                equity = getattr(self, '_last_known_equity', self.initial_capital)

            signal_pct = min(1.0, max(0.1, abs(expected_return) / self.target_return))

            # Half-Kelly from rolling trade history (falls back to position_pct)
            derisk_key = symbol.replace("/", "-")
            derisk_state = self._derisk_states.get(derisk_key)
            kelly_f = None
            if derisk_state is not None:
                kelly_f = derisk_state.half_kelly(
                    window=risk.kelly_window, min_trades=risk.kelly_min_trades
                )
            if kelly_f is not None:
                base_frac = min(kelly_f, risk.kelly_cap)
                size_source = f"kelly={kelly_f:.3f}"
            else:
                base_frac = risk.position_pct
                size_source = "fixed"

            sizing_pct = base_frac * signal_pct
            sizing_pct = min(sizing_pct, risk.max_position_pct)

            # --- Per-symbol cap (OOS Sharpe-tiered) ---
            sym_cap = get_symbol_cap(symbol, self.group or "swing")
            if sym_cap <= 0.0:
                return (f"DISABLED  ({symbol} cap=0% by OOS performance)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")
            sizing_pct = min(sizing_pct, sym_cap)

            # --- Confidence multiplier (OOS Sharpe tier) ---
            conf_mult = get_confidence_multiplier(symbol)
            if conf_mult <= 0.0:
                return (f"DISABLED  ({symbol} conf_mult=0 by negative OOS Sharpe)  "
                        f"ML: {direction} E[r]={expected_return:+.4f}")
            sizing_pct *= conf_mult

            # --- Rank-based sizing (Layer 1 selector, crypto only) ---
            if self.group == "crypto" and self._coin_selector is not None:
                rank_scalar = self._get_rank_scalar(symbol)
                sizing_pct *= rank_scalar

            # --- Auto de-risking check ---
            size_note = f" [{size_source}]"
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

            # Opt #5: reduce sizing after a recent max-loss in this symbol
            if last_loss:
                loss_age_h = (datetime.now(timezone.utc) - last_loss["time"]).total_seconds() / 3600
                if loss_age_h < self.post_loss_size_hours:
                    sizing_pct *= self.post_loss_size_mult
                    size_note += f" [post-loss {self.post_loss_size_mult:.0%}]"

            invest = equity * sizing_pct
            qty = invest / current_price
            if is_crypto:
                qty = round(qty, 6)
            else:
                qty = int(qty)
            if qty <= 0:
                return f"SKIP  (insufficient allocation)  ML: {direction} E[r]={expected_return:+.4f}"

            # Portfolio constraint check (equity already fetched above)
            try:
                allowed, constraint_reason = check_position_allowed(
                    symbol, invest, equity, positions, risk
                )
                if not allowed:
                    return (f"CONSTRAINT-BLOCK  ({constraint_reason})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")
                # Theme cap check
                theme_ok, theme_reason = check_theme_cap(
                    symbol, invest, equity, positions
                )
                if not theme_ok:
                    return (f"THEME-CAP  ({theme_reason})  "
                            f"ML: {direction} E[r]={expected_return:+.4f}")
            except Exception as exc:
                log.debug("Portfolio constraint check failed: %s", exc)

            if enter_dir == "LONG":
                self.buy(symbol, qty, limit_price=current_price)
            else:
                self.sell_short(symbol, qty, limit_price=current_price)

            self._peak_prices[symbol] = current_price
            self._entry_atrs[symbol] = bar_atr
            self._entry_times[symbol] = datetime.now(timezone.utc)
            self._profit_lock_active[symbol] = False

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
        self._entry_times.pop(symbol, None)
        self._profit_lock_active.pop(symbol, None)
        self._breakeven_floor.pop(symbol, None)
        self._signal_flip_counts.pop(symbol, None)
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

        # Build the valid symbol set for this group
        valid_symbols: set = set(self.symbols)
        # For crypto groups with a selector, also include the full selector universe
        # (positions might belong to coins not selected *this* cycle but still valid)
        if self.group in ("crypto", "crypto_intraday") and self._selector_universe:
            # Normalize: universe uses "BTC-USD", positions use "BTC/USD"
            for s in self._selector_universe:
                valid_symbols.add(s)
                valid_symbols.add(s.replace("-", "/"))

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
                if side == "LONG":
                    self.sell(sym, qty, reason="reconcile_stale")
                else:
                    self.buy_to_cover(sym, qty, reason="reconcile_stale")
                log.info("Reconciled %s %s %.6f (stale position closed)", side, sym, qty)
            except Exception as exc:
                log.error("Failed to close stale position %s: %s", sym, exc)
        print()

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
            if self.group == "crypto":
                session = "regular"
                session_label = "24/7 CRYPTO"
            elif self.group == "crypto_intraday":
                session = "regular"
                session_label = "24/7 CRYPTO INTRADAY"
            elif self.group == "swing":
                session = _get_session()
                session_label = f"SWING ({session.upper()})"
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

                # Layer 1: dynamic coin selection (crypto groups, once per cycle)
                # Runs the cross-sectional selector to pick top-K coins.
                # For non-crypto groups, cycle_symbols == self.symbols (static).
                if self.group in ("crypto", "crypto_intraday") and self._coin_selector is not None:
                    now_utc = datetime.now(timezone.utc)
                    elapsed = ((now_utc - self._selector_last_run).total_seconds()
                               if self._selector_last_run else float("inf"))
                    if elapsed >= 86400 or not self._selector_active_symbols:
                        cycle_symbols = self._run_coin_selector()
                        self._selector_last_run = now_utc
                    else:
                        cycle_symbols = self._selector_active_symbols
                else:
                    cycle_symbols = list(self.symbols)

                allocation_per_sym = account["equity"] / max(len(cycle_symbols), 1)

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

                # Check each symbol in this group (cycle_symbols for crypto with selector)
                for sym in cycle_symbols:
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
                    action = self.check_and_trade(sym, positions, sym_alloc)
                    print(f"  {sym:>5}:  {action}")

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
                    initial_capital = 100_000.0
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

    # Resolve API keys: group-specific keys take priority over generic ALPACA_API_KEY
    group = args.group  # e.g. "intraday", "swing", "crypto", "all", or None
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

    # Resolve symbols: --symbols > --group (dynamic for swing) > full DEFAULT_UNIVERSE
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif group and group != "all":
        if group == "swing":
            symbols = _resolve_swing_symbols()
        else:
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
        disaster_stop_max_pct=args.disaster_stop_max_pct,
        profit_lock_atr_mult=args.profit_lock_atr_mult,
        profit_lock_trail_atr_mult=args.profit_lock_trail_atr_mult,
        max_underwater_days=int(args.max_underwater_days),
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
