#!/usr/bin/env python3
"""
Symbol Selection Pipeline — Train, Backtest, Classify, Update Config
====================================================================

Automates the "candidate → backtest → decide → config" workflow:
  1. Loops over candidate symbols per sleeve (swing-etf, crypto, intraday).
  2. Trains each model with a clean OOS boundary (--train-end).
  3. Backtests on OOS data (post-training period).
  4. Reads metrics from backtest_*_summary.json.
  5. Classifies each symbol as core / secondary / disabled.
  6. Emits a JSON snippet for config/trading.json.

Usage:
    python main.py select-symbols --sleeve swing-etf
    python main.py select-symbols --sleeve crypto
    python main.py select-symbols --sleeve intraday
    python main.py select-symbols --sleeve all
    python main.py select-symbols --sleeve swing-etf --dry-run    # no training, read existing summaries
    python main.py select-symbols --sleeve swing-etf --apply      # write changes to trading.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "trading.json")
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
MAIN_PY = os.path.join(PROJECT_ROOT, "main.py")
TRAIN_TIMEOUT_SECONDS = int(os.environ.get("SELECT_SYMBOLS_TRAIN_TIMEOUT", "1800"))
BACKTEST_TIMEOUT_SECONDS = int(os.environ.get("SELECT_SYMBOLS_BACKTEST_TIMEOUT", "1800"))

# ---------------------------------------------------------------------------
# Candidate symbol lists — extend these to screen new symbols
# ---------------------------------------------------------------------------

CANDIDATE_SWING_ETFS: List[str] = [
    # Current lineup (re-evaluated each cycle)
    "GLD", "SLV", "GDX", "SMH", "QQQ", "IGV", "IBIT",
    # New candidates to screen
    "SPY", "VOO", "VTI", "RSP",
    "XLF", "XLI", "XLV", "XLP", "XLU", "XLB", "XLE",
    "IBB", "XBI", "EWT", "EEM", "EWJ", "EWS",
    "MCHI", "INDA", "FXI",
    "TLT", "LQD", "HYG",
    "USO", "UNG",
    "ARKK", "ARKG",
    "SOXX", "XLK",
]

CANDIDATE_CRYPTOS: List[str] = [
    # Current lineup
    "BTC-USD", "ETH-USD", "SOL-USD",
    "CRV-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
    # Extended candidates
    "DOGE-USD", "DOT-USD", "SUSHI-USD", "AAVE-USD", "RENDER-USD",
    "NEAR-USD", "SHIB-USD",
]

CANDIDATE_INTRADAY_ETFS: List[str] = [
    # Current lineup
    "SMH", "IWM", "IGV", "QQQ", "SOXX",
    # New candidates
    "SPY", "EWT", "XLK", "GDX", "SLV", "EEM",
    "XLE", "USO", "GLD",
]


# ---------------------------------------------------------------------------
# Promotion rules
# ---------------------------------------------------------------------------

@dataclass
class PromotionRules:
    """Numeric thresholds for classifying symbols into buckets."""
    # Core
    core_min_sharpe: float = 0.70
    core_max_dd: float = -20.0       # percent (e.g. -20 means -20%)
    core_min_trades: int = 15
    core_min_winrate: float = 0.50
    # Secondary
    secondary_min_sharpe: float = 0.30
    secondary_max_dd: float = -25.0
    secondary_min_trades: int = 10
    secondary_min_winrate: float = 0.45
    # Everything else → disabled

    # Per-sleeve overrides
    @classmethod
    def for_sleeve(cls, sleeve: str) -> "PromotionRules":
        if sleeve == "crypto":
            return cls(
                core_min_sharpe=0.70,
                core_max_dd=-25.0,      # crypto is more volatile
                core_min_trades=10,
                core_min_winrate=0.50,
                secondary_min_sharpe=0.25,
                secondary_max_dd=-30.0,
                secondary_min_trades=8,
                secondary_min_winrate=0.40,
            )
        elif sleeve == "intraday":
            return cls(
                core_min_sharpe=1.50,    # intraday needs higher Sharpe to justify effort
                core_max_dd=-5.0,
                core_min_trades=20,
                core_min_winrate=0.50,
                secondary_min_sharpe=0.70,
                secondary_max_dd=-10.0,
                secondary_min_trades=10,
                secondary_min_winrate=0.45,
            )
        else:  # swing-etf
            return cls()  # defaults


# Sharpe → max equity cap mapping
SHARPE_TO_MAX_EQUITY = [
    (1.50, 0.12),  # Sharpe >= 1.5 → 12%
    (1.00, 0.10),  # 1.0-1.5 → 10%
    (0.70, 0.07),  # 0.7-1.0 → 7%
    (0.50, 0.05),  # 0.5-0.7 → 5%
    (0.30, 0.03),  # 0.3-0.5 → 3%
    (0.00, 0.01),  # 0.0-0.3 → 1% probe
]

CRYPTO_SHARPE_TO_MAX_EQUITY = [
    (1.00, 0.05),
    (0.70, 0.04),
    (0.50, 0.03),
    (0.25, 0.02),
    (0.00, 0.01),
]


def sharpe_to_cap(sharpe: float, sleeve: str) -> float:
    """Map OOS Sharpe to max equity cap."""
    table = CRYPTO_SHARPE_TO_MAX_EQUITY if sleeve == "crypto" else SHARPE_TO_MAX_EQUITY
    for min_sharpe, cap in table:
        if sharpe >= min_sharpe:
            return cap
    return 0.00


@dataclass
class SymbolResult:
    """OOS backtest result for a candidate symbol."""
    symbol: str
    sleeve: str
    sharpe: float
    max_dd: float
    total_trades: int
    win_rate: float
    total_return: float
    profit_factor: Optional[float]
    bucket: str = "disabled"
    max_equity: float = 0.00
    notes: str = ""

    def classify(self, rules: PromotionRules) -> None:
        """Apply promotion rules to set bucket and max_equity."""
        if (self.sharpe >= rules.core_min_sharpe
                and self.max_dd >= rules.core_max_dd
                and self.total_trades >= rules.core_min_trades
                and self.win_rate >= rules.core_min_winrate):
            self.bucket = "core"
        elif (self.sharpe >= rules.secondary_min_sharpe
                and self.max_dd >= rules.secondary_max_dd
                and self.total_trades >= rules.secondary_min_trades
                and self.win_rate >= rules.secondary_min_winrate):
            self.bucket = "secondary"
        else:
            self.bucket = "disabled"
            self.notes = self._failure_reason(rules)

        self.max_equity = sharpe_to_cap(self.sharpe, self.sleeve) if self.bucket != "disabled" else 0.00

    def _failure_reason(self, rules: PromotionRules) -> str:
        reasons = []
        if self.sharpe < rules.secondary_min_sharpe:
            reasons.append(f"Sharpe {self.sharpe:.2f} < {rules.secondary_min_sharpe}")
        if self.max_dd < rules.secondary_max_dd:
            reasons.append(f"MaxDD {self.max_dd:.1f}% < {rules.secondary_max_dd:.0f}%")
        if self.total_trades < rules.secondary_min_trades:
            reasons.append(f"trades {self.total_trades} < {rules.secondary_min_trades}")
        if self.win_rate < rules.secondary_min_winrate:
            reasons.append(f"WR {self.win_rate:.1%} < {rules.secondary_min_winrate:.0%}")
        return "; ".join(reasons) if reasons else "below thresholds"


# ---------------------------------------------------------------------------
# Training + backtest orchestration
# ---------------------------------------------------------------------------

def _sleeve_config(sleeve: str) -> dict:
    """Return training/backtest parameters per sleeve."""
    if sleeve == "crypto":
        return {
            "train_cmd": "train-crypto",
            "train_end": "2025-01-01",
            "backtest_start": "2025-01-01",
            "model_type": "swing",
            "model_dir": os.path.join(PROJECT_ROOT, "models", "crypto"),
            "mode": "daily",
        }
    elif sleeve == "intraday":
        return {
            "train_cmd": "train-intraday",
            "train_end": None,  # intraday uses internal 80/20 split
            "backtest_start": "2025-12-01",  # Yahoo 5min ~90 days
            "model_type": "intraday",
            "model_dir": None,
            "mode": "intraday",
        }
    else:  # swing-etf
        return {
            "train_cmd": "train-swing",
            "train_end": "2024-01-01",
            "backtest_start": "2024-01-01",
            "model_type": "swing",
            "model_dir": None,
            "mode": "daily",
        }


def train_symbol(symbol: str, sleeve: str) -> bool:
    """Train model for a single symbol. Returns True on success."""
    cfg = _sleeve_config(sleeve)

    if sleeve == "crypto":
        # train-crypto hardcodes its own symbol list; train individually via train-swing
        cmd = [PYTHON, MAIN_PY, "train-swing",
               "--symbols", symbol,
               "--provider", "yahoo",
               "--save-dir", cfg["model_dir"]]
        if cfg["train_end"]:
            cmd += ["--train-end", cfg["train_end"]]
    elif sleeve == "intraday":
        cmd = [PYTHON, MAIN_PY, "train-intraday",
               "--symbols", symbol,
               "--provider", "yahoo"]
    else:
        cmd = [PYTHON, MAIN_PY, "train-swing",
               "--symbols", symbol,
               "--provider", "yahoo"]
        if cfg["train_end"]:
            cmd += ["--train-end", cfg["train_end"]]

    log.info("Training %s (%s): %s", symbol, sleeve, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            log.error("Training failed for %s: %s", symbol, result.stderr[-500:] if result.stderr else "")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("Training timed out for %s after %ss", symbol, TRAIN_TIMEOUT_SECONDS)
        return False


def backtest_symbol(symbol: str, sleeve: str) -> bool:
    """Run OOS backtest for a single symbol. Returns True on success."""
    cfg = _sleeve_config(sleeve)

    cmd = [PYTHON, MAIN_PY, "backtest",
           "--symbol", symbol,
           "--start", cfg["backtest_start"],
           "--model", cfg["model_type"]]

    if cfg["mode"] == "intraday":
        cmd += ["--mode", "intraday", "--interval", "5min"]

    if cfg["model_dir"]:
        cmd += ["--model-dir", cfg["model_dir"]]

    log.info("Backtesting %s (%s): %s", symbol, sleeve, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BACKTEST_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            log.error("Backtest failed for %s: %s", symbol, result.stderr[-500:] if result.stderr else "")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("Backtest timed out for %s after %ss", symbol, BACKTEST_TIMEOUT_SECONDS)
        return False


def read_backtest_summary(symbol: str, sleeve: str = "swing-etf") -> Optional[dict]:
    """Read backtest_*_summary.json for a symbol, preferring mode-specific file."""
    # Intraday has a mode-specific suffix
    if sleeve == "intraday":
        path = os.path.join(OUTPUTS_DIR, f"backtest_{symbol}_intraday_summary.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log.error("Failed to read summary for %s: %s", symbol, e)

    # Fall back to non-suffixed (daily / swing / crypto)
    path = os.path.join(OUTPUTS_DIR, f"backtest_{symbol}_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Verify mode matches sleeve expectation
        file_mode = data.get("mode", "daily")
        if sleeve == "intraday" and file_mode != "intraday":
            log.warning("Summary for %s is mode=%s but sleeve=intraday; skipping", symbol, file_mode)
            return None
        if sleeve != "intraday" and file_mode == "intraday":
            log.warning("Summary for %s is mode=intraday but sleeve=%s; skipping", symbol, sleeve)
            return None
        return data
    except (json.JSONDecodeError, IOError) as e:
        log.error("Failed to read summary for %s: %s", symbol, e)
        return None


def evaluate_symbol(symbol: str, sleeve: str, summary: dict) -> SymbolResult:
    """Create a SymbolResult from backtest summary and classify it."""
    result = SymbolResult(
        symbol=symbol,
        sleeve=sleeve,
        sharpe=summary.get("sharpe_ratio", 0.0),
        max_dd=summary.get("max_drawdown_pct", 0.0),
        total_trades=summary.get("total_trades", 0),
        win_rate=summary.get("win_rate", 0.0),
        total_return=summary.get("total_return_pct", 0.0),
        profit_factor=summary.get("profit_factor"),
    )
    rules = PromotionRules.for_sleeve(sleeve)
    result.classify(rules)
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    sleeve: str,
    candidates: List[str],
    dry_run: bool = False,
    apply: bool = False,
) -> List[SymbolResult]:
    """Train, backtest, classify all candidates for a sleeve.

    Args:
        sleeve: "swing-etf", "crypto", or "intraday"
        candidates: list of symbols to evaluate
        dry_run: if True, skip training/backtest, read existing summaries only
        apply: if True, write results to config/trading.json
    """
    results: List[SymbolResult] = []

    for i, symbol in enumerate(candidates, 1):
        log.info("=== [%d/%d] %s (%s) ===", i, len(candidates), symbol, sleeve)

        if not dry_run:
            # Train
            train_ok = train_symbol(symbol, sleeve)
            if not train_ok:
                results.append(SymbolResult(
                    symbol=symbol, sleeve=sleeve, sharpe=0.0, max_dd=0.0,
                    total_trades=0, win_rate=0.0, total_return=0.0,
                    profit_factor=None, bucket="disabled", notes="training failed",
                ))
                continue

            # Backtest
            bt_ok = backtest_symbol(symbol, sleeve)
            if not bt_ok:
                results.append(SymbolResult(
                    symbol=symbol, sleeve=sleeve, sharpe=0.0, max_dd=0.0,
                    total_trades=0, win_rate=0.0, total_return=0.0,
                    profit_factor=None, bucket="disabled", notes="backtest failed",
                ))
                continue

        # Read metrics
        summary = read_backtest_summary(symbol, sleeve)
        if summary is None:
            results.append(SymbolResult(
                symbol=symbol, sleeve=sleeve, sharpe=0.0, max_dd=0.0,
                total_trades=0, win_rate=0.0, total_return=0.0,
                profit_factor=None, bucket="disabled", notes="no summary found",
            ))
            continue

        result = evaluate_symbol(symbol, sleeve, summary)
        results.append(result)

    # Sort by Sharpe descending
    results.sort(key=lambda r: r.sharpe, reverse=True)

    # Print summary table
    print_summary_table(results, sleeve)

    # Generate config snippet
    config_snippet = generate_config_snippet(results, sleeve)
    print("\n--- Config Snippet (symbol_caps for trading.json) ---")
    print(json.dumps(config_snippet, indent=2))

    # Apply to trading.json
    if apply:
        apply_to_config(config_snippet, sleeve)
        log.info("Config updated: %s", CONFIG_PATH)

    return results


def print_summary_table(results: List[SymbolResult], sleeve: str) -> None:
    """Print a formatted table of results."""
    print(f"\n{'='*95}")
    print(f"  Symbol Selection Results — {sleeve.upper()}")
    print(f"{'='*95}")
    print(f"  {'Symbol':<12} {'Bucket':<12} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} "
          f"{'Trades':>7} {'WR':>6} {'MaxEq':>6}  {'Notes'}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6}  {'-'*20}")

    for r in results:
        bucket_display = r.bucket.upper()
        if r.bucket == "core":
            bucket_display = "** CORE **"
        elif r.bucket == "secondary":
            bucket_display = "  second  "

        print(f"  {r.symbol:<12} {bucket_display:<12} {r.sharpe:>+7.2f} {r.total_return:>+7.1f}% "
              f"{r.max_dd:>+7.1f}% {r.total_trades:>6d} {r.win_rate:>5.0%} {r.max_equity:>5.0%}  "
              f"{r.notes[:40]}")

    # Counts
    n_core = sum(1 for r in results if r.bucket == "core")
    n_secondary = sum(1 for r in results if r.bucket == "secondary")
    n_disabled = sum(1 for r in results if r.bucket == "disabled")
    print(f"\n  Summary: {n_core} core, {n_secondary} secondary, {n_disabled} disabled "
          f"(out of {len(results)} candidates)")


def generate_config_snippet(results: List[SymbolResult], sleeve: str) -> dict:
    """Generate a config snippet for trading.json symbol_caps."""
    # Map sleeve to config key
    config_key = {"swing-etf": "swing", "crypto": "crypto", "intraday": "intraday"}.get(sleeve, sleeve)

    caps = {}
    active_symbols = []
    for r in results:
        caps[r.symbol] = r.max_equity
        if r.bucket != "disabled":
            active_symbols.append(r.symbol)

    return {
        "sleeve": config_key,
        "symbol_caps": caps,
        "active_symbols": active_symbols,
        "promotion_summary": {
            r.symbol: {
                "bucket": r.bucket,
                "sharpe": round(r.sharpe, 3),
                "max_dd": round(r.max_dd, 2),
                "trades": r.total_trades,
                "win_rate": round(r.win_rate, 3),
                "max_equity": r.max_equity,
            }
            for r in results
        },
    }


def _backup_config() -> Optional[str]:
    """Create a timestamped backup of trading.json before writing.

    Keeps the last 4 weekly backups, deletes older ones.
    Returns the backup path (or None if backup failed).
    """
    import glob
    import shutil
    from datetime import datetime

    if not os.path.exists(CONFIG_PATH):
        return None

    config_dir = os.path.dirname(CONFIG_PATH)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_path = os.path.join(config_dir, f"trading_backup_{stamp}.json")

    try:
        shutil.copy2(CONFIG_PATH, backup_path)
        log.info("Config backup: %s", backup_path)

        # Prune old backups — keep last 4
        pattern = os.path.join(config_dir, "trading_backup_*.json")
        backups = sorted(glob.glob(pattern))
        for old in backups[:-4]:
            os.remove(old)
            log.info("Pruned old backup: %s", os.path.basename(old))

        return backup_path
    except Exception as exc:
        log.warning("Config backup failed: %s", exc)
        return None


def apply_to_config(snippet: dict, sleeve: str) -> None:
    """Merge the symbol_caps snippet into config/trading.json."""
    config_key = snippet["sleeve"]

    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log.error("Cannot read %s", CONFIG_PATH)
        return

    # Backup before writing
    _backup_config()

    # Update symbol_caps
    if "symbol_caps" not in config:
        config["symbol_caps"] = {}
    config["symbol_caps"][config_key] = snippet["symbol_caps"]

    # Update SYMBOL_OOS_SHARPE in the snippet for reference
    if "oos_sharpe_registry" not in config:
        config["oos_sharpe_registry"] = {}
    for sym, info in snippet["promotion_summary"].items():
        config["oos_sharpe_registry"][sym] = info["sharpe"]

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    log.info("Updated %s: %s sleeve, %d symbols (%d active)",
             CONFIG_PATH, config_key, len(snippet["symbol_caps"]),
             len(snippet["active_symbols"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbol Selection Pipeline — screen candidates, backtest, classify, update config.",
    )
    parser.add_argument("--sleeve", required=True,
                        choices=["swing-etf", "crypto", "intraday", "all"],
                        help="Which asset sleeve to evaluate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip training/backtest, read existing summary JSONs only")
    parser.add_argument("--apply", action="store_true",
                        help="Write results to config/trading.json (default: print only)")
    parser.add_argument("--symbols", default=None,
                        help="Override candidate list (comma-separated). If omitted, uses built-in list.")

    args = parser.parse_args()

    sleeves_to_run = []
    if args.sleeve == "all":
        sleeves_to_run = ["swing-etf", "crypto", "intraday"]
    else:
        sleeves_to_run = [args.sleeve]

    for sleeve in sleeves_to_run:
        # Pick candidate list
        if args.symbols:
            candidates = [s.strip().upper() for s in args.symbols.split(",")]
        else:
            if sleeve == "swing-etf":
                candidates = CANDIDATE_SWING_ETFS
            elif sleeve == "crypto":
                candidates = CANDIDATE_CRYPTOS
            elif sleeve == "intraday":
                candidates = CANDIDATE_INTRADAY_ETFS
            else:
                candidates = []

        if not candidates:
            log.warning("No candidates for sleeve %s", sleeve)
            continue

        run_pipeline(
            sleeve=sleeve,
            candidates=candidates,
            dry_run=args.dry_run,
            apply=args.apply,
        )


if __name__ == "__main__":
    main()
