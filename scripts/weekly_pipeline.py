#!/usr/bin/env python3
"""
Weekly Pipeline — Automated retraining + OOS validation for ALL trading groups.
================================================================================

Orchestrates the full retrain → backtest → registry update cycle for all 4 groups:
  - ETF Swing (XGBoost + TFT)
  - ETF Intraday (LightGBM + GRU)
  - Crypto Swing (XGBoost + TFT)
  - Crypto Intraday (LightGBM + GRU)

Pipeline steps:
    1. screen-etf-universe       — Discover tradeable ETFs (Alpaca + yfinance)
    2. screen-crypto-universe    — Discover tradeable coins (CMC + Kraken)
    3. train-etf-swing           — Retrain swing models for all ETFs (up to cutoff)
    4. train-etf-intraday        — Retrain intraday models for all ETFs (up to cutoff)
    5. train-crypto-swing        — Retrain swing models for all crypto (up to cutoff)
    6. train-crypto-intraday     — Retrain intraday models for all crypto (up to cutoff)
    7. backtest-all              — OOS backtest ALL models (cutoff → today)
    8. update-registries         — Populate OOS Sharpe registries from backtest results
    9. train-selectors           — Retrain LambdaRank selectors (swing, intraday, crypto)
   10. model-health              — Check model health, alert on degradation
   11. slack-summary             — Send pipeline summary to Slack

OOS Cutoff:
    --train-end YYYY-MM-DD sets the boundary. All models train on data BEFORE this date.
    OOS backtests run on data AFTER this date. Default: today - 60 days.

Usage:
    python scripts/weekly_pipeline.py
    python scripts/weekly_pipeline.py --train-end 2026-01-25
    python scripts/weekly_pipeline.py --skip train-etf-swing backtest-all
    python scripts/weekly_pipeline.py --only train-crypto-swing backtest-all update-registries
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging

log = logging.getLogger("weekly_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Log rotation for weekly pipeline (10 MB, 3 backups) + cleanup old timestamped logs
_log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(_log_dir, exist_ok=True)
from logging.handlers import RotatingFileHandler as _RFH
_rfh = _RFH(os.path.join(_log_dir, "weekly_pipeline.log"),
            maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
_rfh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(_rfh)


def _cleanup_old_logs(log_dir: str, prefix: str, keep: int = 10) -> None:
    """Remove old timestamped log files, keeping only the most recent `keep`."""
    import glob
    pattern = os.path.join(log_dir, f"{prefix}*.log")
    # Exclude rotated logs (e.g., weekly_pipeline.log.1)
    files = [f for f in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
             if not any(f.endswith(f".log.{i}") for i in range(1, 10))]
    for old in files[keep:]:
        try:
            os.remove(old)
            log.info("Cleaned up old log: %s", os.path.basename(old))
        except OSError:
            pass
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for _p in (SRC_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
MAIN_PY = os.path.join(PROJECT_ROOT, "main.py")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ENV_FILE = os.path.join(PROJECT_ROOT, "secrets", "alpaca.env")

# Model directories
SWING_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "swing")
INTRADAY_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "intraday")
CRYPTO_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "crypto")
CRYPTO_INTRADAY_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "crypto_intraday")

# Backtest output directory
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "outputs", "backtests")


def _load_env_file() -> None:
    """Load key=value pairs from secrets/alpaca.env into os.environ."""
    if not os.path.exists(ENV_FILE):
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and val:
                os.environ.setdefault(key, val)


@dataclass
class StepResult:
    name: str
    success: bool
    elapsed_seconds: float
    error: str = ""
    details: str = ""


# Global: set by --train-end or default (today - 60 days)
TRAIN_END: str = ""


def _run(args: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output."""
    return subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)


def _count_models(directory: str, pattern: str) -> list[str]:
    """List symbols with trained models matching pattern in directory."""
    files = glob.glob(os.path.join(directory, pattern))
    return sorted(set(
        os.path.basename(f).replace(pattern.replace("*", ""), "").rstrip(".")
        for f in files
    ))


def _get_swing_etf_symbols() -> list[str]:
    """Get all ETF symbols with swing models."""
    return _count_models(SWING_MODEL_DIR, "*_xgb_swing.joblib")


def _get_intraday_etf_symbols() -> list[str]:
    """Get all ETF symbols with intraday models."""
    return _count_models(INTRADAY_MODEL_DIR, "*_lgb_intraday_etf.joblib")


def _get_crypto_swing_symbols() -> list[str]:
    """Get all crypto symbols with swing models."""
    return _count_models(CRYPTO_MODEL_DIR, "*_xgb_swing.joblib")


def _get_crypto_intraday_symbols() -> list[str]:
    """Get all crypto symbols with intraday models."""
    return _count_models(CRYPTO_INTRADAY_MODEL_DIR, "*_lgb_intraday_crypto.joblib")


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def step_screen_etf_universe() -> StepResult:
    """Screen ETF universe via Alpaca API + yfinance validation."""
    t0 = time.time()
    try:
        result = _run([PYTHON, MAIN_PY, "screen-etf-universe"])
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("screen-etf-universe", False, elapsed,
                              error=(result.stderr or "")[-500:])
        details = ""
        for line in (result.stdout or "").splitlines():
            if "pass" in line.lower() or "etf" in line.lower():
                details = line.strip()
        return StepResult("screen-etf-universe", True, elapsed, details=details)
    except Exception as exc:
        return StepResult("screen-etf-universe", False, time.time() - t0, error=str(exc))


def step_screen_crypto_universe() -> StepResult:
    """Screen crypto universe via CMC + Kraken."""
    t0 = time.time()
    try:
        result = _run([PYTHON, MAIN_PY, "screen-universe", "--top-n", "250"])
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("screen-crypto-universe", False, elapsed,
                              error=(result.stderr or "")[-500:])
        details = ""
        for line in (result.stdout or "").splitlines():
            if "coins" in line.lower() or "universe" in line.lower():
                details = line.strip()
                break
        return StepResult("screen-crypto-universe", True, elapsed, details=details)
    except Exception as exc:
        return StepResult("screen-crypto-universe", False, time.time() - t0, error=str(exc))


def step_train_etf_swing() -> StepResult:
    """Train XGBoost+TFT swing models for all ETFs."""
    t0 = time.time()
    try:
        symbols = _get_swing_etf_symbols()
        if not symbols:
            return StepResult("train-etf-swing", False, 0, error="no swing ETF models on disk")
        cmd = [PYTHON, MAIN_PY, "train-swing",
               "--symbols", ",".join(symbols),
               "--provider", "yahoo",
               "--train-end", TRAIN_END]
        result = _run(cmd, timeout=7200)
        elapsed = time.time() - t0
        lines = (result.stdout or "").splitlines() + (result.stderr or "").splitlines()
        ok = sum(1 for l in lines if "Saved swing XGBoost" in l)
        fail = sum(1 for l in lines if "training failed" in l or "skipping" in l.lower())
        if ok == 0 and result.stderr:
            # Log stderr for debugging when no models saved
            err_lines = result.stderr.strip().splitlines()[-10:]
            log.error("train-etf-swing stderr tail:\n%s", "\n".join(err_lines))
        details = f"{len(symbols)} symbols, {ok} saved, {fail} failed, cutoff={TRAIN_END}"
        return StepResult("train-etf-swing", ok > 0, elapsed, details=details)
    except Exception as exc:
        return StepResult("train-etf-swing", False, time.time() - t0, error=str(exc))


def step_train_etf_intraday() -> StepResult:
    """Train LightGBM+GRU intraday models for all ETFs."""
    t0 = time.time()
    try:
        symbols = _get_intraday_etf_symbols()
        if not symbols:
            return StepResult("train-etf-intraday", False, 0, error="no intraday ETF models on disk")
        # Call etf_intraday_model.py directly to ensure --train-end passes through
        etf_intraday_script = os.path.join(SRC_DIR, "etf_intraday_model.py")
        cmd = [PYTHON, etf_intraday_script,
               "--symbols", ",".join(symbols),
               "--provider", "yahoo",
               "--walk-forward",
               "--train-end", TRAIN_END]
        result = _run(cmd, timeout=7200)
        elapsed = time.time() - t0
        lines = (result.stdout or "").splitlines() + (result.stderr or "").splitlines()
        ok = sum(1 for l in lines if "ENSEMBLE:" in l or "LGB-only:" in l)
        details = f"{len(symbols)} symbols, {ok} models, cutoff={TRAIN_END}"
        return StepResult("train-etf-intraday", result.returncode == 0, elapsed, details=details)
    except Exception as exc:
        return StepResult("train-etf-intraday", False, time.time() - t0, error=str(exc))


def step_train_crypto_swing() -> StepResult:
    """Train XGBoost+TFT swing models for all crypto."""
    t0 = time.time()
    try:
        # Load universe from disk
        from universe_screener import load_universe
        universe = load_universe(CRYPTO_MODEL_DIR)
        if not universe:
            universe = _get_crypto_swing_symbols()
        if not universe:
            return StepResult("train-crypto-swing", False, 0, error="no crypto symbols found")
        sym_str = ",".join(s if "-" in s else f"{s}-USD" for s in universe)
        cmd = [PYTHON, MAIN_PY, "train-swing",
               "--symbols", sym_str,
               "--provider", "yahoo",
               "--save-dir", CRYPTO_MODEL_DIR,
               "--train-end", TRAIN_END]
        result = _run(cmd, timeout=10800)
        elapsed = time.time() - t0
        lines = (result.stdout or "").splitlines() + (result.stderr or "").splitlines()
        ok = sum(1 for l in lines if "Saved swing XGBoost" in l)
        details = f"{len(universe)} symbols, {ok} saved, cutoff={TRAIN_END}"
        return StepResult("train-crypto-swing", result.returncode == 0, elapsed, details=details)
    except Exception as exc:
        return StepResult("train-crypto-swing", False, time.time() - t0, error=str(exc))


def step_train_crypto_intraday() -> StepResult:
    """Train LightGBM+GRU intraday models for all crypto."""
    t0 = time.time()
    try:
        from universe_screener import load_universe
        universe = load_universe(CRYPTO_MODEL_DIR)
        if not universe:
            universe = _get_crypto_intraday_symbols()
        if not universe:
            return StepResult("train-crypto-intraday", False, 0, error="no crypto symbols found")
        sym_str = ",".join(s if "-" in s else f"{s}-USD" for s in universe)
        # Call crypto_intraday_model.py directly (main.py rebuilds sys.argv, breaking --train-end)
        crypto_intraday_script = os.path.join(SRC_DIR, "crypto_intraday_model.py")
        cmd = [PYTHON, crypto_intraday_script,
               "--symbols", sym_str,
               "--save-dir", CRYPTO_INTRADAY_MODEL_DIR,
               "--walk-forward",
               "--train-end", TRAIN_END]
        result = _run(cmd, timeout=10800)
        elapsed = time.time() - t0
        lines = (result.stdout or "").splitlines() + (result.stderr or "").splitlines()
        ok = sum(1 for l in lines if "ENSEMBLE:" in l or "LGB-only:" in l)
        details = f"{len(universe)} symbols, {ok} models, cutoff={TRAIN_END}"
        return StepResult("train-crypto-intraday", result.returncode == 0, elapsed, details=details)
    except Exception as exc:
        return StepResult("train-crypto-intraday", False, time.time() - t0, error=str(exc))


def _backtest_group(group: str, symbols: list[str], model_flag: str,
                    model_dir: Optional[str] = None, mode: str = "daily",
                    interval: str = "1d") -> Dict[str, dict]:
    """Run OOS backtests for a list of symbols. Returns {symbol: {sharpe, pf, wr, trades}}."""
    results = {}
    for sym in symbols:
        try:
            cmd = [PYTHON, MAIN_PY, "backtest",
                   "--symbol", sym,
                   "--start", TRAIN_END,
                   "--model", model_flag]
            if mode == "intraday":
                cmd.extend(["--mode", "intraday", "--interval", interval])
            if model_dir:
                cmd.extend(["--model-dir", model_dir])
            r = _run(cmd, timeout=300)
            output = (r.stdout or "") + (r.stderr or "")
            # Parse Sharpe from output
            sharpe = 0.0
            pf = 0.0
            wr = 0.0
            trades = 0
            for line in output.splitlines():
                if "Sharpe Ratio" in line:
                    m = re.search(r"[\-+]?\d+\.?\d*", line.split(":")[-1])
                    if m:
                        sharpe = float(m.group())
                elif "Profit Factor" in line:
                    m = re.search(r"[\-+]?\d+\.?\d*", line.split(":")[-1])
                    if m:
                        pf = float(m.group())
                elif "Win Rate" in line:
                    m = re.search(r"[\-+]?\d+\.?\d*", line.split(":")[-1])
                    if m:
                        wr = float(m.group())
                elif "Total Trades" in line or "trades" in line.lower():
                    m = re.search(r"\d+", line.split(":")[-1])
                    if m:
                        trades = int(m.group())
            results[sym] = {"sharpe": sharpe, "pf": pf, "wr": wr, "trades": trades}
        except Exception as exc:
            results[sym] = {"sharpe": 0.0, "pf": 0.0, "wr": 0.0, "trades": 0, "error": str(exc)}
    return results


def step_backtest_all() -> StepResult:
    """Run OOS backtests for ALL models across all 4 groups."""
    t0 = time.time()
    all_results: Dict[str, Dict[str, dict]] = {}
    try:
        os.makedirs(BACKTEST_DIR, exist_ok=True)

        # 1. Swing ETF
        swing_syms = _get_swing_etf_symbols()
        print(f"    Backtesting {len(swing_syms)} swing ETFs...")
        all_results["swing"] = _backtest_group("swing", swing_syms, "swing")

        # 2. Intraday ETF
        intraday_syms = _get_intraday_etf_symbols()
        print(f"    Backtesting {len(intraday_syms)} intraday ETFs...")
        all_results["intraday"] = _backtest_group(
            "intraday", intraday_syms, "etf_intraday",
            mode="intraday", interval="5min")

        # 3. Crypto swing
        crypto_syms = _get_crypto_swing_symbols()
        print(f"    Backtesting {len(crypto_syms)} crypto swing...")
        all_results["crypto"] = _backtest_group(
            "crypto", crypto_syms, "swing", model_dir=CRYPTO_MODEL_DIR)

        # 4. Crypto intraday
        crypto_id_syms = _get_crypto_intraday_symbols()
        print(f"    Backtesting {len(crypto_id_syms)} crypto intraday...")
        all_results["crypto_intraday"] = _backtest_group(
            "crypto_intraday", crypto_id_syms, "crypto_intraday",
            model_dir=CRYPTO_INTRADAY_MODEL_DIR, mode="intraday", interval="5min")

        # Save results to disk for registry update
        results_path = os.path.join(BACKTEST_DIR, f"oos_results_{TRAIN_END}.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        elapsed = time.time() - t0
        total = sum(len(v) for v in all_results.values())
        positive = sum(
            1 for group in all_results.values()
            for sym_data in group.values()
            if sym_data.get("sharpe", 0) > 0
        )
        details = f"{total} backtests, {positive} positive Sharpe, saved to {results_path}"
        return StepResult("backtest-all", True, elapsed, details=details)
    except Exception as exc:
        return StepResult("backtest-all", False, time.time() - t0, error=str(exc))


def step_update_registries() -> StepResult:
    """Update OOS Sharpe registries from backtest results."""
    t0 = time.time()
    try:
        results_path = os.path.join(BACKTEST_DIR, f"oos_results_{TRAIN_END}.json")
        if not os.path.exists(results_path):
            return StepResult("update-registries", False, 0, error="no backtest results file")

        with open(results_path) as f:
            all_results = json.load(f)

        today = datetime.now().strftime("%Y-%m-%d")

        # --- ETF registries: promoted_symbols.json ---
        for group_key, model_dir in [("swing", SWING_MODEL_DIR), ("intraday", INTRADAY_MODEL_DIR)]:
            registry_path = os.path.join(model_dir, "promoted_symbols.json")
            registry = {}
            if os.path.exists(registry_path):
                with open(registry_path) as f:
                    registry = json.load(f)
            group_results = all_results.get(group_key, {})
            for sym, data in group_results.items():
                registry[sym] = {
                    "oos_sharpe": data.get("sharpe", 0.0),
                    "oos_pf": data.get("pf", 0.0),
                    "oos_wr": data.get("wr", 0.0),
                    "oos_trades": data.get("trades", 0),
                    "updated": today,
                    "oos_start": TRAIN_END,
                }
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            print(f"    Updated {registry_path}: {len(group_results)} symbols")

        # --- Crypto registries: config/trading.json ---
        config_path = os.path.join(PROJECT_ROOT, "config", "trading.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

            for group_key, registry_key in [
                ("crypto", "oos_sharpe_registry"),
                ("crypto_intraday", "oos_sharpe_registry_intraday"),
            ]:
                registry = config.get(registry_key, {})
                group_results = all_results.get(group_key, {})
                for sym, data in group_results.items():
                    registry[sym] = {
                        "oos_sharpe": data.get("sharpe", 0.0),
                        "oos_pf": data.get("pf", 0.0),
                        "oos_wr": data.get("wr", 0.0),
                        "oos_trades": data.get("trades", 0),
                        "updated": today,
                        "oos_start": TRAIN_END,
                    }
                config[registry_key] = registry
                print(f"    Updated trading.json[{registry_key}]: {len(group_results)} symbols")

            # Update retrain metadata
            config["_retrain_cadence"]["_last_retrain"] = today
            config["_retrain_cadence"]["_oos_cutoff"] = TRAIN_END

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        elapsed = time.time() - t0
        total = sum(len(v) for v in all_results.values())
        return StepResult("update-registries", True, elapsed,
                          details=f"{total} symbols updated across 4 registries")
    except Exception as exc:
        return StepResult("update-registries", False, time.time() - t0, error=str(exc))


def step_train_selectors() -> StepResult:
    """Retrain LambdaRank selectors (crypto + ETF intraday)."""
    t0 = time.time()
    trained = []
    try:
        # Crypto selector
        r = _run([PYTHON, MAIN_PY, "train-selector"])
        if r.returncode == 0:
            trained.append("crypto")

        # Crypto intraday selector
        r = _run([PYTHON, MAIN_PY, "train-selector-intraday"])
        if r.returncode == 0:
            trained.append("crypto-intraday")

        # ETF intraday selector
        r = _run([PYTHON, MAIN_PY, "train-intraday-etf-selector"])
        if r.returncode == 0:
            trained.append("etf-intraday")

        elapsed = time.time() - t0
        return StepResult("train-selectors", len(trained) > 0, elapsed,
                          details=f"Trained: {', '.join(trained)}")
    except Exception as exc:
        return StepResult("train-selectors", False, time.time() - t0, error=str(exc))


def step_model_health() -> StepResult:
    """Check model health, report degraded models."""
    t0 = time.time()
    try:
        from model_monitor import ModelMonitor
        monitor = ModelMonitor()
        monitor.generate_report()
        elapsed = time.time() - t0
        health = monitor.get_all_health()
        paused = [s for s, h in health.items() if getattr(h, "status", "ok") == "paused"]
        details = f"PAUSED: {', '.join(paused)}" if paused else "All models healthy"
        return StepResult("model-health", True, elapsed, details=details)
    except Exception as exc:
        return StepResult("model-health", False, time.time() - t0, error=str(exc))


def send_slack_summary(results: list[StepResult]) -> None:
    """Send pipeline summary to Slack."""
    try:
        from alerts import AlertEngine
        engine = AlertEngine()
        lines = [f"Weekly Pipeline Summary (cutoff={TRAIN_END})", ""]
        total_time = sum(r.elapsed_seconds for r in results)
        passed = sum(1 for r in results if r.success)
        for r in results:
            status = "PASS" if r.success else "FAIL"
            line = f"  {status}  {r.name} ({r.elapsed_seconds:.0f}s)"
            if r.details:
                line += f" -- {r.details}"
            if r.error:
                line += f" -- ERROR: {r.error[:200]}"
            lines.append(line)
        lines.append(f"\n{passed}/{len(results)} passed, {total_time/60:.1f} min total")
        engine.notify_pipeline_summary("\n".join(lines))
    except Exception as exc:
        print(f"  [!] Slack summary failed: {exc}")


def print_summary(results: list[StepResult]) -> None:
    """Print pipeline summary to console."""
    print(f"\n{'='*70}")
    print(f"  WEEKLY PIPELINE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  OOS Cutoff: {TRAIN_END}")
    print(f"{'='*70}\n")
    for r in results:
        status = "OK  " if r.success else "FAIL"
        print(f"  [{status}] {r.name:<25s} {r.elapsed_seconds:>7.0f}s", end="")
        if r.details:
            print(f"  {r.details}", end="")
        if r.error:
            print(f"  ERROR: {r.error[:100]}", end="")
        print()
    passed = sum(1 for r in results if r.success)
    total = sum(r.elapsed_seconds for r in results)
    print(f"\n  {passed}/{len(results)} steps passed — {total/60:.1f} min total")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STEPS = [
    ("screen-etf-universe",    step_screen_etf_universe),
    ("screen-crypto-universe", step_screen_crypto_universe),
    ("train-etf-swing",        step_train_etf_swing),
    ("train-etf-intraday",     step_train_etf_intraday),
    ("train-crypto-swing",     step_train_crypto_swing),
    ("train-crypto-intraday",  step_train_crypto_intraday),
    ("backtest-all",           step_backtest_all),
    ("update-registries",      step_update_registries),
    ("train-selectors",        step_train_selectors),
    ("model-health",           step_model_health),
]

# Hard dependencies: if upstream fails, downstream is skipped
HARD_DEPS = {
    "backtest-all": ["train-etf-swing", "train-crypto-swing"],
    "update-registries": ["backtest-all"],
}


def main() -> None:
    global TRAIN_END

    parser = argparse.ArgumentParser(description="Weekly pipeline: retrain + backtest + registry update")
    parser.add_argument("--train-end", default=None,
                        help="OOS cutoff date (YYYY-MM-DD). Default: today - 60 days.")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Steps to skip")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run ONLY these steps (overrides --skip)")
    args = parser.parse_args()

    # Set OOS cutoff
    if args.train_end:
        TRAIN_END = args.train_end
    else:
        TRAIN_END = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    _load_env_file()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BACKTEST_DIR, exist_ok=True)

    # Determine which steps to run
    if args.only:
        active_steps = [(n, fn) for n, fn in ALL_STEPS if n in args.only]
    else:
        active_steps = [(n, fn) for n, fn in ALL_STEPS if n not in (args.skip or [])]

    step_names = [n for n, _ in active_steps]
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Starting weekly pipeline...")
    print(f"  OOS cutoff: {TRAIN_END}")
    print(f"  Steps: {', '.join(step_names)}\n")

    results: list[StepResult] = []
    for name, fn in active_steps:
        # Check hard dependencies
        deps = HARD_DEPS.get(name, [])
        blocked = False
        for dep in deps:
            dep_result = next((r for r in results if r.name == dep), None)
            if dep_result and not dep_result.success:
                msg = f"SKIPPED (upstream {dep} failed)"
                results.append(StepResult(name, False, 0, details=msg))
                print(f"  [GATE] {name} -- {msg}")
                blocked = True
                break
        if blocked:
            continue

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Running {name}...")
        result = fn()
        results.append(result)
        status = "OK" if result.success else "FAIL"
        print(f"  [{status}] {name} ({result.elapsed_seconds:.0f}s)")
        if result.details:
            print(f"         {result.details}")
        if result.error:
            print(f"         Error: {result.error[:200]}")

    print_summary(results)
    send_slack_summary(results)

    # Cleanup old timestamped logs (keep 10 most recent per prefix)
    for prefix in ("weekly_pipeline_", "universe_screen_",
                   "paper_trader_gold_scalper_2", "paper_trader_intraday_2",
                   "paper_trader_swing_2", "paper_trader_crypto_2",
                   "paper_trader_crypto_intraday_2"):
        _cleanup_old_logs(LOG_DIR, prefix, keep=10)


if __name__ == "__main__":
    main()
