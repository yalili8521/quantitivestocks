#!/usr/bin/env python3
"""
Weekly Crypto Pipeline — Automated maintenance for the crypto trading system.
=============================================================================

Chains six steps in sequence:
    1. screen-universe         — Layer 0: discover tradeable coins (CoinGecko x Kraken)
    2. train-selector          — Layer 1: retrain cross-sectional coin selector
    3. train-crypto            — Layer 2a: retrain TFT+XGBoost swing models for all coins
    4. train-crypto-intraday   — Layer 2b: retrain LGB+GRU intraday models for all coins
    5. select-symbols          — Layer 3: OOS backtest + auto-update symbol_caps
    6. model-health            — Layer 4: check model health, alert on degradation

Usage:
    python scripts/weekly_pipeline.py
    python scripts/weekly_pipeline.py --skip train-crypto     # skip slow training step
    python scripts/weekly_pipeline.py --skip screen-universe train-selector

Schedule (PowerShell, admin):
    $action = New-ScheduledTaskAction `
        -Execute "C:\\Users\\yalil\\OneDrive\\Desktop\\AI-projects\\quantitivestocks\\.venv\\Scripts\\python.exe" `
        -Argument "-u scripts/weekly_pipeline.py" `
        -WorkingDirectory "C:\\Users\\yalil\\OneDrive\\Desktop\\AI-projects\\quantitivestocks"
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "10:00PM"
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd
    Register-ScheduledTask -TaskName "QuantStocks-WeeklyPipeline" `
        -Action $action -Trigger $trigger -Settings $settings `
        -Description "Weekly crypto pipeline: screen, train, select, health check"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for _p in (SRC_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
MAIN_PY = os.path.join(PROJECT_ROOT, "main.py")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ENV_FILE = os.path.join(PROJECT_ROOT, "secrets", "alpaca.env")


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


def _run_command(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a subprocess command, capturing output."""
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_screen_universe() -> StepResult:
    """Step 1: Layer 0 — discover tradeable coins from CoinGecko x Kraken."""
    t0 = time.time()
    try:
        result = _run_command([PYTHON, MAIN_PY, "screen-universe", "--top-n", "250"])
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("screen-universe", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")
        # Extract coin count from output
        details = ""
        for line in result.stdout.splitlines():
            if "coins" in line.lower() or "universe" in line.lower():
                details = line.strip()
                break
        return StepResult("screen-universe", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("screen-universe", False, time.time() - t0, error="timeout (1h)")
    except Exception as exc:
        return StepResult("screen-universe", False, time.time() - t0, error=str(exc))


def run_train_selector() -> StepResult:
    """Step 2: Layer 1 — retrain the LambdaRank coin selector."""
    t0 = time.time()
    try:
        result = _run_command([PYTHON, MAIN_PY, "train-selector"])
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("train-selector", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")
        details = ""
        for line in result.stdout.splitlines():
            if "ndcg" in line.lower() or "hit_rate" in line.lower() or "saved" in line.lower():
                details = line.strip()
        return StepResult("train-selector", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("train-selector", False, time.time() - t0, error="timeout (1h)")
    except Exception as exc:
        return StepResult("train-selector", False, time.time() - t0, error=str(exc))


def run_train_crypto() -> StepResult:
    """Step 3: Layer 2 — retrain TFT+XGBoost swing models for all crypto coins."""
    t0 = time.time()
    try:
        result = _run_command([PYTHON, MAIN_PY, "train-crypto"], timeout=7200)
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("train-crypto", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")
        # Count from log output: "Saved swing XGBoost" = success, "training failed" = failure
        lines = result.stdout.splitlines() + result.stderr.splitlines()
        ok_count = sum(1 for l in lines if "Saved swing XGBoost" in l or "[TFT]" in l and "saved" in l)
        fail_count = sum(1 for l in lines if "training failed" in l)
        # Count unique symbols trained (each has "=== Training swing XGBoost for XXX ===")
        trained_syms = sum(1 for l in lines if "Training swing XGBoost for" in l)
        details = f"{trained_syms} symbols, {ok_count} models saved, {fail_count} failed"
        return StepResult("train-crypto", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("train-crypto", False, time.time() - t0, error="timeout (2h)")
    except Exception as exc:
        return StepResult("train-crypto", False, time.time() - t0, error=str(exc))


def run_train_crypto_intraday() -> StepResult:
    """Step 3b: Layer 2b — retrain LGB+GRU intraday models for crypto coins."""
    t0 = time.time()
    try:
        result = _run_command([PYTHON, MAIN_PY, "train-crypto-intraday"], timeout=7200)
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("train-crypto-intraday", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")
        lines = result.stdout.splitlines() + result.stderr.splitlines()
        ok_count = sum(1 for l in lines if "ENSEMBLE:" in l or "LGB-only:" in l)
        fail_count = sum(1 for l in lines if "training failed" in l)
        trained_syms = sum(1 for l in lines if "Training crypto intraday for" in l)
        details = f"{trained_syms} symbols, {ok_count} models saved, {fail_count} failed"
        return StepResult("train-crypto-intraday", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("train-crypto-intraday", False, time.time() - t0, error="timeout (2h)")
    except Exception as exc:
        return StepResult("train-crypto-intraday", False, time.time() - t0, error=str(exc))


def run_select_symbols() -> StepResult:
    """Step 4: Layer 3 — OOS backtest + classify + auto-update config/trading.json."""
    t0 = time.time()
    try:
        result = _run_command(
            [PYTHON, MAIN_PY, "select-symbols", "--sleeve", "crypto", "--apply"],
            timeout=7200,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("select-symbols", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")
        # Extract classification summary (e.g. "Summary: 1 core, 3 secondary, 10 disabled")
        details = ""
        for line in result.stdout.splitlines():
            if line.strip().startswith("Summary:"):
                details = line.strip()
                break
        if not details:
            for line in result.stdout.splitlines():
                if "core" in line.lower() or "applied" in line.lower():
                    details = line.strip()
        return StepResult("select-symbols", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("select-symbols", False, time.time() - t0, error="timeout (2h)")
    except Exception as exc:
        return StepResult("select-symbols", False, time.time() - t0, error=str(exc))


def run_train_new_etfs() -> StepResult:
    """Step 5b: Train swing models for any newly promoted ETFs without models.

    Checks promoted_symbols.json against existing model files. Only trains
    symbols that were promoted but don't have a trained XGBoost model yet.
    Skips entirely if there are no new symbols to train.
    """
    t0 = time.time()
    try:
        from utils import SWING_MODEL_DIR
        from etf_screener import load_promoted_symbols

        promoted = load_promoted_symbols(SWING_MODEL_DIR)
        if not promoted:
            return StepResult("train-new-etfs", True, time.time() - t0,
                              details="No promoted_symbols.json — skipped")

        # Check which promoted symbols lack a model
        import re
        _SYM_RE = re.compile(r"^[A-Z0-9\-/]{1,10}$")
        untrained = []
        for sym in promoted:
            if not _SYM_RE.match(sym):
                print(f"  [WARN] Invalid symbol skipped: {sym!r}")
                continue
            model_file = os.path.join(SWING_MODEL_DIR, f"{sym}_xgb_swing.joblib")
            if not os.path.exists(model_file):
                untrained.append(sym)

        if not untrained:
            return StepResult("train-new-etfs", True, time.time() - t0,
                              details="All promoted ETFs have models — nothing to train")

        # Train only the new ones
        sym_list = ",".join(untrained)
        result = _run_command(
            [PYTHON, MAIN_PY, "train-swing",
             "--symbols", sym_list,
             "--provider", "yahoo"],
            timeout=3600,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("train-new-etfs", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")

        lines = result.stdout.splitlines() + result.stderr.splitlines()
        ok_count = sum(1 for l in lines if "Saved swing XGBoost" in l)
        details = f"{len(untrained)} new symbols, {ok_count} models saved"
        return StepResult("train-new-etfs", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("train-new-etfs", False, time.time() - t0, error="timeout (1h)")
    except Exception as exc:
        return StepResult("train-new-etfs", False, time.time() - t0, error=str(exc))


def run_model_health() -> StepResult:
    """Step 5: Layer 4 — check model health, report degraded models."""
    t0 = time.time()
    try:
        from model_monitor import ModelMonitor
        monitor = ModelMonitor()
        report = monitor.generate_report()
        elapsed = time.time() - t0

        # Find paused/warning models
        paused = []
        warnings = []
        health = monitor.get_all_health()
        for sym, info in health.items():
            status = getattr(info, "status", "ok")
            if status == "paused":
                paused.append(sym)
            elif status == "warning":
                warnings.append(sym)

        details_parts = []
        if paused:
            details_parts.append(f"PAUSED: {', '.join(paused)}")
        if warnings:
            details_parts.append(f"WARNING: {', '.join(warnings)}")
        if not details_parts:
            details_parts.append("All models healthy")
        details = " | ".join(details_parts)

        return StepResult("model-health", True, elapsed, details=details)
    except Exception as exc:
        return StepResult("model-health", False, time.time() - t0, error=str(exc))


def send_slack_summary(results: list[StepResult]) -> None:
    """Send pipeline summary to Slack via AlertEngine."""
    try:
        from alerts import AlertEngine
        engine = AlertEngine()

        lines = ["Weekly Crypto Pipeline Summary", ""]
        total_time = sum(r.elapsed_seconds for r in results)
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        for r in results:
            status = "PASS" if r.success else "FAIL"
            line = f"  {status}  {r.name} ({r.elapsed_seconds:.0f}s)"
            if r.details:
                line += f" -- {r.details}"
            if r.error:
                line += f" -- ERROR: {r.error[:200]}"
            lines.append(line)

        lines.append("")
        lines.append(f"Total: {passed}/{len(results)} passed, {total_time/60:.1f} min")
        if failed:
            lines.append(f"{failed} step(s) FAILED -- check logs")

        engine.notify_pipeline_summary("\n".join(lines))
    except Exception as exc:
        print(f"  [!] Slack summary failed: {exc}")


def print_summary(results: list[StepResult]) -> None:
    """Print pipeline summary to console."""
    print(f"\n{'='*65}")
    print(f"  WEEKLY CRYPTO PIPELINE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}\n")

    total_time = 0
    for r in results:
        total_time += r.elapsed_seconds
        status = "OK  " if r.success else "FAIL"
        print(f"  [{status}] {r.name:<20s} {r.elapsed_seconds:>7.0f}s", end="")
        if r.details:
            print(f"  {r.details}", end="")
        if r.error:
            print(f"  ERROR: {r.error[:100]}", end="")
        print()

    passed = sum(1 for r in results if r.success)
    print(f"\n  {passed}/{len(results)} steps passed — {total_time/60:.1f} min total")
    print(f"{'='*65}\n")


def _step_succeeded(results: list[StepResult], name: str) -> bool:
    """Check if a step succeeded (or was skipped)."""
    for r in results:
        if r.name == name:
            return r.success
    return False


# Dependency gates: which upstream failures should skip downstream steps.
# Key = downstream step, value = (upstream step, reason).
# "soft" = upstream fail logs a warning but downstream still runs (uses cached data)
# "hard" = upstream fail causes downstream to be skipped entirely
DEPENDENCY_GATES: dict[str, list[tuple[str, str, str]]] = {
    # train-selector needs fresh universe, but can use cached → soft
    "train-selector": [
        ("screen-universe", "soft", "universe stale, using cached"),
    ],
    # train-crypto is independent of selector, so no hard deps
    "train-crypto": [],
    # train-crypto-intraday is independent (parallel to swing training)
    "train-crypto-intraday": [],
    # select-symbols MUST have fresh models → hard dep on train-crypto
    "select-symbols": [
        ("train-crypto", "hard", "cannot backtest stale models"),
    ],
    # train-new-etfs is independent — uses promoted_symbols.json (already exists)
    "train-new-etfs": [],
    # model-health is independent — always runs
    "model-health": [],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly crypto maintenance pipeline")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Steps to skip: screen-universe, train-selector, "
                             "train-crypto, train-crypto-intraday, "
                             "select-symbols, train-new-etfs, model-health")
    args = parser.parse_args()

    # Load env vars (CMC_API_KEY, ALERT_WEBHOOK_URL, etc.)
    _load_env_file()

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    steps = [
        ("screen-universe", run_screen_universe),
        ("train-selector", run_train_selector),
        ("train-crypto", run_train_crypto),
        ("train-crypto-intraday", run_train_crypto_intraday),
        ("select-symbols", run_select_symbols),
        ("train-new-etfs", run_train_new_etfs),
        ("model-health", run_model_health),
    ]

    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Starting weekly crypto pipeline...", flush=True)
    print(f"  Steps: {', '.join(name for name, _ in steps)}")
    print(f"  Skipping: {args.skip or 'none'}\n")

    results: list[StepResult] = []
    for name, fn in steps:
        if name in args.skip:
            results.append(StepResult(name=name, success=True, elapsed_seconds=0, details="SKIPPED"))
            print(f"  [SKIP] {name}")
            continue

        # Check dependency gates
        gates = DEPENDENCY_GATES.get(name, [])
        gate_blocked = False
        for upstream, severity, reason in gates:
            if not _step_succeeded(results, upstream):
                if severity == "hard":
                    msg = f"SKIPPED (upstream {upstream} failed: {reason})"
                    results.append(StepResult(name=name, success=False, elapsed_seconds=0,
                                              details=msg))
                    print(f"  [GATE] {name} -- {msg}")
                    gate_blocked = True
                    break
                else:  # soft
                    print(f"  [WARN] {name}: upstream {upstream} failed -- {reason}")

        if gate_blocked:
            continue

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Running {name}...")
        result = fn()
        results.append(result)

        status = "OK" if result.success else "FAIL"
        print(f"  [{status}] {name} ({result.elapsed_seconds:.0f}s)")
        if result.error:
            print(f"         Error: {result.error[:200]}")

    print_summary(results)
    send_slack_summary(results)


if __name__ == "__main__":
    main()
