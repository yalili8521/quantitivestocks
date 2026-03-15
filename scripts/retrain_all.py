#!/usr/bin/env python3
"""
Retrain all models — LSTM (v2 regression), LightGBM intraday, TFT+XGBoost swing.

Meta RF is DEPRECATED in v2 (regression output magnitude IS the confidence).
Use --step meta to force-train it if needed for backward compatibility.

Usage:
    python scripts/retrain_all.py                     # everything except meta RF
    python scripts/retrain_all.py --step lstm          # LSTM only
    python scripts/retrain_all.py --step meta          # meta RF only (DEPRECATED)
    python scripts/retrain_all.py --step new           # intraday + swing only
    python scripts/retrain_all.py --step all           # everything including deprecated meta RF
    python scripts/retrain_all.py --mode daily         # daily symbols only (LSTM)
    python scripts/retrain_all.py --mode intraday      # intraday symbols only (LSTM)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from ml_model import train_model, train_meta_model  # LSTM training (deprecated)
from utils import DEFAULT_MODEL_DIR, SWING_MODEL_DIR, INTRADAY_MODEL_DIR, CRYPTO_MODEL_DIR
from signals_engine import build_adapter

# ---------------------------------------------------------------------------
# Symbol universe
# ---------------------------------------------------------------------------
DAILY_SYMBOLS = [
    # Account 1 (Intraday account — daily models also needed for meta RF timing)
    "SPY", "QQQ", "IWM", "SOXX",
    # Account 2 (Swing — daily mode)
    "EWT", "GLD", "EEM", "SLV",
    # Account 3 (Expansion — daily mode)
    "EWJ", "EWS", "XLE", "INDA",
]

INTRADAY_SYMBOLS = ["SPY", "QQQ", "IWM", "SOXX"]
SWING_SYMBOLS = ["EWT", "GLD", "EEM", "SLV"]
# Crypto symbols in yfinance format (Alpaca BTC/USD → yfinance BTC-USD)
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]
EPOCHS = 60
LOOKBACK = 1000

PYTHON = sys.executable  # use same Python that launched this script


# ---------------------------------------------------------------------------
# LSTM (daily + intraday attention-LSTM)
# ---------------------------------------------------------------------------
def train_all_lstm(mode: str, adapter, fred_key: str | None) -> None:
    symbols = INTRADAY_SYMBOLS if mode == "intraday" else DAILY_SYMBOLS
    interval = "5min"
    print(f"\n{'='*65}")
    print(f"  LSTM Training — {mode.upper()} ({len(symbols)} symbols)")
    print(f"{'='*65}\n")

    failed = []
    for sym in symbols:
        print(f"\n--- Training {sym} ({mode}) ---")
        t0 = time.time()
        try:
            train_model(
                symbol=sym,
                adapter=adapter,
                fred_key=fred_key,
                epochs=EPOCHS,
                lookback=LOOKBACK,
                save_dir=DEFAULT_MODEL_DIR,
                mode=mode,
                intraday_interval=interval,
            )
            elapsed = time.time() - t0
            print(f"  [OK] {sym} ({mode}) — {elapsed:.0f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [FAIL] {sym} ({mode}) after {elapsed:.0f}s: {exc}")
            failed.append(sym)

    print(f"\n{'='*65}")
    print(f"  LSTM {mode.upper()} done. Failed: {failed or 'none'}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Meta RF (daily + intraday)
# ---------------------------------------------------------------------------
def train_all_meta(mode: str, adapter, fred_key: str | None) -> None:
    symbols = INTRADAY_SYMBOLS if mode == "intraday" else DAILY_SYMBOLS
    interval = "5min"
    print(f"\n{'='*65}")
    print(f"  Meta RF Training — {mode.upper()} ({len(symbols)} symbols)")
    print(f"{'='*65}\n")

    failed = []
    for sym in symbols:
        print(f"\n--- Meta RF for {sym} ({mode}) ---")
        t0 = time.time()
        try:
            train_meta_model(
                symbol=sym,
                adapter=adapter,
                fred_key=fred_key,
                lookback=LOOKBACK,
                save_dir=DEFAULT_MODEL_DIR,
                mode=mode,
                intraday_interval=interval,
            )
            elapsed = time.time() - t0
            print(f"  [OK] {sym} ({mode}) meta RF — {elapsed:.0f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [FAIL] {sym} ({mode}) meta RF after {elapsed:.0f}s: {exc}")
            failed.append(sym)

    print(f"\n{'='*65}")
    print(f"  Meta RF {mode.upper()} done. Failed: {failed or 'none'}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# LightGBM intraday (via main.py train-intraday)
# ---------------------------------------------------------------------------
def train_all_intraday() -> None:
    print(f"\n{'='*65}")
    print(f"  LightGBM Intraday Training ({len(INTRADAY_SYMBOLS)} symbols)")
    print(f"{'='*65}\n")

    t0 = time.time()
    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "main.py"),
        "train-intraday",
        "--symbols", ",".join(INTRADAY_SYMBOLS),
        "--provider", "yahoo",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"\n  [{status}] LightGBM intraday — {elapsed:.0f}s")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# PatchTST swing (via main.py train-swing)
# ---------------------------------------------------------------------------
def train_all_swing() -> None:
    print(f"\n{'='*65}")
    print(f"  PatchTST Swing Training ({len(SWING_SYMBOLS)} symbols)")
    print(f"{'='*65}\n")

    t0 = time.time()
    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "main.py"),
        "train-swing",
        "--symbols", ",".join(SWING_SYMBOLS),
        "--provider", "yahoo",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"\n  [{status}] PatchTST swing — {elapsed:.0f}s")
    print(f"{'='*65}\n")



# ---------------------------------------------------------------------------
# Crypto swing models (separate model dir: models/crypto/)
# ---------------------------------------------------------------------------
def train_all_crypto() -> None:
    """Train swing XGBoost+TFT models for crypto symbols, saved to models/crypto/."""
    print(f"\n{'='*65}")
    print(f"  Crypto Swing Training ({len(CRYPTO_SYMBOLS)} symbols → {CRYPTO_MODEL_DIR})")
    print(f"{'='*65}\n")

    t0 = time.time()
    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "main.py"),
        "train-swing",
        "--symbols", ",".join(CRYPTO_SYMBOLS),
        "--provider", "yahoo",
        "--save-dir", CRYPTO_MODEL_DIR,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"\n  [{status}] Crypto swing — {elapsed:.0f}s")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain all models (LSTM v2 regression, intraday, swing).",
    )
    parser.add_argument(
        "--step",
        choices=["lstm", "meta", "both", "new", "crypto", "all"],
        default="all",
        help=(
            "lstm=LSTM only, meta=meta RF only (DEPRECATED), both=lstm+meta, "
            "new=intraday+swing, crypto=crypto swing only, all=everything (default; skips meta RF)"
        ),
    )
    parser.add_argument("--mode", choices=["daily", "intraday", "both"], default="both",
                        help="For LSTM steps: daily, intraday, or both (default: both)")
    args = parser.parse_args()

    fred_key = os.environ.get("FRED_API_KEY", "5e06c25a712146a59c69804dc0cdec4c")
    adapter = build_adapter("yahoo")

    modes = ["daily", "intraday"] if args.mode == "both" else [args.mode]

    t_start = time.time()

    # Step 1: LSTM (daily + intraday attention-LSTM, v2 regression)
    if args.step in ("lstm", "both", "all"):
        for mode in modes:
            train_all_lstm(mode, adapter, fred_key)

    # Step 2: Meta RF (DEPRECATED in v2 — regression output IS the signal)
    # Only run if explicitly requested via --step meta or --step both
    if args.step in ("meta", "both"):
        print("\n  [!] WARNING: Meta RF is DEPRECATED in v2 regression architecture.")
        print("  [!] The regression model's expected return magnitude IS the confidence signal.")
        print("  [!] Training meta RF for backward compatibility only.\n")
        for mode in modes:
            train_all_meta(mode, adapter, fred_key)

    # Step 3: LightGBM intraday
    if args.step in ("new", "all"):
        train_all_intraday()

    # Step 4: TFT+XGBoost swing
    if args.step in ("new", "all"):
        train_all_swing()

    # Step 5: Crypto swing (separate model dir)
    if args.step in ("crypto", "all"):
        train_all_crypto()

    total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  ALL TRAINING COMPLETE — {total/60:.1f} min total")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
