#!/usr/bin/env python3
"""
Mean Decrease Accuracy (MDA) Feature Importance — Separate by Trading Mode
===========================================================================
Runs permutation importance on the primary LSTM across all active symbols,
separately for daily and intraday (5min) modes.

For each feature:
  1. Compute baseline accuracy on the held-out validation set
  2. Shuffle that one feature across all sequences (5 repeats)
  3. Importance = baseline_acc - mean(shuffled_acc)
  Positive importance = feature helps. Near-zero or negative = noise/redundant.

Usage:
    python scripts/mda_analysis.py
    python scripts/mda_analysis.py --mode daily
    python scripts/mda_analysis.py --mode intraday
    python scripts/mda_analysis.py --repeats 10
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml_model import (
    FEATURE_COLS, get_feature_cols, FeatureEngine, DirectionLSTM,
    _fetch_vix_for_training, DEFAULT_MODEL_DIR,
    SEQ_LEN, prepare_sequences_triple_barrier,
)
from signals_engine import build_adapter

# ---------------------------------------------------------------------------
# Active symbol universe by mode
# ---------------------------------------------------------------------------
SYMBOLS_DAILY = [
    "SPY", "QQQ", "IWM", "SOXX",          # intraday account (also have daily models)
    "EWT", "GLD", "EEM", "SLV", "TLT",    # swing account
    "EWJ", "EWS", "IGV", "XLE",            # expansion account
]
SYMBOLS_INTRADAY = [
    "SPY", "QQQ", "IWM", "SOXX",          # intraday account only
]


def build_val_set(
    symbol: str,
    mode: str,
    intraday_interval: str = "5min",
    lookback: int = 1000,
    fred_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct the validation set exactly as done during training."""
    adapter = build_adapter("yahoo")
    suffix = "" if mode == "daily" else f"_{intraday_interval}"

    # Check model exists
    weights_path = os.path.join(DEFAULT_MODEL_DIR, f"{symbol}_lstm{suffix}.pt")
    scaler_path  = os.path.join(DEFAULT_MODEL_DIR, f"{symbol}_scaler{suffix}.json")
    if not os.path.exists(weights_path):
        print(f"  [{symbol}] No model found at {weights_path} — skipping.")
        return None

    try:
        # Fetch data
        if mode == "daily":
            bars = adapter.fetch_daily(symbol, lookback)
        else:
            bars = adapter.fetch_intraday(symbol, intraday_interval, lookback_days=lookback)

        if len(bars) < SEQ_LEN + 50:
            print(f"  [{symbol}] Not enough bars ({len(bars)}) — skipping.")
            return None

        vix_df = _fetch_vix_for_training(fred_key, lookback_days=max(lookback, 500))

        # Build + normalize features (same process as training)
        engine = FeatureEngine()
        features = engine.build_features(bars, vix_df, mode=mode, symbol=symbol)
        split_idx = int(len(features) * 0.8)
        engine.fit_scaler(features.iloc[:split_idx])
        full_norm = engine.transform(features)

        # Triple-barrier labels
        if mode == "intraday":
            pt_pct, sl_pct, horizon = 0.005, 0.003, 10
        else:
            pt_pct, sl_pct, horizon = 0.015, 0.010, 5

        X_all, y_all = prepare_sequences_triple_barrier(
            full_norm, bars, SEQ_LEN, pt_pct=pt_pct, sl_pct=sl_pct, horizon=horizon,
        )

        # Same 80/20 + embargo split as training
        seq_split = int(len(X_all) * 0.8)
        embargo   = SEQ_LEN
        X_val = X_all[seq_split + embargo:]
        y_val = y_all[seq_split + embargo:]

        if len(X_val) < 20:
            print(f"  [{symbol}] Validation set too small ({len(X_val)}) — skipping.")
            return None

        return X_val, y_val

    except Exception as exc:
        print(f"  [{symbol}] Error building val set: {exc}")
        return None


def mda_for_symbol(
    symbol: str,
    mode: str,
    intraday_interval: str = "5min",
    n_repeats: int = 5,
    fred_key: str | None = None,
) -> dict[str, float] | None:
    """Run MDA for one symbol. Returns {feature: importance_score}."""
    suffix = "" if mode == "daily" else f"_{intraday_interval}"

    result = build_val_set(symbol, mode, intraday_interval, fred_key=fred_key)
    if result is None:
        return None
    X_val, y_val = result

    # Load trained model
    weights_path = os.path.join(DEFAULT_MODEL_DIR, f"{symbol}_lstm{suffix}.pt")
    model = DirectionLSTM(n_features=len(get_feature_cols(mode, symbol)))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    # Baseline accuracy
    with torch.no_grad():
        probs = model(torch.FloatTensor(X_val)).squeeze().numpy()
    baseline_preds = (probs > 0.5).astype(int)
    baseline_acc   = float((baseline_preds == y_val).mean())

    importances: dict[str, float] = {}

    for feat_idx, feat_name in enumerate(get_feature_cols(mode, symbol)):
        shuffled_accs = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            # Shuffle this feature's values across ALL time steps in ALL sequences
            flat_vals = X_perm[:, :, feat_idx].flatten()
            np.random.shuffle(flat_vals)
            X_perm[:, :, feat_idx] = flat_vals.reshape(X_perm.shape[0], X_perm.shape[1])

            with torch.no_grad():
                perm_probs = model(torch.FloatTensor(X_perm)).squeeze().numpy()
            perm_acc = float(((perm_probs > 0.5).astype(int) == y_val).mean())
            shuffled_accs.append(perm_acc)

        importance = baseline_acc - float(np.mean(shuffled_accs))
        importances[feat_name] = importance

    print(f"  [{symbol}] baseline_acc={baseline_acc:.3f}  val_n={len(y_val)}")
    return importances


def run_mda(mode: str, n_repeats: int, fred_key: str | None) -> None:
    symbols = SYMBOLS_DAILY if mode == "daily" else SYMBOLS_INTRADAY
    intraday_interval = "5min"

    print(f"\n{'='*65}")
    print(f"  MDA Feature Importance — {mode.upper()} mode ({n_repeats} repeats/feature)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"{'='*65}\n")

    # Accumulate importance scores across symbols
    all_importances: dict[str, list[float]] = defaultdict(list)
    n_successful = 0

    for sym in symbols:
        print(f"  Running MDA for {sym}...")
        result = mda_for_symbol(sym, mode, intraday_interval, n_repeats, fred_key)
        if result is not None:
            for feat, imp in result.items():
                all_importances[feat].append(imp)
            n_successful += 1

    if n_successful == 0:
        print("  No symbols completed successfully.")
        return

    # Compute mean + std importance across symbols
    mode_feat_cols = get_feature_cols(mode)
    summary: list[tuple[str, float, float]] = []
    for feat in mode_feat_cols:
        vals = all_importances[feat]
        mean_imp = float(np.mean(vals)) if vals else 0.0
        std_imp  = float(np.std(vals))  if vals else 0.0
        summary.append((feat, mean_imp, std_imp))

    # Sort by mean importance descending
    summary.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Results averaged over {n_successful}/{len(symbols)} symbols\n")
    print(f"  {'Rank':<5} {'Feature':<20} {'Importance':>12} {'Std':>8}  {'Signal?'}")
    print(f"  {'-'*5} {'-'*20} {'-'*12} {'-'*8}  {'-'*10}")

    for rank, (feat, mean_imp, std_imp) in enumerate(summary, 1):
        if mean_imp > 0.005:
            signal = "STRONG"
        elif mean_imp > 0.001:
            signal = "weak"
        elif mean_imp > -0.001:
            signal = "noise"
        else:
            signal = "HURTS"
        print(f"  {rank:<5} {feat:<20} {mean_imp:>+12.4f} {std_imp:>8.4f}  {signal}")

    # Clear candidates to drop (noise or hurts, consistently)
    drop_candidates = [feat for feat, mean_imp, _ in summary if mean_imp <= 0.001]
    keep_features   = [feat for feat, mean_imp, _ in summary if mean_imp > 0.001]

    print(f"\n  --- VERDICT ({mode.upper()} mode) ---")
    print(f"  KEEP  ({len(keep_features)} features): {', '.join(keep_features)}")
    print(f"  DROP  ({len(drop_candidates)} features): {', '.join(drop_candidates)}")
    print(f"\n  Dropping these would reduce FEATURE_COLS_{mode.upper()} from {len(mode_feat_cols)} -> {len(keep_features)}")
    print(f"  Retrain all {mode} models after pruning.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MDA feature importance by trading mode")
    parser.add_argument("--mode", choices=["daily", "intraday", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Permutation repeats per feature (higher = more stable, slower)")
    args = parser.parse_args()

    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        # Try loading from secrets file
        env_path = os.path.join(PROJECT_ROOT, "secrets", "alpaca.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FRED_API_KEY="):
                        fred_key = line.split("=", 1)[1].strip()
                        break

    modes = ["daily", "intraday"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mda(mode, args.repeats, fred_key)


if __name__ == "__main__":
    main()
