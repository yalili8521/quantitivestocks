#!/usr/bin/env python3
"""
Weekly ETF Pipeline — Automated screening + training for ETF trading groups.
=============================================================================

Chains eight steps in sequence:
    1. read-research          — Ingest last 7 days of daily research reports
    2. screen-etf-universe    — Layer 0: discover tradeable ETFs (yfinance)
    3. rank-etfs              — Rank by tradability + momentum + research bias (swing)
    3b. rank-etfs-intraday    — Rank for intraday: base + activity + swing tilt
    4. batch-backtest         — Train + OOS backtest screened ETFs
    5. train-swing            — Train swing models for newly promoted ETFs
    6. train-intraday         — Train intraday models for promoted ETFs
    7. model-health           — Check model health, report degradation

Research integration:
    Daily markdown files in research/daily/YYYY-MM-DD.md are parsed by
    ResearchReader to extract ticker mentions, sector sentiment, and macro
    regime signals. These become bias weights that adjust the screening
    composite score, boosting research-supported candidates.

Usage:
    python scripts/weekly_etf_pipeline.py
    python scripts/weekly_etf_pipeline.py --skip screen-etf-universe
    python scripts/weekly_etf_pipeline.py --skip batch-backtest train-intraday
    python scripts/weekly_etf_pipeline.py --research-days 14

Schedule (PowerShell, admin):
    $action = New-ScheduledTaskAction `
        -Execute "C:\\Users\\yalil\\OneDrive\\Desktop\\AI-projects\\quantitivestocks\\.venv\\Scripts\\python.exe" `
        -Argument "-u scripts/weekly_etf_pipeline.py" `
        -WorkingDirectory "C:\\Users\\yalil\\OneDrive\\Desktop\\AI-projects\\quantitivestocks"
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "08:00PM"
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd
    Register-ScheduledTask -TaskName "QuantStocks-WeeklyETFPipeline" `
        -Action $action -Trigger $trigger -Settings $settings `
        -Description "Weekly ETF pipeline: research, screen, train, backtest, promote"
"""

from __future__ import annotations

import argparse
import json
import logging
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

log = logging.getLogger("weekly_etf_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_read_research(research_days: int = 7) -> StepResult:
    """Step 1: Read daily research reports and extract signals.

    Saves extracted signals to outputs/research_signals.json for downstream steps.
    """
    t0 = time.time()
    try:
        from research_reader import ResearchReader

        reader = ResearchReader()
        signals = reader.read_last_n_days(research_days)
        elapsed = time.time() - t0

        if not signals.dates_read:
            return StepResult("read-research", True, elapsed,
                              details="No research files found — screening will run without bias")

        bias = reader.compute_bias_weights(signals)
        cat_bias = reader.compute_category_bias(signals)

        # Save signals for downstream steps
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
        signals_path = os.path.join(PROJECT_ROOT, "outputs", "research_signals.json")
        payload = {
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "dates_read": signals.dates_read,
            "ticker_bias": bias,
            "category_bias": cat_bias,
            "ticker_count": len(signals.ticker_mentions),
            "sector_count": len(signals.sector_signals),
            "macro_count": len(signals.macro_signals),
            "themes": signals.themes,
            "macro_summary": [
                {"date": m.date, "regime": m.regime,
                 "rate_direction": m.rate_direction,
                 "confidence": m.confidence}
                for m in signals.macro_signals
            ],
        }
        with open(signals_path, "w") as f:
            json.dump(payload, f, indent=2)

        # Build details string
        top_bull = sorted(bias.items(), key=lambda x: -x[1])[:3]
        top_bear = sorted(bias.items(), key=lambda x: x[1])[:3]
        bull_str = ", ".join(f"{s}={w:+.2f}" for s, w in top_bull if w > 0)
        bear_str = ", ".join(f"{s}={w:+.2f}" for s, w in top_bear if w < 0)
        details = (f"{len(signals.dates_read)} reports, "
                   f"{len(bias)} tickers biased")
        if bull_str:
            details += f" | bull: {bull_str}"
        if bear_str:
            details += f" | bear: {bear_str}"

        return StepResult("read-research", True, elapsed, details=details)
    except Exception as exc:
        return StepResult("read-research", False, time.time() - t0, error=str(exc))


def run_screen_etf_universe() -> StepResult:
    """Step 2: Screen ETF universe via yfinance."""
    t0 = time.time()
    try:
        result = _run_command([PYTHON, MAIN_PY, "screen-etf-universe"])
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("screen-etf-universe", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")
        details = ""
        for line in result.stdout.splitlines():
            if "etf" in line.lower() and "qualified" in line.lower():
                details = line.strip()
                break
        if not details:
            for line in result.stdout.splitlines():
                if "final" in line.lower() or "saved" in line.lower():
                    details = line.strip()
                    break
        return StepResult("screen-etf-universe", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("screen-etf-universe", False, time.time() - t0, error="timeout (1h)")
    except Exception as exc:
        return StepResult("screen-etf-universe", False, time.time() - t0, error=str(exc))


# ---------------------------------------------------------------------------
# Ranking weights (configurable)
# ---------------------------------------------------------------------------
# final_score = w_tradability * base_score
#             + w_momentum   * momentum_score
#             + w_model_perf * model_perf_score
#             + w_research   * (0.3 * research_bias)
#
# When research/model_perf unavailable, weights are redistributed to momentum.
DEFAULT_W_TRADABILITY = 0.25
DEFAULT_W_MOMENTUM = 0.25
DEFAULT_W_MODEL_PERF = 0.35
DEFAULT_W_RESEARCH = 0.15


def _load_oos_metrics(model_dir: str) -> dict:
    """Load OOS backtest metrics from promoted_symbols.json.

    Returns {symbol: {sharpe_ratio, profit_factor, win_rate, ...}} or empty dict.
    """
    path = os.path.join(model_dir, "promoted_symbols.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {d["symbol"]: d for d in data.get("details", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def _compute_model_perf_score(oos: dict) -> float:
    """Convert OOS metrics to a [0, 1] score.

    score = 0.6 * sharpe_norm + 0.4 * pf_norm
    sharpe_norm: Sharpe clipped to [0, 3], divided by 3
    pf_norm:     log(PF) clipped to [0, log(10)], divided by log(10)
    """
    import math
    sharpe = max(0, min(oos.get("sharpe_ratio", 0), 3.0))
    sharpe_norm = sharpe / 3.0

    pf = max(1.0, oos.get("profit_factor", 1.0))
    pf_norm = min(math.log(pf), math.log(10)) / math.log(10)

    return 0.6 * sharpe_norm + 0.4 * pf_norm


def _load_intraday_model_metrics(model_dir: str, symbols: list) -> dict:
    """Load intraday model validation metrics from config JSONs.

    Returns {symbol: {val_ic, val_dir_acc}} or empty dict.
    """
    metrics = {}
    for sym in symbols:
        path = os.path.join(model_dir, f"{sym}_lgb_intraday_etf_config.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                cfg = json.load(f)
            metrics[sym] = {
                "val_ic": cfg.get("val_ic", 0),
                "val_dir_acc": cfg.get("val_dir_acc", 0.5),
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return metrics


def _compute_intraday_model_perf_score(metrics: dict) -> float:
    """Convert intraday model metrics to a [0, 1] score.

    score = 0.5 * ic_norm + 0.5 * acc_norm
    ic_norm: IC clipped to [0, 0.4], divided by 0.4
    acc_norm: dir_acc clipped to [0.5, 0.7], normalized to [0, 1]
    """
    ic = max(0, min(metrics.get("val_ic", 0), 0.4))
    ic_norm = ic / 0.4

    acc = max(0.5, min(metrics.get("val_dir_acc", 0.5), 0.7))
    acc_norm = (acc - 0.5) / 0.2

    return 0.5 * ic_norm + 0.5 * acc_norm


def run_apply_research_bias(
    w_tradability: float = DEFAULT_W_TRADABILITY,
    w_momentum: float = DEFAULT_W_MOMENTUM,
    w_model_perf: float = DEFAULT_W_MODEL_PERF,
    w_research: float = DEFAULT_W_RESEARCH,
) -> StepResult:
    """Step 3: Rank ETFs by tradability + momentum + model performance + research bias.

    Reads etf_universe.json, computes momentum scores from market data,
    loads OOS backtest metrics, loads research signals (optional), and
    produces a final ranked list. Writes etf_candidates_ranked.json.

    final_score = w_tradability * base_score
                + w_momentum   * momentum_score
                + w_model_perf * model_perf_score
                + w_research   * (0.3 * research_bias)

    Weights are redistributed when data is unavailable:
    - No research → w_research goes to momentum
    - No OOS metrics for a symbol → w_model_perf goes to momentum for that symbol
    """
    t0 = time.time()
    try:
        import math
        from utils import SWING_MODEL_DIR
        from etf_screener import compute_etf_momentum_scores

        signals_path = os.path.join(PROJECT_ROOT, "outputs", "research_signals.json")
        universe_path = os.path.join(SWING_MODEL_DIR, "etf_universe.json")

        # Load universe
        if not os.path.exists(universe_path):
            return StepResult("rank-etfs", False, time.time() - t0,
                              error="No etf_universe.json — run screen-etf-universe first")

        with open(universe_path) as f:
            universe = json.load(f)
        etfs = universe.get("coins", [])

        if not etfs:
            return StepResult("rank-etfs", False, time.time() - t0,
                              error="Empty ETF universe")

        # --- Momentum scores (market data, no research needed) ---
        symbols = [e["symbol"] for e in etfs]
        momentum_scores = compute_etf_momentum_scores(symbols)

        # --- OOS backtest metrics (from promoted_symbols.json) ---
        oos_metrics = _load_oos_metrics(SWING_MODEL_DIR)
        oos_available = bool(oos_metrics)
        if oos_available:
            log.info("Loaded OOS metrics for %d symbols", len(oos_metrics))
        else:
            log.warning("No promoted_symbols.json — model_perf weight redistributed to momentum")

        # --- Research bias (optional) ---
        ticker_bias = {}
        category_bias = {}
        research_available = False
        if os.path.exists(signals_path):
            try:
                with open(signals_path) as f:
                    signals = json.load(f)
                ticker_bias = signals.get("ticker_bias", {})
                category_bias = signals.get("category_bias", {})
                research_available = True
            except (json.JSONDecodeError, KeyError):
                log.warning("Could not parse research_signals.json")

        # Effective weights: redistribute unavailable component weights to momentum
        eff_w_trad = w_tradability
        eff_w_res = w_research if research_available else 0.0
        # Global model_perf weight (per-symbol redistribution below if no OOS for that sym)
        eff_w_model_global = w_model_perf if oos_available else 0.0
        # Base momentum weight absorbs unavailable components
        eff_w_mom_base = w_momentum + (w_research if not research_available else 0.0) \
                         + (w_model_perf if not oos_available else 0.0)

        # --- Score each ETF ---
        ranked = []
        for etf in etfs:
            sym = etf["symbol"]
            dollar_vol = etf.get("avg_dollar_volume_30d", 0)
            ann_vol = etf.get("realized_vol_ann", 0)

            # Base score (tradability): liquidity + volatility quality
            vol_score = math.log10(max(dollar_vol, 1e6)) / 12
            if 0.15 <= ann_vol <= 0.40:
                vol_quality = 1.0
            elif ann_vol > 0:
                vol_quality = max(0.3, 1.0 - abs(ann_vol - 0.25) * 2)
            else:
                vol_quality = 0.5
            base_score = 0.5 * vol_score + 0.5 * vol_quality

            # Momentum score
            mom_score = momentum_scores.get(sym, 0.5)

            # Model performance score (OOS backtest)
            oos = oos_metrics.get(sym)
            if oos:
                model_perf = _compute_model_perf_score(oos)
                sym_w_model = eff_w_model_global
            else:
                model_perf = 0.0
                sym_w_model = 0.0  # redistribute to momentum for this symbol

            # Per-symbol momentum weight: absorbs unused model_perf weight
            sym_w_mom = eff_w_mom_base + (eff_w_model_global - sym_w_model)

            # Research bias: ticker-level, falling back to category-level
            bias = ticker_bias.get(sym, 0.0)
            cat = etf.get("category", "")
            if bias == 0.0 and cat:
                bias = category_bias.get(cat, 0.0) * 0.5

            # Final composite score
            final_score = (
                eff_w_trad * base_score
                + sym_w_mom * mom_score
                + sym_w_model * model_perf
                + eff_w_res * (0.3 * bias)
            )

            ranked.append({
                **etf,
                "base_score": round(base_score, 4),
                "momentum_score": round(mom_score, 4),
                "model_perf_score": round(model_perf, 4),
                "research_bias": round(bias, 4),
                "final_score": round(final_score, 4),
            })

        # Sort by final_score descending
        ranked.sort(key=lambda x: -x["final_score"])

        # --- Diagnostic log for key symbols ---
        spotlight = {"SPY", "IWM", "MCHI", "LQD", "SMH", "XLE", "GLD", "EEM", "USO"}
        log.info("--- ETF Ranking Diagnostic (weights: trad=%.2f mom=%.2f model=%.2f res=%.2f) ---",
                 eff_w_trad, eff_w_mom_base, eff_w_model_global, eff_w_res)
        log.info("  %-6s %6s %6s %6s %6s %7s  %s",
                 "Symbol", "Base", "Mom", "Model", "Bias", "Final", "Rank")
        for rank_pos, r in enumerate(ranked, 1):
            sym = r["symbol"]
            if sym in spotlight:
                log.info("  %-6s %6.3f %6.3f %6.3f %+6.3f %7.4f  #%d",
                         sym, r["base_score"], r["momentum_score"],
                         r["model_perf_score"], r["research_bias"],
                         r["final_score"], rank_pos)

        # Save ranked candidates
        out_path = os.path.join(SWING_MODEL_DIR, "etf_candidates_ranked.json")
        payload = {
            "ranked_at": datetime.now(timezone.utc).isoformat(),
            "research_available": research_available,
            "weights": {
                "w_tradability": eff_w_trad,
                "w_momentum": eff_w_mom_base,
                "w_model_perf": eff_w_model_global,
                "w_research": eff_w_res,
            },
            "oos_metrics_available": oos_available,
            "count": len(ranked),
            "candidates": ranked,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        elapsed = time.time() - t0

        # Build details string
        top5 = ", ".join(f"{r['symbol']}" for r in ranked[:5])
        mom_leaders = sorted(ranked, key=lambda x: -x["momentum_score"])[:3]
        mom_str = ", ".join(f"{r['symbol']}={r['momentum_score']:.2f}" for r in mom_leaders)
        details = f"{len(ranked)} ETFs ranked (top5: {top5})"
        details += f" | mom leaders: {mom_str}"
        if research_available:
            boosted = [r["symbol"] for r in ranked[:10] if r["research_bias"] > 0.05]
            if boosted:
                details += f" | research boost: {','.join(boosted[:3])}"

        return StepResult("rank-etfs", True, elapsed, details=details)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return StepResult("rank-etfs", False, time.time() - t0, error=str(exc))


# ---------------------------------------------------------------------------
# Intraday ranking weights
# ---------------------------------------------------------------------------
# intraday_rank_score = w_base * base_score
#                     + w_activity * intraday_activity_score
#                     + w_model_perf * model_perf_score
#                     + w_swing    * swing_final_score
INTRADAY_W_BASE = 0.30
INTRADAY_W_ACTIVITY = 0.25
INTRADAY_W_MODEL_PERF = 0.30
INTRADAY_W_SWING = 0.15


def run_rank_etfs_intraday() -> StepResult:
    """Step 3b: Rank ETFs for intraday trading.

    Uses a separate ranking formula optimized for intraday characteristics:
      intraday_rank_score = 0.5 * base_score
                          + 0.3 * intraday_activity_score
                          + 0.2 * swing_final_score

    Where intraday_activity_score captures:
      - ATR% (daily range as % of price)
      - Average daily high-low range %
      - Overnight gap frequency

    The swing_final_score provides a medium-term tilt so intraday doesn't
    diverge completely from the broader view.

    Reads etf_candidates_ranked.json (swing ranking) and produces
    etf_candidates_intraday_ranked.json.
    """
    t0 = time.time()
    try:
        from utils import SWING_MODEL_DIR
        from etf_screener import compute_etf_intraday_activity_scores

        # Load swing-ranked candidates (need base_score + final_score)
        ranked_path = os.path.join(SWING_MODEL_DIR, "etf_candidates_ranked.json")
        if not os.path.exists(ranked_path):
            return StepResult("rank-etfs-intraday", False, time.time() - t0,
                              error="No etf_candidates_ranked.json — run rank-etfs first")

        with open(ranked_path) as f:
            ranked_data = json.load(f)
        swing_candidates = ranked_data.get("candidates", [])

        if not swing_candidates:
            return StepResult("rank-etfs-intraday", False, time.time() - t0,
                              error="Empty swing candidates")

        symbols = [c["symbol"] for c in swing_candidates]

        # Build lookup for swing scores
        swing_lookup = {c["symbol"]: c for c in swing_candidates}

        # Compute intraday activity scores
        activity_scores = compute_etf_intraday_activity_scores(symbols)

        # Load intraday model validation metrics
        from utils import INTRADAY_MODEL_DIR
        intraday_metrics = _load_intraday_model_metrics(INTRADAY_MODEL_DIR, symbols)
        model_perf_available = bool(intraday_metrics)
        if model_perf_available:
            log.info("Loaded intraday model metrics for %d symbols", len(intraday_metrics))

        # Normalize swing_final_score to [0, 1] range for fair combination
        swing_scores_raw = [c.get("final_score", 0) for c in swing_candidates]
        swing_min = min(swing_scores_raw) if swing_scores_raw else 0
        swing_max = max(swing_scores_raw) if swing_scores_raw else 1
        swing_range = swing_max - swing_min if swing_max > swing_min else 1.0

        # Effective model_perf weight (redistributed if unavailable)
        eff_w_model = INTRADAY_W_MODEL_PERF if model_perf_available else 0.0
        eff_w_activity = INTRADAY_W_ACTIVITY + (INTRADAY_W_MODEL_PERF if not model_perf_available else 0.0)

        # Score each ETF for intraday
        intraday_ranked = []
        for sym in symbols:
            sc = swing_lookup[sym]
            base = sc.get("base_score", 0.5)
            activity = activity_scores.get(sym, 0.5)
            # Normalize swing score to [0, 1]
            swing_norm = (sc.get("final_score", 0) - swing_min) / swing_range

            # Model performance from intraday config (val_ic, val_dir_acc)
            m = intraday_metrics.get(sym)
            if m:
                model_perf = _compute_intraday_model_perf_score(m)
                sym_w_model = eff_w_model
            else:
                model_perf = 0.0
                sym_w_model = 0.0

            # Per-symbol activity weight absorbs unused model_perf
            sym_w_activity = eff_w_activity + (eff_w_model - sym_w_model)

            intraday_score = (
                INTRADAY_W_BASE * base
                + sym_w_activity * activity
                + sym_w_model * model_perf
                + INTRADAY_W_SWING * swing_norm
            )

            intraday_ranked.append({
                **sc,
                "intraday_activity": round(activity, 4),
                "intraday_model_perf": round(model_perf, 4),
                "swing_final_norm": round(swing_norm, 4),
                "intraday_rank_score": round(intraday_score, 4),
            })

        # Sort by intraday_rank_score descending
        intraday_ranked.sort(key=lambda x: -x["intraday_rank_score"])

        # Diagnostic log
        spotlight = {"SPY", "IWM", "QQQ", "SMH", "SOXX", "IGV", "XLE", "GLD", "USO"}
        log.info("--- Intraday ETF Ranking (weights: base=%.2f activity=%.2f model=%.2f swing=%.2f) ---",
                 INTRADAY_W_BASE, eff_w_activity, eff_w_model, INTRADAY_W_SWING)
        log.info("  %-6s %6s %6s %6s %6s %7s  %s",
                 "Symbol", "Base", "Activ", "Model", "SwNrm", "IDrank", "Rank")
        for rank_pos, r in enumerate(intraday_ranked, 1):
            sym = r["symbol"]
            if sym in spotlight:
                log.info("  %-6s %6.3f %6.3f %6.3f %6.3f %7.4f  #%d",
                         sym, r["base_score"], r["intraday_activity"],
                         r.get("intraday_model_perf", 0), r["swing_final_norm"],
                         r["intraday_rank_score"], rank_pos)

        # Save
        out_path = os.path.join(SWING_MODEL_DIR, "etf_candidates_intraday_ranked.json")
        payload = {
            "ranked_at": datetime.now(timezone.utc).isoformat(),
            "weights": {
                "w_base": INTRADAY_W_BASE,
                "w_activity": eff_w_activity,
                "w_model_perf": eff_w_model,
                "w_swing": INTRADAY_W_SWING,
            },
            "count": len(intraday_ranked),
            "candidates": intraday_ranked,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        elapsed = time.time() - t0

        # Comparison: show where intraday ranking diverges from swing
        swing_rank = {c["symbol"]: i for i, c in enumerate(swing_candidates, 1)}
        big_movers = []
        for rank_pos, r in enumerate(intraday_ranked[:15], 1):
            sym = r["symbol"]
            swing_pos = swing_rank.get(sym, 99)
            delta = swing_pos - rank_pos
            if abs(delta) >= 5:
                direction = "up" if delta > 0 else "down"
                big_movers.append(f"{sym}({direction}{abs(delta):+d})")

        top5 = ", ".join(r["symbol"] for r in intraday_ranked[:5])
        details = f"{len(intraday_ranked)} ETFs intraday-ranked (top5: {top5})"
        if big_movers:
            details += f" | vs swing: {', '.join(big_movers[:4])}"

        return StepResult("rank-etfs-intraday", True, elapsed, details=details)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return StepResult("rank-etfs-intraday", False, time.time() - t0, error=str(exc))


def run_batch_backtest() -> StepResult:
    """Step 4: Batch train + OOS backtest for screened ETFs."""
    t0 = time.time()
    try:
        result = _run_command(
            [PYTHON, MAIN_PY, "batch-backtest"],
            timeout=7200,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("batch-backtest", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")

        # Extract summary
        details = ""
        for line in result.stdout.splitlines():
            if "promoted" in line.lower() and "rejected" in line.lower():
                details = line.strip()
                break
        if not details:
            for line in result.stdout.splitlines():
                if "total:" in line.lower():
                    details = line.strip()
                    break

        return StepResult("batch-backtest", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("batch-backtest", False, time.time() - t0, error="timeout (2h)")
    except Exception as exc:
        return StepResult("batch-backtest", False, time.time() - t0, error=str(exc))


def run_train_new_swing() -> StepResult:
    """Step 5: Train swing models for newly promoted ETFs without models."""
    t0 = time.time()
    try:
        from utils import SWING_MODEL_DIR
        from etf_screener import load_promoted_symbols

        promoted = load_promoted_symbols(SWING_MODEL_DIR)
        if not promoted:
            return StepResult("train-swing", True, time.time() - t0,
                              details="No promoted symbols — skipped")

        # Check which promoted symbols lack a model
        import re as _re
        _SYM_RE = _re.compile(r"^[A-Z0-9\-/]{1,10}$")
        untrained = []
        for sym in promoted:
            if not _SYM_RE.match(sym):
                continue
            model_file = os.path.join(SWING_MODEL_DIR, f"{sym}_xgb_swing.joblib")
            if not os.path.exists(model_file):
                untrained.append(sym)

        if not untrained:
            return StepResult("train-swing", True, time.time() - t0,
                              details="All promoted ETFs already have models")

        sym_list = ",".join(untrained)
        result = _run_command(
            [PYTHON, MAIN_PY, "train-swing",
             "--symbols", sym_list,
             "--provider", "yahoo"],
            timeout=3600,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("train-swing", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")

        lines = result.stdout.splitlines() + result.stderr.splitlines()
        ok_count = sum(1 for l in lines if "Saved swing XGBoost" in l)
        details = f"{len(untrained)} new symbols, {ok_count} models saved"
        return StepResult("train-swing", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("train-swing", False, time.time() - t0, error="timeout (1h)")
    except Exception as exc:
        return StepResult("train-swing", False, time.time() - t0, error=str(exc))


def run_train_new_intraday() -> StepResult:
    """Step 6: Train intraday models for promoted ETFs without intraday models."""
    t0 = time.time()
    try:
        from utils import SWING_MODEL_DIR, INTRADAY_MODEL_DIR
        from etf_screener import load_promoted_symbols

        promoted = load_promoted_symbols(SWING_MODEL_DIR)
        if not promoted:
            return StepResult("train-intraday", True, time.time() - t0,
                              details="No promoted symbols — skipped")

        # Check which lack intraday models
        untrained = []
        for sym in promoted:
            lgb_path = os.path.join(INTRADAY_MODEL_DIR, f"{sym}_lgb_intraday_etf.joblib")
            if not os.path.exists(lgb_path):
                untrained.append(sym)

        if not untrained:
            return StepResult("train-intraday", True, time.time() - t0,
                              details="All promoted ETFs already have intraday models")

        sym_list = ",".join(untrained)
        result = _run_command(
            [PYTHON, MAIN_PY, "train-intraday",
             "--symbols", sym_list,
             "--provider", "alpaca"],
            timeout=3600,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return StepResult("train-intraday", False, elapsed,
                              error=result.stderr[-500:] if result.stderr else "non-zero exit")

        lines = result.stdout.splitlines() + result.stderr.splitlines()
        ok_count = sum(1 for l in lines if "ENSEMBLE:" in l or "LGB-only:" in l)
        details = f"{len(untrained)} new symbols, {ok_count} models saved"
        return StepResult("train-intraday", True, elapsed, details=details)
    except subprocess.TimeoutExpired:
        return StepResult("train-intraday", False, time.time() - t0, error="timeout (1h)")
    except Exception as exc:
        return StepResult("train-intraday", False, time.time() - t0, error=str(exc))


def run_model_health() -> StepResult:
    """Step 7: Check model health, report degraded models."""
    t0 = time.time()
    try:
        from model_monitor import ModelMonitor
        monitor = ModelMonitor()
        report = monitor.generate_report()
        elapsed = time.time() - t0

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


# ---------------------------------------------------------------------------
# Slack summary (reuses AlertEngine from crypto pipeline)
# ---------------------------------------------------------------------------

def send_slack_summary(results: list[StepResult]) -> None:
    """Send pipeline summary to Slack via AlertEngine."""
    try:
        from alerts import AlertEngine
        engine = AlertEngine()

        lines = ["Weekly ETF Pipeline Summary", ""]
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
    print(f"\n{'='*70}")
    print(f"  WEEKLY ETF PIPELINE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}\n")

    total_time = 0
    for r in results:
        total_time += r.elapsed_seconds
        status = "OK  " if r.success else "FAIL"
        print(f"  [{status}] {r.name:<25s} {r.elapsed_seconds:>7.0f}s", end="")
        if r.details:
            print(f"  {r.details}", end="")
        if r.error:
            print(f"  ERROR: {r.error[:100]}", end="")
        print()

    passed = sum(1 for r in results if r.success)
    print(f"\n  {passed}/{len(results)} steps passed — {total_time/60:.1f} min total")
    print(f"{'='*70}\n")


def _step_succeeded(results: list[StepResult], name: str) -> bool:
    """Check if a step succeeded (or was skipped)."""
    for r in results:
        if r.name == name:
            return r.success
    return False


# Dependency gates
DEPENDENCY_GATES: dict[str, list[tuple[str, str, str]]] = {
    "read-research": [],
    "screen-etf-universe": [],
    # rank-etfs needs universe, soft dep on research
    "rank-etfs": [
        ("screen-etf-universe", "hard", "no universe to rank"),
        ("read-research", "soft", "no research signals, using momentum + tradability only"),
    ],
    # intraday ranking needs swing ranking
    "rank-etfs-intraday": [
        ("rank-etfs", "hard", "no swing ranking to build on"),
    ],
    # batch-backtest needs ranked candidates
    "batch-backtest": [
        ("screen-etf-universe", "hard", "no screened ETFs to backtest"),
    ],
    # training needs promoted symbols from backtest
    "train-swing": [
        ("batch-backtest", "soft", "using existing promoted_symbols.json"),
    ],
    "train-intraday": [
        ("batch-backtest", "soft", "using existing promoted_symbols.json"),
    ],
    # model-health always runs
    "model-health": [],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly ETF pipeline: research, screen, train, backtest, promote"
    )
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Steps to skip: read-research, screen-etf-universe, "
                             "rank-etfs, rank-etfs-intraday, batch-backtest, "
                             "train-swing, train-intraday, model-health")
    parser.add_argument("--research-days", type=int, default=7,
                        help="Number of days of research to read (default: 7)")
    args = parser.parse_args()

    # Load env vars
    _load_env_file()

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Capture research_days for step 1
    _research_days = args.research_days

    steps: list[tuple[str, object]] = [
        ("read-research", lambda: run_read_research(_research_days)),
        ("screen-etf-universe", run_screen_etf_universe),
        ("rank-etfs", run_apply_research_bias),
        ("rank-etfs-intraday", run_rank_etfs_intraday),
        ("batch-backtest", run_batch_backtest),
        ("train-swing", run_train_new_swing),
        ("train-intraday", run_train_new_intraday),
        ("model-health", run_model_health),
    ]

    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Starting weekly ETF pipeline...",
          flush=True)
    print(f"  Steps: {', '.join(name for name, _ in steps)}")
    print(f"  Skipping: {args.skip or 'none'}")
    print(f"  Research lookback: {_research_days} days\n")

    results: list[StepResult] = []
    for name, fn in steps:
        if name in args.skip:
            results.append(StepResult(name=name, success=True,
                                      elapsed_seconds=0, details="SKIPPED"))
            print(f"  [SKIP] {name}")
            continue

        # Check dependency gates
        gates = DEPENDENCY_GATES.get(name, [])
        gate_blocked = False
        for upstream, severity, reason in gates:
            if not _step_succeeded(results, upstream):
                if severity == "hard":
                    msg = f"SKIPPED (upstream {upstream} failed: {reason})"
                    results.append(StepResult(name=name, success=False,
                                              elapsed_seconds=0, details=msg))
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
        if result.details:
            print(f"         {result.details}")
        if result.error:
            print(f"         Error: {result.error[:200]}")

    print_summary(results)
    send_slack_summary(results)


if __name__ == "__main__":
    main()
