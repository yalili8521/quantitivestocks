"""
Unified OOS Feedback Loop — shared by ETF and crypto groups.

Provides:
  - load_promoted_oos(model_dir)      → {symbol: {sharpe, pf, win_rate, ...}}
  - compute_composite_scores(...)     → {symbol: composite_score} sorted desc
  - blended_sharpe(...)               → OOS+live Sharpe blend with exponential decay

Extracted from paper_trader._compute_composite_scores() so both ETF and
crypto paths use the same James-Stein shrinkage + rank-normalization logic.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load promoted symbols (OOS backtest metrics)
# ---------------------------------------------------------------------------

def load_promoted_oos(model_dir: str) -> Dict[str, dict]:
    """Load OOS backtest metrics from promoted_symbols.json.

    Returns {symbol: {sharpe_ratio, profit_factor, win_rate, ...}} keyed by
    symbol for fast lookup.  Returns empty dict if file missing or corrupt.
    """
    path = os.path.join(model_dir, "promoted_symbols.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        details = data.get("details", [])
        return {d["symbol"]: d for d in details if "symbol" in d}
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        log.warning("Failed to load promoted_symbols from %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Blended Sharpe (OOS + live rolling Sharpe with exponential decay)
# ---------------------------------------------------------------------------

def blended_sharpe(
    oos_sharpe: float,
    live_sharpe: Optional[float],
    n_trades: int,
    min_trades: int = 10,
    decay_halflife: float = 60.0,
    max_live_weight: float = 0.70,
) -> float:
    """Blend OOS Sharpe with live rolling Sharpe using exponential decay.

    w_live = min(1 - 0.5^(n_trades/decay_halflife), max_live_weight)
    blended = (1 - w_live) * oos_sharpe + w_live * live_sharpe

    After ``decay_halflife`` trades, live gets 50% weight.  Cap at
    ``max_live_weight`` — never fully abandon the OOS prior (Rob Carver's
    shrinkage principle).

    Returns oos_sharpe if insufficient live data.
    """
    if live_sharpe is None or n_trades < min_trades:
        return oos_sharpe
    w_live = min(1.0 - 0.5 ** (n_trades / decay_halflife), max_live_weight)
    return (1.0 - w_live) * oos_sharpe + w_live * live_sharpe


# ---------------------------------------------------------------------------
# Rank normalization (robust to outliers)
# ---------------------------------------------------------------------------

def _rank_normalize(values: List[float]) -> List[float]:
    """Map values to [0, 1] by rank-percentile.  Robust to outliers."""
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank_pos, idx in enumerate(order):
        ranks[idx] = rank_pos / (n - 1)
    return ranks


# ---------------------------------------------------------------------------
# Composite scoring (James-Stein shrinkage: selector + OOS Sharpe)
# ---------------------------------------------------------------------------

def compute_composite_scores(
    rankings: List[Tuple[str, float]],
    oos_sharpe: Dict[str, float],
    derisk_lookup: Optional[Callable[[str], Optional[Tuple[int, Optional[float]]]]] = None,
    btc_correlations: Optional[Dict[str, float]] = None,
    fee_costs: Optional[Dict[str, float]] = None,
    min_oos_sharpe: float = -0.20,
) -> Dict[str, float]:
    """Composite score = w_sel * selector + w_sharpe * blended_sharpe (rank-normalized).

    Dynamic weighting (James-Stein shrinkage):
      - Cold start (< 20 live trades):  30% selector / 70% OOS Sharpe
      - Warming   (20-60 trades):       linear interpolation
      - Live      (60+ trades):         70% selector / 30% OOS Sharpe

    Parameters
    ----------
    rankings : list of (symbol, selector_score)
        Output from ETFSelector.rank() or CoinSelector.rank(), sorted desc.
    oos_sharpe : dict
        {symbol: OOS Sharpe ratio} from promoted_symbols or trading.json registry.
    derisk_lookup : callable, optional
        ``fn(symbol) -> (n_trades, live_rolling_sharpe) | None``
        Used for James-Stein weight computation and live Sharpe blending.
        If None, cold-start weights (30/70) are used.
    btc_correlations : dict, optional
        {symbol: correlation_with_anchor} — BTC for crypto, SPY for ETFs.
    fee_costs : dict, optional
        {symbol: round_trip_cost_fraction} from cost_model. Symbols with
        above-median costs are penalised; neutral when cost == median.
    min_oos_sharpe : float
        Hard floor: symbols below this OOS Sharpe are excluded.

    Returns
    -------
    {symbol: composite_score} sorted descending.
    """
    # --- Determine dynamic weights based on median live trade count ----------
    n_trades_list = []
    if derisk_lookup is not None:
        for sym, _ in rankings:
            info = derisk_lookup(sym)
            if info is not None:
                n_trades_list.append(info[0])

    median_trades = (
        sorted(n_trades_list)[len(n_trades_list) // 2]
        if n_trades_list else 0
    )

    if median_trades < 20:
        w_selector, w_sharpe = 0.30, 0.70  # cold start — trust OOS prior
    elif median_trades < 60:
        frac = (median_trades - 20) / 40.0
        w_selector = 0.30 + 0.40 * frac
        w_sharpe = 1.0 - w_selector
    else:
        w_selector, w_sharpe = 0.70, 0.30  # trust live selector

    log.info(
        "Composite weights: selector=%.0f%% sharpe=%.0f%% (median_trades=%d)",
        w_selector * 100, w_sharpe * 100, median_trades,
    )

    # --- Build candidate list, blending OOS + live Sharpe --------------------
    candidates: List[Tuple[str, float, float]] = []
    for sym, sel_score in rankings:
        oos_s = oos_sharpe.get(sym, 0.0)
        if oos_s < min_oos_sharpe:
            log.info(
                "Composite: %s excluded (OOS Sharpe %.2f < %.2f)",
                sym, oos_s, min_oos_sharpe,
            )
            continue

        # Blend OOS + live Sharpe
        live_sharpe_val = None
        n_trades = 0
        if derisk_lookup is not None:
            info = derisk_lookup(sym)
            if info is not None:
                n_trades, live_sharpe_val = info
        blended = blended_sharpe(oos_s, live_sharpe_val, n_trades)
        candidates.append((sym, sel_score, blended))

    if not candidates:
        return {}

    # --- Rank-normalize each dimension to [0, 1] ----------------------------
    sel_scores = [c[1] for c in candidates]
    sharpe_vals = [c[2] for c in candidates]

    sel_norm = _rank_normalize(sel_scores)
    sharpe_norm = _rank_normalize(sharpe_vals)

    # --- Weighted composite + optional BTC penalty ---------------------------
    result: Dict[str, float] = {}
    for i, (sym, _sel, _blended) in enumerate(candidates):
        composite = w_selector * sel_norm[i] + w_sharpe * sharpe_norm[i]

        if btc_correlations and sym in btc_correlations:
            btc_corr = btc_correlations[sym]
            composite *= (1 - 0.5 * btc_corr)

        if fee_costs and sym in fee_costs:
            _median_cost = np.median(list(fee_costs.values())) if fee_costs else 1e-4
            _cost_ratio = fee_costs[sym] / max(_median_cost, 1e-6)
            composite *= max(0.3, 1.0 - 0.15 * (_cost_ratio - 1.0))

        result[sym] = round(composite, 4)

    return dict(sorted(result.items(), key=lambda x: -x[1]))
