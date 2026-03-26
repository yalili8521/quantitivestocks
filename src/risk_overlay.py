"""
Cross-Group Risk Overlay — Phase 5

Replaces static `cross_group_kelly_discount = 0.50` with data-driven
per-group scaling based on aggregate drawdown and cross-group correlation.

State is shared via ``outputs/risk_overlay.json`` (atomically written).
Each group reads its scaling factor at cycle start.

Scaling rules:
  - Aggregate DD > 10% → all groups scale 0.50
  - Aggregate DD > 15% → halt (factor 0.0)
  - Pair correlation > 0.80 → smaller group scales 0.70
  - Group Sharpe < 0 over rolling 20d → that group scales 0.50
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils import TRADES_DIR, OUTPUT_DIR

log = logging.getLogger(__name__)

OVERLAY_FILE = os.path.join(OUTPUT_DIR, "risk_overlay.json")
GROUPS = ["intraday", "swing", "crypto", "crypto_intraday"]


# ---------------------------------------------------------------------------
# Daily PnL per group
# ---------------------------------------------------------------------------

def compute_group_daily_pnl() -> Dict[str, pd.Series]:
    """Read daily_trades_*.csv per group, sum PnL per day.

    Returns {group: pd.Series(date → daily_pnl)}.
    """
    result: Dict[str, pd.Series] = {}
    for group in GROUPS:
        pattern = f"daily_trades_{group}"
        csv_path = os.path.join(TRADES_DIR, f"{pattern}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if "pnl" not in df.columns or "date" not in df.columns:
                continue
            daily = df.groupby("date")["pnl"].sum()
            if len(daily) > 0:
                result[group] = daily
        except Exception as exc:
            log.warning("Failed to read PnL for %s: %s", group, exc)
    return result


# ---------------------------------------------------------------------------
# Cross-group correlation matrix
# ---------------------------------------------------------------------------

def compute_cross_group_correlation(
    daily_pnl: Dict[str, pd.Series],
    window: int = 20,
) -> pd.DataFrame:
    """Rolling pairwise correlation matrix of group daily PnL.

    Returns DataFrame indexed by group, columns by group.
    Missing groups get NaN.
    """
    if len(daily_pnl) < 2:
        return pd.DataFrame()

    aligned = pd.DataFrame(daily_pnl)
    if len(aligned) < window:
        return aligned.corr()

    return aligned.iloc[-window:].corr()


# ---------------------------------------------------------------------------
# Aggregate drawdown
# ---------------------------------------------------------------------------

def compute_aggregate_drawdown(daily_pnl: Dict[str, pd.Series]) -> float:
    """Sum all groups' PnL, compute current drawdown from peak.

    Returns drawdown as a negative percentage (e.g., -0.12 = 12% DD).
    Returns 0.0 if no data.
    """
    if not daily_pnl:
        return 0.0

    aligned = pd.DataFrame(daily_pnl).fillna(0)
    total = aligned.sum(axis=1).cumsum()

    if len(total) == 0:
        return 0.0

    # Prepend 0 as initial capital reference so drawdown is measured
    # from the starting point, not just from the first day's PnL.
    total = pd.concat([pd.Series([0.0]), total], ignore_index=True)
    peak = total.cummax()
    current = total.iloc[-1]
    peak_val = peak.iloc[-1]

    if peak_val <= 0:
        # Never went positive — can't compute % drawdown from peak.
        # Return 0.0 (no scaling) since we lack a reference base.
        return 0.0

    dd = (current - peak_val) / peak_val
    return float(dd)


# ---------------------------------------------------------------------------
# Group rolling Sharpe
# ---------------------------------------------------------------------------

def compute_group_sharpe(
    daily_pnl: Dict[str, pd.Series],
    window: int = 20,
) -> Dict[str, float]:
    """Rolling Sharpe ratio per group (annualized, 252 trading days).

    Returns {group: sharpe}. Groups with insufficient data get 0.0.
    """
    result: Dict[str, float] = {}
    for group, pnl in daily_pnl.items():
        if len(pnl) < window:
            result[group] = 0.0
            continue
        recent = pnl.iloc[-window:]
        mean_ret = recent.mean()
        std_ret = recent.std()
        if std_ret > 0:
            result[group] = float(mean_ret / std_ret * np.sqrt(252))
        else:
            result[group] = 0.0
    return result


# ---------------------------------------------------------------------------
# Core: compute scaling factors
# ---------------------------------------------------------------------------

def compute_scaling_factors() -> Dict[str, float]:
    """Compute per-group scaling factors based on aggregate risk.

    Rules (applied in order):
      1. Aggregate DD > 15% → all groups halt (factor = 0.0)
      2. Aggregate DD > 10% → all groups scale 0.50
      3. Pair correlation > 0.80 → smaller group (by PnL) scales 0.70
      4. Group Sharpe < 0 → that group scales 0.50

    Returns {group: factor} where factor in [0.0, 1.0].
    Default factor = 1.0 (no scaling).
    """
    factors: Dict[str, float] = {g: 1.0 for g in GROUPS}

    daily_pnl = compute_group_daily_pnl()
    if not daily_pnl:
        log.info("Risk overlay: no daily PnL data — default factors 1.0")
        return factors

    # Rule 1 & 2: Aggregate drawdown
    agg_dd = compute_aggregate_drawdown(daily_pnl)
    if agg_dd < -0.15:
        log.warning("Risk overlay: aggregate DD %.1f%% > 15%% → HALT all groups", agg_dd * 100)
        return {g: 0.0 for g in GROUPS}
    elif agg_dd < -0.10:
        log.warning("Risk overlay: aggregate DD %.1f%% > 10%% → scale all to 0.50", agg_dd * 100)
        factors = {g: 0.50 for g in GROUPS}

    # Rule 3: Cross-group correlation penalty
    corr_matrix = compute_cross_group_correlation(daily_pnl)
    if not corr_matrix.empty:
        for i, g1 in enumerate(corr_matrix.index):
            for g2 in corr_matrix.columns[i + 1:]:
                corr = corr_matrix.loc[g1, g2]
                if pd.notna(corr) and corr > 0.80:
                    # Scale the smaller group
                    pnl_g1 = daily_pnl.get(g1, pd.Series(dtype=float)).sum()
                    pnl_g2 = daily_pnl.get(g2, pd.Series(dtype=float)).sum()
                    smaller = g2 if pnl_g1 >= pnl_g2 else g1
                    factors[smaller] = min(factors[smaller], 0.70)
                    log.info("Risk overlay: %s-%s corr=%.2f → %s scaled to %.2f",
                             g1, g2, corr, smaller, factors[smaller])

    # Rule 4: Group Sharpe < 0
    group_sharpe = compute_group_sharpe(daily_pnl)
    for group, sharpe in group_sharpe.items():
        if sharpe < 0:
            factors[group] = min(factors[group], 0.50)
            log.info("Risk overlay: %s Sharpe=%.2f < 0 → scaled to %.2f",
                     group, sharpe, factors[group])

    return factors


# ---------------------------------------------------------------------------
# File I/O (atomic write + read)
# ---------------------------------------------------------------------------

def save_scaling_factors(factors: Dict[str, float]) -> None:
    """Atomically write scaling factors to risk_overlay.json."""
    os.makedirs(os.path.dirname(OVERLAY_FILE), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "factors": factors,
    }
    # Atomic write: tmp file + rename
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(OVERLAY_FILE), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, OVERLAY_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_scaling_factor(group: str) -> float:
    """Read scaling factor for a group from risk_overlay.json.

    Returns 1.0 if file doesn't exist or group not found (no scaling).
    """
    if not os.path.exists(OVERLAY_FILE):
        return 1.0
    try:
        with open(OVERLAY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("factors", {}).get(group, 1.0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------
# Main entry point (called from watchdog or at cycle start)
# ---------------------------------------------------------------------------

def update_overlay() -> Dict[str, float]:
    """Recompute and save scaling factors. Returns the factors dict."""
    factors = compute_scaling_factors()
    save_scaling_factors(factors)
    log.info("Risk overlay updated: %s", factors)
    return factors


# ---------------------------------------------------------------------------
# Cross-group position sharing (for theme/sector cap enforcement)
# ---------------------------------------------------------------------------

POSITIONS_FILE = os.path.join(OUTPUT_DIR, "cross_group_positions.json")


def publish_positions(group: str, positions: Dict[str, dict]) -> None:
    """Write this group's current positions to shared file.

    Each group publishes its positions every cycle so other groups can
    see cross-group theme/sector exposure.
    """
    snapshot = {
        sym: {
            "qty": abs(p.get("qty", 0)),
            "current_price": p.get("current_price", 0),
            "side": p.get("side", "long"),
        }
        for sym, p in positions.items()
        if p.get("qty", 0) != 0
    }
    # Read existing, update this group, write back atomically
    existing = _read_cross_positions()
    existing[group] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "positions": snapshot,
    }
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(POSITIONS_FILE) or ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp_path, POSITIONS_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_all_group_positions(exclude_group: Optional[str] = None) -> Dict[str, dict]:
    """Read positions from ALL other groups (excludes ``exclude_group``).

    Returns a flat dict {symbol: {qty, current_price, side}} merging
    all groups except the caller's. If a symbol appears in multiple groups,
    notionals are summed.
    """
    all_data = _read_cross_positions()
    merged: Dict[str, dict] = {}
    for grp, grp_data in all_data.items():
        if grp == exclude_group:
            continue
        # Skip stale data (> 15 min old)
        updated = grp_data.get("updated_at", "")
        if updated:
            try:
                ts = datetime.fromisoformat(updated)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > 900:  # 15 minutes
                    continue
            except (ValueError, TypeError):
                continue
        for sym, pos in grp_data.get("positions", {}).items():
            if sym in merged:
                merged[sym]["qty"] += pos.get("qty", 0)
                # Keep higher price for conservative notional
                merged[sym]["current_price"] = max(
                    merged[sym]["current_price"], pos.get("current_price", 0)
                )
            else:
                merged[sym] = dict(pos)
    return merged


def _read_cross_positions() -> dict:
    """Read the shared cross-group positions file."""
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
