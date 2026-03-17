"""
Model monitoring: rolling IC, hit rate, realized vs predicted tracking,
backtest-vs-live divergence, and auto-pause thresholds.

Persists metrics to JSON files in outputs/monitoring/ for cross-session tracking.
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """One prediction + realized outcome."""
    timestamp: str
    symbol: str
    predicted_return: float
    realized_return: Optional[float] = None
    spread_at_trade: Optional[float] = None
    slippage: Optional[float] = None
    direction: Optional[str] = None
    model_type: Optional[str] = None


@dataclass
class ModelHealth:
    """Health metrics for one symbol's model."""
    symbol: str
    model_type: str
    rolling_ic: float = 0.0               # rank correlation (pred vs realized)
    rolling_hit_rate: float = 0.0          # % correct direction
    rolling_mean_pred: float = 0.0         # mean prediction (drift check)
    rolling_mean_realized: float = 0.0     # mean realized return
    pred_realized_ratio: float = 1.0       # predicted / realized (calibration)
    n_predictions: int = 0
    n_trades: int = 0
    last_updated: str = ""
    status: str = "ok"                     # ok, warning, paused
    warning_reason: str = ""


# ---------------------------------------------------------------------------
# Thresholds for auto-warning / auto-pause
# ---------------------------------------------------------------------------

IC_WARNING_THRESHOLD = 0.0       # IC below this → warning
IC_PAUSE_THRESHOLD = -0.10       # IC below this → recommend pause
HIT_RATE_WARNING = 0.45          # hit rate below this → warning
HIT_RATE_PAUSE = 0.35            # hit rate below this → recommend pause
CALIBRATION_WARNING = 2.0        # |pred/realized| ratio above this → warning
MIN_SAMPLES_FOR_ALERT = 30       # need at least this many samples


class ModelMonitor:
    """Track and persist per-symbol model health metrics."""

    def __init__(self, output_dir: Optional[str] = None, window: int = 100):
        if output_dir is None:
            from signals_engine import PROJECT_ROOT
            output_dir = os.path.join(PROJECT_ROOT, "outputs", "monitoring")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.window = window
        # In-memory buffers: {symbol: deque of PredictionRecord}
        self._buffers: Dict[str, deque] = {}
        # Loaded health snapshots
        self._health: Dict[str, ModelHealth] = {}
        self._load_state()

    def _state_path(self) -> str:
        return os.path.join(self.output_dir, "model_health.json")

    def _predictions_path(self, symbol: str) -> str:
        return os.path.join(self.output_dir, f"predictions_{symbol}.jsonl")

    def _load_state(self) -> None:
        """Load persisted health metrics."""
        path = self._state_path()
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for sym, h in data.items():
                    self._health[sym] = ModelHealth(**h)
            except Exception as exc:
                log.warning("Failed to load model health: %s", exc)

    def _save_state(self) -> None:
        """Persist health metrics."""
        try:
            data = {sym: asdict(h) for sym, h in self._health.items()}
            with open(self._state_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning("Failed to save model health: %s", exc)

    def record_prediction(
        self,
        symbol: str,
        predicted_return: float,
        model_type: str = "unknown",
        direction: Optional[str] = None,
        spread_at_trade: Optional[float] = None,
    ) -> None:
        """Record a new prediction (realized return filled in later)."""
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self.window)

        rec = PredictionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            predicted_return=predicted_return,
            direction=direction,
            model_type=model_type,
            spread_at_trade=spread_at_trade,
        )
        self._buffers[symbol].append(rec)

        # Append to JSONL for persistence
        try:
            with open(self._predictions_path(symbol), "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
        except Exception:
            pass

    def record_realized(
        self,
        symbol: str,
        realized_return: float,
        slippage: Optional[float] = None,
    ) -> None:
        """Fill in the realized return for the most recent unfilled prediction."""
        buf = self._buffers.get(symbol)
        if not buf:
            return
        # Find most recent unfilled record
        for rec in reversed(buf):
            if rec.realized_return is None:
                rec.realized_return = realized_return
                rec.slippage = slippage
                break

    def compute_health(self, symbol: str) -> ModelHealth:
        """Compute rolling health metrics for a symbol."""
        buf = self._buffers.get(symbol, deque())
        # Only use records with realized returns
        filled = [r for r in buf if r.realized_return is not None]
        n = len(filled)

        model_type = filled[-1].model_type if filled else "unknown"
        health = ModelHealth(
            symbol=symbol,
            model_type=model_type,
            n_predictions=len(buf),
            n_trades=n,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        if n < 5:
            self._health[symbol] = health
            return health

        preds = np.array([r.predicted_return for r in filled])
        reals = np.array([r.realized_return for r in filled])

        # Rolling IC (Spearman rank correlation)
        has_pred_variation = preds.std() > 0
        has_real_variation = reals.std() > 0
        if has_pred_variation and has_real_variation:
            try:
                from scipy.stats import spearmanr

                ic, _ = spearmanr(preds, reals)
                health.rolling_ic = float(ic) if not np.isnan(ic) else 0.0
            except ImportError:
                # Fallback: Pearson
                health.rolling_ic = float(np.corrcoef(preds, reals)[0, 1])

        # Hit rate (directional accuracy)
        correct = ((preds > 0) == (reals > 0)).sum()
        health.rolling_hit_rate = correct / n

        # Mean predicted vs realized
        health.rolling_mean_pred = float(preds.mean())
        health.rolling_mean_realized = float(reals.mean())

        # Calibration ratio
        if abs(health.rolling_mean_realized) > 1e-6:
            health.pred_realized_ratio = (
                health.rolling_mean_pred / health.rolling_mean_realized
            )

        # Status assessment
        if n >= MIN_SAMPLES_FOR_ALERT:
            warnings = []
            if health.rolling_ic < IC_PAUSE_THRESHOLD:
                health.status = "paused"
                warnings.append(f"IC={health.rolling_ic:.3f} < {IC_PAUSE_THRESHOLD}")
            elif health.rolling_ic < IC_WARNING_THRESHOLD:
                health.status = "warning"
                warnings.append(f"IC={health.rolling_ic:.3f} < {IC_WARNING_THRESHOLD}")

            if health.rolling_hit_rate < HIT_RATE_PAUSE:
                health.status = "paused"
                warnings.append(f"hit_rate={health.rolling_hit_rate:.1%} < {HIT_RATE_PAUSE:.0%}")
            elif health.rolling_hit_rate < HIT_RATE_WARNING:
                if health.status != "paused":
                    health.status = "warning"
                warnings.append(f"hit_rate={health.rolling_hit_rate:.1%} < {HIT_RATE_WARNING:.0%}")

            if abs(health.pred_realized_ratio) > CALIBRATION_WARNING:
                if health.status != "paused":
                    health.status = "warning"
                warnings.append(f"calibration_ratio={health.pred_realized_ratio:.2f}")

            health.warning_reason = "; ".join(warnings)

        self._health[symbol] = health
        self._save_state()
        return health

    def should_pause_model(self, symbol: str) -> tuple:
        """Check if a model should be paused based on health metrics.

        Returns (should_pause: bool, reason: str).
        """
        health = self._health.get(symbol)
        if health is None:
            return False, "no health data"
        if health.status == "paused":
            return True, health.warning_reason
        return False, "ok"

    def should_retrain_model(self, symbol: str) -> tuple:
        """Check if a model should be retrained.

        Returns (should_retrain: bool, reason: str).
        """
        health = self._health.get(symbol)
        if health is None:
            return False, "no health data"
        if health.status in ("paused", "warning"):
            return True, health.warning_reason
        return False, "ok"

    def get_all_health(self) -> Dict[str, ModelHealth]:
        """Get health metrics for all tracked symbols."""
        return dict(self._health)

    def generate_report(self) -> str:
        """Generate a text summary of all model health."""
        lines = ["Model Health Report", "=" * 60]
        for sym, h in sorted(self._health.items()):
            status_icon = {"ok": "[OK]", "warning": "[!]", "paused": "[X]"}.get(
                h.status, "[?]"
            )
            lines.append(
                f"  {status_icon} {sym:>12} | IC={h.rolling_ic:+.3f} | "
                f"HR={h.rolling_hit_rate:.1%} | "
                f"pred/real={h.pred_realized_ratio:.2f} | "
                f"n={h.n_trades}"
            )
            if h.warning_reason:
                lines.append(f"               | {h.warning_reason}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature importance stability check
# ---------------------------------------------------------------------------

RANK_SHIFT_THRESHOLD = 10  # flag features whose rank changed by more than this


def check_feature_stability(
    current: dict,
    previous: dict,
) -> tuple:
    """Compare feature importance rankings between two training runs.

    Args:
        current:  importance dict with keys "feature_cols" and "importances".
        previous: importance dict from an earlier training run (same structure).

    Returns:
        (stable: bool, warnings: list[str])
        stable is True when Spearman rank correlation >= 0.70 and no feature
        shifted more than RANK_SHIFT_THRESHOLD positions.
    """
    warnings: List[str] = []

    cur_cols = current.get("feature_cols", [])
    prev_cols = previous.get("feature_cols", [])
    cur_imp = current.get("importances", [])
    prev_imp = previous.get("importances", [])

    if not cur_cols or not prev_cols:
        return True, ["no feature columns to compare"]

    # Build a shared feature set (intersection, preserving current order)
    shared = [f for f in cur_cols if f in prev_cols]
    if len(shared) < 3:
        return True, ["fewer than 3 shared features — skipping stability check"]

    # Rank arrays for shared features (higher importance = lower rank number)
    cur_rank = _importance_ranks(cur_cols, cur_imp)
    prev_rank = _importance_ranks(prev_cols, prev_imp)

    cur_ranks_shared = np.array([cur_rank[f] for f in shared])
    prev_ranks_shared = np.array([prev_rank[f] for f in shared])

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(cur_ranks_shared, prev_ranks_shared)
        rho = float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        # Fallback: Pearson on ranks
        if cur_ranks_shared.std() > 0 and prev_ranks_shared.std() > 0:
            rho = float(np.corrcoef(cur_ranks_shared, prev_ranks_shared)[0, 1])
        else:
            rho = 1.0

    if rho < 0.70:
        warnings.append(f"Spearman rank correlation = {rho:.3f} (< 0.70 threshold)")

    # Per-feature rank shift
    for feat in shared:
        shift = abs(cur_rank[feat] - prev_rank[feat])
        if shift > RANK_SHIFT_THRESHOLD:
            warnings.append(
                f"'{feat}' rank shifted by {shift} "
                f"(was #{prev_rank[feat]+1}, now #{cur_rank[feat]+1})"
            )

    stable = len(warnings) == 0
    return stable, warnings


def _importance_ranks(feature_cols: list, importances: list) -> Dict[str, int]:
    """Return {feature_name: rank} where rank 0 = most important."""
    arr = np.array(importances, dtype=float)
    order = np.argsort(arr)[::-1]  # descending by importance
    return {feature_cols[i]: rank for rank, i in enumerate(order)
            if i < len(feature_cols)}


# ---------------------------------------------------------------------------
# Calibration mapping (quantile-based)
# ---------------------------------------------------------------------------

class CalibrationMap:
    """Maps raw model predictions to calibrated confidence using
    historical quantile bins of predicted vs realized returns.

    Built after training; used at inference time.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_realized_mean: Optional[np.ndarray] = None
        self.bin_realized_std: Optional[np.ndarray] = None

    def fit(self, predicted: np.ndarray, realized: np.ndarray) -> None:
        """Fit calibration from training predictions and realized returns."""
        if len(predicted) < self.n_bins * 3:
            log.warning("Too few samples (%d) for calibration; skipping.", len(predicted))
            return

        # Create quantile bins
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        self.bin_edges = np.quantile(predicted, quantiles)

        # Compute realized stats per bin
        bin_indices = np.digitize(predicted, self.bin_edges[1:-1])
        self.bin_realized_mean = np.zeros(self.n_bins)
        self.bin_realized_std = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                self.bin_realized_mean[i] = realized[mask].mean()
                self.bin_realized_std[i] = realized[mask].std()

    def calibrated_return(self, raw_prediction: float) -> float:
        """Map a raw prediction to calibrated expected return."""
        if self.bin_edges is None:
            return raw_prediction  # no calibration available

        bin_idx = np.searchsorted(self.bin_edges[1:-1], raw_prediction)
        bin_idx = min(bin_idx, self.n_bins - 1)
        return float(self.bin_realized_mean[bin_idx])

    def calibrated_confidence(
        self,
        raw_prediction: float,
        target_return: float = 0.02,
    ) -> float:
        """Map a raw prediction to calibrated confidence [0, 1].

        Uses calibrated expected return instead of raw prediction.
        """
        cal_return = self.calibrated_return(raw_prediction)
        return min(1.0, abs(cal_return) / target_return)

    def save(self, path: str) -> None:
        """Save calibration map to JSON."""
        data = {
            "n_bins": self.n_bins,
            "bin_edges": self.bin_edges.tolist() if self.bin_edges is not None else None,
            "bin_realized_mean": self.bin_realized_mean.tolist() if self.bin_realized_mean is not None else None,
            "bin_realized_std": self.bin_realized_std.tolist() if self.bin_realized_std is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> CalibrationMap:
        """Load calibration map from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cm = cls(n_bins=data["n_bins"])
        if data["bin_edges"] is not None:
            cm.bin_edges = np.array(data["bin_edges"])
            cm.bin_realized_mean = np.array(data["bin_realized_mean"])
            cm.bin_realized_std = np.array(data["bin_realized_std"])
        return cm
