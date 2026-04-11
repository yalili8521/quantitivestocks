"""Ensemble Predictor — blends a primary predictor with TimesFM zero-shot forecasts.

Handles three modes:
  1. Both available: weighted blend (primary gets more weight by default)
  2. Only primary: use primary alone
  3. Only TimesFM (cold-start): 100% TimesFM with damped confidence
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COST_THRESHOLD = 0.001
TARGET_RETURN = 0.02


class EnsemblePredictor:
    """Blends a primary predictor with TimesFM zero-shot forecasts.

    Dynamic weighting: starts at fixed split, transitions to inverse-MAE
    weighting after MIN_OBS realized observations.
    """

    model_type = "ensemble"

    # Dynamic weighting params
    MIN_OBS = 20            # min realized observations before dynamic weighting
    MIN_WEIGHT = 0.20       # floor for either model
    MAX_WEIGHT = 0.80       # cap for either model
    COLD_START_CAP = 0.50   # max confidence when primary is None

    def __init__(
        self,
        primary: object | None,
        timesfm_pred,
        primary_weight: float = 0.70,
    ):
        """
        Args:
            primary: Any predictor with .predict(bars_df, vix_df) → dict.
                     None for cold-start mode (TimesFM only).
            timesfm_pred: TimesFMPredictor instance.
            primary_weight: Starting weight for the primary model [0, 1].
        """
        self.primary = primary
        self.tfm = timesfm_pred
        self.symbol = getattr(timesfm_pred, "symbol", "?")
        self._primary_weight = primary_weight
        self._tfm_weight = 1.0 - primary_weight

        # Rolling error tracking for dynamic weighting
        self._primary_errors: deque = deque(maxlen=60)
        self._tfm_errors: deque = deque(maxlen=60)
        self._pending_preds: deque = deque(maxlen=50)
        # (date, primary_er, tfm_er, entry_price)

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame = None,
                seq_len: int = 20) -> dict:
        """Produce blended prediction from primary + TimesFM.

        Returns standard dict matching ml_model.Predictor.predict() output.
        """
        # --- Get individual predictions ---
        primary_pred = None
        if self.primary is not None:
            try:
                primary_pred = self.primary.predict(bars_df, vix_df, seq_len=seq_len)
                if not primary_pred.get("tradeable", False) and primary_pred.get("expected_return", 0) == 0:
                    primary_pred = None  # model returned flat/empty
            except Exception as exc:
                log.warning("[Ensemble] %s primary predict failed: %s", self.symbol, exc)
                primary_pred = None

        tfm_pred = None
        try:
            tfm_pred = self.tfm.predict(bars_df, vix_df)
        except Exception as exc:
            log.warning("[Ensemble] %s TimesFM predict failed: %s", self.symbol, exc)

        # --- Resolve pending predictions for dynamic weighting ---
        self._resolve_pending(bars_df)

        # --- Blend ---
        if primary_pred is not None and tfm_pred is not None:
            return self._blend(primary_pred, tfm_pred, bars_df)
        elif primary_pred is not None:
            return primary_pred
        elif tfm_pred is not None:
            return self._cold_start(tfm_pred)
        else:
            return self._flat_result()

    def _blend(self, primary_pred: dict, tfm_pred: dict,
               bars_df: pd.DataFrame) -> dict:
        """Weighted blend of two predictions."""
        p_er = primary_pred["expected_return"]
        t_er = tfm_pred["expected_return"]

        # Update dynamic weights if enough observations
        self._update_weights()

        w_p = self._primary_weight
        w_t = self._tfm_weight
        blended_er = w_p * p_er + w_t * t_er

        # Record for future weight adaptation
        self._record_pending(bars_df, p_er, t_er)

        # Direction from blended return
        if blended_er > COST_THRESHOLD:
            direction = "UP"
        elif blended_er < -COST_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Blend confidence too
        p_conf = primary_pred.get("confidence", 0)
        t_conf = tfm_pred.get("confidence", 0)
        confidence = w_p * p_conf + w_t * t_conf

        probability = max(0.05, min(0.95, 0.5 + blended_er * 10))

        return {
            "expected_return": round(float(blended_er), 6),
            "calibrated_return": round(float(blended_er), 6),
            "direction": direction,
            "probability": round(float(probability), 4),
            "confidence": round(float(confidence), 4),
            "meta_confidence": 1.0,
            "tradeable": abs(blended_er) > COST_THRESHOLD,
            "model_type": "ensemble",
            "ensemble_weights": {"primary": round(w_p, 3), "timesfm": round(w_t, 3)},
        }

    def _cold_start(self, tfm_pred: dict) -> dict:
        """Use TimesFM only (no primary model). Cap confidence."""
        pred = dict(tfm_pred)
        pred["confidence"] = min(pred.get("confidence", 0), self.COLD_START_CAP)
        pred["model_type"] = "timesfm_cold_start"
        return pred

    def _update_weights(self):
        """Inverse-MAE dynamic weighting (from SwingPredictor pattern)."""
        if (len(self._primary_errors) < self.MIN_OBS or
                len(self._tfm_errors) < self.MIN_OBS):
            return  # keep default weights until enough data

        p_mae = sum(self._primary_errors) / len(self._primary_errors)
        t_mae = sum(self._tfm_errors) / len(self._tfm_errors)

        if p_mae > 0 and t_mae > 0:
            # Lower error → higher weight
            w_tfm = (1 / t_mae) / (1 / t_mae + 1 / p_mae)
            w_tfm = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, w_tfm))
            self._tfm_weight = w_tfm
            self._primary_weight = 1.0 - w_tfm
            log.debug("[Ensemble] %s dynamic weights: primary=%.2f tfm=%.2f "
                      "(p_mae=%.4f t_mae=%.4f)",
                      self.symbol, self._primary_weight, self._tfm_weight,
                      p_mae, t_mae)

    def _record_pending(self, bars_df: pd.DataFrame, p_er: float, t_er: float):
        """Store prediction for future weight adaptation."""
        try:
            current_date = bars_df.index[-1]
            current_price = float(bars_df["close"].iloc[-1])
            if current_price > 0:
                self._pending_preds.append(
                    (current_date, p_er, t_er, current_price)
                )
        except Exception:
            pass

    def _resolve_pending(self, bars_df: pd.DataFrame):
        """Check if 10 days have passed for any pending predictions."""
        try:
            current_price = float(bars_df["close"].iloc[-1])
            current_date = bars_df.index[-1]
        except (IndexError, KeyError):
            return

        resolved = []
        for pp in self._pending_preds:
            pred_date, p_er, t_er, entry_px = pp
            try:
                days_elapsed = (current_date - pred_date).days
            except (TypeError, AttributeError):
                continue
            if days_elapsed >= 10 and entry_px > 0:
                realized = current_price / entry_px - 1
                self._primary_errors.append(abs(p_er - realized))
                self._tfm_errors.append(abs(t_er - realized))
                resolved.append(pp)

        for pp in resolved:
            self._pending_preds.remove(pp)

    @staticmethod
    def _flat_result() -> dict:
        return {
            "expected_return": 0.0,
            "calibrated_return": 0.0,
            "direction": "FLAT",
            "probability": 0.5,
            "confidence": 0.0,
            "meta_confidence": 1.0,
            "tradeable": False,
            "model_type": "ensemble",
        }
