"""TimesFM Predictor — zero-shot time-series forecasting via Google TimesFM.

Uses the pretrained TimesFM foundation model (200M-500M params) to forecast
future prices from raw close-price series. No per-symbol training needed.

Interface matches ml_model.Predictor.predict() for drop-in compatibility.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Thresholds must match ml_model.py
COST_THRESHOLD = 0.001   # 0.1%
TARGET_RETURN = 0.02     # 2%

# ── Singleton model (shared across all symbols) ──────────────────────

_TIMESFM_MODEL = None
_TIMESFM_LOCK = threading.Lock()


def _get_timesfm_model(
    context_len: int = 256,
    horizon_len: int = 10,
    backend: str = "cpu",
    repo_id: str = "google/timesfm-2.0-500m-pytorch",
):
    """Lazy-load the TimesFM model (one instance for all symbols)."""
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL

    with _TIMESFM_LOCK:
        if _TIMESFM_MODEL is not None:
            return _TIMESFM_MODEL

        import timesfm

        log.info("[TimesFM] Loading model %s (backend=%s)...", repo_id, backend)
        _TIMESFM_MODEL = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                per_core_batch_size=32,
                horizon_len=horizon_len,
                context_len=context_len,
                backend=backend,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=repo_id,
            ),
        )
        log.info("[TimesFM] Model loaded.")
        return _TIMESFM_MODEL


class TimesFMPredictor:
    """Zero-shot time-series forecaster using Google TimesFM.

    Compatible with ml_model.Predictor.predict() interface.
    No training required — uses pretrained foundation model on raw close prices.
    """

    model_type = "timesfm"

    def __init__(
        self,
        symbol: str,
        horizon: int = 10,
        context_len: int = 256,
        backend: str = "cpu",
        repo_id: str = "google/timesfm-2.0-500m-pytorch",
        damping_factor: float = 0.6,
        freq: int = 0,  # 0=daily, 1=weekly, 2=monthly
    ):
        self.symbol = symbol
        self.horizon = horizon
        self.context_len = context_len
        self.backend = backend
        self.repo_id = repo_id
        self.damping_factor = damping_factor
        self.freq = freq
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            self._model = _get_timesfm_model(
                context_len=self.context_len,
                horizon_len=self.horizon,
                backend=self.backend,
                repo_id=self.repo_id,
            )
        return self._model

    def predict(self, bars_df: pd.DataFrame, vix_df: pd.DataFrame = None,
                seq_len: int = 20) -> dict:
        """Produce prediction from raw close prices (zero-shot).

        Args:
            bars_df: DataFrame with 'close' column (and optionally 'volume').
            vix_df: Ignored in base implementation (interface compat).
            seq_len: Ignored (TimesFM uses context_len instead).

        Returns:
            Standard dict matching ml_model.Predictor.predict() output.
        """
        if "close" not in bars_df.columns:
            return self._flat_result()

        close = bars_df["close"].dropna().values.astype(np.float64)
        if len(close) < 30:
            return self._flat_result()

        # Use up to context_len recent prices
        context = close[-self.context_len:]
        current_price = context[-1]

        if current_price <= 0:
            return self._flat_result()

        try:
            model = self._ensure_model()
            forecast_out = model.forecast([context], freq=[self.freq])

            # forecast_out is (point_forecasts, quantile_forecasts) or similar
            if isinstance(forecast_out, tuple):
                point_forecast = np.array(forecast_out[0]).flatten()
            else:
                point_forecast = np.array(forecast_out).flatten()

            if len(point_forecast) == 0:
                return self._flat_result()

            # Use last forecast point as the horizon-end price
            forecast_price = point_forecast[min(self.horizon - 1, len(point_forecast) - 1)]
            expected_return = (forecast_price / current_price) - 1.0

        except Exception as exc:
            log.warning("[TimesFM] Forecast failed for %s: %s", self.symbol, exc)
            return self._flat_result()

        # Direction
        if expected_return > COST_THRESHOLD:
            direction = "UP"
        elif expected_return < -COST_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Damped confidence (conservative for zero-shot)
        confidence = min(1.0, abs(expected_return) / TARGET_RETURN) * self.damping_factor

        # Legacy probability compat
        probability = max(0.05, min(0.95, 0.5 + expected_return * 10))

        return {
            "expected_return": round(float(expected_return), 6),
            "calibrated_return": round(float(expected_return), 6),
            "direction": direction,
            "probability": round(float(probability), 4),
            "confidence": round(float(confidence), 4),
            "meta_confidence": 1.0,
            "tradeable": abs(expected_return) > COST_THRESHOLD,
            "model_type": "timesfm",
        }

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
            "model_type": "timesfm",
        }

    def batch_predict(self, symbols_bars: dict[str, pd.DataFrame],
                      freq: int = None) -> dict[str, dict]:
        """Batch-predict for multiple symbols at once (faster than one-by-one).

        Args:
            symbols_bars: {symbol: bars_df} mapping.
            freq: Override frequency for all symbols.

        Returns:
            {symbol: prediction_dict} mapping.
        """
        model = self._ensure_model()
        f = freq if freq is not None else self.freq

        contexts = []
        current_prices = []
        valid_symbols = []

        for sym, bars_df in symbols_bars.items():
            if "close" not in bars_df.columns:
                continue
            close = bars_df["close"].dropna().values.astype(np.float64)
            if len(close) < 30 or close[-1] <= 0:
                continue
            contexts.append(close[-self.context_len:])
            current_prices.append(close[-1])
            valid_symbols.append(sym)

        if not contexts:
            return {}

        try:
            forecast_out = model.forecast(contexts, freq=[f] * len(contexts))
            if isinstance(forecast_out, tuple):
                all_forecasts = np.array(forecast_out[0])
            else:
                all_forecasts = np.array(forecast_out)
        except Exception as exc:
            log.warning("[TimesFM] Batch forecast failed: %s", exc)
            return {}

        results = {}
        for i, sym in enumerate(valid_symbols):
            fc = all_forecasts[i].flatten()
            forecast_price = fc[min(self.horizon - 1, len(fc) - 1)]
            er = (forecast_price / current_prices[i]) - 1.0

            if er > COST_THRESHOLD:
                direction = "UP"
            elif er < -COST_THRESHOLD:
                direction = "DOWN"
            else:
                direction = "FLAT"

            confidence = min(1.0, abs(er) / TARGET_RETURN) * self.damping_factor
            probability = max(0.05, min(0.95, 0.5 + er * 10))

            results[sym] = {
                "expected_return": round(float(er), 6),
                "calibrated_return": round(float(er), 6),
                "direction": direction,
                "probability": round(float(probability), 4),
                "confidence": round(float(confidence), 4),
                "meta_confidence": 1.0,
                "tradeable": abs(er) > COST_THRESHOLD,
                "model_type": "timesfm",
            }

        return results
