"""
Regime Detector — 2-state HMM for adaptive exits (Phase 4)

Uses a GaussianHMM on (returns, volatility) features to classify
the current market as low-vol (regime 0) or high-vol (regime 1).

Stop multipliers:
  - Low-vol regime:  0.75 (tighter stops → capture gains)
  - High-vol regime: 1.50 (wider stops → avoid noise shakeouts)

Time-decay tightening:
  After holding 50% of max bars, linearly tighten stops so positions
  don't sit indefinitely in sideways markets.

Falls back to multiplier 1.0 if hmmlearn unavailable or fit fails.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Attempt hmmlearn import — graceful fallback if missing
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    log.info("hmmlearn not installed — regime detector disabled (multiplier=1.0)")


class RegimeDetector:
    """2-state HMM regime detector.

    States are identified post-hoc by volatility: the state with higher
    mean volatility is labeled "high-vol" (regime 1).
    """

    def __init__(
        self,
        n_states: int = 2,
        stop_mult_low: float = 0.75,
        stop_mult_high: float = 1.50,
    ):
        self.n_states = n_states
        self.stop_mult_low = stop_mult_low
        self.stop_mult_high = stop_mult_high
        self._model: Optional[object] = None
        self._high_vol_state: int = 1  # will be set after fit
        self._fitted = False

    def fit(self, returns: pd.Series, window: int = 252) -> bool:
        """Fit HMM on (returns, rolling_vol) features.

        Args:
            returns: daily return series (at least ``window`` observations)
            window: lookback for fitting (default 252 = 1 year)

        Returns True if fit succeeded, False otherwise.
        """
        if not _HMM_AVAILABLE:
            return False

        try:
            returns = returns.dropna()
            if len(returns) < max(window, 60):
                log.warning("RegimeDetector: insufficient data (%d < %d)", len(returns), window)
                return False

            recent = returns.iloc[-window:]
            vol_20d = recent.rolling(20).std().dropna()
            ret_aligned = recent.loc[vol_20d.index]

            X = np.column_stack([ret_aligned.values, vol_20d.values])
            if len(X) < 40:
                return False

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)
            self._model = model

            # Identify which state is high-vol by comparing mean vol feature
            vol_means = model.means_[:, 1]  # second feature = volatility
            self._high_vol_state = int(np.argmax(vol_means))
            self._fitted = True

            log.info(
                "RegimeDetector fitted: high-vol state=%d, vol_means=%s",
                self._high_vol_state,
                [f"{v:.4f}" for v in vol_means],
            )
            return True

        except Exception as exc:
            log.warning("RegimeDetector fit failed: %s", exc)
            self._fitted = False
            return False

    def predict_regime(self, returns: pd.Series) -> int:
        """Predict current regime (0=low-vol, 1=high-vol).

        Returns 0 if model not fitted or prediction fails.
        """
        if not self._fitted or self._model is None:
            return 0

        try:
            returns = returns.dropna()
            if len(returns) < 25:
                return 0

            vol_20d = returns.rolling(20).std().dropna()
            ret_aligned = returns.loc[vol_20d.index]
            X = np.column_stack([ret_aligned.values, vol_20d.values])

            states = self._model.predict(X)
            current_state = int(states[-1])

            # Map to 0=low-vol, 1=high-vol
            if current_state == self._high_vol_state:
                return 1
            return 0

        except Exception:
            return 0

    def get_stop_multiplier(self, regime: Optional[int] = None) -> float:
        """Return stop width multiplier for the given regime.

        Low-vol  → tighter stops (0.75)
        High-vol → wider stops (1.50)
        Not fitted → 1.0 (no adjustment)
        """
        if not self._fitted:
            return 1.0
        if regime is None:
            return 1.0
        if regime == 1:
            return self.stop_mult_high
        return self.stop_mult_low


def time_decay_tightening(
    bars_held: int,
    max_bars: int,
    start_frac: float = 0.50,
) -> float:
    """Linearly tighten stops after holding ``start_frac`` of max_bars.

    Returns a multiplier in [0.5, 1.0]:
      - Before start_frac * max_bars: 1.0 (no tightening)
      - At max_bars: 0.5 (stops halved)
      - Linear interpolation between

    This prevents positions from sitting indefinitely in choppy markets.
    """
    if max_bars <= 0:
        return 1.0

    threshold = int(start_frac * max_bars)
    if bars_held <= threshold:
        return 1.0

    remaining_frac = (bars_held - threshold) / max(max_bars - threshold, 1)
    remaining_frac = min(remaining_frac, 1.0)

    # Linear from 1.0 → 0.5
    return 1.0 - 0.5 * remaining_frac
