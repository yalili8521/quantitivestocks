"""
HRP Portfolio Sizer — Phase 3

Hierarchical Risk Parity determines relative weights within each group
(correlation-aware). Kelly determines total exposure. HRP distributes
that exposure across correlated assets more intelligently than equal-weight.

Falls back to equal weights if skfolio not installed or < 2 symbols.

Usage:
    sizer = HRPSizer(lookback=60)
    weights = sizer.compute_weights(returns_dict)
    # weights = {"SPY": 0.35, "GLD": 0.40, "QQQ": 0.25}  (sum=1.0)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Attempt skfolio import — graceful fallback
try:
    from skfolio import RiskMeasure
    from skfolio.optimization import HierarchicalRiskParity
    _SKFOLIO_AVAILABLE = True
except ImportError:
    _SKFOLIO_AVAILABLE = False
    log.info("skfolio not installed — HRP sizing disabled (equal weights)")


class HRPSizer:
    """HRP-based portfolio weight allocator.

    Wraps skfolio's HierarchicalRiskParity with CVaR risk measure.
    Falls back to equal weights when skfolio is unavailable or when
    there are fewer than ``min_symbols`` candidates.
    """

    def __init__(
        self,
        risk_measure: str = "CVAR",
        lookback: int = 60,
        min_symbols: int = 2,
    ):
        self.risk_measure_name = risk_measure
        self.lookback = lookback
        self.min_symbols = min_symbols

    def compute_weights(
        self,
        returns_dict: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """Compute HRP portfolio weights from daily returns.

        Args:
            returns_dict: {symbol: pd.Series of daily returns}
                          Each series should have at least ``lookback`` observations.

        Returns:
            {symbol: weight} summing to 1.0. Equal weights on fallback.
        """
        symbols = list(returns_dict.keys())
        n = len(symbols)

        if n < self.min_symbols:
            return self._equal_weights(symbols)

        if not _SKFOLIO_AVAILABLE:
            return self._equal_weights(symbols)

        try:
            # Build aligned returns matrix
            ret_df = pd.DataFrame(returns_dict)
            ret_df = ret_df.dropna(how="all")

            # Use only recent lookback period
            if len(ret_df) > self.lookback:
                ret_df = ret_df.iloc[-self.lookback:]

            if len(ret_df) < 20:
                log.warning("HRP: insufficient data (%d rows) — equal weights", len(ret_df))
                return self._equal_weights(symbols)

            # Fill remaining NaN with 0 (symbol may have started later)
            ret_df = ret_df.fillna(0)

            # Drop columns with zero variance (can't compute correlation)
            std = ret_df.std()
            zero_var = std[std == 0].index.tolist()
            if zero_var:
                log.info("HRP: dropping zero-variance symbols: %s", zero_var)
                ret_df = ret_df.drop(columns=zero_var)

            if ret_df.shape[1] < self.min_symbols:
                return self._equal_weights(symbols)

            # Run HRP
            risk_measure = getattr(RiskMeasure, self.risk_measure_name, RiskMeasure.CVAR)
            model = HierarchicalRiskParity(risk_measure=risk_measure)
            model.fit(ret_df)

            weights_array = model.weights_
            weight_symbols = ret_df.columns.tolist()

            weights: Dict[str, float] = {}
            for sym, w in zip(weight_symbols, weights_array):
                weights[sym] = float(w)

            # Add back zero-variance symbols with small weight
            if zero_var:
                total = sum(weights.values())
                for sym in zero_var:
                    if sym in symbols:
                        weights[sym] = 0.01
                # Re-normalize
                total = sum(weights.values())
                weights = {s: w / total for s, w in weights.items()}

            log.info("HRP weights: %s", {s: f"{w:.3f}" for s, w in weights.items()})
            return weights

        except Exception as exc:
            log.warning("HRP failed: %s — equal weights", exc)
            return self._equal_weights(symbols)

    def adjust_kelly_with_hrp(
        self,
        kelly_total: float,
        hrp_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Distribute total Kelly exposure using HRP weights.

        size[sym] = kelly_total * hrp_weight[sym]

        Args:
            kelly_total: total portfolio fraction from Kelly criterion
            hrp_weights: {symbol: weight} summing to 1.0

        Returns:
            {symbol: position_fraction}
        """
        return {sym: kelly_total * w for sym, w in hrp_weights.items()}

    @staticmethod
    def _equal_weights(symbols: list) -> Dict[str, float]:
        """Fallback: equal weight across all symbols."""
        n = len(symbols)
        if n == 0:
            return {}
        w = 1.0 / n
        return {s: w for s in symbols}
