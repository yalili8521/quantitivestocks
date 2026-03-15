"""
CryptoSelector — Layer 1 cross-sectional coin ranking.

Wraps the existing coin_selector.py (LambdaRank LightGBM) with the
BaseSelector interface. Training still uses coin_selector.main().
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from selectors.base import BaseSelector, SelectorOutput
from coin_selector import (
    CoinSelector as _LegacyCoinSelector,
    CRYPTO_UNIVERSE,
    fetch_universe_data,
)
from utils import CRYPTO_MODEL_DIR

log = logging.getLogger(__name__)


class CryptoSelector(BaseSelector):
    """Cross-sectional crypto coin selector using LambdaRank."""

    UNIVERSE = list(CRYPTO_UNIVERSE)

    def __init__(self, model_dir: str = CRYPTO_MODEL_DIR, top_k: int = 6):
        self._legacy: _LegacyCoinSelector = None  # type: ignore
        super().__init__(model_dir=model_dir, top_k=top_k)

    def _load_model(self) -> None:
        self._legacy = _LegacyCoinSelector(model_dir=self.model_dir)
        log.info("CryptoSelector loaded (top_k=%d, universe=%d coins)",
                 self.top_k, len(self.UNIVERSE))

    def rank(self, universe_data: Dict[str, pd.DataFrame]) -> SelectorOutput:
        output = self._legacy.rank(universe_data, top_k=self.top_k)
        return SelectorOutput(
            selected=output.selected,
            scores=output.scores,
            all_ranked=output.all_ranked,
            timestamp=output.timestamp,
        )

    @staticmethod
    def fetch_data(lookback_days: int = 630) -> Dict[str, pd.DataFrame]:
        """Convenience: fetch universe data for ranking."""
        return fetch_universe_data(lookback_days=lookback_days)
