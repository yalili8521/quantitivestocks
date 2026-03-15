"""
SwingSelector — Layer 1 cross-sectional selector for swing group.

Ranks ETF symbols by momentum regime, relative strength, and vol regime.
Uses same LambdaRank approach as CryptoSelector.

TODO: Implement training pipeline and ranking logic.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from selectors.base import BaseSelector, SelectorOutput
from utils import SWING_MODEL_DIR

log = logging.getLogger(__name__)


# Full candidate universe for swing selection
SWING_UNIVERSE = [
    "GDX", "GLD", "SLV",           # Commodities
    "SMH", "SOXX", "IGV", "XLK",   # Tech/semis
    "QQQ", "IWM", "SPY",           # Broad market
    "IBIT",                         # Crypto ETF
    "EWT", "EEM", "MCHI",          # Emerging markets
]


class SwingSelector(BaseSelector):
    """Cross-sectional ETF selector for the swing trading group."""

    UNIVERSE = SWING_UNIVERSE

    def __init__(self, model_dir: str = SWING_MODEL_DIR, top_k: int = 6):
        super().__init__(model_dir=model_dir, top_k=top_k)

    def _load_model(self) -> None:
        import os
        model_path = os.path.join(self.model_dir, "swing_selector_lgb.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained swing selector at {model_path}. "
                "Train with: python main.py train-swing-selector"
            )
        import lightgbm as lgb
        self._model = lgb.Booster(model_file=model_path)
        log.info("SwingSelector loaded (top_k=%d, universe=%d symbols)",
                 self.top_k, len(self.UNIVERSE))

    def rank(self, universe_data: Dict[str, pd.DataFrame]) -> SelectorOutput:
        # TODO: implement cross-sectional feature computation and ranking
        raise NotImplementedError("SwingSelector.rank() not yet implemented")
