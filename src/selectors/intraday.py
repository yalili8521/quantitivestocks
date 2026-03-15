"""
IntradaySelector — Layer 1 cross-sectional selector for intraday group.

Ranks intraday symbols by pre-market gap, RVOL, overnight momentum,
and VIX regime. Simpler than crypto/swing — small universe (6 symbols)
may use a scoring function instead of ML ranking.

TODO: Implement scoring/ranking logic.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from selectors.base import BaseSelector, SelectorOutput
from utils import INTRADAY_MODEL_DIR

log = logging.getLogger(__name__)


# Full candidate universe for intraday selection
INTRADAY_UNIVERSE = [
    "SMH", "IWM", "IGV", "QQQ", "SOXX",  # Current active
    "EWT", "SPY", "XLK",                   # Previously tested
]


class IntradaySelector(BaseSelector):
    """Cross-sectional selector for the intraday trading group."""

    UNIVERSE = INTRADAY_UNIVERSE

    def __init__(self, model_dir: str = INTRADAY_MODEL_DIR, top_k: int = 4):
        super().__init__(model_dir=model_dir, top_k=top_k)

    def _load_model(self) -> None:
        import os
        model_path = os.path.join(self.model_dir, "intraday_selector_lgb.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained intraday selector at {model_path}. "
                "Train with: python main.py train-intraday-selector"
            )
        import lightgbm as lgb
        self._model = lgb.Booster(model_file=model_path)
        log.info("IntradaySelector loaded (top_k=%d, universe=%d symbols)",
                 self.top_k, len(self.UNIVERSE))

    def rank(self, universe_data: Dict[str, pd.DataFrame]) -> SelectorOutput:
        # TODO: implement cross-sectional feature computation and ranking
        raise NotImplementedError("IntradaySelector.rank() not yet implemented")
