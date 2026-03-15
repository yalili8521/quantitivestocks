"""
BaseSelector — abstract base class for Layer 1 cross-sectional selectors.

All selectors share the same interface:
    1. Load a trained ranking model from disk
    2. Fetch universe data (OHLCV for all candidate symbols)
    3. Compute cross-sectional features (z-scored across symbols per date)
    4. Rank symbols and return top-K

Subclasses implement:
    - UNIVERSE: list of candidate symbols
    - XS_FEATURES: cross-sectional feature names
    - compute_features(): per-symbol feature computation
    - rank(): run the model and return SelectorOutput
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SelectorOutput:
    """Result of a selector ranking."""
    selected: List[str]                    # top-K symbols (ordered best → worst)
    scores: Dict[str, float]              # symbol → ranking score
    all_ranked: List[str]                  # full ranking (all symbols)
    timestamp: Optional[str] = None       # when the ranking was computed
    metadata: Dict = field(default_factory=dict)


class BaseSelector(ABC):
    """Abstract base class for Layer 1 cross-sectional selectors."""

    # Subclasses must define these
    UNIVERSE: List[str] = []
    XS_FEATURES: List[str] = []

    def __init__(self, model_dir: str, top_k: int = 6):
        self.model_dir = model_dir
        self.top_k = top_k
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the trained ranking model from disk."""
        ...

    @abstractmethod
    def rank(self, universe_data: Dict[str, pd.DataFrame]) -> SelectorOutput:
        """Rank all symbols and return SelectorOutput with top-K selected."""
        ...

    def select(self, universe_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Convenience: return just the top-K symbol list."""
        return self.rank(universe_data).selected
