"""
SwingSelector — Layer 1 cross-sectional selector for swing group.

Ranks ETF symbols by composite score: momentum + quality + regime.
Uses heuristic scoring initially; can be upgraded to LambdaRank
once sufficient training data is collected.

Joint evaluation with swing model:
  1. Selector ranks all candidates by cross-sectional score
  2. Top-K symbols get traded by the swing XGBoost model
  3. Combined signal = selector_rank_score × swing_model_E[r]
  This ensures we only trade symbols that are both top-ranked
  cross-sectionally AND have positive expected returns from the model.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
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

# Cross-sectional features for ranking
XS_FEATURES = [
    "ret21_zscore",         # 21-day momentum z-scored across symbols
    "ret63_zscore",         # 63-day momentum z-scored
    "vol20_zscore",         # 20-day vol z-scored (lower = better for momentum)
    "rel_strength_zscore",  # relative strength vs SPY z-scored
    "momentum_quality",     # trend consistency
]


class SwingSelector(BaseSelector):
    """Cross-sectional ETF selector for the swing trading group.

    Heuristic scoring approach (pre-LambdaRank):
      score = 0.35 × ret63_zscore + 0.25 × ret21_zscore
            + 0.20 × rel_strength_zscore + 0.10 × momentum_quality
            - 0.10 × vol20_zscore  (penalize high vol)

    Once a LambdaRank model is trained (swing_selector_lgb.txt),
    the selector automatically upgrades to model-based ranking.
    """

    UNIVERSE = SWING_UNIVERSE
    XS_FEATURES = XS_FEATURES

    def __init__(self, model_dir: str = SWING_MODEL_DIR, top_k: int = 6):
        self.model_dir = model_dir
        self.top_k = top_k
        self._model = None
        self._use_heuristic = True
        try:
            self._load_model()
            self._use_heuristic = False
        except FileNotFoundError:
            log.info("No trained LambdaRank model — using heuristic ranking")

    def _load_model(self) -> None:
        model_path = os.path.join(self.model_dir, "swing_selector_lgb.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained swing selector at {model_path}. "
                "Using heuristic ranking instead."
            )
        import lightgbm as lgb
        self._model = lgb.Booster(model_file=model_path)
        log.info("SwingSelector loaded LambdaRank model (top_k=%d, universe=%d symbols)",
                 self.top_k, len(self.UNIVERSE))

    def _compute_features(self, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute cross-sectional features for all symbols.

        Args:
            universe_data: {symbol: OHLCV DataFrame} for each symbol

        Returns:
            DataFrame with one row per symbol, columns = XS_FEATURES
        """
        records = []
        for sym, bars in universe_data.items():
            if bars is None or bars.empty or len(bars) < 63:
                continue
            close = bars["close"].astype(float)
            try:
                ret21 = float((close.iloc[-1] / close.iloc[-21]) - 1) if len(close) >= 21 else 0.0
                ret63 = float((close.iloc[-1] / close.iloc[-63]) - 1) if len(close) >= 63 else 0.0
                vol20 = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
                mq = float(close.pct_change().rolling(20).apply(
                    lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
                ).iloc[-1])
            except (IndexError, TypeError):
                continue
            records.append({
                "symbol": sym,
                "ret21": ret21,
                "ret63": ret63,
                "vol20": vol20,
                "momentum_quality": mq,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("symbol")

        # Z-score across symbols (cross-sectional standardization)
        for col in ["ret21", "ret63", "vol20"]:
            std = df[col].std()
            if std > 0:
                df[f"{col}_zscore"] = (df[col] - df[col].mean()) / std
            else:
                df[f"{col}_zscore"] = 0.0

        # Relative strength = ret63 rank percentile
        df["rel_strength_zscore"] = df["ret63_zscore"]  # proxy for now

        return df

    def rank(self, universe_data: Dict[str, pd.DataFrame]) -> SelectorOutput:
        """Rank all symbols by composite cross-sectional score."""
        features = self._compute_features(universe_data)

        if features.empty:
            return SelectorOutput(
                selected=list(self.UNIVERSE[:self.top_k]),
                scores={},
                all_ranked=list(self.UNIVERSE),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"method": "fallback_no_data"},
            )

        if self._use_heuristic:
            # Heuristic composite score
            scores = (
                0.35 * features.get("ret63_zscore", 0)
                + 0.25 * features.get("ret21_zscore", 0)
                + 0.20 * features.get("rel_strength_zscore", 0)
                + 0.10 * features.get("momentum_quality", 0)
                - 0.10 * features.get("vol20_zscore", 0)
            )
            method = "heuristic"
        else:
            # LambdaRank model prediction
            feature_cols = [c for c in features.columns if c.endswith("_zscore") or c == "momentum_quality"]
            X = features[feature_cols].values
            scores = pd.Series(self._model.predict(X), index=features.index)
            method = "lambdarank"

        # Sort descending by score
        ranked = scores.sort_values(ascending=False)
        all_ranked = ranked.index.tolist()
        selected = all_ranked[:self.top_k]

        return SelectorOutput(
            selected=selected,
            scores=ranked.to_dict(),
            all_ranked=all_ranked,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"method": method, "top_k": self.top_k},
        )
