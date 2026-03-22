"""Position sizing — scaled contracts with profit-based growth + margin check.

仓位大小计算：
- 基础6手合约
- 每赚$7,500利润增加6手（1倍 → 2倍 → 3倍 → 4倍上限）
- 保证金检查：合约数不能超过账户权益/保证金要求
- 止盈按比例分配：TP1=1手, TP2=2手, TP3=1手, TP4=1手, 跑者=1手
- 缩放时所有止盈数量等比例增长
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from src.gold_scalper.config import GoldScalperConfig


@dataclass
class SizingResult:
    """Position sizing result with TP split breakdown."""
    total_contracts: int
    scale_mult: int           # 1x, 2x, 3x, 4x
    tp_splits: List[int]      # [tp1_qty, tp2_qty, tp3_qty, tp4_qty]
    runner_qty: int
    margin_limited: bool = False  # True if margin capped the size

    def __str__(self) -> str:
        splits = "/".join(str(s) for s in self.tp_splits)
        cap_note = " [MARGIN LIMITED]" if self.margin_limited else ""
        return (f"{self.total_contracts} contracts ({self.scale_mult}x scale) "
                f"— TPs: {splits}/R{self.runner_qty}{cap_note}")


class PositionSizer:
    """Computes trade size based on cumulative profit + margin constraints.

    仓位计算逻辑：
    1. scale_mult = min(max_scale, 1 + floor(cumulative_pnl / scale_per_profit))
    2. total = base_contracts × scale_mult
    3. margin_check: total = min(total, floor(equity / margin_per_contract))
    4. 每个TP的合约数 = 基础TP合约数 × effective_scale
    """

    def __init__(self, config: GoldScalperConfig):
        self.base = config.base_contracts
        self.scale_step = config.scale_per_profit
        self.max_scale = config.max_scale_mult
        self.margin_long = config.margin_long
        self.margin_short = config.margin_short
        self.contract_value = config.contract_value
        self.tp_base = [
            config.tp1_contracts,
            config.tp2_contracts,
            config.tp3_contracts,
            config.tp4_contracts,
        ]

    def compute(
        self,
        cumulative_pnl: float,
        equity: float = 0.0,
        price: float = 0.0,
        direction: str = "LONG",
    ) -> SizingResult:
        """Compute position size and TP splits with margin enforcement.

        Args:
            cumulative_pnl: Total profit since inception (USD).
            equity: Current account equity (for margin check). 0 = skip margin check.
            price: Current gold price (for margin calculation). 0 = skip.
            direction: "LONG" or "SHORT" (determines which margin rate to use).

        Returns:
            SizingResult with total contracts and per-TP breakdown.
        """
        # Scale multiplier: +1x per $7,500 profit, capped
        if self.scale_step <= 0 or cumulative_pnl <= 0:
            extra = 0
        else:
            extra = math.floor(cumulative_pnl / self.scale_step)

        scale_mult = min(self.max_scale, 1 + extra)
        total = self.base * scale_mult
        margin_limited = False

        # Margin check: how many contracts can equity support?
        if equity > 0 and price > 0:
            margin_rate = self.margin_long if direction == "LONG" else self.margin_short
            # Margin required per contract = price × contract_value × margin_rate
            # For MCG: $5000 × $10/point × 0.20 = $10,000 per contract
            margin_per_ct = price * self.contract_value * margin_rate
            if margin_per_ct > 0:
                max_by_margin = math.floor(equity / margin_per_ct)
                if max_by_margin < total:
                    total = max(1, max_by_margin)  # at least 1 contract
                    margin_limited = True
                    # Recalculate effective scale
                    scale_mult = max(1, total // self.base) if self.base > 0 else 1

        # Scale TP quantities proportionally
        if total >= self.base:
            effective_scale = total // self.base
            tp_splits = [qty * effective_scale for qty in self.tp_base]
            runner_qty = total - sum(tp_splits)
        else:
            # Fewer contracts than base — distribute 1 each until exhausted
            tp_splits = []
            remaining = total
            for qty in self.tp_base:
                take = min(1, remaining)
                tp_splits.append(take)
                remaining -= take
                if remaining <= 0:
                    tp_splits.extend([0] * (len(self.tp_base) - len(tp_splits)))
                    break
            runner_qty = max(0, remaining)

        return SizingResult(
            total_contracts=total,
            scale_mult=scale_mult,
            tp_splits=tp_splits,
            runner_qty=max(0, runner_qty),
            margin_limited=margin_limited,
        )
