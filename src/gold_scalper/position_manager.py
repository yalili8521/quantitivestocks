"""Position manager — tiered TP/SL ladder, runner, partial fills.

仓位管理器（策略核心）：
管理从入场到完全平仓的整个仓位生命周期：

止损移动规则：
  入场 → 硬止损(300点)
  TP1命中(+60点) → 止损移到盈亏平衡+10点
  TP2命中(+220点) → 止损移到TP1水平
  TP3命中(+400点) → 止损移到TP2水平
  TP4命中(+600点) → 止损移到TP3水平

出场优先级：
  1. 交易时段强制平仓（12:45 PT / 周五12:50 PT）
  2. 硬止损触发（300点）
  3. TP1超时（2小时内TP1没命中 → 全部平仓）
  4. TP阶梯（TP1→TP2→TP3→TP4）
  5. 跑者偏向翻转出场（TP2命中后，偏向堆栈不再对齐时平仓）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from src.gold_scalper.config import GoldScalperConfig
from src.gold_scalper.session_filter import SessionFilter


@dataclass
class GoldPosition:
    """Tracks the full state of an open gold position."""
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    total_contracts: int        # initial size (e.g. 6, 12, 18, 24)
    remaining_contracts: int    # currently open
    hard_stop: float            # initial SL price
    current_stop: float         # ratcheting SL price

    # TP tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    tp4_hit: bool = False

    # Per-TP contract amounts (scaled)
    tp1_qty: int = 0
    tp2_qty: int = 0
    tp3_qty: int = 0
    tp4_qty: int = 0
    runner_qty: int = 0

    # Peak tracking
    peak_price: float = 0.0
    peak_pips: float = 0.0

    # P&L tracking
    realized_pnl: float = 0.0  # accumulated from partial closes
    tp_log: List[str] = field(default_factory=list)

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"

    @property
    def is_flat(self) -> bool:
        return self.remaining_contracts <= 0

    @property
    def is_runner_only(self) -> bool:
        """True if only the runner contract(s) remain."""
        return self.remaining_contracts <= self.runner_qty and self.tp4_hit

    def pips_from_entry(self, current_price: float, pip_value: float) -> float:
        """Calculate pips from entry (positive = in profit direction)."""
        if self.is_long:
            return (current_price - self.entry_price) / pip_value
        else:
            return (self.entry_price - current_price) / pip_value

    def unrealized_pnl(self, current_price: float, pip_value: float) -> float:
        """Unrealized P&L in USD for remaining contracts."""
        pips = self.pips_from_entry(current_price, pip_value)
        # Each pip per contract = $1 for micro gold (pip_value=0.10, multiplier=10oz)
        # For simplicity, 1 pip = 1 pip_value move, PnL = pips × contracts × pip_value × 10
        # Actually for MCG: each $0.10 move = $1 per contract
        # So pips × $1/pip × contracts
        return pips * self.remaining_contracts


@dataclass
class TradeAction:
    """An action the engine must execute."""
    action: str                 # "CLOSE_PARTIAL", "CLOSE_ALL", "MOVE_STOP"
    contracts: int              # how many to close (0 for MOVE_STOP)
    new_stop: Optional[float]   # new SL price after this action
    reason: str                 # "TP1", "TP2", etc.
    pnl_pips: float = 0.0      # P&L in pips for this partial
    price: float = 0.0         # execution price


class PositionManager:
    """Manages position lifecycle with tiered TP/SL."""

    def __init__(self, config: GoldScalperConfig):
        self.config = config
        self.session = SessionFilter(config)

    def create_position(
        self,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        total_contracts: int,
        tp_splits: List[int],
        runner_qty: int,
    ) -> GoldPosition:
        """Create a new position with initial hard stop.

        创建新仓位，设置初始硬止损：
        - 做多：止损 = 入场价 - 硬止损距离
        - 做空：止损 = 入场价 + 硬止损距离
        """
        sl_dist = self.config.pips_to_price(self.config.hard_stop_pips)

        if direction == "LONG":
            hard_stop = entry_price - sl_dist
        else:
            hard_stop = entry_price + sl_dist

        return GoldPosition(
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            total_contracts=total_contracts,
            remaining_contracts=total_contracts,
            hard_stop=hard_stop,
            current_stop=hard_stop,
            peak_price=entry_price,
            tp1_qty=tp_splits[0] if len(tp_splits) > 0 else 0,
            tp2_qty=tp_splits[1] if len(tp_splits) > 1 else 0,
            tp3_qty=tp_splits[2] if len(tp_splits) > 2 else 0,
            tp4_qty=tp_splits[3] if len(tp_splits) > 3 else 0,
            runner_qty=runner_qty,
        )

    def update(
        self,
        pos: GoldPosition,
        current_price: float,
        now: datetime,
        bias_direction: Optional[str],
    ) -> List[TradeAction]:
        """Check all exit conditions, return ordered list of actions.

        出场检查顺序：
        1. 交易时段强制平仓
        2. 硬止损 / 移动止损触发
        3. TP1超时（2小时未命中TP1）
        4. TP阶梯命中（TP1 → TP2 → TP3 → TP4）
        5. 跑者偏向翻转出场

        Returns:
            List of TradeAction to execute (may be empty).
        """
        if pos.is_flat:
            return []

        actions: List[TradeAction] = []
        pip_val = self.config.pip_value
        pips = pos.pips_from_entry(current_price, pip_val)

        # Update peak
        if pos.is_long:
            pos.peak_price = max(pos.peak_price, current_price)
        else:
            pos.peak_price = min(pos.peak_price, current_price)
        pos.peak_pips = max(pos.peak_pips, pips)

        # --- EXIT 1: Session force-close ---
        should_close, reason = self.session.should_force_close(now)
        if should_close:
            actions.append(TradeAction(
                action="CLOSE_ALL",
                contracts=pos.remaining_contracts,
                new_stop=None,
                reason=reason,
                pnl_pips=pips,
                price=current_price,
            ))
            return actions

        # --- EXIT 2: Stop-loss hit ---
        stop_hit = False
        if pos.is_long and current_price <= pos.current_stop:
            stop_hit = True
        elif not pos.is_long and current_price >= pos.current_stop:
            stop_hit = True

        if stop_hit:
            # Determine reason based on what TPs were hit
            if pos.tp4_hit:
                reason = "Runner Stopped (SL at TP3)"
            elif pos.tp3_hit:
                reason = "Stopped after TP3 (SL at TP2)"
            elif pos.tp2_hit:
                reason = "Stopped after TP2 (risk-free)"
            elif pos.tp1_hit:
                reason = "Stopped after TP1 (at BE)"
            else:
                reason = "Hard Stop Hit"

            actions.append(TradeAction(
                action="CLOSE_ALL",
                contracts=pos.remaining_contracts,
                new_stop=None,
                reason=reason,
                pnl_pips=pips,
                price=current_price,
            ))
            return actions

        # --- EXIT 3: TP2 timeout (matches Pine Script) ---
        if not pos.tp2_hit:
            elapsed = (now - pos.entry_time).total_seconds() / 60.0
            if elapsed >= self.config.tp1_timeout_minutes:
                actions.append(TradeAction(
                    action="CLOSE_ALL",
                    contracts=pos.remaining_contracts,
                    new_stop=None,
                    reason=f"Timeout ({self.config.tp1_timeout_minutes}min) - TP2 not hit",
                    pnl_pips=pips,
                    price=current_price,
                ))
                return actions

        # --- EXIT 4: TP ladder ---
        tp_actions = self._check_tp_levels(pos, current_price, pips)
        actions.extend(tp_actions)

        # --- EXIT 5: Runner bias flip ---
        runner_action = self._check_runner_exit(pos, bias_direction, current_price, pips)
        if runner_action:
            actions.append(runner_action)

        return actions

    def _check_tp_levels(
        self,
        pos: GoldPosition,
        price: float,
        pips: float,
    ) -> List[TradeAction]:
        """Check if any TP levels have been reached.

        TP阶梯检查（每个TP命中后移动止损）：
        TP1 (+60点) → 平tp1_qty手，止损移到BE+10点
        TP2 (+220点) → 平tp2_qty手，止损移到TP1水平
        TP3 (+400点) → 平tp3_qty手，止损移到TP2水平
        TP4 (+600点) → 平tp4_qty手，止损移到TP3水平
        """
        actions = []
        cfg = self.config
        entry = pos.entry_price
        pip_val = cfg.pip_value

        # TP1
        if not pos.tp1_hit and pips >= cfg.tp1_pips:
            pos.tp1_hit = True
            # SL → BE + offset
            new_sl = self._be_stop(pos, entry, cfg.be_offset_pips)
            pos.current_stop = new_sl

            actions.append(TradeAction(
                action="CLOSE_PARTIAL",
                contracts=pos.tp1_qty,
                new_stop=new_sl,
                reason="TP1",
                pnl_pips=cfg.tp1_pips,
                price=price,
            ))
            pos.remaining_contracts -= pos.tp1_qty
            pos.tp_log.append(f"TP1: {pos.tp1_qty}ct @ +{cfg.tp1_pips:.0f} pips")

        # TP2
        if not pos.tp2_hit and pips >= cfg.tp2_pips:
            pos.tp2_hit = True
            # SL → TP1 level
            new_sl = self._tp_level_stop(pos, entry, cfg.tp1_pips)
            pos.current_stop = new_sl

            actions.append(TradeAction(
                action="CLOSE_PARTIAL",
                contracts=pos.tp2_qty,
                new_stop=new_sl,
                reason="TP2",
                pnl_pips=cfg.tp2_pips,
                price=price,
            ))
            pos.remaining_contracts -= pos.tp2_qty
            pos.tp_log.append(f"TP2: {pos.tp2_qty}ct @ +{cfg.tp2_pips:.0f} pips")

        # TP3
        if not pos.tp3_hit and pips >= cfg.tp3_pips:
            pos.tp3_hit = True
            # SL → TP2 level
            new_sl = self._tp_level_stop(pos, entry, cfg.tp2_pips)
            pos.current_stop = new_sl

            actions.append(TradeAction(
                action="CLOSE_PARTIAL",
                contracts=pos.tp3_qty,
                new_stop=new_sl,
                reason="TP3",
                pnl_pips=cfg.tp3_pips,
                price=price,
            ))
            pos.remaining_contracts -= pos.tp3_qty
            pos.tp_log.append(f"TP3: {pos.tp3_qty}ct @ +{cfg.tp3_pips:.0f} pips")

        # TP4
        if not pos.tp4_hit and pips >= cfg.tp4_pips:
            pos.tp4_hit = True
            # SL → TP2 level (gives runner more room; matches Pine Script)
            new_sl = self._tp_level_stop(pos, entry, cfg.tp2_pips)
            pos.current_stop = new_sl

            actions.append(TradeAction(
                action="CLOSE_PARTIAL",
                contracts=pos.tp4_qty,
                new_stop=new_sl,
                reason="TP4",
                pnl_pips=cfg.tp4_pips,
                price=price,
            ))
            pos.remaining_contracts -= pos.tp4_qty
            pos.tp_log.append(f"TP4: {pos.tp4_qty}ct @ +{cfg.tp4_pips:.0f} pips")

        return actions

    def _check_runner_exit(
        self,
        pos: GoldPosition,
        bias_direction: Optional[str],
        price: float,
        pips: float,
    ) -> Optional[TradeAction]:
        """Check if runner should exit on bias flip.

        跑者出场条件（TP2命中后生效）：
        - 做多仓位：偏向堆栈不再全部看涨 → 平仓
        - 做空仓位：偏向堆栈不再全部看跌 → 平仓
        """
        # Runner exit activates after TP4 hit (matches Pine Script)
        if not pos.tp4_hit:
            return None

        if pos.remaining_contracts <= 0:
            return None

        should_exit = False
        if pos.is_long and bias_direction != "LONG":
            should_exit = True
        elif not pos.is_long and bias_direction != "SHORT":
            should_exit = True

        if should_exit:
            return TradeAction(
                action="CLOSE_ALL",
                contracts=pos.remaining_contracts,
                new_stop=None,
                reason="Runner Exit - Bias Flipped",
                pnl_pips=pips,
                price=price,
            )

        return None

    def _be_stop(self, pos: GoldPosition, entry: float, offset_pips: float) -> float:
        """Calculate breakeven stop with offset.

        Pine Script convention: sl_after_tp1 = -10 means SL is placed 10 pips
        BELOW entry for longs (giving breathing room, accepting small loss risk).
        Positive offset = lock in profit above entry.  Negative = allow dip below.

        盈亏平衡止损 = 入场价 - offset（做多）/ 入场价 + offset（做空）
        """
        offset = offset_pips * self.config.pip_value
        if pos.is_long:
            return entry - offset
        else:
            return entry + offset

    def _tp_level_stop(self, pos: GoldPosition, entry: float, tp_pips: float) -> float:
        """Calculate stop at a TP level price.

        将止损移到某个TP水平的价格：
        做多：止损 = 入场价 + TP距离
        做空：止损 = 入场价 - TP距离
        """
        dist = tp_pips * self.config.pip_value
        if pos.is_long:
            return entry + dist
        else:
            return entry - dist
