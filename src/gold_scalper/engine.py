"""Gold Scalper Engine — main loop orchestrating all components.

主引擎 — 把所有组件串起来的协调器：

主循环（每10秒）：
  1. 检查交易时段（是否在NYSE窗口内）
  2. 获取5个时间框架的K线数据
  3. 计算偏向堆栈
  4. 如果有仓位 → 检查所有出场条件
  5. 如果空仓 → 检查入场信号
  6. 执行交易动作（部分平仓、全部平仓、移动止损）
  7. 发送Discord警报

支持 --dry-run 模式：只记录信号，不执行交易
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.gold_scalper.alerts import ScalperAlertEngine
from src.gold_scalper.bias_stack import BiasStack
from src.gold_scalper.broker_adapter import GoldBrokerAdapter, PaperGoldBroker
from src.gold_scalper.config import GoldScalperConfig, load_config
from src.gold_scalper.position_manager import (
    GoldPosition, PositionManager, TradeAction,
)
from src.gold_scalper.session_filter import SessionFilter
from src.gold_scalper.signals import EntrySignalGenerator
from src.gold_scalper.sizing import PositionSizer

logger = logging.getLogger(__name__)


class GoldScalperEngine:
    """Main engine — orchestrates bias, signals, sizing, exits, alerts.

    黄金剥头皮引擎：
    - 单仓位模型（同一时间最多一个仓位，与Pine Script一致）
    - 支持纸上交易（PaperGoldBroker）和未来的真实交易
    - 每日盈亏跟踪 + 熔断机制
    """

    def __init__(
        self,
        config: GoldScalperConfig,
        broker: GoldBrokerAdapter,
        alerts: ScalperAlertEngine,
        dry_run: bool = False,
    ):
        self.config = config
        self.broker = broker
        self.alerts = alerts
        self.dry_run = dry_run
        self.tz = ZoneInfo(config.timezone)

        # Components
        self.bias_stack = BiasStack(config)
        self.signal_gen = EntrySignalGenerator(config)
        self.position_mgr = PositionManager(config)
        self.sizer = PositionSizer(config)
        self.session = SessionFilter(config)

        # State
        self.position: Optional[GoldPosition] = None
        self.daily_pnl: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.daily_losses: int = 0
        self._last_day: Optional[int] = None
        self._circuit_breaker_fired: bool = False
        self._cooldown_until: Optional[datetime] = None  # no new entries until this time
        self._last_heartbeat: Optional[datetime] = None   # periodic liveness log

        # Restore position from broker state (survives process restarts)
        self._restore_position()

    def _restore_position(self) -> None:
        """Restore GoldPosition from broker's persisted state on restart.

        Without this, a process restart while in a position would silently
        abandon the remaining contracts (engine.position = None but broker
        still has open contracts).
        """
        broker_pos = self.broker.get_position(self.config.symbol)
        if broker_pos is None:
            return

        direction = broker_pos.get("direction", "LONG")
        entry_price = broker_pos.get("entry_price", 0)
        contracts = broker_pos.get("contracts", 0)
        if contracts <= 0:
            return

        # Rebuild position with proper TP ladder based on current price.
        # Compute fresh splits as if this were a new entry, then retroactively
        # mark TPs that price has already passed as hit.
        sizing = self.sizer.compute(
            self.cumulative_pnl,
            equity=self.broker.get_account_equity(),
            direction=direction,
        )
        tp_splits = sizing.tp_splits
        runner_qty = sizing.runner_qty

        self.position = self.position_mgr.create_position(
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(self.tz),
            total_contracts=contracts,
            tp_splits=tp_splits,
            runner_qty=runner_qty,
        )

        # Ratchet stop based on current price, but do NOT mark TPs as hit.
        # The normal _tick → position_mgr.update() flow will detect passed
        # TP levels and execute the partial close orders through the broker.
        # Previously this method marked TPs hit without executing partials,
        # causing the engine (fewer contracts) and broker (full position) to
        # desync — the next runner-exit check would then dump the entire
        # position as "Runner Exit - Bias Flipped".
        try:
            current_price = self.broker.get_current_price(self.config.symbol)
            pips = self.position.pips_from_entry(current_price, self.config.pip_value)

            # Only set the protective stop — let _tick handle TP partials
            sign = 1 if direction == "LONG" else -1
            if pips >= self.config.tp4_pips:
                self.position.current_stop = entry_price + sign * self.config.pips_to_price(self.config.tp3_pips)
            elif pips >= self.config.tp3_pips:
                self.position.current_stop = entry_price + sign * self.config.pips_to_price(self.config.tp2_pips)
            elif pips >= self.config.tp2_pips:
                self.position.current_stop = entry_price + sign * self.config.pips_to_price(self.config.tp1_pips)
            elif pips >= self.config.tp1_pips:
                be_offset = self.config.pips_to_price(self.config.be_offset_pips)
                self.position.current_stop = entry_price + sign * be_offset

            logger.info(
                f"[RESTORE] Recovered {direction} {contracts}ct "
                f"@ {entry_price:.2f} (pips={pips:.0f}, "
                f"stop={self.position.current_stop:.2f}) — "
                f"TPs will fire on next tick"
            )
        except Exception as e:
            logger.warning(
                f"[RESTORE] Price fetch failed ({e}), keeping hard stop "
                f"at {self.position.current_stop:.2f}"
            )

    def run(self, poll_interval: int = 10) -> None:
        """Main loop — runs until interrupted.

        主循环：
        - 交易时段内每10秒执行一次tick
        - 非交易时段等待下一个开盘
        - Ctrl+C 中断退出
        """
        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}"
            f"Gold Scalper Engine v1.0 starting — "
            f"symbol={self.config.symbol}, "
            f"base={self.config.base_contracts}ct, "
            f"max_scale={self.config.max_scale_mult}x"
        )

        try:
            while True:
                now = datetime.now(self.tz)
                self._reset_daily_if_needed(now)

                if not self.session.can_trade(now):
                    time.sleep(60 if self.session.minutes_to_session_end(now) <= 0 else 30)
                    continue

                try:
                    self._tick(now)
                except RuntimeError as e:
                    logger.warning("Tick skipped: %s", e)

                # Heartbeat every 5 minutes — proves the process is alive
                # to the watchdog (which checks log file staleness).
                if (self._last_heartbeat is None or
                        (now - self._last_heartbeat).total_seconds() >= 300):
                    pos_status = "FLAT"
                    if self.position and not self.position.is_flat:
                        p = self.position
                        pos_status = (
                            f"{p.direction} {p.remaining_contracts}ct "
                            f"@ {p.entry_price:.2f}"
                        )
                    logger.info(
                        "Heartbeat: %s | daily PnL=$%.2f | W/L=%d/%d",
                        pos_status, self.daily_pnl,
                        self.daily_wins, self.daily_losses,
                    )
                    self._last_heartbeat = now

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Engine stopped by user (Ctrl+C)")
            if self.position and not self.position.is_flat:
                logger.warning(
                    f"WARNING: Open position with "
                    f"{self.position.remaining_contracts} contracts!"
                )

    def _tick(self, now: datetime) -> None:
        """Single iteration of the main loop.

        单次tick：
        1. 获取多时间框架数据
        2. 计算偏向
        3. 有仓位 → 检查出场
        4. 空仓 → 检查入场
        """
        # Fetch multi-TF bars
        bars = self._fetch_all_timeframes()
        if not bars:
            logger.warning("Failed to fetch bars, skipping tick")
            return

        # Compute bias
        biases = self.bias_stack.compute(bars)
        direction = self.bias_stack.is_aligned(biases)

        # Log bias state
        logger.debug(f"Bias: {self.bias_stack.bias_icons(biases)} → {direction or 'MIXED'}")

        # ── EXIT CHECK (if in position) ──
        if self.position and not self.position.is_flat:
            price = self.broker.get_current_price(self.config.symbol)
            actions = self.position_mgr.update(
                self.position, price, now, direction
            )
            for action in actions:
                self._execute_action(action, biases)

            # Check if position fully closed
            if self.position.is_flat:
                self._on_position_closed(now)
            return  # Don't enter while in position

        # ── ENTRY CHECK (if flat) ──

        # Post-trade cooldown — prevents immediate re-entry after a loss/BE
        if self._cooldown_until and now < self._cooldown_until:
            return

        # Circuit breaker
        if self._circuit_breaker_fired:
            return

        if self.daily_pnl <= self.config.daily_loss_limit:
            if not self._circuit_breaker_fired:
                self._circuit_breaker_fired = True
                self.alerts.on_circuit_breaker(
                    self.daily_pnl, self.config.daily_loss_limit
                )
                logger.warning(
                    f"⛔ Circuit breaker: daily PnL ${self.daily_pnl:.2f} "
                    f"<= limit ${self.config.daily_loss_limit:.2f}"
                )
            return

        # Avoid entry near session end
        avoid, reason = self.session.should_avoid_entry(now)
        if avoid:
            logger.debug(f"Skipping entry: {reason}")
            return

        # Pre-signal filters (ATR, volume, VWAP) — all configurable, disabled by default
        filter_reason = self._apply_signal_filters(direction, bars)
        if filter_reason:
            logger.debug(f"Filtered: {filter_reason}")
            return

        # Generate signal
        signal = self.signal_gen.evaluate(direction, bars.get("5m", pd.DataFrame()))

        if not signal.is_valid:
            logger.debug(f"No signal: {signal.reason_rejected}")
            return

        # ── ENTRY ──
        sizing = self.sizer.compute(self.cumulative_pnl)
        price = self.broker.get_current_price(self.config.symbol)

        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}"
            f"ENTRY SIGNAL: {signal.direction} @ ${price:.2f} — "
            f"{sizing}"
        )

        if self.dry_run:
            return

        # Place order
        side = "BUY" if signal.direction == "LONG" else "SELL"
        order_id = self.broker.place_market_order(
            self.config.symbol, sizing.total_contracts, side
        )

        if not order_id:
            logger.error("Order placement failed!")
            return

        # Create position
        self.position = self.position_mgr.create_position(
            direction=signal.direction,
            entry_price=price,
            entry_time=now,
            total_contracts=sizing.total_contracts,
            tp_splits=sizing.tp_splits,
            runner_qty=sizing.runner_qty,
        )

        # Send entry alert
        bias_summary = self.bias_stack.bias_summary(biases)
        tp_pips = [tp[0] for tp in self.config.tp_levels]
        tp_cts = sizing.tp_splits

        self.alerts.on_entry(
            direction=signal.direction,
            symbol=self.config.symbol,
            price=price,
            contracts=sizing.total_contracts,
            scale_mult=sizing.scale_mult,
            hard_stop_pips=self.config.hard_stop_pips,
            tp_pips=tp_pips,
            tp_contracts=tp_cts,
            runner_qty=sizing.runner_qty,
            bias_summary=bias_summary,
            rsi=signal.rsi_value,
        )

    def _apply_signal_filters(
        self, direction: Optional[str], bars: Dict[str, pd.DataFrame]
    ) -> str:
        """Pre-signal filters — reject entries based on volatility/volume/VWAP.

        Returns empty string if all filters pass, or rejection reason.
        All filters are disabled by default (config fields default False).
        """
        if direction is None:
            return ""  # no bias alignment, nothing to filter

        bars_5m = bars.get("5m", pd.DataFrame())
        if bars_5m.empty or len(bars_5m) < 30:
            return ""  # insufficient data for indicators

        cfg = self.config

        # Filter 1: ATR volatility band
        if cfg.filter_atr_enabled:
            from src.signals_engine import compute_atr
            atr_series = compute_atr(
                bars_5m["high"], bars_5m["low"], bars_5m["close"], period=14
            )
            if len(atr_series) >= 20:
                atr_now = float(atr_series.iloc[-1])
                atr_ma = float(atr_series.tail(20).mean())
                if atr_ma > 0:
                    ratio = atr_now / atr_ma
                    if ratio < cfg.filter_atr_low:
                        return f"ATR ratio {ratio:.2f} < {cfg.filter_atr_low} (dead market)"
                    if ratio > cfg.filter_atr_high:
                        return f"ATR ratio {ratio:.2f} > {cfg.filter_atr_high} (too volatile)"

        # Filter 2: Volume regime
        if cfg.filter_vol_enabled:
            vol = bars_5m["volume"]
            if len(vol) >= 20:
                vol_now = float(vol.iloc[-1])
                vol_ma = float(vol.tail(20).mean())
                if vol_ma > 0 and vol_now < cfg.filter_vol_min * vol_ma:
                    return (
                        f"Volume {vol_now:.0f} < {cfg.filter_vol_min}× MA "
                        f"({vol_ma:.0f}) — thin market"
                    )

        # Filter 3: VWAP distance
        if cfg.filter_vwap_enabled:
            from src.signals_engine import compute_atr
            atr_series = compute_atr(
                bars_5m["high"], bars_5m["low"], bars_5m["close"], period=14
            )
            if len(atr_series) >= 1:
                atr_now = float(atr_series.iloc[-1])
                # Compute VWAP inline (need typical_price × volume / cumulative volume)
                tp = (bars_5m["high"] + bars_5m["low"] + bars_5m["close"]) / 3.0
                cum_tpv = (tp * bars_5m["volume"]).cumsum()
                cum_vol = bars_5m["volume"].cumsum()
                vwap = float(cum_tpv.iloc[-1] / max(cum_vol.iloc[-1], 1))
                price = float(bars_5m["close"].iloc[-1])
                if atr_now > 0:
                    dist = abs(price - vwap) / atr_now
                    if dist > cfg.filter_vwap_max_atr:
                        return (
                            f"VWAP distance {dist:.1f}× ATR > "
                            f"{cfg.filter_vwap_max_atr} (overstretched)"
                        )

        return ""  # all filters passed

    def _execute_action(self, action: TradeAction, biases) -> None:
        """Execute a trade action from the position manager.

        执行交易动作：
        - CLOSE_PARTIAL: 部分平仓（某个TP命中）
        - CLOSE_ALL: 全部平仓（止损、超时、时段关闭等）
        """
        symbol = self.config.symbol

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would execute: {action.action} "
                f"{action.contracts}ct — {action.reason}"
            )
            return

        if action.action == "CLOSE_ALL":
            self.broker.close_all(symbol, action.reason)
            # Mark engine position as flat so is_flat returns True this tick
            if self.position:
                self.position.remaining_contracts = 0

        elif action.action == "CLOSE_PARTIAL":
            self.broker.close_partial(
                symbol, action.contracts, action.reason
            )

        # Send alerts based on reason
        if action.reason.startswith("TP"):
            tp_num = int(action.reason[2]) if len(action.reason) > 2 and action.reason[2].isdigit() else 0
            stop_labels = {
                1: f"BE +{self.config.be_offset_pips:.0f} pips",
                2: f"TP1 level (+{self.config.tp1_pips:.0f} pips)",
                3: f"TP2 level (+{self.config.tp2_pips:.0f} pips)",
                4: f"TP3 level (+{self.config.tp3_pips:.0f} pips)",
            }
            self.alerts.on_tp_hit(
                tp_level=tp_num,
                symbol=symbol,
                price=action.price,
                tp_pips=action.pnl_pips,
                contracts_closed=action.contracts,
                remaining=self.position.remaining_contracts if self.position else 0,
                new_stop=action.new_stop or 0,
                new_stop_label=stop_labels.get(tp_num, ""),
            )

        elif "Bias Flipped" in action.reason:
            if self.position:
                pnl = self.position.unrealized_pnl(
                    action.price, self.config.pip_value
                )
                self.alerts.on_runner_exit(
                    symbol=symbol,
                    entry_price=self.position.entry_price,
                    exit_price=action.price,
                    pnl=pnl,
                    reason="Bias Flipped",
                )

        elif "Timeout" in action.reason:
            if self.position:
                self.alerts.on_timeout(
                    symbol=symbol,
                    entry_price=self.position.entry_price,
                    exit_price=action.price,
                    timeout_minutes=self.config.tp1_timeout_minutes,
                )

        elif "Close" in action.reason or "Weekend" in action.reason:
            self.alerts.on_session_close(action.price, action.reason)

    def _on_position_closed(self, now: datetime) -> None:
        """Called when position is fully closed — update stats + send summary.

        仓位完全关闭后：
        - 更新每日盈亏统计
        - 发送交易总结警报
        """
        if not self.position:
            return

        pos = self.position
        total_pnl = pos.realized_pnl
        duration = (now - pos.entry_time).total_seconds() / 60.0

        # Update stats
        self.daily_pnl += total_pnl
        self.cumulative_pnl += total_pnl
        self.daily_trades += 1
        if total_pnl >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        # Determine exit reason
        if pos.tp4_hit:
            exit_reason = "Runner Exit (all TPs hit)"
        elif pos.tp3_hit:
            exit_reason = "Stopped after TP3"
        elif pos.tp2_hit:
            exit_reason = "Stopped after TP2 (risk-free)"
        elif pos.tp1_hit:
            exit_reason = "Stopped after TP1"
        else:
            exit_reason = "Hard Stop / Timeout / Session Close"

        # Get exit price from broker
        exit_price = self.broker.get_current_price(self.config.symbol)

        # Send summary alert
        self.alerts.on_trade_summary(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            total_pnl=total_pnl,
            duration_minutes=duration,
            exit_reason=exit_reason,
            tp_log=pos.tp_log,
            equity=self.broker.get_account_equity(),
            daily_wins=self.daily_wins,
            daily_losses=self.daily_losses,
            daily_trades=self.daily_trades,
        )

        logger.info(
            f"Trade closed: {pos.direction} {pos.total_contracts}ct — "
            f"PnL: ${total_pnl:.2f} — {exit_reason} — "
            f"Duration: {duration:.0f}min"
        )

        # Cooldown after losing / breakeven trades to prevent immediate re-entry
        # into the same (now-invalidated) signal.  Winners get a shorter cooldown.
        from datetime import timedelta
        if total_pnl <= 0:
            # Loss or breakeven: 10-minute cooldown
            self._cooldown_until = now + timedelta(minutes=10)
            logger.info("Cooldown: 10 min after loss/BE (until %s)",
                        self._cooldown_until.strftime("%H:%M:%S"))
        else:
            # Winner: 2-minute cooldown (avoid chasing the same move)
            self._cooldown_until = now + timedelta(minutes=2)

        # Clear position
        self.position = None

    def _fetch_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """Fetch D, 4H, 1H, 15m, 5m bars.

        获取所有时间框架的K线数据：
        - 1d: 30根日线（足够计算EMA(15)）
        - 4h: 从1H重采样得到
        - 1h: 5天的1小时线
        - 15m: 3天的15分钟线
        - 5m: 2天的5分钟线（足够计算RSI(14) + 吞没形态）
        """
        symbol = self.config.symbol
        bars = {}

        lookbacks = {
            "1d": 30,
            "4h": 30,  # will be resampled from 1h
            "1h": 120,
            "15m": 200,
            "5m": 200,
        }

        for tf, lookback in lookbacks.items():
            try:
                df = self.broker.fetch_bars(symbol, tf, lookback)
                if df is not None and not df.empty:
                    bars[tf] = df
                else:
                    logger.warning("Empty bars for %s", tf)
            except Exception as e:
                logger.error("Failed to fetch %s bars: %s", tf, e)

        return bars

    def _reset_daily_if_needed(self, now: datetime) -> None:
        """Reset daily stats at session open.

        每日重置：
        - 每个交易日开始时重置每日盈亏、胜/负次数
        - 重置熔断状态
        """
        day = now.date().toordinal()
        if self._last_day != day:
            if self._last_day is not None:
                logger.info(
                    f"New trading day — yesterday: "
                    f"PnL=${self.daily_pnl:.2f}, "
                    f"W/L={self.daily_wins}/{self.daily_losses}"
                )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self._circuit_breaker_fired = False
            self._last_day = day


def main():
    """CLI entry point for gold-scalper command.

    命令行入口：
    python main.py gold-scalper              # 运行纸上交易
    python main.py gold-scalper --dry-run    # 只记录信号
    python main.py gold-scalper --config path/to/config.json
    """
    parser = argparse.ArgumentParser(
        description="Gold Multi-TF NYSE Scalper — TC baby V1.0"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to gold_scalper.json config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log signals without executing trades"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=10,
        help="Seconds between ticks (default: 10)"
    )
    parser.add_argument(
        "--initial-equity", type=float, default=5000.0,
        help="Initial paper account equity (default: $5000)"
    )
    parser.add_argument(
        "--broker", type=str, default="paper",
        choices=["paper", "cqg", "ibkr", "hybrid"],
        help="Broker backend: 'paper' (Yahoo data + sim), 'hybrid' (IBKR data + sim), 'ibkr' (IBKR live), 'cqg' (AMP demo)"
    )
    parser.add_argument(
        "--mode", type=str, default="signal",
        choices=["signal", "webhook"],
        help="Run mode: 'signal' (generate signals locally), 'webhook' (receive from TradingView)"
    )
    parser.add_argument(
        "--webhook-port", type=int, default=8000,
        help="Port for webhook server (default: 8000)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Log rotation: RotatingFileHandler (10 MB, 5 backups)
    # Separate log files for signal mode vs webhook mode
    from logging.handlers import RotatingFileHandler as _RFH
    _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _log_name = ("gold_webhook.log" if args.mode == "webhook"
                 else "gold_signal.log")
    _rfh = _RFH(os.path.join(_log_dir, _log_name),
                maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    _rfh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_rfh)

    # Load config
    config = load_config(args.config)
    logger.info(f"Config loaded: {config.symbol}, base={config.base_contracts}ct")

    # Load webhook URL from env if not in config
    webhook_url = config.discord_webhook_url
    if not webhook_url:
        webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "")

    # Initialize components
    if args.broker == "cqg":
        from src.gold_scalper.cqg_broker import CQGGoldBroker
        broker = CQGGoldBroker(config)
        logger.info("Using CQG broker (AMP Futures demo)")
    elif args.broker == "ibkr":
        from src.gold_scalper.ibkr_broker import IBKRGoldBroker
        # Webhook mode uses clientId 20, signal mode uses 10 — prevents conflicts
        _default_cid = "20" if args.mode == "webhook" else "10"
        _ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
        _ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
        _ibkr_cid = int(os.environ.get("IBKR_CLIENT_ID", _default_cid))
        # Retry IBKR connection — TWS may not be up yet after a restart
        broker = None
        for _attempt in range(1, 61):  # up to ~10 min of retries
            try:
                broker = IBKRGoldBroker(
                    config, host=_ibkr_host, port=_ibkr_port, client_id=_ibkr_cid,
                )
                break
            except Exception as _e:
                logger.warning(
                    "IBKR connect attempt %d/60 failed: %s — retrying in 10s",
                    _attempt, _e,
                )
                import time as _time
                _time.sleep(10)
        if broker is None:
            raise RuntimeError(
                f"Cannot connect to IBKR at {_ibkr_host}:{_ibkr_port} after 60 attempts"
            )
        logger.info("Using IBKR broker (Interactive Brokers)")
    elif args.broker == "hybrid":
        from src.gold_scalper.broker_adapter import HybridGoldBroker
        broker = HybridGoldBroker(
            config,
            initial_equity=args.initial_equity,
            ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
            ibkr_port=int(os.environ.get("IBKR_PORT", "7497")),
            ibkr_client_id=int(os.environ.get("IBKR_CLIENT_ID", "10")),
        )
        logger.info("Using hybrid broker (IBKR data + paper execution)")
    else:
        broker = PaperGoldBroker(config, initial_equity=args.initial_equity)
        logger.info("Using paper broker (local simulation)")
    alerts = ScalperAlertEngine(webhook_url)

    # Webhook mode: start webhook server instead of signal loop
    if args.mode == "webhook":
        from src.gold_scalper.webhook_server import start_server
        logger.info("Starting webhook mode — waiting for TradingView signals")
        start_server(
            broker=broker,
            config=config,
            host="0.0.0.0",
            port=args.webhook_port,
            webhook_url=webhook_url,
        )
        return

    # Signal mode: run normal engine loop
    engine = GoldScalperEngine(
        config=config,
        broker=broker,
        alerts=alerts,
        dry_run=args.dry_run,
    )
    engine.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
