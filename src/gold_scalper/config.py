"""GoldScalperConfig — all strategy parameters in one place.

配置说明：
- 所有止盈/止损用点数(pips)表示，1 pip = $0.10（黄金）
- 交易时段用太平洋时间(PT)
- 仓位缩放：每赚$7,500利润加一倍基础合约数，最高4倍
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass(frozen=True)
class GoldScalperConfig:
    """Frozen config — all TC baby V1.0 parameters."""

    # ── Instrument ──
    symbol: str = "GC=F"           # Yahoo Finance symbol for gold futures
    pip_value: float = 0.10        # USD per pip (gold: $0.10 = 1 pip)

    # ── Multi-TF Bias ──
    bias_ema_periods: Dict[str, int] = field(default_factory=lambda: {
        "1d": 15, "4h": 15, "1h": 15, "15m": 15, "5m": 5
    })

    # ── Entry Filters ──
    rsi_period: int = 14
    rsi_long_threshold: float = 55.0   # RSI must be > this for LONG
    rsi_short_threshold: float = 35.0  # RSI must be < this for SHORT
    require_engulfing: bool = True     # require engulfing candle confirmation

    # ── Session (Pacific Time) ──
    # COMEX/Globex gold futures: Sun 3:00 PM - Fri 2:00 PM PT
    # Daily maintenance halt: 2:00 PM - 3:00 PM PT (Mon-Thu)
    session_start: str = "15:00"   # Globex open (3:00 PM PT)
    session_end: str = "14:00"     # Globex close (2:00 PM PT next day)
    daily_close: str = "13:45"     # force-close before maintenance halt
    friday_close: str = "13:45"    # force-close Friday before weekly close
    timezone: str = "America/Los_Angeles"

    # ── Dead Zone Hours (PT) ──
    # Gold futures halt 2:00-3:00 PM PT daily; closed Sat + Sun until 3 PM
    dead_zone_ranges: tuple = (
        ("14:00", "14:59", "Mon-Thu"),    # daily maintenance halt
        ("13:45", "23:59", "Fri"),        # Friday close through weekend
        ("00:00", "23:59", "Sat"),        # all Saturday
        ("00:00", "14:59", "Sun"),        # Sunday until 3:00 PM PT
    )

    # ── Margin ──
    margin_long: float = 0.20         # margin requirement for longs (0.20 = 20%)
    margin_short: float = 0.20        # margin requirement for shorts
    contract_value: float = 10.0      # USD value per $1 move per contract (MCG: 10 oz × $1 = $10)

    # ── Position Sizing ──
    base_contracts: int = 6
    scale_per_profit: float = 7500.0   # add base_contracts per this $ profit
    max_scale_mult: int = 4            # cap multiplier (max 24 contracts at 6 base)

    # ── Take-Profit Levels (pips) ──
    tp1_pips: float = 60.0
    tp1_contracts: int = 1    # contracts to close at TP1 (per base_contracts=6)
    tp2_pips: float = 220.0
    tp2_contracts: int = 2
    tp3_pips: float = 400.0
    tp3_contracts: int = 1
    tp4_pips: float = 600.0
    tp4_contracts: int = 1
    # Runner = base_contracts - tp1 - tp2 - tp3 - tp4 = 1

    # ── Stop-Loss ──
    hard_stop_pips: float = 300.0      # initial hard stop
    be_offset_pips: float = 10.0       # SL moves to entry + this after TP trigger

    # ── Timeout ──
    tp1_timeout_minutes: int = 120     # close if TP1 not hit within 2 hours

    # ── Circuit Breaker ──
    daily_loss_limit: float = -1000.0  # stop new entries after this daily loss (USD)

    # ── Alerts ──
    discord_webhook_url: str = ""

    # ── Data Provider ──
    data_provider: str = "yahoo"       # "yahoo" or "alpaca"

    # ── Signal Filters (all disabled by default — no behavior change) ──
    filter_atr_enabled: bool = False    # ATR volatility band filter
    filter_atr_low: float = 0.7        # skip if ATR < this × 20-bar ATR MA
    filter_atr_high: float = 2.0       # skip if ATR > this × 20-bar ATR MA
    filter_vol_enabled: bool = False    # volume regime filter
    filter_vol_min: float = 0.5        # skip if volume < this × 20-bar vol MA
    filter_vwap_enabled: bool = False   # VWAP distance filter
    filter_vwap_max_atr: float = 1.5   # skip if |price - VWAP| > this × ATR

    # ── Derived ──
    @property
    def runner_contracts(self) -> int:
        """Contracts left as runner (per base unit)."""
        return self.base_contracts - (
            self.tp1_contracts + self.tp2_contracts +
            self.tp3_contracts + self.tp4_contracts
        )

    @property
    def tp_levels(self) -> list:
        """List of (pips, base_contracts) for each TP level."""
        return [
            (self.tp1_pips, self.tp1_contracts),
            (self.tp2_pips, self.tp2_contracts),
            (self.tp3_pips, self.tp3_contracts),
            (self.tp4_pips, self.tp4_contracts),
        ]

    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price distance."""
        return pips * self.pip_value


def load_config(path: str | None = None) -> GoldScalperConfig:
    """Load config from JSON file, falling back to defaults.

    参数优先级：JSON文件 > 默认值
    """
    if path is None:
        # Default path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(project_root, "config", "gold_scalper.json")

    if not os.path.exists(path):
        return GoldScalperConfig()

    with open(path, "r") as f:
        raw = json.load(f)

    # Remove comments
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Convert bias_ema_periods from JSON (always dict)
    if "bias_ema_periods" in raw:
        raw["bias_ema_periods"] = dict(raw["bias_ema_periods"])

    # Convert dead_zone_ranges from JSON list to tuple of tuples
    if "dead_zone_ranges" in raw:
        raw["dead_zone_ranges"] = tuple(
            tuple(r) for r in raw["dead_zone_ranges"]
        )

    return GoldScalperConfig(**raw)


def save_config(config: GoldScalperConfig, path: str | None = None) -> None:
    """Save config to JSON file."""
    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(project_root, "config", "gold_scalper.json")

    data = asdict(config)
    # Convert tuples to lists for JSON
    if "dead_zone_ranges" in data:
        data["dead_zone_ranges"] = [list(r) for r in data["dead_zone_ranges"]]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
