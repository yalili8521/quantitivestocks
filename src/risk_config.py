"""
Centralized risk configuration and horizon validation.

All risk parameters, exposure limits, and model-horizon constraints live here.
Both backtester and paper_trader import from this module to stay aligned.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Horizon definitions — enforce model/mode separation
# ---------------------------------------------------------------------------

VALID_HORIZONS = {
    "10d":  {"mode": "daily",    "models": ["lstm", "xgb_swing", "tft_swing"]},
    "1d":   {"mode": "intraday", "models": ["lgb_intraday"]},
}

# Model type → allowed horizons
MODEL_HORIZON_MAP = {
    "lstm":          "10d",
    "xgb_swing":     "10d",
    "tft_swing":     "10d",
    "lgb_intraday":  "1d",
}


def validate_model_mode(model_type: str, mode: str) -> None:
    """Fail fast if model type doesn't match trading mode.

    Raises ValueError if e.g. LSTM is used with intraday mode or
    LightGBM intraday model is used with daily/swing mode.
    """
    expected_horizon = MODEL_HORIZON_MAP.get(model_type)
    if expected_horizon is None:
        return  # unknown model type, skip validation

    allowed_mode = VALID_HORIZONS[expected_horizon]["mode"]
    if mode != allowed_mode:
        raise ValueError(
            f"Model type '{model_type}' (horizon={expected_horizon}) cannot be used "
            f"with mode='{mode}'. Expected mode='{allowed_mode}'. "
            f"LSTM/XGBoost swing models are daily-only; LightGBM is intraday-only."
        )


# ---------------------------------------------------------------------------
# Sector / theme classification for exposure caps
# ---------------------------------------------------------------------------

SYMBOL_SECTOR: Dict[str, str] = {
    # Tech / semiconductors
    "SMH":  "tech", "SOXX": "tech", "IGV":  "tech", "QQQ":  "tech", "XLK": "tech",
    # Commodities
    "GDX":  "commodities", "GLD":  "commodities", "SLV":  "commodities",
    # Emerging / international
    "EWT":  "emerging", "EEM":  "emerging", "MCHI": "emerging",
    # Broad market
    "SPY":  "broad", "IWM":  "broad",
    # Crypto
    "IBIT": "crypto",
    "BTC/USD": "crypto", "ETH/USD": "crypto", "SOL/USD": "crypto",
    "AVAX/USD": "crypto", "LINK/USD": "crypto", "DOGE/USD": "crypto",
    "DOT/USD": "crypto", "SUSHI/USD": "crypto", "ADA/USD": "crypto",
    "CRV/USD": "crypto", "AAVE/USD": "crypto", "RENDER/USD": "crypto",
    # Extended crypto universe (Layer 1 selector candidates)
    "NEAR/USD": "crypto", "UNI/USD": "crypto", "LTC/USD": "crypto",
    "ARB/USD": "crypto", "OP/USD": "crypto", "FIL/USD": "crypto",
    "APT/USD": "crypto", "INJ/USD": "crypto",
}

# Crypto beta mapping (relative to BTC)
CRYPTO_BTC_BETA: Dict[str, float] = {
    "BTC/USD": 1.0, "ETH/USD": 1.3, "SOL/USD": 1.8,
    "AVAX/USD": 2.0, "LINK/USD": 1.5, "DOGE/USD": 2.2,
    "DOT/USD": 1.6, "SUSHI/USD": 2.0, "ADA/USD": 1.4,
    "CRV/USD": 2.0, "AAVE/USD": 1.5, "RENDER/USD": 2.0,
    "IBIT": 1.0,
    # Extended crypto universe (Layer 1 selector candidates)
    "NEAR/USD": 1.7, "UNI/USD": 1.8, "LTC/USD": 1.0,
    "ARB/USD": 2.0, "OP/USD": 2.0, "FIL/USD": 1.8,
    "APT/USD": 1.9, "INJ/USD": 2.2,
}


# ---------------------------------------------------------------------------
# Risk parameter dataclass — one source of truth
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Centralized risk parameters for a trading group."""

    # --- Position sizing ---
    position_pct: float = 0.50          # base capital fraction (was 0.90)
    max_position_pct: float = 0.15      # hard cap per-position (% of equity)
    max_sector_pct: float = 0.40        # max exposure per sector
    max_total_exposure: float = 0.80    # max total invested (% of equity)
    max_positions: int = 8              # max simultaneous positions

    # --- Cost thresholds ---
    cost_threshold: float = 0.001       # min |E[r]| to trade
    target_return: float = 0.02         # E[r] for full-size position
    cost_safety_margin: float = 1.5     # cost_threshold must be >= estimated_cost * this

    # --- Stop loss ---
    disaster_stop_atr_mult: float = 3.0
    disaster_stop_max_pct: float = 0.20
    profit_lock_atr_mult: float = 2.0
    profit_lock_trail_atr_mult: float = 1.5
    max_underwater_days: int = 90

    # --- Cooldowns ---
    loss_cooldown_hours: float = 4.0
    max_loss_cooldown_hours: float = 4.0
    post_loss_size_mult: float = 0.5
    post_loss_size_hours: float = 4.0
    same_dir_confidence_mult: float = 1.5

    # --- Trend / regime ---
    trend_sma_period: int = 50
    vix_threshold: float = 30.0
    regime_cooldown_days: int = 7
    regime_min_trades: int = 20
    regime_min_winrate: float = 0.50

    # --- Kelly ---
    kelly_cap: float = 0.25            # max Kelly fraction
    kelly_min_trades: int = 20         # min trades before Kelly kicks in
    kelly_window: int = 60             # rolling window for Kelly

    # --- Stress test multipliers ---
    stress_cost_mult: float = 1.0      # 1.0 = normal, 2.0 = stress
    stress_spread_mult: float = 1.0


# Pre-built configs per group (v3 — OOS-validated sizing)
SWING_RISK = RiskConfig(
    position_pct=0.45,
    max_position_pct=0.12,
    max_sector_pct=0.35,
    max_total_exposure=0.75,
    max_positions=6,
    cost_threshold=0.002,
    target_return=0.02,
    kelly_cap=0.25,
    loss_cooldown_hours=4.0,
    max_loss_cooldown_hours=4.0,
)

INTRADAY_RISK = RiskConfig(
    position_pct=0.35,
    max_position_pct=0.15,
    max_sector_pct=0.45,
    max_total_exposure=0.65,
    max_positions=5,
    cost_threshold=0.0015,
    target_return=0.015,
    kelly_cap=0.20,
    loss_cooldown_hours=0.5,
    max_loss_cooldown_hours=2.0,
    max_underwater_days=1,  # intraday exits EOD
)

CRYPTO_RISK = RiskConfig(
    position_pct=0.15,
    max_position_pct=0.05,
    max_sector_pct=0.15,
    max_total_exposure=0.15,
    max_positions=6,              # up from 3 — selector picks top-6
    cost_threshold=0.005,
    target_return=0.04,
    disaster_stop_atr_mult=4.0,   # wider stops for crypto vol
    disaster_stop_max_pct=0.20,
    kelly_cap=0.10,               # very conservative Kelly for crypto
    loss_cooldown_hours=6.0,
    max_loss_cooldown_hours=6.0,
)

CRYPTO_INTRADAY_RISK = RiskConfig(
    position_pct=0.12,                # smaller size for fast turnover
    max_position_pct=0.04,
    max_sector_pct=0.12,
    max_total_exposure=0.12,
    max_positions=6,                  # selector picks top-6
    cost_threshold=0.003,             # 30bps (matches model's COST_THRESHOLD)
    target_return=0.01,               # 1% for full size (1-hour horizon)
    disaster_stop_atr_mult=3.0,
    disaster_stop_max_pct=0.02,       # 2% hard stop (tight for intraday)
    kelly_cap=0.10,
    loss_cooldown_hours=0.5,          # 30min cooldown (fast turnover)
    max_loss_cooldown_hours=1.0,
    post_loss_size_mult=0.5,
    post_loss_size_hours=1.0,
    max_underwater_days=0,            # max-hold handled separately (4h)
)

GROUP_RISK_CONFIGS: Dict[str, RiskConfig] = {
    "swing": SWING_RISK,
    "intraday": INTRADAY_RISK,
    "crypto": CRYPTO_RISK,
    "crypto_intraday": CRYPTO_INTRADAY_RISK,
}


def get_risk_config(group: Optional[str] = None) -> RiskConfig:
    """Get risk config for a group, with fallback to swing defaults."""
    if group and group in GROUP_RISK_CONFIGS:
        return GROUP_RISK_CONFIGS[group]
    return SWING_RISK


# ---------------------------------------------------------------------------
# Exposure calculation helpers
# ---------------------------------------------------------------------------

def compute_sector_exposure(
    positions: Dict[str, dict],
    equity: float,
) -> Dict[str, float]:
    """Compute current exposure per sector as fraction of equity.

    positions: {symbol: {"qty": float, "current_price": float, "side": str}}
    """
    sector_exposure: Dict[str, float] = {}
    for sym, pos in positions.items():
        sector = SYMBOL_SECTOR.get(sym, "other")
        notional = pos["qty"] * pos["current_price"]
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + notional
    return {k: v / equity for k, v in sector_exposure.items()} if equity > 0 else {}


def compute_btc_beta_exposure(
    positions: Dict[str, dict],
    equity: float,
) -> float:
    """Compute total BTC-beta-weighted exposure for crypto positions."""
    total_beta = 0.0
    for sym, pos in positions.items():
        beta = CRYPTO_BTC_BETA.get(sym, 0.0)
        if beta > 0:
            notional = pos["qty"] * pos["current_price"]
            total_beta += beta * notional / equity if equity > 0 else 0
    return total_beta


def check_position_allowed(
    symbol: str,
    proposed_notional: float,
    equity: float,
    positions: Dict[str, dict],
    risk: RiskConfig,
) -> tuple:
    """Check if a new position passes all portfolio constraints.

    Returns (allowed: bool, reason: str).
    """
    if equity <= 0:
        return False, "zero equity"

    # 1. Per-position cap
    position_pct = proposed_notional / equity
    if position_pct > risk.max_position_pct:
        return False, (f"position size {position_pct:.1%} > max {risk.max_position_pct:.1%}")

    # 2. Total exposure cap
    current_exposure = sum(
        p["qty"] * p["current_price"] for p in positions.values()
    ) / equity
    new_exposure = current_exposure + position_pct
    if new_exposure > risk.max_total_exposure:
        return False, (f"total exposure {new_exposure:.1%} > max {risk.max_total_exposure:.1%}")

    # 3. Max positions
    active_positions = sum(1 for p in positions.values() if p.get("qty", 0) > 0)
    if active_positions >= risk.max_positions:
        return False, f"max positions reached ({active_positions} >= {risk.max_positions})"

    # 4. Sector cap
    sector = SYMBOL_SECTOR.get(symbol, "other")
    sector_exp = compute_sector_exposure(positions, equity)
    current_sector = sector_exp.get(sector, 0.0)
    if current_sector + position_pct > risk.max_sector_pct:
        return False, (f"sector '{sector}' exposure {current_sector + position_pct:.1%} "
                       f"> max {risk.max_sector_pct:.1%}")

    # 5. BTC beta cap (crypto only)
    if CRYPTO_BTC_BETA.get(symbol, 0) > 0:
        current_beta = compute_btc_beta_exposure(positions, equity)
        new_beta = current_beta + CRYPTO_BTC_BETA[symbol] * position_pct
        max_btc_beta = 2.0  # max 2x BTC-equivalent exposure
        if new_beta > max_btc_beta:
            return False, (f"BTC beta exposure {new_beta:.2f} > max {max_btc_beta:.1f}")

    return True, "ok"


# ---------------------------------------------------------------------------
# Per-symbol caps (OOS Sharpe-tiered) — loaded from config/trading.json
# ---------------------------------------------------------------------------

# OOS Sharpe registry — updated after each retrain+backtest cycle
_SYMBOL_OOS_SHARPE_FALLBACK: Dict[str, float] = {
    # Swing ETFs (OOS 2024-01-01 → present)
    "GLD": 1.55, "IBIT": 2.20, "SLV": 1.11,
    "SMH": 0.72, "QQQ": 0.65, "GDX": 0.59,
    "IGV": 0.54, "XLK": 0.19,
    # Intraday ETFs (OOS ~90 days)
    "SMH_intraday": 1.0, "SOXX_intraday": 0.8,
    "IWM_intraday": 0.7, "QQQ_intraday": 0.9,
    "IGV_intraday": 0.6,
}


def _load_oos_sharpe_from_config() -> Dict[str, float]:
    """Load OOS Sharpe from config/trading.json, merged with ETF fallbacks."""
    result = dict(_SYMBOL_OOS_SHARPE_FALLBACK)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "trading.json",
    )
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        registry = cfg.get("oos_sharpe_registry", {})
        result.update(registry)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return result


SYMBOL_OOS_SHARPE: Dict[str, float] = _load_oos_sharpe_from_config()

# Default per-symbol caps by OOS Sharpe tier
# These are overridden by config/trading.json symbol_caps if present
_DEFAULT_SYMBOL_CAPS: Dict[str, float] = {
    # Swing
    "GLD": 0.12, "IBIT": 0.10, "SLV": 0.10,
    "SMH": 0.07, "QQQ": 0.07, "GDX": 0.05,
    "IGV": 0.03, "XLK": 0.00,
    # Crypto
    "CRV-USD": 0.05, "ADA-USD": 0.03, "AVAX-USD": 0.03, "LINK-USD": 0.03,
    "DOT-USD": 0.00, "ETH-USD": 0.00, "BTC-USD": 0.00, "SOL-USD": 0.00,
    "DOGE-USD": 0.00, "SUSHI-USD": 0.00, "AAVE-USD": 0.00, "RENDER-USD": 0.00,
    # Intraday
    "SMH_intraday": 0.15, "SOXX_intraday": 0.12, "IWM_intraday": 0.12,
    "QQQ_intraday": 0.12, "IGV_intraday": 0.09, "EWT_intraday": 0.00,
}

# Theme caps across all sleeves
THEME_CAPS: Dict[str, float] = {
    "tech": 0.30, "commodities": 0.25, "crypto": 0.15,
    "emerging": 0.10, "broad": 0.20,
}


def _load_symbol_caps_from_config() -> Dict[str, Dict[str, float]]:
    """Load per-symbol caps from config/trading.json if available."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "trading.json",
    )
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return cfg.get("symbol_caps", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_symbol_cap(symbol: str, group: str) -> float:
    """Get max % equity for a specific symbol in a group.

    Checks config/trading.json first, falls back to _DEFAULT_SYMBOL_CAPS.
    Returns 1.0 (uncapped) if symbol not found in either.
    """
    caps = _load_symbol_caps_from_config()
    group_caps = caps.get(group, {})

    # Normalize crypto symbol format: BTC/USD → BTC-USD for lookup
    lookup_sym = symbol.replace("/", "-")

    if lookup_sym in group_caps:
        return group_caps[lookup_sym]

    # Fallback to defaults
    if group == "intraday":
        return _DEFAULT_SYMBOL_CAPS.get(f"{symbol}_intraday", 0.15)
    return _DEFAULT_SYMBOL_CAPS.get(lookup_sym, 0.15)


def is_symbol_disabled(symbol: str, group: str) -> bool:
    """Return True if symbol cap is 0 (disabled by OOS performance)."""
    return get_symbol_cap(symbol, group) <= 0.0


def get_confidence_multiplier(symbol: str) -> float:
    """Map OOS Sharpe to a confidence multiplier (scales position size).

    Sharpe >= 1.5  → 1.0  (full confidence)
    0.7 <= S < 1.5 → 0.75
    0.3 <= S < 0.7 → 0.50
    0.0 <= S < 0.3 → 0.25 (probe only)
    S < 0.0        → 0.00 (disabled)
    """
    lookup = symbol.replace("/", "-")
    sharpe = SYMBOL_OOS_SHARPE.get(lookup)
    if sharpe is None:
        return 0.50  # unknown symbol → conservative default

    if sharpe >= 1.5:
        return 1.0
    elif sharpe >= 0.7:
        return 0.75
    elif sharpe >= 0.3:
        return 0.50
    elif sharpe >= 0.0:
        return 0.25
    else:
        return 0.0


def compute_theme_exposure(
    positions: Dict[str, dict],
    equity: float,
) -> Dict[str, float]:
    """Compute current exposure per theme as fraction of equity."""
    theme_exp: Dict[str, float] = {}
    for sym, pos in positions.items():
        theme = SYMBOL_SECTOR.get(sym, "other")
        notional = abs(pos.get("qty", 0) * pos.get("current_price", 0))
        theme_exp[theme] = theme_exp.get(theme, 0.0) + notional
    return {k: v / equity for k, v in theme_exp.items()} if equity > 0 else {}


def check_theme_cap(
    symbol: str,
    proposed_notional: float,
    equity: float,
    positions: Dict[str, dict],
) -> tuple:
    """Check if adding a position would breach theme caps.

    Returns (allowed: bool, reason: str).
    """
    if equity <= 0:
        return False, "zero equity"

    theme = SYMBOL_SECTOR.get(symbol, "other")
    cap = THEME_CAPS.get(theme)
    if cap is None:
        return True, "ok"

    theme_exp = compute_theme_exposure(positions, equity)
    current = theme_exp.get(theme, 0.0)
    proposed_pct = proposed_notional / equity
    if current + proposed_pct > cap:
        return False, (f"theme '{theme}' exposure {current + proposed_pct:.1%} "
                       f"> cap {cap:.1%}")
    return True, "ok"


# ---------------------------------------------------------------------------
# Auto de-risking: rolling performance monitor
# ---------------------------------------------------------------------------

@dataclass
class DeRiskState:
    """Tracks rolling performance for auto de-risking decisions."""
    returns: List[float] = field(default_factory=list)
    peak_equity: float = 0.0
    current_equity: float = 0.0
    halved: bool = False
    disabled: bool = False
    disable_reason: str = ""
    trades_since_disable: int = 0

    def record_trade(self, pnl_pct: float) -> None:
        self.returns.append(pnl_pct)

    def rolling_sharpe(self, window: int = 50) -> Optional[float]:
        recent = self.returns[-window:]
        if len(recent) < 10:
            return None
        mean_r = sum(recent) / len(recent)
        var = sum((r - mean_r) ** 2 for r in recent) / len(recent)
        std = var ** 0.5
        if std < 1e-8:
            return 0.0
        return mean_r / std

    def rolling_winrate(self, window: int = 50) -> Optional[float]:
        recent = self.returns[-window:]
        if len(recent) < 10:
            return None
        return sum(1 for r in recent if r > 0) / len(recent)

    def drawdown_from_peak(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity


def evaluate_derisk(
    state: DeRiskState,
    window: int = 50,
    sharpe_warning: float = 0.0,
    sharpe_disable: float = -0.30,
    dd_halfsize: float = 0.10,
    dd_disable: float = 0.20,
    winrate_warning: float = 0.40,
) -> tuple:
    """Evaluate whether to de-risk a symbol based on rolling performance.

    Returns (action: str, reason: str) where action is one of:
      "ok"       — no action needed
      "warning"  — log warning but keep trading
      "halfsize" — cut position size by 50%
      "disable"  — stop trading this symbol
    """
    if state.disabled:
        return "disable", state.disable_reason

    # Check drawdown
    dd = state.drawdown_from_peak()
    if dd >= dd_disable:
        state.disabled = True
        state.disable_reason = f"drawdown {dd:.1%} >= {dd_disable:.1%}"
        return "disable", state.disable_reason
    if dd >= dd_halfsize:
        state.halved = True
        return "halfsize", f"drawdown {dd:.1%} >= {dd_halfsize:.1%}"

    # Check rolling Sharpe
    rs = state.rolling_sharpe(window)
    if rs is not None:
        if rs <= sharpe_disable:
            state.disabled = True
            state.disable_reason = f"rolling Sharpe {rs:.2f} <= {sharpe_disable}"
            return "disable", state.disable_reason
        if rs <= sharpe_warning:
            state.halved = True
            return "halfsize", f"rolling Sharpe {rs:.2f} <= {sharpe_warning}"

    # Check win rate
    wr = state.rolling_winrate(window)
    if wr is not None and wr < winrate_warning:
        return "warning", f"rolling win rate {wr:.1%} < {winrate_warning:.1%}"

    # If previously halved but now recovered, restore
    if state.halved and rs is not None and rs > 0.20:
        state.halved = False

    return "ok", ""
