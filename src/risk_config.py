"""
Centralized risk configuration and horizon validation.

All risk parameters, exposure limits, and model-horizon constraints live here.
Both backtester and paper_trader import from this module to stay aligned.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Horizon definitions — enforce model/mode separation
# ---------------------------------------------------------------------------

VALID_HORIZONS = {
    "10d":  {"mode": "daily",    "models": ["lstm", "xgb_swing", "tft_swing"]},
    "1d":   {"mode": "intraday", "models": ["lgb_intraday"]},
    "1h":   {"mode": "intraday", "models": ["lgb_intraday_crypto", "lgb_intraday_etf"]},
}

# Model type → allowed horizons
MODEL_HORIZON_MAP = {
    "lstm":          "10d",
    "xgb_swing":     "10d",
    "tft_swing":     "10d",
    "tft_xgb_swing": "10d",
    "lgb_intraday":  "1d",
    "intraday_momentum": "1d",
    "lgb_intraday_crypto": "1h",
    "lgb_intraday_etf": "1h",
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

# Correlation clusters — highly correlated pairs with a combined exposure cap.
# More effective than sector labels for small universes (e.g. intraday 5-symbol).
# Each entry: (frozenset of symbols, max combined exposure as fraction of equity)
CORRELATION_CLUSTERS: List[Tuple[frozenset, float]] = [
    # Semiconductors: SMH/SOXX corr ~0.92 — cap combined at 20%
    (frozenset({"SMH", "SOXX"}), 0.20),
    # Tech overlap: QQQ/IGV corr ~0.88 — cap combined at 25%
    (frozenset({"QQQ", "IGV"}), 0.25),
    # Tech/semi triple overlap: QQQ/SMH/SOXX corr ~0.85+ — cap combined at 30%
    (frozenset({"QQQ", "SMH", "SOXX"}), 0.30),
    # Precious metals: GLD/SLV corr ~0.85 — cap combined at 20%
    (frozenset({"GLD", "SLV"}), 0.20),
]

# Drawdown throttle — equity-curve-based position size reduction.
# When current equity drops below peak by these thresholds, multiply
# position sizes by the corresponding factor.
# Format: list of (drawdown_pct, size_multiplier) sorted ascending.
DRAWDOWN_THROTTLE = [
    (0.05, 0.75),   # -5% from peak → 75% size
    (0.10, 0.50),   # -10% → 50% size
    (0.15, 0.00),   # -15% → halt new entries
]


def drawdown_size_mult(equity: float, peak_equity: float) -> float:
    """Return position size multiplier based on drawdown from peak equity.

    Returns 1.0 if no drawdown, or the throttle multiplier if drawdown
    exceeds a threshold. Returns 0.0 to halt new entries at deep drawdown.
    """
    if peak_equity <= 0 or equity >= peak_equity:
        return 1.0
    dd = (peak_equity - equity) / peak_equity
    mult = 1.0
    for threshold, factor in DRAWDOWN_THROTTLE:
        if dd >= threshold:
            mult = factor
    return mult


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
    max_positions: int = 5              # max simultaneous positions
    min_signal_scale: float = 0.1       # floor for signal_pct (crypto uses 0.2)

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

    # --- Regime gate vs soft scaling ---
    # When False (default), hardcoded SMA/VIX gates are disabled and
    # the model's own regime features drive entry decisions.
    use_hardcoded_regime_gates: bool = False
    # Soft regime scaling: reduce size (not block) in adverse regimes.
    use_soft_regime_scaling: bool = True
    bear_scaling_factor: float = 0.5    # size mult when SPY < SMA(200)
    high_vix_scaling_factor: float = 0.5  # size mult when VIX >= 30

    # --- Kelly ---
    kelly_cap: float = 0.25            # max Kelly fraction
    kelly_min_trades: int = 20         # min trades before Kelly kicks in
    kelly_window: int = 60             # rolling window for Kelly
    # Cross-group Kelly discount: with N concurrent strategies, each group's
    # Kelly is scaled by 1/sqrt(N) (quarter-Kelly for N=4). Applied at sizing
    # time to reduce correlated drawdown risk across the total portfolio.
    cross_group_kelly_discount: float = 0.50  # 1/sqrt(4) ≈ 0.50 for 4 groups

    # --- Stress test multipliers ---
    stress_cost_mult: float = 1.0      # 1.0 = normal, 2.0 = stress
    stress_spread_mult: float = 1.0


# Pre-built configs per group (v3 — OOS-validated sizing)
SWING_RISK = RiskConfig(
    position_pct=0.45,
    max_position_pct=0.12,
    max_sector_pct=0.35,
    max_total_exposure=0.65,
    max_positions=5,    # Kelly-focused: fewer slots = higher conviction per position
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
    max_position_pct=0.10,        # was 0.05 — allow 10% per coin
    max_sector_pct=0.40,          # was 0.15
    max_total_exposure=0.40,      # was 0.15 — deploy up to 40% of equity
    max_positions=6,              # max concurrent positions (risk limit, not selection)
    min_signal_scale=0.2,         # crypto floor: 20% of base (was 10%)
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
    max_positions=3,                  # 12% total / 4% max per-pos = 3 realistic slots
    cost_threshold=0.005,             # 50bps (Kraken taker ~26bps RT; 50bps gives margin)
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


GROUP_EXPECTED_METADATA: Dict[str, Dict[str, object]] = {
    "swing": {
        "mode": "daily",
        "horizon": "10d",
        "required_models": {"xgb_swing", "tft_swing"},
    },
    "intraday": {
        "mode": "intraday",
        "horizon": "1h",
        "required_models": {"lgb_intraday_etf"},
    },
    "crypto": {
        "mode": "daily",
        "horizon": "10d",
        "required_models": {"xgb_swing", "tft_swing"},
    },
    "crypto_intraday": {
        "mode": "intraday",
        "horizon": "1h",
        "required_models": {"lgb_intraday_crypto"},
    },
}

CONFIG_TO_RISK_FIELD: Dict[str, str] = {
    "position_pct": "position_pct",
    "max_position_pct": "max_position_pct",
    "max_sector_pct": "max_sector_pct",
    "max_total_exposure": "max_total_exposure",
    "max_positions": "max_positions",
    "cost_threshold": "cost_threshold",
    "target_return": "target_return",
    "disaster_stop_mult": "disaster_stop_atr_mult",
    "disaster_stop_max_pct": "disaster_stop_max_pct",
    "profit_lock_atr_mult": "profit_lock_atr_mult",
    "profit_lock_trail_atr_mult": "profit_lock_trail_atr_mult",
    "max_underwater_days": "max_underwater_days",
    "loss_cooldown_hours": "loss_cooldown_hours",
    "max_loss_cooldown_hours": "max_loss_cooldown_hours",
    "post_loss_size_mult": "post_loss_size_mult",
    "post_loss_size_hours": "post_loss_size_hours",
    "same_dir_confidence_mult": "same_dir_confidence_mult",
    "trend_sma": "trend_sma_period",
    "kelly_cap": "kelly_cap",
    "cross_group_kelly_discount": "cross_group_kelly_discount",
}


def _default_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "trading.json",
    )


def _get_config_path(config_path: Optional[str] = None) -> str:
    if config_path:
        return config_path
    return os.environ.get("QUANT_STOCKS_CONFIG_PATH", _default_config_path())


@lru_cache(maxsize=4)
def _load_trading_config(config_path: str) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid trading config JSON at {config_path}: {exc}") from exc


def _validate_group_metadata(group: str, group_cfg: dict, config_path: str) -> None:
    expected = GROUP_EXPECTED_METADATA.get(group)
    if not expected or not group_cfg:
        return

    mode = group_cfg.get("mode")
    if mode is not None and mode != expected["mode"]:
        raise ValueError(
            f"{config_path} [{group}] mode={mode!r} does not match authoritative "
            f"mode={expected['mode']!r}"
        )

    horizon = group_cfg.get("horizon")
    if horizon is not None and horizon != expected["horizon"]:
        raise ValueError(
            f"{config_path} [{group}] horizon={horizon!r} does not match authoritative "
            f"horizon={expected['horizon']!r}"
        )

    allowed_models = group_cfg.get("allowed_models")
    if isinstance(allowed_models, list):
        missing = set(expected["required_models"]) - set(allowed_models)
        if missing:
            raise ValueError(
                f"{config_path} [{group}] allowed_models is missing required model(s): "
                f"{sorted(missing)}"
            )


def _apply_group_overrides(group: str, base: RiskConfig, config_path: Optional[str] = None) -> RiskConfig:
    path = _get_config_path(config_path)
    cfg = _load_trading_config(path)
    group_cfg = cfg.get(group, {})
    if not isinstance(group_cfg, dict):
        return base

    _validate_group_metadata(group, group_cfg, path)

    resolved = replace(base)
    for cfg_key, risk_field in CONFIG_TO_RISK_FIELD.items():
        if cfg_key in group_cfg:
            setattr(resolved, risk_field, group_cfg[cfg_key])
    return resolved


def get_risk_config(group: Optional[str] = None, config_path: Optional[str] = None) -> RiskConfig:
    """Get risk config for a group, with fallback to swing defaults."""
    selected_group = group if group in GROUP_RISK_CONFIGS else "swing"
    base = GROUP_RISK_CONFIGS[selected_group]
    return _apply_group_overrides(selected_group, base, config_path=config_path)


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

    # 3. Max positions (count both long and short positions)
    active_positions = sum(1 for p in positions.values() if p.get("qty", 0) != 0)
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

    # 6. Correlation cluster cap — limits combined exposure for highly correlated pairs.
    # More effective than sector labels for small universes where corr > 0.85.
    for cluster_symbols, max_combined in CORRELATION_CLUSTERS:
        if symbol in cluster_symbols:
            cluster_exposure = sum(
                p["qty"] * p["current_price"] / equity
                for s, p in positions.items()
                if s in cluster_symbols and s != symbol
            )
            if cluster_exposure + position_pct > max_combined:
                return False, (
                    f"correlated cluster {cluster_symbols} exposure "
                    f"{cluster_exposure + position_pct:.1%} > max {max_combined:.1%}"
                )

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


def _equal_cap_for_group(group: str) -> float:
    """Compute equal-weight per-symbol cap from group risk config.

    equal_cap = max_total_exposure / max_positions
    This is used as the shrinkage prior for Sharpe-tiered caps.
    """
    risk = get_risk_config(group)
    if risk.max_positions > 0:
        return risk.max_total_exposure / risk.max_positions
    return risk.max_position_pct


# Shrinkage blend: 50% equal cap + 50% Sharpe-tiered cap (Rob Carver).
# Guards against noisy OOS Sharpe estimates from short backtest windows.
# Symbols with cap=0 (disabled) stay at 0 — shrinkage doesn't override disabling.
_SYMBOL_CAP_SHRINKAGE = 0.50


def get_symbol_cap(symbol: str, group: str) -> float:
    """Get max % equity for a specific symbol in a group.

    Uses shrinkage blend: 50% equal cap + 50% Sharpe-tiered cap.
    This limits damage from noisy OOS Sharpe estimates while still
    rewarding genuinely better models (Rob Carver, Systematic Trading).

    Symbols with cap=0 (disabled) remain at 0 — shrinkage doesn't
    override the disable gate.

    Checks config/trading.json first, falls back to _DEFAULT_SYMBOL_CAPS.
    Returns 1.0 (uncapped) if symbol not found in either.
    """
    caps = _load_symbol_caps_from_config()
    group_caps = caps.get(group, {})

    # Normalize crypto symbol format: BTC/USD → BTC-USD for lookup
    lookup_sym = symbol.replace("/", "-")

    raw_cap = None
    if lookup_sym in group_caps:
        raw_cap = group_caps[lookup_sym]
    elif group == "intraday":
        raw_cap = _DEFAULT_SYMBOL_CAPS.get(f"{symbol}_intraday")
    else:
        raw_cap = _DEFAULT_SYMBOL_CAPS.get(lookup_sym)

    if raw_cap is None:
        return 0.15  # unknown symbol → conservative default (no shrinkage)

    # Disabled symbols stay disabled — shrinkage doesn't override
    if raw_cap <= 0.0:
        return 0.0

    # Shrinkage blend: 50% equal + 50% Sharpe-tiered
    equal_cap = _equal_cap_for_group(group)
    blended = _SYMBOL_CAP_SHRINKAGE * equal_cap + (1 - _SYMBOL_CAP_SHRINKAGE) * raw_cap
    return blended


def is_symbol_disabled(symbol: str, group: str) -> bool:
    """Return True if symbol cap is 0 (disabled by OOS performance)."""
    return get_symbol_cap(symbol, group) <= 0.0


def get_confidence_multiplier(symbol: str) -> float:
    """Map OOS Sharpe to a confidence multiplier (scales position size).

    Sharpe >= 1.5  → 1.0  (full confidence)
    0.7 <= S < 1.5 → 0.75
    0.3 <= S < 0.7 → 0.50
    0.0 <= S < 0.3 → 0.25 (probe only)
    S < 0.0        → 0.00 (disabled) for ETFs
                   → 0.30 for crypto (OOS Sharpe swings between quarters;
                     ML signal does the real work, so keep coin tradeable)
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
        # Negative OOS Sharpe: trade at reduced size, don't exclude.
        # Ranking already handles priority; conf_mult blocking trades at 0.0
        # violates the no-exclusion principle (ranking should sort, not gate).
        # Crypto floor 0.30 (Sharpe swings between regimes).
        # ETF floor 0.25 (probe size — ranking keeps it low priority).
        is_crypto = "/" in symbol or (lookup.endswith("-USD") and len(lookup) >= 6)
        return 0.30 if is_crypto else 0.25


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

    def half_kelly(self, window: int = 60, min_trades: int = 20) -> Optional[float]:
        """Compute half-Kelly fraction from rolling trade returns.

        f* = (b*p - q) / b   where p = win rate, q = 1-p, b = avg_win/avg_loss
        Returns f*/2 (half-Kelly) or None if insufficient data.
        """
        recent = self.returns[-window:]
        if len(recent) < min_trades:
            return None
        wins = [r for r in recent if r > 0]
        losses = [abs(r) for r in recent if r <= 0]
        if not wins or not losses:
            return None
        p = len(wins) / len(recent)
        q = 1.0 - p
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss < 1e-10:
            return None
        b = avg_win / avg_loss  # payoff ratio
        full_kelly = (b * p - q) / b
        if full_kelly <= 0:
            return 0.0
        return full_kelly / 2.0

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


# ---------------------------------------------------------------------------
# Volatility tiers — asset-aware exit parameters
# ---------------------------------------------------------------------------

class VolTier(Enum):
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass(frozen=True)
class ExitParams:
    """Fixed-percentage exit thresholds for a (group, vol_tier) pair."""
    disaster_stop_pct: float          # max loss before forced exit
    profit_lock_arm_pct: float        # PnL threshold to arm profit-lock
    profit_lock_trail_pct: float      # retrace from peak to trigger exit once armed
    breakeven_ratchet_pct: float      # once PnL >= this, floor stop = entry price
    max_underwater_days: int          # max days underwater before forced exit
    use_atr: bool = False             # True = use ATR multipliers instead of fixed %
    # ATR multipliers (only used when use_atr=True)
    disaster_atr_mult: float = 3.0
    arm_atr_mult: float = 2.0
    trail_atr_mult: float = 1.5
    # Signal-decay interaction
    signal_flip_consecutive: int = 2  # require N consecutive flips before exit
    min_hold_bars: int = 0            # minimum checks to hold before signal_decay can trigger


def get_effective_min_hold(
    base_min_hold: int,
    entry_atr: float,
    current_atr: float,
) -> int:
    """Return ATR-adaptive min hold clamped to [base/2, base*2]."""
    if base_min_hold <= 0:
        return 0
    if entry_atr <= 0 or current_atr <= 0:
        return base_min_hold

    vol_ratio = current_atr / entry_atr
    scaled = int(base_min_hold * vol_ratio)
    return max(base_min_hold // 2, min(scaled, base_min_hold * 2))

# Crypto intraday trade lifecycle (matched to crypto_intraday_model.py)
CRYPTO_INTRADAY_MIN_HOLD_BARS = 12    # 1 hour — 1 full prediction horizon (FORWARD_BARS)
CRYPTO_INTRADAY_MAX_HOLD_BARS = 48    # 4 hours — match model MAX_HOLD_BARS
CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS = 3  # 15-min cooldown on flip exits
CRYPTO_INTRADAY_BAR_SECONDS = 300     # 5-min bars = 300 seconds


# Per-group, per-tier exit configs (calibrated from MFE/MAE analysis, 4040 trades)
# EV breakeven at ~-3% MAE; equity recovery 0% past -10%; crypto ~44% to -8%
EXIT_PARAMS: Dict[str, Dict[VolTier, ExitParams]] = {
    "swing": {
        # Daily-horizon XGBoost+TFT checked every 5 min: min_hold=36 bars (3h)
        # prevents intraday noise from killing multi-day swing positions.
        VolTier.MEDIUM: ExitParams(
            disaster_stop_pct=0.04, profit_lock_arm_pct=0.02,
            profit_lock_trail_pct=0.015, breakeven_ratchet_pct=0.015,
            max_underwater_days=90, use_atr=True,
            disaster_atr_mult=3.0, arm_atr_mult=2.0, trail_atr_mult=1.5,
            signal_flip_consecutive=2, min_hold_bars=36,
        ),
        VolTier.HIGH: ExitParams(
            disaster_stop_pct=0.07, profit_lock_arm_pct=0.04,
            profit_lock_trail_pct=0.025, breakeven_ratchet_pct=0.02,
            max_underwater_days=60, signal_flip_consecutive=2,
            min_hold_bars=36,
        ),
        VolTier.ULTRA: ExitParams(
            disaster_stop_pct=0.06, profit_lock_arm_pct=0.03,
            profit_lock_trail_pct=0.015, breakeven_ratchet_pct=0.015,
            max_underwater_days=45, signal_flip_consecutive=2,
            min_hold_bars=36,
        ),
    },
    "intraday": {
        # 5-min bars, 1-day horizon: min_hold=6 bars (30 min) prevents
        # noise exits in the first confirmation window after entry.
        VolTier.MEDIUM: ExitParams(
            disaster_stop_pct=0.03, profit_lock_arm_pct=0.008,
            profit_lock_trail_pct=0.004, breakeven_ratchet_pct=0.005,
            max_underwater_days=1, signal_flip_consecutive=1,
            min_hold_bars=6,
        ),
        VolTier.HIGH: ExitParams(
            disaster_stop_pct=0.04, profit_lock_arm_pct=0.012,
            profit_lock_trail_pct=0.006, breakeven_ratchet_pct=0.007,
            max_underwater_days=1, signal_flip_consecutive=1,
            min_hold_bars=6,
        ),
        VolTier.ULTRA: ExitParams(
            disaster_stop_pct=0.04, profit_lock_arm_pct=0.012,
            profit_lock_trail_pct=0.006, breakeven_ratchet_pct=0.007,
            max_underwater_days=1, signal_flip_consecutive=1,
            min_hold_bars=6,
        ),
    },
    "crypto": {
        # Daily-horizon model checked every 5 min: min_hold=36 bars (3h)
        # prevents noise-driven exits. signal_flip_consecutive=3 requires
        # 15 min of sustained disagreement before exiting.
        VolTier.MEDIUM: ExitParams(
            disaster_stop_pct=0.08, profit_lock_arm_pct=0.04,
            profit_lock_trail_pct=0.02, breakeven_ratchet_pct=0.02,
            max_underwater_days=30, signal_flip_consecutive=3,
            min_hold_bars=36,
        ),
        VolTier.HIGH: ExitParams(
            disaster_stop_pct=0.08, profit_lock_arm_pct=0.04,
            profit_lock_trail_pct=0.02, breakeven_ratchet_pct=0.02,
            max_underwater_days=30, signal_flip_consecutive=3,
            min_hold_bars=36,
        ),
        VolTier.ULTRA: ExitParams(
            disaster_stop_pct=0.08, profit_lock_arm_pct=0.04,
            profit_lock_trail_pct=0.02, breakeven_ratchet_pct=0.02,
            max_underwater_days=30, signal_flip_consecutive=3,
            min_hold_bars=36,
        ),
    },
    "crypto_intraday": {
        # signal_flip_consecutive and min_hold_bars are unused for this group —
        # crypto_intraday uses horizon-aware hold logic (CRYPTO_INTRADAY_*_BARS)
        # in paper_trader.py instead of generic signal_decay.
        VolTier.MEDIUM: ExitParams(
            disaster_stop_pct=0.03, profit_lock_arm_pct=0.007,
            profit_lock_trail_pct=0.004, breakeven_ratchet_pct=0.005,
            max_underwater_days=0,
        ),
        VolTier.HIGH: ExitParams(
            disaster_stop_pct=0.03, profit_lock_arm_pct=0.007,
            profit_lock_trail_pct=0.004, breakeven_ratchet_pct=0.005,
            max_underwater_days=0,
        ),
        VolTier.ULTRA: ExitParams(
            disaster_stop_pct=0.03, profit_lock_arm_pct=0.007,
            profit_lock_trail_pct=0.004, breakeven_ratchet_pct=0.005,
            max_underwater_days=0,
        ),
    },
}


# ---------------------------------------------------------------------------
# A/B exit variant system
# ---------------------------------------------------------------------------

class ExitVariant(Enum):
    A = "A"  # tighter stops + trails
    B = "B"  # current/looser stops + trails


# Only override the groups/tiers under test; everything else uses EXIT_PARAMS
EXIT_VARIANT_OVERRIDES: Dict[ExitVariant, Dict[str, Dict[VolTier, ExitParams]]] = {
    ExitVariant.A: {
        "swing": {
            VolTier.ULTRA: ExitParams(
                disaster_stop_pct=0.06, profit_lock_arm_pct=0.03,
                profit_lock_trail_pct=0.015, breakeven_ratchet_pct=0.015,
                max_underwater_days=45, signal_flip_consecutive=2,
            ),
        },
        "crypto": {
            VolTier.HIGH: ExitParams(
                disaster_stop_pct=0.08, profit_lock_arm_pct=0.04,
                profit_lock_trail_pct=0.02, breakeven_ratchet_pct=0.02,
                max_underwater_days=30, signal_flip_consecutive=2,
            ),
        },
    },
    ExitVariant.B: {
        "swing": {
            VolTier.ULTRA: ExitParams(
                disaster_stop_pct=0.08, profit_lock_arm_pct=0.03,
                profit_lock_trail_pct=0.020, breakeven_ratchet_pct=0.015,
                max_underwater_days=45, signal_flip_consecutive=2,
            ),
        },
        "crypto": {
            VolTier.HIGH: ExitParams(
                disaster_stop_pct=0.10, profit_lock_arm_pct=0.04,
                profit_lock_trail_pct=0.025, breakeven_ratchet_pct=0.02,
                max_underwater_days=30, signal_flip_consecutive=2,
            ),
        },
    },
}


def classify_vol_tier(
    atr_price_ratio: float,
    vol20_annualized: float,
) -> VolTier:
    """Classify a symbol into a volatility tier.

    Args:
        atr_price_ratio: ATR(14) / current_price
        vol20_annualized: 20-day realized volatility (annualized, e.g. 0.50 = 50%)
    """
    if atr_price_ratio > 0.05 or vol20_annualized > 0.50:
        return VolTier.ULTRA
    if atr_price_ratio > 0.025 or vol20_annualized > 0.35:
        return VolTier.HIGH
    return VolTier.MEDIUM


def compute_vol_metrics(prices_df) -> Tuple[float, float]:
    """Compute ATR/Price ratio and 20d annualized vol from a DataFrame with OHLC columns.

    Args:
        prices_df: DataFrame with 'High', 'Low', 'Close' columns (at least 21 rows)

    Returns:
        (atr_price_ratio, vol20_annualized)
    """
    import numpy as np

    close = prices_df["Close"].values.flatten()
    high = prices_df["High"].values.flatten()
    low = prices_df["Low"].values.flatten()

    # ATR(14)
    trs = []
    for i in range(1, len(close)):
        trs.append(max(high[i] - low[i],
                       abs(high[i] - close[i - 1]),
                       abs(low[i] - close[i - 1])))
    atr14 = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))
    atr_price_ratio = atr14 / close[-1] if close[-1] > 0 else 0.0

    # 20d realized vol (annualized)
    if len(close) >= 21:
        rets = np.diff(np.log(close[-21:]))
        vol20 = float(np.std(rets) * np.sqrt(252))
    else:
        rets = np.diff(np.log(close))
        vol20 = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0

    return atr_price_ratio, vol20


def get_exit_params(group: str, tier: VolTier,
                    variant: Optional[ExitVariant] = None) -> ExitParams:
    """Look up exit parameters for a (group, tier) pair, optionally with A/B variant.

    If variant is set, checks EXIT_VARIANT_OVERRIDES first for that (group, tier).
    Falls back to EXIT_PARAMS, then to HIGH tier, then to swing group.
    """
    if variant is not None:
        overrides = EXIT_VARIANT_OVERRIDES.get(variant, {})
        group_overrides = overrides.get(group, {})
        if tier in group_overrides:
            return group_overrides[tier]
    group_params = EXIT_PARAMS.get(group, EXIT_PARAMS["swing"])
    return group_params.get(tier, group_params.get(VolTier.HIGH, list(group_params.values())[0]))
