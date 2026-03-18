#!/usr/bin/env python3
"""
Feature Selection Pipeline for Crypto Swing Models
====================================================
4-step pipeline:
  1. Factor taxonomy — group all 88 features by economic role
  2. Univariate IC analysis — rolling/segmented IC, stability metrics
  3. Feature selection — dead removal, redundancy filter, group representation
  4. Config output — write config/features_crypto_swing.json

Usage:
    python scripts/select_features.py --asset-class crypto_swing --target-count 20
    python scripts/select_features.py --asset-class crypto_swing --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("select_features")

# ---------------------------------------------------------------------------
# Step 1: Factor Taxonomy
# ---------------------------------------------------------------------------

FACTOR_TAXONOMY: Dict[str, List[str]] = {
    "momentum": [
        "ret252", "ret63", "ret21", "ret5",
        "rel_momentum_vs_spy_63", "rel_momentum_vs_spy_252",
        "momentum_quality", "factor_momentum",
    ],
    "volatility": [
        "vol20", "vol_regime", "bb_bandwidth",
        "btc_realized_vol_30",
        "iv_rv_spread", "iv_rv_zscore", "vvix_level",
        "sector_dispersion",
    ],
    "technical": [
        "rsi14", "bb_pct_b", "macd_hist_norm",
        "ibs", "adx", "dv_accel",
    ],
    "microstructure": [
        "volume_zscore", "volume_spike", "price_volume_divergence", "volume_trend",
        "vpoc_distance", "vol_support_strength", "vol_resistance_strength", "vol_sr_imbalance",
    ],
    "regime": [
        "spy_above_sma200", "spy_trend_strength",
        "vix_regime", "vix_zscore", "vix_pctrank",
        "sector_etf_rel_spy_21",
        "btc_sma200_flag", "btc_drawdown", "eth_btc_ratio_zscore",
        "btc_ret5", "btc_ret21", "btc_trend_strength", "btc_momentum_accel",
    ],
    "macro": [
        "treasury_slope", "yield_curve_3m10y",
        "real_yield_10y", "breakeven_inflation_5y", "fed_funds_rate",
        "credit_spread", "credit_spread_chg5", "credit_spread_zscore",
    ],
    "risk_appetite": [
        "risk_appetite_ratio", "risk_appetite_trend",
        "smallcap_rotation", "equity_carry", "correlation_regime",
    ],
    "fama_french": [
        "beta_mkt", "beta_smb", "beta_hml",
        "beta_rmw", "beta_cma", "beta_mom",
    ],
    "calendar": [
        "day_of_week", "turn_of_month",
        "month_sin", "month_cos", "fomc_proximity",
    ],
    "cboe_options": [
        "pc_volume_ratio", "pc_oi_ratio", "vix_term_ratio", "vix_term_inverted",
        "cboe_vix_term_spread", "cboe_vix_term_ratio",
        "cboe_vix9d_ratio", "cboe_skew", "cboe_vix_percentile",
    ],
    "breadth": [
        "sector_breadth", "breadth_momentum",
    ],
    "sentiment": [
        "okx_long_short_ratio", "okx_ls_zscore",
        "fear_greed_index", "fear_greed_zscore",
        "polymarket_btc_bullish",
    ],
    "cross_asset": [
        "real_yield_spread", "gold_silver_ratio",
        "crude_ret5", "crude_ret20",
        "usdjpy_ret5", "usdjpy_ret20",
        "usd_strength_ret5", "usd_strength_ret20",
        "t10y3m_slope", "hy_credit_spread",
        "eurusd_ret5", "eurusd_ret20",
        "gbpusd_ret5", "gbpusd_ret20",
        "copper_ret5",
        # treasury_slope excluded — already in "macro" group
    ],
}

# Reverse lookup: feature -> group
FEATURE_TO_GROUP: Dict[str, str] = {}
for group, feats in FACTOR_TAXONOMY.items():
    for f in feats:
        FEATURE_TO_GROUP[f] = group

ALL_FEATURES = list(FEATURE_TO_GROUP.keys())

# Coin group classification (crypto)
COIN_GROUPS = {
    "majors": ["BTC-USD", "ETH-USD"],
    "large_caps": [
        "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD",
        "LINK-USD", "DOT-USD", "BCH-USD", "LTC-USD",
    ],
    "meme": ["DOGE-USD", "SHIB-USD", "WIF-USD"],
}

# ETF group classification (equity)
ETF_GROUPS = {
    "us_factor": [
        "SPY", "QQQ", "IWM", "IWF", "VTV", "IWB", "QUAL", "USMV", "MTUM", "VBR",
    ],
    "us_sector": [
        "XLE", "SMH", "XLK", "XLF", "XLI", "XLV", "XLP", "XLY", "XLU",
        "XLB", "XLRE", "IGV", "SOXX", "ARKK", "CIBR",
    ],
    "commodities": ["GLD", "SLV", "USO", "GDX", "GDXJ", "IAU", "PDBC", "URNM"],
    "fixed_income": ["TLT", "IEF", "SHY", "BND", "TIP", "LQD", "HYG", "EMB"],
    "intl_em": ["EEM", "EWY", "EWZ", "EWT", "INDA", "MCHI", "EWW", "VWO"],
    "intl_developed": ["EWJ", "VGK", "EWA", "EWC", "EWH", "EWU", "EWG"],
    "crypto_etf": ["IBIT", "ETHA", "FBTC"],
}

# Groups that MUST have at least 1 surviving feature (per asset class)
REQUIRED_GROUPS = {
    "crypto_swing": ["momentum", "volatility", "technical", "microstructure", "regime", "macro"],
    "equity_swing": ["momentum", "volatility", "technical", "microstructure", "regime", "macro", "cross_asset"],
}

# ---------------------------------------------------------------------------
# Domain priors: per-asset-class group weights
# ---------------------------------------------------------------------------
# These multiply the composite score for features in each group.
# >1.0 = "this group matters more for this asset class, protect from pruning"
# <1.0 = "this group matters less, easier to prune"
# =1.0 = "no opinion, let data decide"
#
# Rationale (crypto_swing):
#   - Crypto alpha is primarily in price/volume/momentum (Liu et al. 2022,
#     Cong et al. 2023). Multi-horizon TSMOM (1-8wk lookback) is the strongest
#     documented crypto factor.
#   - BTC regime features are structurally important for altcoins (BTC leads).
#   - Macro/equity factors can have spuriously high IC over short samples
#     because crypto correlates with risk-on/off regimes during bull/bear
#     transitions. Their predictive value for *individual coin* swings is low.
#   - Fama-French has no theoretical basis for crypto (no earnings, book value,
#     investment, or profitability metrics for tokens).
#   - CBOE options flow is equity-market-specific.
#
# Rationale (equity_swing):
#   - Traditional factor investing applies: macro/yield curve, Fama-French
#     factor betas, CBOE vol surface are all well-documented.
#   - Calendar effects (FOMC, turn-of-month) are strongest in equities.
#   - Price/volume still matters but less dominant than crypto.

DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "crypto_swing": {
        "momentum":      1.5,   # crypto TSMOM is the #1 factor
        "volatility":    1.3,   # vol regime / breakout critical
        "technical":     1.3,   # mean-reversion, trend signals
        "microstructure": 1.3,  # volume/liquidity signals
        "regime":        1.2,   # BTC regime important for alts
        "macro":         0.6,   # keep minimal — context only
        "risk_appetite": 0.7,   # some value as risk-on/off proxy
        "fama_french":   0.3,   # no theoretical basis for crypto
        "calendar":      0.4,   # 24/7 market, weak effects
        "cboe_options":  0.4,   # equity options market
        "breadth":       0.5,   # equity-centric
        "sentiment":     0.8,   # OKX/Fear&Greed somewhat useful but unreliable
        "cross_asset":   0.3,   # FRED/FX signals not useful for crypto
    },
    "equity_swing": {
        "momentum":      1.0,
        "volatility":    1.0,
        "technical":     1.0,
        "microstructure": 1.0,
        "regime":        1.0,
        "macro":         1.2,   # macro matters for equities
        "risk_appetite": 1.1,
        "fama_french":   1.2,   # well-documented factor premia
        "calendar":      1.0,
        "cboe_options":  1.1,   # vol surface is informative
        "breadth":       1.0,
        "sentiment":     0.5,   # crypto sentiment not relevant
        "cross_asset":   1.3,   # FX/commodity/yield drivers are core for ETFs
    },
}

# Per-asset-class group min/max slots — hard constraints
# min: minimum features to keep from this group (even if low IC)
# max: maximum features to keep (cap over-represented groups)
GROUP_SLOTS: Dict[str, Dict[str, Dict[str, int]]] = {
    "crypto_swing": {
        "momentum":      {"min": 3, "max": 5},   # full horizon ladder: 5d/21d/63d/252d + style
        "volatility":    {"min": 2, "max": 3},
        "technical":     {"min": 2, "max": 4},
        "microstructure": {"min": 1, "max": 3},
        "regime":        {"min": 1, "max": 3},   # spy_trend + vix as context
        "macro":         {"min": 1, "max": 1},   # single macro context feature
        "risk_appetite": {"min": 0, "max": 1},   # single risk-on/off proxy
        "fama_french":   {"min": 0, "max": 1},
        "calendar":      {"min": 0, "max": 1},
        "cboe_options":  {"min": 0, "max": 1},
        "breadth":       {"min": 0, "max": 1},
        "sentiment":     {"min": 0, "max": 1},
        "cross_asset":   {"min": 0, "max": 1},   # not useful for crypto
    },
    "equity_swing": {
        "momentum":      {"min": 2, "max": 4},
        "volatility":    {"min": 2, "max": 4},
        "technical":     {"min": 2, "max": 4},
        "microstructure": {"min": 1, "max": 3},
        "regime":        {"min": 1, "max": 3},
        "macro":         {"min": 2, "max": 4},
        "risk_appetite": {"min": 1, "max": 2},
        "fama_french":   {"min": 1, "max": 3},
        "calendar":      {"min": 0, "max": 2},
        "cboe_options":  {"min": 1, "max": 3},
        "breadth":       {"min": 0, "max": 2},
        "sentiment":     {"min": 0, "max": 1},
        "cross_asset":   {"min": 2, "max": 5},   # FX/commodity/yield — core for ETFs
    },
}

# Features that MUST be included regardless of composite score.
# These encode domain priors that pure IC/importance can't capture:
#   - ret5: matches 10d holding period (short-horizon momentum, weak mean IC
#           but structurally tied to the prediction target)
#   - dv_accel: event/volume acceleration signal — low mean IC but high
#           conditional edge on extreme-vol subsamples (single-day spike → 3-10d
#           repair). Valuable as a conditional filter even if unconditional IC ≈ 0.05.
FORCED_FEATURES: Dict[str, List[str]] = {
    "crypto_swing": ["ret5", "ret63", "dv_accel"],
    "equity_swing": [],
}


# ---------------------------------------------------------------------------
# Step 2: IC Analysis
# ---------------------------------------------------------------------------

def load_importance_data(model_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load XGBoost feature importances from all *_importance.json files."""
    result = {}
    for fpath in model_dir.glob("*_xgb_swing_importance.json"):
        try:
            data = json.loads(fpath.read_text())
            symbol = data.get("symbol", fpath.stem.replace("_xgb_swing_importance", ""))
            feature_cols = data.get("feature_cols", [])
            importances = data.get("importances", [])
            if len(feature_cols) == len(importances):
                result[symbol] = dict(zip(feature_cols, importances))
        except (json.JSONDecodeError, KeyError):
            continue
    return result


def compute_ic_metrics(
    feature_series: pd.Series,
    forward_returns: pd.Series,
    spy_trend: Optional[pd.Series] = None,
    window: int = 60,
) -> Dict[str, float]:
    """Compute IC metrics for one feature vs forward returns."""
    # Drop NaN in both
    mask = feature_series.notna() & forward_returns.notna()
    feat = feature_series[mask]
    fwd = forward_returns[mask]

    if len(feat) < 30:
        return {
            "full_ic": 0.0, "ic_mean": 0.0, "ic_std": 1.0,
            "ic_ir": 0.0, "pct_positive": 0.0,
            "direction_consistency": 0.0,
            "bull_ic": 0.0, "bear_ic": 0.0,
            "n_samples": len(feat),
        }

    # Full-sample IC
    full_ic, _ = stats.spearmanr(feat, fwd)
    if np.isnan(full_ic):
        full_ic = 0.0

    # Rolling IC
    rolling_ic = feat.rolling(window).corr(fwd)
    rolling_ic = rolling_ic.dropna()

    if len(rolling_ic) < 5:
        return {
            "full_ic": full_ic, "ic_mean": full_ic, "ic_std": 0.5,
            "ic_ir": 0.0, "pct_positive": 0.5,
            "direction_consistency": 0.5,
            "bull_ic": full_ic, "bear_ic": full_ic,
            "n_samples": len(feat),
        }

    ic_mean = rolling_ic.mean()
    ic_std = rolling_ic.std()
    pct_positive = (rolling_ic > 0).mean()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

    # Direction consistency
    if full_ic != 0:
        direction_consistency = ((rolling_ic > 0) == (full_ic > 0)).mean()
    else:
        direction_consistency = 0.0

    # Regime-segmented IC
    bull_ic = 0.0
    bear_ic = 0.0
    if spy_trend is not None:
        spy_aligned = spy_trend.reindex(feat.index)
        bull_mask = spy_aligned > 0
        bear_mask = spy_aligned <= 0
        if bull_mask.sum() > 20:
            c, _ = stats.spearmanr(feat[bull_mask], fwd[bull_mask])
            bull_ic = 0.0 if np.isnan(c) else c
        if bear_mask.sum() > 20:
            c, _ = stats.spearmanr(feat[bear_mask], fwd[bear_mask])
            bear_ic = 0.0 if np.isnan(c) else c

    return {
        "full_ic": round(full_ic, 4),
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "ic_ir": round(ic_ir, 4),
        "pct_positive": round(pct_positive, 4),
        "direction_consistency": round(direction_consistency, 4),
        "bull_ic": round(bull_ic, 4),
        "bear_ic": round(bear_ic, 4),
        "n_samples": len(feat),
    }


def run_ic_analysis(
    symbols: List[str],
    model_dir: Path,
    provider: str = "yahoo",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full IC analysis across all symbols and features.

    Returns:
        ic_df: DataFrame with columns [symbol, feature, group, full_ic, ic_mean, ...]
        corr_matrix: pairwise feature correlation matrix (averaged across symbols)
    """
    import swing_model as _sm
    from swing_model import SwingFeatureEngine, get_swing_feature_cols, FORWARD_DAYS
    from signals_engine import build_adapter
    from utils import _fetch_vix_for_training

    importance_data = load_importance_data(model_dir)
    log.info("Loaded importance data for %d symbols", len(importance_data))

    # Monkey-patch get_swing_feature_cols to return ALL features.
    # build_features() uses this to filter its output columns; we need
    # the full superset for IC analysis, not just the config-selected subset.
    _original_get_cols = _sm.get_swing_feature_cols
    _sm.get_swing_feature_cols = lambda symbol=None: ALL_FEATURES

    adapter = build_adapter(provider)
    all_ic_rows = []
    corr_matrices = []

    # Pre-fetch VIX and SPY data (shared across all symbols)
    fred_key = os.environ.get("FRED_API_KEY", "")
    try:
        vix_df = _fetch_vix_for_training(fred_key=fred_key, lookback_days=800, include_live=False)
    except Exception as e:
        log.warning("VIX fetch failed: %s", e)
        vix_df = pd.DataFrame()
    try:
        spy_bars = adapter.fetch_daily("SPY", lookback=800)
    except Exception:
        spy_bars = None

    engine = SwingFeatureEngine()

    for symbol in symbols:
        log.info("=== IC analysis: %s ===", symbol)
        try:
            bars = adapter.fetch_daily(symbol, lookback=800)
            if bars is None or len(bars) < 100:
                log.warning("Skip %s — insufficient data (%s bars)", symbol, len(bars) if bars is not None else 0)
                continue
        except Exception as e:
            log.warning("Skip %s — fetch failed: %s", symbol, e)
            continue

        # Build features
        try:
            feat_df = engine.build_features(bars, vix_df, spy_bars=spy_bars, symbol=symbol)
        except Exception as e:
            log.warning("Skip %s — build_features failed: %s", symbol, e)
            continue

        if feat_df is None or len(feat_df) < 60:
            log.warning("Skip %s — too few rows after feature build (%s)", symbol, len(feat_df) if feat_df is not None else 0)
            continue

        # Compute forward returns from original bars (feat_df has features only)
        close_col = "Close" if "Close" in bars.columns else "close"
        close_series = bars[close_col].reindex(feat_df.index)
        feat_df["fwd_ret"] = close_series.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
        feat_df = feat_df.dropna(subset=["fwd_ret"])

        if len(feat_df) < 60:
            log.warning("Skip %s — too few rows after fwd_ret (%d)", symbol, len(feat_df))
            continue

        log.info("  %s: %d rows, computing IC for %d features", symbol, len(feat_df), len(ALL_FEATURES))

        # Get spy_trend for regime segmentation
        spy_trend = feat_df.get("spy_trend_strength")

        # Get symbol's importance scores
        sym_importance = importance_data.get(symbol, {})

        # Feature correlation matrix for this symbol
        available_feats = [f for f in ALL_FEATURES if f in feat_df.columns]
        if available_feats:
            sym_corr = feat_df[available_feats].corr(method="spearman")
            corr_matrices.append(sym_corr)

        # IC per feature
        for feat_name in ALL_FEATURES:
            if feat_name not in feat_df.columns:
                # Feature not computed for this symbol — skip
                continue
            if feat_df[feat_name].nunique() < 3:
                # Constant or near-constant — skip
                ic_row = {
                    "symbol": symbol, "feature": feat_name,
                    "group": FEATURE_TO_GROUP.get(feat_name, "unknown"),
                    "full_ic": 0.0, "ic_mean": 0.0, "ic_std": 1.0,
                    "ic_ir": 0.0, "pct_positive": 0.5,
                    "direction_consistency": 0.0,
                    "bull_ic": 0.0, "bear_ic": 0.0,
                    "n_samples": 0,
                    "xgb_importance": sym_importance.get(feat_name, 0.0),
                }
                all_ic_rows.append(ic_row)
                continue

            metrics = compute_ic_metrics(
                feat_df[feat_name], feat_df["fwd_ret"],
                spy_trend=spy_trend, window=60,
            )
            metrics["symbol"] = symbol
            metrics["feature"] = feat_name
            metrics["group"] = FEATURE_TO_GROUP.get(feat_name, "unknown")
            metrics["xgb_importance"] = sym_importance.get(feat_name, 0.0)
            all_ic_rows.append(metrics)

    # Restore original get_swing_feature_cols
    _sm.get_swing_feature_cols = _original_get_cols

    ic_df = pd.DataFrame(all_ic_rows)

    # Average correlation matrix across symbols
    if corr_matrices:
        avg_corr = pd.concat(corr_matrices).groupby(level=0).mean()
        # Re-index to make it square
        all_idx = sorted(set().union(*[m.columns for m in corr_matrices]))
        avg_corr = avg_corr.reindex(index=all_idx, columns=all_idx).fillna(0)
    else:
        avg_corr = pd.DataFrame()

    return ic_df, avg_corr


# ---------------------------------------------------------------------------
# Step 3: Feature Selection
# ---------------------------------------------------------------------------

def select_features(
    ic_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    asset_class: str = "crypto_swing",
    target_count: int = 20,
    ic_threshold: float = 0.02,
    importance_threshold: float = 0.005,
    corr_threshold: float = 0.80,
) -> Tuple[List[str], Dict]:
    """Select features using IC + importance + domain priors.

    Scoring: composite = (0.4 * ic_score + 0.6 * imp_score) * domain_weight
    The 60/40 split favors XGBoost importance over raw IC because:
      - IC is noisy at N=188 (Spearman CI ≈ ±0.07 at this sample size)
      - XGBoost importance reflects nonlinear, conditional usefulness
      - IC captures only linear rank correlation, missing interaction effects
    Domain weights then adjust per asset class.

    Returns:
        selected: list of selected feature names
        metadata: selection metadata dict
    """
    if ic_df.empty:
        log.error("No IC data — cannot select features")
        return [], {}

    domain_weights = DOMAIN_WEIGHTS.get(asset_class, {})
    group_slots = GROUP_SLOTS.get(asset_class, {})

    # Aggregate IC across symbols (mean of absolute values)
    agg = ic_df.groupby("feature").agg(
        mean_abs_ic=("full_ic", lambda x: np.abs(x).mean()),
        mean_ic=("full_ic", "mean"),
        mean_ic_ir=("ic_ir", lambda x: np.abs(x).mean()),
        mean_pct_positive=("pct_positive", "mean"),
        mean_dir_consistency=("direction_consistency", "mean"),
        mean_xgb_importance=("xgb_importance", "mean"),
        max_xgb_importance=("xgb_importance", "max"),
        n_symbols=("symbol", "nunique"),
    ).reset_index()
    agg["group"] = agg["feature"].map(FEATURE_TO_GROUP)

    # Score: composite of IC and importance (normalized), weighted by domain prior
    max_ic = agg["mean_abs_ic"].max()
    max_imp = agg["mean_xgb_importance"].max()
    agg["ic_score"] = agg["mean_abs_ic"] / max_ic if max_ic > 0 else 0
    agg["imp_score"] = agg["mean_xgb_importance"] / max_imp if max_imp > 0 else 0

    # Raw composite: 40% IC + 60% importance
    agg["raw_score"] = 0.4 * agg["ic_score"] + 0.6 * agg["imp_score"]

    # Apply domain weight per group
    agg["domain_weight"] = agg["group"].map(lambda g: domain_weights.get(g, 1.0))
    agg["composite_score"] = agg["raw_score"] * agg["domain_weight"]

    log.info("Scoring: 0.4*IC + 0.6*IMP × domain_weight (%s)", asset_class)

    # --- Step 3a: Hard remove dead features ---
    # Dead = near-zero IC AND near-zero importance (truly useless)
    dead_mask = (agg["mean_abs_ic"] < ic_threshold) & (agg["max_xgb_importance"] < importance_threshold)
    dead_features = set(agg.loc[dead_mask, "feature"].tolist())
    log.info("Dead features (|IC|<%.2f AND imp<%.3f): %d — %s",
             ic_threshold, importance_threshold, len(dead_features), sorted(dead_features))

    alive = agg[~dead_mask].copy()

    # --- Step 3b: Redundancy removal within groups ---
    removed_redundant = []
    features_to_drop = set()

    for group_name in FACTOR_TAXONOMY:
        group_feats = alive[alive["group"] == group_name]["feature"].tolist()
        if len(group_feats) < 2:
            continue
        for i, f1 in enumerate(group_feats):
            if f1 in features_to_drop:
                continue
            for f2 in group_feats[i+1:]:
                if f2 in features_to_drop:
                    continue
                if f1 in corr_matrix.index and f2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[f1, f2]
                    if abs(corr_val) > corr_threshold:
                        score1 = alive.loc[alive["feature"] == f1, "composite_score"].values
                        score2 = alive.loc[alive["feature"] == f2, "composite_score"].values
                        s1 = score1[0] if len(score1) > 0 else 0
                        s2 = score2[0] if len(score2) > 0 else 0
                        drop = f2 if s1 >= s2 else f1
                        keep = f1 if drop == f2 else f2
                        features_to_drop.add(drop)
                        removed_redundant.append({
                            "kept": keep, "dropped": drop,
                            "group": group_name,
                            "corr": round(corr_val, 3),
                        })
                        log.info("  Redundant: %s ↔ %s (corr=%.3f, group=%s) → drop %s",
                                 f1, f2, corr_val, group_name, drop)

    alive = alive[~alive["feature"].isin(features_to_drop)]

    # --- Step 3c: Group slot enforcement (three-pass) ---
    # Pass 0: forced features — domain priors that override pure scoring
    forced = FORCED_FEATURES.get(asset_class, [])
    selected_features = []
    for f in forced:
        if f in alive["feature"].values or f in agg["feature"].values:
            selected_features.append(f)
            log.info("  Forced: %s (%s)", f, FEATURE_TO_GROUP.get(f, "unknown"))

    # Pass 1: take min_slots from each group (guaranteed floor),
    # accounting for forced features already counted
    for group_name, slots in group_slots.items():
        min_slots = slots.get("min", 0)
        if min_slots == 0:
            continue

        already_in_group = sum(1 for f in selected_features if FEATURE_TO_GROUP.get(f) == group_name)
        need = min_slots - already_in_group
        if need <= 0:
            continue

        group_df = alive[
            (alive["group"] == group_name) & (~alive["feature"].isin(selected_features))
        ].sort_values("composite_score", ascending=False)

        if len(group_df) == 0 and need > 0:
            # Rescue from dead pool — pick best even if dead
            rescue_df = agg[
                (agg["group"] == group_name) & (~agg["feature"].isin(selected_features))
            ].sort_values("composite_score", ascending=False)
            n_rescue = min(need, len(rescue_df))
            for _, row in rescue_df.head(n_rescue).iterrows():
                selected_features.append(row["feature"])
                dead_features.discard(row["feature"])
                log.info("  Rescued %s from %s (min_slots=%d)", row["feature"], group_name, min_slots)
        else:
            n_take = min(need, len(group_df))
            for _, row in group_df.head(n_take).iterrows():
                selected_features.append(row["feature"])

    # Pass 2: fill remaining budget from best unselected, respecting max_slots
    remaining = target_count - len(selected_features)
    if remaining > 0:
        unselected = alive[~alive["feature"].isin(selected_features)].sort_values("composite_score", ascending=False)
        for _, row in unselected.iterrows():
            if remaining <= 0:
                break
            group = row["group"]
            max_for_group = group_slots.get(group, {}).get("max", 99)
            already_in_group = sum(1 for f in selected_features if FEATURE_TO_GROUP.get(f) == group)
            if already_in_group < max_for_group:
                selected_features.append(row["feature"])
                remaining -= 1

    log.info("Selected %d features (target=%d)", len(selected_features), target_count)

    # Build metadata
    alive_selected = agg[agg["feature"].isin(selected_features)].sort_values("composite_score", ascending=False)
    metadata = {
        "total_features_analyzed": len(agg),
        "features_selected": len(selected_features),
        "dead_features_removed": sorted(dead_features),
        "redundant_pairs_removed": removed_redundant,
        "scoring_formula": "0.4 * IC_score + 0.6 * IMP_score × domain_weight",
        "domain_weights_used": domain_weights,
        "group_slots_used": group_slots,
        "ic_summary": {},
    }
    for _, row in alive_selected.iterrows():
        metadata["ic_summary"][row["feature"]] = {
            "mean_abs_ic": round(row["mean_abs_ic"], 4),
            "mean_ic_ir": round(row["mean_ic_ir"], 4),
            "mean_dir_consistency": round(row["mean_dir_consistency"], 4),
            "mean_xgb_importance": round(row["mean_xgb_importance"], 4),
            "raw_score": round(row["raw_score"], 4),
            "domain_weight": round(row["domain_weight"], 2),
            "composite_score": round(row["composite_score"], 4),
            "group": row["group"],
        }

    return selected_features, metadata


# ---------------------------------------------------------------------------
# Step 4: Config Output
# ---------------------------------------------------------------------------

def build_config(
    selected_features: List[str],
    metadata: Dict,
    asset_class: str = "crypto_swing",
) -> Dict:
    """Build the feature config JSON structure."""
    if asset_class == "equity_swing":
        return _build_equity_config(selected_features, metadata)
    return _build_crypto_config(selected_features, metadata, asset_class)


def _build_crypto_config(
    selected_features: List[str],
    metadata: Dict,
    asset_class: str = "crypto_swing",
) -> Dict:
    """Build crypto-specific feature config."""
    # Separate core (non-crypto-regime) from crypto-regime features
    crypto_regime_feats = {
        "btc_sma200_flag", "btc_realized_vol_30", "btc_drawdown",
        "eth_btc_ratio_zscore", "btc_ret5", "btc_ret21",
        "btc_trend_strength", "btc_momentum_accel",
    }

    core = [f for f in selected_features if f not in crypto_regime_feats]
    crypto_add = [f for f in selected_features if f in crypto_regime_feats]

    # BTC-specific overrides: remove self-referential BTC regime features
    btc_remove = [f for f in crypto_add if f.startswith("btc_")]

    config = {
        "version": "3.0",
        "asset_class": asset_class,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "target_horizon": "10d",
        "design_notes": (
            f"v3: domain-weighted selection. Scoring = (0.4*IC + 0.6*XGB_imp) × domain_weight. "
            f"Asset class '{asset_class}' uses group weights and slot constraints "
            f"to enforce domain priors while letting data rank within groups."
        ),
        "core_features": core,
        "group_overrides": {
            "crypto_all": {
                "add": crypto_add,
                "remove": [],
            },
        },
        "per_coin_overrides": {
            "BTC-USD": {
                "add": [],
                "remove": btc_remove,
            },
        },
        "coin_groups": COIN_GROUPS,
        "factor_taxonomy": FACTOR_TAXONOMY,
        "selection_metadata": metadata,
    }

    samples_est = 188
    n_feats = len(selected_features)
    config["selection_metadata"]["samples_per_feature_ratio"] = round(samples_est / n_feats, 1) if n_feats > 0 else 0

    return config


def _build_equity_config(
    selected_features: List[str],
    metadata: Dict,
) -> Dict:
    """Build equity-specific feature config.

    All selected features go to core_features — XGBoost handles irrelevant
    features by not splitting on them. Per-category overrides can be added
    manually after reviewing IC results if desired.
    """
    # Crypto-specific features that should never be in equity config
    crypto_only = {
        "btc_sma200_flag", "btc_realized_vol_30", "btc_drawdown",
        "eth_btc_ratio_zscore", "btc_ret5", "btc_ret21",
        "btc_trend_strength", "btc_momentum_accel",
        "okx_long_short_ratio", "okx_ls_zscore",
        "fear_greed_index", "fear_greed_zscore",
        "polymarket_btc_bullish",
    }
    core = [f for f in selected_features if f not in crypto_only]

    config = {
        "version": "3.0",
        "asset_class": "equity_swing",
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "target_horizon": "10d",
        "design_notes": (
            "v3: domain-weighted selection. Scoring = (0.4*IC + 0.6*XGB_imp) × domain_weight. "
            "Asset class 'equity_swing' uses group weights and slot constraints "
            "to enforce domain priors while letting data rank within groups. "
            "Cross-asset features (FX, commodity, yield signals) selected by IC."
        ),
        "core_features": core,
        "group_overrides": {},
        "per_coin_overrides": {},
        "coin_groups": ETF_GROUPS,
        "factor_taxonomy": FACTOR_TAXONOMY,
        "selection_metadata": metadata,
    }

    samples_est = 500  # ETFs have ~2yr daily = ~500 training samples
    n_feats = len(core)
    config["selection_metadata"]["samples_per_feature_ratio"] = round(samples_est / n_feats, 1) if n_feats > 0 else 0

    return config


def print_summary(ic_df: pd.DataFrame, selected: List[str], metadata: Dict):
    """Print a summary table of the selection results."""
    if ic_df.empty:
        print("No data to summarize.")
        return

    agg = ic_df.groupby("feature").agg(
        mean_abs_ic=("full_ic", lambda x: np.abs(x).mean()),
        mean_ic_ir=("ic_ir", lambda x: np.abs(x).mean()),
        mean_dir_consistency=("direction_consistency", "mean"),
        mean_xgb_importance=("xgb_importance", "mean"),
    ).reset_index()
    agg["group"] = agg["feature"].map(FEATURE_TO_GROUP)
    agg["selected"] = agg["feature"].isin(selected)
    agg = agg.sort_values("mean_abs_ic", ascending=False)

    print("\n" + "=" * 110)
    print(f"{'Feature':<30} {'Group':<15} {'|IC|':>6} {'IC_IR':>6} {'DirCon':>6} {'XGBImp':>7} {'Decision':>10}")
    print("-" * 110)
    for _, row in agg.iterrows():
        decision = "KEEP" if row["selected"] else "DROP"
        print(f"{row['feature']:<30} {row['group']:<15} {row['mean_abs_ic']:>6.4f} "
              f"{row['mean_ic_ir']:>6.3f} {row['mean_dir_consistency']:>6.3f} "
              f"{row['mean_xgb_importance']:>7.4f} {decision:>10}")
    print("=" * 110)

    n_selected = len(selected)
    n_dead = len(metadata.get("dead_features_removed", []))
    n_redundant = len(metadata.get("redundant_pairs_removed", []))
    ratio = metadata.get("samples_per_feature_ratio", 0)
    print(f"\nSelected: {n_selected} features | Dead removed: {n_dead} | Redundant removed: {n_redundant}")
    print(f"Samples/feature ratio: ~{ratio}")

    print("\nSelected features by group:")
    groups = {}
    for f in selected:
        g = FEATURE_TO_GROUP.get(f, "unknown")
        groups.setdefault(g, []).append(f)
    for g, feats in sorted(groups.items()):
        print(f"  {g}: {', '.join(feats)}")

    print(f"\nRedundant pairs removed:")
    for pair in metadata.get("redundant_pairs_removed", []):
        print(f"  {pair['kept']} kept, {pair['dropped']} dropped (corr={pair['corr']}, group={pair['group']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature selection pipeline")
    parser.add_argument("--asset-class", default="crypto_swing", choices=["crypto_swing", "equity_swing"])
    parser.add_argument("--target-count", type=int, default=20, help="Target feature count")
    parser.add_argument("--dry-run", action="store_true", help="Analysis only, don't write config")
    parser.add_argument("--provider", default="yahoo", help="Data provider for IC analysis")
    parser.add_argument("--ic-threshold", type=float, default=0.02, help="Min |IC| to survive")
    parser.add_argument("--importance-threshold", type=float, default=0.005, help="Min XGB importance to survive")
    parser.add_argument("--corr-threshold", type=float, default=0.80, help="Max within-group correlation")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols for IC analysis (default: top coins from importance files)")
    args = parser.parse_args()

    if args.asset_class == "equity_swing":
        model_dir = PROJECT_ROOT / "models" / "swing"
    else:
        model_dir = PROJECT_ROOT / "models" / "crypto"
    config_dir = PROJECT_ROOT / "config"

    # Determine symbols for IC analysis
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.asset_class == "equity_swing":
        # Use symbols from etf_universe.json if available, else hardcoded priority
        importance_data = load_importance_data(model_dir)
        universe_path = model_dir / "etf_universe.json"
        if universe_path.exists():
            try:
                with open(universe_path) as f:
                    universe = json.load(f)
                all_etfs = [c["symbol"] for c in universe.get("coins", [])]
                # Prefer symbols with importance data, then fill from universe
                with_imp = [s for s in all_etfs if s in importance_data]
                without_imp = [s for s in all_etfs if s not in importance_data]
                symbols = with_imp + without_imp
                # Cap at 30 to keep runtime reasonable
                symbols = symbols[:30]
                log.info("Loaded %d ETF symbols from etf_universe.json (%d with importance data)",
                         len(symbols), len(with_imp))
            except (json.JSONDecodeError, KeyError):
                symbols = []
        if not symbols:
            # Hardcoded priority: diverse ETF set covering all categories
            priority = [
                "SPY", "QQQ", "IWM", "GLD", "SLV", "XLE", "XLF", "TLT",
                "EWJ", "EEM", "EWT", "SMH", "SOXX", "IGV", "GDX", "HYG",
                "VGK", "EWZ", "IBIT", "INDA",
            ]
            if importance_data:
                symbols = [s for s in priority if s in importance_data]
                if len(symbols) < 10:
                    symbols = list(importance_data.keys())[:20]
            else:
                symbols = priority
    else:
        # Crypto: use symbols that have importance files (already trained)
        importance_data = load_importance_data(model_dir)
        # Pick a representative subset (top coins by market cap + diversity)
        priority = [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
            "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "DOGE-USD",
            "LTC-USD", "NEAR-USD", "AAVE-USD", "CRV-USD", "RENDER-USD",
        ]
        symbols = [s for s in priority if s in importance_data]
        if len(symbols) < 5:
            # Fallback to all available
            symbols = list(importance_data.keys())[:20]

    log.info("Running IC analysis for %d symbols: %s", len(symbols), symbols)

    # Step 2: IC Analysis
    ic_df, corr_matrix = run_ic_analysis(symbols, model_dir, provider=args.provider)

    if ic_df.empty:
        log.error("IC analysis returned no data. Check data availability.")
        sys.exit(1)

    log.info("IC analysis complete: %d rows across %d symbols",
             len(ic_df), ic_df["symbol"].nunique())

    # Step 3: Feature Selection
    selected, metadata = select_features(
        ic_df, corr_matrix,
        asset_class=args.asset_class,
        target_count=args.target_count,
        ic_threshold=args.ic_threshold,
        importance_threshold=args.importance_threshold,
        corr_threshold=args.corr_threshold,
    )

    # Print summary
    print_summary(ic_df, selected, metadata)

    if args.dry_run:
        log.info("Dry run — not writing config file")
        return

    # Step 4: Write config
    config = build_config(selected, metadata, asset_class=args.asset_class)
    config_path = config_dir / f"features_{args.asset_class}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Config written to %s", config_path)
    log.info("Selected %d features (samples/feature ratio: %.1f)",
             len(selected), metadata.get("samples_per_feature_ratio", 0))


if __name__ == "__main__":
    main()
