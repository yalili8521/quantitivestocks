"""
Portfolio-level backtester: simulates a single portfolio with shared capital,
exposure limits, and portfolio-level metrics across multiple symbols.

Unlike per-symbol backtesting, this runs all symbols through one cash pool
so that exposure caps, sector limits, and cash management are realistic.

Usage (via main.py):
    python main.py backtest-portfolio --symbols SPY,QQQ,IWM --start 2024-01-01
    python main.py backtest-portfolio --group swing --start 2024-01-01
    python main.py backtest-portfolio --group swing --start 2024-01-01 --stress-cost-mult 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signals_engine import build_adapter, PROJECT_ROOT, compute_atr
from utils import DEFAULT_MODEL_DIR, SWING_MODEL_DIR, BACKTEST_DIR, COST_THRESHOLD, TARGET_RETURN, _fetch_vix_for_training
from risk_config import (
    get_risk_config,
    check_position_allowed,
    check_theme_cap,
    get_symbol_cap,
    SYMBOL_SECTOR,
    RiskConfig,
    validate_model_mode,
    get_effective_min_hold,
    classify_vol_tier,
    compute_vol_metrics,
    get_exit_params,
    VolTier,
    DeRiskState,
    evaluate_derisk,
)
from cost_model import get_symbol_costs, simulate_fill
from coin_selector import CoinSelector
from etf_selector import ETFSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("portfolio_backtester")

OUTPUT_DIR = BACKTEST_DIR


@dataclass
class PortfolioTrade:
    symbol: str
    entry_date: object
    entry_price: float
    direction: str
    size: float
    atr_at_entry: float = 0.0
    peak_price: float = 0.0
    tp_activated: bool = False
    breakeven_armed: bool = False
    signal_flip_count: int = 0
    bars_held: int = 0
    exit_date: object = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ""


@dataclass
class PortfolioState:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    positions: Dict[str, PortfolioTrade] = field(default_factory=dict)
    closed_trades: List[PortfolioTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


@dataclass
class PortfolioResult:
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_exposure: float
    max_exposure: float
    max_positions_held: int
    turnover: float
    trades_by_symbol: Dict[str, int]
    equity_curve: pd.DataFrame
    trades: List[PortfolioTrade] = field(default_factory=list)


class PortfolioBacktester:
    """Walk-forward multi-symbol backtester with shared capital and portfolio constraints."""

    _RANK_SCALAR_RANGE = {
        "swing": (0.50, 1.0),
        "intraday": (0.50, 1.0),
        "crypto": (0.25, 1.0),
        "crypto_intraday": (0.25, 1.0),
    }

    def __init__(
        self,
        symbols: List[str],
        adapter,
        fred_key: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        initial_capital: float = 100_000.0,
        group: str = "swing",
        model_type: str = "swing",
        stress_cost_mult: float = 1.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.symbols = symbols
        self.adapter = adapter
        self.fred_key = fred_key
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.group = group
        self.model_type = model_type
        self.stress_cost_mult = stress_cost_mult
        self.risk = risk_config or get_risk_config(group)
        if self.model_type == "intraday":
            validate_model_mode("lgb_intraday", "intraday")
        else:
            validate_model_mode("tft_swing", "daily")
        self._exit_params_by_symbol: Dict[str, object] = {}
        self._btc_correlations_by_symbol: Dict[str, float] = {}
        self._derisk_states: Dict[str, DeRiskState] = {}
        self._max_loss_exits: Dict[str, dict] = {}
        self._etf_selector = ETFSelector(top_k=10, min_pool=0) if not self._is_crypto_group() else None
        self._coin_selector = None
        if self._is_crypto_group():
            try:
                self._coin_selector = CoinSelector()
            except (FileNotFoundError, ImportError):
                self._coin_selector = None

    def _is_crypto_group(self) -> bool:
        return self.group in ("crypto", "crypto_intraday")

    def _calendar_symbol(self) -> str:
        """Return the market calendar anchor for this portfolio."""
        return "BTC-USD" if self._is_crypto_group() else "SPY"

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.replace("/", "-")

    @staticmethod
    def _hours_since(then, now) -> float:
        try:
            return max(0.0, (pd.Timestamp(now) - pd.Timestamp(then)).total_seconds() / 3600.0)
        except Exception:
            return float("inf")

    @staticmethod
    def _is_btc_anchor(symbol: str) -> bool:
        return PortfolioBacktester._normalize_symbol(symbol) in ("BTC-USD", "ETH-USD")

    def _compute_btc_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Compute a BTC-correlation penalty map for crypto sizing."""
        btc_key = None
        for sym in data:
            if self._normalize_symbol(sym) == "BTC-USD":
                btc_key = sym
                break
        if btc_key is None:
            return {}

        btc_df = data.get(btc_key)
        if btc_df is None or len(btc_df) < 35:
            return {}

        btc_ret = btc_df["close"].astype(float).pct_change().rename("btc")
        correlations: Dict[str, float] = {}
        for sym, df in data.items():
            if self._is_btc_anchor(sym):
                correlations[sym] = 0.0
                continue
            try:
                sym_ret = df["close"].astype(float).pct_change().rename("alt")
                aligned = pd.concat([btc_ret, sym_ret], axis=1, join="inner").dropna()
                if len(aligned) < 30:
                    correlations[sym] = 0.5
                    continue
                corr = aligned.iloc[-30:]["btc"].corr(aligned.iloc[-30:]["alt"])
                correlations[sym] = max(0.0, corr) if not np.isnan(corr) else 0.5
            except Exception:
                correlations[sym] = 0.5
        return correlations

    def _regime_scalar(
        self,
        day,
        spy_date_map: dict,
        spy_sma200_map: dict,
        vix_map: dict,
    ) -> float:
        """Apply soft regime scaling for ETF groups only."""
        if self._is_crypto_group():
            return 1.0

        spy_price = spy_date_map.get(day)
        spy_sma = spy_sma200_map.get(day)
        vix_val = vix_map.get(day, 20.0)
        adverse_regime = False
        if spy_price and spy_sma and not np.isnan(spy_sma):
            if spy_price <= spy_sma:
                adverse_regime = True
        if self.group != "intraday" and vix_val >= 30:
            adverse_regime = True
        return 0.5 if adverse_regime else 1.0

    def _rank_scalar_from_scores(self, symbol: str, scores: Dict[str, float]) -> float:
        if not scores:
            return 1.0

        floor, ceiling = self._RANK_SCALAR_RANGE.get(self.group, (0.50, 1.0))
        if symbol not in scores:
            return floor * 0.8

        vals = list(scores.values())
        max_score = max(vals) if vals else 1.0
        min_score = min(vals) if vals else 0.0
        score_range = max_score - min_score
        if score_range < 1e-6:
            return (floor + ceiling) / 2

        normalized = (scores[symbol] - min_score) / score_range
        return floor + (ceiling - floor) * normalized

    def _load_oos_registries(self) -> tuple[Dict[str, float], Dict[str, dict]]:
        config_path = os.path.join(PROJECT_ROOT, "config", "trading.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("oos_sharpe_registry", {}), cfg.get("oos_performance_registry", {})
        except Exception:
            return {}, {}

    def _compute_composite_scores(
        self,
        rankings: list,
        oos_sharpe: dict,
        oos_perf: dict,
        btc_correlations: Optional[Dict[str, float]] = None,
        w_selector: float = 0.3,
        w_sharpe: float = 0.4,
        w_hitrate: float = 0.3,
    ) -> Dict[str, float]:
        candidates = []
        for sym, sel_score in rankings:
            sharpe = oos_sharpe.get(sym, 0)
            perf = oos_perf.get(sym, {})
            hitrate = perf.get("win_rate", 0.5) if isinstance(perf, dict) else 0.5
            candidates.append((sym, sel_score, sharpe, hitrate))

        if not candidates:
            return {}

        def _rank_normalize(values):
            n = len(values)
            if n <= 1:
                return [1.0] * n
            order = sorted(range(n), key=lambda i: values[i])
            ranks = [0.0] * n
            for rank_pos, idx in enumerate(order):
                ranks[idx] = rank_pos / (n - 1)
            return ranks

        sel_norm = _rank_normalize([c[1] for c in candidates])
        sharpe_norm = _rank_normalize([c[2] for c in candidates])
        hitrate_norm = _rank_normalize([c[3] for c in candidates])

        result = {}
        for i, (sym, _sel, _sharpe, _hitrate) in enumerate(candidates):
            composite = (w_selector * sel_norm[i]
                         + w_sharpe * sharpe_norm[i]
                         + w_hitrate * hitrate_norm[i])
            if btc_correlations and sym in btc_correlations:
                composite *= (1 - 0.5 * btc_correlations[sym])
            result[sym] = round(composite, 4)
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def _rank_scalars_for_day(
        self,
        day,
        all_bars: Dict[str, pd.DataFrame],
        spy_bars: Optional[pd.DataFrame],
        predictors: Dict[str, object],
        vix_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute same-day rank scalars without lookahead."""
        if self._is_crypto_group():
            if self._coin_selector is None:
                return {}

            selector_data: Dict[str, pd.DataFrame] = {}
            symbol_map: Dict[str, str] = {}
            for sym, bars in all_bars.items():
                dates = pd.to_datetime(bars["ts"])
                mask = dates.dt.date <= day
                subset = bars.loc[mask, ["ts", "open", "high", "low", "close", "volume"]].copy()
                if len(subset) < 60:
                    continue
                norm_sym = self._normalize_symbol(sym)
                selector_data[norm_sym] = pd.DataFrame({
                    "date": pd.to_datetime(subset["ts"]).dt.tz_localize(None),
                    "open": subset["open"].astype(float).values,
                    "high": subset["high"].astype(float).values,
                    "low": subset["low"].astype(float).values,
                    "close": subset["close"].astype(float).values,
                    "volume": subset["volume"].astype(float).values,
                })
                symbol_map[norm_sym] = sym

            result = self._coin_selector.rank(selector_data) if len(selector_data) >= 2 else None
            if result is None or not result.rankings:
                return {}

            btc_corr = self._compute_btc_correlations(selector_data)
            oos_sharpe, oos_perf = self._load_oos_registries()
            composite = self._compute_composite_scores(
                result.rankings,
                oos_sharpe,
                oos_perf,
                btc_correlations=btc_corr,
            )

            if composite:
                for norm_sym in list(composite.keys()):
                    original_sym = symbol_map.get(norm_sym)
                    predictor = predictors.get(original_sym) if original_sym else None
                    if predictor is None:
                        continue
                    bars_df = selector_data.get(norm_sym)
                    if bars_df is None:
                        continue
                    try:
                        pred = predictor.predict(bars_df, vix_df)
                        if pred and "confidence" in pred:
                            composite[norm_sym] += 0.10 * min(1.0, abs(float(pred["confidence"])))
                    except Exception:
                        pass
                composite = dict(sorted(composite.items(), key=lambda x: -x[1]))

            return {
                symbol_map[norm_sym]: self._rank_scalar_from_scores(norm_sym, composite)
                for norm_sym in composite
                if norm_sym in symbol_map
            }

        if self._etf_selector is None:
            return {}

        closes: Dict[str, pd.Series] = {}
        for sym, bars in all_bars.items():
            dates = pd.to_datetime(bars["ts"])
            mask = dates.dt.date <= day
            subset = bars.loc[mask, ["ts", "close"]].copy()
            if subset.empty:
                continue
            closes[sym] = pd.Series(
                subset["close"].astype(float).values,
                index=pd.to_datetime(subset["ts"]),
                name=sym,
            )

        if spy_bars is not None:
            spy_dates = pd.to_datetime(spy_bars["ts"])
            spy_mask = spy_dates.dt.date <= day
            spy_subset = spy_bars.loc[spy_mask, ["ts", "close"]].copy()
            if not spy_subset.empty:
                closes["SPY"] = pd.Series(
                    spy_subset["close"].astype(float).values,
                    index=pd.to_datetime(spy_subset["ts"]),
                    name="SPY",
                )

        ranking = self._etf_selector.rank(list(self.symbols), closes=closes)
        if ranking.empty or "score" not in ranking.columns:
            return {}

        score_map = dict(zip(ranking["symbol"], ranking["score"]))
        return {
            sym: self._rank_scalar_from_scores(sym, score_map)
            for sym in self.symbols
        }

    def _classify_exit_params(self, bars: pd.DataFrame):
        """Pick the configured exit params for this symbol's vol regime."""
        try:
            metrics_df = pd.DataFrame({
                "High": bars["high"].astype(float),
                "Low": bars["low"].astype(float),
                "Close": bars["close"].astype(float),
            })
            atr_ratio, vol20 = compute_vol_metrics(metrics_df)
            tier = classify_vol_tier(atr_ratio, vol20)
        except Exception:
            tier = VolTier.HIGH
        return get_exit_params(self.group, tier)

    def run(self, start_date: str, end_date: Optional[str] = None) -> PortfolioResult:
        """Run portfolio-level backtest across all symbols."""
        log.info("=== Portfolio Backtest: %s ===", ", ".join(self.symbols))
        log.info("Group: %s | Model: %s | Capital: $%,.0f | Stress: %.1fx",
                 self.group, self.model_type, self.initial_capital, self.stress_cost_mult)
        log.info("Risk: pos=%.0f%% max_pos=%.0f%% max_sector=%.0f%% max_exp=%.0f%%",
                 self.risk.position_pct * 100, self.risk.max_position_pct * 100,
                 self.risk.max_sector_pct * 100, self.risk.max_total_exposure * 100)

        # Load predictors for all symbols
        predictors = {}
        for sym in self.symbols:
            predictor = self._create_predictor(sym)
            if predictor is not None:
                predictors[sym] = predictor
                log.info("Loaded model for %s", sym)
            else:
                log.warning("No model for %s — skipping", sym)

        if not predictors:
            log.error("No models loaded. Nothing to backtest.")
            return self._empty_result(start_date, end_date)

        # Fetch all data
        all_bars = {}
        for sym in predictors:
            bars = self.adapter.fetch_daily(sym, 1200)
            all_bars[sym] = bars
            self._exit_params_by_symbol[sym] = self._classify_exit_params(bars)
            log.info("Fetched %d bars for %s", len(bars), sym)
        if self._is_crypto_group():
            self._btc_correlations_by_symbol = self._compute_btc_correlations(all_bars)

        calendar_symbol = self._calendar_symbol()
        calendar_bars = self.adapter.fetch_daily(calendar_symbol, 1200)
        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=1200)

        # Find common date range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        # Build the trade calendar from the relevant market anchor.
        calendar_dates = pd.to_datetime(calendar_bars["ts"]).dt.date
        trade_dates = sorted(set(d for d in calendar_dates if start_ts.date() <= d <= end_ts.date()))
        log.info("Trading dates: %d (from %s to %s)", len(trade_dates), trade_dates[0], trade_dates[-1])

        # Initialize portfolio
        portfolio = PortfolioState(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )

        # Regime filter: ETF groups use SPY + VIX soft scaling; crypto groups do not.
        if self._is_crypto_group():
            spy_date_map = {}
            spy_sma200_map = {}
        else:
            spy_bars = self.adapter.fetch_daily("SPY", 1200)
            spy_dates = pd.to_datetime(spy_bars["ts"]).dt.date
            spy_close = spy_bars["close"].astype(float)
            spy_sma200 = spy_close.rolling(200).mean()
            spy_date_map = dict(zip(spy_dates.values, spy_close.values))
            spy_sma200_map = dict(zip(spy_dates.values, spy_sma200.values))

        vix_map = {}
        if not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_map[d] = row["vix"]

        exposure_log = []
        max_positions_held = 0

        for day in trade_dates:
            equity = self._compute_equity(portfolio, all_bars, day)

            regime_scalar = self._regime_scalar(day, spy_date_map, spy_sma200_map, vix_map)
            rank_scalars = self._rank_scalars_for_day(
                day,
                all_bars,
                None if self._is_crypto_group() else spy_bars,
                predictors,
                vix_df,
            )

            # Check exits for all open positions
            for sym in list(portfolio.positions.keys()):
                pos = portfolio.positions[sym]
                bars = all_bars.get(sym)
                if bars is None:
                    continue
                bar_dates = pd.to_datetime(bars["ts"]).dt.date
                day_mask = bar_dates == day
                if not day_mask.any():
                    continue
                idx = day_mask.values.nonzero()[0][-1]
                current_price = float(bars["close"].iloc[idx])
                ep = self._exit_params_by_symbol.get(sym) or get_exit_params(self.group, VolTier.HIGH)
                high = bars["high"].astype(float)
                low = bars["low"].astype(float)
                close_series = bars["close"].astype(float)
                atr_s = compute_atr(high, low, close_series, period=14)
                bar_atr = float(atr_s.iloc[idx]) if not np.isnan(float(atr_s.iloc[idx])) else 0.0

                # Get prediction
                pred = self._predict(predictors.get(sym), bars, vix_df, idx)
                expected_return = pred.get("expected_return", 0.0)

                # Disaster stop
                entry_atr = pos.atr_at_entry
                entry_price = pos.entry_price
                if pos.direction == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                    pos.peak_price = max(pos.peak_price, current_price) if pos.peak_price > 0 else current_price
                    drawdown_from_peak = ((pos.peak_price - current_price) / pos.peak_price
                                          if pos.peak_price > 0 else 0.0)
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                    pos.peak_price = min(pos.peak_price, current_price) if pos.peak_price > 0 else current_price
                    drawdown_from_peak = ((current_price - pos.peak_price) / pos.peak_price
                                          if pos.peak_price > 0 else 0.0)

                if ep.use_atr and entry_atr > 0 and entry_price > 0:
                    disaster_pct = min(
                        ep.disaster_stop_pct,
                        ep.disaster_atr_mult * entry_atr / entry_price,
                    )
                else:
                    disaster_pct = ep.disaster_stop_pct

                if pnl_pct <= -disaster_pct:
                    self._close_position(portfolio, sym, day, current_price, "disaster_stop")
                    continue

                # Breakeven ratchet
                if pnl_pct >= ep.breakeven_ratchet_pct:
                    pos.breakeven_armed = True
                if pos.breakeven_armed and pnl_pct <= 0:
                    self._close_position(portfolio, sym, day, current_price, "breakeven_ratchet")
                    continue

                # Profit-lock arm and trail
                if ep.use_atr and entry_atr > 0 and entry_price > 0:
                    arm_pct = min(
                        ep.profit_lock_arm_pct,
                        ep.arm_atr_mult * entry_atr / entry_price,
                    )
                    trail_pct = min(
                        ep.profit_lock_trail_pct,
                        ep.trail_atr_mult * entry_atr / entry_price,
                    )
                else:
                    arm_pct = ep.profit_lock_arm_pct
                    trail_pct = ep.profit_lock_trail_pct

                if pnl_pct >= arm_pct:
                    pos.tp_activated = True

                # Signal decay exit respects ATR-adaptive minimum hold.
                next_bars_held = pos.bars_held + 1
                signal_flipped = (
                    (pos.direction == "LONG" and expected_return <= 0)
                    or (pos.direction == "SHORT" and expected_return >= 0)
                )
                min_hold = get_effective_min_hold(
                    ep.min_hold_bars,
                    pos.atr_at_entry,
                    bar_atr,
                )
                if pos.tp_activated:
                    if signal_flipped and pnl_pct > 0 and next_bars_held >= min_hold:
                        self._close_position(portfolio, sym, day, current_price, "profit_lock+signal_decay")
                        continue
                    if drawdown_from_peak >= trail_pct:
                        self._close_position(portfolio, sym, day, current_price, "profit_lock_trail")
                        continue

                pos.bars_held = next_bars_held
                if signal_flipped:
                    pos.signal_flip_count += 1
                else:
                    pos.signal_flip_count = 0

                if pos.bars_held >= min_hold and pos.signal_flip_count >= ep.signal_flip_consecutive:
                    self._close_position(portfolio, sym, day, current_price, "signal_decay")
                    continue

                if ep.max_underwater_days > 0 and pnl_pct < 0:
                    try:
                        days_held = (day - pos.entry_date).days
                    except TypeError:
                        days_held = 0
                    if days_held >= ep.max_underwater_days:
                        self._close_position(
                            portfolio, sym, day, current_price,
                            f"max_underwater_{ep.max_underwater_days}d"
                        )
                        continue

            # Check entries
            ordered_symbols = sorted(
                predictors.keys(),
                key=lambda sym: rank_scalars.get(sym, 1.0),
                reverse=True,
            )
            for sym in ordered_symbols:
                if sym in portfolio.positions:
                    continue  # already in position
                bars = all_bars.get(sym)
                if bars is None:
                    continue
                bar_dates = pd.to_datetime(bars["ts"]).dt.date
                day_mask = bar_dates == day
                if not day_mask.any():
                    continue
                idx = day_mask.values.nonzero()[0][-1]
                current_price = float(bars["close"].iloc[idx])

                pred = self._predict(predictors.get(sym), bars, vix_df, idx)
                expected_return = pred.get("expected_return", 0.0)

                enter_dir = None
                cost_threshold = max(COST_THRESHOLD, get_symbol_costs(sym).round_trip_pct * 1.5)
                if expected_return > cost_threshold:
                    enter_dir = "LONG"
                elif expected_return < -cost_threshold:
                    enter_dir = "SHORT"

                if enter_dir is None:
                    continue

                last_loss = self._max_loss_exits.get(sym)
                if last_loss is not None:
                    loss_age_h = self._hours_since(last_loss["time"], day)
                    if loss_age_h < self.risk.max_loss_cooldown_hours:
                        continue
                    if loss_age_h < 24 and enter_dir == last_loss["direction"]:
                        effective_threshold = cost_threshold * self.risk.same_dir_confidence_mult
                        if abs(expected_return) < effective_threshold:
                            continue

                # Position sizing with soft regime scaling, not a hard gate.
                signal_pct = min(1.0, max(self.risk.min_signal_scale, abs(expected_return) / TARGET_RETURN))
                derisk_key = self._normalize_symbol(sym)
                derisk_state = self._derisk_states.get(derisk_key)
                kelly_f = None
                if derisk_state is not None:
                    kelly_f = derisk_state.half_kelly(
                        window=self.risk.kelly_window,
                        min_trades=self.risk.kelly_min_trades,
                    )
                if kelly_f is not None:
                    effective_cap = self.risk.kelly_cap * self.risk.cross_group_kelly_discount
                    base_frac = min(
                        kelly_f * self.risk.cross_group_kelly_discount,
                        effective_cap,
                    )
                else:
                    base_frac = self.risk.position_pct

                sizing_pct = base_frac * signal_pct * regime_scalar
                sizing_pct = min(sizing_pct, self.risk.max_position_pct)
                sizing_pct = min(sizing_pct, get_symbol_cap(sym, self.group))
                if self._is_crypto_group() and not self._is_btc_anchor(sym):
                    btc_open = any(self._normalize_symbol(open_sym) == "BTC-USD"
                                   for open_sym in portfolio.positions)
                    if btc_open:
                        btc_corr = self._btc_correlations_by_symbol.get(sym, 0.5)
                        sizing_pct *= (1 - 0.5 * btc_corr)
                if derisk_state is not None:
                    derisk_action, _ = evaluate_derisk(derisk_state, window=50)
                    if derisk_action == "disable":
                        continue
                    if derisk_action == "halfsize":
                        sizing_pct *= 0.5
                if last_loss is not None:
                    loss_age_h = self._hours_since(last_loss["time"], day)
                    if loss_age_h < self.risk.post_loss_size_hours:
                        sizing_pct *= self.risk.post_loss_size_mult
                if sizing_pct <= 0:
                    continue

                invest = equity * sizing_pct

                # Portfolio constraint check
                pos_dict = self._positions_as_dict(portfolio, all_bars, day)
                allowed, reason = check_position_allowed(
                    sym, invest, equity, pos_dict, self.risk
                )
                if not allowed:
                    continue
                theme_ok, _theme_reason = check_theme_cap(
                    sym, invest, equity, pos_dict
                )
                if not theme_ok:
                    continue

                # ATR for stops
                high = bars["high"].astype(float)
                low = bars["low"].astype(float)
                atr_s = compute_atr(high, low, close_series, period=14)
                bar_atr = float(atr_s.iloc[idx]) if not np.isnan(float(atr_s.iloc[idx])) else 0.0

                # Apply cost model
                fill_price, filled = simulate_fill(
                    sym, current_price, enter_dir,
                    stress_mult=self.stress_cost_mult,
                )
                if not filled:
                    continue

                shares = invest / fill_price
                portfolio.cash -= invest
                portfolio.positions[sym] = PortfolioTrade(
                    symbol=sym, entry_date=day, entry_price=fill_price,
                    direction=enter_dir, size=shares, atr_at_entry=bar_atr,
                    peak_price=fill_price,
                )

            # Record equity
            equity = self._compute_equity(portfolio, all_bars, day)
            n_positions = len(portfolio.positions)
            max_positions_held = max(max_positions_held, n_positions)
            total_invested = sum(
                pos.size * self._get_price(all_bars, pos.symbol, day)
                for pos in portfolio.positions.values()
            )
            exposure_pct = total_invested / equity if equity > 0 else 0
            exposure_log.append(exposure_pct)
            portfolio.equity_curve.append({"date": day, "equity": equity, "positions": n_positions})

        # Close remaining positions at end
        for sym in list(portfolio.positions.keys()):
            bars = all_bars.get(sym)
            if bars is not None:
                last_price = float(bars["close"].iloc[-1])
            else:
                last_price = portfolio.positions[sym].entry_price
            self._close_position(portfolio, sym, trade_dates[-1], last_price, "end_of_backtest")

        return self._compute_results(portfolio, trade_dates, exposure_log, max_positions_held)

    def _create_predictor(self, symbol: str):
        """Create predictor based on model type."""
        if self.model_type == "swing":
            try:
                from swing_model import SwingPredictor
                return SwingPredictor(symbol, model_dir=self.model_dir)
            except (FileNotFoundError, ImportError):
                pass
        try:
            from ml_model import Predictor
            return Predictor(symbol, model_dir=self.model_dir)
        except (FileNotFoundError, RuntimeError):
            return None

    def _predict(self, predictor, bars_df, vix_df, up_to_idx: int) -> dict:
        """Run prediction using data up to (and including) up_to_idx."""
        if predictor is None:
            return {"expected_return": 0.0, "direction": "FLAT"}
        try:
            # Use only bars up to current date to prevent lookahead
            bars_subset = bars_df.iloc[:up_to_idx + 1].copy()
            return predictor.predict(bars_subset, vix_df)
        except Exception:
            return {"expected_return": 0.0, "direction": "FLAT"}

    def _close_position(self, portfolio: PortfolioState, symbol: str,
                        date, price: float, reason: str) -> None:
        pos = portfolio.positions.pop(symbol)
        exit_side = "SELL" if pos.direction == "LONG" else "BUY"
        fill_price, _ = simulate_fill(symbol, price, exit_side, stress_mult=self.stress_cost_mult)
        pos.exit_date = date
        pos.exit_price = fill_price
        pos.exit_reason = reason
        if pos.direction == "LONG":
            pos.pnl = pos.size * (fill_price - pos.entry_price)
            portfolio.cash += pos.size * fill_price
        else:
            pos.pnl = pos.size * (pos.entry_price - fill_price)
            portfolio.cash += pos.size * pos.entry_price + pos.pnl
        portfolio.closed_trades.append(pos)
        if pos.entry_price > 0:
            pnl_pct = (pos.pnl / (pos.size * pos.entry_price)) if pos.size > 0 else 0.0
            derisk_key = self._normalize_symbol(symbol)
            if derisk_key not in self._derisk_states:
                self._derisk_states[derisk_key] = DeRiskState()
            self._derisk_states[derisk_key].record_trade(float(pnl_pct))
        if reason == "disaster_stop":
            self._max_loss_exits[symbol] = {"time": date, "direction": pos.direction}

    def _compute_equity(self, portfolio: PortfolioState, all_bars: dict, day) -> float:
        equity = portfolio.cash
        for sym, pos in portfolio.positions.items():
            price = self._get_price(all_bars, sym, day)
            if pos.direction == "LONG":
                equity += pos.size * price
            else:
                equity += pos.size * pos.entry_price + pos.size * (pos.entry_price - price)
        return equity

    def _get_price(self, all_bars: dict, symbol: str, day) -> float:
        bars = all_bars.get(symbol)
        if bars is None:
            return 0.0
        bar_dates = pd.to_datetime(bars["ts"]).dt.date
        day_mask = bar_dates == day
        if not day_mask.any():
            # Use most recent available price
            prior = bar_dates <= day
            if prior.any():
                return float(bars["close"][prior].iloc[-1])
            return 0.0
        return float(bars["close"][day_mask].iloc[-1])

    def _positions_as_dict(self, portfolio: PortfolioState, all_bars: dict, day) -> dict:
        """Convert positions to dict format expected by check_position_allowed."""
        result = {}
        for sym, pos in portfolio.positions.items():
            price = self._get_price(all_bars, sym, day)
            result[sym] = {
                "qty": pos.size,
                "current_price": price,
                "side": pos.direction,
            }
        return result

    def _compute_results(self, portfolio: PortfolioState, trade_dates: list,
                         exposure_log: list, max_positions_held: int) -> PortfolioResult:
        eq_df = pd.DataFrame(portfolio.equity_curve)
        final_equity = eq_df["equity"].iloc[-1] if not eq_df.empty else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100

        # Annualized return
        years = len(trade_dates) / 252
        ann_return = ((final_equity / self.initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

        # Sharpe
        daily_returns = eq_df["equity"].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                  if daily_returns.std() > 0 else 0.0)

        # Max drawdown
        peak = eq_df["equity"].cummax()
        drawdown = (eq_df["equity"] - peak) / peak
        max_dd = abs(drawdown.min()) * 100

        # Trade stats
        trades = portfolio.closed_trades
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

        # Trades by symbol
        trades_by_sym = {}
        for t in trades:
            trades_by_sym[t.symbol] = trades_by_sym.get(t.symbol, 0) + 1

        # Exposure stats
        avg_exposure = np.mean(exposure_log) if exposure_log else 0
        max_exposure = max(exposure_log) if exposure_log else 0

        # Turnover
        total_volume = sum(abs(t.size * t.entry_price) for t in trades)
        turnover = total_volume / (self.initial_capital * max(years, 0.01))

        result = PortfolioResult(
            symbols=self.symbols,
            start_date=str(trade_dates[0]),
            end_date=str(trade_dates[-1]),
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(ann_return, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            total_trades=n_trades,
            win_rate=round(win_rate, 3),
            profit_factor=round(profit_factor, 2),
            avg_exposure=round(avg_exposure, 3),
            max_exposure=round(max_exposure, 3),
            max_positions_held=max_positions_held,
            turnover=round(turnover, 1),
            trades_by_symbol=trades_by_sym,
            equity_curve=eq_df,
            trades=trades,
        )
        return result

    def _empty_result(self, start_date, end_date) -> PortfolioResult:
        return PortfolioResult(
            symbols=self.symbols, start_date=start_date or "",
            end_date=end_date or "", initial_capital=self.initial_capital,
            final_equity=self.initial_capital, total_return_pct=0,
            annualized_return_pct=0, sharpe_ratio=0, max_drawdown_pct=0,
            total_trades=0, win_rate=0, profit_factor=0,
            avg_exposure=0, max_exposure=0, max_positions_held=0,
            turnover=0, trades_by_symbol={}, equity_curve=pd.DataFrame(),
        )


def print_portfolio_report(result: PortfolioResult) -> None:
    """Print portfolio backtest results."""
    print("\n" + "=" * 70)
    print("  PORTFOLIO BACKTEST REPORT")
    print("=" * 70)
    print(f"  Symbols:     {', '.join(result.symbols)}")
    print(f"  Period:      {result.start_date} to {result.end_date}")
    print(f"  Capital:     ${result.initial_capital:,.0f} → ${result.final_equity:,.0f}")
    print(f"  Return:      {result.total_return_pct:+.2f}% (ann: {result.annualized_return_pct:+.2f}%)")
    print(f"  Sharpe:      {result.sharpe_ratio:.3f}")
    print(f"  Max DD:      {result.max_drawdown_pct:.2f}%")
    print(f"  Trades:      {result.total_trades} (WR: {result.win_rate:.1%}, PF: {result.profit_factor:.2f})")
    print(f"  Exposure:    avg={result.avg_exposure:.1%}, max={result.max_exposure:.1%}")
    print(f"  Max pos:     {result.max_positions_held}")
    print(f"  Turnover:    {result.turnover:.1f}x/yr")
    print()
    if result.trades_by_symbol:
        print("  Trades by symbol:")
        for sym, n in sorted(result.trades_by_symbol.items(), key=lambda x: -x[1]):
            print(f"    {sym:>8}: {n}")
    print("=" * 70)

    # Save summary JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "symbols": result.symbols,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "total_return_pct": result.total_return_pct,
        "annualized_return_pct": result.annualized_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "avg_exposure": result.avg_exposure,
        "max_exposure": result.max_exposure,
        "trades_by_symbol": result.trades_by_symbol,
    }
    path = os.path.join(OUTPUT_DIR, "portfolio_backtest_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {path}")


def main() -> None:
    from paper_trader import SYMBOL_GROUPS

    parser = argparse.ArgumentParser(description="Portfolio-level multi-symbol backtester.")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (overrides --group)")
    parser.add_argument("--group", type=str, default="swing",
                        choices=list(SYMBOL_GROUPS.keys()),
                        help="Symbol group (default: swing)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--model", default="swing", choices=["lstm", "swing"])
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--stress-cost-mult", type=float, default=1.0,
                        help="Cost multiplier for stress testing (default: 1.0)")

    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = SYMBOL_GROUPS.get(args.group, [])

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    model_dir = args.model_dir or DEFAULT_MODEL_DIR

    bt = PortfolioBacktester(
        symbols=symbols,
        adapter=adapter,
        fred_key=fred_key,
        model_dir=model_dir,
        initial_capital=args.capital,
        group=args.group,
        model_type=args.model,
        stress_cost_mult=args.stress_cost_mult,
    )

    result = bt.run(start_date=args.start, end_date=args.end)
    print_portfolio_report(result)


if __name__ == "__main__":
    main()
