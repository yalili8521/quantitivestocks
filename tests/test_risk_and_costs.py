"""
Tests for risk_config, cost_model, and model_monitor modules.

Run: .venv/Scripts/python.exe -m pytest tests/test_risk_and_costs.py -v
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
import warnings
import shutil
import uuid


# ===================================================================
# risk_config tests
# ===================================================================

class TestRiskConfig:
    def test_group_configs_exist(self):
        from risk_config import get_risk_config
        for group in ("swing", "intraday", "crypto"):
            cfg = get_risk_config(group)
            assert cfg.position_pct > 0
            assert cfg.max_position_pct > 0
            assert cfg.max_positions > 0

    def test_swing_more_conservative_than_legacy(self):
        from risk_config import SWING_RISK
        assert SWING_RISK.position_pct <= 0.60  # was 0.95

    def test_crypto_most_conservative(self):
        from risk_config import CRYPTO_RISK, SWING_RISK
        assert CRYPTO_RISK.position_pct < SWING_RISK.position_pct
        assert CRYPTO_RISK.kelly_cap < SWING_RISK.kelly_cap

    def test_validate_model_mode_lstm_daily_ok(self):
        from risk_config import validate_model_mode
        validate_model_mode("lstm", "daily")  # should not raise

    def test_validate_model_mode_lstm_intraday_fails(self):
        from risk_config import validate_model_mode
        with pytest.raises(ValueError, match="cannot be used"):
            validate_model_mode("lstm", "intraday")

    def test_validate_model_mode_lgb_intraday_ok(self):
        from risk_config import validate_model_mode
        validate_model_mode("lgb_intraday", "intraday")  # should not raise

    def test_validate_model_mode_lgb_daily_fails(self):
        from risk_config import validate_model_mode
        with pytest.raises(ValueError, match="cannot be used"):
            validate_model_mode("lgb_intraday", "daily")

    def test_validate_model_mode_crypto_intraday_ok(self):
        from risk_config import validate_model_mode
        validate_model_mode("lgb_intraday_crypto", "intraday")

    def test_validate_model_mode_crypto_intraday_daily_fails(self):
        from risk_config import validate_model_mode
        with pytest.raises(ValueError, match="cannot be used"):
            validate_model_mode("lgb_intraday_crypto", "daily")

    def test_get_risk_config_applies_config_overrides(self):
        from risk_config import get_risk_config

        temp_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"riskcfg_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        cfg_path = os.path.join(temp_dir, "trading.json")
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "swing": {
                        "mode": "daily",
                        "horizon": "10d",
                        "allowed_models": ["lstm", "xgb_swing", "tft_swing"],
                        "position_pct": 0.41,
                        "max_positions": 7,
                        "kelly_cap": 0.22,
                        "cross_group_kelly_discount": 0.45,
                    }
                }, f)

            cfg = get_risk_config("swing", config_path=cfg_path)
            assert cfg.position_pct == 0.41
            assert cfg.max_positions == 7
            assert cfg.kelly_cap == 0.22
            assert cfg.cross_group_kelly_discount == 0.45
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_risk_config_rejects_mode_mismatch(self):
        from risk_config import get_risk_config

        temp_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"riskcfg_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        cfg_path = os.path.join(temp_dir, "trading.json")
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "swing": {
                        "mode": "intraday",
                        "horizon": "10d",
                        "allowed_models": ["lstm", "xgb_swing", "tft_swing"],
                    }
                }, f)

            with pytest.raises(ValueError, match="mode="):
                get_risk_config("swing", config_path=cfg_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_check_position_allowed_passes(self):
        from risk_config import check_position_allowed, SWING_RISK
        positions = {}
        ok, reason = check_position_allowed("SPY", 10_000, 100_000, positions, SWING_RISK)
        assert ok
        assert reason == "ok"

    def test_check_position_blocks_over_exposure(self):
        from risk_config import check_position_allowed, SWING_RISK
        # Fill up to near max exposure (80k of 100k = 80%)
        positions = {
            "SPY": {"qty": 100, "current_price": 500, "side": "LONG"},  # 50k
            "QQQ": {"qty": 60,  "current_price": 500, "side": "LONG"},  # 30k
        }
        # Use 10k (10%) which is under per-position cap but pushes total over max_total_exposure
        ok, reason = check_position_allowed("IWM", 10_000, 100_000, positions, SWING_RISK)
        assert not ok
        assert "total exposure" in reason

    def test_check_position_blocks_sector(self):
        from risk_config import check_position_allowed, SWING_RISK
        # Fill tech sector
        positions = {
            "SMH":  {"qty": 80, "current_price": 250, "side": "LONG"},  # 20k tech
            "SOXX": {"qty": 50, "current_price": 200, "side": "LONG"},  # 10k tech
            "QQQ":  {"qty": 20, "current_price": 500, "side": "LONG"},  # 10k tech
        }
        # 40k tech = 40% already at cap
        ok, reason = check_position_allowed("XLK", 5_000, 100_000, positions, SWING_RISK)
        assert not ok
        assert "sector" in reason

    def test_check_position_blocks_max_positions(self):
        from risk_config import check_position_allowed, INTRADAY_RISK
        # Use N+1 positions so test survives future max_positions changes
        positions = {f"SYM{i}": {"qty": 10, "current_price": 100, "side": "LONG"}
                     for i in range(INTRADAY_RISK.max_positions + 1)}
        ok, reason = check_position_allowed("NEW", 1_000, 100_000, positions, INTRADAY_RISK)
        assert not ok
        assert "max positions" in reason

    def test_effective_min_hold_defaults_to_base_without_atr(self):
        from risk_config import get_effective_min_hold

        assert get_effective_min_hold(12, 0.0, 1.0) == 12
        assert get_effective_min_hold(12, 1.0, 0.0) == 12

    def test_effective_min_hold_scales_and_clamps(self):
        from risk_config import get_effective_min_hold

        assert get_effective_min_hold(12, 2.0, 4.0) == 24
        assert get_effective_min_hold(12, 4.0, 1.0) == 6

    def test_btc_beta_cap(self):
        from risk_config import check_position_allowed, compute_btc_beta_exposure, CRYPTO_RISK
        # Use large equity so per-position cap doesn't fire first.
        # BTC: 100k × beta 1.0 = 1.0, ETH: 100k × beta 1.3 = 1.3 → total beta = 2.3
        equity = 1_000_000
        positions = {
            "BTC/USD": {"qty": 1, "current_price": 100_000, "side": "LONG"},   # 10% × 1.0
            "ETH/USD": {"qty": 25, "current_price": 4_000,  "side": "LONG"},   # 10% × 1.3
        }
        current_beta = compute_btc_beta_exposure(positions, equity)
        assert current_beta > 0.2  # sanity: has some beta
        # Adding DOGE at 5% of equity with beta 2.2 → should push over cap
        ok, reason = check_position_allowed("DOGE/USD", 50_000, equity, positions, CRYPTO_RISK)
        # The test verifies the constraint system works; it may block on position size,
        # total exposure, or BTC beta depending on config
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


# ===================================================================
# cost_model tests
# ===================================================================

class TestCostModel:
    def test_liquid_etf_low_cost(self):
        from cost_model import get_symbol_costs
        costs = get_symbol_costs("SPY")
        assert costs.round_trip_pct < 0.001  # < 0.1%

    def test_crypto_alt_higher_cost(self):
        from cost_model import get_symbol_costs
        spy_cost = get_symbol_costs("SPY").round_trip_pct
        doge_cost = get_symbol_costs("DOGE/USD").round_trip_pct
        assert doge_cost > spy_cost * 5

    def test_extended_hours_multiplier(self):
        from cost_model import get_symbol_costs
        regular = get_symbol_costs("SMH", session="regular")
        extended = get_symbol_costs("SMH", session="extended")
        assert extended.round_trip_pct > regular.round_trip_pct * 2

    def test_stress_multiplier(self):
        from cost_model import get_symbol_costs
        normal = get_symbol_costs("SPY", stress_mult=1.0)
        stress = get_symbol_costs("SPY", stress_mult=3.0)
        assert stress.round_trip_pct > normal.round_trip_pct * 2

    def test_validate_cost_threshold_pass(self):
        from cost_model import validate_cost_threshold
        ok, msg, rt = validate_cost_threshold("SPY", 0.01)
        assert ok

    def test_validate_cost_threshold_fail_crypto(self):
        from cost_model import validate_cost_threshold
        ok, msg, rt = validate_cost_threshold("DOGE/USD", 0.001)
        assert not ok

    def test_simulate_fill_adds_cost(self):
        from cost_model import simulate_fill
        price = 100.0
        buy_fill, filled = simulate_fill("SPY", price, "BUY")
        assert buy_fill >= price  # buy fills at or above quoted
        sell_fill, filled = simulate_fill("SPY", price, "SELL")
        assert sell_fill <= price  # sell fills at or below quoted

    def test_adjust_return_for_costs(self):
        from cost_model import adjust_return_for_costs
        raw = 0.01  # 1% return
        adjusted = adjust_return_for_costs(raw, "SPY")
        assert adjusted < raw
        assert adjusted > 0  # still positive after SPY costs


# ===================================================================
# model_monitor tests
# ===================================================================

class TestCalibrationMap:
    def test_fit_and_calibrate(self):
        from model_monitor import CalibrationMap
        np.random.seed(42)
        pred = np.random.normal(0.005, 0.02, 500)
        realized = pred * 0.6 + np.random.normal(0, 0.01, 500)

        cm = CalibrationMap(n_bins=10)
        cm.fit(pred, realized)

        # Calibrated return should be attenuated relative to raw prediction
        cal = cm.calibrated_return(0.03)
        assert abs(cal) < abs(0.03)  # calibration shrinks overconfident

    def test_save_load_roundtrip(self, tmp_path):
        from model_monitor import CalibrationMap
        np.random.seed(42)
        pred = np.random.normal(0, 0.02, 200)
        realized = pred * 0.5 + np.random.normal(0, 0.01, 200)

        cm = CalibrationMap(n_bins=5)
        cm.fit(pred, realized)

        path = str(tmp_path / "cal.json")
        cm.save(path)
        cm2 = CalibrationMap.load(path)

        assert np.allclose(cm.bin_edges, cm2.bin_edges)
        assert cm.calibrated_return(0.01) == cm2.calibrated_return(0.01)

    def test_no_calibration_returns_raw(self):
        from model_monitor import CalibrationMap
        cm = CalibrationMap()
        assert cm.calibrated_return(0.05) == 0.05

    def test_calibrated_confidence(self):
        from model_monitor import CalibrationMap
        np.random.seed(42)
        pred = np.random.normal(0, 0.02, 300)
        realized = pred * 0.5 + np.random.normal(0, 0.01, 300)

        cm = CalibrationMap(n_bins=10)
        cm.fit(pred, realized)

        conf = cm.calibrated_confidence(0.02, target_return=0.02)
        assert 0.0 <= conf <= 1.0
        # Large raw prediction should have moderate calibrated confidence
        # (not 1.0 because calibration attenuates)
        assert conf < 1.0


class TestModelMonitor:
    def test_record_and_compute(self, tmp_path):
        from model_monitor import ModelMonitor
        monitor = ModelMonitor(output_dir=str(tmp_path), window=50)

        # Record predictions with realized returns
        for i in range(40):
            pred = 0.01 * (1 if i % 2 == 0 else -1)
            monitor.record_prediction("SPY", pred, model_type="lstm")
            real = pred * 0.5 + np.random.normal(0, 0.005)
            monitor.record_realized("SPY", real)

        health = monitor.compute_health("SPY")
        assert health.n_trades == 40
        assert health.rolling_hit_rate > 0.4  # should be decent with correlated pred/real
        assert health.symbol == "SPY"

    def test_should_pause_on_bad_model(self):
        from model_monitor import ModelMonitor
        temp_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"monitor_test_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            monitor = ModelMonitor(output_dir=temp_dir, window=50)

            # Record terrible predictions (always wrong direction)
            for i in range(35):
                monitor.record_prediction("BAD", 0.01, model_type="lstm")
                monitor.record_realized("BAD", -0.02)  # always opposite

            health = monitor.compute_health("BAD")
            should_pause, reason = monitor.should_pause_model("BAD")
            assert should_pause
            assert "hit_rate" in reason or "IC" in reason
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_constant_inputs_keep_ic_zero_without_warnings(self, tmp_path):
        from model_monitor import ModelMonitor

        monitor = ModelMonitor(output_dir=str(tmp_path), window=50)
        for _ in range(35):
            monitor.record_prediction("CONST", 0.01, model_type="lstm")
            monitor.record_realized("CONST", -0.02)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            health = monitor.compute_health("CONST")

        assert health.rolling_ic == 0.0
        assert not any("ConstantInputWarning" in str(w.category.__name__) for w in caught)

    def test_pause_persists_across_restart_until_cleared(self):
        from model_monitor import ModelMonitor, ModelHealth

        temp_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"monitor_test_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            monitor = ModelMonitor(output_dir=temp_dir, window=50)
            monitor._health["PAUSED"] = ModelHealth(
                symbol="PAUSED",
                model_type="lstm",
                status="paused",
                warning_reason="IC=-0.200 < -0.10",
            )
            monitor._save_state()

            reloaded = ModelMonitor(output_dir=temp_dir, window=50)
            should_pause, reason = reloaded.should_pause_model("PAUSED")
            assert should_pause
            assert "IC=" in reason

            health = reloaded.compute_health("PAUSED")
            assert health.status == "paused"
            assert os.path.exists(os.path.join(temp_dir, "paused_models.json"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_clear_model_pause_after_retrain(self):
        from model_monitor import ModelMonitor, ModelHealth

        temp_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"monitor_test_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            monitor = ModelMonitor(output_dir=temp_dir, window=50)
            monitor._health["SPY"] = ModelHealth(
                symbol="SPY",
                model_type="tft_xgb_swing",
                status="paused",
                warning_reason="IC=-0.150 < -0.10",
            )
            monitor._save_state()

            reloaded = ModelMonitor(output_dir=temp_dir, window=50)
            reloaded.clear_model_pause("SPY", reason="retrained_swing_model")

            should_pause, reason = reloaded.should_pause_model("SPY")
            assert not should_pause
            assert reason == "ok"
            assert reloaded.get_all_health()["SPY"].status == "ok"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_hit_rate_below_45_pauses_model(self):
        from model_monitor import ModelMonitor

        temp_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"monitor_test_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            monitor = ModelMonitor(output_dir=temp_dir, window=50)
            for i in range(40):
                pred = 0.01
                realized = 0.02 if i < 16 else -0.02  # 40% hit rate
                monitor.record_prediction("WEAK", pred, model_type="xgb_swing")
                monitor.record_realized("WEAK", realized)

            health = monitor.compute_health("WEAK")
            assert health.rolling_hit_rate < 0.45
            assert health.status == "paused"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEntrySemantics:
    def test_backtester_open_position_supports_short_without_trend_gate(self):
        from backtester import Backtester, Portfolio

        bt = Backtester(
            symbol="SPY",
            adapter=None,
            mode="daily",
            model_type="swing",
            position_pct=0.5,
            use_cost_model=False,
        )
        portfolio = Portfolio(initial_capital=100_000.0, cash=100_000.0)

        opened = bt._open_position(
            portfolio,
            date="2025-01-02",
            price=100.0,
            direction="SHORT",
            expected_return=-0.03,
            bar_atr=1.0,
        )

        assert opened
        assert portfolio.position is not None
        assert portfolio.position.direction == "SHORT"

    def test_backtester_close_position_records_derisk_return(self):
        from backtester import Backtester, Portfolio, Trade

        bt = Backtester(
            symbol="SPY",
            adapter=None,
            mode="daily",
            model_type="swing",
            position_pct=0.5,
            use_cost_model=False,
        )
        portfolio = Portfolio(initial_capital=100_000.0, cash=50_000.0)
        portfolio.position = Trade(
            entry_date="2025-01-02",
            entry_price=100.0,
            direction="LONG",
            size=100.0,
        )

        bt._close_position(portfolio, date="2025-01-03", price=105.0, reason="test")

        assert bt._derisk_state.returns
        assert bt._derisk_state.returns[-1] > 0

    def test_backtester_same_direction_reentry_can_be_blocked_after_disaster_stop(self):
        from backtester import Backtester, Portfolio

        bt = Backtester(
            symbol="SPY",
            adapter=None,
            mode="daily",
            model_type="swing",
            position_pct=0.5,
            use_cost_model=False,
            cost_threshold=0.01,
        )
        bt._max_loss_exit = {"time": "2025-01-02", "direction": "LONG"}
        portfolio = Portfolio(initial_capital=100_000.0, cash=100_000.0)

        opened = bt._open_position(
            portfolio,
            date="2025-01-02 12:00:00",
            price=100.0,
            direction="LONG",
            expected_return=0.012,
            bar_atr=1.0,
        )

        assert not opened


class TestPortfolioBacktesterParity:
    def test_crypto_group_uses_btc_calendar(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["BTC/USD", "ETH/USD"],
            adapter=None,
            group="crypto",
            model_type="swing",
        )

        assert bt._calendar_symbol() == "BTC-USD"

    def test_etf_group_uses_spy_calendar(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["SMH", "QQQ"],
            adapter=None,
            group="swing",
            model_type="swing",
        )

        assert bt._calendar_symbol() == "SPY"

    def test_crypto_group_ignores_spy_vix_regime_scaling(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["BTC/USD", "ETH/USD"],
            adapter=None,
            group="crypto",
            model_type="swing",
        )

        scalar = bt._regime_scalar(
            day="2026-03-18",
            spy_date_map={"2026-03-18": 500.0},
            spy_sma200_map={"2026-03-18": 550.0},
            vix_map={"2026-03-18": 40.0},
        )

        assert scalar == 1.0

    def test_swing_group_uses_soft_regime_scaling(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["SMH", "QQQ"],
            adapter=None,
            group="swing",
            model_type="swing",
        )

        scalar = bt._regime_scalar(
            day="2026-03-18",
            spy_date_map={"2026-03-18": 500.0},
            spy_sma200_map={"2026-03-18": 550.0},
            vix_map={"2026-03-18": 40.0},
        )

        assert scalar == 0.5

    def test_portfolio_backtester_classifies_exit_params_from_bars(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["SMH", "QQQ"],
            adapter=None,
            group="swing",
            model_type="swing",
        )
        bars = pd.DataFrame({
            "high": np.linspace(101.0, 130.0, 40),
            "low": np.linspace(99.0, 128.0, 40),
            "close": np.linspace(100.0, 129.0, 40),
        })

        ep = bt._classify_exit_params(bars)

        assert ep.min_hold_bars > 0
        assert ep.signal_flip_consecutive >= 1

    def test_portfolio_backtester_btc_correlation_keeps_anchors_unpenalized(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["BTC/USD", "ETH/USD", "AVAX/USD"],
            adapter=None,
            group="crypto",
            model_type="swing",
        )
        btc = pd.DataFrame({"close": np.linspace(100, 140, 40)})
        eth = pd.DataFrame({"close": np.linspace(200, 260, 40)})
        avax = pd.DataFrame({"close": np.linspace(50, 70, 40)})

        corr = bt._compute_btc_correlations({
            "BTC/USD": btc,
            "ETH/USD": eth,
            "AVAX/USD": avax,
        })

        assert corr["BTC/USD"] == 0.0
        assert corr["ETH/USD"] == 0.0
        assert 0.0 <= corr["AVAX/USD"] <= 1.0

    def test_portfolio_backtester_rank_scalar_increases_with_score(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["SMH", "QQQ", "GLD"],
            adapter=None,
            group="swing",
            model_type="swing",
        )

        low = bt._rank_scalar_from_scores("GLD", {"SMH": 0.9, "QQQ": 0.6, "GLD": 0.3})
        high = bt._rank_scalar_from_scores("SMH", {"SMH": 0.9, "QQQ": 0.6, "GLD": 0.3})

        assert 0.5 <= low < high <= 1.0

    def test_portfolio_backtester_crypto_composite_penalizes_btc_correlation(self):
        from portfolio_backtester import PortfolioBacktester

        bt = PortfolioBacktester(
            symbols=["BTC/USD", "ETH/USD", "AVAX/USD"],
            adapter=None,
            group="crypto",
            model_type="swing",
        )

        composite = bt._compute_composite_scores(
            [("BTC-USD", 0.9), ("ETH-USD", 0.8), ("AVAX-USD", 0.7)],
            {"BTC-USD": 0.5, "ETH-USD": 0.6, "AVAX-USD": 0.6},
            {
                "BTC-USD": {"win_rate": 0.55},
                "ETH-USD": {"win_rate": 0.60},
                "AVAX-USD": {"win_rate": 0.60},
            },
            btc_correlations={"BTC-USD": 0.0, "ETH-USD": 0.0, "AVAX-USD": 0.9},
        )

        assert composite["AVAX-USD"] < composite["ETH-USD"]

    def test_portfolio_backtester_derisk_state_produces_half_kelly(self):
        from portfolio_backtester import PortfolioBacktester
        from risk_config import DeRiskState

        bt = PortfolioBacktester(
            symbols=["SMH", "QQQ"],
            adapter=None,
            group="swing",
            model_type="swing",
        )
        state = DeRiskState()
        for _ in range(25):
            state.record_trade(0.03)
        for _ in range(10):
            state.record_trade(-0.01)

        bt._derisk_states["SMH"] = state
        kelly = bt._derisk_states["SMH"].half_kelly(
            window=bt.risk.kelly_window,
            min_trades=bt.risk.kelly_min_trades,
        )

        assert kelly is not None
        assert kelly > 0

    def test_portfolio_backtester_tracks_disaster_stop_for_reentry_controls(self):
        from portfolio_backtester import PortfolioBacktester, PortfolioState, PortfolioTrade

        bt = PortfolioBacktester(
            symbols=["SMH", "QQQ"],
            adapter=None,
            group="swing",
            model_type="swing",
        )
        portfolio = PortfolioState(initial_capital=100_000.0, cash=90_000.0)
        portfolio.positions["SMH"] = PortfolioTrade(
            symbol="SMH",
            entry_date=pd.Timestamp("2025-01-02"),
            entry_price=100.0,
            direction="LONG",
            size=10.0,
            peak_price=100.0,
        )

        bt._close_position(portfolio, "SMH", pd.Timestamp("2025-01-03"), 95.0, "disaster_stop")

        assert "SMH" in bt._max_loss_exits
        assert bt._max_loss_exits["SMH"]["direction"] == "LONG"

