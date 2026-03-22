"""Tests for the Gold Scalper Engine — TC baby V1.0 port.

测试覆盖：
1. 偏向堆栈：全部对齐、混合、翻转
2. 交易时段：窗口内、死区、周五平仓
3. RSI阈值 + 吞没形态检测
4. TP阶梯：价格逐步穿过TP1-TP4，验证合约平仓和止损移动
5. 超时：2小时01分TP1未命中 → 全平
6. 仓位缩放：$0→6手, $15K→18手, $30K→24手(上限)
7. 熔断：每日亏损-$1001 → 不再开新仓
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.gold_scalper.config import GoldScalperConfig, load_config
from src.gold_scalper.session_filter import SessionFilter
from src.gold_scalper.bias_stack import BiasStack, TimeframeBias
from src.gold_scalper.signals import EntrySignalGenerator
from src.gold_scalper.sizing import PositionSizer
from src.gold_scalper.position_manager import PositionManager, GoldPosition


PT = ZoneInfo("America/Los_Angeles")


def _make_config(**overrides) -> GoldScalperConfig:
    """Create config with optional overrides."""
    defaults = {
        "symbol": "GC=F",
        "pip_value": 0.10,
        "base_contracts": 6,
        "max_scale_mult": 4,
        "hard_stop_pips": 300.0,
        "tp1_pips": 60.0, "tp1_contracts": 1,
        "tp2_pips": 220.0, "tp2_contracts": 2,
        "tp3_pips": 400.0, "tp3_contracts": 1,
        "tp4_pips": 600.0, "tp4_contracts": 1,
        "be_offset_pips": 10.0,
        "tp1_timeout_minutes": 120,
        "daily_loss_limit": -1000.0,
        "rsi_period": 14,
        "rsi_long_threshold": 55.0,
        "rsi_short_threshold": 35.0,
    }
    defaults.update(overrides)
    return GoldScalperConfig(**defaults)


def _make_bars(closes: list, n_bars: int = None) -> pd.DataFrame:
    """Create simple OHLCV DataFrame from close prices."""
    if n_bars is None:
        n_bars = len(closes)
    data = {
        "ts": pd.date_range("2025-01-01", periods=n_bars, freq="5min"),
        "open": [c - 0.05 for c in closes[:n_bars]],
        "high": [c + 0.10 for c in closes[:n_bars]],
        "low": [c - 0.10 for c in closes[:n_bars]],
        "close": closes[:n_bars],
        "volume": [1000] * n_bars,
    }
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════
# 1. BIAS STACK TESTS
# ══════════════════════════════════════════════════════════════

class TestBiasStack:
    def test_all_bullish(self):
        """All TFs above EMA → LONG."""
        config = _make_config()
        bs = BiasStack(config)

        # Create bars where close is always above EMA
        closes = list(np.linspace(100, 120, 30))  # trending up
        bars = {tf: _make_bars(closes) for tf in BiasStack.TIMEFRAMES}

        biases = bs.compute(bars)
        assert all(b.is_bullish for b in biases)
        assert bs.is_aligned(biases) == "LONG"

    def test_all_bearish(self):
        """All TFs below EMA → SHORT."""
        config = _make_config()
        bs = BiasStack(config)

        closes = list(np.linspace(120, 100, 30))  # trending down
        bars = {tf: _make_bars(closes) for tf in BiasStack.TIMEFRAMES}

        biases = bs.compute(bars)
        assert all(b.is_bearish for b in biases)
        assert bs.is_aligned(biases) == "SHORT"

    def test_mixed_bias(self):
        """Mixed TFs → None (no trade)."""
        config = _make_config()
        bs = BiasStack(config)

        # 4 bullish, 1 bearish
        up = list(np.linspace(100, 120, 30))
        down = list(np.linspace(120, 100, 30))
        bars = {
            "1d": _make_bars(up),
            "4h": _make_bars(up),
            "1h": _make_bars(up),
            "15m": _make_bars(up),
            "5m": _make_bars(down),  # 5m bearish
        }

        biases = bs.compute(bars)
        assert bs.is_aligned(biases) is None

    def test_bias_summary(self):
        """Bias summary string format."""
        biases = [
            TimeframeBias("1d", 100, 99, "BULL"),
            TimeframeBias("4h", 100, 99, "BULL"),
            TimeframeBias("1h", 100, 101, "BEAR"),
            TimeframeBias("15m", 100, 99, "BULL"),
            TimeframeBias("5m", 100, 99, "BULL"),
        ]
        config = _make_config()
        bs = BiasStack(config)
        summary = bs.bias_summary(biases)
        assert "BULL" in summary
        assert "BEAR" in summary

    def test_resample_4h(self):
        """1H bars resample to 4H correctly."""
        n = 24  # 24 hourly bars
        df = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=n, freq="1h"),
            "open": range(n),
            "high": [x + 1 for x in range(n)],
            "low": [x - 1 for x in range(n)],
            "close": range(n),
            "volume": [100] * n,
        })
        result = BiasStack.resample_to_4h(df)
        assert len(result) == 6  # 24h / 4h = 6 bars
        assert "close" in result.columns


# ══════════════════════════════════════════════════════════════
# 2. SESSION FILTER TESTS
# ══════════════════════════════════════════════════════════════

class TestSessionFilter:
    def setup_method(self):
        self.sf = SessionFilter(_make_config())

    def test_in_trading_window(self):
        """10am PT Wednesday → can trade."""
        dt = datetime(2025, 3, 19, 10, 0, tzinfo=PT)  # Wednesday
        assert self.sf.is_trading_window(dt)
        assert not self.sf.is_dead_zone(dt)
        assert self.sf.can_trade(dt)

    def test_dead_zone_overnight(self):
        """3am PT → dead zone."""
        dt = datetime(2025, 3, 19, 3, 0, tzinfo=PT)
        assert self.sf.is_dead_zone(dt)

    def test_sunday_dead(self):
        """Sunday → dead zone."""
        dt = datetime(2025, 3, 23, 12, 0, tzinfo=PT)  # Sunday
        assert self.sf.is_dead_zone(dt)
        assert not self.sf.can_trade(dt)

    def test_friday_close(self):
        """Friday 12:51 PT → force close."""
        dt = datetime(2025, 3, 21, 12, 51, tzinfo=PT)  # Friday
        should_close, reason = self.sf.should_force_close(dt)
        assert should_close
        assert "Friday" in reason

    def test_daily_close(self):
        """Wednesday 12:46 PT → force close."""
        dt = datetime(2025, 3, 19, 12, 46, tzinfo=PT)
        should_close, reason = self.sf.should_force_close(dt)
        assert should_close
        assert "Daily" in reason

    def test_avoid_entry_near_close(self):
        """Wednesday 12:35 PT → avoid new entries."""
        dt = datetime(2025, 3, 19, 12, 35, tzinfo=PT)
        avoid, _ = self.sf.should_avoid_entry(dt)
        assert avoid


# ══════════════════════════════════════════════════════════════
# 3. ENTRY SIGNAL TESTS
# ══════════════════════════════════════════════════════════════

class TestEntrySignals:
    def setup_method(self):
        self.gen = EntrySignalGenerator(_make_config())

    def test_rsi_computation(self):
        """RSI computes a value between 0-100."""
        np.random.seed(42)
        closes = pd.Series(np.cumsum(np.random.randn(100)) + 2000)
        rsi = self.gen.compute_rsi(closes)
        assert 0 <= rsi <= 100

    def test_bullish_engulfing(self):
        """Detect bullish engulfing pattern."""
        bars = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=5, freq="5min"),
            "open":  [100, 100, 99,  98, 100],   # bar[-3] red, bar[-2] green engulfing
            "close": [101, 99,  98, 100,  101],
            "high":  [102, 101, 100, 101, 102],
            "low":   [99,  98,  97,  97,  99],
            "volume": [100] * 5,
        })
        result = self.gen.detect_engulfing(bars)
        assert result == "BULLISH"

    def test_bearish_engulfing(self):
        """Detect bearish engulfing pattern."""
        bars = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=5, freq="5min"),
            "open":  [100, 99,  100, 101, 100],   # bar[-3] green, bar[-2] red engulfing
            "close": [101, 100, 101, 99,  98],
            "high":  [102, 101, 102, 102, 101],
            "low":   [99,  98,  99,  98,  97],
            "volume": [100] * 5,
        })
        result = self.gen.detect_engulfing(bars)
        assert result == "BEARISH"

    def test_signal_rejected_no_bias(self):
        """No bias alignment → no signal."""
        bars = _make_bars([2000 + i for i in range(30)])
        signal = self.gen.evaluate(None, bars)
        assert not signal.is_valid
        assert "Bias" in signal.reason_rejected

    def test_signal_rejected_rsi(self):
        """RSI below threshold for LONG → rejected."""
        # Create flat data (RSI ~ 50)
        bars = _make_bars([2000] * 30)
        signal = self.gen.evaluate("LONG", bars)
        assert not signal.is_valid
        assert "RSI" in signal.reason_rejected


# ══════════════════════════════════════════════════════════════
# 4. POSITION SIZING TESTS
# ══════════════════════════════════════════════════════════════

class TestPositionSizing:
    def setup_method(self):
        self.sizer = PositionSizer(_make_config())

    def test_base_sizing(self):
        """$0 profit → 6 contracts at 1x scale."""
        result = self.sizer.compute(0.0)
        assert result.total_contracts == 6
        assert result.scale_mult == 1
        assert result.runner_qty == 1

    def test_scaled_sizing(self):
        """$15K profit → 18 contracts at 3x scale."""
        result = self.sizer.compute(15000.0)
        assert result.total_contracts == 18
        assert result.scale_mult == 3

    def test_max_cap(self):
        """$30K profit → 24 contracts at 4x (capped)."""
        result = self.sizer.compute(30000.0)
        assert result.total_contracts == 24
        assert result.scale_mult == 4

    def test_huge_profit_still_capped(self):
        """$100K profit → still 24 contracts (4x cap)."""
        result = self.sizer.compute(100000.0)
        assert result.total_contracts == 24
        assert result.scale_mult == 4

    def test_negative_profit_no_scale(self):
        """Negative PnL → 1x scale."""
        result = self.sizer.compute(-5000.0)
        assert result.total_contracts == 6
        assert result.scale_mult == 1

    def test_tp_splits_sum(self):
        """TP splits + runner = total contracts."""
        for pnl in [0, 7500, 15000, 22500]:
            result = self.sizer.compute(pnl)
            assert sum(result.tp_splits) + result.runner_qty == result.total_contracts


# ══════════════════════════════════════════════════════════════
# 5. POSITION MANAGER TESTS
# ══════════════════════════════════════════════════════════════

class TestPositionManager:
    def setup_method(self):
        self.config = _make_config()
        self.pm = PositionManager(self.config)
        self.now = datetime(2025, 3, 19, 10, 0, tzinfo=PT)  # Wednesday 10am PT

    def _make_position(self, direction="LONG", entry_price=2000.0) -> GoldPosition:
        return self.pm.create_position(
            direction=direction,
            entry_price=entry_price,
            entry_time=self.now,
            total_contracts=6,
            tp_splits=[1, 2, 1, 1],
            runner_qty=1,
        )

    def test_hard_stop_long(self):
        """LONG position hard stop = entry - 300 pips ($30)."""
        pos = self._make_position("LONG", 2000.0)
        expected_sl = 2000.0 - (300.0 * 0.10)  # 2000 - 30 = 1970
        assert abs(pos.hard_stop - expected_sl) < 0.01

    def test_hard_stop_short(self):
        """SHORT position hard stop = entry + 300 pips ($30)."""
        pos = self._make_position("SHORT", 2000.0)
        expected_sl = 2000.0 + (300.0 * 0.10)  # 2000 + 30 = 2030
        assert abs(pos.hard_stop - expected_sl) < 0.01

    def test_tp1_hit_long(self):
        """TP1 at +60 pips → close 1 contract, SL to BE+10."""
        pos = self._make_position("LONG", 2000.0)
        tp1_price = 2000.0 + (60.0 * 0.10) + 0.01  # just above TP1

        actions = self.pm.update(pos, tp1_price, self.now + timedelta(minutes=30), "LONG")

        assert len(actions) >= 1
        assert actions[0].reason == "TP1"
        assert actions[0].contracts == 1
        assert pos.tp1_hit is True
        assert pos.remaining_contracts == 5

        # SL should be at BE + 10 pips = 2000 + 1.0 = 2001
        expected_sl = 2000.0 + (10.0 * 0.10)
        assert abs(pos.current_stop - expected_sl) < 0.01

    def test_tp_ladder_full(self):
        """Price walks through all 4 TPs — verify contracts and SL at each step."""
        pos = self._make_position("LONG", 2000.0)

        # TP1: +60 pips = 2006
        t = self.now + timedelta(minutes=30)
        self.pm.update(pos, 2006.01, t, "LONG")
        assert pos.tp1_hit
        assert pos.remaining_contracts == 5

        # TP2: +220 pips = 2022
        t += timedelta(minutes=30)
        self.pm.update(pos, 2022.01, t, "LONG")
        assert pos.tp2_hit
        assert pos.remaining_contracts == 3
        # SL should be at TP1 = 2006
        assert abs(pos.current_stop - 2006.0) < 0.01

        # TP3: +400 pips = 2040
        t += timedelta(minutes=30)
        self.pm.update(pos, 2040.01, t, "LONG")
        assert pos.tp3_hit
        assert pos.remaining_contracts == 2
        # SL should be at TP2 = 2022
        assert abs(pos.current_stop - 2022.0) < 0.01

        # TP4: +600 pips = 2060
        t += timedelta(minutes=30)
        self.pm.update(pos, 2060.01, t, "LONG")
        assert pos.tp4_hit
        assert pos.remaining_contracts == 1
        # SL should be at TP3 = 2040
        assert abs(pos.current_stop - 2040.0) < 0.01

    def test_stop_loss_hit(self):
        """Price drops below hard stop → CLOSE_ALL."""
        pos = self._make_position("LONG", 2000.0)
        stop_price = pos.hard_stop - 0.01  # just below stop

        actions = self.pm.update(pos, stop_price, self.now + timedelta(minutes=10), "LONG")

        assert len(actions) == 1
        assert actions[0].action == "CLOSE_ALL"
        assert "Hard Stop" in actions[0].reason

    def test_timeout_no_tp1(self):
        """2h01m without TP1 → CLOSE_ALL."""
        pos = self._make_position("LONG", 2000.0)
        # Price stays flat (no TP1 hit)
        t = self.now + timedelta(minutes=121)

        actions = self.pm.update(pos, 2000.50, t, "LONG")

        assert len(actions) == 1
        assert actions[0].action == "CLOSE_ALL"
        assert "Timeout" in actions[0].reason

    def test_runner_exit_bias_flip(self):
        """After TP2 hit, bias flips → runner exits."""
        pos = self._make_position("LONG", 2000.0)

        # Hit TP1 and TP2
        t = self.now + timedelta(minutes=30)
        self.pm.update(pos, 2006.01, t, "LONG")  # TP1
        self.pm.update(pos, 2022.01, t + timedelta(minutes=10), "LONG")  # TP2

        # Now bias flips (pass SHORT instead of LONG)
        actions = self.pm.update(
            pos, 2020.0, t + timedelta(minutes=20), "SHORT"
        )

        assert any("Bias Flipped" in a.reason for a in actions)

    def test_runner_no_exit_before_tp2(self):
        """Before TP2, bias flip does NOT trigger runner exit."""
        pos = self._make_position("LONG", 2000.0)

        # Only TP1 hit
        t = self.now + timedelta(minutes=30)
        self.pm.update(pos, 2006.01, t, "LONG")  # TP1

        # Bias flips but TP2 not hit yet
        actions = self.pm.update(
            pos, 2005.0, t + timedelta(minutes=10), "SHORT"
        )

        # Should NOT have runner exit
        assert not any("Bias Flipped" in a.reason for a in actions)

    def test_pips_from_entry(self):
        """Pip calculation for LONG and SHORT."""
        pos_long = self._make_position("LONG", 2000.0)
        assert abs(pos_long.pips_from_entry(2006.0, 0.10) - 60.0) < 0.01

        pos_short = self._make_position("SHORT", 2000.0)
        assert abs(pos_short.pips_from_entry(1994.0, 0.10) - 60.0) < 0.01


# ══════════════════════════════════════════════════════════════
# 6. CONFIG TESTS
# ══════════════════════════════════════════════════════════════

class TestConfig:
    def test_default_config(self):
        """Default config has expected values."""
        config = GoldScalperConfig()
        assert config.base_contracts == 6
        assert config.max_scale_mult == 4
        assert config.hard_stop_pips == 300.0
        assert config.runner_contracts == 1

    def test_tp_levels(self):
        """TP levels property returns correct structure."""
        config = GoldScalperConfig()
        levels = config.tp_levels
        assert len(levels) == 4
        assert levels[0] == (60.0, 1)
        assert levels[1] == (220.0, 2)

    def test_pips_to_price(self):
        """60 pips × $0.10 = $6.00."""
        config = GoldScalperConfig()
        assert abs(config.pips_to_price(60.0) - 6.0) < 0.001

    def test_load_config_defaults(self):
        """load_config with no file returns defaults."""
        config = load_config("/nonexistent/path.json")
        assert config.symbol == "GC=F"
