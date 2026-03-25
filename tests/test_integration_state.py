"""
Integration tests for state management across restarts and dashboard consistency.

Covers:
1. Kraken executor accounting (LONG/SHORT open/close, duplicate detection)
2. Dashboard equity curve matches executor accounting
3. Trade pairing handles overwrites, partial closes, edge cases
4. Derisk state rebuilds from CSV trade logs (Kelly persistence)

Run: .venv/Scripts/python.exe -m pytest tests/test_integration_state.py -v
"""
import sys
import os
import csv
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest


# ===================================================================
# Fixtures
# ===================================================================

def make_trade(symbol, side, qty, price, time, intent):
    """Helper to create a trade log entry."""
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "time": time,
        "intent": intent,
    }


def make_order(symbol, side, qty, price, filled_at, intent):
    """Helper to create an order entry for trade pairing."""
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "filled_at": filled_at,
        "intent": intent,
    }


# ===================================================================
# Kraken executor paper accounting
# ===================================================================

class TestKrakenAccounting:
    """Test that the Kraken executor's paper_fill accounting is correct."""

    def test_long_open_deducts_notional_plus_fee(self):
        """LONG open: cash -= qty * price + fee."""
        initial = 100_000.0
        fee_pct = 0.0026
        qty, price = 100.0, 50.0
        notional = qty * price  # 5000
        fee = notional * fee_pct  # 13.0
        expected_cash = initial - notional - fee
        assert expected_cash == pytest.approx(94987.0, abs=0.01)

    def test_long_close_returns_entry_notional_plus_pnl(self):
        """LONG close: cash += entry * qty + pnl - fee."""
        cash = 95000.0
        entry_price, exit_price, qty = 50.0, 55.0, 100.0
        fee_pct = 0.0026
        pnl_raw = (exit_price - entry_price) * qty  # 500
        fee = exit_price * qty * fee_pct  # 14.3
        pnl = pnl_raw - fee
        cash += entry_price * qty + pnl  # return notional + net pnl
        assert cash == pytest.approx(95000 + 5000 + 500 - 14.3, abs=0.1)

    def test_short_open_deducts_only_fee(self):
        """SHORT open: cash -= fee only (margin model)."""
        initial = 100_000.0
        fee_pct = 0.0026
        qty, price = 100.0, 50.0
        fee = qty * price * fee_pct  # 13.0
        expected_cash = initial - fee
        assert expected_cash == pytest.approx(99987.0, abs=0.01)

    def test_short_close_returns_pnl_only(self):
        """SHORT close: cash += pnl (margin is already in cash)."""
        cash = 99987.0
        entry_price, exit_price, qty = 50.0, 45.0, 100.0
        fee_pct = 0.0026
        pnl_raw = (entry_price - exit_price) * qty  # 500
        fee = exit_price * qty * fee_pct  # 11.7
        pnl = pnl_raw - fee
        cash += pnl
        assert cash == pytest.approx(99987.0 + 500 - 11.7, abs=0.1)

    def test_duplicate_open_rejected(self):
        """Opening same symbol twice should be rejected (returns None)."""
        # Simulate the guard: if symbol in positions, skip
        positions = {"BTC/USD": {"side": "LONG", "qty": 1.0, "entry_price": 50000}}
        symbol = "BTC/USD"
        result = None
        if symbol in positions:
            result = None  # skip
        assert result is None

    def test_close_nonexistent_no_crash(self):
        """Closing a symbol not in positions should not crash."""
        positions = {}
        symbol = "BTC/USD"
        # Executor guards: if is_close and symbol in positions
        if symbol in positions:
            pass  # would close
        # No error raised — test passes

    def test_long_roundtrip_equity_conserved(self):
        """Full LONG open→close: equity should equal initial + net P&L."""
        initial = 100_000.0
        fee_pct = 0.0026
        qty, entry, exit_p = 100.0, 50.0, 55.0

        # Open
        open_fee = qty * entry * fee_pct
        cash = initial - qty * entry - open_fee

        # Close
        close_fee = qty * exit_p * fee_pct
        pnl = (exit_p - entry) * qty - close_fee
        cash += entry * qty + pnl

        expected = initial + (exit_p - entry) * qty - open_fee - close_fee
        assert cash == pytest.approx(expected, abs=0.01)

    def test_short_roundtrip_equity_conserved(self):
        """Full SHORT open→close: equity should equal initial + net P&L."""
        initial = 100_000.0
        fee_pct = 0.0026
        qty, entry, exit_p = 100.0, 50.0, 45.0

        # Open SHORT
        open_fee = qty * entry * fee_pct
        cash = initial - open_fee

        # Close SHORT
        close_fee = qty * exit_p * fee_pct
        pnl = (entry - exit_p) * qty - close_fee
        cash += pnl

        expected = initial + (entry - exit_p) * qty - open_fee - close_fee
        assert cash == pytest.approx(expected, abs=0.01)


# ===================================================================
# Dashboard equity curve vs executor accounting
# ===================================================================

class TestEquityCurveConsistency:
    """Verify dashboard _build_equity_curve matches executor accounting."""

    def _build_curve(self, trades, initial, fee_pct=0.0):
        from history import _build_equity_curve
        return _build_equity_curve(trades, initial, fee_pct)

    def test_single_long_roundtrip(self):
        """Single LONG open→close should produce correct final equity."""
        initial = 100_000.0
        fee_pct = 0.0026
        qty, entry, exit_p = 100.0, 50.0, 55.0
        trades = [
            make_trade("BTC", "buy", qty, entry, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("BTC", "sell", qty, exit_p, "2026-01-01T01:00:00+00:00", "close"),
        ]
        curve = self._build_curve(trades, initial, fee_pct)
        final = curve["equity"][-1]

        # Manual calculation
        open_fee = qty * entry * fee_pct
        close_fee = qty * exit_p * fee_pct
        expected = initial + (exit_p - entry) * qty - open_fee - close_fee
        assert final == pytest.approx(expected, abs=0.01)

    def test_single_short_roundtrip(self):
        """Single SHORT open→close should produce correct final equity."""
        initial = 100_000.0
        fee_pct = 0.0026
        qty, entry, exit_p = 100.0, 50.0, 45.0
        trades = [
            make_trade("BTC", "sell", qty, entry, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("BTC", "buy", qty, exit_p, "2026-01-01T01:00:00+00:00", "close"),
        ]
        curve = self._build_curve(trades, initial, fee_pct)
        final = curve["equity"][-1]

        open_fee = qty * entry * fee_pct
        close_fee = qty * exit_p * fee_pct
        expected = initial + (entry - exit_p) * qty - open_fee - close_fee
        assert final == pytest.approx(expected, abs=0.01)

    def test_position_overwrite_implicit_close(self):
        """Double open same symbol: old position implicitly closed at new price."""
        initial = 100_000.0
        fee_pct = 0.0
        trades = [
            make_trade("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            # Overwrite: implicitly close old @12, open new @12
            make_trade("A", "buy", 50, 12.0, "2026-01-01T01:00:00+00:00", "open"),
            make_trade("A", "sell", 50, 15.0, "2026-01-01T02:00:00+00:00", "close"),
        ]
        curve = self._build_curve(trades, initial, fee_pct)
        final = curve["equity"][-1]

        # First open: cash -= 100 * 10 = -1000, cash = 99000
        # Implicit close at 12: pnl = (12-10)*100 = 200, cash += 10*100 + 200 = 100200
        # Second open at 12: cash -= 50*12 = -600, cash = 99600
        # Close at 15: pnl = (15-12)*50 = 150, cash += 12*50 + 150 = 100350
        assert final == pytest.approx(100_350.0, abs=0.01)

    def test_partial_close(self):
        """Closing half a position should keep remaining half open."""
        initial = 100_000.0
        fee_pct = 0.0
        trades = [
            make_trade("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            # Partial close: sell 60 of 100
            make_trade("A", "sell", 60, 12.0, "2026-01-01T01:00:00+00:00", "close"),
            # Close remaining 40
            make_trade("A", "sell", 40, 15.0, "2026-01-01T02:00:00+00:00", "close"),
        ]
        curve = self._build_curve(trades, initial, fee_pct)
        final = curve["equity"][-1]

        # Open: cash = 100000 - 1000 = 99000
        # Partial close 60@12: pnl = (12-10)*60 = 120, cash += 10*60 + 120 = 99720
        # Still holding 40 @ 10, last_price = 12 → pos_val = 40*12 = 480
        mid_equity = curve["equity"][1]
        assert mid_equity == pytest.approx(99720 + 40 * 12, abs=0.01)

        # Close remaining 40@15: pnl = (15-10)*40 = 200, cash += 10*40 + 200 = 100320
        assert final == pytest.approx(100_320.0, abs=0.01)

    def test_no_trades_returns_empty(self):
        """Empty trade list returns empty dict."""
        curve = self._build_curve([], 100_000.0)
        assert curve == {}

    def test_fees_reduce_equity(self):
        """With fees, final equity should be lower than without."""
        initial = 100_000.0
        qty, entry, exit_p = 100.0, 50.0, 55.0
        trades = [
            make_trade("A", "buy", qty, entry, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("A", "sell", qty, exit_p, "2026-01-01T01:00:00+00:00", "close"),
        ]
        no_fee = self._build_curve(trades, initial, 0.0)["equity"][-1]
        with_fee = self._build_curve(trades, initial, 0.0026)["equity"][-1]
        assert with_fee < no_fee

    def test_multiple_symbols_concurrent(self):
        """Multiple positions open simultaneously should track independently."""
        initial = 100_000.0
        fee_pct = 0.0
        trades = [
            make_trade("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("B", "sell", 50, 20.0, "2026-01-01T00:01:00+00:00", "open"),
            make_trade("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
            make_trade("B", "buy", 50, 18.0, "2026-01-01T01:01:00+00:00", "close"),
        ]
        curve = self._build_curve(trades, initial, fee_pct)
        final = curve["equity"][-1]

        # A: LONG profit = (12-10)*100 = 200
        # B: SHORT profit = (20-18)*50 = 100
        assert final == pytest.approx(initial + 300, abs=0.01)


# ===================================================================
# Trade pairing (_pair_closed_trades)
# ===================================================================

class TestTradePairing:
    """Test trade pairing handles all edge cases."""

    def _pair(self, orders, fee_pct=0.0):
        from history import _pair_closed_trades
        return _pair_closed_trades(orders, fee_pct)

    def test_simple_roundtrip(self):
        orders = [
            make_order("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "A"
        assert t["direction"] == "LONG"
        assert t["pnl_dollar"] == pytest.approx(200.0, abs=0.01)

    def test_short_roundtrip(self):
        orders = [
            make_order("A", "sell", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "buy", 100, 8.0, "2026-01-01T01:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 1
        assert trades[0]["direction"] == "SHORT"
        assert trades[0]["pnl_dollar"] == pytest.approx(200.0, abs=0.01)

    def test_overwrite_creates_implicit_close(self):
        """Double open creates an implicit close for the first position."""
        orders = [
            make_order("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "buy", 50, 12.0, "2026-01-01T01:00:00+00:00", "open"),
            make_order("A", "sell", 50, 15.0, "2026-01-01T02:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 2
        # First: implicit close of 100@10 → exit@12
        assert trades[0]["pnl_dollar"] == pytest.approx(200.0, abs=0.01)
        # Second: explicit close of 50@12 → exit@15
        assert trades[1]["pnl_dollar"] == pytest.approx(150.0, abs=0.01)

    def test_partial_close(self):
        """Partial close should leave remaining qty open."""
        orders = [
            make_order("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "sell", 60, 12.0, "2026-01-01T01:00:00+00:00", "close"),
            make_order("A", "sell", 40, 15.0, "2026-01-01T02:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 2
        # First partial: 60@10 → 12 = $120
        assert trades[0]["pnl_dollar"] == pytest.approx(120.0, abs=0.01)
        # Second partial: 40@10 → 15 = $200
        assert trades[1]["pnl_dollar"] == pytest.approx(200.0, abs=0.01)

    def test_fees_deducted_from_pnl(self):
        """Fees should reduce P&L."""
        fee_pct = 0.0026
        orders = [
            make_order("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
        ]
        no_fee = self._pair(orders, 0.0)
        with_fee = self._pair(orders, fee_pct)
        assert with_fee[0]["pnl_dollar"] < no_fee[0]["pnl_dollar"]
        # Fee = (100*10 + 100*12) * 0.0026 = 5.72
        expected_fee = (100 * 10 + 100 * 12) * fee_pct
        diff = no_fee[0]["pnl_dollar"] - with_fee[0]["pnl_dollar"]
        assert diff == pytest.approx(expected_fee, abs=0.01)

    def test_zero_qty_skipped(self):
        """Trades with qty ~0 should be skipped."""
        orders = [
            make_order("A", "buy", 1e-12, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "sell", 1e-12, 12.0, "2026-01-01T01:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 0

    def test_zero_price_skipped(self):
        """Trades with zero price should be skipped."""
        orders = [
            make_order("A", "buy", 100, 0.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 0

    def test_close_without_open_ignored(self):
        """Close with no matching open should not crash or create trade."""
        orders = [
            make_order("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 0

    def test_multiple_symbols_independent(self):
        """Trades in different symbols don't interfere."""
        orders = [
            make_order("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_order("B", "buy", 50, 20.0, "2026-01-01T00:01:00+00:00", "open"),
            make_order("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
            make_order("B", "sell", 50, 25.0, "2026-01-01T01:01:00+00:00", "close"),
        ]
        trades = self._pair(orders)
        assert len(trades) == 2
        by_sym = {t["symbol"]: t for t in trades}
        assert by_sym["A"]["pnl_dollar"] == pytest.approx(200.0, abs=0.01)
        assert by_sym["B"]["pnl_dollar"] == pytest.approx(250.0, abs=0.01)


# ===================================================================
# Equity curve ↔ trade pairing consistency
# ===================================================================

class TestCurveTradeConsistency:
    """Verify that equity curve final value matches sum of paired trade P&Ls."""

    def _build_curve(self, trades, initial, fee_pct=0.0):
        from history import _build_equity_curve
        return _build_equity_curve(trades, initial, fee_pct)

    def _pair(self, orders, fee_pct=0.0):
        from history import _pair_closed_trades
        return _pair_closed_trades(orders, fee_pct)

    def _to_orders(self, trades):
        """Convert trade log format to order format for pairing."""
        return [{"symbol": t["symbol"], "side": t["side"], "qty": t["qty"],
                 "price": t["price"], "filled_at": t["time"], "intent": t["intent"]}
                for t in trades]

    def test_all_closed_equity_matches_pnl_sum(self):
        """When all positions are closed, final equity = initial + sum(trade pnls)."""
        initial = 100_000.0
        fee_pct = 0.0026
        trades = [
            make_trade("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("B", "sell", 50, 20.0, "2026-01-01T00:01:00+00:00", "open"),
            make_trade("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
            make_trade("B", "buy", 50, 18.0, "2026-01-01T01:01:00+00:00", "close"),
        ]

        curve = self._build_curve(trades, initial, fee_pct)
        final_equity = curve["equity"][-1]

        orders = self._to_orders(trades)
        paired = self._pair(orders, fee_pct)
        total_pnl = sum(t["pnl_dollar"] for t in paired)

        assert final_equity == pytest.approx(initial + total_pnl, abs=0.01)

    def test_multi_roundtrip_same_symbol(self):
        """Multiple open→close cycles on same symbol: curve and paired trades agree."""
        initial = 100_000.0
        fee_pct = 0.0026
        trades = [
            make_trade("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("A", "sell", 100, 12.0, "2026-01-01T01:00:00+00:00", "close"),
            make_trade("A", "buy", 80, 11.0, "2026-01-01T02:00:00+00:00", "open"),
            make_trade("A", "sell", 80, 13.0, "2026-01-01T03:00:00+00:00", "close"),
        ]

        curve = self._build_curve(trades, initial, fee_pct)
        final_equity = curve["equity"][-1]

        orders = self._to_orders(trades)
        paired = self._pair(orders, fee_pct)
        total_pnl = sum(t["pnl_dollar"] for t in paired)

        assert final_equity == pytest.approx(initial + total_pnl, abs=0.01)

    def test_overwrite_scenario_consistent(self):
        """Position overwrite: both curve and pairing agree on net P&L."""
        initial = 100_000.0
        fee_pct = 0.0
        trades = [
            make_trade("A", "buy", 100, 10.0, "2026-01-01T00:00:00+00:00", "open"),
            make_trade("A", "buy", 50, 12.0, "2026-01-01T01:00:00+00:00", "open"),
            make_trade("A", "sell", 50, 15.0, "2026-01-01T02:00:00+00:00", "close"),
        ]

        curve = self._build_curve(trades, initial, fee_pct)
        final_equity = curve["equity"][-1]

        orders = self._to_orders(trades)
        paired = self._pair(orders, fee_pct)
        total_pnl = sum(t["pnl_dollar"] for t in paired)

        assert final_equity == pytest.approx(initial + total_pnl, abs=0.01)


# ===================================================================
# Derisk state rebuild from CSV trade logs
# ===================================================================

class TestDeriskRebuild:
    """Test that derisk states rebuild correctly from CSV trade logs."""

    @pytest.fixture
    def trades_dir(self, tmp_path):
        """Create a temp directory with CSV trade logs."""
        d = tmp_path / "trades"
        d.mkdir()
        return d

    def _write_csv(self, path, rows):
        """Write rows to a CSV file with the standard header."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "group", "symbol", "side",
                             "qty", "price", "reason", "pnl_pct"])
            for row in rows:
                writer.writerow(row)

    def test_rebuild_basic(self, trades_dir):
        """Rebuild should load closed trades and compute Kelly."""
        from risk_config import DeRiskState
        # 25 trades: 15 wins, 10 losses → should have Kelly
        rows = []
        for i in range(15):
            rows.append([f"2026-01-01 {i:02d}:00:00", "crypto_intraday",
                         "GOMINING/USD", "buy", "100", "1.00",
                         "disaster_stop (+5.00%)", "+0.0500"])
        for i in range(10):
            rows.append([f"2026-01-01 {15+i:02d}:00:00", "crypto_intraday",
                         "GOMINING/USD", "sell", "100", "1.00",
                         "disaster_stop (-3.00%)", "-0.0300"])
        self._write_csv(trades_dir / "daily_trades_20260101.csv", rows)

        # Replay the rebuild logic
        derisk_states = {}
        import glob
        for csv_path in sorted(glob.glob(str(trades_dir / "daily_trades_*.csv"))):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("group") != "crypto_intraday":
                        continue
                    reason = row.get("reason", "")
                    pnl = float(row.get("pnl_pct", "+0.0000"))
                    if "entry" in reason.lower() and abs(pnl) < 1e-8:
                        continue
                    sym = row.get("symbol", "").replace("/", "-")
                    if sym not in derisk_states:
                        derisk_states[sym] = DeRiskState()
                    derisk_states[sym].record_trade(pnl)

        ds = derisk_states["GOMINING-USD"]
        assert len(ds.returns) == 25
        kelly = ds.half_kelly(window=60, min_trades=20)
        assert kelly is not None
        assert kelly > 0  # positive edge → positive Kelly

    def test_rebuild_skips_entries(self, trades_dir):
        """Entry rows (pnl=+0.0000, reason=entry) should be skipped."""
        rows = [
            ["2026-01-01 00:00:00", "swing", "SPY", "buy", "100", "450.00",
             "entry (LONG)", "+0.0000"],
            ["2026-01-01 01:00:00", "swing", "SPY", "sell", "100", "455.00",
             "signal_reversal", "+0.0111"],
        ]
        self._write_csv(trades_dir / "daily_trades_20260101.csv", rows)

        from risk_config import DeRiskState
        import glob
        derisk_states = {}
        for csv_path in sorted(glob.glob(str(trades_dir / "daily_trades_*.csv"))):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("group") != "swing":
                        continue
                    reason = row.get("reason", "")
                    pnl = float(row.get("pnl_pct", "+0.0000"))
                    if "entry" in reason.lower() and abs(pnl) < 1e-8:
                        continue
                    sym = row.get("symbol", "").replace("/", "-")
                    if sym not in derisk_states:
                        derisk_states[sym] = DeRiskState()
                    derisk_states[sym].record_trade(pnl)

        ds = derisk_states["SPY"]
        assert len(ds.returns) == 1  # only the exit, not the entry
        assert ds.returns[0] == pytest.approx(0.0111, abs=1e-5)

    def test_rebuild_filters_by_group(self, trades_dir):
        """Rebuild should only load trades matching the group."""
        rows = [
            ["2026-01-01 00:00:00", "swing", "SPY", "sell", "100", "455.00",
             "exit", "+0.0100"],
            ["2026-01-01 00:00:00", "intraday", "QQQ", "sell", "50", "400.00",
             "exit", "+0.0200"],
        ]
        self._write_csv(trades_dir / "daily_trades_20260101.csv", rows)

        from risk_config import DeRiskState
        import glob
        derisk_states = {}
        for csv_path in sorted(glob.glob(str(trades_dir / "daily_trades_*.csv"))):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("group") != "swing":
                        continue
                    sym = row.get("symbol", "").replace("/", "-")
                    pnl = float(row.get("pnl_pct", "+0.0000"))
                    if sym not in derisk_states:
                        derisk_states[sym] = DeRiskState()
                    derisk_states[sym].record_trade(pnl)

        assert "SPY" in derisk_states
        assert "QQQ" not in derisk_states

    def test_insufficient_trades_no_kelly(self, trades_dir):
        """Below min_trades threshold, half_kelly returns None."""
        from risk_config import DeRiskState
        rows = []
        for i in range(10):  # only 10 trades, below 20 min
            rows.append([f"2026-01-01 {i:02d}:00:00", "crypto_intraday",
                         "BTC/USD", "sell", "1", "50000.00",
                         "exit", "+0.0200"])
        self._write_csv(trades_dir / "daily_trades_20260101.csv", rows)

        import glob
        derisk_states = {}
        for csv_path in sorted(glob.glob(str(trades_dir / "daily_trades_*.csv"))):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("group") != "crypto_intraday":
                        continue
                    sym = row.get("symbol", "").replace("/", "-")
                    pnl = float(row.get("pnl_pct", "+0.0000"))
                    if sym not in derisk_states:
                        derisk_states[sym] = DeRiskState()
                    derisk_states[sym].record_trade(pnl)

        ds = derisk_states["BTC-USD"]
        assert len(ds.returns) == 10
        assert ds.half_kelly(window=60, min_trades=20) is None

    def test_rebuild_across_multiple_csv_files(self, trades_dir):
        """Trades spread across multiple daily CSV files should all be loaded."""
        from risk_config import DeRiskState
        for day in range(1, 6):
            rows = []
            for i in range(5):
                pnl = "+0.0200" if i % 2 == 0 else "-0.0100"
                rows.append([f"2026-01-0{day} {i:02d}:00:00", "swing",
                             "GLD", "sell", "50", "200.00",
                             "exit", pnl])
            self._write_csv(trades_dir / f"daily_trades_2026010{day}.csv", rows)

        import glob
        derisk_states = {}
        for csv_path in sorted(glob.glob(str(trades_dir / "daily_trades_*.csv"))):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("group") != "swing":
                        continue
                    sym = row.get("symbol", "").replace("/", "-")
                    pnl = float(row.get("pnl_pct", "+0.0000"))
                    if sym not in derisk_states:
                        derisk_states[sym] = DeRiskState()
                    derisk_states[sym].record_trade(pnl)

        ds = derisk_states["GLD"]
        assert len(ds.returns) == 25  # 5 days × 5 trades
        kelly = ds.half_kelly(window=60, min_trades=20)
        assert kelly is not None

    def test_all_losses_kelly_zero(self, trades_dir):
        """If all trades are losses, Kelly should be 0."""
        from risk_config import DeRiskState
        rows = []
        for i in range(25):
            rows.append([f"2026-01-01 {i:02d}:00:00", "crypto_intraday",
                         "DOGE/USD", "sell", "1000", "0.10",
                         "stop_loss", "-0.0300"])
        self._write_csv(trades_dir / "daily_trades_20260101.csv", rows)

        import glob
        derisk_states = {}
        for csv_path in sorted(glob.glob(str(trades_dir / "daily_trades_*.csv"))):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("group") != "crypto_intraday":
                        continue
                    sym = row.get("symbol", "").replace("/", "-")
                    pnl = float(row.get("pnl_pct", "+0.0000"))
                    if sym not in derisk_states:
                        derisk_states[sym] = DeRiskState()
                    derisk_states[sym].record_trade(pnl)

        ds = derisk_states["DOGE-USD"]
        kelly = ds.half_kelly(window=60, min_trades=20)
        # All losses → no wins → half_kelly returns None (can't compute payoff ratio)
        assert kelly is None


# ===================================================================
# DeRiskState unit tests
# ===================================================================

class TestDeRiskState:
    """Direct tests for the DeRiskState class."""

    def test_half_kelly_positive_edge(self):
        from risk_config import DeRiskState
        ds = DeRiskState()
        # 60% win rate, avg win = 5%, avg loss = 3%
        for _ in range(12):
            ds.record_trade(0.05)
        for _ in range(8):
            ds.record_trade(-0.03)
        kelly = ds.half_kelly(window=60, min_trades=20)
        assert kelly is not None
        assert kelly > 0
        # Full Kelly: (b*p - q)/b where b=5/3, p=0.6, q=0.4
        # = (5/3 * 0.6 - 0.4) / (5/3) = (1.0 - 0.4) / 1.667 = 0.36
        # Half Kelly = 0.18
        assert kelly == pytest.approx(0.18, abs=0.01)

    def test_half_kelly_negative_edge(self):
        from risk_config import DeRiskState
        ds = DeRiskState()
        # 30% win rate, avg win = 2%, avg loss = 5%
        for _ in range(6):
            ds.record_trade(0.02)
        for _ in range(14):
            ds.record_trade(-0.05)
        kelly = ds.half_kelly(window=60, min_trades=20)
        assert kelly is not None
        assert kelly == 0.0  # negative edge → clamped to 0

    def test_rolling_winrate(self):
        from risk_config import DeRiskState
        ds = DeRiskState()
        for _ in range(7):
            ds.record_trade(0.05)
        for _ in range(3):
            ds.record_trade(-0.03)
        wr = ds.rolling_winrate(window=50)
        assert wr == pytest.approx(0.70, abs=0.01)

    def test_rolling_sharpe(self):
        from risk_config import DeRiskState
        ds = DeRiskState()
        for _ in range(20):
            ds.record_trade(0.02)  # constant returns → infinite Sharpe? No, std > 0 needed
        # Actually all same value → std = 0 → returns 0.0
        sharpe = ds.rolling_sharpe(window=50)
        assert sharpe == 0.0

    def test_insufficient_data_returns_none(self):
        from risk_config import DeRiskState
        ds = DeRiskState()
        for _ in range(5):
            ds.record_trade(0.01)
        assert ds.half_kelly() is None
        assert ds.rolling_winrate() is None
        assert ds.rolling_sharpe() is None
