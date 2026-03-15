#!/usr/bin/env python3
"""
Divergence Report — Backtest vs Live paper-trading comparison.

Compares backtest summary JSONs with live daily trade CSVs and produces
a self-contained HTML report highlighting significant divergences.

Usage:
    python main.py divergence-report --group crypto
    python main.py divergence-report --group swing --output outputs/div_swing.html
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root (same convention as other src/ modules)
# ---------------------------------------------------------------------------
try:
    from signals_engine import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# Symbol groups — import from paper_trader when available, else hardcode
try:
    from paper_trader import SYMBOL_GROUPS
except ImportError:
    SYMBOL_GROUPS: Dict[str, List[str]] = {
        "intraday": ["SMH", "IWM", "IGV", "QQQ", "EWT", "SOXX"],
        "swing":    ["GDX", "SLV", "IGV", "QQQ", "GLD", "SMH", "XLK", "IBIT"],
        "crypto":   ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
                      "DOGE/USD", "DOT/USD", "SUSHI/USD", "ADA/USD",
                      "CRV/USD", "AAVE/USD", "RENDER/USD"],
    }

DEFAULT_OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Helper: normalise crypto symbol formats
# ---------------------------------------------------------------------------
def _to_yahoo(symbol: str) -> str:
    """BTC/USD -> BTC-USD (for file matching)."""
    return symbol.replace("/", "-")


def _find_backtest_summary(symbol: str, backtest_dir: str) -> Optional[str]:
    """Return path to backtest_<symbol>_summary.json if it exists."""
    yf = _to_yahoo(symbol)
    path = os.path.join(backtest_dir, f"backtest_{yf}_summary.json")
    return path if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SymbolMetrics:
    """Comparison metrics for one symbol."""
    symbol: str
    # Backtest
    bt_win_rate: Optional[float] = None
    bt_avg_return: Optional[float] = None       # avg_win weighted by WR + avg_loss weighted by (1-WR)
    bt_sharpe: Optional[float] = None
    bt_avg_holding: Optional[float] = None       # days
    bt_trades_total: Optional[int] = None
    bt_trades_per_week: Optional[float] = None
    bt_start: Optional[str] = None
    bt_end: Optional[str] = None
    # Live
    live_win_rate: Optional[float] = None
    live_avg_return: Optional[float] = None
    live_sharpe: Optional[float] = None
    live_avg_holding: Optional[float] = None
    live_trades_total: Optional[int] = None
    live_trades_per_week: Optional[float] = None
    live_start: Optional[str] = None
    live_end: Optional[str] = None
    # Divergences (metric_name -> pct_diff)
    divergences: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DivergenceReport
# ---------------------------------------------------------------------------
class DivergenceReport:
    """Load backtest + live results and compute per-symbol divergences."""

    DIVERGENCE_THRESHOLD = 0.20  # flag if >20% relative diff

    def __init__(self, backtest_dir: str = DEFAULT_OUTPUTS,
                 trades_dir: str = DEFAULT_OUTPUTS,
                 symbols: Optional[List[str]] = None):
        self.backtest_dir = backtest_dir
        self.trades_dir = trades_dir
        self.symbols = symbols or []
        self.metrics: List[SymbolMetrics] = []

    # ------------------------------------------------------------------
    # Load backtest summary
    # ------------------------------------------------------------------
    def _load_backtest(self, symbol: str) -> Optional[dict]:
        path = _find_backtest_summary(symbol, self.backtest_dir)
        if not path:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Load live trades from daily_trades_*.csv
    # ------------------------------------------------------------------
    def _load_live_trades(self, symbol: str) -> pd.DataFrame:
        """Concatenate all daily_trades_YYYYMMDD.csv and filter for symbol."""
        import glob
        pattern = os.path.join(self.trades_dir, "daily_trades_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return pd.DataFrame()

        frames = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()

        all_trades = pd.concat(frames, ignore_index=True)
        # Normalise symbol column: BTC/USD, BTC-USD, BTCUSD all match
        yf = _to_yahoo(symbol)
        mask = all_trades["symbol"].apply(
            lambda s: _to_yahoo(str(s).replace("USD", "-USD")
                                if str(s).endswith("USD") and "/" not in str(s) and "-" not in str(s)
                                else str(s).replace("/", "-"))
        ) == yf
        # Also try direct match
        mask = mask | (all_trades["symbol"] == symbol)
        return all_trades[mask].copy()

    # ------------------------------------------------------------------
    # Compute live metrics from trade rows
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_live_metrics(df: pd.DataFrame) -> dict:
        """Compute win rate, avg return, approx Sharpe, trades/week from live trade rows."""
        # Only look at exit rows (pnl_pct != 0 and reason contains 'exit' or 'stop' or 'decay' or 'lock')
        exits = df[
            df["reason"].str.contains("exit|stop|decay|lock|underwater", case=False, na=False)
            & (df["pnl_pct"] != 0)
        ].copy()
        if exits.empty:
            return {}

        # Parse pnl_pct — may be stored as "+0.0312" string or numeric
        exits["pnl_val"] = pd.to_numeric(
            exits["pnl_pct"].astype(str).str.replace("+", "", regex=False),
            errors="coerce",
        )
        exits = exits.dropna(subset=["pnl_val"])
        if exits.empty:
            return {}

        n = len(exits)
        wins = (exits["pnl_val"] > 0).sum()
        win_rate = wins / n if n > 0 else 0.0
        avg_ret = exits["pnl_val"].mean()

        # Approx Sharpe: mean / std (daily-return proxy — not annualised)
        std = exits["pnl_val"].std()
        sharpe = (avg_ret / std) if std > 0 else 0.0

        # Date range & trades per week
        exits["ts"] = pd.to_datetime(exits["timestamp"], errors="coerce")
        ts_valid = exits.dropna(subset=["ts"])
        if len(ts_valid) >= 2:
            span_days = (ts_valid["ts"].max() - ts_valid["ts"].min()).days or 1
            trades_per_week = n / (span_days / 7.0)
            start = str(ts_valid["ts"].min().date())
            end = str(ts_valid["ts"].max().date())
        else:
            trades_per_week = 0.0
            start = end = "N/A"

        return {
            "win_rate": win_rate,
            "avg_return": avg_ret,
            "sharpe": sharpe,
            "trades_total": n,
            "trades_per_week": trades_per_week,
            "start": start,
            "end": end,
        }

    # ------------------------------------------------------------------
    # Relative divergence
    # ------------------------------------------------------------------
    @staticmethod
    def _rel_diff(bt_val: Optional[float], live_val: Optional[float]) -> Optional[float]:
        """Return relative difference: |bt - live| / max(|bt|, |live|, 1e-9)."""
        if bt_val is None or live_val is None:
            return None
        denom = max(abs(bt_val), abs(live_val), 1e-9)
        return abs(bt_val - live_val) / denom

    # ------------------------------------------------------------------
    # Build report data
    # ------------------------------------------------------------------
    def compute(self) -> List[SymbolMetrics]:
        self.metrics = []
        for symbol in self.symbols:
            m = SymbolMetrics(symbol=symbol)

            # --- Backtest ---
            bt = self._load_backtest(symbol)
            if bt:
                m.bt_win_rate = bt.get("win_rate")
                # Weighted avg return per trade
                wr = bt.get("win_rate", 0)
                avg_win = bt.get("avg_win_pct", 0)
                avg_loss = bt.get("avg_loss_pct", 0)
                m.bt_avg_return = wr * avg_win + (1 - wr) * avg_loss
                m.bt_sharpe = bt.get("sharpe_ratio")
                m.bt_avg_holding = bt.get("avg_trade_duration_days")
                m.bt_trades_total = bt.get("total_trades")
                # Approx trades per week
                try:
                    sd = datetime.strptime(bt["start_date"], "%Y-%m-%d")
                    ed = datetime.strptime(bt["end_date"], "%Y-%m-%d")
                    span_weeks = max((ed - sd).days / 7.0, 1)
                    m.bt_trades_per_week = (bt.get("total_trades", 0)) / span_weeks
                except Exception:
                    m.bt_trades_per_week = None
                m.bt_start = bt.get("start_date")
                m.bt_end = bt.get("end_date")

            # --- Live ---
            live_df = self._load_live_trades(symbol)
            lm = self._compute_live_metrics(live_df)
            if lm:
                m.live_win_rate = lm["win_rate"]
                m.live_avg_return = lm["avg_return"]
                m.live_sharpe = lm["sharpe"]
                m.live_trades_total = lm["trades_total"]
                m.live_trades_per_week = lm["trades_per_week"]
                m.live_start = lm["start"]
                m.live_end = lm["end"]

            # --- Flag divergences ---
            checks = [
                ("win_rate", m.bt_win_rate, m.live_win_rate),
                ("avg_return", m.bt_avg_return, m.live_avg_return),
                ("sharpe", m.bt_sharpe, m.live_sharpe),
                ("trades/wk", m.bt_trades_per_week, m.live_trades_per_week),
            ]
            for name, bv, lv in checks:
                rd = self._rel_diff(bv, lv)
                if rd is not None and rd > self.DIVERGENCE_THRESHOLD:
                    m.divergences[name] = rd

            self.metrics.append(m)

        return self.metrics

    # ------------------------------------------------------------------
    # Generate HTML
    # ------------------------------------------------------------------
    def generate_html(self) -> str:
        if not self.metrics:
            self.compute()

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        rows_html = []
        for m in self.metrics:
            has_bt = m.bt_win_rate is not None
            has_live = m.live_win_rate is not None
            flags = ", ".join(
                f"{k} ({v:.0%})" for k, v in m.divergences.items()
            ) if m.divergences else ""
            row_class = "divergent" if m.divergences else ""

            def _fmt(val, fmt=".2f", pct=False):
                if val is None:
                    return '<span class="na">--</span>'
                s = f"{val:{fmt}}"
                if pct:
                    s += "%"
                return s

            def _fmt_pct(val):
                if val is None:
                    return '<span class="na">--</span>'
                return f"{val:.1%}"

            rows_html.append(f"""
            <tr class="{row_class}">
                <td class="symbol">{m.symbol}</td>
                <td>{_fmt_pct(m.bt_win_rate)}</td>
                <td>{_fmt_pct(m.live_win_rate)}</td>
                <td>{_fmt(m.bt_avg_return, '.2f')}%</td>
                <td>{_fmt(m.live_avg_return, '.4f')}</td>
                <td>{_fmt(m.bt_sharpe)}</td>
                <td>{_fmt(m.live_sharpe)}</td>
                <td>{_fmt(m.bt_trades_per_week)}</td>
                <td>{_fmt(m.live_trades_per_week)}</td>
                <td>{m.bt_trades_total if m.bt_trades_total is not None else '--'}</td>
                <td>{m.live_trades_total if m.live_trades_total is not None else '--'}</td>
                <td class="flags">{flags or '<span class="ok">OK</span>'}</td>
            </tr>""")

        # Count summaries
        total = len(self.metrics)
        with_bt = sum(1 for m in self.metrics if m.bt_win_rate is not None)
        with_live = sum(1 for m in self.metrics if m.live_win_rate is not None)
        with_diverg = sum(1 for m in self.metrics if m.divergences)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest vs Live Divergence Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px;
    }}
    h1 {{ color: #58a6ff; margin-bottom: 4px; }}
    .subtitle {{ color: #8b949e; font-size: 0.9em; margin-bottom: 20px; }}
    .summary {{ display: flex; gap: 24px; margin-bottom: 20px; }}
    .summary .card {{
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 20px; min-width: 120px;
    }}
    .summary .card .label {{ font-size: 0.8em; color: #8b949e; }}
    .summary .card .value {{ font-size: 1.4em; font-weight: 600; }}
    table {{
        border-collapse: collapse; width: 100%; background: #161b22;
        border: 1px solid #30363d; border-radius: 8px; overflow: hidden;
    }}
    th {{
        background: #21262d; color: #58a6ff; text-align: left;
        padding: 10px 12px; font-size: 0.85em; border-bottom: 2px solid #30363d;
    }}
    td {{
        padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 0.9em;
        font-variant-numeric: tabular-nums;
    }}
    tr:hover {{ background: #1c2128; }}
    tr.divergent {{ background: #2d1b1b; }}
    tr.divergent:hover {{ background: #3b2222; }}
    .symbol {{ font-weight: 600; color: #f0f6fc; }}
    .na {{ color: #484f58; }}
    .flags {{ color: #f85149; font-size: 0.85em; }}
    .ok {{ color: #3fb950; }}
    .note {{
        margin-top: 16px; font-size: 0.8em; color: #8b949e;
        border-top: 1px solid #30363d; padding-top: 12px;
    }}
</style>
</head>
<body>
<h1>Backtest vs Live Divergence Report</h1>
<div class="subtitle">Generated {now}</div>

<div class="summary">
    <div class="card"><div class="label">Symbols</div><div class="value">{total}</div></div>
    <div class="card"><div class="label">With Backtest</div><div class="value">{with_bt}</div></div>
    <div class="card"><div class="label">With Live Trades</div><div class="value">{with_live}</div></div>
    <div class="card"><div class="label">Divergences</div>
        <div class="value" style="color: {'#f85149' if with_diverg else '#3fb950'}">{with_diverg}</div>
    </div>
</div>

<table>
<thead>
<tr>
    <th rowspan="2">Symbol</th>
    <th colspan="2">Win Rate</th>
    <th colspan="2">Avg Return/Trade</th>
    <th colspan="2">Sharpe</th>
    <th colspan="2">Trades/Week</th>
    <th colspan="2">Total Trades</th>
    <th rowspan="2">Flags</th>
</tr>
<tr>
    <th>BT</th><th>Live</th>
    <th>BT</th><th>Live</th>
    <th>BT</th><th>Live</th>
    <th>BT</th><th>Live</th>
    <th>BT</th><th>Live</th>
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>

<div class="note">
    <strong>Divergence threshold:</strong> &gt;20% relative difference in any metric is flagged.<br>
    <strong>BT</strong> = Backtest (from backtest_*_summary.json) &nbsp;|&nbsp;
    <strong>Live</strong> = Paper trading (from daily_trades_*.csv)<br>
    Live Sharpe is per-trade (not annualised). BT avg return is win-rate-weighted (WR * avg_win + (1-WR) * avg_loss).
</div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # Write to file
    # ------------------------------------------------------------------
    def write_html(self, output_path: str) -> str:
        html = self.generate_html()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate backtest-vs-live divergence HTML report.",
    )
    parser.add_argument(
        "--group", default=None,
        choices=list(SYMBOL_GROUPS.keys()),
        help="Filter symbols to a trading group (swing, intraday, crypto).",
    )
    parser.add_argument(
        "--output", default=os.path.join(DEFAULT_OUTPUTS, "divergence_report.html"),
        help="Output HTML path (default: outputs/divergence_report.html).",
    )
    parser.add_argument(
        "--backtest-dir", default=DEFAULT_OUTPUTS,
        help="Directory containing backtest_*_summary.json files.",
    )
    parser.add_argument(
        "--trades-dir", default=DEFAULT_OUTPUTS,
        help="Directory containing daily_trades_*.csv files.",
    )
    args = parser.parse_args()

    if args.group:
        symbols = SYMBOL_GROUPS[args.group]
    else:
        # All groups combined (deduplicated, preserving order)
        seen = set()
        symbols = []
        for grp in SYMBOL_GROUPS.values():
            for s in grp:
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)

    report = DivergenceReport(
        backtest_dir=args.backtest_dir,
        trades_dir=args.trades_dir,
        symbols=symbols,
    )
    report.compute()
    out = report.write_html(args.output)
    n_div = sum(1 for m in report.metrics if m.divergences)
    print(f"\n  Divergence report: {out}")
    print(f"  Symbols: {len(report.metrics)}, flagged: {n_div}\n")


if __name__ == "__main__":
    main()
