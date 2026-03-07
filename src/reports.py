#!/usr/bin/env python3
"""
Quantitative Stocks — Live Log Viewer
======================================
Parses the latest paper-trader log with pandas and produces a clean
self-contained HTML page showing:
  • Account equity chart for the session
  • Current status per symbol (action, ML signal, ADX, P&L)
  • Action counts (HOLD / BUY / SHORT / SKIP / EXIT) per symbol

Usage:
    python reports.py            # writes outputs/report.html
    python reports.py --open     # writes + opens in browser
    python reports.py --log <path>   # use a specific log file
"""

from __future__ import annotations

import argparse
import os
import re
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).parent
LOGS_DIR     = PROJECT_ROOT / "logs"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
REPORT_PATH  = OUTPUTS_DIR / "report.html"

# ── colour palette ─────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
BLUE   = "#58a6ff"
MUTED  = "#8b949e"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Find the latest log (by timestamp in filename, not alphabetical order)
# ═══════════════════════════════════════════════════════════════════════════════

_TS_RE = re.compile(r"paper_trader_(\d{8}_\d{6})\.log$")


def latest_log_file() -> Optional[Path]:
    """Return the log file with the newest timestamp in its name (not mtime)."""
    candidates = []
    for p in LOGS_DIR.glob("paper_trader_*.log"):
        if "_err" in p.name:
            continue
        m = _TS_RE.search(p.name)
        if m:
            candidates.append((m.group(1), p))   # (YYYYMMDD_HHMMSS, path)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Log parser  (plain-text → pandas DataFrames)
# ═══════════════════════════════════════════════════════════════════════════════

_CYCLE_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\].*Cycle #(\d+)")
_ACCT_RE  = re.compile(r"Account:\s+\$([0-9,]+\.?\d*)\s+equity")
_SYM_RE   = re.compile(
    r"^\s{2,4}(\w+):\s+"                                   # symbol
    r"(HOLD|BUY|SELL|SHORT|EXIT|SKIP|FLIP\S*)"             # action verb
    r".*?ML:\s+(\w+)\s+([\d.]+)"                           # ML dir + conf
    r"(?:.*?ADX=([\d.]+))?",                               # optional ADX
    re.MULTILINE,
)
_PNL_RE   = re.compile(r"P&L:\s+([-+]?[\d.]+)%")


def parse_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse log → (account_df[time, cycle, equity], sym_df[time, cycle, ...])."""
    text  = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    m = re.search(r"paper_trader_(\d{4})(\d{2})(\d{2})_", path.name)
    log_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "unknown"

    acct_rows, sym_rows = [], []
    cur_cycle, cur_time = 0, ""

    for i, line in enumerate(lines):
        cm = _CYCLE_RE.search(line)
        if cm:
            cur_time  = f"{log_date} {cm.group(1)}"
            cur_cycle = int(cm.group(2))
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            am  = _ACCT_RE.search(nxt)
            if am:
                acct_rows.append({
                    "time":   cur_time,
                    "cycle":  cur_cycle,
                    "equity": float(am.group(1).replace(",", "")),
                })
            continue

        sm = _SYM_RE.search(line)
        if sm and cur_cycle:
            pnl_m = _PNL_RE.search(line)
            sym_rows.append({
                "time":    cur_time,
                "cycle":   cur_cycle,
                "symbol":  sm.group(1),
                "action":  sm.group(2),
                "ml_dir":  sm.group(3),
                "ml_conf": float(sm.group(4)),
                "adx":     float(sm.group(5)) if sm.group(5) else None,
                "pnl_pct": float(pnl_m.group(1)) if pnl_m else None,
            })

    acct_df = pd.DataFrame(acct_rows)
    sym_df  = pd.DataFrame(sym_rows)

    if not acct_df.empty:
        acct_df["time"] = pd.to_datetime(acct_df["time"], errors="coerce")
    if not sym_df.empty:
        sym_df["time"] = pd.to_datetime(sym_df["time"], errors="coerce")

    return acct_df, sym_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HTML helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _act_badge(action: str) -> str:
    a = action.upper()
    if "HOLD"  in a: css, label = "act-hold",  "HOLD"
    elif "BUY" in a: css, label = "act-buy",   "BUY"
    elif "SHORT" in a: css, label = "act-short", "SHORT"
    elif "SKIP" in a: css, label = "act-skip",  "SKIP"
    elif "EXIT" in a: css, label = "act-exit",  "EXIT"
    elif "FLIP" in a: css, label = "act-flip",  action[:8]
    else:             css, label = "act-skip",  action[:8]
    return f'<span class="act {css}">{label}</span>'


def _pct(v: float) -> str:
    cls = "pos" if v > 0 else "neg" if v < 0 else "neu"
    return f'<span class="{cls}">{v:+.2f}%</span>'


def _plotly_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Build the HTML page
# ═══════════════════════════════════════════════════════════════════════════════

def build_html(log_path: Path) -> str:
    acct_df, sym_df = parse_log(log_path)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    if not acct_df.empty:
        start_eq = acct_df["equity"].iloc[0]
        latest_eq = acct_df["equity"].iloc[-1]
        delta_eq  = latest_eq - start_eq
        cycles    = int(acct_df["cycle"].max())
        period_s  = str(acct_df["time"].min())[:16]
        period_e  = str(acct_df["time"].max())[:16]
        eq_class  = "green" if delta_eq >= 0 else "red"
    else:
        start_eq = latest_eq = delta_eq = 0.0
        cycles = 0
        period_s = period_e = "—"
        eq_class = "neu"

    symbols = sorted(sym_df["symbol"].unique()) if not sym_df.empty else []

    kpi = f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-label">Equity</div>
    <div class="kpi-value">${latest_eq:,.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Session P&L</div>
    <div class="kpi-value {eq_class}">${delta_eq:+,.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Cycles</div>
    <div class="kpi-value blue">{cycles}</div></div>
  <div class="kpi"><div class="kpi-label">Symbols</div>
    <div class="kpi-value blue">{len(symbols)}</div></div>
  <div class="kpi"><div class="kpi-label">Period</div>
    <div class="kpi-value small">{period_s}<br>{period_e}</div></div>
</div>"""

    # ── Equity chart ──────────────────────────────────────────────────────────
    if not acct_df.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=acct_df["time"], y=acct_df["equity"],
            mode="lines", name="Equity",
            line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.07)",
            hovertemplate="<b>$%{y:,.2f}</b>  %{x}<extra></extra>",
        ))
        fig_eq.add_hline(
            y=start_eq, line_dash="dot", line_color=MUTED, line_width=1,
            annotation_text=f"Start ${start_eq:,.0f}",
            annotation_font_color=MUTED,
        )
        fig_eq.update_layout(
            template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
            height=260, margin=dict(l=70, r=24, t=32, b=40),
            yaxis_title="Equity ($)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor=BORDER),
            showlegend=False,
        )
        eq_chart = f'<div class="chart">{_plotly_div(fig_eq, "eq_live")}</div>'
    else:
        eq_chart = '<p class="empty">No equity data parsed.</p>'

    # ── Current status table ──────────────────────────────────────────────────
    if not sym_df.empty:
        latest = (sym_df.sort_values("cycle")
                        .groupby("symbol", sort=False)
                        .last()
                        .reset_index()
                        .sort_values("symbol"))

        rows = ""
        for _, r in latest.iterrows():
            adx  = f"{r['adx']:.0f}" if pd.notna(r.get("adx")) else "—"
            pnl  = _pct(r["pnl_pct"]) if pd.notna(r.get("pnl_pct")) else '<span class="neu">—</span>'
            ml_c = "pos" if r["ml_dir"] == "UP" else "neg" if r["ml_dir"] == "DOWN" else "neu"
            rows += f"""<tr>
              <td><b>{r['symbol']}</b></td>
              <td>{_act_badge(r['action'])}</td>
              <td class="{ml_c}">{r['ml_dir']}</td>
              <td>{r['ml_conf']:.2f}</td>
              <td>{adx}</td>
              <td>{pnl}</td>
            </tr>"""

        status_table = f"""
<div class="tbl-wrap"><table>
  <thead><tr><th>Symbol</th><th>Action</th><th>ML Dir</th>
              <th>ML Conf</th><th>ADX</th><th>P&L %</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>"""

        # ── Action counts chart ────────────────────────────────────────────────
        counts = (sym_df.groupby(["symbol", "action"])
                        .size()
                        .unstack(fill_value=0))
        action_colors = {
            "HOLD":  BLUE,
            "BUY":   GREEN,
            "SHORT": RED,
            "SKIP":  MUTED,
            "EXIT":  YELLOW,
        }
        fig_bar = go.Figure()
        for act in counts.columns:
            fig_bar.add_trace(go.Bar(
                name=act,
                x=counts.index,
                y=counts[act],
                marker_color=action_colors.get(act.upper(), "#bc8cff"),
                hovertemplate=f"<b>%{{x}}</b> — {act}: %{{y}}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="stack",
            template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
            height=280, margin=dict(l=50, r=24, t=32, b=48),
            yaxis_title="Cycles",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(tickfont=dict(size=12)),
        )
        action_chart = f'<div class="chart">{_plotly_div(fig_bar, "action_counts")}</div>'

        # ── P&L over time per symbol ───────────────────────────────────────────
        pnl_data = sym_df.dropna(subset=["pnl_pct"])
        if not pnl_data.empty:
            COLORS = ["#58a6ff","#3fb950","#f0883e","#d29922","#bc8cff",
                      "#ff7b72","#79c0ff","#56d364","#ffa657","#e3b341"]
            fig_pnl = go.Figure()
            for i, sym in enumerate(sorted(pnl_data["symbol"].unique())):
                sdf = pnl_data[pnl_data["symbol"] == sym].sort_values("cycle")
                fig_pnl.add_trace(go.Scatter(
                    x=sdf["time"], y=sdf["pnl_pct"],
                    mode="lines", name=sym,
                    line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                    hovertemplate=f"<b>{sym}</b> %{{y:+.2f}}%<extra></extra>",
                ))
            fig_pnl.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
            fig_pnl.update_layout(
                template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
                height=300, margin=dict(l=56, r=24, t=32, b=40),
                yaxis_title="Open P&L (%)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            pnl_chart = f'<div class="chart">{_plotly_div(fig_pnl, "pnl_time")}</div>'
        else:
            pnl_chart = ""
    else:
        status_table = '<p class="empty">No symbol data parsed.</p>'
        action_chart = ""
        pnl_chart    = ""

    # ── Assemble ──────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Log — {log_path.name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{BG};color:{TEXT};font-family:'Segoe UI',system-ui,sans-serif;
       font-size:14px;line-height:1.5}}

  .header{{background:{PANEL};border-bottom:1px solid {BORDER};
           padding:16px 28px;display:flex;align-items:center;gap:16px}}
  .header h1{{font-size:20px;font-weight:600}}
  .header .meta{{color:{MUTED};font-size:12px}}
  .badge{{background:{BLUE}22;color:{BLUE};border:1px solid {BLUE}44;
          border-radius:4px;padding:2px 8px;font-size:12px}}

  .body{{padding:24px 28px;max-width:1400px;margin:0 auto}}

  .kpi-row{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:28px}}
  .kpi{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;
        padding:14px 18px;min-width:140px}}
  .kpi-label{{color:{MUTED};font-size:11px;text-transform:uppercase;
              letter-spacing:.6px;margin-bottom:4px}}
  .kpi-value{{font-size:20px;font-weight:700}}
  .kpi-value.green{{color:{GREEN}}}
  .kpi-value.red{{color:{RED}}}
  .kpi-value.blue{{color:{BLUE}}}
  .kpi-value.small{{font-size:13px;color:{TEXT}}}

  .section{{margin-bottom:32px}}
  .section-title{{font-size:14px;font-weight:600;margin-bottom:12px;
                  padding-left:10px;border-left:3px solid {BLUE};color:{TEXT}}}

  .chart{{border:1px solid {BORDER};border-radius:8px;overflow:hidden;
          background:{PANEL};margin-bottom:0}}

  .tbl-wrap{{overflow-x:auto;border:1px solid {BORDER};border-radius:8px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:{PANEL};color:{MUTED};font-weight:500;padding:10px 14px;
      text-align:left;border-bottom:1px solid {BORDER};white-space:nowrap}}
  td{{padding:9px 14px;border-bottom:1px solid {BORDER}22}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:{PANEL}66}}

  .pos{{color:{GREEN}}} .neg{{color:{RED}}} .neu{{color:{MUTED}}}

  .act{{display:inline-block;padding:2px 7px;border-radius:4px;
        font-size:11px;font-weight:600;letter-spacing:.4px}}
  .act-hold {{background:#58a6ff22;color:{BLUE}}}
  .act-buy  {{background:#3fb95022;color:{GREEN}}}
  .act-short{{background:#f8514922;color:{RED}}}
  .act-skip {{background:#8b949e22;color:{MUTED}}}
  .act-exit {{background:#d2992222;color:{YELLOW}}}
  .act-flip {{background:#bc8cff22;color:#bc8cff}}

  .empty{{color:{MUTED};padding:16px 0}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Live Paper-Trader Log</h1>
    <div class="meta">{log_path.name} &nbsp;·&nbsp;
      Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  </div>
  <span class="badge">Paper Trading</span>
</div>

<div class="body">
  {kpi}

  <div class="section">
    <div class="section-title">Account Equity</div>
    {eq_chart}
  </div>

  <div class="section">
    <div class="section-title">Open P&L per Symbol Over Time</div>
    {pnl_chart if pnl_chart else '<p class="empty">No open positions with P&L data.</p>'}
  </div>

  <div class="section">
    <div class="section-title">Current Status (latest cycle per symbol)</div>
    {status_table}
  </div>

  <div class="section">
    <div class="section-title">Action Counts This Session</div>
    {action_chart}
  </div>
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Live log HTML viewer")
    parser.add_argument("--open",  action="store_true",
                        help="Open in browser after generating")
    parser.add_argument("--log",   type=str, default=None,
                        help="Path to a specific log file (default: latest)")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else latest_log_file()
    if log_path is None:
        print("No paper-trader log files found in logs/")
        raise SystemExit(1)

    print(f"Parsing: {log_path.name}")
    html = build_html(log_path)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report saved -> {REPORT_PATH}")

    if args.open:
        webbrowser.open(REPORT_PATH.as_uri())


if __name__ == "__main__":
    main()
