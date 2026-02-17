"""Streamlit dashboard for QuantitativeStocks backtest outputs.

Reads CSV outputs from `outputs/`:
- outputs/backtest_<SYMBOL>.csv (equity curve)
- outputs/trades_<SYMBOL>.csv  (trade blotter)

Run:
  pip install streamlit
  streamlit run streamlit_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class SymbolFiles:
    symbol: str
    equity_csv: Path
    trades_csv: Path | None


def _discover_symbols(output_dir: Path) -> list[SymbolFiles]:
    symbols: list[SymbolFiles] = []
    
    if not output_dir.exists():
        return symbols
        
    for equity_csv in sorted(output_dir.glob("backtest_*.csv")):
        # Skip chart html files that match the glob on some systems
        if equity_csv.name.endswith("_chart.html"):
            continue

        name = equity_csv.stem  # backtest_SPY
        if not name.startswith("backtest_"):
            continue
        symbol = name.removeprefix("backtest_")

        trades_csv = output_dir / f"trades_{symbol}.csv"
        symbols.append(
            SymbolFiles(
                symbol=symbol,
                equity_csv=equity_csv,
                trades_csv=trades_csv if trades_csv.exists() else None,
            )
        )

    return symbols


@st.cache_data(show_spinner=False)
def _load_equity_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError("Equity CSV must have columns: date,equity")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["equity"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def _load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("entry_price", "exit_price", "shares", "pnl", "return_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd


def _equity_fig(eq_df: pd.DataFrame) -> go.Figure:
    dd = _compute_drawdown(eq_df["equity"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq_df["date"],
            y=eq_df["equity"],
            mode="lines",
            name="Equity",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Equity Curve",
        hovermode="x unified",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)

    return fig


def _drawdown_fig(eq_df: pd.DataFrame) -> go.Figure:
    dd = _compute_drawdown(eq_df["equity"]) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq_df["date"],
            y=dd,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Drawdown",
        hovermode="x unified",
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def _metrics(eq_df: pd.DataFrame) -> dict[str, float]:
    initial = float(eq_df["equity"].iloc[0])
    final = float(eq_df["equity"].iloc[-1])
    total_return = (final / initial) - 1.0 if initial else 0.0

    days = max((eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days, 1)
    years = days / 365.25
    cagr = (final / initial) ** (1 / years) - 1.0 if initial and years > 0 else 0.0

    dd = _compute_drawdown(eq_df["equity"]).min() if len(eq_df) else 0.0
    return {
        "initial": initial,
        "final": final,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(dd),
    }


@st.cache_data(show_spinner=False)
def _load_price_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    # yfinance is already a dependency in this project; import lazily
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns in some configurations; flatten to strings.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x not in (None, "")]) for col in df.columns]

    df = df.reset_index()
    # yfinance's reset_index() may create 'Date', 'Datetime', 'date', or 'index'
    date_col = None
    for candidate in ("date", "Date", "Datetime", "index"):
        if candidate in df.columns:
            date_col = candidate
            break
    if not date_col:
        return pd.DataFrame()

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # auto_adjust=True usually provides 'Close' column
    if "Close" not in df.columns:
        return pd.DataFrame()

    return df[["date", "Close"]].rename(columns={"Close": "close"})


def _trade_fig(symbol: str, trade_row: pd.Series) -> go.Figure | None:
    entry_date = trade_row.get("entry_date")
    exit_date = trade_row.get("exit_date")
    entry_price = trade_row.get("entry_price")
    exit_price = trade_row.get("exit_price")

    if pd.isna(entry_date) or pd.isna(exit_date):
        return None

    start = (pd.to_datetime(entry_date) - timedelta(days=20)).date().isoformat()
    end = (pd.to_datetime(exit_date) + timedelta(days=20)).date().isoformat()

    px = _load_price_history(symbol, start=start, end=end)
    if px.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name=f"{symbol} (adj)"))

    fig.add_trace(
        go.Scatter(
            x=[entry_date],
            y=[entry_price],
            mode="markers",
            name="Entry",
            marker=dict(size=10, symbol="triangle-up"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[exit_date],
            y=[exit_price],
            mode="markers",
            name="Exit",
            marker=dict(size=10, symbol="x"),
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Selected Trade (Entry/Exit)",
        hovermode="x unified",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="QuantitativeStocks Backtest", layout="wide")

    st.title("QuantitativeStocks — Backtest Dashboard")
    st.caption(f"Reading outputs from: {OUTPUT_DIR}")

    symbols = _discover_symbols(OUTPUT_DIR)
    if not symbols:
        st.error("No backtest CSVs found in outputs/. Run `python main.py backtest --symbol SPY ...` first.")
        return

    symbol_list = [s.symbol for s in symbols]

    with st.sidebar:
        st.header("Controls")
        selected_symbol = st.selectbox("Symbol", symbol_list, index=0)
        show_trades = st.checkbox("Show trade blotter", value=True)
        show_trade_chart = st.checkbox("Per-trade chart", value=True)

    selected = next(s for s in symbols if s.symbol == selected_symbol)

    eq_df = _load_equity_curve(str(selected.equity_csv))
    m = _metrics(eq_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Initial", f"${m['initial']:,.0f}")
    c2.metric("Final", f"${m['final']:,.0f}")
    c3.metric("Total Return", f"{m['total_return']*100:+.2f}%")
    c4.metric("CAGR (approx)", f"{m['cagr']*100:+.2f}%")
    c5.metric("Max Drawdown", f"{m['max_drawdown']*100:.2f}%")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(_equity_fig(eq_df), width="stretch")
    with right:
        st.plotly_chart(_drawdown_fig(eq_df), width="stretch")

    if show_trades:
        if selected.trades_csv and selected.trades_csv.exists():
            trades_df = _load_trades(str(selected.trades_csv))
            st.subheader("Trades")
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

            if show_trade_chart and not trades_df.empty:
                st.subheader("Trade Chart")
                options = [
                    f"#{i+1} {row.get('direction','')} {row.get('entry_date','')} → {row.get('exit_date','')} ({row.get('return_pct',0):+.2f}%)"
                    for i, (_, row) in enumerate(trades_df.iterrows())
                ]
                pick = st.selectbox("Select trade", options, index=0)
                trade_index = options.index(pick)
                trade_row = trades_df.iloc[trade_index]

                fig = _trade_fig(selected_symbol, trade_row)
                if fig is None:
                    st.info("Could not load price history for the per-trade chart (yfinance returned no data).")
                else:
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("No trades CSV found for this symbol.")


if __name__ == "__main__":
    main()
