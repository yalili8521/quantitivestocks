"""Broker adapter — ABC interface + local paper trading implementation.

经纪商适配器：
- GoldBrokerAdapter: 抽象基类，定义所有经纪商必须实现的接口
- PaperGoldBroker: 本地模拟交易，用JSON文件保存状态
- 未来可以实现 IBKRBroker, NinjaTraderBroker 等

设计原则：引擎只通过这个接口与经纪商交互，完全解耦
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.gold_scalper.config import GoldScalperConfig

logger = logging.getLogger(__name__)


class GoldBrokerAdapter(ABC):
    """Abstract broker interface for gold trading.

    所有经纪商适配器必须实现这些方法。
    引擎只调用这些方法，不直接与任何经纪商API交互。
    """

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Return current price (mid-point of bid/ask)."""

    @abstractmethod
    def place_market_order(
        self, symbol: str, contracts: int, side: str
    ) -> Optional[str]:
        """Place a market order.

        Args:
            symbol: Instrument symbol.
            contracts: Number of contracts.
            side: "BUY" or "SELL".

        Returns:
            Order ID string, or None if failed.
        """

    @abstractmethod
    def close_partial(
        self, symbol: str, contracts: int, reason: str
    ) -> Optional[str]:
        """Close N contracts of an existing position.

        Args:
            symbol: Instrument symbol.
            contracts: How many contracts to close.
            reason: Why (for logging).

        Returns:
            Order ID string, or None if failed.
        """

    @abstractmethod
    def close_all(self, symbol: str, reason: str) -> Optional[str]:
        """Close entire position."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return current position details.

        Returns dict with: direction, entry_price, contracts, unrealized_pnl.
        Or None if flat.
        """

    @abstractmethod
    def get_account_equity(self) -> float:
        """Return current account equity in USD."""

    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, lookback: int
    ) -> pd.DataFrame:
        """Fetch OHLCV bars.

        Args:
            symbol: Instrument symbol.
            timeframe: "1d", "1h", "15m", "5m", etc.
            lookback: Number of bars to fetch.

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume.
        """


class PaperGoldBroker(GoldBrokerAdapter):
    """Local paper trading broker with JSON state persistence.

    本地模拟交易经纪商：
    - 用Yahoo Finance获取实时价格和K线数据
    - 订单立即以当前价格成交（无滑点模拟）
    - 状态保存到 outputs/paper_state/gold_scalper_*.json
    - 交易日志保存到 outputs/paper_state/gold_scalper_trade_log.json
    """

    def __init__(self, config: GoldScalperConfig, initial_equity: float = 5000.0):
        self.config = config
        self.symbol = config.symbol

        # State
        self._equity = initial_equity
        self._position: Optional[Dict] = None  # {direction, entry_price, contracts}
        self._trade_log: list = []
        self._order_counter = 0

        # State file paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        self._state_dir = os.path.join(project_root, "outputs", "paper_state")
        os.makedirs(self._state_dir, exist_ok=True)
        self._state_file = os.path.join(self._state_dir, "gold_scalper_state.json")
        self._log_file = os.path.join(self._state_dir, "gold_scalper_trade_log.json")

        # Load existing state if available
        self._load_state()

        # Lazy-load data adapter
        self._data_adapter = None

    def _get_data_adapter(self):
        """Lazy-load Yahoo or Alpaca data adapter."""
        if self._data_adapter is None:
            if self.config.data_provider == "alpaca":
                from src.signals_engine import AlpacaAdapter
                self._data_adapter = AlpacaAdapter()
            else:
                from src.signals_engine import YahooFinanceAdapter
                self._data_adapter = YahooFinanceAdapter()
        return self._data_adapter

    def get_current_price(self, symbol: str) -> float:
        """Get latest price from Yahoo Finance (with 15s subprocess timeout)."""
        import subprocess

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")

        script = (
            "import yfinance as yf\n"
            f"t = yf.Ticker('{symbol}')\n"
            "d = t.history(period='1d', interval='1m')\n"
            "if not d.empty:\n"
            "    print(float(d['Close'].iloc[-1]))\n"
        )
        try:
            result = subprocess.run(
                [python_exe, "-c", script],
                cwd=project_root,
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout getting price for {symbol} (15s)")
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")

        # Fallback: fetch 5m bars and use last close
        bars = self.fetch_bars(symbol, "5m", 5)
        if not bars.empty:
            return float(bars["close"].iloc[-1])

        raise RuntimeError(f"Cannot get current price for {symbol}")

    def place_market_order(
        self, symbol: str, contracts: int, side: str
    ) -> Optional[str]:
        """Simulate market order fill at current price."""
        price = self.get_current_price(symbol)
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        if side == "BUY":
            self._position = {
                "direction": "LONG",
                "entry_price": price,
                "contracts": contracts,
            }
        elif side == "SELL":
            self._position = {
                "direction": "SHORT",
                "entry_price": price,
                "contracts": contracts,
            }

        logger.info(
            f"[PAPER] {side} {contracts} {symbol} @ ${price:.2f} "
            f"(order {order_id})"
        )
        self._save_state()
        return order_id

    def close_partial(
        self, symbol: str, contracts: int, reason: str
    ) -> Optional[str]:
        """Close N contracts at current price."""
        if self._position is None:
            logger.warning(f"No position to close for {symbol}")
            return None

        price = self.get_current_price(symbol)
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        # Calculate P&L for closed contracts
        entry = self._position["entry_price"]
        if self._position["direction"] == "LONG":
            pnl = (price - entry) * contracts  # $1 per $0.10 move per contract
        else:
            pnl = (entry - price) * contracts

        # Scale by pip value (each pip = pip_value move, each contract = $1/pip for MCG)
        pnl_per_pip = contracts  # $1 per pip per contract for micro gold
        pips = (price - entry) / self.config.pip_value
        if self._position["direction"] == "SHORT":
            pips = -pips
        pnl = pips * pnl_per_pip

        self._equity += pnl
        self._position["contracts"] -= contracts

        # Log trade
        self._trade_log.append({
            "time": datetime.now().isoformat(),
            "symbol": symbol,
            "direction": self._position["direction"],
            "contracts": contracts,
            "entry": entry,
            "exit": price,
            "pnl": round(pnl, 2),
            "reason": reason,
            "order_id": order_id,
        })

        logger.info(
            f"[PAPER] Close {contracts}ct {symbol} @ ${price:.2f} "
            f"({reason}) PnL: ${pnl:.2f} (order {order_id})"
        )

        # Clear position if fully closed
        if self._position["contracts"] <= 0:
            self._position = None

        self._save_state()
        return order_id

    def close_all(self, symbol: str, reason: str) -> Optional[str]:
        """Close entire position."""
        if self._position is None:
            return None
        return self.close_partial(symbol, self._position["contracts"], reason)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return current position or None."""
        if self._position and self._position.get("contracts", 0) > 0:
            return dict(self._position)
        return None

    def get_account_equity(self) -> float:
        """Return current equity."""
        return self._equity

    def fetch_bars(
        self, symbol: str, timeframe: str, lookback: int,
        _max_retries: int = 3,
        _timeout: int = 30,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars via data adapter with retry on transient failures.

        获取K线数据（带重试+超时）：
        - 日线(1d): 通过fetch_daily获取
        - 其他周期: 通过fetch_intraday获取
        - 4H: 获取1H数据后重采样
        - yfinance uses curl_cffi (C extension) which holds the GIL during TLS.
          Thread-based timeouts cannot interrupt it. We use subprocess instead:
          spawn a child process for the fetch, kill it if it exceeds _timeout.
        - Retry up to 3 times with 2s backoff for transient issues.
        """
        import time as _time

        adapter = self._get_data_adapter()
        last_exc = None

        for attempt in range(_max_retries):
            try:
                df = self._fetch_with_timeout(
                    adapter, symbol, timeframe, lookback, _timeout
                )

                if df is not None and not df.empty:
                    if len(df) > lookback:
                        df = df.tail(lookback).reset_index(drop=True)
                    return df

                # Empty result — retry
                last_exc = None
            except TimeoutError as e:
                last_exc = e
                logger.warning("Timeout fetching %s bars for %s (attempt %d/%d)",
                               timeframe, symbol, attempt + 1, _max_retries)
                if attempt < _max_retries - 1:
                    _time.sleep(2 * (attempt + 1))
            except Exception as e:
                last_exc = e
                if attempt < _max_retries - 1:
                    _time.sleep(2 * (attempt + 1))
                    continue

        if last_exc:
            logger.error(
                "Failed to fetch %s bars for %s after %d attempts: %s",
                timeframe, symbol, _max_retries, last_exc,
            )
        else:
            logger.warning("Empty %s bars for %s after %d attempts", timeframe, symbol, _max_retries)
        return pd.DataFrame()

    def _fetch_with_timeout(
        self, adapter, symbol: str, timeframe: str, lookback: int, timeout: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch bars with a hard process-level timeout.

        curl_cffi holds the GIL during TLS, so thread timeouts don't work.
        We use subprocess: spawn a child that writes CSV to stdout, parse it.
        """
        import subprocess
        import io

        # Build a minimal Python script that does the fetch and prints CSV
        if timeframe == "1d":
            fetch_code = f"df = adapter.fetch_daily('{symbol}', lookback={lookback})"
        elif timeframe == "4h":
            fetch_code = (
                f"df_1h = adapter.fetch_intraday('{symbol}', interval='1h', "
                f"lookback_days={max(5, lookback // 6)})\n"
                f"from src.gold_scalper.bias_stack import BiasStack\n"
                f"df = BiasStack.resample_to_4h(df_1h)"
            )
        else:
            interval_map = {"4h": "4h", "1h": "1h", "15m": "15m",
                            "5m": "5min", "5min": "5min"}
            interval = interval_map.get(timeframe, timeframe)
            days = max(2, lookback // (390 // _tf_minutes(timeframe)))
            fetch_code = (
                f"df = adapter.fetch_intraday('{symbol}', "
                f"interval='{interval}', lookback_days={days})"
            )

        script = (
            "import sys, os\n"
            "os.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n"
            "from src.signals_engine import YahooFinanceAdapter\n"
            "adapter = YahooFinanceAdapter()\n"
            f"{fetch_code}\n"
            "if df is not None and not df.empty:\n"
            "    df.to_csv(sys.stdout, index=False)\n"
            "else:\n"
            "    sys.exit(1)\n"
        )

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")

        try:
            result = subprocess.run(
                [python_exe, "-c", script],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode == 0 and result.stdout.strip():
                return pd.read_csv(io.StringIO(result.stdout))
            return None
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"fetch_bars({timeframe}) subprocess timed out after {timeout}s"
            )

    def _save_state(self) -> None:
        """Persist state to JSON and sync to Gist for dashboard."""
        state = {
            "equity": self._equity,
            "position": self._position,
            "order_counter": self._order_counter,
            "initial_balance": 5000.0,
            "trade_count": len(self._trade_log),
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

        # Append-only trade log
        with open(self._log_file, "w") as f:
            json.dump(self._trade_log, f, indent=2)

        # Sync to GitHub Gist for Vercel dashboard
        self._sync_to_gist(state)

    def _sync_to_gist(self, state: dict) -> None:
        """Upload gold scalper state to GitHub Gist for the dashboard."""
        gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        if not gist_id or not gh_token:
            # Try loading from secrets/alpaca.env
            env_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "secrets", "alpaca.env",
            )
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            v = v.strip().strip('"').strip("'")
                            if k.strip() == "KRAKEN_STATE_GIST_ID" and not gist_id:
                                gist_id = v
                            elif k.strip() == "GITHUB_TOKEN" and not gh_token:
                                gh_token = v
        if not gist_id or not gh_token:
            return
        try:
            import requests
            files = {
                "gold_scalper_state.json": {
                    "content": json.dumps(state, indent=2),
                },
            }
            if self._trade_log:
                files["gold_scalper_trade_log.json"] = {
                    "content": json.dumps(self._trade_log[-200:], indent=2),
                }
            requests.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers={
                    "Authorization": f"token {gh_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                json={"files": files},
                timeout=10,
            )
        except Exception as exc:
            logger.debug("Gist sync error: %s", exc)

    def _load_state(self) -> None:
        """Load state from JSON if exists."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                self._equity = state.get("equity", self._equity)
                self._position = state.get("position")
                self._order_counter = state.get("order_counter", 0)
                logger.info(
                    f"[PAPER] Loaded state: equity=${self._equity:.2f}, "
                    f"position={self._position}"
                )
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        if os.path.exists(self._log_file):
            try:
                with open(self._log_file, "r") as f:
                    self._trade_log = json.load(f)
            except Exception:
                self._trade_log = []


def _tf_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    mapping = {"1d": 1440, "4h": 240, "1h": 60, "15m": 15, "5m": 5, "5min": 5}
    return mapping.get(timeframe, 5)
