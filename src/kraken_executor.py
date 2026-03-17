"""
Kraken Executor — ccxt-based crypto execution layer.

Replaces Alpaca for the crypto sleeve, supporting both LONG and SHORT positions
via Kraken margin trading. Implements the same interface as AlpacaPaperTrader's
order methods so paper_trader.py can swap executors transparently.

Kraken spot has no sandbox — this module includes a local paper-trading mode
that simulates fills without touching the real exchange.

Usage:
    executor = KrakenExecutor(api_key, api_secret, paper=True)
    executor.buy("BTC/USD", 0.01)
    executor.sell_short("ETH/USD", 0.5)
    positions = executor.get_positions()
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger("kraken_executor")


@dataclass
class PaperPosition:
    """Local paper position for simulation mode."""
    symbol: str
    qty: float
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: str  # ISO format

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - current_price) * self.qty

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return (current_price / self.entry_price) - 1.0
        else:
            return 1.0 - (current_price / self.entry_price)


class KrakenExecutor:
    """Crypto execution layer using ccxt Kraken client.

    Supports both live and paper (simulated) modes. Paper mode keeps positions
    in a local JSON file and simulates fills at current market price.

    Interface matches AlpacaPaperTrader's order methods:
        get_account_summary() → {equity, cash, buying_power}
        get_positions() → {symbol: {qty, side, entry_price, current_price, ...}}
        buy(symbol, qty) → order_id
        sell(symbol, qty, reason) → order_id
        sell_short(symbol, qty) → order_id
        buy_to_cover(symbol, qty, reason) → order_id
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        leverage: int = 2,
        state_dir: Optional[str] = None,
        state_file: Optional[str] = None,
        initial_balance: float = 10_000.0,
    ):
        import ccxt

        self.paper = paper
        self.leverage = leverage
        self._order_counter = 0

        # Always create real exchange for price data
        self.exchange = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

        if paper:
            # Paper mode: local state, no real orders
            log.info("KrakenExecutor initialized in PAPER mode (no real orders)")
            self._state_dir = state_dir or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "outputs"
            )
            os.makedirs(self._state_dir, exist_ok=True)
            fname = state_file or "kraken_paper_state.json"
            self._state_file = os.path.join(self._state_dir, fname)
            self._load_state(initial_balance)
        else:
            log.info("KrakenExecutor initialized in LIVE mode — real orders will be placed")
            self._paper_positions: Dict[str, PaperPosition] = {}
            self._paper_cash: float = 0.0
            self._paper_initial: float = 0.0

    # -- State persistence (paper mode) ------------------------------------

    def _load_state(self, initial_balance: float) -> None:
        """Load paper positions and balance from disk."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                self._paper_cash = state.get("cash", initial_balance)
                self._paper_initial = state.get("initial_balance", initial_balance)
                self._paper_positions = {}
                for sym, pos_data in state.get("positions", {}).items():
                    self._paper_positions[sym] = PaperPosition(**pos_data)
                log.info("Loaded paper state: cash=$%.2f, %d positions",
                         self._paper_cash, len(self._paper_positions))
                return
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                log.warning("Corrupt paper state file, resetting: %s", exc)

        self._paper_cash = initial_balance
        self._paper_initial = initial_balance
        self._paper_positions = {}
        self._save_state()

    def _save_state(self) -> None:
        """Persist paper positions and balance to disk, then sync to GitHub Gist."""
        if not self.paper:
            return
        state = {
            "cash": self._paper_cash,
            "initial_balance": self._paper_initial,
            "positions": {sym: asdict(pos) for sym, pos in self._paper_positions.items()},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except OSError as exc:
            log.error("Failed to save paper state: %s", exc)

        # Sync to GitHub Gist so Vercel API can read it
        self._sync_to_gist(state)

    def _sync_to_gist(self, state: dict) -> None:
        """Upload paper state to a GitHub Gist for the Vercel dashboard."""
        gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        if not gist_id or not gh_token:
            return  # silently skip if not configured
        try:
            import requests
            resp = requests.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers={
                    "Authorization": f"token {gh_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                json={"files": {os.path.basename(self._state_file): {"content": json.dumps(state, indent=2)}}},
                timeout=10,
            )
            if resp.ok:
                log.debug("Synced paper state to Gist %s", gist_id)
            else:
                log.warning("Gist sync failed (%d): %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            log.warning("Gist sync error: %s", exc)

    # -- Price fetching ----------------------------------------------------

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert our symbol format to ccxt/Kraken format.

        BTC/USD → BTC/USD (already correct)
        BTC-USD → BTC/USD
        BTCUSD → BTC/USD
        """
        if "/" in symbol:
            return symbol
        if "-" in symbol:
            return symbol.replace("-", "/")
        # Handle BTCUSD format
        if symbol.upper().endswith("USD") and len(symbol) > 3:
            return symbol[:-3] + "/" + symbol[-3:]
        return symbol

    def _get_price(self, symbol: str) -> Optional[float]:
        """Fetch current market price from Kraken."""
        ccxt_sym = self._normalize_symbol(symbol)
        try:
            ticker = self.exchange.fetch_ticker(ccxt_sym)
            return float(ticker["last"])
        except Exception as exc:
            log.error("Failed to fetch price for %s: %s", ccxt_sym, exc)
            return None

    # -- Account info ------------------------------------------------------

    def get_account_summary(self) -> dict:
        """Get account equity, cash, and buying power."""
        if self.paper:
            equity = self._paper_cash
            for sym, pos in self._paper_positions.items():
                price = self._get_price(sym)
                if price is not None:
                    equity += pos.unrealized_pnl(price)
                    # Add position notional back for long positions
                    if pos.side == "LONG":
                        equity += pos.entry_price * pos.qty
            return {
                "equity": equity,
                "cash": self._paper_cash,
                "buying_power": self._paper_cash * self.leverage,
            }
        else:
            balance = self.exchange.fetch_balance()
            total_usd = float(balance.get("total", {}).get("USD", 0))
            free_usd = float(balance.get("free", {}).get("USD", 0))
            return {
                "equity": total_usd,
                "cash": free_usd,
                "buying_power": free_usd * self.leverage,
            }

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions as {symbol: {qty, side, entry_price, current_price, pnl, pnl_pct}}."""
        if self.paper:
            result = {}
            for sym, pos in self._paper_positions.items():
                price = self._get_price(sym)
                if price is None:
                    price = pos.entry_price  # fallback
                result[sym] = {
                    "qty": pos.qty,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "current_price": price,
                    "unrealized_pnl": pos.unrealized_pnl(price),
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct(price),
                }
            return result
        else:
            # Live mode: fetch from Kraken
            try:
                positions = self.exchange.fetch_positions()
            except Exception as exc:
                log.error("Failed to fetch positions: %s", exc)
                return {}

            result = {}
            for pos in positions:
                if float(pos.get("contracts", 0)) == 0:
                    continue
                sym = pos["symbol"]
                side = pos["side"].upper()
                qty = abs(float(pos["contracts"]))
                entry = float(pos.get("entryPrice", 0))
                current = float(pos.get("markPrice", entry))
                pnl = float(pos.get("unrealizedPnl", 0))
                notional = entry * qty if entry > 0 else 1.0
                pnl_pct = pnl / notional if notional > 0 else 0.0
                result[sym] = {
                    "qty": qty,
                    "side": side,
                    "entry_price": entry,
                    "current_price": current,
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct": pnl_pct,
                }
            return result

    # -- Order execution ---------------------------------------------------

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"kraken-paper-{self._order_counter}"

    def _submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        is_close: bool = False,
    ) -> Optional[str]:
        """Submit an order (market).

        Args:
            symbol: Trading pair (e.g. BTC/USD, ETH/USD)
            qty: Amount in base currency
            side: "buy" or "sell"
            is_close: True if closing an existing position
        """
        ccxt_sym = self._normalize_symbol(symbol)

        if self.paper:
            return self._paper_fill(symbol, ccxt_sym, qty, side, is_close)
        else:
            return self._live_order(ccxt_sym, qty, side, is_close)

    def _paper_fill(
        self,
        symbol: str,
        ccxt_sym: str,
        qty: float,
        side: str,
        is_close: bool,
    ) -> Optional[str]:
        """Simulate a fill in paper mode."""
        price = self._get_price(symbol)
        if price is None:
            log.error("Cannot fill paper order — no price for %s", symbol)
            return None

        order_id = self._next_order_id()

        if is_close and symbol in self._paper_positions:
            # Closing: realize P&L
            pos = self._paper_positions[symbol]
            pnl = pos.unrealized_pnl(price)
            # Return notional for long positions
            if pos.side == "LONG":
                self._paper_cash += pos.entry_price * pos.qty + pnl
            else:
                # Short: we received cash at entry, now buy back
                self._paper_cash += pnl  # margin is already in cash
            log.info("PAPER CLOSE %s %s x%.6f @ $%.2f — P&L: $%.2f — order %s",
                     pos.side, symbol, pos.qty, price, pnl, order_id)
            del self._paper_positions[symbol]

        elif not is_close:
            # Opening new position
            notional = price * qty
            if side == "buy":
                # Long: deduct cash
                if notional > self._paper_cash:
                    log.error("Insufficient paper cash ($%.2f) for %s buy $%.2f",
                              self._paper_cash, symbol, notional)
                    return None
                self._paper_cash -= notional
                self._paper_positions[symbol] = PaperPosition(
                    symbol=symbol, qty=qty, side="LONG",
                    entry_price=price,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                )
                log.info("PAPER BUY %s x%.6f @ $%.2f (notional $%.2f) — order %s",
                         symbol, qty, price, notional, order_id)
            else:
                # Short: margin trade, reserve margin
                margin_required = notional / self.leverage
                if margin_required > self._paper_cash:
                    log.error("Insufficient paper margin ($%.2f) for %s short (need $%.2f)",
                              self._paper_cash, symbol, margin_required)
                    return None
                # Don't deduct full notional — only margin
                self._paper_positions[symbol] = PaperPosition(
                    symbol=symbol, qty=qty, side="SHORT",
                    entry_price=price,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                )
                log.info("PAPER SHORT %s x%.6f @ $%.2f (margin $%.2f) — order %s",
                         symbol, qty, price, margin_required, order_id)

        self._save_state()
        return order_id

    def _live_order(
        self,
        ccxt_sym: str,
        qty: float,
        side: str,
        is_close: bool,
    ) -> Optional[str]:
        """Place a real order on Kraken via ccxt."""
        try:
            params = {}
            if side == "sell" and not is_close:
                # Opening a short — need leverage
                params["leverage"] = self.leverage
            if is_close:
                params["reduce_only"] = True

            order = self.exchange.create_order(
                symbol=ccxt_sym,
                type="market",
                side=side,
                amount=qty,
                params=params,
            )
            order_id = str(order.get("id", "unknown"))
            log.info("KRAKEN %s %s %s x%.6f — order %s",
                     "CLOSE" if is_close else "OPEN",
                     side.upper(), ccxt_sym, qty, order_id)
            return order_id
        except Exception as exc:
            log.error("Kraken order failed %s %s x%.6f: %s",
                      side.upper(), ccxt_sym, qty, exc)
            return None

    # -- High-level order methods (match AlpacaPaperTrader interface) ------

    def buy(self, symbol: str, qty: float,
            limit_price: Optional[float] = None) -> Optional[str]:
        """Open LONG position."""
        if qty <= 0:
            return None
        return self._submit_order(symbol, qty, "buy", is_close=False)

    def sell(self, symbol: str, qty: float, reason: str = "",
             limit_price: Optional[float] = None) -> Optional[str]:
        """Close LONG position."""
        if qty <= 0:
            return None
        oid = self._submit_order(symbol, qty, "sell", is_close=True)
        if oid:
            log.info("SELL reason: %s", reason)
        return oid

    def sell_short(self, symbol: str, qty: float,
                   limit_price: Optional[float] = None) -> Optional[str]:
        """Open SHORT position (margin)."""
        if qty <= 0:
            return None
        return self._submit_order(symbol, qty, "sell", is_close=False)

    def buy_to_cover(self, symbol: str, qty: float, reason: str = "",
                     limit_price: Optional[float] = None) -> Optional[str]:
        """Close SHORT position."""
        if qty <= 0:
            return None
        oid = self._submit_order(symbol, qty, "buy", is_close=True)
        if oid:
            log.info("COVER reason: %s", reason)
        return oid
