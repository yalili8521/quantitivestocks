#!/usr/bin/env python3
"""
Microstructure Alert System
====================================
Detects abnormal market conditions and sends alerts via console + webhook.

Alert types:
    - RVOL_SPIKE:          Relative volume > 2.5x (unusual activity)
    - VIX_TERM_INVERSION:  VIX > VIX3M (near-term fear spike)
    - PC_RATIO_EXTREME:    Put/call ratio > 1.5 or < 0.5 (extreme sentiment)
    - CONSECUTIVE_LOSSES:  3+ consecutive losing trades (strategy degradation)
    - EQUITY_DRAWDOWN:     Portfolio drawdown > 5% (risk management)

Delivery:
    - Console logging (always)
    - Webhook POST to ALERT_WEBHOOK_URL env var (Discord/Slack compatible)

Usage:
    from alerts import AlertEngine
    engine = AlertEngine()
    alerts = engine.check(market_context, options_flow, portfolio_state)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("alerts")


# ---------------------------------------------------------------------------
# Alert Types and Severity
# ---------------------------------------------------------------------------
class AlertType(Enum):
    RVOL_SPIKE = "rvol_spike"
    VIX_TERM_INVERSION = "vix_term_inversion"
    PC_RATIO_EXTREME = "pc_ratio_extreme"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    EQUITY_DRAWDOWN = "equity_drawdown"
    POSITION_OPENED = "position_opened"
    MODEL_PAUSED = "model_paused"
    PIPELINE_SUMMARY = "pipeline_summary"


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    alert_type: AlertType
    symbol: str              # symbol or "MARKET" for market-wide alerts
    message: str
    severity: Severity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    value: float = 0.0       # the numeric value that triggered the alert

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type.value,
            "symbol": self.symbol,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
        }


# ---------------------------------------------------------------------------
# Alert Deduplicator
# ---------------------------------------------------------------------------
class AlertDeduplicator:
    """Prevent alert spam by enforcing a cooldown per (alert_type, symbol).

    Default cooldown: 30 minutes.
    """

    def __init__(self, cooldown_seconds: float = 1800.0):
        self._cooldown = cooldown_seconds
        self._last_fired: Dict[str, float] = {}

    def should_fire(self, alert_type: AlertType, symbol: str) -> bool:
        key = f"{alert_type.value}:{symbol}"
        last = self._last_fired.get(key, 0)
        if time.time() - last < self._cooldown:
            return False
        return True

    def mark_fired(self, alert_type: AlertType, symbol: str) -> None:
        key = f"{alert_type.value}:{symbol}"
        self._last_fired[key] = time.time()

    def reset(self) -> None:
        self._last_fired.clear()


# ---------------------------------------------------------------------------
# Webhook Notifier
# ---------------------------------------------------------------------------
class WebhookNotifier:
    """Send alerts via HTTP POST to Discord/Slack webhook.

    Reads webhook URL from ALERT_WEBHOOK_URL environment variable.
    """

    SEVERITY_COLORS = {
        Severity.INFO:     0x3498DB,   # blue
        Severity.WARNING:  0xF39C12,   # orange
        Severity.CRITICAL: 0xE74C3C,   # red
    }

    SEVERITY_EMOJI = {
        Severity.INFO:     "info",
        Severity.WARNING:  "warning",
        Severity.CRITICAL: "rotating_light",
    }

    def __init__(self, webhook_url: Optional[str] = None):
        self._url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def send(self, alert: Alert) -> bool:
        """Send alert to webhook. Returns True on success."""
        if not self._url:
            return False

        color = self.SEVERITY_COLORS.get(alert.severity, 0x95A5A6)
        emoji = self.SEVERITY_EMOJI.get(alert.severity, "bell")

        is_slack = "hooks.slack.com" in self._url

        if is_slack:
            color_hex = f"#{color:06x}"
            payload = {
                "attachments": [{
                    "color": color_hex,
                    "title": f":{emoji}: [{alert.severity.value}] {alert.alert_type.value.upper()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Symbol", "value": alert.symbol, "short": True},
                        {"title": "Value", "value": f"{alert.value:.4f}", "short": True},
                    ],
                    "footer": "QuantStocks Alert Engine",
                    "ts": int(alert.timestamp.timestamp()),
                }]
            }
        else:
            # Discord-compatible embed format
            payload = {
                "embeds": [{
                    "title": f"[{alert.severity.value}] {alert.alert_type.value.upper()}",
                    "description": alert.message,
                    "color": color,
                    "fields": [
                        {"name": "Symbol", "value": alert.symbol, "inline": True},
                        {"name": "Value", "value": f"{alert.value:.4f}", "inline": True},
                    ],
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {"text": "QuantStocks Alert Engine"},
                }]
            }

        try:
            resp = requests.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                return True
            log.warning("Webhook returned status %d: %s", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:
            log.warning("Webhook send failed: %s", exc)
            return False

    def send_raw(self, content: str) -> bool:
        """Send a plain-text message (used by gold scalper alerts)."""
        if not self._url:
            return False
        is_slack = "hooks.slack.com" in self._url
        if is_slack:
            payload = {"text": content}
        else:
            payload = {"content": content[:2000]}
        try:
            resp = requests.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            return resp.status_code in (200, 204)
        except Exception as exc:
            log.warning("Webhook send_raw failed: %s", exc)
            return False

    def send_batch(self, alerts: List[Alert]) -> int:
        """Send multiple alerts. Returns count of successful sends."""
        sent = 0
        for alert in alerts:
            if self.send(alert):
                sent += 1
        return sent


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------
class AlertEngine:
    """Central alert engine that checks market conditions and fires alerts.

    Usage:
        engine = AlertEngine()
        alerts = engine.check(
            market_context={"rvol": 3.2, "symbol": "SPY"},
            options_flow={"vix_term_inverted": 1.0, "pc_volume_ratio": 1.8},
            portfolio_state={"consecutive_losses": 4, "drawdown_pct": -6.2},
        )
    """

    # Thresholds
    RVOL_THRESHOLD = 2.5          # relative volume spike
    PC_HIGH_THRESHOLD = 1.5       # extreme put buying
    PC_LOW_THRESHOLD = 0.5        # extreme call buying
    CONSECUTIVE_LOSS_THRESHOLD = 3
    DRAWDOWN_THRESHOLD = -5.0     # drawdown % (negative)

    def __init__(self,
                 cooldown_seconds: float = 1800.0,
                 webhook_url: Optional[str] = None):
        self._dedup = AlertDeduplicator(cooldown_seconds)
        self._notifier = WebhookNotifier(webhook_url)
        log.info("AlertEngine initialized. Webhook: %s",
                 "enabled" if self._notifier.enabled else "disabled (set ALERT_WEBHOOK_URL)")

    def check(self,
              market_context: Optional[Dict] = None,
              options_flow: Optional[Dict] = None,
              portfolio_state: Optional[Dict] = None) -> List[Alert]:
        """Run all alert checks and return fired alerts.

        Args:
            market_context: Per-symbol dict with {symbol, rvol, adx, vol20, ...}
            options_flow: Market-wide dict from OptionsFlowEngine.get_features()
            portfolio_state: Portfolio dict with {consecutive_losses, drawdown_pct,
                            equity, initial_capital}

        Returns:
            List of Alert objects that were fired (after dedup).
        """
        fired: List[Alert] = []

        if market_context:
            fired.extend(self._check_rvol(market_context))

        if options_flow:
            fired.extend(self._check_vix_term(options_flow))
            fired.extend(self._check_pc_ratio(options_flow))

        if portfolio_state:
            fired.extend(self._check_consecutive_losses(portfolio_state))
            fired.extend(self._check_drawdown(portfolio_state))

        # Deliver alerts
        for alert in fired:
            self._deliver(alert)

        return fired

    def _check_rvol(self, ctx: Dict) -> List[Alert]:
        """Check for relative volume spike."""
        symbol = ctx.get("symbol", "UNKNOWN")
        rvol = ctx.get("rvol", 0)

        if rvol is None or rvol < self.RVOL_THRESHOLD:
            return []
        if not self._dedup.should_fire(AlertType.RVOL_SPIKE, symbol):
            return []

        severity = Severity.CRITICAL if rvol > 4.0 else Severity.WARNING
        alert = Alert(
            alert_type=AlertType.RVOL_SPIKE,
            symbol=symbol,
            message=f"Relative volume spike detected for {symbol}: "
                    f"RVOL={rvol:.1f}x (threshold: {self.RVOL_THRESHOLD}x). "
                    f"Unusual activity may indicate institutional order flow or news event.",
            severity=severity,
            value=rvol,
        )
        self._dedup.mark_fired(AlertType.RVOL_SPIKE, symbol)
        return [alert]

    def _check_vix_term(self, flow: Dict) -> List[Alert]:
        """Check for VIX term structure inversion (backwardation)."""
        inverted = flow.get("vix_term_inverted", 0)
        ratio = flow.get("vix_term_ratio", 0)

        if not inverted or inverted == 0:
            return []
        if not self._dedup.should_fire(AlertType.VIX_TERM_INVERSION, "MARKET"):
            return []

        severity = Severity.CRITICAL if ratio > 1.15 else Severity.WARNING
        alert = Alert(
            alert_type=AlertType.VIX_TERM_INVERSION,
            symbol="MARKET",
            message=f"VIX term structure INVERTED (backwardation): "
                    f"VIX/VIX3M ratio={ratio:.3f}. "
                    f"Near-term fear exceeds medium-term — historically signals "
                    f"elevated risk of sharp selloff.",
            severity=severity,
            value=ratio,
        )
        self._dedup.mark_fired(AlertType.VIX_TERM_INVERSION, "MARKET")
        return [alert]

    def _check_pc_ratio(self, flow: Dict) -> List[Alert]:
        """Check for extreme put/call ratio."""
        pc_ratio = flow.get("pc_volume_ratio")
        if pc_ratio is None or (pc_ratio != pc_ratio):  # NaN check
            return []
        if not self._dedup.should_fire(AlertType.PC_RATIO_EXTREME, "MARKET"):
            return []

        if pc_ratio > self.PC_HIGH_THRESHOLD:
            alert = Alert(
                alert_type=AlertType.PC_RATIO_EXTREME,
                symbol="MARKET",
                message=f"Extreme put/call ratio: {pc_ratio:.3f} "
                        f"(threshold: >{self.PC_HIGH_THRESHOLD}). "
                        f"High put buying indicates fear/hedging — "
                        f"contrarian bullish signal at extremes.",
                severity=Severity.WARNING,
                value=pc_ratio,
            )
            self._dedup.mark_fired(AlertType.PC_RATIO_EXTREME, "MARKET")
            return [alert]

        if pc_ratio < self.PC_LOW_THRESHOLD:
            alert = Alert(
                alert_type=AlertType.PC_RATIO_EXTREME,
                symbol="MARKET",
                message=f"Extreme put/call ratio: {pc_ratio:.3f} "
                        f"(threshold: <{self.PC_LOW_THRESHOLD}). "
                        f"Extreme call buying indicates complacency — "
                        f"contrarian bearish signal at extremes.",
                severity=Severity.WARNING,
                value=pc_ratio,
            )
            self._dedup.mark_fired(AlertType.PC_RATIO_EXTREME, "MARKET")
            return [alert]

        return []

    def _check_consecutive_losses(self, state: Dict) -> List[Alert]:
        """Check for consecutive losing trades."""
        losses = state.get("consecutive_losses", 0)
        if losses < self.CONSECUTIVE_LOSS_THRESHOLD:
            return []
        if not self._dedup.should_fire(AlertType.CONSECUTIVE_LOSSES, "PORTFOLIO"):
            return []

        severity = Severity.CRITICAL if losses >= 5 else Severity.WARNING
        alert = Alert(
            alert_type=AlertType.CONSECUTIVE_LOSSES,
            symbol="PORTFOLIO",
            message=f"{losses} consecutive losing trades detected. "
                    f"Strategy may be in drawdown regime — review model "
                    f"performance and consider reducing position sizes.",
            severity=severity,
            value=float(losses),
        )
        self._dedup.mark_fired(AlertType.CONSECUTIVE_LOSSES, "PORTFOLIO")
        return [alert]

    def _check_drawdown(self, state: Dict) -> List[Alert]:
        """Check for portfolio equity drawdown."""
        dd_pct = state.get("drawdown_pct", 0)
        if dd_pct > self.DRAWDOWN_THRESHOLD:  # drawdown is negative
            return []
        if not self._dedup.should_fire(AlertType.EQUITY_DRAWDOWN, "PORTFOLIO"):
            return []

        severity = Severity.CRITICAL if dd_pct < -10.0 else Severity.WARNING
        alert = Alert(
            alert_type=AlertType.EQUITY_DRAWDOWN,
            symbol="PORTFOLIO",
            message=f"Portfolio drawdown: {dd_pct:.1f}% "
                    f"(threshold: {self.DRAWDOWN_THRESHOLD}%). "
                    f"Consider risk reduction measures.",
            severity=severity,
            value=dd_pct,
        )
        self._dedup.mark_fired(AlertType.EQUITY_DRAWDOWN, "PORTFOLIO")
        return [alert]

    def notify_entry(self, symbol: str, direction: str, qty,
                     price: float, notional: float,
                     group: str = "", expected_return: float = 0.0) -> None:
        """Fire an alert when a new position is opened."""
        qty_str = f"{qty:.6f}" if isinstance(qty, float) else str(qty)
        msg = (f"{direction} {symbol} — {qty_str} shares @ ${price:.2f} "
               f"(${notional:,.0f})")
        if group:
            msg += f" [{group}]"
        if expected_return:
            msg += f" E[r]={expected_return:+.4f}"

        alert = Alert(
            alert_type=AlertType.POSITION_OPENED,
            symbol=symbol,
            message=msg,
            severity=Severity.INFO,
            value=price,
        )
        self._deliver(alert)

    def notify_model_paused(self, symbol: str, reason: str, group: str = "") -> None:
        """Fire an alert when a model is paused due to health degradation."""
        if not self._dedup.should_fire(AlertType.MODEL_PAUSED, symbol):
            return
        msg = f"Model PAUSED for {symbol}: {reason}"
        if group:
            msg += f" [{group}]"
        alert = Alert(
            alert_type=AlertType.MODEL_PAUSED,
            symbol=symbol,
            message=msg,
            severity=Severity.WARNING,
            value=0.0,
        )
        self._dedup.mark_fired(AlertType.MODEL_PAUSED, symbol)
        self._deliver(alert)

    def notify_pipeline_summary(self, summary_text: str) -> None:
        """Send weekly pipeline run summary via Slack."""
        alert = Alert(
            alert_type=AlertType.PIPELINE_SUMMARY,
            symbol="PIPELINE",
            message=summary_text,
            severity=Severity.INFO,
            value=0.0,
        )
        self._deliver(alert)

    def _deliver(self, alert: Alert) -> None:
        """Deliver alert via console logging and webhook."""
        # Console (always)
        level = {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        log.log(level, "ALERT [%s] %s | %s | value=%.4f",
                alert.severity.value, alert.alert_type.value,
                alert.message, alert.value)

        # Webhook (if configured)
        if self._notifier.enabled:
            self._notifier.send(alert)
