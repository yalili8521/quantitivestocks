"""Session filter — NYSE trading window, dead zones, forced closes.

交易时段过滤器：
- 只在纽约交易所开盘时间交易（太平洋时间6:30-13:00，周一到周五）
- "死区"时段禁止交易（凌晨、下午收盘后、周末）
- 每天12:45 PT强制平仓，周五12:50 PT提前平仓
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.gold_scalper.config import GoldScalperConfig


def _parse_time(s: str) -> time:
    """Parse 'HH:MM' to datetime.time."""
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time format: {s!r}, expected HH:MM")
    return time(int(parts[0]), int(parts[1]))


class SessionFilter:
    """Determines whether trading is allowed based on time of day/week."""

    def __init__(self, config: GoldScalperConfig):
        self.tz = ZoneInfo(config.timezone)
        self.session_start = _parse_time(config.session_start)
        self.session_end = _parse_time(config.session_end)
        self.daily_close = _parse_time(config.daily_close)
        self.friday_close = _parse_time(config.friday_close)

    def _to_pt(self, dt: datetime) -> datetime:
        """Convert any datetime to Pacific Time."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def is_weekday(self, now: datetime) -> bool:
        """Monday=0 .. Friday=4 are weekdays."""
        pt = self._to_pt(now)
        return pt.weekday() < 5  # Mon-Fri

    def is_trading_window(self, now: datetime) -> bool:
        """True if inside NYSE session (6:30am-1:00pm PT, weekdays only).

        交易窗口：太平洋时间6:30-13:00，仅工作日
        """
        pt = self._to_pt(now)
        if not self.is_weekday(pt):
            return False
        t = pt.time()
        return self.session_start <= t <= self.session_end

    def is_dead_zone(self, now: datetime) -> bool:
        """True if in a dead zone where no new trades should open.

        死区判断：
        - 每天 00:30-07:14 PT（凌晨死区）
        - 每天 12:44-20:15 PT（下午死区）
        - 周六 20:15-23:59 PT
        - 周日 全天
        - 周一 00:00-20:14 PT（周一开盘前）

        注意：这些范围来自Pine Script原版，某些与NYSE窗口有重叠。
        is_trading_window() 和 is_dead_zone() 应该一起用：
        can_trade = is_trading_window(now) and not is_dead_zone(now)
        """
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()  # Mon=0, Sun=6

        # Sunday — all day dead zone
        if dow == 6:
            return True

        # Saturday 20:15-23:59
        if dow == 5 and t >= time(20, 15):
            return True

        # Monday 00:00-20:14
        if dow == 0 and t < time(20, 14):
            return True

        # Every day 00:30-07:14 (overnight)
        if time(0, 30) <= t <= time(7, 14):
            return True

        # Every day 12:44-20:15 (afternoon/evening)
        if time(12, 44) <= t <= time(20, 15):
            return True

        return False

    def can_trade(self, now: datetime) -> bool:
        """Combined check: in trading window AND not in dead zone.

        可以交易 = 在交易窗口内 且 不在死区内
        """
        return self.is_trading_window(now) and not self.is_dead_zone(now)

    def should_force_close(self, now: datetime) -> tuple[bool, str]:
        """Return (True, reason) if positions must be force-closed.

        强制平仓条件：
        1. 周一到周四：12:45 PT 平仓
        2. 周五：12:50 PT 平仓（提前5分钟因为周末）

        Returns (should_close, reason_string).
        """
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()

        # Friday 12:50-12:54 PT — weekend close
        if dow == 4 and time(12, 50) <= t <= time(12, 54):
            return True, "Friday Weekend Close (12:50 PT)"

        # Mon-Thu 12:45-12:49 PT — daily close
        if dow < 4 and time(12, 45) <= t <= time(12, 49):
            return True, "NYSE Daily Close (12:45 PT)"

        return False, ""

    def should_avoid_entry(self, now: datetime) -> tuple[bool, str]:
        """True if too close to session end for new entries.

        避免入场：距离强制平仓不到15分钟时不再开新仓
        """
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()

        # Within 15 min of force close
        if dow == 4 and t >= time(12, 35):  # Friday
            return True, "Too close to Friday close"
        if dow < 4 and t >= time(12, 30):   # Mon-Thu
            return True, "Too close to daily close"

        return False, ""

    def minutes_to_session_end(self, now: datetime) -> float:
        """Minutes remaining until session end (13:00 PT)."""
        pt = self._to_pt(now)
        end_dt = pt.replace(
            hour=self.session_end.hour,
            minute=self.session_end.minute,
            second=0, microsecond=0
        )
        delta = (end_dt - pt).total_seconds() / 60.0
        return max(0.0, delta)
