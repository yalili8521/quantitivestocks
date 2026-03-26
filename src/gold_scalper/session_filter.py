"""Session filter — COMEX/Globex gold futures trading window.

Gold futures (GC) trade nearly 24 hours on weekdays:
  - Sunday 3:00 PM PT through Friday 2:00 PM PT
  - Daily maintenance halt: 2:00 PM - 3:00 PM PT (Mon-Thu)
  - Weekly close: Friday 2:00 PM PT
  - Weekly open: Sunday 3:00 PM PT
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.gold_scalper.config import GoldScalperConfig

_DAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def _parse_time(s: str) -> time:
    """Parse 'HH:MM' to datetime.time."""
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time format: {s!r}, expected HH:MM")
    return time(int(parts[0]), int(parts[1]))


def _parse_day_range(day_spec: str) -> set[int]:
    """Parse day spec like 'Mon-Thu', 'Sat', 'Mon-Sun' into weekday ints."""
    if "-" in day_spec:
        start_name, end_name = day_spec.split("-")
        start_idx = _DAY_MAP[start_name]
        end_idx = _DAY_MAP[end_name]
        if start_idx <= end_idx:
            return set(range(start_idx, end_idx + 1))
        # Wrap around (e.g. Fri-Mon)
        return set(range(start_idx, 7)) | set(range(0, end_idx + 1))
    return {_DAY_MAP[day_spec]}


class SessionFilter:
    """Determines whether trading is allowed based on time of day/week."""

    def __init__(self, config: GoldScalperConfig):
        self.tz = ZoneInfo(config.timezone)
        self.session_start = _parse_time(config.session_start)
        self.session_end = _parse_time(config.session_end)
        self.daily_close = _parse_time(config.daily_close)
        self.friday_close = _parse_time(config.friday_close)
        # Parse dead zone ranges from config
        self._dead_zones: list[tuple[time, time, set[int]]] = []
        for start_s, end_s, day_spec in config.dead_zone_ranges:
            self._dead_zones.append((
                _parse_time(start_s),
                _parse_time(end_s),
                _parse_day_range(day_spec),
            ))

    def _to_pt(self, dt: datetime) -> datetime:
        """Convert any datetime to Pacific Time."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def is_weekday(self, now: datetime) -> bool:
        """Monday=0 .. Friday=4 are weekdays."""
        pt = self._to_pt(now)
        return pt.weekday() < 5

    def is_trading_window(self, now: datetime) -> bool:
        """True if inside Globex session.

        Handles overnight sessions where session_start > session_end
        (e.g., 15:00 start, 14:00 end = crosses midnight).
        """
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()

        # Weekend: only open Sunday >= session_start
        if dow == 5:  # Saturday
            return False
        if dow == 6:  # Sunday: only after session_start
            return t >= self.session_start

        # Weekdays: overnight session (start > end means crosses midnight)
        if self.session_start > self.session_end:
            # In session if: time >= start OR time < end
            return t >= self.session_start or t < self.session_end
        else:
            return self.session_start <= t < self.session_end

    def is_dead_zone(self, now: datetime) -> bool:
        """True if in a dead zone where no new trades should open."""
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()

        for dz_start, dz_end, dz_days in self._dead_zones:
            if dow in dz_days and dz_start <= t <= dz_end:
                return True
        return False

    def can_trade(self, now: datetime) -> bool:
        """Combined check: in trading window AND not in dead zone."""
        return self.is_trading_window(now) and not self.is_dead_zone(now)

    def should_force_close(self, now: datetime) -> tuple[bool, str]:
        """Return (True, reason) if positions must be force-closed.

        - Friday: force-close at friday_close before weekly shutdown
        - Mon-Thu: force-close at daily_close before maintenance halt
        """
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()

        # Friday close (before weekly shutdown)
        if dow == 4:
            close_t = self.friday_close
            close_end_min = close_t.minute + 4
            close_end_hour = close_t.hour + close_end_min // 60
            close_end_min = close_end_min % 60
            if close_t <= t <= time(close_end_hour, close_end_min):
                return True, f"Friday Weekly Close ({self.friday_close.strftime('%H:%M')} PT)"

        # Mon-Thu close (before daily maintenance halt)
        if 0 <= dow <= 3:
            close_t = self.daily_close
            close_end_min = close_t.minute + 4
            close_end_hour = close_t.hour + close_end_min // 60
            close_end_min = close_end_min % 60
            if close_t <= t <= time(close_end_hour, close_end_min):
                return True, f"Daily Maintenance Close ({self.daily_close.strftime('%H:%M')} PT)"

        return False, ""

    def should_avoid_entry(self, now: datetime) -> tuple[bool, str]:
        """True if too close to session end for new entries (15 min buffer)."""
        pt = self._to_pt(now)
        t = pt.time()
        dow = pt.weekday()

        # Within 15 min of force close
        close_t = self.friday_close if dow == 4 else self.daily_close
        buffer_hour = close_t.hour
        buffer_min = close_t.minute - 15
        if buffer_min < 0:
            buffer_hour -= 1
            buffer_min += 60
        buffer_time = time(buffer_hour, buffer_min)

        avoid_end_min = close_t.minute + 4
        avoid_end_hour = close_t.hour + avoid_end_min // 60
        avoid_end_min = avoid_end_min % 60
        if t >= buffer_time and t <= time(avoid_end_hour, avoid_end_min):
            label = "Friday close" if dow == 4 else "daily maintenance"
            return True, f"Too close to {label}"

        return False, ""

    def minutes_to_session_end(self, now: datetime) -> float:
        """Minutes remaining until daily force-close."""
        pt = self._to_pt(now)
        close_t = self.friday_close if pt.weekday() == 4 else self.daily_close
        end_dt = pt.replace(
            hour=close_t.hour,
            minute=close_t.minute,
            second=0, microsecond=0,
        )
        delta = (end_dt - pt).total_seconds() / 60.0
        return max(0.0, delta)
