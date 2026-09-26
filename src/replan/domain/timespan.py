"""Time primitives, and the exact serialisation the AWS Events API demands.

Two independent time conventions meet in this API, and conflating them is the
easiest way to put an attendee in the wrong room at the wrong hour:

* **Sessions** are published as ``sessionTime`` = ``{date, time, length,
  timezone}``: a local wall-clock date and start time, a length in minutes as a
  *string*, and the IANA zone those are expressed in. There is no end time; you
  compute it.
* **Personal time** is written as ``YYYY-MM-DDTHH:MM:SS`` and is *always UTC*.
  The docs are explicit that a trailing ``Z`` or a ``+01:00`` offset is
  **rejected rather than converted**, seconds must be ``00``, and the duration
  must be a whole number of 5-minute increments.

So: everything inside replan is timezone-aware, converted at the edges, and
serialised back through :func:`format_personal_time` so the naive-UTC rule lives
in exactly one place. Ruff's DTZ rules are enabled to stop naive datetimes
leaking in.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# re:Invent's home zone. Confirmed live: GetEvent for `reinvent2026` reports
# timezone "America/Los_Angeles", running 2026-11-30 to 2026-12-04.
REINVENT_TZ = ZoneInfo("America/Los_Angeles")

_WEEKDAY_ABBR = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_PERSONAL_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
PERSONAL_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
PERSONAL_TIME_INCREMENT_MINUTES = 5


def resolve_timezone(name: str | None, fallback: ZoneInfo = REINVENT_TZ) -> ZoneInfo:
    """Look up an IANA zone, falling back instead of exploding.

    A catalog that names a zone we cannot load should degrade to showing times in
    the event's default zone, not crash the planner.
    """
    if not name:
        return fallback
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return fallback


def now_utc() -> datetime:
    """The single source of 'now', so tests freeze one function."""
    return datetime.now(UTC)


def require_aware(value: datetime, *, field: str = "datetime") -> datetime:
    if value.tzinfo is None:
        msg = f"{field} must be timezone-aware; refusing to guess a zone"
        raise ValueError(msg)
    return value


# --------------------------------------------------------------------------- #
# Session time parsing
# --------------------------------------------------------------------------- #
def parse_session_time(
    *,
    day: str | None,
    clock: str | None,
    length_minutes: str | int | None,
    timezone_name: str | None = None,
    fallback_tz: ZoneInfo = REINVENT_TZ,
) -> TimeSpan | None:
    """Turn an API ``sessionTime`` into a :class:`TimeSpan`.

    Returns ``None`` rather than raising when the catalog omits the pieces, which
    it does for all-day and unscheduled entries. A planner that crashed on those
    would be unable to read a real catalog at all.
    """
    if not day or not clock:
        return None
    try:
        parsed_day = date.fromisoformat(day.strip())
    except ValueError:
        return None

    clock_text = clock.strip()
    parsed_clock = _parse_clock(clock_text)
    if parsed_clock is None:
        return None

    minutes = _parse_length(length_minutes)
    if minutes is None or minutes <= 0:
        return None

    zone = resolve_timezone(timezone_name, fallback_tz)
    start = datetime.combine(parsed_day, parsed_clock, tzinfo=zone)
    return TimeSpan(start, start + timedelta(minutes=minutes))


def _parse_clock(text: str) -> time | None:
    """Accept ``"16:00"``, ``"16:00:00"``, ``"4:00 PM"``, ``"4 PM"``."""
    cleaned = text.strip().upper().replace(".", "")
    meridiem: str | None = None
    for marker in ("AM", "PM"):
        if cleaned.endswith(marker):
            meridiem = marker
            cleaned = cleaned[: -len(marker)].strip()
            break
    parts = cleaned.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except (ValueError, IndexError):
        return None
    if meridiem == "PM" and hour < 12:
        hour += 12
    elif meridiem == "AM" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour, minute)


def _parse_length(raw: str | int | None) -> int | None:
    """``length`` arrives as a string of minutes, e.g. ``"70"``."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return int(digits) if digits else None


# --------------------------------------------------------------------------- #
# Personal time serialisation: naive UTC, 5-minute increments, :00 seconds
# --------------------------------------------------------------------------- #
class PersonalTimeFormatError(ValueError):
    """Raised when a block cannot be expressed in the API's required form."""


def format_personal_time(value: datetime) -> str:
    """Render an aware datetime as the API's naive-UTC string.

    The API rejects any offset or ``Z`` suffix, so this converts to UTC and then
    deliberately drops the marker. Seconds must be zero; we raise rather than
    silently truncating, because silently moving someone's dinner block by 40
    seconds is the kind of bug that never gets found.
    """
    require_aware(value, field="personal time boundary")
    as_utc = value.astimezone(UTC)
    if as_utc.second or as_utc.microsecond:
        msg = (
            f"personal time must land on a whole minute; got {as_utc.isoformat()}. "
            "Round it before calling."
        )
        raise PersonalTimeFormatError(msg)
    return as_utc.strftime(PERSONAL_TIME_FORMAT)


def parse_personal_time(text: str) -> datetime:
    """Read the API's naive-UTC string back into an aware UTC datetime."""
    candidate = text.strip()
    if not _PERSONAL_TIME_RE.match(candidate):
        msg = f"expected YYYY-MM-DDTHH:MM:SS with no zone marker, got {text!r}"
        raise PersonalTimeFormatError(msg)
    return datetime.strptime(candidate, PERSONAL_TIME_FORMAT).replace(tzinfo=UTC)


def validate_personal_time_span(span: TimeSpan) -> None:
    """Enforce the documented 5-minute-increment rule before we spend a request."""
    if span.minutes % PERSONAL_TIME_INCREMENT_MINUTES:
        msg = (
            f"personal time must be a whole number of {PERSONAL_TIME_INCREMENT_MINUTES}-minute "
            f"increments; {span.minutes} minutes is not"
        )
        raise PersonalTimeFormatError(msg)


def round_to_increment(value: datetime, *, up: bool = False) -> datetime:
    """Snap a datetime to the API's 5-minute grid, and to a whole minute."""
    require_aware(value, field="datetime")
    trimmed = value.replace(second=0, microsecond=0)
    remainder = trimmed.minute % PERSONAL_TIME_INCREMENT_MINUTES
    if not remainder:
        return trimmed
    if up:
        return trimmed + timedelta(minutes=PERSONAL_TIME_INCREMENT_MINUTES - remainder)
    return trimmed - timedelta(minutes=remainder)


# --------------------------------------------------------------------------- #
# Day helpers
# --------------------------------------------------------------------------- #
def day_label(value: date) -> str:
    """``mon``, ``tue``, ... - the shorthand ``--day`` accepts."""
    return _WEEKDAY_ABBR[value.weekday()]


def resolve_day(token: str, days: Sequence[date]) -> date:
    """Turn ``tue``, ``2026-12-01``, or ``day2`` into a real catalog date.

    Resolved against the days the catalog actually contains, so "Tuesday" means
    the Tuesday *of the event* even when you run this in September.
    """
    if not days:
        msg = "cannot resolve a day without the event's calendar"
        raise ValueError(msg)
    calendar = sorted(set(days))
    text = token.strip().lower()
    if not text:
        msg = "empty day token"
        raise ValueError(msg)
    try:
        exact = date.fromisoformat(text)
    except ValueError:
        pass
    else:
        if exact in calendar:
            return exact
        msg = f"{exact.isoformat()} is not a day of this event"
        raise ValueError(msg)

    abbr = text[:3]
    if abbr in _WEEKDAY_ABBR:
        wanted = _WEEKDAY_ABBR.index(abbr)
        for candidate in calendar:
            if candidate.weekday() == wanted:
                return candidate
        msg = f"no day of this event falls on {abbr!r}"
        raise ValueError(msg)

    if text.startswith("day") and text[3:].strip().isdigit():
        index = int(text[3:].strip()) - 1
        if 0 <= index < len(calendar):
            return calendar[index]
        msg = f"day index out of range: {token!r} (event has {len(calendar)} days)"
        raise ValueError(msg)

    msg = f"unrecognised day {token!r} (try 'tue', '2026-12-01', or 'day2')"
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
# TimeSpan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, order=True)
class TimeSpan:
    """A half-open interval ``[start, end)``.

    Half-open is what makes back-to-back sessions work correctly: a 10:00-11:00
    talk and an 11:00-12:00 talk do not *overlap*, but they leave zero minutes to
    cross the Strip, which is exactly the distinction the travel constraint
    needs to see.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        require_aware(self.start, field="TimeSpan.start")
        require_aware(self.end, field="TimeSpan.end")
        if self.end <= self.start:
            msg = (
                f"TimeSpan must have positive length: "
                f"{self.start.isoformat()} -> {self.end.isoformat()}"
            )
            raise ValueError(msg)

    @classmethod
    def starting(cls, start: datetime, minutes: int) -> TimeSpan:
        return cls(start, start + timedelta(minutes=minutes))

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    def local(self, zone: ZoneInfo) -> tuple[datetime, datetime]:
        return self.start.astimezone(zone), self.end.astimezone(zone)

    def day_in(self, zone: ZoneInfo) -> date:
        """The calendar day this span starts on, as seen in ``zone``."""
        return self.start.astimezone(zone).date()

    def overlaps(self, other: TimeSpan) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    def gap_before(self, other: TimeSpan) -> int | None:
        """Minutes of clear air between this span ending and ``other`` starting.

        ``None`` when they overlap or are out of order, which forces callers to
        handle "these cannot be sequenced" rather than reading a negative number
        as a tight connection.
        """
        if other.start < self.end:
            return None
        return int((other.start - self.end).total_seconds() // 60)

    def shifted(self, minutes: int) -> TimeSpan:
        delta = timedelta(minutes=minutes)
        return TimeSpan(self.start + delta, self.end + delta)

    def format_in(self, zone: ZoneInfo) -> str:
        local_start, local_end = self.local(zone)
        return f"{local_start:%a %H:%M}-{local_end:%H:%M}"

    def __str__(self) -> str:
        return self.format_in(REINVENT_TZ)


def iter_overlapping_pairs(
    spans: Sequence[tuple[str, TimeSpan]],
) -> Iterator[tuple[str, str]]:
    """Yield every colliding pair of keyed spans.

    Sorting first makes this O(n log n + collisions) instead of O(n^2). On a real
    re:Invent catalog that is the difference between a few thousand comparisons
    and several million, and the solver rebuilds this on every run.
    """
    ordered = sorted(spans, key=lambda item: (item[1].start, item[1].end, item[0]))
    for index, (left_key, left) in enumerate(ordered):
        for right_key, right in ordered[index + 1 :]:
            if right.start >= left.end:
                break
            yield (left_key, right_key)


__all__ = [
    "PERSONAL_TIME_FORMAT",
    "PERSONAL_TIME_INCREMENT_MINUTES",
    "REINVENT_TZ",
    "PersonalTimeFormatError",
    "TimeSpan",
    "day_label",
    "format_personal_time",
    "iter_overlapping_pairs",
    "now_utc",
    "parse_personal_time",
    "parse_session_time",
    "require_aware",
    "resolve_day",
    "resolve_timezone",
    "round_to_increment",
    "validate_personal_time_span",
]
