"""Pure domain model for the AWS Events API catalog.

No I/O, no HTTP, no framework. Everything is immutable and hashable so it can be
cached, memoised, and handed to a constraint solver without defensive copying.
Adapters translate the API's payloads into these types once, at the boundary, and
nothing downstream ever sees a raw dict.
"""

from __future__ import annotations

from replan.domain.enums import (
    CODE_PREFIX_TRACKS,
    BulkFailureCode,
    KnownVenue,
    Level,
    SeatAvailability,
    SessionFormat,
)
from replan.domain.models import (
    REPLAN_BLOCK_MARKER,
    AttendeeSchedule,
    BulkFailure,
    BulkResult,
    Catalog,
    CatalogStats,
    Event,
    EventAddress,
    Frozen,
    PersonalTime,
    Room,
    Session,
    Speaker,
    slugify,
)
from replan.domain.timespan import (
    PERSONAL_TIME_FORMAT,
    PERSONAL_TIME_INCREMENT_MINUTES,
    REINVENT_TZ,
    PersonalTimeFormatError,
    TimeSpan,
    day_label,
    format_personal_time,
    iter_overlapping_pairs,
    now_utc,
    parse_personal_time,
    parse_session_time,
    require_aware,
    resolve_day,
    resolve_timezone,
    round_to_increment,
    validate_personal_time_span,
)

__all__ = [
    "CODE_PREFIX_TRACKS",
    "PERSONAL_TIME_FORMAT",
    "PERSONAL_TIME_INCREMENT_MINUTES",
    "REINVENT_TZ",
    "REPLAN_BLOCK_MARKER",
    "AttendeeSchedule",
    "BulkFailure",
    "BulkFailureCode",
    "BulkResult",
    "Catalog",
    "CatalogStats",
    "Event",
    "EventAddress",
    "Frozen",
    "KnownVenue",
    "Level",
    "PersonalTime",
    "PersonalTimeFormatError",
    "Room",
    "SeatAvailability",
    "Session",
    "SessionFormat",
    "Speaker",
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
    "slugify",
    "validate_personal_time_span",
]
