"""Decode AWS Events API payloads into domain objects, and encode what we send.

This module is deliberately paranoid. The API documents that "a field is present
only when the event provides it. Treat any single field as optional, and read
defensively", and live data confirms it: a Summit catalog returns no ``venue``,
no ``seatAvailability``, no ``isReservable`` and no ``isAllDaySession`` at all,
while re:Invent supplies them. Nothing here raises on a missing field; instead
the domain model carries the absence honestly so downstream code can decide.

Unknown keys are ignored rather than rejected. The API adds fields as the event
platform grows, and a decoder that refused new keys would break on the first
addition.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from replan.domain.enums import BulkFailureCode, Level, SeatAvailability, SessionFormat
from replan.domain.models import (
    AttendeeSchedule,
    BulkFailure,
    BulkResult,
    Catalog,
    Event,
    EventAddress,
    PersonalTime,
    Room,
    Session,
    Speaker,
)
from replan.domain.timespan import (
    TimeSpan,
    format_personal_time,
    now_utc,
    parse_personal_time,
    parse_session_time,
    validate_personal_time_span,
)

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if item is not None and str(item).strip())
    return ()


def _bool_or_none(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _aware_datetime(payload: dict[str, Any], key: str) -> datetime | None:
    """Parse an ISO-8601 timestamp with offset, as ``Event`` dates carry.

    Live example: ``"2026-11-30T05:30:00.000-08:00"``.
    """
    raw = _text(payload, key)
    if raw is None:
        return None
    candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
def decode_event(payload: dict[str, Any]) -> Event:
    address_payload = payload.get("address")
    address = None
    if isinstance(address_payload, dict):
        address = EventAddress(
            city=_text(address_payload, "city"),
            country=_text(address_payload, "country"),
            state=_text(address_payload, "state"),
            street=_text(address_payload, "street"),
            postal_code=_text(address_payload, "postalCode"),
        )

    return Event(
        event_id=_text(payload, "eventId") or "unknown",
        name=_text(payload, "name") or "Untitled event",
        event_type=_text(payload, "eventType"),
        start_date=_aware_datetime(payload, "startDate"),
        end_date=_aware_datetime(payload, "endDate"),
        is_online=bool(payload.get("isOnline")),
        timezone_name=_text(payload, "timezone"),
        timezone_abbreviation=_text(payload, "timezoneAbbreviation"),
        time_format=_text(payload, "timeFormat"),
        address=address,
        supported_language_codes=_str_tuple(payload, "supportedLanguageCodes"),
        authentication_required=bool(payload.get("authenticationRequired")),
    )


def decode_events(payload: dict[str, Any]) -> tuple[Event, ...]:
    items = payload.get("items")
    if not isinstance(items, list):
        return ()
    return tuple(decode_event(item) for item in items if isinstance(item, dict))


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
def decode_session(
    payload: dict[str, Any],
    *,
    fallback_timezone: str | None = None,
) -> Session | None:
    """Decode one session. ``None`` when it lacks the two required fields.

    Returning ``None`` rather than raising means one malformed entry cannot
    destroy a 2,000-session catalog walk; the caller counts and reports skips.
    """
    session_id = _text(payload, "sessionId")
    title = _text(payload, "title")
    if not session_id or not title:
        return None

    session_time = payload.get("sessionTime")
    span: TimeSpan | None = None
    zone_name = fallback_timezone
    if isinstance(session_time, dict):
        zone_name = _text(session_time, "timezone") or fallback_timezone
        span = parse_session_time(
            day=_text(session_time, "date"),
            clock=_text(session_time, "time"),
            length_minutes=session_time.get("length"),
            timezone_name=zone_name,
        )

    level_raw = _text(payload, "level")
    type_raw = _text(payload, "type")

    room = Room(
        venue=_text(payload, "venue"),
        name=_text(payload, "room"),
    )

    speakers = tuple(
        Speaker(name=name)
        for name in (
            _text(item, "name")
            for item in payload.get("speakers", []) or []
            if isinstance(item, dict)
        )
        if name
    )

    return Session(
        session_id=session_id,
        title=title,
        abbreviation=_text(payload, "abbreviation"),
        abstract=_text(payload, "abstract"),
        level=Level.parse(level_raw),
        level_raw=level_raw,
        format=SessionFormat.parse(type_raw),
        type_raw=type_raw,
        room=room,
        span=span,
        timezone_name=zone_name,
        is_all_day=bool(payload.get("isAllDaySession")),
        is_reservable=_bool_or_none(payload, "isReservable"),
        seat_availability=SeatAvailability.parse(payload.get("seatAvailability")),
        tracks=_str_tuple(payload, "tracks"),
        topics=_str_tuple(payload, "topics"),
        industries=_str_tuple(payload, "industries"),
        areas_of_interest=_str_tuple(payload, "areasOfInterest"),
        roles=_str_tuple(payload, "roles"),
        services=_str_tuple(payload, "services"),
        segments=_str_tuple(payload, "segments"),
        features=_str_tuple(payload, "features"),
        customer_personas=_str_tuple(payload, "customerPersonas"),
        experiences=_str_tuple(payload, "experiences"),
        additional_activities=_str_tuple(payload, "additionalActivities"),
        focus_areas=_str_tuple(payload, "focusAreas"),
        speakers=speakers,
    )


class CatalogPage(tuple[tuple[Session, ...], str | None, int | None, int]):
    """``(sessions, next_token, total_count, skipped)`` from one ListSessions page."""

    __slots__ = ()

    @property
    def sessions(self) -> tuple[Session, ...]:
        return self[0]

    @property
    def next_token(self) -> str | None:
        return self[1]

    @property
    def total_count(self) -> int | None:
        return self[2]

    @property
    def skipped(self) -> int:
        return self[3]


def decode_catalog_page(
    payload: dict[str, Any],
    *,
    fallback_timezone: str | None = None,
) -> CatalogPage:
    """Decode a ``ListSessions`` page.

    The API warns that "page sizes can vary, and a short page does not mean the
    last page. Stop when nextToken is absent, never when a page looks short." So
    the token is returned verbatim and the walk decision is left to the caller.
    """
    raw_items = payload.get("items")
    items = raw_items if isinstance(raw_items, list) else []

    sessions: list[Session] = []
    skipped = 0
    for item in items:
        if not isinstance(item, dict):
            skipped += 1
            continue
        decoded = decode_session(item, fallback_timezone=fallback_timezone)
        if decoded is None:
            skipped += 1
            continue
        sessions.append(decoded)

    total_raw = payload.get("totalCount")
    total = int(total_raw) if isinstance(total_raw, int | float) else None
    next_token = _text(payload, "nextToken")

    return CatalogPage((tuple(sessions), next_token, total, skipped))


def build_catalog(
    *,
    event_id: str,
    sessions: tuple[Session, ...],
    event: Event | None = None,
    source: str,
    total_count: int | None = None,
    locale: str | None = None,
    fetched_at: datetime | None = None,
) -> Catalog:
    return Catalog(
        event_id=event_id,
        sessions=sessions,
        event=event,
        source=source,
        total_count=total_count,
        locale=locale,
        fetched_at=fetched_at or now_utc(),
    )


# --------------------------------------------------------------------------- #
# Schedule and personal time
# --------------------------------------------------------------------------- #
def decode_personal_time(payload: dict[str, Any]) -> PersonalTime | None:
    entry_id = _text(payload, "personalTimeId")
    title = _text(payload, "title")
    start_raw = _text(payload, "startDateTime")
    end_raw = _text(payload, "endDateTime")
    if not entry_id or not title or not start_raw or not end_raw:
        return None
    try:
        start = parse_personal_time(start_raw)
        end = parse_personal_time(end_raw)
        span = TimeSpan(start, end)
    except ValueError:
        return None
    return PersonalTime(
        personal_time_id=entry_id,
        title=title,
        description=_text(payload, "description") or "",
        span=span,
        location=_text(payload, "location"),
    )


def decode_schedule(payload: dict[str, Any], *, event_id: str) -> AttendeeSchedule:
    """Decode ``GetSchedule``.

    The response nests everything under ``schedule``, and the API guarantees each
    list is present and empty rather than absent - but we tolerate absence anyway.
    """
    body = payload.get("schedule") if isinstance(payload.get("schedule"), dict) else payload
    if not isinstance(body, dict):
        body = {}

    blocks = tuple(
        block
        for block in (
            decode_personal_time(item)
            for item in body.get("personalTime", []) or []
            if isinstance(item, dict)
        )
        if block is not None
    )

    return AttendeeSchedule(
        event_id=event_id,
        reserved=_str_tuple(body, "reserved"),
        favorites=_str_tuple(body, "favorites"),
        personal_time=blocks,
        fetched_at=now_utc(),
    )


def encode_personal_time(
    *,
    span: TimeSpan,
    title: str,
    description: str,
    location: str | None = None,
) -> dict[str, str]:
    """Build a ``PersonalTimeInput`` body.

    Enforces the documented rules *before* spending a request: naive UTC with no
    zone marker, seconds at ``00``, and a duration that is a whole number of
    5-minute increments. ``title`` and ``description`` are both required, so an
    empty description is filled rather than omitted.
    """
    validate_personal_time_span(span)
    body: dict[str, str] = {
        "startDateTime": format_personal_time(span.start),
        "endDateTime": format_personal_time(span.end),
        "title": title.strip() or "Personal time",
        "description": description.strip() or title.strip() or "Personal time",
    }
    if location and location.strip():
        body["location"] = location.strip()
    return body


# --------------------------------------------------------------------------- #
# Bulk results
# --------------------------------------------------------------------------- #
def decode_bulk_result(payload: dict[str, Any]) -> BulkResult:
    """Decode ``ReserveSessions`` / ``AssociateFavorites``.

    A 200 here does not mean success: the API reports per session, and sessions
    that were *already* reserved or favorited come back in ``failed``. The domain
    object folds those into ``effective_successes`` so callers reason about the
    end state instead of the transport.
    """
    body = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if not isinstance(body, dict):
        body = {}

    failures: list[BulkFailure] = []
    for item in body.get("failed", []) or []:
        if not isinstance(item, dict):
            continue
        session_id = _text(item, "sessionId")
        if not session_id:
            continue
        raw_code = _text(item, "code")
        failures.append(
            BulkFailure(
                session_id=session_id,
                code=BulkFailureCode.parse(raw_code),
                code_raw=raw_code,
                conflicts_with=_str_tuple(item, "conflictsWith"),
            )
        )

    return BulkResult(
        successful=_str_tuple(body, "successful"),
        failed=tuple(failures),
    )


__all__ = [
    "CatalogPage",
    "build_catalog",
    "decode_bulk_result",
    "decode_catalog_page",
    "decode_event",
    "decode_events",
    "decode_personal_time",
    "decode_schedule",
    "decode_session",
    "encode_personal_time",
]
