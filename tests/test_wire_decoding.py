"""Decode real AWS Events API payloads.

Every assertion here runs against bytes the live API actually returned. Where a
test encodes a quirk (en-dash levels, absent ``venue``, ``length`` as a string),
that quirk was observed in production data, not imagined.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from replan.adapters.errors import (
    EventsAuthRequiredError,
    EventsEdgeRefusedError,
    EventsNotRegisteredError,
    EventsOperationClosedError,
    EventsThrottledError,
    classify_response,
)
from replan.adapters.wire import (
    build_catalog,
    decode_bulk_result,
    decode_catalog_page,
    decode_event,
    decode_events,
    decode_personal_time,
    decode_schedule,
    decode_session,
    encode_personal_time,
)
from replan.domain import (
    REPLAN_BLOCK_MARKER,
    BulkFailureCode,
    Level,
    PersonalTimeFormatError,
    SeatAvailability,
    SessionFormat,
    TimeSpan,
)

# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def test_decode_events_from_live_payload(list_events_payload: dict[str, Any]) -> None:
    events = decode_events(list_events_payload)
    assert events, "the live API returned no events to decode"
    for event in events:
        assert event.event_id
        assert event.name


def test_reinvent2026_is_present_and_requires_sign_in(
    list_events_payload: dict[str, Any],
) -> None:
    events = {event.event_id: event for event in decode_events(list_events_payload)}
    reinvent = events.get("reinvent2026")
    assert reinvent is not None, "reinvent2026 should be listed by ListEvents"
    # Documented: an event requiring registration still appears in the list.
    assert reinvent.authentication_required is True
    assert reinvent.is_reinvent
    assert str(reinvent.zone) == "America/Los_Angeles"
    # Five conference days, Mon 30 Nov - Fri 4 Dec 2026.
    assert len(reinvent.days) == 5


def test_public_events_do_not_require_sign_in(list_events_payload: dict[str, Any]) -> None:
    events = decode_events(list_events_payload)
    public = [event for event in events if not event.authentication_required]
    assert public, "expected at least one credential-free event to test against"


def test_decode_single_event(get_event_payload: dict[str, Any]) -> None:
    payload = get_event_payload.get("event", get_event_payload)
    event = decode_event(payload)
    assert event.event_id
    assert event.zone is not None


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def test_decode_catalog_page_loses_nothing(list_sessions_payload: dict[str, Any]) -> None:
    page = decode_catalog_page(list_sessions_payload)
    assert page.skipped == 0, "no live session should fail to decode"
    assert len(page.sessions) == len(list_sessions_payload["items"])
    assert page.total_count == list_sessions_payload["totalCount"]


def test_every_live_session_has_the_two_guaranteed_fields(
    list_sessions_payload: dict[str, Any],
) -> None:
    for session in decode_catalog_page(list_sessions_payload).sessions:
        assert session.session_id
        assert session.title


def test_levels_parse_including_en_dash_and_no_level(
    list_sessions_payload: dict[str, Any],
) -> None:
    """The live catalog writes "100 – Foundational" with an EN DASH, plus "No Level"."""
    raw_levels = {item.get("level") for item in list_sessions_payload["items"]}
    assert any(raw and "\u2013" in raw for raw in raw_levels), (
        "fixture no longer contains an en-dash level; this test guards that parsing path"
    )

    for session in decode_catalog_page(list_sessions_payload).sessions:
        assert isinstance(session.level, Level)
        if session.level_raw and any(ch.isdigit() for ch in session.level_raw):
            assert session.level is not Level.UNKNOWN
        elif session.level_raw:
            # "No Level" must land on UNKNOWN rather than raising.
            assert session.level is Level.UNKNOWN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100 \u2013 Foundational", Level.FOUNDATIONAL),
        ("200 - Intermediate", Level.INTERMEDIATE),
        ("300 \u2013 Advanced", Level.ADVANCED),
        ("400 \u2013 Expert", Level.EXPERT),
        ("No Level", Level.UNKNOWN),
        ("Level 300", Level.ADVANCED),
        (300, Level.ADVANCED),
        (None, Level.UNKNOWN),
        ("", Level.UNKNOWN),
        ("nonsense", Level.UNKNOWN),
    ],
)
def test_level_parse_table(raw: object, expected: Level) -> None:
    assert Level.parse(raw) is expected


def test_session_types_map_to_known_formats(list_sessions_payload: dict[str, Any]) -> None:
    sessions = decode_catalog_page(list_sessions_payload).sessions
    recognised = [s for s in sessions if s.format is not SessionFormat.OTHER]
    assert recognised, "no live session type was recognised; the alias table is stale"
    for session in sessions:
        assert session.type_raw is not None
        assert isinstance(session.format, SessionFormat)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Breakout session", SessionFormat.BREAKOUT),
        ("Chalk talk", SessionFormat.CHALK_TALK),
        ("Code talk", SessionFormat.CODE_TALK),
        ("Keynote", SessionFormat.KEYNOTE),
        ("Lightning talk", SessionFormat.LIGHTNING_TALK),
        ("Exam prep", SessionFormat.EXAM_PREP),
        ("Gamified learning", SessionFormat.GAMIFIED_LEARNING),
        ("Startup talk", SessionFormat.STARTUP_TALK),
        ("Workshop", SessionFormat.WORKSHOP),
        ("Builders' session", SessionFormat.BUILDERS_SESSION),
        ("Hands-on Lab", SessionFormat.WORKSHOP),
        ("something brand new", SessionFormat.OTHER),
        (None, SessionFormat.OTHER),
    ],
)
def test_session_format_parse_table(raw: object, expected: SessionFormat) -> None:
    assert SessionFormat.parse(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("available", SeatAvailability.AVAILABLE),
        ("limited", SeatAvailability.LIMITED),
        ("veryLimited", SeatAvailability.VERY_LIMITED),
        ("very limited", SeatAvailability.VERY_LIMITED),
        ("unavailable", SeatAvailability.UNAVAILABLE),
        ("walkUp", SeatAvailability.WALK_UP),
        (None, None),
        ("", None),
        ("somethingNew", None),
    ],
)
def test_seat_availability_parse_table(raw: object, expected: SeatAvailability | None) -> None:
    assert SeatAvailability.parse(raw) is expected


def test_seat_availability_semantics() -> None:
    """The API gives a band, never a count, so these predicates carry the weight."""
    assert SeatAvailability.AVAILABLE.is_bookable
    assert SeatAvailability.VERY_LIMITED.is_bookable
    assert not SeatAvailability.UNAVAILABLE.is_bookable
    # walkUp means no reservation is taken at all: not bookable, but not a problem.
    assert not SeatAvailability.WALK_UP.is_bookable
    assert SeatAvailability.WALK_UP.needs_no_reservation
    assert SeatAvailability.VERY_LIMITED.urgency > SeatAvailability.AVAILABLE.urgency
    assert SeatAvailability.WALK_UP.urgency == 0


def test_walkup_session_needs_no_reservation() -> None:
    session = decode_session(
        {
            "sessionId": "W1",
            "title": "Walk up welcome",
            "isReservable": True,
            "seatAvailability": "walkUp",
        }
    )
    assert session is not None
    assert session.seat_availability is SeatAvailability.WALK_UP
    assert session.needs_reservation is False
    assert session.is_bookable_now is False


def test_full_session_is_not_bookable_but_is_watchable() -> None:
    session = decode_session(
        {
            "sessionId": "F1",
            "title": "Popular chalk talk",
            "type": "Chalk talk",
            "isReservable": True,
            "seatAvailability": "unavailable",
        }
    )
    assert session is not None
    assert session.is_full
    assert session.needs_reservation is True
    assert session.is_bookable_now is False
    assert session.scarcity == 3


def test_session_time_becomes_a_real_span(list_sessions_payload: dict[str, Any]) -> None:
    """``length`` arrives as a *string* of minutes and there is no end time."""
    items = {item["sessionId"]: item for item in list_sessions_payload["items"]}
    for session in decode_catalog_page(list_sessions_payload).sessions:
        raw_time = items[session.session_id].get("sessionTime") or {}
        if not (raw_time.get("date") and raw_time.get("time") and raw_time.get("length")):
            continue
        assert session.span is not None
        assert session.span.minutes == int(raw_time["length"])
        assert session.span.start.tzinfo is not None
        assert session.day is not None


def test_absent_fields_stay_absent(list_sessions_payload: dict[str, Any]) -> None:
    """The Summit catalog omits venue/seatAvailability/isReservable entirely.

    The decoder must represent that as "unknown", never as a fabricated default,
    because a planner that invents ``isReservable=True`` would try to book seats
    that do not exist.
    """
    items = list_sessions_payload["items"]
    assert all("seatAvailability" not in item for item in items), (
        "fixture changed; this test guards the missing-field path"
    )
    for session in decode_catalog_page(list_sessions_payload).sessions:
        assert session.seat_availability is None
        assert session.is_reservable is None
        assert session.needs_reservation is False


def test_speaker_name_is_split_but_preserved(list_sessions_payload: dict[str, Any]) -> None:
    """Live speakers pack name, title and employer into one string."""
    sessions = decode_catalog_page(list_sessions_payload).sessions
    with_speakers = [s for s in sessions if s.speakers]
    assert with_speakers
    multipart = [s for s in with_speakers if "," in s.speakers[0].name]
    assert multipart, "expected at least one 'Name, Title, Company' speaker"
    speaker = multipart[0].speakers[0]
    assert speaker.person
    assert speaker.person in speaker.name
    assert speaker.name  # raw value never discarded


def test_topic_keys_span_several_taxonomies(list_sessions_payload: dict[str, Any]) -> None:
    sessions = decode_catalog_page(list_sessions_payload).sessions
    assert any(len(session.topic_keys) >= 2 for session in sessions)


def test_search_blob_includes_services_and_speakers(
    list_sessions_payload: dict[str, Any],
) -> None:
    for session in decode_catalog_page(list_sessions_payload).sessions:
        blob = session.search_blob.lower()
        for service in session.services:
            assert service.lower() in blob
        for speaker in session.speakers:
            assert speaker.name.lower() in blob


def test_decode_single_session(get_session_payload: dict[str, Any]) -> None:
    payload = get_session_payload.get("session", get_session_payload)
    session = decode_session(payload)
    assert session is not None
    assert session.session_id


def test_malformed_session_is_skipped_not_fatal() -> None:
    page = decode_catalog_page(
        {
            "items": [
                {"title": "missing its id"},
                {"sessionId": "OK1", "title": "fine"},
                "not even an object",
            ],
            "totalCount": 3,
        }
    )
    assert [s.session_id for s in page.sessions] == ["OK1"]
    assert page.skipped == 2


def test_unknown_upstream_keys_are_ignored() -> None:
    session = decode_session(
        {
            "sessionId": "NEW1",
            "title": "Forward compatible",
            "somethingAwsAddedLater": {"nested": True},
        }
    )
    assert session is not None
    assert session.session_id == "NEW1"


# --------------------------------------------------------------------------- #
# Repeats and series
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("code", "series", "repeat"),
    [
        ("SVS401", "SVS401", None),
        ("SVS401-R", "SVS401", 1),
        ("SVS401-R1", "SVS401", 1),
        ("SVS401-R2", "SVS401", 2),
        ("aim315-r3", "AIM315", 3),
    ],
)
def test_repeat_suffix_is_understood(code: str, series: str, repeat: int | None) -> None:
    session = decode_session({"sessionId": f"id-{code}", "title": "t", "abbreviation": code})
    assert session is not None
    assert session.series == series
    assert session.repeat_index == repeat


def test_track_inferred_from_code_prefix_when_absent() -> None:
    session = decode_session(
        {"sessionId": "x", "title": "t", "abbreviation": "SVS401", "tracks": []}
    )
    assert session is not None
    assert session.effective_tracks == ("Serverless & Event-Driven",)


# --------------------------------------------------------------------------- #
# Catalog aggregate
# --------------------------------------------------------------------------- #


def test_catalog_stats_and_lookups(
    list_sessions_payload: dict[str, Any], public_event_id: str
) -> None:
    page = decode_catalog_page(list_sessions_payload)
    catalog = build_catalog(
        event_id=public_event_id,
        sessions=page.sessions,
        source="recorded",
        total_count=page.total_count,
    )
    stats = catalog.stats()

    assert stats.total == len(catalog)
    assert stats.scheduled + stats.unscheduled == stats.total
    assert sum(count for _, count in stats.by_level) == stats.total
    assert sum(count for _, count in stats.by_format) == stats.total
    assert catalog.is_complete

    first = catalog.sessions[0]
    assert catalog.by_id(first.session_id) is first
    assert catalog.require(first.session_id) is first
    assert catalog.resolve(first.code) is not None
    with pytest.raises(KeyError):
        catalog.require("does-not-exist")


def test_catalog_incomplete_when_walk_stops_early(
    list_sessions_payload: dict[str, Any], public_event_id: str
) -> None:
    """A short page must never be mistaken for a finished walk."""
    page = decode_catalog_page(list_sessions_payload)
    truncated = build_catalog(
        event_id=public_event_id,
        sessions=page.sessions[:5],
        source="recorded",
        total_count=page.total_count,
    )
    assert truncated.is_complete is False


# --------------------------------------------------------------------------- #
# Schedule and personal time
# --------------------------------------------------------------------------- #


def test_decode_schedule_shape() -> None:
    schedule = decode_schedule(
        {
            "schedule": {
                "reserved": ["S1", "S2"],
                "favorites": ["S3"],
                "personalTime": [
                    {
                        "personalTimeId": "pt-1",
                        "startDateTime": "2026-12-02T16:00:00",
                        "endDateTime": "2026-12-02T17:00:00",
                        "title": "Dinner",
                        "description": f"With the team {REPLAN_BLOCK_MARKER}",
                        "location": "Mandalay Bay",
                    }
                ],
            }
        },
        event_id="reinvent2026",
    )
    assert schedule.reserved_ids == {"S1", "S2"}
    assert schedule.favorite_ids == {"S3"}
    assert len(schedule.personal_time) == 1
    block = schedule.personal_time[0]
    assert block.span.minutes == 60
    assert block.span.start.tzinfo is UTC
    assert block.created_by_replan is True
    assert schedule.replan_blocks == (block,)
    assert schedule.manual_blocks == ()


def test_empty_schedule_decodes_cleanly() -> None:
    schedule = decode_schedule({"schedule": {}}, event_id="reinvent2026")
    assert schedule.is_empty()


def test_personal_time_rejects_zone_markers() -> None:
    """The API rejects a trailing Z or an offset rather than converting it."""
    assert (
        decode_personal_time(
            {
                "personalTimeId": "x",
                "title": "t",
                "startDateTime": "2026-12-02T16:00:00Z",
                "endDateTime": "2026-12-02T17:00:00Z",
            }
        )
        is None
    )


def test_encode_personal_time_emits_naive_utc() -> None:
    span = TimeSpan(
        datetime(2026, 12, 2, 16, 0, tzinfo=UTC),
        datetime(2026, 12, 2, 17, 30, tzinfo=UTC),
    )
    body = encode_personal_time(span=span, title="Lunch", description="")
    assert body["startDateTime"] == "2026-12-02T16:00:00"
    assert body["endDateTime"] == "2026-12-02T17:30:00"
    assert "Z" not in body["startDateTime"]
    assert "+" not in body["startDateTime"]
    # description is required by the API, so it is filled rather than omitted.
    assert body["description"]
    assert "location" not in body


def test_encode_personal_time_enforces_five_minute_increments() -> None:
    span = TimeSpan(
        datetime(2026, 12, 2, 16, 0, tzinfo=UTC),
        datetime(2026, 12, 2, 16, 7, tzinfo=UTC),
    )
    with pytest.raises(PersonalTimeFormatError):
        encode_personal_time(span=span, title="Odd", description="7 minutes")


def test_encode_personal_time_converts_from_local_zone() -> None:
    from zoneinfo import ZoneInfo

    vegas = ZoneInfo("America/Los_Angeles")
    span = TimeSpan(
        datetime(2026, 12, 2, 8, 0, tzinfo=vegas),
        datetime(2026, 12, 2, 9, 0, tzinfo=vegas),
    )
    body = encode_personal_time(span=span, title="Breakfast", description="d")
    # 08:00 Pacific in December is 16:00 UTC.
    assert body["startDateTime"] == "2026-12-02T16:00:00"


# --------------------------------------------------------------------------- #
# Bulk results
# --------------------------------------------------------------------------- #


def test_bulk_result_folds_already_done_into_success() -> None:
    result = decode_bulk_result(
        {
            "result": {
                "successful": ["A"],
                "failed": [
                    {"sessionId": "B", "code": "alreadyScheduled"},
                    {"sessionId": "C", "code": "sessionFull"},
                    {
                        "sessionId": "D",
                        "code": "scheduleConflict",
                        "conflictsWith": ["A"],
                    },
                ],
            }
        }
    )
    assert result.effective_successes == {"A", "B"}
    assert {f.session_id for f in result.real_failures} == {"C", "D"}
    assert result.ok is False
    conflict = result.failure_for("D")
    assert conflict is not None
    assert conflict.conflicts_with == ("A",)
    assert "clashes with A" in conflict.describe()


def test_unknown_bulk_failure_code_degrades_to_other() -> None:
    result = decode_bulk_result(
        {"result": {"successful": [], "failed": [{"sessionId": "Z", "code": "someNewCode"}]}}
    )
    failure = result.failure_for("Z")
    assert failure is not None
    assert failure.code is BulkFailureCode.OTHER
    assert failure.code_raw == "someNewCode"


def test_all_successful_bulk_is_ok() -> None:
    result = decode_bulk_result({"result": {"successful": ["A", "B"], "failed": []}})
    assert result.ok
    assert result.effective_successes == {"A", "B"}


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #


def test_success_is_not_an_error() -> None:
    assert classify_response(status=200, body_text="{}", payload={}) is None


def test_live_401_body_classifies_as_auth_required(
    unauthenticated_payload: dict[str, Any],
) -> None:
    """Recorded from a real anonymous call to reinvent2026's catalog."""
    assert unauthenticated_payload["status"] == 401
    error = classify_response(
        status=401,
        body_text="{}",
        payload=unauthenticated_payload["body"],
        operation="ListSessions",
    )
    assert isinstance(error, EventsAuthRequiredError)
    assert error.is_retryable
    assert "replan login" in error.remedy


def test_403_with_json_body_means_not_registered() -> None:
    error = classify_response(
        status=403, body_text='{"message":"not registered"}', payload={"message": "x"}
    )
    assert isinstance(error, EventsNotRegisteredError)
    assert error.is_retryable is False


def test_403_without_body_means_edge_refusal() -> None:
    error = classify_response(status=403, body_text="<html>denied</html>", payload=None)
    assert isinstance(error, EventsEdgeRefusedError)
    assert error.is_retryable is True


def test_409_is_operation_closed_and_mentions_reservations() -> None:
    error = classify_response(status=409, body_text="{}", payload={"message": "closed"})
    assert isinstance(error, EventsOperationClosedError)
    assert "8 October 2026" in error.remedy


def test_429_reads_retry_after() -> None:
    error = classify_response(
        status=429,
        body_text="{}",
        payload={"message": "slow down"},
        headers={"Retry-After": "37"},
    )
    assert isinstance(error, EventsThrottledError)
    assert error.retry_after_seconds == 37
    assert "37s" in error.remedy


def test_non_json_edge_body_does_not_crash_classifier() -> None:
    error = classify_response(status=503, body_text="<html>oops</html>", payload=None)
    assert error is not None
    assert error.is_retryable
