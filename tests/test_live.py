"""Tests for the real-time re-planner.

The behaviour worth pinning down is the part the main solver cannot do: ruling a
session out because of where you are standing, rather than because of the catalog. A
14:30 in the Venetian is available if you are in the Venetian at 14:10 and not if you
are in Mandalay Bay, and nothing about the session changed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from replan.adapters.wire import build_catalog, decode_event, decode_session
from replan.domain import AttendeeSchedule, Catalog, Room, Session
from replan.live import NowReport, Urgency, parse_location, parse_moment, replan_now
from replan.profiles import Profile
from replan.travel import default_graph

EVENT = "reinvent2026"
DAY = "2026-12-01"
NEXT_DAY = "2026-12-02"


def session_at(
    code: str,
    clock: str,
    *,
    minutes: int = 60,
    venue: str | None = "Venetian",
    room: str = "Level 2 | Hall C",
    day: str = DAY,
    reservable: bool | None = None,
    seats: str | None = None,
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": code,
        "abbreviation": code,
        "title": f"Session {code}",
        "abstract": "About AWS Lambda and serverless.",
        "level": "300 \u2013 Advanced",
        "type": "Breakout session",
        "room": room,
        "services": ["AWS Lambda"],
        "topics": ["Serverless"],
        "sessionTime": {
            "date": day,
            "time": clock,
            "length": str(minutes),
            "timezone": "America/Los_Angeles",
        },
    }
    if venue is not None:
        payload["venue"] = venue
    if reservable is not None:
        payload["isReservable"] = reservable
    if seats is not None:
        payload["seatAvailability"] = seats
    decoded = decode_session(payload)
    assert decoded is not None
    return decoded


def catalog_of(*sessions: Session) -> Catalog:
    event = decode_event(
        {
            "eventId": EVENT,
            "name": "re:Invent 2026",
            "startDate": "2026-11-30T05:30:00.000-08:00",
            "endDate": "2026-12-04T23:59:00.000-08:00",
            "timezone": "America/Los_Angeles",
            "authenticationRequired": True,
        }
    )
    return build_catalog(event_id=EVENT, sessions=sessions, event=event, source="test")


PROFILE = Profile(topics={"lambda": 5.0}, min_break_minutes=0, walk_tolerance_minutes=120)


def report_for(
    catalog: Catalog,
    *,
    at: str,
    location: str | None = None,
    profile: Profile | None = None,
    schedule: AttendeeSchedule | None = None,
    limit: int = 3,
) -> NowReport:
    return replan_now(
        catalog,
        profile or PROFILE,
        default_graph(),
        at=at,
        location=location,
        schedule=schedule,
        limit=limit,
    )


# --------------------------------------------------------------------------- #
# Moment parsing
# --------------------------------------------------------------------------- #
def test_bare_clock_resolves_to_a_conference_day() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    moment = parse_moment("14:10", catalog)
    assert moment.at.hour == 14
    assert moment.at.minute == 10
    assert moment.day in catalog.days


def test_weekday_and_clock_resolve_together() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", day=DAY), session_at("B1", "09:00", day=NEXT_DAY)
    )
    moment = parse_moment("wed 14:10", catalog)
    assert moment.day.isoformat() == NEXT_DAY  # 2026-12-02 is a Wednesday
    assert moment.at.hour == 14


def test_iso_timestamp_is_accepted() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    moment = parse_moment("2026-12-01T16:45", catalog)
    assert moment.day.isoformat() == DAY
    assert moment.at.hour == 16
    assert moment.at.minute == 45


def test_now_uses_the_reference_clock() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    reference = datetime(2026, 12, 1, 18, 30, tzinfo=UTC)
    moment = parse_moment("now", catalog, reference=reference)
    assert moment.day.isoformat() == DAY
    assert moment.is_during_event


def test_now_outside_the_event_falls_forward_and_says_so() -> None:
    """Running this in October must still be useful, and must not pretend otherwise."""
    catalog = catalog_of(session_at("A1", "09:00"))
    reference = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    moment = parse_moment("now", catalog, reference=reference)
    assert moment.is_during_event is False
    assert moment.day in catalog.days


@pytest.mark.parametrize("bad", ["lunchtime", "25:00", "14:99", "half past two"])
def test_unparseable_times_are_rejected_clearly(bad: str) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    with pytest.raises(ValueError, match="time"):
        parse_moment(bad, catalog)


@pytest.mark.parametrize("blank", [None, "", "   ", "now"])
def test_blank_or_now_means_the_current_moment(blank: str | None) -> None:
    """An omitted --at is not an error; it means now."""
    catalog = catalog_of(session_at("A1", "09:00"))
    reference = datetime(2026, 12, 1, 18, 30, tzinfo=UTC)
    moment = parse_moment(blank, catalog, reference=reference)
    assert moment.day.isoformat() == DAY


def test_moment_reports_the_event_timezone() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    moment = parse_moment("14:10", catalog)
    assert str(moment.zone) == "America/Los_Angeles"
    assert "14:10" in moment.clock


# --------------------------------------------------------------------------- #
# Location parsing
# --------------------------------------------------------------------------- #
def test_location_accepts_a_room_string() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    room = parse_location("Venetian | Level 2 | Hall C", catalog)
    assert room is not None
    assert room.known_venue is not None


def test_location_accepts_a_session_code() -> None:
    """Mid-conference you know which session you are in, not the room's full name."""
    catalog = catalog_of(session_at("A1", "09:00", room="Level 5 | Palazzo Ballroom B"))
    room = parse_location("A1", catalog)
    assert room is not None
    assert room.name == "Level 5 | Palazzo Ballroom B"


def test_no_location_is_allowed() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    assert parse_location(None, catalog) is None
    assert parse_location("   ", catalog) is None


# --------------------------------------------------------------------------- #
# Position changes what is reachable
# --------------------------------------------------------------------------- #
def test_position_rules_out_an_otherwise_valid_session() -> None:
    """The thing the main solver cannot know.

    A 14:30 in the Venetian is fine from the Venetian and impossible from Mandalay Bay,
    and the session itself did not change.
    """
    catalog = catalog_of(session_at("NEAR1", "14:30", venue="Venetian", room="Level 2 | Hall C"))

    close = report_for(catalog, at="14:10", location="Venetian | Level 2 | Hall C")
    assert [move.item.code for move in close.moves] == ["NEAR1"]

    far = report_for(catalog, at="14:10", location="Mandalay Bay | Level 1 | South Pacific F")
    assert far.is_empty
    assert far.unreachable


def test_without_a_location_only_the_clock_constrains() -> None:
    catalog = catalog_of(
        session_at("A1", "14:30", venue="Mandalay Bay", room="Level 1 | South Pacific F")
    )
    report = report_for(catalog, at="14:10")
    assert [move.item.code for move in report.moves] == ["A1"]


def test_sessions_already_started_are_not_offered() -> None:
    catalog = catalog_of(
        session_at("PAST1", "09:00"),
        session_at("FUTURE1", "16:00"),
    )
    report = report_for(catalog, at="14:10")
    assert [move.item.code for move in report.moves] == ["FUTURE1"]


def test_walk_tolerance_still_applies() -> None:
    catalog = catalog_of(
        session_at("FAR1", "18:00", venue="Mandalay Bay", room="Level 1 | South Pacific F")
    )
    generous = report_for(
        catalog,
        at="09:00",
        location="Venetian | Level 2 | Hall C",
        profile=PROFILE.model_copy(update={"walk_tolerance_minutes": 120}),
    )
    assert generous.moves

    strict = report_for(
        catalog,
        at="09:00",
        location="Venetian | Level 2 | Hall C",
        profile=PROFILE.model_copy(update={"walk_tolerance_minutes": 15}),
    )
    assert strict.is_empty
    assert strict.unreachable


def test_only_todays_sessions_are_considered() -> None:
    catalog = catalog_of(
        session_at("TODAY1", "16:00", day=DAY),
        session_at("TOMORROW1", "09:00", day=NEXT_DAY),
    )
    report = report_for(catalog, at="14:10")
    assert [move.item.code for move in report.moves] == ["TODAY1"]


# --------------------------------------------------------------------------- #
# Departure times
# --------------------------------------------------------------------------- #
def test_departure_time_is_reported_not_just_a_duration() -> None:
    catalog = catalog_of(session_at("A1", "14:30", venue="Wynn", room="Level 1 | Latour 1"))
    report = report_for(catalog, at="12:00", location="Venetian | Level 2 | Hall C")
    move = report.next_move
    assert move is not None
    assert move.travel_minutes > 0
    assert move.depart_at is not None
    assert move.depart_at < move.item.span.start
    assert move.depart_clock is not None
    # The departure must be exactly start minus travel.
    expected = move.item.span.start - (move.item.span.start - move.depart_at)
    assert expected == move.depart_at


def test_leave_now_when_departure_is_this_minute() -> None:
    """Venetian to Wynn is 23 minutes, so a 14:30 session departs at 14:07."""
    catalog = catalog_of(session_at("A1", "14:30", venue="Wynn", room="Level 1 | Latour 1"))
    travel = default_graph().minutes(
        Room(venue="Venetian", name="Level 2 | Hall C"),
        Room(venue="Wynn", name="Level 1 | Latour 1"),
    )
    assert travel == 23

    report = report_for(catalog, at="14:07", location="Venetian | Level 2 | Hall C")
    move = report.next_move
    assert move is not None
    assert move.urgency is Urgency.LEAVE_NOW
    assert move.minutes_until_departure == 0


def test_a_session_you_cannot_reach_is_dropped_not_urged() -> None:
    """Better to omit it than urge someone to run for a train that has gone."""
    catalog = catalog_of(session_at("A1", "14:30", venue="Wynn", room="Level 1 | Latour 1"))
    # Ten minutes to cover a 23-minute transfer: not possible.
    report = report_for(catalog, at="14:20", location="Venetian | Level 2 | Hall C")
    assert report.is_empty
    assert report.unreachable
    assert report.unreachable[0][0].code == "A1"


def test_leave_soon_inside_ten_minutes() -> None:
    catalog = catalog_of(session_at("A1", "14:30", venue="Wynn", room="Level 1 | Latour 1"))
    report = report_for(catalog, at="14:02", location="Venetian | Level 2 | Hall C")
    move = report.next_move
    assert move is not None
    assert move.urgency is Urgency.LEAVE_SOON


def test_comfortable_when_there_is_plenty_of_time() -> None:
    catalog = catalog_of(session_at("A1", "17:00", venue="Wynn", room="Level 1 | Latour 1"))
    report = report_for(catalog, at="12:00", location="Venetian | Level 2 | Hall C")
    move = report.next_move
    assert move is not None
    assert move.urgency is Urgency.COMFORTABLE


def test_same_room_says_stay_put_rather_than_leave_now() -> None:
    """ "Leave now (0 min)" is a contradiction; the answer is that you are already there."""
    catalog = catalog_of(session_at("A1", "14:30", venue="Venetian", room="Level 2 | Hall C"))
    report = report_for(catalog, at="14:25", location="Venetian | Level 2 | Hall C")
    move = report.next_move
    assert move is not None
    assert move.travel_minutes == 0
    assert move.urgency is Urgency.STAY_PUT
    assert "stay put" in move.describe()
    assert move.urgency.needs_movement is False


# --------------------------------------------------------------------------- #
# Being mid-session
# --------------------------------------------------------------------------- #
def test_current_session_is_detected_from_the_schedule() -> None:
    catalog = catalog_of(
        session_at("CURRENT1", "14:00", minutes=60),
        session_at("NEXT1", "16:00"),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("CURRENT1",))
    report = report_for(catalog, at="14:20", schedule=schedule)

    assert report.in_progress is not None
    assert report.in_progress.code == "CURRENT1"
    assert any("until" in note for note in report.notes)


def test_next_move_waits_for_the_current_session_to_end() -> None:
    catalog = catalog_of(
        session_at("CURRENT1", "14:00", minutes=60, room="Level 2 | Hall C"),
        session_at("SOON1", "14:30", room="Level 2 | Titian 2205"),
        session_at("AFTER1", "16:00", room="Level 2 | Titian 2205"),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("CURRENT1",))
    report = report_for(catalog, at="14:20", schedule=schedule)

    codes = [move.item.code for move in report.moves]
    # SOON1 starts while you are still in CURRENT1, so it cannot be attended.
    assert "SOON1" not in codes
    assert "AFTER1" in codes


def test_first_move_after_a_current_session_is_marked_accordingly() -> None:
    catalog = catalog_of(
        session_at("CURRENT1", "14:00", minutes=60),
        session_at("AFTER1", "16:00", room="Level 5 | Palazzo Ballroom B"),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("CURRENT1",))
    report = report_for(catalog, at="14:20", schedule=schedule)
    move = report.next_move
    assert move is not None
    assert move.urgency is Urgency.AFTER_CURRENT


def test_travel_is_measured_from_the_current_session_room() -> None:
    """You will be leaving from where the session is, not where you were standing."""
    catalog = catalog_of(
        session_at("CURRENT1", "14:00", minutes=60, venue="Wynn", room="Level 1 | Latour 1"),
        session_at("AFTER1", "17:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("CURRENT1",))
    report = report_for(
        catalog,
        at="14:20",
        location="Mandalay Bay | Level 1 | South Pacific F",
        schedule=schedule,
    )
    move = report.next_move
    assert move is not None
    # Same room as the session you are in, so no travel at all.
    assert move.travel_minutes == 0


# --------------------------------------------------------------------------- #
# Held reservations
# --------------------------------------------------------------------------- #
def test_existing_reservation_later_today_is_kept() -> None:
    catalog = catalog_of(
        session_at("HELD1", "16:00", reservable=True, seats="available"),
        session_at("BETTER1", "16:00", room="Level 5 | Palazzo Ballroom B"),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("HELD1",))
    report = report_for(catalog, at="14:10", schedule=schedule)

    codes = [move.item.code for move in report.moves]
    assert "HELD1" in codes
    assert any("holding your existing reservation" in note for note in report.notes)


def test_reservations_earlier_today_are_not_mentioned() -> None:
    catalog = catalog_of(
        session_at("PAST1", "09:00", reservable=True, seats="available"),
        session_at("LATER1", "16:00"),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("PAST1",))
    report = report_for(catalog, at="14:10", schedule=schedule)
    assert not any("holding your existing" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# Limits and reporting
# --------------------------------------------------------------------------- #
def test_limit_caps_the_number_of_moves() -> None:
    catalog = catalog_of(*[session_at(f"S{i}", f"{14 + i:02d}:00") for i in range(6)])
    profile = PROFILE.model_copy(update={"max_sessions_per_day": 10})
    report = report_for(catalog, at="13:00", profile=profile, limit=2)
    assert len(report.moves) == 2


def test_empty_day_is_reported_clearly() -> None:
    catalog = catalog_of(session_at("MORNING1", "09:00"))
    report = report_for(catalog, at="20:00")
    assert report.is_empty
    assert any("nothing left" in note for note in report.notes)


def test_unreachable_sessions_are_listed_with_their_cost() -> None:
    catalog = catalog_of(
        session_at("FAR1", "14:20", venue="Mandalay Bay", room="Level 1 | South Pacific F")
    )
    report = report_for(catalog, at="14:10", location="Venetian | Level 2 | Hall C")
    assert report.is_empty
    assert report.unreachable
    session, minutes = report.unreachable[0]
    assert session.code == "FAR1"
    assert minutes > 10


def test_moves_are_in_time_order() -> None:
    catalog = catalog_of(
        session_at("LATE1", "17:00", room="Level 2 | Hall C"),
        session_at("EARLY1", "15:00", room="Level 2 | Hall C"),
    )
    profile = PROFILE.model_copy(update={"max_sessions_per_day": 5})
    report = report_for(catalog, at="14:00", profile=profile)
    starts = [move.item.span.start for move in report.moves]
    assert starts == sorted(starts)


def test_report_carries_the_underlying_plan() -> None:
    catalog = catalog_of(session_at("A1", "16:00"))
    report = report_for(catalog, at="14:00")
    assert report.plan.items
    assert report.plan.stats.solver


def test_seat_state_survives_into_the_moves() -> None:
    catalog = catalog_of(session_at("A1", "16:00", reservable=True, seats="limited"))
    report = report_for(catalog, at="14:00")
    move = report.next_move
    assert move is not None
    assert move.item.needs_reservation


# --------------------------------------------------------------------------- #
# Real catalog
# --------------------------------------------------------------------------- #
def test_works_on_the_real_catalog(
    list_sessions_payload: dict[str, Any], get_event_payload: dict[str, Any]
) -> None:
    from replan.adapters.wire import decode_catalog_page
    from replan.profiles import DEFAULT_PROFILE_YAML

    event = decode_event(get_event_payload.get("event", get_event_payload))
    page = decode_catalog_page(list_sessions_payload, fallback_timezone=event.timezone_name)
    catalog = build_catalog(
        event_id=event.event_id,
        sessions=page.sessions,
        event=event,
        source="recorded",
        total_count=page.total_count,
    )
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)

    report = replan_now(catalog, profile, default_graph(), at="09:40", location="DVT301", limit=3)
    assert report.moment.day in catalog.days
    assert report.location is not None

    for move in report.moves:
        # Nothing offered may start before you could possibly arrive.
        assert move.item.span.start >= report.moment.at
        if move.depart_at is not None:
            assert move.depart_at <= move.item.span.start


def test_real_catalog_respects_position(
    list_sessions_payload: dict[str, Any], get_event_payload: dict[str, Any]
) -> None:
    from replan.adapters.wire import decode_catalog_page
    from replan.profiles import DEFAULT_PROFILE_YAML

    event = decode_event(get_event_payload.get("event", get_event_payload))
    page = decode_catalog_page(list_sessions_payload, fallback_timezone=event.timezone_name)
    catalog = build_catalog(
        event_id=event.event_id, sessions=page.sessions, event=event, source="recorded"
    )
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML).model_copy(
        update={"walk_tolerance_minutes": 5}
    )

    # A five-minute tolerance cannot even cross a single building.
    report = replan_now(
        catalog,
        profile,
        default_graph(),
        at="09:40",
        location="Venetian | Level 2 | Hall C",
    )
    assert report.is_empty or all(move.travel_minutes <= 5 for move in report.moves)


def test_location_can_be_omitted_on_the_real_catalog(
    list_sessions_payload: dict[str, Any], get_event_payload: dict[str, Any]
) -> None:
    from replan.adapters.wire import decode_catalog_page
    from replan.profiles import DEFAULT_PROFILE_YAML

    event = decode_event(get_event_payload.get("event", get_event_payload))
    page = decode_catalog_page(list_sessions_payload, fallback_timezone=event.timezone_name)
    catalog = build_catalog(
        event_id=event.event_id, sessions=page.sessions, event=event, source="recorded"
    )
    report = replan_now(
        catalog, Profile.from_yaml(DEFAULT_PROFILE_YAML), default_graph(), at="09:00"
    )
    assert report.location is None
    assert report.moves


def test_room_parse_round_trips_through_location() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    room = Room(venue="Venetian", name="Level 2 | Hall C")
    parsed = parse_location(room.display, catalog)
    assert parsed is not None
    assert parsed.known_venue == room.known_venue
    assert parsed.name == room.name
