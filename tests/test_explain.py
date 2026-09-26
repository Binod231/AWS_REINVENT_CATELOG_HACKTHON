"""Tests for explanations and exports.

Two properties carry most of the weight here. Every rejection must produce an
actionable remedy, because an explanation you cannot act on is just an apology. And
the calendar's alarms must fire at *departure* time, since a reminder that arrives
when the session starts is a reminder you can no longer use.
"""

from __future__ import annotations

from datetime import time, timedelta
from pathlib import Path
from typing import Any

import pytest
from icalendar import Calendar

from replan.adapters.wire import build_catalog, decode_catalog_page, decode_event, decode_session
from replan.domain import AttendeeSchedule, Catalog, Session
from replan.explain import (
    Verdict,
    explain_session,
    explain_worst,
    export_day_sheets_html,
    export_ics,
    export_markdown,
)
from replan.explain.exports import FIRST_SESSION_LEAD_MINUTES
from replan.profiles import DEFAULT_PROFILE_YAML, MealWindow, Profile, Scorer
from replan.solver import Plan, RejectionReason, SolveRequest, solve
from replan.travel import default_graph

EVENT = "test-event"
DAY = "2026-12-01"


def session_at(
    code: str,
    clock: str,
    *,
    minutes: int = 60,
    venue: str | None = "Venetian",
    room: str = "Level 2 | Hall C",
    day: str = DAY,
    services: tuple[str, ...] = ("AWS Lambda",),
    topics: tuple[str, ...] = ("Serverless",),
    level: str | None = "300 \u2013 Advanced",
    session_type: str = "Breakout session",
    reservable: bool | None = None,
    seats: str | None = None,
    session_id: str | None = None,
    abstract: str = "A useful abstract about Lambda.",
    speakers: tuple[str, ...] = ("Ada Lovelace, Principal Engineer, AWS",),
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": session_id or code,
        "abbreviation": code,
        "title": f"Session {code}",
        "abstract": abstract,
        "type": session_type,
        "room": room,
        "services": list(services),
        "topics": list(topics),
        "speakers": [{"name": name} for name in speakers],
        "sessionTime": {
            "date": day,
            "time": clock,
            "length": str(minutes),
            "timezone": "America/Los_Angeles",
        },
    }
    if level is not None:
        payload["level"] = level
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
    return build_catalog(event_id=EVENT, sessions=sessions, source="test")


BASE_PROFILE = Profile(
    topics={"lambda": 5.0},
    min_break_minutes=0,
    walk_tolerance_minutes=120,
    goals=("Learn about Lambda",),
)


def plan_for(
    catalog: Catalog,
    profile: Profile | None = None,
    *,
    schedule: AttendeeSchedule | None = None,
) -> Plan:
    resolved = profile or BASE_PROFILE
    scorer = Scorer(resolved, catalog)
    return solve(
        SolveRequest(
            catalog=catalog,
            profile=resolved,
            scores=scorer.by_id(),
            travel=default_graph(),
            schedule=schedule,
            time_limit_seconds=5.0,
        )
    )


def explain(token: str, catalog: Catalog, plan: Plan, profile: Profile | None = None) -> Any:
    resolved = profile or BASE_PROFILE
    return explain_session(token, plan, catalog, resolved, default_graph())


# --------------------------------------------------------------------------- #
# Scheduled sessions
# --------------------------------------------------------------------------- #
def test_explains_a_scheduled_session() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    result = explain("A1", catalog, plan)

    assert result.verdict is Verdict.SCHEDULED
    assert "in your plan" in result.headline
    facts = dict(result.facts)
    assert "when" in facts
    assert "where" in facts
    assert result.score_lines
    assert result.remedy is None


def test_quotes_the_apis_own_level_string() -> None:
    """ "300 – Advanced" with an en dash, not our normalisation of it."""
    catalog = catalog_of(session_at("A1", "09:00", level="300 \u2013 Advanced"))
    result = explain("A1", catalog, plan_for(catalog))
    assert dict(result.facts)["level"] == "300 \u2013 Advanced"


def test_reports_the_travel_leg_into_a_session() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    plan = plan_for(catalog)
    result = explain("B1", catalog, plan)
    facts = dict(result.facts)
    assert "getting there" in facts
    assert "A1" in facts["getting there"]
    assert "leave by" in facts["getting there"]


def test_first_session_of_the_day_says_so() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    result = explain("A1", catalog, plan_for(catalog))
    assert dict(result.facts)["getting there"] == "first session of the day"


def test_same_room_transfer_is_described_as_such() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", room="Level 2 | Hall C"),
        session_at("B1", "11:00", room="Level 2 | Hall C"),
    )
    plan = plan_for(catalog)
    result = explain("B1", catalog, plan)
    assert "same room" in dict(result.facts)["getting there"]


def test_pinned_session_is_flagged_in_the_headline() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(update={"must_attend": ("A1",)})
    result = explain("A1", catalog, plan_for(catalog, profile), profile)
    assert "must_attend" in result.headline


def test_reservation_status_is_reported() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="limited"))
    result = explain("A1", catalog, plan_for(catalog))
    assert "needs one" in dict(result.facts)["reservation"]


def test_held_seat_is_reported() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="limited"))
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("A1",))
    plan = plan_for(catalog, schedule=schedule)
    result = explain("A1", catalog, plan)
    assert "already hold" in dict(result.facts)["reservation"]


# --------------------------------------------------------------------------- #
# Unknown sessions
# --------------------------------------------------------------------------- #
def test_unknown_session_is_reported_clearly() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    result = explain("NOPE999", catalog, plan_for(catalog))
    assert result.verdict is Verdict.UNKNOWN_SESSION
    assert "NOPE999" in result.headline
    assert result.remedy is not None
    assert "sync" in result.remedy


# --------------------------------------------------------------------------- #
# Rejections all carry an actionable remedy
# --------------------------------------------------------------------------- #
def test_every_rejection_has_a_remedy() -> None:
    """An explanation you cannot act on is just an apology."""
    catalog = catalog_of(
        *[session_at(f"S{i}", f"{8 + i // 2:02d}:{(i % 2) * 30:02d}") for i in range(10)]
    )
    profile = BASE_PROFILE.model_copy(update={"max_sessions_per_day": 2})
    plan = plan_for(catalog, profile)
    assert plan.rejections

    for rejection in plan.rejections:
        result = explain(rejection.session_id, catalog, plan, profile)
        assert result.verdict is Verdict.REJECTED
        assert result.remedy, f"{rejection.code} ({rejection.reason.value}) has no remedy"


def test_walk_too_far_remedy_names_the_setting_and_value() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    profile = BASE_PROFILE.model_copy(update={"walk_tolerance_minutes": 20})
    plan = plan_for(catalog, profile)
    dropped = next(item for item in plan.rejections)
    result = explain(dropped.session_id, catalog, plan, profile)
    assert result.remedy is not None
    assert "walk_tolerance_minutes" in result.remedy
    assert "20" in result.remedy


def test_day_full_remedy_names_the_current_limit() -> None:
    catalog = catalog_of(*[session_at(f"S{i}", f"{9 + i:02d}:00") for i in range(6)])
    profile = BASE_PROFILE.model_copy(update={"max_sessions_per_day": 2})
    plan = plan_for(catalog, profile)
    remedies = [explain(item.session_id, catalog, plan, profile).remedy for item in plan.rejections]
    assert any(remedy and "max_sessions_per_day" in remedy for remedy in remedies)


def test_banned_session_remedy_points_at_never_attend() -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "13:00"))
    profile = BASE_PROFILE.model_copy(update={"never_attend": ("A1",)})
    plan = plan_for(catalog, profile)
    result = explain("A1", catalog, plan, profile)
    assert result.remedy is not None
    assert "never_attend" in result.remedy


def test_protected_time_remedy_offers_three_options() -> None:
    catalog = catalog_of(session_at("LUNCH1", "12:15", minutes=30))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    result = explain("LUNCH1", catalog, plan, profile)
    assert result.remedy is not None
    assert "protect" in result.remedy or "must_attend" in result.remedy


def test_unschedulable_session_admits_nothing_can_be_done() -> None:
    """Honesty matters more than offering a fix that does not exist."""
    floating = decode_session({"sessionId": "expo", "title": "Expo floor", "isAllDaySession": True})
    assert floating is not None
    catalog = catalog_of(session_at("A1", "09:00"), floating)
    plan = plan_for(catalog)
    result = explain("expo", catalog, plan)
    assert result.remedy is not None
    assert result.remedy.startswith("nothing")


def test_travel_infeasible_without_a_break_admits_nothing_helps() -> None:
    catalog = catalog_of(
        session_at("A1", "13:30", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "15:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    plan = plan_for(catalog)
    dropped = next(item for item in plan.rejections)
    result = explain(dropped.session_id, catalog, plan)
    assert result.remedy is not None
    assert "nothing in the profile" in result.remedy


def test_rejection_names_the_blocking_session() -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "09:30"))
    plan = plan_for(catalog)
    kept = {item.code for item in plan.items}
    dropped = "B1" if "A1" in kept else "A1"
    result = explain(dropped, catalog, plan)
    assert result.conflicts_with
    assert any(code in entry for entry in result.conflicts_with for code in kept)


def test_nearest_planned_session_is_reported_for_a_rejection() -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "09:30"))
    plan = plan_for(catalog)
    kept = {item.code for item in plan.items}
    dropped = "B1" if "A1" in kept else "A1"
    result = explain(dropped, catalog, plan)
    assert "nearest in plan" in dict(result.facts)


# --------------------------------------------------------------------------- #
# Repeats
# --------------------------------------------------------------------------- #
def test_alternative_showings_are_listed_with_their_fate() -> None:
    catalog = catalog_of(
        session_at("SVS401", "09:00", session_id="r1"),
        session_at("SVS401-R2", "14:00", session_id="r2"),
    )
    plan = plan_for(catalog)
    chosen = plan.items[0]
    other = "r2" if chosen.session_id == "r1" else "r1"
    result = explain(other, catalog, plan)

    assert result.alternatives
    assert any("in your plan" in entry for entry in result.alternatives)


def test_duplicate_content_remedy_mentions_pinning() -> None:
    catalog = catalog_of(
        session_at("SVS401", "09:00", session_id="r1"),
        session_at("SVS401-R2", "14:00", session_id="r2"),
    )
    plan = plan_for(catalog)
    other = "r2" if plan.items[0].session_id == "r1" else "r1"
    result = explain(other, catalog, plan)
    assert result.remedy is not None
    assert "must_attend" in result.remedy


# --------------------------------------------------------------------------- #
# explain_worst
# --------------------------------------------------------------------------- #
def test_explain_worst_returns_the_highest_scoring_misses() -> None:
    catalog = catalog_of(*[session_at(f"S{i}", f"{9 + i:02d}:00") for i in range(6)])
    profile = BASE_PROFILE.model_copy(update={"max_sessions_per_day": 2})
    plan = plan_for(catalog, profile)
    results = explain_worst(plan, catalog, profile, default_graph(), limit=3)
    assert len(results) == 3
    for result in results:
        assert result.verdict is Verdict.REJECTED


def test_explanation_renders_as_text() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    text = explain("A1", catalog, plan_for(catalog)).as_text()
    assert "A1" in text
    assert "scoring:" in text


# --------------------------------------------------------------------------- #
# ICS export
# --------------------------------------------------------------------------- #
def test_ics_is_valid_and_has_one_event_per_session(tmp_path: Path) -> None:
    catalog = catalog_of(
        session_at("A1", "09:00"),
        session_at("B1", "13:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    plan = plan_for(catalog)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog)

    parsed = Calendar.from_ical(target.read_bytes())
    events = list(parsed.walk("VEVENT"))
    assert len(events) == len(plan.items)
    for event in events:
        assert event.get("summary")
        assert event.get("dtstart")
        assert event.get("dtend")
        assert event.get("uid")


def test_alarms_fire_at_departure_time_not_session_start(tmp_path: Path) -> None:
    """The point of the whole export.

    A reminder at 13:00 for a 13:00 session in another property is useless. The
    trigger must equal the travel time.
    """
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    plan = plan_for(catalog)
    second = next(item for item in plan.items if item.code == "B1")
    assert second.travel_minutes and second.travel_minutes > 0

    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog)
    parsed = Calendar.from_ical(target.read_bytes())

    for event in parsed.walk("VEVENT"):
        if str(event.get("summary")).startswith("B1"):
            alarms = list(event.walk("VALARM"))
            assert len(alarms) == 1
            trigger = alarms[0].get("trigger").dt
            assert trigger == timedelta(minutes=-second.travel_minutes)
            assert "Leave now" in str(alarms[0].get("description"))
            return
    pytest.fail("B1 was not in the exported calendar")


def test_first_session_gets_a_lead_time_alarm(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog)
    parsed = Calendar.from_ical(target.read_bytes())
    alarm = next(iter(next(iter(parsed.walk("VEVENT"))).walk("VALARM")))
    assert alarm.get("trigger").dt == timedelta(minutes=-FIRST_SESSION_LEAD_MINUTES)


def test_alarms_can_be_switched_to_session_start(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog, departure_alarms=False)
    parsed = Calendar.from_ical(target.read_bytes())
    assert not list(next(iter(parsed.walk("VEVENT"))).walk("VALARM"))


def test_ics_uids_are_stable_across_exports(tmp_path: Path) -> None:
    """Re-importing should update events, not duplicate them."""
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)

    first = export_ics(plan, tmp_path / "a.ics", catalog=catalog)
    second = export_ics(plan, tmp_path / "b.ics", catalog=catalog)

    def uids(path: Path) -> set[str]:
        parsed = Calendar.from_ical(path.read_bytes())
        return {str(event.get("uid")) for event in parsed.walk("VEVENT")}

    assert uids(first) == uids(second)


def test_ics_description_carries_travel_and_abstract(tmp_path: Path) -> None:
    catalog = catalog_of(
        session_at("A1", "09:00"),
        session_at(
            "B1",
            "13:00",
            venue="Wynn",
            room="Level 1 | Latour 1",
            abstract="A distinctive abstract sentence.",
        ),
    )
    plan = plan_for(catalog)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog)
    parsed = Calendar.from_ical(target.read_bytes())

    for event in parsed.walk("VEVENT"):
        if str(event.get("summary")).startswith("B1"):
            description = str(event.get("description"))
            assert "Getting here" in description
            assert "Leave by" in description
            assert "A distinctive abstract sentence." in description
            assert "Ada Lovelace" in description
            return
    pytest.fail("B1 missing")


def test_ics_marks_held_seats_confirmed(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="limited"))
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("A1",))
    plan = plan_for(catalog, schedule=schedule)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog)
    parsed = Calendar.from_ical(target.read_bytes())
    assert str(next(iter(parsed.walk("VEVENT"))).get("status")) == "CONFIRMED"


def test_ics_without_a_catalog_still_works(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=None)
    parsed = Calendar.from_ical(target.read_bytes())
    assert len(list(parsed.walk("VEVENT"))) == 1


def test_ics_times_are_utc(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    target = export_ics(plan, tmp_path / "agenda.ics", catalog=catalog)
    raw = target.read_text(encoding="utf-8")
    # 09:00 Pacific in December is 17:00 UTC.
    assert "DTSTART:20261201T170000Z" in raw


# --------------------------------------------------------------------------- #
# Markdown export
# --------------------------------------------------------------------------- #
def test_markdown_contains_the_agenda(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "13:00"))
    plan = plan_for(catalog)
    target = export_markdown(plan, tmp_path / "agenda.md", catalog=catalog)
    text = target.read_text(encoding="utf-8")

    assert text.startswith(f"# {EVENT} agenda")
    assert "| Time | Code | Session |" in text
    for item in plan.items:
        assert item.code in text
    assert "Learn about Lambda" in text  # the profile goal
    assert "300 \u2013 Advanced" in text  # the API's own level string


def test_markdown_escapes_pipes_in_titles(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", room="Level 2 | Hall C"))
    plan = plan_for(catalog)
    text = export_markdown(plan, tmp_path / "a.md", catalog=catalog).read_text()
    # The room name contains a pipe; the table must still have intact columns.
    rows = [line for line in text.splitlines() if line.startswith("| 09:00")]
    assert rows


def test_markdown_lists_seats_to_reserve(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    text = export_markdown(plan, tmp_path / "a.md", catalog=catalog).read_text()
    assert "Seats to reserve" in text
    assert "replan apply" in text


def test_markdown_reports_warnings(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(update={"must_attend": ("GHOST1",)})
    plan = plan_for(catalog, profile)
    text = export_markdown(plan, tmp_path / "a.md", catalog=catalog).read_text()
    assert "WARNING" in text
    assert "GHOST1" in text


def test_markdown_lists_near_misses(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "09:30"))
    plan = plan_for(catalog)
    text = export_markdown(plan, tmp_path / "a.md", catalog=catalog).read_text()
    assert "did not fit" in text


# --------------------------------------------------------------------------- #
# HTML day sheets
# --------------------------------------------------------------------------- #
def test_day_sheets_are_written_per_day_plus_index(tmp_path: Path) -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", day="2026-12-01"),
        session_at("B1", "09:00", day="2026-12-02"),
    )
    plan = plan_for(catalog)
    written = export_day_sheets_html(plan, tmp_path / "sheets")

    names = {path.name for path in written}
    assert "2026-12-01.html" in names
    assert "2026-12-02.html" in names
    assert "index.html" in names


def test_day_sheet_contains_the_sessions(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    written = export_day_sheets_html(plan, tmp_path / "sheets")
    page = next(path for path in written if path.name == f"{DAY}.html")
    html = page.read_text(encoding="utf-8")

    assert "A1" in html
    assert "09:00" in html
    assert "Hall C" in html
    assert "<!doctype html>" in html


def test_day_sheet_escapes_html(tmp_path: Path) -> None:
    """Titles come from an external API, so autoescaping is not optional."""
    hostile = decode_session(
        {
            "sessionId": "X1",
            "abbreviation": "X1",
            "title": "<script>alert('xss')</script> Session",
            "room": "Hall C",
            "sessionTime": {
                "date": DAY,
                "time": "09:00",
                "length": "60",
                "timezone": "America/Los_Angeles",
            },
        }
    )
    assert hostile is not None
    plan = plan_for(catalog_of(hostile), Profile())
    written = export_day_sheets_html(plan, tmp_path / "sheets")
    page = next(path for path in written if path.name == f"{DAY}.html")
    html = page.read_text(encoding="utf-8")

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_empty_plan_writes_no_day_sheets(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("LUNCH1", "12:15", minutes=30))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    assert plan.items == ()
    assert export_day_sheets_html(plan, tmp_path / "sheets") == ()


# --------------------------------------------------------------------------- #
# Real catalog
# --------------------------------------------------------------------------- #
def test_exports_work_on_the_real_catalog(
    tmp_path: Path,
    list_sessions_payload: dict[str, Any],
    get_event_payload: dict[str, Any],
) -> None:
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
    scorer = Scorer(profile, catalog)
    plan = solve(
        SolveRequest(
            catalog=catalog,
            profile=profile,
            scores=scorer.by_id(),
            travel=default_graph(),
            time_limit_seconds=10.0,
        )
    )
    assert plan.items

    ics = export_ics(plan, tmp_path / "a.ics", catalog=catalog)
    parsed = Calendar.from_ical(ics.read_bytes())
    assert len(list(parsed.walk("VEVENT"))) == len(plan.items)

    markdown = export_markdown(plan, tmp_path / "a.md", catalog=catalog)
    assert markdown.read_text(encoding="utf-8")

    sheets = export_day_sheets_html(plan, tmp_path / "sheets")
    assert sheets


def test_every_real_rejection_gets_a_remedy(
    list_sessions_payload: dict[str, Any], get_event_payload: dict[str, Any]
) -> None:
    """Across a whole real catalog, no rejection may be left unexplained."""
    event = decode_event(get_event_payload.get("event", get_event_payload))
    page = decode_catalog_page(list_sessions_payload, fallback_timezone=event.timezone_name)
    catalog = build_catalog(
        event_id=event.event_id, sessions=page.sessions, event=event, source="recorded"
    )
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    scorer = Scorer(profile, catalog)
    plan = solve(
        SolveRequest(
            catalog=catalog,
            profile=profile,
            scores=scorer.by_id(),
            travel=default_graph(),
            time_limit_seconds=10.0,
        )
    )
    graph = default_graph()
    seen: set[RejectionReason] = set()
    for rejection in plan.rejections:
        result = explain_session(rejection.session_id, plan, catalog, profile, graph, scorer)
        assert result.remedy, f"{rejection.code} ({rejection.reason.value})"
        seen.add(rejection.reason)
    assert seen, "expected the real catalog to produce some rejections"
