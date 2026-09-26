"""Tests for the seat watcher.

The central behaviour is transition detection. The API publishes no seat count and no
waitlist, so "a seat opened" is only knowable by comparing against what was seen last
time. Everything else here protects the attendee from an automated booking making their
schedule worse.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from replan.adapters.auth import NotSignedInError
from replan.adapters.errors import (
    EventsApiError,
    EventsNotFoundError,
    EventsOperationClosedError,
    EventsThrottledError,
)
from replan.adapters.wire import build_catalog, decode_session
from replan.domain import (
    AttendeeSchedule,
    BulkFailure,
    BulkFailureCode,
    BulkResult,
    Catalog,
    SeatAvailability,
    Session,
)
from replan.profiles import Profile, Scorer
from replan.travel import default_graph
from replan.watch import (
    ConsoleNotifier,
    MultiNotifier,
    Notification,
    NotificationKind,
    SlackNotifier,
    SnsNotifier,
    Verdict,
    Watcher,
    WatchPolicy,
    WatchStore,
    build_notifier,
)
from replan.watch.policy import band_improved
from replan.watch.watcher import MIN_INTERVAL_SECONDS

EVENT = "reinvent2026"
DAY = "2026-12-01"


def session_at(
    code: str,
    clock: str = "14:00",
    *,
    minutes: int = 60,
    seats: str | None = "unavailable",
    reservable: bool | None = True,
    venue: str | None = "Venetian",
    room: str = "Level 2 | Hall C",
    day: str = DAY,
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": code,
        "abbreviation": code,
        "title": f"Session {code}",
        "abstract": "About AWS Lambda.",
        "level": "300 \u2013 Advanced",
        "type": "Chalk talk",
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
    if seats is not None:
        payload["seatAvailability"] = seats
    if reservable is not None:
        payload["isReservable"] = reservable
    decoded = decode_session(payload)
    assert decoded is not None
    return decoded


def catalog_of(*sessions: Session) -> Catalog:
    return build_catalog(event_id=EVENT, sessions=sessions, source="test")


PROFILE = Profile(topics={"lambda": 5.0}, min_break_minutes=0, walk_tolerance_minutes=120)
BEFORE_SESSIONS = datetime(2026, 12, 1, 16, 0, tzinfo=UTC)  # 08:00 Pacific


class WatchPort:
    """A port whose seat bands can be changed between polls."""

    surface_name = "fake"

    def __init__(self, *sessions: Session) -> None:
        self.sessions = {item.session_id: item for item in sessions}
        self.reserved: set[str] = set()
        self.favorites: set[str] = set()
        self.calls: list[str] = []
        self.schedule_raises: Exception | None = None
        self.session_raises: dict[str, Exception] = {}
        self.reserve_raises: Exception | None = None
        self.reserve_result: BulkResult | None = None

    def set_band(self, session_id: str, band: str | None) -> None:
        current = self.sessions[session_id]
        self.sessions[session_id] = current.model_copy(
            update={"seat_availability": SeatAvailability.parse(band)}
        )

    def get_session(self, event_id: str, session_id: str, *, locale: str | None = None) -> Session:
        self.calls.append(f"GetSession:{session_id}")
        if session_id in self.session_raises:
            raise self.session_raises[session_id]
        found = self.sessions.get(session_id)
        if found is None:
            raise EventsNotFoundError("no such session", status=404)
        return found

    def get_schedule(self, event_id: str) -> AttendeeSchedule:
        self.calls.append("GetSchedule")
        if self.schedule_raises is not None:
            raise self.schedule_raises
        return AttendeeSchedule(
            event_id=event_id,
            reserved=tuple(sorted(self.reserved)),
            favorites=tuple(sorted(self.favorites)),
        )

    def reserve_sessions(self, event_id: str, session_ids: list[str]) -> BulkResult:
        self.calls.append(f"ReserveSessions:{','.join(session_ids)}")
        if self.reserve_raises is not None:
            raise self.reserve_raises
        if self.reserve_result is not None:
            return self.reserve_result
        self.reserved.update(session_ids)
        return BulkResult(successful=tuple(session_ids), failed=())


class CapturingNotifier:
    name = "capture"

    def __init__(self) -> None:
        self.sent: list[Notification] = []

    def send(self, notification: Notification) -> bool:
        self.sent.append(notification)
        return True


def make_watcher(
    port: WatchPort,
    store: WatchStore,
    *,
    catalog: Catalog,
    policy: WatchPolicy | None = None,
    notifier: Any = None,
    profile: Profile | None = None,
    clock: Any = None,
) -> Watcher:
    return Watcher(
        port,
        store,
        event_id=EVENT,
        catalog=catalog,
        profile=profile or PROFILE,
        travel=default_graph(),
        policy=policy or WatchPolicy(),
        notifier=notifier,
        sleeper=lambda _seconds: None,
        clock=clock or (lambda: BEFORE_SESSIONS),
        rng=random.Random(1),
    )


@pytest.fixture
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path / "watch.db")


# --------------------------------------------------------------------------- #
# Transition detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("unavailable", "available", True),
        ("unavailable", "limited", True),
        ("unavailable", "veryLimited", True),
        (None, "available", True),
        ("veryLimited", "available", True),
        ("limited", "available", True),
        ("available", "available", False),
        ("limited", "veryLimited", False),
        ("available", "limited", False),
        ("available", "unavailable", False),
        ("unavailable", "unavailable", False),
        ("unavailable", None, False),
        ("unavailable", "walkUp", False),
    ],
)
def test_band_improvement_detection(
    previous: str | None, current: str | None, expected: bool
) -> None:
    """The entire mechanism: a comparison, because there is no seat count to threshold."""
    assert (
        band_improved(SeatAvailability.parse(previous), SeatAvailability.parse(current)) is expected
    )


def test_store_records_the_band_on_add_so_the_first_poll_has_a_baseline(
    store: WatchStore,
) -> None:
    """Without a baseline the first check would look like a transition."""
    session = session_at("A1", seats="unavailable")
    target = store.add(EVENT, session)
    assert target.last_band is SeatAvailability.UNAVAILABLE


def test_no_notification_when_the_band_does_not_change(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    notifier = CapturingNotifier()

    result = make_watcher(port, store, catalog=catalog_of(session), notifier=notifier).poll_once()

    assert result.checked == 1
    assert not result.openings
    assert notifier.sent == []


def test_a_seat_that_is_already_free_is_reported_on_the_first_check(store: WatchStore) -> None:
    """Otherwise a watch added while the seat was open would sit silent forever.

    ``band_improved`` says nothing here because the band never changes, so the watcher
    treats the first sighting of a bookable seat as an opening in its own right.
    """
    session = session_at("A1", seats="available")
    port = WatchPort(session)
    store.add(EVENT, session)
    notifier = CapturingNotifier()
    watcher = make_watcher(port, store, catalog=catalog_of(session), notifier=notifier)

    first = watcher.poll_once()
    assert len(first.openings) == 1
    assert "already, on the first check" in notifier.sent[0].body

    # Only once: a still-open seat must not notify on every poll thereafter.
    assert watcher.poll_once().openings == ()


def test_notification_when_a_seat_frees_up(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    notifier = CapturingNotifier()
    watcher = make_watcher(port, store, catalog=catalog_of(session), notifier=notifier)

    assert watcher.poll_once().openings == ()

    port.set_band("A1", "limited")
    result = watcher.poll_once()

    assert len(result.openings) == 1
    assert result.notified == ("A1",)
    assert len(notifier.sent) == 1
    assert notifier.sent[0].kind is NotificationKind.SEAT_OPEN
    assert "unavailable -> limited" in notifier.sent[0].body


# --------------------------------------------------------------------------- #
# Auto-reserve
# --------------------------------------------------------------------------- #
def test_auto_reserve_is_off_by_default(store: WatchStore) -> None:
    """Spending a reservation unasked is a big step."""
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(port, store, catalog=catalog_of(session))

    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()

    assert result.openings[0].verdict is Verdict.NOTIFY_ONLY
    assert "--auto-reserve" in result.openings[0].reason
    assert port.reserved == set()


def test_auto_reserve_takes_the_seat(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    notifier = CapturingNotifier()
    watcher = make_watcher(
        port,
        store,
        catalog=catalog_of(session),
        policy=WatchPolicy(auto_reserve=True),
        notifier=notifier,
    )

    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()

    assert result.reserved == ("A1",)
    assert "A1" in port.reserved
    assert notifier.sent[-1].kind is NotificationKind.RESERVED

    target = store.get(EVENT, "A1")
    assert target is not None
    assert target.reserved_at is not None
    # Settled targets are not polled again.
    assert store.pending(EVENT) == ()


def test_minimum_score_gates_auto_reserve(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(
        port,
        store,
        catalog=catalog_of(session),
        policy=WatchPolicy(auto_reserve=True, minimum_score=10_000.0),
    )
    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()

    assert result.openings[0].verdict is Verdict.NOTIFY_ONLY
    assert "threshold" in result.openings[0].reason
    assert port.reserved == set()


# --------------------------------------------------------------------------- #
# Protecting the existing schedule
# --------------------------------------------------------------------------- #
def test_conflicting_seat_is_never_auto_reserved(store: WatchStore) -> None:
    """Taking an overlapping seat trades a certainty for a clash."""
    held = session_at("HELD1", "14:00", seats="available")
    wanted = session_at("WANT1", "14:30", seats="unavailable")
    catalog = catalog_of(held, wanted)
    port = WatchPort(held, wanted)
    port.reserved.add("HELD1")
    store.add(EVENT, wanted)

    watcher = make_watcher(port, store, catalog=catalog, policy=WatchPolicy(auto_reserve=True))
    watcher.poll_once()
    port.set_band("WANT1", "available")
    result = watcher.poll_once()

    decision = result.openings[0]
    assert decision.verdict is Verdict.NOTIFY_ONLY
    assert "overlaps a seat you already hold" in decision.reason
    assert decision.conflicts_with == ("HELD1",)
    assert "WANT1" not in port.reserved


def test_unreachable_seat_is_not_auto_reserved(store: WatchStore) -> None:
    """Non-overlapping is not the same as reachable."""
    held = session_at("HELD1", "13:00", venue="Venetian", room="Level 2 | Hall C")
    wanted = session_at(
        "WANT1",
        "14:10",
        seats="unavailable",
        venue="Mandalay Bay",
        room="Level 1 | South Pacific F",
    )
    catalog = catalog_of(held, wanted)
    port = WatchPort(held, wanted)
    port.reserved.add("HELD1")
    store.add(EVENT, wanted)

    watcher = make_watcher(port, store, catalog=catalog, policy=WatchPolicy(auto_reserve=True))
    watcher.poll_once()
    port.set_band("WANT1", "available")
    result = watcher.poll_once()

    decision = result.openings[0]
    assert decision.verdict is Verdict.NOTIFY_ONLY
    assert "could not get there" in decision.reason
    assert "WANT1" not in port.reserved


def test_conflicts_can_be_allowed_explicitly(store: WatchStore) -> None:
    held = session_at("HELD1", "14:00", seats="available")
    wanted = session_at("WANT1", "14:30", seats="unavailable")
    port = WatchPort(held, wanted)
    port.reserved.add("HELD1")
    store.add(EVENT, wanted)

    watcher = make_watcher(
        port,
        store,
        catalog=catalog_of(held, wanted),
        policy=WatchPolicy(auto_reserve=True, allow_conflicts=True, require_travel_feasible=False),
    )
    watcher.poll_once()
    port.set_band("WANT1", "available")
    result = watcher.poll_once()
    assert result.reserved == ("WANT1",)


def test_already_held_seat_is_ignored(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(port, store, catalog=catalog_of(session))
    watcher.poll_once()

    port.reserved.add("A1")
    port.set_band("A1", "available")
    result = watcher.poll_once()
    assert result.openings == ()


def test_started_session_is_ignored(store: WatchStore) -> None:
    session = session_at("A1", "09:00", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    # 20:00 UTC is midday Pacific, well after a 09:00 session began.
    late = datetime(2026, 12, 1, 20, 0, tzinfo=UTC)
    watcher = make_watcher(port, store, catalog=catalog_of(session), clock=lambda: late)

    watcher.poll_once()
    port.set_band("A1", "available")
    assert watcher.poll_once().openings == ()


def test_walkup_is_reported_but_not_reserved(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(
        port, store, catalog=catalog_of(session), policy=WatchPolicy(auto_reserve=True)
    )
    watcher.poll_once()
    port.set_band("A1", "walkUp")
    result = watcher.poll_once()
    # walkUp is not an improvement, so nothing fires at all.
    assert result.openings == ()


def test_policy_reports_walkup_as_notify_only_when_evaluated(store: WatchStore) -> None:
    session = session_at("A1", seats="walkUp")
    target = store.add(EVENT, session)
    decision = WatchPolicy(auto_reserve=True).evaluate(
        session,
        target,
        schedule=AttendeeSchedule(event_id=EVENT),
        catalog=catalog_of(session),
        travel=default_graph(),
        profile=PROFILE,
        reference=BEFORE_SESSIONS,
    )
    assert decision.verdict is Verdict.NOTIFY_ONLY
    assert "walk-up" in decision.reason


# --------------------------------------------------------------------------- #
# Reservations closed, and degraded modes
# --------------------------------------------------------------------------- #
def test_closed_reservations_degrade_to_notify_only(store: WatchStore) -> None:
    """Exactly today's situation: reservations 409 until 8 October 2026."""
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(
        port,
        store,
        catalog=catalog_of(session),
        policy=WatchPolicy(auto_reserve=True, reservations_open=False),
    )
    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()

    decision = result.openings[0]
    assert decision.verdict is Verdict.NOTIFY_ONLY
    assert "8 October 2026" in decision.reason
    assert port.reserved == set()


def test_a_409_during_reserve_latches_and_is_reported(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    port.reserve_raises = EventsOperationClosedError("closed", status=409)
    store.add(EVENT, session)
    watcher = make_watcher(
        port, store, catalog=catalog_of(session), policy=WatchPolicy(auto_reserve=True)
    )
    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()

    assert result.reserved == ()
    assert any("8 October 2026" in problem for problem in result.errors)
    assert watcher.reservations_closed


def test_unreadable_schedule_degrades_to_notify_only(store: WatchStore) -> None:
    """A seat we cannot check against the schedule is a seat we must not book."""
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    port.schedule_raises = NotSignedInError("not signed in; run `replan login`")
    store.add(EVENT, session)
    watcher = make_watcher(
        port, store, catalog=catalog_of(session), policy=WatchPolicy(auto_reserve=True)
    )

    first = watcher.poll_once()
    assert first.checked == 1
    assert any("notify-only" in note for note in first.skipped)

    port.set_band("A1", "available")
    result = watcher.poll_once()
    assert result.openings
    assert result.reserved == ()
    assert port.reserved == set()


def test_reserve_refusal_is_reported(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    port.reserve_result = BulkResult(
        successful=(),
        failed=(BulkFailure(session_id="A1", code=BulkFailureCode.SESSION_FULL),),
    )
    store.add(EVENT, session)
    watcher = make_watcher(
        port, store, catalog=catalog_of(session), policy=WatchPolicy(auto_reserve=True)
    )
    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()

    assert result.reserved == ()
    assert any("full" in problem for problem in result.errors)


def test_already_reserved_refusal_counts_as_success(store: WatchStore) -> None:
    """The API reports an existing reservation as a failure; the end state is the point."""
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    port.reserve_result = BulkResult(
        successful=(),
        failed=(BulkFailure(session_id="A1", code=BulkFailureCode.ALREADY_SCHEDULED),),
    )
    store.add(EVENT, session)
    watcher = make_watcher(
        port, store, catalog=catalog_of(session), policy=WatchPolicy(auto_reserve=True)
    )
    watcher.poll_once()
    port.set_band("A1", "available")
    assert watcher.poll_once().reserved == ("A1",)


# --------------------------------------------------------------------------- #
# Error handling and politeness
# --------------------------------------------------------------------------- #
def test_a_removed_session_is_dropped_from_the_watch_list(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    port.session_raises["A1"] = EventsNotFoundError("gone", status=404)
    store.add(EVENT, session)

    result = make_watcher(port, store, catalog=catalog_of(session)).poll_once()
    assert any("no longer in the catalog" in note for note in result.skipped)
    assert store.pending(EVENT) == ()


def test_throttling_waits_for_retry_after(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    port.session_raises["A1"] = EventsThrottledError(
        "slow down", status=429, retry_after_seconds=17
    )
    store.add(EVENT, session)

    slept: list[float] = []
    watcher = Watcher(
        port,
        store,
        event_id=EVENT,
        catalog=catalog_of(session),
        profile=PROFILE,
        travel=default_graph(),
        notifier=None,
        sleeper=slept.append,
        clock=lambda: BEFORE_SESSIONS,
    )
    result = watcher.poll_once()
    assert 17 in slept
    assert any("throttled" in problem for problem in result.errors)


def test_other_api_errors_are_collected_not_fatal(store: WatchStore) -> None:
    good = session_at("A1", seats="unavailable")
    bad = session_at("B1", seats="unavailable")
    port = WatchPort(good, bad)
    port.session_raises["B1"] = EventsApiError("boom", status=500)
    store.add(EVENT, good)
    store.add(EVENT, bad)

    result = make_watcher(port, store, catalog=catalog_of(good, bad)).poll_once()
    assert result.checked == 1
    assert len(result.errors) == 1


def test_polling_interval_has_a_floor(store: WatchStore) -> None:
    session = session_at("A1")
    watcher = Watcher(
        WatchPort(session),
        store,
        event_id=EVENT,
        catalog=catalog_of(session),
        profile=PROFILE,
        travel=default_graph(),
        interval_seconds=1,
    )
    assert watcher.interval_seconds == MIN_INTERVAL_SECONDS


def test_run_stops_when_the_list_is_settled(store: WatchStore) -> None:
    session = session_at("A1", seats="available")
    port = WatchPort(session)
    store.add(EVENT, session)
    notifier = CapturingNotifier()
    watcher = make_watcher(
        port,
        store,
        catalog=catalog_of(session),
        policy=WatchPolicy(auto_reserve=True),
        notifier=notifier,
    )

    results = watcher.run(iterations=5)
    assert len(results) == 1  # reserved on the first pass, so nothing left to watch
    assert any(item.kind is NotificationKind.WATCHER_STOPPED for item in notifier.sent)


def test_run_announces_itself(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    notifier = CapturingNotifier()
    watcher = make_watcher(port, store, catalog=catalog_of(session), notifier=notifier)

    watcher.run(iterations=1)
    assert notifier.sent[0].kind is NotificationKind.WATCHER_STARTED


def test_empty_watch_list_polls_nothing(store: WatchStore) -> None:
    port = WatchPort()
    result = make_watcher(port, store, catalog=catalog_of()).poll_once()
    assert result.checked == 0
    assert port.calls == []


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def test_store_round_trip(store: WatchStore) -> None:
    session = session_at("A1", seats="limited")
    store.add(EVENT, session, priority=7)
    target = store.get(EVENT, "A1")

    assert target is not None
    assert target.code == "A1"
    assert target.priority == 7
    assert target.last_band is SeatAvailability.LIMITED
    assert target.is_settled is False


def test_store_remove_and_clear(store: WatchStore) -> None:
    store.add(EVENT, session_at("A1"))
    store.add(EVENT, session_at("B1"))
    assert store.remove(EVENT, "A1") is True
    assert store.remove(EVENT, "A1") is False
    assert store.clear(EVENT) == 1
    assert store.list(EVENT) == ()


def test_store_orders_by_priority(store: WatchStore) -> None:
    store.add(EVENT, session_at("LOW1"), priority=1)
    store.add(EVENT, session_at("HIGH1"), priority=9)
    assert [item.code for item in store.list(EVENT)] == ["HIGH1", "LOW1"]


def test_store_isolates_events(store: WatchStore) -> None:
    store.add("one", session_at("A1"))
    store.add("two", session_at("B1"))
    assert [item.code for item in store.list("one")] == ["A1"]
    assert store.events() == ("one", "two")


def test_store_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "watch.db"
    WatchStore(path).add(EVENT, session_at("A1"))
    assert len(WatchStore(path).list(EVENT)) == 1


def test_checks_counter_increments(store: WatchStore) -> None:
    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(port, store, catalog=catalog_of(session))
    watcher.poll_once()
    watcher.poll_once()

    target = store.get(EVENT, "A1")
    assert target is not None
    assert target.checks == 2


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def test_seed_from_plan_adds_unbooked_reservable_sessions(store: WatchStore) -> None:
    from replan.solver import SolveRequest, solve

    wanted = session_at("A1", "14:00", seats="unavailable", reservable=True)
    walkin = session_at("B1", "16:00", seats=None, reservable=False)
    catalog = catalog_of(wanted, walkin)
    scorer = Scorer(PROFILE, catalog)
    plan = solve(
        SolveRequest(
            catalog=catalog,
            profile=PROFILE,
            scores=scorer.by_id(),
            travel=default_graph(),
            time_limit_seconds=5.0,
        )
    )

    watcher = make_watcher(WatchPort(wanted, walkin), store, catalog=catalog)
    added = watcher.seed_from_plan(plan)

    codes = {item.code for item in added}
    assert "A1" in codes
    assert "B1" not in codes  # takes no reservation, so nothing to watch


def test_add_by_code_and_unknown_token(store: WatchStore) -> None:
    session = session_at("A1")
    watcher = make_watcher(WatchPort(session), store, catalog=catalog_of(session))
    assert watcher.add("A1") is not None
    assert watcher.add("NOPE999") is None


# --------------------------------------------------------------------------- #
# Notifiers
# --------------------------------------------------------------------------- #
def test_console_notifier_writes(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleNotifier().send(
        Notification(kind=NotificationKind.SEAT_OPEN, title="Seat available: A1")
    )
    assert "Seat available: A1" in capsys.readouterr().out


def test_console_notifier_accepts_a_writer() -> None:
    lines: list[str] = []
    ConsoleNotifier(lines.append).send(Notification(kind=NotificationKind.SEAT_OPEN, title="hello"))
    assert lines and "hello" in lines[0]


def test_notification_renders_session_details() -> None:
    session = session_at("A1", "14:00", seats="limited")
    notification = Notification(
        kind=NotificationKind.SEAT_OPEN,
        title="Seat available: A1",
        body="unavailable -> limited",
        session=session,
    )
    text = notification.as_text()
    assert "Hall C" in text
    assert "unavailable -> limited" in text

    blocks = notification.as_slack_blocks()
    assert blocks
    rendered = str(blocks)
    assert "Seat available: A1" in rendered
    assert "300 \u2013 Advanced" in rendered


def test_slack_notifier_never_leaks_the_webhook_url() -> None:
    """The webhook is a credential, and httpx puts URLs in its error messages."""
    notifier = SlackNotifier("https://hooks.slack.test/services/SECRET-TOKEN-VALUE")
    ok = notifier.send(Notification(kind=NotificationKind.SEAT_OPEN, title="hi"))
    assert ok is False
    assert notifier.last_error is not None
    assert "SECRET-TOKEN-VALUE" not in notifier.last_error


def test_sns_notifier_reports_missing_boto3_or_fails_cleanly() -> None:
    notifier = SnsNotifier("arn:aws:sns:us-west-2:000000000000:nope")
    assert notifier.send(Notification(kind=NotificationKind.ERROR, title="x")) is False
    assert notifier.last_error is not None


def test_multi_notifier_succeeds_if_any_sink_works() -> None:
    lines: list[str] = []
    multi = MultiNotifier(
        SlackNotifier("https://hooks.slack.test/broken"), ConsoleNotifier(lines.append)
    )
    assert multi.send(Notification(kind=NotificationKind.SEAT_OPEN, title="hi")) is True
    assert lines
    assert multi.sinks == ("slack", "console")


def test_build_notifier_selects_sinks() -> None:
    assert build_notifier(console=True).name == "console"
    combined = build_notifier(console=True, slack_webhook_url="https://hooks.slack.test/x")
    assert combined.name == "multi"


def test_notifier_failure_does_not_stop_the_watcher(store: WatchStore) -> None:
    class BrokenNotifier:
        name = "broken"

        def send(self, notification: Notification) -> bool:
            return False

    session = session_at("A1", seats="unavailable")
    port = WatchPort(session)
    store.add(EVENT, session)
    watcher = make_watcher(port, store, catalog=catalog_of(session), notifier=BrokenNotifier())
    watcher.poll_once()
    port.set_band("A1", "available")
    result = watcher.poll_once()
    assert result.openings  # detection still worked


# --------------------------------------------------------------------------- #
# Jitter
# --------------------------------------------------------------------------- #
def test_sleep_is_jittered_within_bounds(store: WatchStore) -> None:
    """A hundred attendees must not synchronise onto the minute boundary."""
    session = session_at("A1", seats="unavailable")
    slept: list[float] = []
    watcher = Watcher(
        WatchPort(session),
        store,
        event_id=EVENT,
        catalog=catalog_of(session),
        profile=PROFILE,
        travel=default_graph(),
        interval_seconds=100,
        sleeper=slept.append,
        clock=lambda: BEFORE_SESSIONS,
        rng=random.Random(7),
    )
    store.add(EVENT, session)
    watcher.run(iterations=4, announce=False)

    assert slept
    assert len({round(value, 3) for value in slept}) > 1, "sleeps should not be identical"
    for value in slept:
        assert MIN_INTERVAL_SECONDS <= value <= 125


def test_poll_result_summary_reads_well() -> None:
    from replan.watch.watcher import PollResult

    result = PollResult(checked=3, notified=("A1",), reserved=("A1",))
    text = result.summary()
    assert "checked 3" in text
    assert "1 reserved" in text


# --------------------------------------------------------------------------- #
# Real catalog
# --------------------------------------------------------------------------- #
def test_watcher_runs_against_the_recorded_catalog(tmp_path: Path, recorded_dir: Path) -> None:
    from replan.adapters.recorded import RecordedCatalogClient

    port = RecordedCatalogClient(recorded_dir, reservations_open=True)
    event_id = port.recorded_event_id
    catalog = port.list_sessions(event_id)
    store = WatchStore(tmp_path / "watch.db")

    first = catalog.sessions[0]
    store.add(event_id, first)

    watcher = Watcher(
        port,
        store,
        event_id=event_id,
        catalog=catalog,
        profile=PROFILE,
        travel=default_graph(),
        notifier=CapturingNotifier(),
        sleeper=lambda _seconds: None,
    )
    result = watcher.poll_once()
    assert result.checked == 1
    # This Summit publishes no seatAvailability at all, so there is no band to improve.
    assert result.openings == ()
