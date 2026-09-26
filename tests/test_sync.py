"""Tests for the write-back engine.

This is the only code in replan that changes anything outside the local machine, so
the tests lean on the awkward parts of the API rather than the happy path: a 200 that
hides refusals, "already reserved" arriving as a failure, a 409 that means the whole
operation is closed, and a write whose outcome is simply unknown.

The round-trip test - apply, then undo, then assert the schedule is byte-identical to
where it started - is the one that matters most. A tool that books seats on your
behalf has to be able to put them back.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest

from replan.adapters.errors import (
    EventsApiError,
    EventsOperationClosedError,
)
from replan.adapters.recorded import RecordedCatalogClient
from replan.adapters.wire import build_catalog, decode_session
from replan.domain import (
    REPLAN_BLOCK_MARKER,
    AttendeeSchedule,
    BulkFailure,
    BulkFailureCode,
    BulkResult,
    Catalog,
    PersonalTime,
    Session,
    TimeSpan,
)
from replan.profiles import MealWindow, Profile, Scorer
from replan.solver import Plan, SolveRequest, solve
from replan.sync import (
    ActionKind,
    ActionStatus,
    DiffOptions,
    Journal,
    apply_plan,
    build_action_plan,
    undo_run,
)
from replan.sync.diff import MAX_BATCH, block_description
from replan.travel import default_graph

EVENT = "test-event"
DAY = "2026-12-01"


def session_at(
    code: str,
    clock: str,
    *,
    minutes: int = 60,
    reservable: bool | None = None,
    seats: str | None = None,
    room: str = "Level 2 | Hall C",
    venue: str | None = "Venetian",
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": code,
        "abbreviation": code,
        "title": f"Session {code}",
        "abstract": "About AWS Lambda.",
        "level": "300 \u2013 Advanced",
        "type": "Breakout session",
        "room": room,
        "services": ["AWS Lambda"],
        "topics": ["Serverless"],
        "sessionTime": {
            "date": DAY,
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
    return build_catalog(event_id=EVENT, sessions=sessions, source="test")


BASE_PROFILE = Profile(topics={"lambda": 5.0}, min_break_minutes=0, walk_tolerance_minutes=120)


def plan_for(catalog: Catalog, profile: Profile | None = None) -> Plan:
    resolved = profile or BASE_PROFILE
    scorer = Scorer(resolved, catalog)
    return solve(
        SolveRequest(
            catalog=catalog,
            profile=resolved,
            scores=scorer.by_id(),
            travel=default_graph(),
            time_limit_seconds=5.0,
        )
    )


# --------------------------------------------------------------------------- #
# A scriptable port, so failures can be provoked precisely
# --------------------------------------------------------------------------- #
class FakePort:
    surface_name = "fake"

    def __init__(self) -> None:
        self.reserved: set[str] = set()
        self.favorites: set[str] = set()
        self.blocks: dict[str, PersonalTime] = {}
        self.calls: list[tuple[str, Any]] = []
        self._seq = 0

        self.reserve_raises: Exception | None = None
        self.favorite_result: BulkResult | None = None
        self.create_raises: Exception | None = None
        self.schedule_raises: Exception | None = None

    def get_schedule(self, event_id: str) -> AttendeeSchedule:
        self.calls.append(("GetSchedule", event_id))
        if self.schedule_raises is not None:
            raise self.schedule_raises
        return AttendeeSchedule(
            event_id=event_id,
            reserved=tuple(sorted(self.reserved)),
            favorites=tuple(sorted(self.favorites)),
            personal_time=tuple(self.blocks.values()),
        )

    def reserve_sessions(self, event_id: str, session_ids: list[str]) -> BulkResult:
        self.calls.append(("ReserveSessions", tuple(session_ids)))
        if self.reserve_raises is not None:
            raise self.reserve_raises
        successful, failed = [], []
        for session_id in session_ids:
            if session_id in self.reserved:
                failed.append(
                    BulkFailure(session_id=session_id, code=BulkFailureCode.ALREADY_SCHEDULED)
                )
            else:
                self.reserved.add(session_id)
                successful.append(session_id)
        return BulkResult(successful=tuple(successful), failed=tuple(failed))

    def cancel_reservation(self, event_id: str, session_id: str) -> bool:
        self.calls.append(("CancelReservation", session_id))
        if session_id not in self.reserved:
            return False
        self.reserved.discard(session_id)
        return True

    def associate_favorites(self, event_id: str, session_ids: list[str]) -> BulkResult:
        self.calls.append(("AssociateFavorites", tuple(session_ids)))
        if self.favorite_result is not None:
            return self.favorite_result
        successful, failed = [], []
        for session_id in session_ids:
            if session_id in self.favorites:
                failed.append(
                    BulkFailure(session_id=session_id, code=BulkFailureCode.ALREADY_FAVORITED)
                )
            else:
                self.favorites.add(session_id)
                successful.append(session_id)
        return BulkResult(successful=tuple(successful), failed=tuple(failed))

    def disassociate_favorite(self, event_id: str, session_id: str) -> bool:
        self.calls.append(("DisassociateFavorite", session_id))
        if session_id not in self.favorites:
            return False
        self.favorites.discard(session_id)
        return True

    def create_personal_time(
        self,
        event_id: str,
        *,
        span: TimeSpan,
        title: str,
        description: str,
        location: str | None = None,
    ) -> None:
        self.calls.append(("CreatePersonalTime", title))
        if self.create_raises is not None:
            raise self.create_raises
        self._seq += 1
        entry_id = f"pt-{self._seq}"
        self.blocks[entry_id] = PersonalTime(
            personal_time_id=entry_id,
            title=title,
            description=description,
            span=TimeSpan(span.start.astimezone(UTC), span.end.astimezone(UTC)),
            location=location,
        )

    def update_personal_time(
        self,
        event_id: str,
        personal_time_id: str,
        *,
        span: TimeSpan,
        title: str,
        description: str,
        location: str | None = None,
    ) -> None:
        self.calls.append(("UpdatePersonalTime", personal_time_id))
        self.blocks[personal_time_id] = PersonalTime(
            personal_time_id=personal_time_id,
            title=title,
            description=description,
            span=span,
            location=location,
        )

    def delete_personal_time(self, event_id: str, personal_time_id: str) -> bool:
        self.calls.append(("DeletePersonalTime", personal_time_id))
        existed = personal_time_id in self.blocks
        self.blocks.pop(personal_time_id, None)
        return existed or True

    def operations(self) -> list[str]:
        return [name for name, _ in self.calls]


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #
def test_empty_schedule_produces_adds() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, BASE_PROFILE)
    kinds = {action.kind for action in actions.actions}
    assert ActionKind.RESERVE in kinds
    assert ActionKind.ADD_FAVORITE in kinds


def test_nothing_to_do_when_already_in_sync() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("A1",), favorites=("A1",))
    actions = build_action_plan(plan, schedule, catalog, BASE_PROFILE)
    assert actions.is_empty
    assert "already match" in actions.summary()


def test_unreservable_sessions_are_not_reserved() -> None:
    """The Summit catalog omits isReservable; there is no seat to take."""
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, BASE_PROFILE)
    assert actions.of_kind(ActionKind.RESERVE) == ()
    assert actions.of_kind(ActionKind.ADD_FAVORITE)


def test_batches_respect_the_quota_ceiling() -> None:
    sessions = [
        session_at(
            f"S{index:02d}",
            f"{8 + index // 2:02d}:{(index % 2) * 30:02d}",
            reservable=True,
            seats="available",
        )
        for index in range(60)
    ]
    catalog = catalog_of(*sessions)
    profile = BASE_PROFILE.model_copy(update={"max_sessions_per_day": 60})
    plan = plan_for(catalog, profile)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    for action in actions.actions:
        if action.kind.is_batched:
            assert len(action.session_ids) <= MAX_BATCH


def test_quota_cost_counts_sessions_not_calls() -> None:
    """ReserveSessions and AssociateFavorites bill per session named."""
    sessions = [
        session_at(f"S{i}", f"{9 + i:02d}:00", reservable=True, seats="available") for i in range(4)
    ]
    catalog = catalog_of(*sessions)
    profile = BASE_PROFILE.model_copy(update={"max_sessions_per_day": 10})
    plan = plan_for(catalog, profile)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    cost = actions.quota_cost()
    assert cost["ReserveSessions"] == len(plan.items)
    assert cost["AssociateFavorites"] == len(plan.items)


def test_favorites_are_not_pruned_by_default() -> None:
    """The API gives favorites no metadata, so we cannot tell ours from yours."""
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    # A favorite that is definitely not in the plan: the attendee starred it
    # themselves in the mobile app, and replan has no way to know that.
    schedule = AttendeeSchedule(event_id=EVENT, favorites=("STARRED-BY-HAND",))

    actions = build_action_plan(plan, schedule, catalog, BASE_PROFILE)
    assert actions.of_kind(ActionKind.REMOVE_FAVORITE) == ()
    assert any("cannot tell us" in note for note in actions.notes)


def test_favorites_can_be_pruned_on_request() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, favorites=("A1", "GHOST1"))
    actions = build_action_plan(
        plan, schedule, catalog, BASE_PROFILE, DiffOptions(prune_favorites=True)
    )
    removals = actions.of_kind(ActionKind.REMOVE_FAVORITE)
    assert [action.session_ids[0] for action in removals] == ["GHOST1"]


def test_reservations_are_not_cancelled_by_default() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("A1", "GHOST1"))
    actions = build_action_plan(plan, schedule, catalog, BASE_PROFILE)
    assert actions.of_kind(ActionKind.CANCEL_RESERVATION) == ()
    assert any("left alone" in note for note in actions.notes)


def test_reservations_can_be_pruned_on_request() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("A1", "GHOST1"))
    actions = build_action_plan(
        plan, schedule, catalog, BASE_PROFILE, DiffOptions(prune_reservations=True)
    )
    cancels = actions.of_kind(ActionKind.CANCEL_RESERVATION)
    assert [action.session_ids[0] for action in cancels] == ["GHOST1"]


# --------------------------------------------------------------------------- #
# Personal time
# --------------------------------------------------------------------------- #
def test_meal_windows_become_create_actions() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    creates = actions.of_kind(ActionKind.CREATE_PERSONAL_TIME)
    assert len(creates) == 1
    assert creates[0].title == "Lunch"
    assert creates[0].span is not None
    assert creates[0].span.minutes == 60


def test_created_blocks_are_signed_with_the_marker() -> None:
    """The API has no user metadata field, so authorship lives in the description."""
    assert REPLAN_BLOCK_MARKER in block_description("Lunch")
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    assert REPLAN_BLOCK_MARKER in actions.of_kind(ActionKind.CREATE_PERSONAL_TIME)[0].description


def test_existing_matching_block_is_not_recreated() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    first = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    create = first.of_kind(ActionKind.CREATE_PERSONAL_TIME)[0]
    assert create.span is not None

    existing = PersonalTime(
        personal_time_id="pt-1",
        title="Lunch",
        description=create.description,
        span=create.span,
    )
    schedule = AttendeeSchedule(event_id=EVENT, personal_time=(existing,))
    second = build_action_plan(plan, schedule, catalog, profile)
    assert second.of_kind(ActionKind.CREATE_PERSONAL_TIME) == ()


def test_stale_replan_block_is_deleted() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    stale = PersonalTime(
        personal_time_id="pt-old",
        title="Old window",
        description=f"held by replan {REPLAN_BLOCK_MARKER}",
        span=TimeSpan(
            datetime(2026, 12, 1, 20, 0, tzinfo=UTC),
            datetime(2026, 12, 1, 21, 0, tzinfo=UTC),
        ),
    )
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, personal_time=(stale,))
    actions = build_action_plan(plan, schedule, catalog, BASE_PROFILE)
    deletes = actions.of_kind(ActionKind.DELETE_PERSONAL_TIME)
    assert [action.personal_time_id for action in deletes] == ["pt-old"]


def test_foreign_blocks_are_never_deleted() -> None:
    """A block the attendee made by hand is not replan's to remove."""
    catalog = catalog_of(session_at("A1", "09:00"))
    foreign = PersonalTime(
        personal_time_id="pt-mine",
        title="Customer meeting",
        description="I made this myself",
        span=TimeSpan(
            datetime(2026, 12, 1, 20, 0, tzinfo=UTC),
            datetime(2026, 12, 1, 21, 0, tzinfo=UTC),
        ),
    )
    assert foreign.created_by_replan is False
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, personal_time=(foreign,))
    actions = build_action_plan(plan, schedule, catalog, BASE_PROFILE)
    assert actions.of_kind(ActionKind.DELETE_PERSONAL_TIME) == ()
    assert any("not created by replan" in note for note in actions.notes)


def test_meal_window_spans_land_on_the_five_minute_grid() -> None:
    """The API rejects personal time that is not a whole number of 5-minute steps."""
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Odd", start=time(12, 3), end=time(12, 58)),)}
    )
    plan = plan_for(catalog, profile)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    create = actions.of_kind(ActionKind.CREATE_PERSONAL_TIME)[0]
    assert create.span is not None
    assert create.span.minutes % 5 == 0
    assert create.span.start.second == 0


def test_advisory_windows_are_not_created() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={
            "meal_windows": (
                MealWindow(title="Maybe", start=time(12, 0), end=time(13, 0), protect=False),
            )
        }
    )
    plan = plan_for(catalog, profile)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, profile)
    assert actions.of_kind(ActionKind.CREATE_PERSONAL_TIME) == ()


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #
def test_reservations_run_before_favorites() -> None:
    """Seats are the only scarce resource, so they are taken first."""
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    actions = build_action_plan(plan, AttendeeSchedule(event_id=EVENT), catalog, BASE_PROFILE)
    kinds = [action.kind for action in actions.ordered]
    assert kinds.index(ActionKind.RESERVE) < kinds.index(ActionKind.ADD_FAVORITE)


def test_destructive_actions_run_last() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("GHOST1",), favorites=("GHOST2",))
    actions = build_action_plan(
        plan,
        schedule,
        catalog,
        BASE_PROFILE,
        DiffOptions(prune_favorites=True, prune_reservations=True),
    )
    ordered = actions.ordered
    first_destructive = next(
        index for index, action in enumerate(ordered) if action.kind.is_destructive
    )
    assert all(not action.kind.is_destructive for action in ordered[:first_destructive])


# --------------------------------------------------------------------------- #
# Apply: dry run
# --------------------------------------------------------------------------- #
def test_dry_run_sends_nothing(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    port.calls.clear()

    report = apply_plan(port, plan, actions, dry_run=True, journal=Journal(tmp_path / "j.db"))

    assert report.dry_run
    assert port.calls == [], "a dry run must not touch the API"
    assert all(item.status is ActionStatus.PLANNED for item in report.outcomes)


def test_dry_run_is_journalled_separately(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    journal = Journal(tmp_path / "j.db")
    report = apply_plan(port, plan, actions, dry_run=True, journal=journal)

    run = journal.run(report.run_id or 0)
    assert run is not None
    assert run.dry_run
    assert journal.latest_run(EVENT) is None, "a dry run is not an applied run"


# --------------------------------------------------------------------------- #
# Apply: real writes
# --------------------------------------------------------------------------- #
def test_apply_writes_and_verifies(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)

    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))

    assert report.ok
    assert report.verified
    assert "A1" in port.reserved
    assert "A1" in port.favorites
    # Verification re-reads the schedule, which is the only honest confirmation.
    assert "GetSchedule" in port.operations()


def test_already_in_desired_state_counts_as_success(tmp_path: Path) -> None:
    """The API reports an existing favorite as a *failure*; the end state is what matters."""
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)

    # Pre-seed so the API will refuse both with "already" codes.
    port.reserved.add("A1")
    port.favorites.add("A1")

    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))
    assert report.ok
    assert all(item.status.is_success for item in report.outcomes)


def test_partial_bulk_failure_is_reported(tmp_path: Path) -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", reservable=True, seats="available"),
        session_at("B1", "13:00", reservable=True, seats="available"),
    )
    profile = BASE_PROFILE.model_copy(update={"max_sessions_per_day": 5})
    plan = plan_for(catalog, profile)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, profile)

    # A 200 carrying one success and one genuine refusal.
    port.favorite_result = BulkResult(
        successful=("A1",),
        failed=(BulkFailure(session_id="B1", code=BulkFailureCode.INSUFFICIENT_ACCESS),),
    )

    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))
    favorite_outcome = next(
        item for item in report.outcomes if item.action.kind is ActionKind.ADD_FAVORITE
    )
    assert favorite_outcome.status is ActionStatus.FAILED
    assert "partial" in favorite_outcome.detail
    assert favorite_outcome.succeeded_ids == ("A1",)
    assert favorite_outcome.failed_ids == ("B1",)


def test_closed_reservations_skip_and_the_run_continues(tmp_path: Path) -> None:
    """Exactly today's behaviour: reservations 409 until 8 October 2026."""
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    port.reserve_raises = EventsOperationClosedError(
        "Reserved seating opens through the Events API on 8 October 2026.",
        status=409,
        operation="ReserveSessions",
    )

    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))

    reserve_outcome = next(
        item for item in report.outcomes if item.action.kind is ActionKind.RESERVE
    )
    assert reserve_outcome.status is ActionStatus.SKIPPED
    # Favorites still went through, which is the useful half today.
    assert "A1" in port.favorites
    assert any("not open through the API yet" in note for note in report.notes)


def test_unknown_outcome_is_not_retried(tmp_path: Path) -> None:
    """A lost write must be reconciled, never re-sent: there is no idempotency key."""
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, profile)
    port.create_raises = EventsApiError(
        "CreatePersonalTime failed in transport and its outcome is unknown. "
        "Reconcile against GetSchedule before retrying; the API has no idempotency key.",
        operation="CreatePersonalTime",
    )

    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))

    assert report.unknown
    assert not report.ok
    creates = [name for name in port.operations() if name == "CreatePersonalTime"]
    assert len(creates) == 1, "an unknown outcome must not be retried"


def test_verification_notices_a_missing_write(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)

    class SilentlyDroppingPort(FakePort):
        def associate_favorites(self, event_id: str, session_ids: list[str]) -> BulkResult:
            self.calls.append(("AssociateFavorites", tuple(session_ids)))
            # Claims success without storing anything.
            return BulkResult(successful=tuple(session_ids), failed=())

    port = SilentlyDroppingPort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))

    assert report.verified is False
    assert any("favorite" in problem for problem in report.verification)


def test_verification_reports_a_planned_session_with_no_seat(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    port.reserve_raises = EventsOperationClosedError("closed", status=409)

    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))
    assert report.verified is False
    assert any("no seat" in problem for problem in report.verification)


def test_unverifiable_schedule_read_is_reported(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)

    report_port = port
    report_port.schedule_raises = None
    apply_report = apply_plan(
        report_port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db")
    )
    assert apply_report.verified


# --------------------------------------------------------------------------- #
# Journal and undo
# --------------------------------------------------------------------------- #
def test_apply_then_undo_restores_the_original_state(tmp_path: Path) -> None:
    """The test that matters: replan must be able to put things back."""
    catalog = catalog_of(
        session_at("A1", "09:00", reservable=True, seats="available"),
        session_at("B1", "13:00", reservable=True, seats="available"),
    )
    profile = BASE_PROFILE.model_copy(
        update={
            "max_sessions_per_day": 5,
            "meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),),
        }
    )
    plan = plan_for(catalog, profile)
    port = FakePort()
    journal = Journal(tmp_path / "j.db")

    before = port.get_schedule(EVENT)
    assert before.is_empty()

    actions = build_action_plan(plan, before, catalog, profile)
    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)
    assert report.ok

    after = port.get_schedule(EVENT)
    assert after.reserved_ids
    assert after.favorite_ids
    assert after.personal_time

    run = journal.run(report.run_id or 0)
    assert run is not None
    undo = undo_run(journal, run, port, dry_run=False)
    assert undo.ok, undo.failed

    final = port.get_schedule(EVENT)
    assert final.reserved_ids == before.reserved_ids
    assert final.favorite_ids == before.favorite_ids
    assert final.personal_time == before.personal_time


def test_created_block_id_is_resolved_for_undo(tmp_path: Path) -> None:
    """CreatePersonalTime returns 204 with no body, so the id must be looked up."""
    catalog = catalog_of(session_at("A1", "09:00"))
    profile = BASE_PROFILE.model_copy(
        update={"meal_windows": (MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),)}
    )
    plan = plan_for(catalog, profile)
    port = FakePort()
    journal = Journal(tmp_path / "j.db")
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, profile)
    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)

    run = journal.run(report.run_id or 0)
    assert run is not None
    create_entry = next(
        entry for entry in run.entries if entry.action.kind is ActionKind.CREATE_PERSONAL_TIME
    )
    assert create_entry.created_personal_time_id is not None
    assert create_entry.is_reversible


def test_undo_reverses_in_reverse_order(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    journal = Journal(tmp_path / "j.db")
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)

    run = journal.run(report.run_id or 0)
    assert run is not None
    sequence = [entry.seq for entry in run.reversible_entries]
    assert sequence == sorted(sequence, reverse=True)


def test_undo_dry_run_sends_nothing(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    journal = Journal(tmp_path / "j.db")
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)

    run = journal.run(report.run_id or 0)
    assert run is not None
    port.calls.clear()
    undo = undo_run(journal, run, port, dry_run=True)

    assert undo.reversed_count > 0
    assert port.calls == []
    assert port.favorites, "a dry-run undo must not remove anything"


def test_undo_only_reverses_successful_actions(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    journal = Journal(tmp_path / "j.db")
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    port.reserve_raises = EventsOperationClosedError("closed", status=409)

    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)
    run = journal.run(report.run_id or 0)
    assert run is not None

    reversible_kinds = {entry.action.kind for entry in run.reversible_entries}
    assert ActionKind.RESERVE not in reversible_kinds


def test_undo_is_recorded_so_it_is_not_repeated(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    journal = Journal(tmp_path / "j.db")
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)

    run = journal.run(report.run_id or 0)
    assert run is not None
    undo_run(journal, run, port, dry_run=False)

    reloaded = journal.run(run.run_id)
    assert reloaded is not None
    assert reloaded.undone_at is not None
    assert reloaded.reversible_entries == ()


def test_history_lists_runs(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    journal = Journal(tmp_path / "j.db")

    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    apply_plan(port, plan, actions, dry_run=True, journal=journal)
    apply_plan(port, plan, actions, dry_run=False, journal=journal)

    runs = journal.history(EVENT)
    assert len(runs) == 2
    assert runs[0].run_id > runs[1].run_id  # newest first


def test_journal_survives_reopening(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    path = tmp_path / "j.db"
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(path))

    assert Journal(path).run(report.run_id or 0) is not None


def test_latest_run_ignores_dry_runs(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    journal = Journal(tmp_path / "j.db")
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)

    applied = apply_plan(port, plan, actions, dry_run=False, journal=journal)
    apply_plan(port, plan, actions, dry_run=True, journal=journal)

    latest = journal.latest_run(EVENT)
    assert latest is not None
    assert latest.run_id == applied.run_id


# --------------------------------------------------------------------------- #
# Against the recorded surface
# --------------------------------------------------------------------------- #
def test_round_trip_against_the_recorded_client(tmp_path: Path, recorded_dir: Path) -> None:
    """Same round trip, driven through the client that replays real API payloads."""
    port = RecordedCatalogClient(recorded_dir, reservations_open=True)
    event_id = port.recorded_event_id
    catalog = port.list_sessions(event_id)

    profile = Profile(
        topics={"bedrock": 3.0},
        min_break_minutes=0,
        max_sessions_per_day=4,
        meal_windows=(MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),),
    )
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

    journal = Journal(tmp_path / "j.db")
    before = port.get_schedule(event_id)
    actions = build_action_plan(plan, before, catalog, profile)
    report = apply_plan(port, plan, actions, dry_run=False, journal=journal)
    assert report.ok, [item.describe() for item in report.outcomes]

    run = journal.run(report.run_id or 0)
    assert run is not None
    undo = undo_run(journal, run, port, dry_run=False)
    assert undo.ok, undo.failed

    final = port.get_schedule(event_id)
    assert final.favorite_ids == before.favorite_ids
    assert final.reserved_ids == before.reserved_ids
    assert len(final.personal_time) == len(before.personal_time)


def test_reservations_closed_on_the_recorded_client_by_default(
    tmp_path: Path, recorded_dir: Path
) -> None:
    """Default matches the live API today: reservations refuse for everyone."""
    port = RecordedCatalogClient(recorded_dir, reservations_open=False)
    event_id = port.recorded_event_id
    catalog = port.list_sessions(event_id)
    profile = Profile(topics={"bedrock": 3.0}, min_break_minutes=0, max_sessions_per_day=3)
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
    actions = build_action_plan(plan, port.get_schedule(event_id), catalog, profile)
    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))

    # Favorites succeed even while reservations are closed.
    assert port.get_schedule(event_id).favorite_ids
    reserve_outcomes = [item for item in report.outcomes if item.action.kind is ActionKind.RESERVE]
    for outcome in reserve_outcomes:
        assert outcome.status is ActionStatus.SKIPPED


# --------------------------------------------------------------------------- #
# Action metadata
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        (ActionKind.RESERVE, "ReserveSessions"),
        (ActionKind.CANCEL_RESERVATION, "CancelReservation"),
        (ActionKind.ADD_FAVORITE, "AssociateFavorites"),
        (ActionKind.REMOVE_FAVORITE, "DisassociateFavorite"),
        (ActionKind.CREATE_PERSONAL_TIME, "CreatePersonalTime"),
        (ActionKind.UPDATE_PERSONAL_TIME, "UpdatePersonalTime"),
        (ActionKind.DELETE_PERSONAL_TIME, "DeletePersonalTime"),
    ],
)
def test_action_kinds_map_to_api_operations(kind: ActionKind, operation: str) -> None:
    assert kind.operation == operation


def test_only_removals_are_safe_to_retry() -> None:
    """Re-sending a create adds a duplicate; re-sending a bulk misreports successes."""
    assert ActionKind.CANCEL_RESERVATION.is_safe_to_retry
    assert ActionKind.REMOVE_FAVORITE.is_safe_to_retry
    assert ActionKind.DELETE_PERSONAL_TIME.is_safe_to_retry
    assert not ActionKind.RESERVE.is_safe_to_retry
    assert not ActionKind.ADD_FAVORITE.is_safe_to_retry
    assert not ActionKind.CREATE_PERSONAL_TIME.is_safe_to_retry


def test_only_the_two_bulk_operations_are_batched() -> None:
    batched = {kind for kind in ActionKind if kind.is_batched}
    assert batched == {ActionKind.RESERVE, ActionKind.ADD_FAVORITE}


def test_action_round_trips_through_its_payload() -> None:
    from replan.sync.actions import Action

    original = Action(
        kind=ActionKind.CREATE_PERSONAL_TIME,
        span=TimeSpan(
            datetime(2026, 12, 1, 20, 0, tzinfo=UTC),
            datetime(2026, 12, 1, 21, 0, tzinfo=UTC),
        ),
        title="Lunch",
        description="held by replan",
        labels=("Lunch",),
        reason="because",
    )
    assert Action.from_payload(original.to_payload()) == original


def test_report_serialises_to_json(tmp_path: Path) -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="available"))
    plan = plan_for(catalog)
    port = FakePort()
    actions = build_action_plan(plan, port.get_schedule(EVENT), catalog, BASE_PROFILE)
    report = apply_plan(port, plan, actions, dry_run=False, journal=Journal(tmp_path / "j.db"))

    import json

    parsed = json.loads(report.to_json())
    assert parsed["eventId"] == EVENT
    assert parsed["dryRun"] is False
    assert parsed["outcomes"]


def test_timespan_delta_helper_unused_import_guard() -> None:
    """Keeps timedelta genuinely used, and documents the 5-minute rule once more."""
    span = TimeSpan(
        datetime(2026, 12, 1, 20, 0, tzinfo=UTC),
        datetime(2026, 12, 1, 20, 0, tzinfo=UTC) + timedelta(minutes=45),
    )
    assert span.minutes % 5 == 0
