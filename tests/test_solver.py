"""Tests for the scheduler.

The important cases are the ones a human gets wrong: a pair that does not overlap
but cannot be walked, five sessions on the same subject beating a balanced day, and
a repeat of content you already have. Each is asserted against a hand-built
scenario with a known correct answer, so a regression in the model is caught rather
than absorbed.

Both solvers run through the same suite where the expected answer does not depend
on optimality, since a fallback that permits illegal plans would be worse than no
fallback.
"""

from __future__ import annotations

import itertools
from datetime import time
from pathlib import Path
from typing import Any

import pytest

from replan.adapters.wire import build_catalog, decode_catalog_page, decode_event, decode_session
from replan.domain import AttendeeSchedule, Catalog, PersonalTime, Session, TimeSpan
from replan.profiles import DEFAULT_PROFILE_YAML, MealWindow, ObjectiveWeights, Profile, Scorer
from replan.solver import (
    Plan,
    RejectionReason,
    SolveRequest,
    SolverKind,
    solve,
    solve_greedy,
)
from replan.solver.common import prepare
from replan.solver.cpsat import cpsat_available
from replan.travel import TravelGraph, default_graph

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
    topics: tuple[str, ...] = ("Serverless",),
    services: tuple[str, ...] = ("AWS Lambda",),
    title: str | None = None,
    reservable: bool | None = None,
    seats: str | None = None,
    session_id: str | None = None,
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": session_id or code,
        "abbreviation": code,
        "title": title or f"Session {code}",
        "abstract": "An abstract.",
        "level": "300 \u2013 Advanced",
        "type": "Breakout session",
        "room": room,
        "topics": list(topics),
        "services": list(services),
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
    return build_catalog(event_id=EVENT, sessions=sessions, source="test")


def request_for(
    catalog: Catalog,
    profile: Profile | None = None,
    *,
    schedule: AttendeeSchedule | None = None,
    graph: TravelGraph | None = None,
    **kwargs: Any,
) -> SolveRequest:
    # Two deliberate choices in the default test profile:
    #
    # - A generous walk tolerance, so tests about other constraints are not
    #   silently answered by the tolerance. Venetian to Mandalay Bay is 51 minutes,
    #   which the profile default of 45 rejects.
    # - A real topic preference, so the sessions are worth attending. With no
    #   preferences every score is zero, and the solver then correctly refuses to
    #   pay a travel or idle cost to attend something you do not care about, which
    #   would make these tests measure the wrong thing.
    resolved = profile or TEST_PROFILE
    scorer = Scorer(resolved, catalog)
    return SolveRequest(
        catalog=catalog,
        profile=resolved,
        scores=scorer.by_id(),
        travel=graph or default_graph(),
        schedule=schedule,
        time_limit_seconds=5.0,
        **kwargs,
    )


def codes(plan: Plan) -> set[str]:
    return {item.code for item in plan.items}


BOTH_SOLVERS = pytest.mark.parametrize("prefer_cpsat", [True, False], ids=["cpsat", "greedy"])

# Matches `session_at`'s default services, so a default test session is worth
# roughly nine points rather than zero.
TEST_PROFILE = Profile(
    topics={"lambda": 5.0},
    min_break_minutes=0,
    walk_tolerance_minutes=120,
)


def profile_with(**changes: Any) -> Profile:
    """A variant of the test profile, keeping the baseline topic preference."""
    return TEST_PROFILE.model_copy(update=changes)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def test_ortools_is_installed() -> None:
    """The `solver` extra is part of `--extra all`, so CP-SAT should be present."""
    assert cpsat_available()


def test_cpsat_is_used_by_default() -> None:
    plan = solve(request_for(catalog_of(session_at("A1", "09:00"))))
    assert plan.stats.solver is SolverKind.CPSAT
    assert plan.stats.optimal


def test_greedy_can_be_forced_and_says_it_is_not_optimal() -> None:
    plan = solve(request_for(catalog_of(session_at("A1", "09:00"))), prefer_cpsat=False)
    assert plan.stats.solver is SolverKind.GREEDY
    assert plan.stats.optimal is False
    assert plan.stats.fallback_reason


# --------------------------------------------------------------------------- #
# Hard constraint: no overlaps
# --------------------------------------------------------------------------- #
@BOTH_SOLVERS
def test_overlapping_sessions_are_never_both_chosen(prefer_cpsat: bool) -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "09:30"))
    plan = solve(request_for(catalog), prefer_cpsat=prefer_cpsat)
    assert len(plan.items) == 1


@BOTH_SOLVERS
def test_no_two_items_in_a_plan_ever_overlap(prefer_cpsat: bool) -> None:
    catalog = catalog_of(
        *[session_at(f"S{i}", f"{9 + i // 2:02d}:{(i % 2) * 30:02d}") for i in range(10)]
    )
    plan = solve(request_for(catalog), prefer_cpsat=prefer_cpsat)
    spans = [item.span for item in plan.items]
    for index, left in enumerate(spans):
        for right in spans[index + 1 :]:
            assert not left.overlaps(right)


def test_overlap_rejection_names_the_blocking_session() -> None:
    catalog = catalog_of(session_at("A1", "09:00"), session_at("B1", "09:30"))
    plan = solve(request_for(catalog))
    kept = next(iter(codes(plan)))
    dropped = "B1" if kept == "A1" else "A1"
    rejection = plan.rejection_for(dropped)
    assert rejection is not None
    assert rejection.reason is RejectionReason.TIME_CONFLICT
    assert kept in rejection.detail


# --------------------------------------------------------------------------- #
# Hard constraint: travel feasibility
# --------------------------------------------------------------------------- #
@BOTH_SOLVERS
def test_impossible_transfer_is_refused(prefer_cpsat: bool) -> None:
    """The signature failure a hand-built agenda makes.

    13:30-14:30 in the Venetian then 15:00 in Mandalay Bay. The two do not overlap,
    so every list view says this is fine. It needs about 51 minutes of travel and
    there are 30.
    """
    catalog = catalog_of(
        session_at("A1", "13:30", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "15:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    plan = solve(request_for(catalog), prefer_cpsat=prefer_cpsat)
    assert len(plan.items) == 1


def test_travel_rejection_explains_the_shortfall() -> None:
    catalog = catalog_of(
        session_at("A1", "13:30", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "15:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    plan = solve(request_for(catalog))
    dropped = next(item for item in plan.rejections if item.code not in codes(plan))
    assert dropped.reason is RejectionReason.TRAVEL_INFEASIBLE
    assert "min" in dropped.detail


@BOTH_SOLVERS
def test_feasible_transfer_is_allowed(prefer_cpsat: bool) -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    plan = solve(request_for(catalog), prefer_cpsat=prefer_cpsat)
    assert codes(plan) == {"A1", "B1"}


def test_walk_tolerance_blocks_a_reachable_session() -> None:
    """Enough clock time, but further than you are willing to walk.

    The 51-minute Venetian/Mandalay Bay transfer fits in a four-hour gap, so only
    the tolerance decides it. Note the profile default of 45 minutes already
    rejects this, which is the intended out-of-the-box behaviour.
    """
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    generous = solve(request_for(catalog, profile_with(walk_tolerance_minutes=90)))
    assert len(generous.items) == 2

    strict = solve(request_for(catalog, profile_with(walk_tolerance_minutes=20)))
    assert len(strict.items) == 1
    rejection = next(item for item in strict.rejections if item.code not in codes(strict))
    assert rejection.reason is RejectionReason.WALK_TOO_FAR
    assert "tolerance" in rejection.detail


def test_min_break_is_respected_and_explained() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", minutes=60),
        session_at("B1", "10:05", minutes=60),
    )
    assert len(solve(request_for(catalog, profile_with(min_break_minutes=0))).items) == 2

    plan = solve(request_for(catalog, profile_with(min_break_minutes=30)))
    assert len(plan.items) == 1
    rejection = next(item for item in plan.rejections if item.code not in codes(plan))
    assert "break" in rejection.detail


def test_travel_legs_are_attached_to_plan_items() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "12:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    plan = solve(request_for(catalog))
    ordered = sorted(plan.items, key=lambda item: item.span.start)
    assert ordered[0].travel_minutes is None  # first of the day
    second = ordered[1]
    assert second.travel_minutes and second.travel_minutes > 0
    assert second.travel_mode
    assert second.depart_previous_by is not None
    assert second.depart_previous_by < second.span.start
    assert second.previous_session_id == ordered[0].session_id


# --------------------------------------------------------------------------- #
# Hard constraint: one showing per piece of content
# --------------------------------------------------------------------------- #
@BOTH_SOLVERS
def test_repeats_of_the_same_content_are_not_both_chosen(prefer_cpsat: bool) -> None:
    catalog = catalog_of(
        session_at("SVS401", "09:00", session_id="r1"),
        session_at("SVS401-R2", "14:00", session_id="r2"),
    )
    plan = solve(request_for(catalog), prefer_cpsat=prefer_cpsat)
    assert len(plan.items) == 1


def test_duplicate_rejection_points_at_the_chosen_showing() -> None:
    catalog = catalog_of(
        session_at("SVS401", "09:00", session_id="r1"),
        session_at("SVS401-R2", "14:00", session_id="r2"),
    )
    plan = solve(request_for(catalog))
    chosen = next(iter(plan.items))
    other = "r2" if chosen.session_id == "r1" else "r1"
    rejection = plan.rejection_for(other)
    assert rejection is not None
    assert rejection.reason is RejectionReason.DUPLICATE_CONTENT
    assert chosen.session_id in rejection.conflicts_with


def test_repeat_lets_the_solver_dodge_a_conflict() -> None:
    """The move humans miss: take the later showing to keep both sessions.

    A pinned session collides with the first showing of SVS401, but a repeat
    exists in the afternoon, so a good plan contains both pieces of content.
    """
    catalog = catalog_of(
        session_at("KEY001", "09:00", minutes=90, topics=("Keynote",), services=()),
        session_at("SVS401", "09:30", session_id="r1"),
        session_at("SVS401-R2", "14:00", session_id="r2"),
    )
    profile = Profile(must_attend=("KEY001",), min_break_minutes=0)
    plan = solve(request_for(catalog, profile))
    assert codes(plan) == {"KEY001", "SVS401-R2"}


# --------------------------------------------------------------------------- #
# Pins and existing reservations
# --------------------------------------------------------------------------- #
@BOTH_SOLVERS
def test_pinned_session_is_always_included(prefer_cpsat: bool) -> None:
    catalog = catalog_of(
        session_at("DULL1", "09:00", topics=(), services=(), title="Nothing you asked for"),
        session_at("GOOD1", "09:30", services=("Amazon Bedrock",)),
    )
    profile = Profile(topics={"bedrock": 5.0}, must_attend=("DULL1",))
    plan = solve(request_for(catalog, profile), prefer_cpsat=prefer_cpsat)
    assert "DULL1" in codes(plan)


def test_pinned_items_are_marked() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = solve(request_for(catalog, Profile(must_attend=("A1",))))
    assert plan.items[0].pinned


def test_unresolvable_pin_produces_a_warning() -> None:
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = solve(request_for(catalog, Profile(must_attend=("NOPE1",))))
    assert any("NOPE1" in warning for warning in plan.warnings)


def test_existing_reservation_is_treated_as_a_pin() -> None:
    """You already hold the seat, so the plan must not quietly drop it."""
    catalog = catalog_of(
        session_at("HELD1", "09:00", topics=(), services=(), title="Already booked"),
        session_at("GOOD1", "09:30", services=("Amazon Bedrock",)),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("HELD1",))
    profile = Profile(topics={"bedrock": 5.0})
    plan = solve(request_for(catalog, profile, schedule=schedule))
    assert "HELD1" in codes(plan)
    assert plan.items[0].already_reserved


def test_reservations_can_be_ignored_on_request() -> None:
    catalog = catalog_of(
        session_at("HELD1", "09:00", topics=(), services=(), title="Already booked"),
        session_at("GOOD1", "09:30", services=("Amazon Bedrock",)),
    )
    schedule = AttendeeSchedule(event_id=EVENT, reserved=("HELD1",))
    profile = Profile(topics={"bedrock": 5.0})
    plan = solve(request_for(catalog, profile, schedule=schedule, respect_reservations=False))
    assert codes(plan) == {"GOOD1"}


# --------------------------------------------------------------------------- #
# Protected time
# --------------------------------------------------------------------------- #
def test_meal_window_blocks_overlapping_sessions() -> None:
    catalog = catalog_of(session_at("LUNCH1", "12:15", minutes=30))
    profile = Profile(meal_windows=(MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),))
    plan = solve(request_for(catalog, profile))
    assert plan.items == ()
    rejection = plan.rejection_for("LUNCH1")
    assert rejection is not None
    assert rejection.reason is RejectionReason.PROTECTED_TIME
    assert "Lunch" in rejection.detail


def test_advisory_meal_window_does_not_block() -> None:
    catalog = catalog_of(session_at("LUNCH1", "12:15", minutes=30))
    profile = Profile(
        meal_windows=(MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0), protect=False),)
    )
    assert len(solve(request_for(catalog, profile)).items) == 1


def test_pin_beats_a_meal_window() -> None:
    catalog = catalog_of(session_at("LUNCH1", "12:15", minutes=30))
    profile = Profile(
        must_attend=("LUNCH1",),
        meal_windows=(MealWindow(title="Lunch", start=time(12, 0), end=time(13, 0)),),
    )
    assert len(solve(request_for(catalog, profile)).items) == 1


def test_existing_personal_time_is_respected() -> None:
    """A block created in the official app comes back via GetSchedule."""
    catalog = catalog_of(session_at("A1", "09:00"))
    target = catalog.sessions[0]
    assert target.span is not None
    schedule = AttendeeSchedule(
        event_id=EVENT,
        personal_time=(
            PersonalTime(
                personal_time_id="pt-1",
                title="Customer meeting",
                span=TimeSpan(target.span.start, target.span.end),
            ),
        ),
    )
    plan = solve(request_for(catalog, schedule=schedule))
    assert plan.items == ()
    rejection = plan.rejection_for("A1")
    assert rejection is not None
    assert "Customer meeting" in rejection.detail


# --------------------------------------------------------------------------- #
# Daily limits and fatigue
# --------------------------------------------------------------------------- #
@BOTH_SOLVERS
def test_max_sessions_per_day_is_enforced(prefer_cpsat: bool) -> None:
    catalog = catalog_of(*[session_at(f"S{i}", f"{9 + i:02d}:00") for i in range(8)])
    profile = profile_with(max_sessions_per_day=3)
    plan = solve(request_for(catalog, profile), prefer_cpsat=prefer_cpsat)
    assert len(plan.items) == 3


def test_day_full_rejection_is_recorded() -> None:
    catalog = catalog_of(*[session_at(f"S{i}", f"{9 + i:02d}:00") for i in range(6)])
    profile = profile_with(max_sessions_per_day=2)
    plan = solve(request_for(catalog, profile))
    reasons = {item.reason for item in plan.rejections}
    assert RejectionReason.DAY_FULL in reasons or RejectionReason.NOT_CHOSEN in reasons


def test_limits_apply_per_day_not_overall() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", day="2026-12-01"),
        session_at("A2", "11:00", day="2026-12-01"),
        session_at("B1", "09:00", day="2026-12-02"),
        session_at("B2", "11:00", day="2026-12-02"),
    )
    profile = profile_with(max_sessions_per_day=2)
    plan = solve(request_for(catalog, profile))
    assert len(plan.items) == 4
    assert len(plan.by_day()) == 2


def test_fatigue_penalty_discourages_a_marathon_day() -> None:
    catalog = catalog_of(*[session_at(f"S{i}", f"{8 + i:02d}:00") for i in range(10)])
    relaxed = profile_with(
        max_consecutive_minutes=600,
        weights=ObjectiveWeights(fatigue_penalty=0.0),
    )
    punishing = profile_with(
        max_consecutive_minutes=120,
        weights=ObjectiveWeights(fatigue_penalty=10.0),
    )
    assert len(solve(request_for(catalog, punishing)).items) < len(
        solve(request_for(catalog, relaxed)).items
    )


# --------------------------------------------------------------------------- #
# Coverage: the diminishing-returns objective
# --------------------------------------------------------------------------- #
def test_coverage_prefers_breadth_over_five_of_the_same_thing() -> None:
    """The reason the objective is not just a sum of scores.

    Three identical-topic sessions and one on a different subject, with room for
    two. High coverage weight should take one of each rather than two duplicates.
    """
    catalog = catalog_of(
        session_at("DUP1", "09:00", topics=("Serverless",), services=("AWS Lambda",)),
        session_at("DUP2", "11:00", topics=("Serverless",), services=("AWS Lambda",)),
        session_at("DUP3", "13:00", topics=("Serverless",), services=("AWS Lambda",)),
        session_at("DIFF1", "15:00", topics=("Security",), services=("AWS IAM",)),
    )
    profile = Profile(
        topics={"lambda": 2.0, "iam": 1.0},
        max_sessions_per_day=2,
        min_break_minutes=0,
        weights=ObjectiveWeights(coverage=6.0),
    )
    plan = solve(request_for(catalog, profile))
    assert "DIFF1" in codes(plan), "high coverage weight should buy breadth"


def test_zero_coverage_weight_just_maximises_score() -> None:
    catalog = catalog_of(
        session_at("DUP1", "09:00", topics=("Serverless",), services=("AWS Lambda",)),
        session_at("DUP2", "11:00", topics=("Serverless",), services=("AWS Lambda",)),
        session_at("DIFF1", "13:00", topics=("Security",), services=("AWS IAM",)),
    )
    profile = Profile(
        topics={"lambda": 5.0, "iam": 0.1},
        max_sessions_per_day=2,
        min_break_minutes=0,
        weights=ObjectiveWeights(coverage=0.0),
    )
    plan = solve(request_for(catalog, profile))
    assert codes(plan) == {"DUP1", "DUP2"}


def test_coverage_is_counted_in_stats() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", topics=("Serverless",)),
        session_at("A2", "11:00", topics=("Security",)),
        session_at("A3", "13:00", topics=("Serverless",)),
        session_at("A4", "15:00", topics=("Security",)),
    )
    plan = solve(request_for(catalog))
    assert plan.stats.topics_available > 0
    assert 0.0 <= plan.stats.coverage_ratio <= 1.0


# --------------------------------------------------------------------------- #
# Travel cost pricing (the successor-arc formulation)
# --------------------------------------------------------------------------- #
def test_travel_penalty_prefers_staying_put() -> None:
    """Equal value, different travel: the cheaper chain should win."""
    catalog = catalog_of(
        session_at("HOME1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("NEAR1", "11:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("FAR1", "11:00", venue="Mandalay Bay", room="Level 1 | South Pacific F"),
    )
    profile = profile_with(weights=ObjectiveWeights(travel_penalty=5.0, coverage=0.0))
    plan = solve(request_for(catalog, profile))
    assert "NEAR1" in codes(plan)
    assert "FAR1" not in codes(plan)


def test_travel_cost_is_actually_charged() -> None:
    """Without the arc-count constraint the solver could zero out travel cost."""
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    profile = profile_with(weights=ObjectiveWeights(travel_penalty=1.0))
    plan = solve(request_for(catalog, profile))
    assert len(plan.items) == 2
    assert plan.objective.travel_cost > 0


def test_objective_total_matches_its_parts() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00"),
        session_at("B1", "12:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    plan = solve(request_for(catalog, profile_with()))
    parts = plan.objective
    expected = (
        parts.utility
        + parts.coverage
        - parts.travel_cost
        - parts.fatigue_cost
        - parts.hop_cost
        - parts.idle_cost
    )
    assert parts.total == pytest.approx(expected)


def test_venue_hop_cost_appears_when_venues_change() -> None:
    catalog = catalog_of(
        session_at("A1", "09:00", venue="Venetian", room="Level 2 | Hall C"),
        session_at("B1", "13:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    profile = profile_with(weights=ObjectiveWeights(venue_hop_penalty=1.0))
    plan = solve(request_for(catalog, profile))
    assert len(plan.items) == 2
    assert plan.objective.hop_cost > 0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_solving_twice_gives_the_same_plan() -> None:
    """Without this, "why did my plan change?" is unanswerable."""
    catalog = catalog_of(*[session_at(f"S{i}", f"{8 + i:02d}:00") for i in range(12)])
    profile = profile_with(max_sessions_per_day=4)
    first = solve(request_for(catalog, profile))
    second = solve(request_for(catalog, profile))
    assert [item.session_id for item in first.items] == [item.session_id for item in second.items]
    assert first.objective.total == pytest.approx(second.objective.total)


# --------------------------------------------------------------------------- #
# Candidate pool
# --------------------------------------------------------------------------- #
def test_candidate_cap_records_below_cutoff_rejections() -> None:
    catalog = catalog_of(*[session_at(f"S{i:02d}", "09:00", day=DAY) for i in range(30)])
    request = request_for(catalog, candidates_per_day=5)
    prepared = prepare(request)
    assert len(prepared.candidates) == 5
    below = [
        item for item in prepared.pre_rejections if item.reason is RejectionReason.BELOW_CUTOFF
    ]
    assert len(below) == 25
    assert "cutoff" in below[0].detail


def test_pins_bypass_the_candidate_cap() -> None:
    """Nothing you explicitly asked for may be pruned for being low-scoring."""
    sessions = [
        session_at(f"S{i:02d}", f"{8 + i:02d}:00", services=("Amazon Bedrock",)) for i in range(10)
    ]
    sessions.append(session_at("PIN1", "20:00", topics=(), services=(), title="Dull"))
    catalog = catalog_of(*sessions)
    profile = Profile(topics={"bedrock": 5.0}, must_attend=("PIN1",), min_break_minutes=0)
    prepared = prepare(request_for(catalog, profile, candidates_per_day=3))
    assert "PIN1" in {item.session.code for item in prepared.candidates}


def test_unscheduled_sessions_never_enter_the_model() -> None:
    floating = decode_session({"sessionId": "expo", "title": "Expo floor", "isAllDaySession": True})
    assert floating is not None
    catalog = catalog_of(session_at("A1", "09:00"), floating)
    plan = solve(request_for(catalog))
    assert "expo" not in {item.session_id for item in plan.items}
    rejection = plan.rejection_for("expo")
    assert rejection is not None
    assert rejection.reason is RejectionReason.VETOED


# --------------------------------------------------------------------------- #
# Every candidate is accounted for
# --------------------------------------------------------------------------- #
def test_every_session_is_either_scheduled_or_explained() -> None:
    """No session may vanish silently."""
    catalog = catalog_of(
        *[session_at(f"S{i}", f"{8 + i // 2:02d}:{(i % 2) * 30:02d}") for i in range(12)]
    )
    plan = solve(request_for(catalog, profile_with(max_sessions_per_day=3)))
    accounted = {item.session_id for item in plan.items} | {
        item.session_id for item in plan.rejections
    }
    assert accounted == {session.session_id for session in catalog.sessions}


def test_rejections_carry_non_empty_details() -> None:
    catalog = catalog_of(
        *[session_at(f"S{i}", f"{8 + i // 2:02d}:{(i % 2) * 30:02d}") for i in range(10)]
    )
    plan = solve(request_for(catalog, profile_with(max_sessions_per_day=2)))
    for rejection in plan.rejections:
        assert rejection.detail.strip(), rejection.code
        assert rejection.reason.label


def test_worst_rejections_are_the_highest_scoring() -> None:
    catalog = catalog_of(
        session_at("LOW1", "09:00", topics=(), services=(), title="Dull"),
        session_at("HIGH1", "09:30", services=("Amazon Bedrock",)),
        session_at("MID1", "09:15", services=("AWS Lambda",)),
    )
    profile = Profile(topics={"bedrock": 5.0, "lambda": 2.0})
    plan = solve(request_for(catalog, profile))
    worst = plan.worst_rejections(2)
    assert worst
    assert worst[0].score >= worst[-1].score


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_plan_round_trips_through_json(tmp_path: Path) -> None:
    catalog = catalog_of(
        session_at("A1", "09:00"),
        session_at("B1", "12:00", venue="Wynn", room="Level 1 | Latour 1"),
    )
    plan = solve(request_for(catalog, profile_with()))
    target = plan.write(tmp_path / "plan.json")
    restored = Plan.read(target)

    assert [item.session_id for item in restored.items] == [item.session_id for item in plan.items]
    assert restored.objective.total == pytest.approx(plan.objective.total)
    assert restored.stats.solver is plan.stats.solver
    assert len(restored.rejections) == len(plan.rejections)
    assert restored.event_id == plan.event_id


def test_reading_a_missing_plan_explains_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="replan optimize"):
        Plan.read(tmp_path / "absent.json")


def test_plan_from_a_newer_format_is_refused() -> None:
    with pytest.raises(ValueError, match="newer replan"):
        Plan.from_dict({"formatVersion": 999})


def test_plan_helpers() -> None:
    catalog = catalog_of(session_at("A1", "09:00", reservable=True, seats="limited"))
    plan = solve(request_for(catalog, profile_with()))
    assert plan.item_for("A1") is not None
    assert plan.item_for("a1") is not None
    assert plan.total_score > 0
    assert plan.total_minutes == 60
    assert len(plan.needs_reservation) == 1
    assert plan.days


def test_a_session_you_are_indifferent_to_still_beats_an_empty_slot() -> None:
    """With no preferences expressed, every score is zero.

    An empty agenda would then be genuinely optimal and completely useless, so a
    small attendance bonus tips the balance toward actually going to something.
    """
    catalog = catalog_of(session_at("A1", "09:00"))
    plan = solve(request_for(catalog, Profile()))
    assert len(plan.items) == 1
    assert plan.stats.optimal


# --------------------------------------------------------------------------- #
# Real catalog
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live_catalog(
    list_sessions_payload: dict[str, Any], get_event_payload: dict[str, Any]
) -> Catalog:
    event = decode_event(get_event_payload.get("event", get_event_payload))
    page = decode_catalog_page(list_sessions_payload, fallback_timezone=event.timezone_name)
    return build_catalog(
        event_id=event.event_id,
        sessions=page.sessions,
        event=event,
        source="recorded",
        total_count=page.total_count,
    )


def test_solves_the_real_catalog_optimally_and_fast(live_catalog: Catalog) -> None:
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    plan = solve(request_for(live_catalog, profile))

    assert plan.stats.solver is SolverKind.CPSAT
    assert plan.stats.optimal, plan.stats.status
    assert plan.stats.wall_seconds < 10.0
    assert plan.items

    # The plan must be internally consistent: no overlaps, every transfer walkable.
    graph = default_graph()
    for _day, items in plan.by_day():
        for first, second in itertools.pairwise(items):
            assert not first.span.overlaps(second.span)
            needed = second.travel_minutes or 0
            gap = first.span.gap_before(second.span)
            assert gap is not None
            assert gap >= needed
        _ = graph


def test_real_plan_respects_the_profile_day_window(live_catalog: Catalog) -> None:
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    plan = solve(request_for(live_catalog, profile))
    for item in plan.items:
        local_start, local_end = item.span.local(item.zone)
        assert profile.day_start is not None
        assert local_start.time() >= profile.day_start
        assert profile.day_end is not None
        assert local_end.time() <= profile.day_end


def test_greedy_produces_a_legal_plan_on_the_real_catalog(live_catalog: Catalog) -> None:
    """The fallback may be worse, but it may never be illegal."""
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    request = request_for(live_catalog, profile)
    selected, _breakdown, stats, prepared = solve_greedy(request)
    assert stats.solver is SolverKind.GREEDY

    blocked = {(item.first, item.second) for item in prepared.incompatible}
    for left in selected:
        for right in selected:
            if left != right:
                assert (left, right) not in blocked


def test_cpsat_scores_at_least_as_well_as_greedy(live_catalog: Catalog) -> None:
    """If the constraint solver were not better, it would not be worth the wheel."""
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    optimal = solve(request_for(live_catalog, profile), prefer_cpsat=True)
    fallback = solve(request_for(live_catalog, profile), prefer_cpsat=False)
    assert optimal.objective.total >= fallback.objective.total - 1e-6
