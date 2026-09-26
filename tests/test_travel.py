"""Tests for the travel-time graph.

The properties asserted here are the ones a planner depends on being true:
symmetry, self-distance of zero, monotonic pessimism when data is missing, and
above all that an impossible transfer is never reported as possible. A travel
model that is optimistic in the unknown case is worse than none at all, because it
produces confident schedules that cannot be walked.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from replan.adapters.wire import decode_session
from replan.domain import KnownVenue, Room, Session
from replan.travel import (
    DEFAULT_DATA_PATH,
    Confidence,
    TravelGraph,
    TravelMode,
    default_graph,
)


@pytest.fixture(scope="module")
def graph() -> TravelGraph:
    return TravelGraph.load()


def room(venue: str | None, name: str) -> Room:
    return Room(venue=venue, name=name)


def session_at(
    code: str,
    *,
    venue: str | None,
    room_name: str,
    clock: str,
    minutes: int = 60,
    day: str = "2026-12-01",
    reservable: bool | None = None,
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": code,
        "title": f"Session {code}",
        "abbreviation": code,
        "room": room_name,
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
    decoded = decode_session(payload)
    assert decoded is not None
    return decoded


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #
def test_bundled_data_loads(graph: TravelGraph) -> None:
    assert graph.known_venue_count() >= 6
    assert graph.pairs
    assert graph.zone_defaults


def test_every_known_venue_has_a_profile(graph: TravelGraph) -> None:
    """A venue in the enum with no data would silently get pessimistic fallbacks."""
    for venue in KnownVenue:
        assert venue in graph.venues, f"{venue.value} has no travel profile"


def test_every_venue_pair_is_curated(graph: TravelGraph) -> None:
    """No pair should need the zone fallback; it exists only as a safety net."""
    venues = list(KnownVenue)
    missing: list[str] = []
    for index, left in enumerate(venues):
        for right in venues[index + 1 :]:
            if graph._pair_key(left, right) not in graph.pairs:
                missing.append(f"{left.value}|{right.value}")
    assert not missing, f"uncurated venue pairs: {missing}"


def test_data_file_declares_a_version() -> None:
    raw = yaml.safe_load(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
    assert raw["version"] == 1


# --------------------------------------------------------------------------- #
# Core graph properties
# --------------------------------------------------------------------------- #
def test_same_room_is_free(graph: TravelGraph) -> None:
    here = room("Venetian", "Level 2 | Hall C")
    leg = graph.leg(here, here)
    assert leg.minutes == 0
    assert leg.mode is TravelMode.STAY
    assert leg.is_free


def test_travel_is_symmetric(graph: TravelGraph) -> None:
    left = room("Venetian", "Level 2 | Hall C")
    right = room("Mandalay Bay", "Level 1 | South Pacific F")
    assert graph.minutes(left, right) == graph.minutes(right, left)


def test_all_venue_pairs_are_symmetric(graph: TravelGraph) -> None:
    venues = list(KnownVenue)
    for index, left in enumerate(venues):
        for right in venues[index + 1 :]:
            a = room(left.label, "Level 1 | A")
            b = room(right.label, "Level 1 | B")
            assert graph.minutes(a, b) == graph.minutes(b, a), f"{left} vs {right}"


def test_every_transfer_costs_something(graph: TravelGraph) -> None:
    """Only staying in the same room is free. Crossing a corridor is not."""
    a = room("Caesars Forum", "Level 1 | Forum 105")
    b = room("Caesars Forum", "Level 1 | Summit 220")
    assert graph.minutes(a, b) > 0


def test_far_venues_cost_more_than_near_ones(graph: TravelGraph) -> None:
    origin = room("Venetian", "Level 2 | Hall C")
    near = room("Wynn", "Level 1 | Latour 1")
    far = room("Mandalay Bay", "Level 1 | South Pacific F")
    assert graph.minutes(origin, near) < graph.minutes(origin, far)


def test_adjacent_properties_are_walked_not_shuttled(graph: TravelGraph) -> None:
    """Wynn and Encore are connected; waiting for a shuttle would be absurd."""
    leg = graph.leg(
        room("Wynn", "Level 1 | Latour 1"), room("Encore", "Level 1 | Encore Ballroom A")
    )
    assert leg.mode is TravelMode.WALK


def test_distant_properties_prefer_the_shuttle(graph: TravelGraph) -> None:
    leg = graph.leg(
        room("Venetian", "Level 2 | Hall C"),
        room("Mandalay Bay", "Level 1 | South Pacific F"),
    )
    assert leg.mode is TravelMode.SHUTTLE
    # The wait is counted, not wished away.
    assert leg.overhead_minutes > graph.overheads.fixed


def test_mode_choice_minimises_total_time(graph: TravelGraph) -> None:
    """Whichever gets you there sooner wins, wait included."""
    for left in KnownVenue:
        for right in KnownVenue:
            if left == right:
                continue
            leg = graph.leg(room(left.label, "A"), room(right.label, "B"))
            times = graph._times_for(left, right)[0]
            options = []
            if "walk" in times:
                options.append(times["walk"] + graph.overheads.fixed)
            if "shuttle" in times:
                options.append(
                    times["shuttle"] + graph.overheads.shuttle_wait + graph.overheads.fixed
                )
            if options:
                assert leg.minutes == min(options)


# --------------------------------------------------------------------------- #
# Internal transfers
# --------------------------------------------------------------------------- #
def test_floor_changes_add_time(graph: TravelGraph) -> None:
    same_floor = graph.minutes(
        room("Venetian", "Level 2 | Hall C"), room("Venetian", "Level 2 | Titian 2205")
    )
    four_floors = graph.minutes(
        room("Venetian", "Level 1 | Marcello 4301"),
        room("Venetian", "Level 5 | Palazzo Ballroom B"),
    )
    assert four_floors > same_floor


def test_large_property_internal_can_exceed_a_short_hop(graph: TravelGraph) -> None:
    """The counterintuitive result worth surfacing to a human.

    Crossing the Venetian with several floor changes genuinely takes longer than
    walking between two adjacent northern properties, which is exactly the sort of
    thing someone reading a list of room names never notices.
    """
    across_venetian = graph.minutes(
        room("Venetian", "Level 1 | Marcello 4301"),
        room("Venetian", "Level 5 | Palazzo Ballroom B"),
    )
    wynn_to_encore = graph.minutes(
        room("Wynn", "Level 1 | Latour 1"), room("Encore", "Level 1 | Encore Ballroom A")
    )
    assert across_venetian > wynn_to_encore


def test_compact_venue_beats_sprawling_one(graph: TravelGraph) -> None:
    forum = graph.minutes(
        room("Caesars Forum", "Level 1 | Forum 105"),
        room("Caesars Forum", "Level 1 | Summit 220"),
    )
    venetian = graph.minutes(
        room("Venetian", "Level 2 | Hall C"), room("Venetian", "Level 2 | Titian 2205")
    )
    assert forum < venetian


def test_missing_floor_hint_does_not_crash(graph: TravelGraph) -> None:
    leg = graph.leg(room("Venetian", "Somewhere unnumbered"), room("Venetian", "Another place"))
    assert leg.minutes > 0
    assert leg.mode is TravelMode.WALK_INTERNAL


# --------------------------------------------------------------------------- #
# Unknown venues: the Summit case
# --------------------------------------------------------------------------- #
def test_absent_venue_is_treated_as_one_building(graph: TravelGraph) -> None:
    """Summits publish no venue at all; a cross-Strip estimate would be nonsense."""
    leg = graph.leg(room(None, "Floor 0, Industry Theatre"), room(None, "Floor 0, Code Talks"))
    assert leg.mode is TravelMode.WALK_INTERNAL
    assert leg.confidence is Confidence.SAME_BUILDING_ASSUMED
    assert leg.minutes < 30


def test_unrecognised_differing_venues_are_pessimistic(graph: TravelGraph) -> None:
    leg = graph.leg(room("Some Convention Centre", "Hall 1"), room("A Different Place", "Room 2"))
    assert leg.confidence is Confidence.UNKNOWN
    assert leg.minutes >= graph.unknown_different_building


def test_half_known_pair_is_pessimistic(graph: TravelGraph) -> None:
    """One recognised venue and one not cannot be routed, so do not pretend."""
    leg = graph.leg(room("Venetian", "Level 2 | Hall C"), room("Unknown Hotel", "Room 1"))
    assert leg.confidence is Confidence.UNKNOWN
    assert leg.minutes >= graph.unknown_different_building


def test_confidence_is_ordered(graph: TravelGraph) -> None:
    assert Confidence.CURATED > Confidence.ZONE_DEFAULT
    assert Confidence.ZONE_DEFAULT > Confidence.SAME_BUILDING_ASSUMED
    assert Confidence.SAME_BUILDING_ASSUMED > Confidence.UNKNOWN


def test_zone_fallback_is_used_when_a_pair_is_absent(tmp_path: Path) -> None:
    trimmed = yaml.safe_load(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
    trimmed["pairs"] = {}  # force the fallback path
    target = tmp_path / "venues.yaml"
    target.write_text(yaml.safe_dump(trimmed), encoding="utf-8")

    sparse = TravelGraph.load(target)
    leg = sparse.leg(
        room("Venetian", "Level 2 | Hall C"),
        room("Mandalay Bay", "Level 1 | South Pacific F"),
    )
    assert leg.confidence is Confidence.ZONE_DEFAULT
    assert leg.minutes > 0


# --------------------------------------------------------------------------- #
# Feasibility: the question the solver asks
# --------------------------------------------------------------------------- #
def test_impossible_transfer_is_rejected(graph: TravelGraph) -> None:
    """The exact failure a hand-built agenda makes.

    13:30-14:30 in the Venetian, then 15:00 in Mandalay Bay. Thirty minutes of
    clear air, and the transfer needs closer to an hour.
    """
    first = session_at("SVS401", venue="Venetian", room_name="Level 2 | Hall C", clock="13:30")
    second = session_at(
        "STG301", venue="Mandalay Bay", room_name="Level 1 | South Pacific F", clock="15:00"
    )
    result = graph.feasibility(first, second)
    assert result.ok is False
    assert result.slack_minutes < 0
    assert "Mandalay Bay" in result.leg.detail or "transfer needs" in result.reason


def test_comfortable_transfer_is_accepted(graph: TravelGraph) -> None:
    first = session_at("A1", venue="Venetian", room_name="Level 2 | Hall C", clock="09:00")
    second = session_at("B1", venue="Venetian", room_name="Level 2 | Titian 2205", clock="11:00")
    result = graph.feasibility(first, second)
    assert result.ok
    assert result.is_tight is False
    assert "comfortable" in result.reason


def test_tight_transfer_is_flagged(graph: TravelGraph) -> None:
    """Makeable but with no slack should not read the same as comfortable."""
    first = session_at("A1", venue="Caesars Forum", room_name="Level 1 | Forum 105", clock="09:00")
    minutes = graph.minutes(
        room("Caesars Forum", "Level 1 | Forum 105"),
        room("Caesars Forum", "Level 1 | Summit 220"),
    )
    second = session_at(
        "B1",
        venue="Caesars Forum",
        room_name="Level 1 | Summit 220",
        clock=_clock_after("09:00", 60 + minutes + 2),
    )
    result = graph.feasibility(first, second)
    assert result.ok
    assert result.is_tight


def test_overlapping_sessions_are_never_feasible(graph: TravelGraph) -> None:
    first = session_at("A1", venue="Venetian", room_name="Hall C", clock="10:00")
    second = session_at("B1", venue="Venetian", room_name="Hall D", clock="10:30")
    result = graph.feasibility(first, second)
    assert result.ok is False
    assert "overlap" in result.reason


def test_argument_order_does_not_matter(graph: TravelGraph) -> None:
    first = session_at("A1", venue="Venetian", room_name="Hall C", clock="09:00")
    second = session_at("B1", venue="Wynn", room_name="Latour 1", clock="14:00")
    forward = graph.feasibility(first, second)
    backward = graph.feasibility(second, first)
    assert forward.ok == backward.ok
    assert forward.gap_minutes == backward.gap_minutes


def test_unscheduled_session_is_not_feasible(graph: TravelGraph) -> None:
    scheduled = session_at("A1", venue="Venetian", room_name="Hall C", clock="09:00")
    floating = decode_session({"sessionId": "expo", "title": "Expo", "venue": "Venetian"})
    assert floating is not None
    result = graph.feasibility(scheduled, floating)
    assert result.ok is False
    assert "no published time" in result.reason


def test_extra_buffer_tightens_the_check(graph: TravelGraph) -> None:
    first = session_at("A1", venue="Venetian", room_name="Hall C", clock="09:00")
    second = session_at("B1", venue="Venetian", room_name="Titian 2205", clock="10:30")
    assert graph.feasibility(first, second).ok
    assert graph.feasibility(first, second, extra_buffer=120).ok is False


def test_back_to_back_across_venues_is_impossible(graph: TravelGraph) -> None:
    """Zero gap, different buildings. Half-open intervals mean no overlap, but
    there is also no time to move, which is the distinction that matters."""
    first = session_at("A1", venue="Venetian", room_name="Hall C", clock="10:00", minutes=60)
    second = session_at("B1", venue="Wynn", room_name="Latour 1", clock="11:00", minutes=60)
    assert first.span is not None and second.span is not None
    assert first.span.overlaps(second.span) is False
    assert graph.feasibility(first, second).ok is False


# --------------------------------------------------------------------------- #
# depart_by
# --------------------------------------------------------------------------- #
def test_depart_by_is_the_start_minus_travel(graph: TravelGraph) -> None:
    first = session_at("A1", venue="Venetian", room_name="Hall C", clock="09:00")
    second = session_at("B1", venue="Wynn", room_name="Latour 1", clock="14:00")
    depart = graph.depart_by(first, second)
    assert depart is not None
    assert second.span is not None
    minutes = graph.minutes(first.room, second.room)
    assert depart == second.span.start - timedelta(minutes=minutes)


def test_depart_by_is_none_without_times(graph: TravelGraph) -> None:
    scheduled = session_at("A1", venue="Venetian", room_name="Hall C", clock="09:00")
    floating = decode_session({"sessionId": "x", "title": "T"})
    assert floating is not None
    assert graph.depart_by(scheduled, floating) is None


def test_reserved_seat_formats_get_extra_entry_time(graph: TravelGraph) -> None:
    walkin = session_at("A1", venue="Venetian", room_name="Hall C", clock="09:00")
    scanned = session_at("W1", venue="Venetian", room_name="Hall C", clock="09:00", reservable=True)
    assert graph.entry_overhead_for(scanned) > graph.entry_overhead_for(walkin)


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def test_overrides_replace_specific_pairs(tmp_path: Path) -> None:
    """An attendee who has walked the route should be able to win the argument."""
    override = tmp_path / "mine.yaml"
    override.write_text(yaml.safe_dump({"pairs": {"venetian|wynn": {"walk": 3}}}), encoding="utf-8")
    base = TravelGraph.load()
    patched = TravelGraph.load(overrides=override)

    pair = (room("Venetian", "Hall C"), room("Wynn", "Latour 1"))
    assert patched.minutes(*pair) < base.minutes(*pair)
    assert patched.minutes(*pair) == 3 + patched.overheads.fixed
    # Untouched pairs keep the bundled values.
    other = (room("Venetian", "Hall C"), room("Mandalay Bay", "South Pacific F"))
    assert patched.minutes(*other) == base.minutes(*other)


def test_overrides_can_change_overheads(tmp_path: Path) -> None:
    override = tmp_path / "mine.yaml"
    override.write_text(
        yaml.safe_dump({"overheads": {"room_exit": 0, "room_entry": 0, "shuttle_wait": 0}}),
        encoding="utf-8",
    )
    patched = TravelGraph.load(overrides=override)
    assert patched.overheads.fixed == 0
    here = room("Caesars Forum", "Level 1 | Forum 105")
    there = room("Caesars Forum", "Level 1 | Summit 220")
    assert patched.minutes(here, there) < TravelGraph.load().minutes(here, there)


def test_missing_override_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TravelGraph.load(overrides=tmp_path / "nope.yaml")


def test_source_is_reported(tmp_path: Path) -> None:
    assert "venues.yaml" in TravelGraph.load().describe_source()
    override = tmp_path / "mine.yaml"
    override.write_text(yaml.safe_dump({"pairs": {}}), encoding="utf-8")
    assert "mine.yaml" in TravelGraph.load(overrides=override).describe_source()


# --------------------------------------------------------------------------- #
# Performance and caching
# --------------------------------------------------------------------------- #
def test_legs_are_memoised(graph: TravelGraph) -> None:
    a = room("Venetian", "Hall C")
    b = room("Wynn", "Latour 1")
    assert graph.leg(a, b) is graph.leg(a, b)


def test_default_graph_is_shared() -> None:
    assert default_graph() is default_graph()


def test_matrix_covers_every_curated_route(graph: TravelGraph) -> None:
    rows = graph.matrix()
    assert len(rows) == len(graph.pairs)
    for _left, _right, minutes, mode in rows:
        assert minutes > 0
        assert mode in {item.value for item in TravelMode}


def _clock_after(clock: str, minutes: int) -> str:
    hour, _, minute = clock.partition(":")
    total = int(hour) * 60 + int(minute) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"
