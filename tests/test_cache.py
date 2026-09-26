"""Tests for the local cache, full-text search, and catalog change detection.

The catalog fixture is a real recorded AWS Events API payload, so the index is
exercised against genuine titles, abstracts, and speaker strings rather than
invented text.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from replan.adapters.wire import build_catalog, decode_catalog_page, decode_session
from replan.cache import (
    CatalogCache,
    ChangeKind,
    SearchFilters,
    build_fts_query,
    session_fingerprint,
)
from replan.domain import (
    Catalog,
    KnownVenue,
    Level,
    SeatAvailability,
    Session,
    SessionFormat,
    TimeSpan,
)

EVENT = "test-event"


@pytest.fixture
def cache(tmp_path: Path) -> CatalogCache:
    return CatalogCache(tmp_path / "catalog.db")


@pytest.fixture
def live_catalog(list_sessions_payload: dict[str, Any], public_event_id: str) -> Catalog:
    page = decode_catalog_page(list_sessions_payload)
    return build_catalog(
        event_id=public_event_id,
        sessions=page.sessions,
        source="recorded",
        total_count=page.total_count,
    )


def make_session(
    session_id: str = "S1",
    *,
    code: str = "SVS401",
    title: str = "Building serverless things",
    level: str = "300 \u2013 Advanced",
    session_type: str = "Chalk talk",
    venue: str | None = "Venetian",
    room: str | None = "Level 2 | Hall C",
    clock: str = "10:00",
    day: str = "2026-12-01",
    minutes: str = "60",
    seats: str | None = "available",
    reservable: bool | None = True,
    abstract: str = "Learn about AWS Lambda and Amazon EventBridge in production.",
    speakers: tuple[str, ...] = ("Ada Lovelace, Principal Engineer, AWS",),
    services: tuple[str, ...] = ("AWS Lambda",),
    tracks: tuple[str, ...] = ("Serverless & Event-Driven",),
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": session_id,
        "title": title,
        "abbreviation": code,
        "abstract": abstract,
        "level": level,
        "type": session_type,
        "room": room,
        "speakers": [{"name": name} for name in speakers],
        "services": list(services),
        "tracks": list(tracks),
        "sessionTime": {
            "date": day,
            "time": clock,
            "length": minutes,
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


def catalog_of(*sessions: Session, total: int | None = None) -> Catalog:
    return build_catalog(
        event_id=EVENT,
        sessions=sessions,
        source="test",
        total_count=total if total is not None else len(sessions),
    )


# --------------------------------------------------------------------------- #
# FTS query construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("bedrock", '"bedrock"*'),
        ("bedrock agents", '"bedrock"* AND "agents"*'),
        ('"exact phrase"', '"exact phrase"'),
        ("", ""),
        ("   ", ""),
    ],
)
def test_build_fts_query(raw: str, expected: str) -> None:
    assert build_fts_query(raw) == expected


@pytest.mark.parametrize(
    "hostile",
    [
        'unbalanced "quote',
        "trailing operator AND",
        "colon:syntax",
        "NEAR(a b",
        "*",
        "^caret",
        "-minus",
        "a)b(c",
        "''",
    ],
)
def test_fts_query_never_produces_invalid_syntax(hostile: str, cache: CatalogCache) -> None:
    """A search box must not crash on punctuation.

    FTS5 raises OperationalError on its own operator syntax, so input is rebuilt
    into safe terms rather than passed through.
    """
    cache.save(catalog_of(make_session()))
    hits = cache.search(EVENT, hostile)
    assert isinstance(hits, tuple)  # no exception is the assertion


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(cache: CatalogCache, live_catalog: Catalog) -> None:
    cache.save(live_catalog)
    restored = cache.load(live_catalog.event_id)

    assert restored is not None
    assert len(restored) == len(live_catalog)
    assert {s.session_id for s in restored.sessions} == {
        s.session_id for s in live_catalog.sessions
    }

    original = {s.session_id: s for s in live_catalog.sessions}
    for session in restored.sessions:
        # Full object equality: the cache must not quietly lose a field.
        assert session == original[session.session_id]


def test_load_missing_event_returns_none(cache: CatalogCache) -> None:
    assert cache.load("never-synced") is None
    assert cache.info("never-synced") is None


def test_cache_records_provenance(cache: CatalogCache, live_catalog: Catalog) -> None:
    cache.save(live_catalog)
    restored = cache.load(live_catalog.event_id)
    assert restored is not None
    # Provenance must survive caching: the user has to know this is cached data.
    assert "cache" in restored.source
    assert "recorded" in restored.source


def test_info_reports_completeness_and_age(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A"), make_session("B"), total=2))
    info = cache.info(EVENT)
    assert info is not None
    assert info.session_count == 2
    assert info.is_complete
    assert info.age_seconds is not None
    assert info.age_seconds < 60
    assert info.age_human.endswith("ago")


def test_info_detects_incomplete_walk(cache: CatalogCache) -> None:
    """totalCount says 50 but only 2 arrived: that must be visible."""
    cache.save(catalog_of(make_session("A"), make_session("B"), total=50))
    info = cache.info(EVENT)
    assert info is not None
    assert info.is_complete is False


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def test_search_finds_real_sessions(cache: CatalogCache, live_catalog: Catalog) -> None:
    cache.save(live_catalog)
    hits = cache.search(live_catalog.event_id, "bedrock", limit=50)
    assert hits, "the recorded catalog definitely mentions Bedrock"
    for hit in hits:
        assert "bedrock" in hit.session.search_blob.lower()


def test_search_matches_stemmed_words(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session(abstract="Operating agentic workflows at scale")))
    assert cache.search(EVENT, "agent")
    assert cache.search(EVENT, "operate") or cache.search(EVENT, "operating")


def test_search_covers_speakers_and_services(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session(
                "S1",
                speakers=("Grace Hopper, Distinguished Engineer, AWS",),
                services=("Amazon EventBridge",),
            )
        )
    )
    assert cache.search(EVENT, "Hopper")
    assert cache.search(EVENT, "EventBridge")


def test_empty_query_lists_in_time_order(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session("late", code="B2", clock="16:00"),
            make_session("early", code="A1", clock="09:00"),
        )
    )
    hits = cache.search(EVENT, "")
    assert [hit.session.session_id for hit in hits] == ["early", "late"]


def test_filter_by_level(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session("a", code="X100", level="100 \u2013 Foundational"),
            make_session("b", code="X300", level="300 \u2013 Advanced"),
            make_session("c", code="X400", level="400 \u2013 Expert"),
        )
    )
    hits = cache.search(EVENT, "", filters=SearchFilters(levels=frozenset({Level.ADVANCED})))
    assert [hit.session.session_id for hit in hits] == ["b"]

    several = cache.search(
        EVENT,
        "",
        filters=SearchFilters(levels=frozenset({Level.ADVANCED, Level.EXPERT})),
    )
    assert {hit.session.session_id for hit in several} == {"b", "c"}


def test_filter_by_day(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session("mon", day="2026-11-30"),
            make_session("tue", day="2026-12-01"),
        )
    )
    from datetime import date

    hits = cache.search(EVENT, "", filters=SearchFilters(days=frozenset({date(2026, 12, 1)})))
    assert [hit.session.session_id for hit in hits] == ["tue"]


def test_filter_by_format_and_venue(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session("ct", session_type="Chalk talk", venue="Venetian"),
            make_session("ws", session_type="Workshop", venue="Mandalay Bay"),
        )
    )
    by_format = cache.search(
        EVENT, "", filters=SearchFilters(formats=frozenset({SessionFormat.WORKSHOP}))
    )
    assert [hit.session.session_id for hit in by_format] == ["ws"]

    by_venue = cache.search(
        EVENT, "", filters=SearchFilters(venues=frozenset({KnownVenue.VENETIAN}))
    )
    assert [hit.session.session_id for hit in by_venue] == ["ct"]


def test_bookable_filter_excludes_full_and_walkup(cache: CatalogCache) -> None:
    """Seat state is a band, so 'bookable' is defined over bands, not counts."""
    cache.save(
        catalog_of(
            make_session("open", seats="available"),
            make_session("tight", seats="veryLimited"),
            make_session("full", seats="unavailable"),
            make_session("walk", seats="walkUp"),
            make_session("nores", seats=None, reservable=False),
        )
    )
    hits = cache.search(EVENT, "", filters=SearchFilters(bookable_only=True))
    assert {hit.session.session_id for hit in hits} == {"open", "tight"}


def test_seat_band_filter(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session("a", seats="available"),
            make_session("b", seats="unavailable"),
        )
    )
    hits = cache.search(
        EVENT,
        "",
        filters=SearchFilters(seat_bands=frozenset({SeatAvailability.UNAVAILABLE})),
    )
    assert [hit.session.session_id for hit in hits] == ["b"]


def test_scheduled_only_filter(cache: CatalogCache) -> None:
    unscheduled = decode_session(
        {"sessionId": "expo", "title": "Expo floor", "isAllDaySession": True}
    )
    assert unscheduled is not None
    cache.save(catalog_of(make_session("real"), unscheduled))

    assert len(cache.search(EVENT, "")) == 2
    hits = cache.search(EVENT, "", filters=SearchFilters(scheduled_only=True))
    assert [hit.session.session_id for hit in hits] == ["real"]


def test_exclude_repeats_keeps_earliest_showing(cache: CatalogCache) -> None:
    """Repeats are separate catalog entries sharing a series code."""
    cache.save(
        catalog_of(
            make_session("r2", code="SVS401-R2", clock="15:00"),
            make_session("r1", code="SVS401", clock="09:00"),
        )
    )
    hits = cache.search(EVENT, "", filters=SearchFilters(exclude_repeats=True))
    assert [hit.session.session_id for hit in hits] == ["r1"]


def test_filter_by_track_and_service(cache: CatalogCache) -> None:
    cache.save(
        catalog_of(
            make_session("sls", tracks=("Serverless & Event-Driven",), services=("AWS Lambda",)),
            make_session("ai", tracks=("Generative AI",), services=("Amazon Bedrock",)),
        )
    )
    hits = cache.search(EVENT, "", filters=SearchFilters(tracks=frozenset({"Generative AI"})))
    assert [hit.session.session_id for hit in hits] == ["ai"]

    hits = cache.search(EVENT, "", filters=SearchFilters(services=frozenset({"AWS Lambda"})))
    assert [hit.session.session_id for hit in hits] == ["sls"]


def test_limit_is_respected(cache: CatalogCache, live_catalog: Catalog) -> None:
    cache.save(live_catalog)
    assert len(cache.search(live_catalog.event_id, "", limit=7)) == 7


def test_search_isolates_events(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("mine")))
    other = build_catalog(event_id="other-event", sessions=(make_session("theirs"),), source="test")
    cache.save(other)
    assert [hit.session.session_id for hit in cache.search(EVENT, "")] == ["mine"]
    assert [hit.session.session_id for hit in cache.search("other-event", "")] == ["theirs"]


# --------------------------------------------------------------------------- #
# Change detection
# --------------------------------------------------------------------------- #
def test_first_sync_is_flagged_and_lists_no_changes(cache: CatalogCache) -> None:
    diff = cache.save(catalog_of(make_session("A"), make_session("B")))
    assert diff.is_first_sync
    assert diff.is_empty
    assert diff.total_now == 2
    assert "first sync" in diff.summary()


def test_identical_resync_reports_nothing(cache: CatalogCache, live_catalog: Catalog) -> None:
    cache.save(live_catalog)
    diff = cache.save(live_catalog)
    assert diff.is_first_sync is False
    assert diff.is_empty
    assert diff.unchanged == len(live_catalog)
    assert "no changes" in diff.summary()


def test_detects_added_and_removed(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A"), make_session("B")))
    diff = cache.save(catalog_of(make_session("A"), make_session("C", code="NEW1")))

    assert {c.session_id for c in diff.added} == {"C"}
    assert {c.session_id for c in diff.removed} == {"B"}
    assert diff.unchanged == 1


def test_detects_time_change_as_rescheduled(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A", clock="10:00")))
    diff = cache.save(catalog_of(make_session("A", clock="14:30")))

    changed = diff.of(ChangeKind.RESCHEDULED)
    assert len(changed) == 1
    assert changed[0].session_id == "A"
    assert any(item.field_name == "start" for item in changed[0].fields)
    assert changed[0].kind.breaks_a_plan


def test_detects_room_change_as_moved(cache: CatalogCache) -> None:
    """The change that silently breaks a hand-built agenda."""
    cache.save(catalog_of(make_session("A", venue="Venetian", room="Level 2 | Hall C")))
    diff = cache.save(catalog_of(make_session("A", venue="Mandalay Bay", room="South Pacific F")))

    moved = diff.of(ChangeKind.MOVED)
    assert len(moved) == 1
    assert moved[0].kind.breaks_a_plan
    fields = {item.field_name for item in moved[0].fields}
    assert "venue" in fields


def test_detects_seat_change_and_treats_it_as_harmless(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A", seats="available")))
    diff = cache.save(catalog_of(make_session("A", seats="unavailable")))

    seats = diff.of(ChangeKind.SEATS)
    assert len(seats) == 1
    # A seat filling does not invalidate the plan; it is the watcher's business.
    assert seats[0].kind.breaks_a_plan is False
    assert diff.plan_breaking == ()


def test_detects_title_edit(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A", title="Old title")))
    diff = cache.save(catalog_of(make_session("A", title="New title")))
    edits = diff.of(ChangeKind.EDITED)
    assert len(edits) == 1
    assert edits[0].fields[0].before == "Old title"
    assert edits[0].fields[0].after == "New title"


def test_abstract_edit_is_detected_without_storing_the_text(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A", abstract="First version of the abstract.")))
    diff = cache.save(catalog_of(make_session("A", abstract="Second version of the abstract.")))
    edits = diff.of(ChangeKind.EDITED)
    assert len(edits) == 1
    change = next(item for item in edits[0].fields if item.field_name == "abstract")
    # Compared by digest, so the long text is never duplicated into the diff.
    assert change.before != change.after
    assert len(change.before or "") == 16


def test_most_severe_change_wins(cache: CatalogCache) -> None:
    """A session that both moved and filled up is reported as a move."""
    cache.save(catalog_of(make_session("A", venue="Venetian", seats="available")))
    diff = cache.save(catalog_of(make_session("A", venue="Wynn", seats="unavailable")))

    assert len(diff.changes) == 1
    assert diff.changes[0].kind is ChangeKind.MOVED
    fields = {item.field_name for item in diff.changes[0].fields}
    assert {"venue", "seats"} <= fields


def test_taxonomy_churn_is_not_reported_as_a_change(cache: CatalogCache) -> None:
    """Tracks gaining a label must not look like a schedule change."""
    cache.save(catalog_of(make_session("A", tracks=("Compute",))))
    diff = cache.save(catalog_of(make_session("A", tracks=("Compute", "Architecture"))))
    assert diff.is_empty


def test_fingerprint_is_stable_and_sensitive() -> None:
    base = make_session("A")
    assert session_fingerprint(base) == session_fingerprint(make_session("A"))
    assert session_fingerprint(base) != session_fingerprint(make_session("A", clock="12:00"))
    # Taxonomy is outside the fingerprint by design.
    assert session_fingerprint(base) == session_fingerprint(
        make_session("A", tracks=("Compute", "Storage"))
    )


def test_change_history_is_queryable(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A", clock="10:00")))
    cache.save(catalog_of(make_session("A", clock="11:00")))

    rows = cache.recent_changes(EVENT, syncs=1)
    assert len(rows) == 1
    kind, code, _title, session_id, details = rows[0]
    assert kind == ChangeKind.RESCHEDULED.value
    assert session_id == "A"
    assert code == "SVS401"
    assert any("start" in item for item in details)


def test_sync_history_is_recorded(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A")))
    cache.save(catalog_of(make_session("A", clock="12:00")))
    history = cache.sync_history(EVENT, limit=5)
    assert len(history) == 2
    assert "first sync" in history[-1][1]


def test_sync_replaces_rather_than_accumulates(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A"), make_session("B"), make_session("C")))
    cache.save(catalog_of(make_session("A")))
    restored = cache.load(EVENT)
    assert restored is not None
    assert len(restored) == 1


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #
def test_clear_one_event(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A"), make_session("B")))
    cache.save(build_catalog(event_id="other", sessions=(make_session("X"),), source="t"))

    removed = cache.clear(EVENT)
    assert removed == 2
    assert cache.load(EVENT) is None
    assert cache.load("other") is not None


def test_clear_everything(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A")))
    cache.save(build_catalog(event_id="other", sessions=(make_session("X"),), source="t"))
    cache.clear(None)
    assert cache.cached_events() == ()


def test_distinct_rejects_unknown_columns(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A")))
    assert cache.distinct(EVENT, "day")
    with pytest.raises(ValueError, match="not a filterable column"):
        cache.distinct(EVENT, "payload; DROP TABLE sessions")


def test_cached_events_lists_everything(cache: CatalogCache) -> None:
    cache.save(catalog_of(make_session("A")))
    cache.save(build_catalog(event_id="zzz", sessions=(make_session("X"),), source="t"))
    assert cache.cached_events() == (EVENT, "zzz")


def test_reopening_the_cache_keeps_data(tmp_path: Path) -> None:
    path = tmp_path / "catalog.db"
    CatalogCache(path).save(catalog_of(make_session("A")))
    assert len(CatalogCache(path).search(EVENT, "")) == 1


def test_unscheduled_sessions_survive_the_cache(cache: CatalogCache) -> None:
    unscheduled = decode_session({"sessionId": "tbd", "title": "To be announced"})
    assert unscheduled is not None
    cache.save(catalog_of(unscheduled))
    restored = cache.load(EVENT)
    assert restored is not None
    assert restored.sessions[0].span is None
    assert restored.sessions[0].is_scheduled is False


def test_session_with_absent_optional_fields_survives(cache: CatalogCache) -> None:
    """A Summit catalog omits venue, seats and isReservable entirely."""
    sparse = decode_session(
        {
            "sessionId": "sparse",
            "title": "Minimal session",
            "room": "Floor 0, Industry Theatre",
            "sessionTime": {"date": "2026-09-30", "time": "09:00", "length": "30"},
        }
    )
    assert sparse is not None
    cache.save(catalog_of(sparse))
    restored = cache.load(EVENT)
    assert restored is not None
    recovered = restored.sessions[0]
    assert recovered.seat_availability is None
    assert recovered.is_reservable is None
    assert recovered.room.venue is None
    assert recovered == sparse


def test_timespan_survives_serialisation(cache: CatalogCache) -> None:
    original = make_session("A", clock="14:30", minutes="90")
    cache.save(catalog_of(original))
    restored = cache.load(EVENT)
    assert restored is not None
    span = restored.sessions[0].span
    assert span is not None
    assert span.minutes == 90
    assert span == TimeSpan(
        datetime(2026, 12, 1, 14, 30, tzinfo=original.zone),
        datetime(2026, 12, 1, 14, 30, tzinfo=original.zone) + timedelta(minutes=90),
    )
    assert span.start.tzinfo is not None
    assert span.start.astimezone(UTC).hour == 22  # 14:30 Pacific in December
