"""One contract suite, run against every ``CatalogPort`` implementation.

Parameterising over the implementations is what stops them drifting. The
recorded client is not a convenient lie: it is held to the same behavioural
contract as the live REST client, including the API's genuinely awkward parts -
partial success on a 200, "already favorited" arriving as a *failure*,
404-means-already-done on single removals, and reservations refused with 409
until reserved seating opens through the API on 8 October 2026.

The live-REST case is driven through a mock transport rather than the network, so
this suite is deterministic and offline. A separate ``@pytest.mark.live`` suite
hits the real API.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
import pytest

from replan.adapters.auth import TokenProvider, TokenSet, TokenStore
from replan.adapters.errors import EventsNotFoundError, EventsOperationClosedError
from replan.adapters.port import CatalogPort
from replan.adapters.recorded import RecordedCatalogClient
from replan.adapters.rest import RestCatalogClient
from replan.adapters.throttle import QuotaLedger
from replan.config import RetryPolicy, Settings
from replan.domain import BulkFailureCode, PersonalTimeFormatError, TimeSpan

pytestmark = pytest.mark.contract


class PortFactory(Protocol):
    def __call__(self) -> CatalogPort: ...


# --------------------------------------------------------------------------- #
# A mock-transport REST client backed by the same recordings
# --------------------------------------------------------------------------- #
class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


def _rest_over_recordings(recorded_dir: Path, tmp_path: Path) -> RestCatalogClient:
    """A real ``RestCatalogClient`` whose transport replays the recorded payloads.

    This exercises the actual production code path - pagination, quota
    accounting, error classification, header discipline - against real bytes.
    """
    import json

    backing = RecordedCatalogClient(recorded_dir, reservations_open=True)
    event_id = backing.recorded_event_id
    events_payload = json.loads((recorded_dir / "list_events.json").read_text())
    sessions_payload = json.loads((recorded_dir / "list_sessions_page.json").read_text())

    state: dict[str, Any] = {"reserved": set(), "favorites": set(), "blocks": {}, "seq": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if path == "/v1/events":
            return httpx.Response(200, json=events_payload)
        if path == f"/v1/events/{event_id}":
            for item in events_payload["items"]:
                if item["eventId"] == event_id:
                    return httpx.Response(200, json=item)
            return httpx.Response(404, json={"message": "no such event"})
        if path == f"/v1/events/{event_id}/sessions":
            # The real service strips the abstract server-side when asked, so the
            # double must too, otherwise this is not testing the same contract.
            if request.url.params.get("includeAbstracts") == "false":
                trimmed = {
                    **sessions_payload,
                    "items": [
                        {key: value for key, value in item.items() if key != "abstract"}
                        for item in sessions_payload["items"]
                    ],
                }
                return httpx.Response(200, json=trimmed)
            return httpx.Response(200, json=sessions_payload)
        if path.startswith(f"/v1/events/{event_id}/sessions/"):
            wanted = path.rsplit("/", 1)[-1]
            for item in sessions_payload["items"]:
                if item["sessionId"] == wanted:
                    return httpx.Response(200, json=item)
            return httpx.Response(404, json={"message": "no such session"})
        if path == f"/v1/events/{event_id}/schedule":
            return httpx.Response(
                200,
                json={
                    "schedule": {
                        "reserved": sorted(state["reserved"]),
                        "favorites": sorted(state["favorites"]),
                        "personalTime": list(state["blocks"].values()),
                    }
                },
            )
        if path == f"/v1/events/{event_id}/favorites" and method == "POST":
            body = json.loads(request.content)
            successful, failed = [], []
            for session_id in body["sessionIds"]:
                if session_id in state["favorites"]:
                    failed.append({"sessionId": session_id, "code": "alreadyFavorited"})
                else:
                    state["favorites"].add(session_id)
                    successful.append(session_id)
            return httpx.Response(
                200, json={"result": {"successful": successful, "failed": failed}}
            )
        if path.startswith(f"/v1/events/{event_id}/favorites/") and method == "DELETE":
            session_id = path.rsplit("/", 1)[-1]
            if session_id not in state["favorites"]:
                return httpx.Response(404, json={"message": "not favorited"})
            state["favorites"].discard(session_id)
            return httpx.Response(204)
        if path == f"/v1/events/{event_id}/reservations" and method == "POST":
            body = json.loads(request.content)
            successful, failed = [], []
            for session_id in body["sessionIds"]:
                if session_id in state["reserved"]:
                    failed.append({"sessionId": session_id, "code": "alreadyScheduled"})
                else:
                    state["reserved"].add(session_id)
                    successful.append(session_id)
            return httpx.Response(
                200, json={"result": {"successful": successful, "failed": failed}}
            )
        if path.startswith(f"/v1/events/{event_id}/reservations/") and method == "DELETE":
            session_id = path.rsplit("/", 1)[-1]
            if session_id not in state["reserved"]:
                return httpx.Response(404, json={"message": "not reserved"})
            state["reserved"].discard(session_id)
            return httpx.Response(204)
        if path == f"/v1/events/{event_id}/personal-time" and method == "POST":
            body = json.loads(request.content)
            state["seq"] += 1
            entry_id = f"pt-{state['seq']}"
            state["blocks"][entry_id] = {"personalTimeId": entry_id, **body}
            return httpx.Response(204)
        if path.startswith(f"/v1/events/{event_id}/personal-time/"):
            entry_id = path.rsplit("/", 1)[-1]
            if method == "PUT":
                if entry_id not in state["blocks"]:
                    return httpx.Response(404, json={"message": "gone"})
                body = json.loads(request.content)
                state["blocks"][entry_id] = {"personalTimeId": entry_id, **body}
                return httpx.Response(204)
            if method == "DELETE":
                state["blocks"].pop(entry_id, None)
                return httpx.Response(204)

        return httpx.Response(404, json={"message": f"unhandled {method} {path}"})

    settings = Settings(
        state_dir=tmp_path,
        event_id=event_id,
        retry=RetryPolicy(max_attempts=2, base_delay_seconds=0.0),
    )
    store = TokenStore(settings.token_path)
    store.save(
        TokenSet(
            access_token="contract-token",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            refresh_token="r",
        )
    )
    clock = _Clock()
    return RestCatalogClient(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.api_base_url),
        tokens=TokenProvider(settings, store),
        ledger=QuotaLedger(settings, clock=clock.time, sleeper=clock.sleep),
        sleeper=clock.sleep,
    )


@pytest.fixture(params=["recorded", "live-rest"])
def port(request: pytest.FixtureRequest, recorded_dir: Path, tmp_path: Path) -> Iterator[Any]:
    if request.param == "recorded":
        yield RecordedCatalogClient(recorded_dir, reservations_open=True)
    else:
        client = _rest_over_recordings(recorded_dir, tmp_path)
        try:
            yield client
        finally:
            client.close()


@pytest.fixture
def event_id(recorded_dir: Path) -> str:
    return RecordedCatalogClient(recorded_dir).recorded_event_id


# --------------------------------------------------------------------------- #
# Structural conformance
# --------------------------------------------------------------------------- #
def test_implements_the_protocol(port: Any) -> None:
    assert isinstance(port, CatalogPort)


def test_declares_its_provenance(port: Any) -> None:
    """Provenance is mandatory: a tool that books real seats must never leave the
    user guessing whether the data was live."""
    assert port.surface_name
    assert isinstance(port.surface_name, str)


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def test_list_events(port: Any) -> None:
    events = port.list_events()
    assert events
    assert all(event.event_id for event in events)


def test_get_event_round_trips(port: Any, event_id: str) -> None:
    event = port.get_event(event_id)
    assert event.event_id == event_id


def test_get_unknown_event_raises_not_found(port: Any) -> None:
    with pytest.raises(EventsNotFoundError):
        port.get_event("no-such-event-anywhere")


def test_list_sessions_is_complete(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    assert len(catalog) > 0
    assert catalog.is_complete, "the walk must not stop before totalCount is reached"
    assert catalog.source


def test_progress_hook_is_called(port: Any, event_id: str) -> None:
    calls: list[tuple[int, int, int | None]] = []
    port.list_sessions(event_id, progress=lambda *args: calls.append(args))
    assert calls
    assert calls[-1][1] > 0


def test_include_abstracts_false_drops_abstracts(port: Any, event_id: str) -> None:
    trimmed = port.list_sessions(event_id, include_abstracts=False)
    assert all(session.abstract is None for session in trimmed.sessions)


def test_get_session_matches_the_catalog_entry(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    wanted = catalog.sessions[0]
    fetched = port.get_session(event_id, wanted.session_id)
    assert fetched.session_id == wanted.session_id
    assert fetched.title == wanted.title


def test_get_unknown_session_raises_not_found(port: Any, event_id: str) -> None:
    with pytest.raises(EventsNotFoundError):
        port.get_session(event_id, "definitely-not-a-session")


# --------------------------------------------------------------------------- #
# Favorites
# --------------------------------------------------------------------------- #
def test_favorites_round_trip(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    first, second = catalog.sessions[0].session_id, catalog.sessions[1].session_id

    result = port.associate_favorites(event_id, [first, second])
    assert result.effective_successes == {first, second}
    assert result.ok

    schedule = port.get_schedule(event_id)
    assert {first, second} <= schedule.favorite_ids

    assert port.disassociate_favorite(event_id, first) is True
    assert first not in port.get_schedule(event_id).favorite_ids


def test_refavoriting_reports_already_favorited_as_a_failure(port: Any, event_id: str) -> None:
    """The documented reason a bulk request is not a safe retry."""
    catalog = port.list_sessions(event_id)
    target = catalog.sessions[0].session_id

    port.associate_favorites(event_id, [target])
    again = port.associate_favorites(event_id, [target])

    failure = again.failure_for(target)
    assert failure is not None
    assert failure.code is BulkFailureCode.ALREADY_FAVORITED
    # ...but the end state is what the caller wanted, so it folds into success.
    assert again.effective_successes == {target}
    assert again.ok


def test_removing_a_favorite_that_is_absent_reports_already_done(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    target = catalog.sessions[2].session_id
    assert port.disassociate_favorite(event_id, target) is False


def test_empty_bulk_is_a_no_op(port: Any, event_id: str) -> None:
    assert port.associate_favorites(event_id, []).successful == ()


# --------------------------------------------------------------------------- #
# Reservations
# --------------------------------------------------------------------------- #
def test_reservations_round_trip(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    target = catalog.sessions[0].session_id

    result = port.reserve_sessions(event_id, [target])
    assert target in result.effective_successes
    assert target in port.get_schedule(event_id).reserved_ids

    assert port.cancel_reservation(event_id, target) is True
    assert target not in port.get_schedule(event_id).reserved_ids


def test_cancelling_an_unreserved_session_reports_already_done(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    assert port.cancel_reservation(event_id, catalog.sessions[3].session_id) is False


def test_reserving_twice_reports_already_scheduled(port: Any, event_id: str) -> None:
    catalog = port.list_sessions(event_id)
    target = catalog.sessions[1].session_id
    port.reserve_sessions(event_id, [target])
    again = port.reserve_sessions(event_id, [target])
    failure = again.failure_for(target)
    assert failure is not None
    assert failure.code is BulkFailureCode.ALREADY_SCHEDULED
    assert again.effective_successes == {target}


# --------------------------------------------------------------------------- #
# Personal time
# --------------------------------------------------------------------------- #
def _span(hours_from_now: int = 0, minutes: int = 60) -> TimeSpan:
    start = datetime(2026, 12, 2, 16, 0, tzinfo=UTC) + timedelta(hours=hours_from_now)
    return TimeSpan(start, start + timedelta(minutes=minutes))


def test_personal_time_create_and_delete(port: Any, event_id: str) -> None:
    before = len(port.get_schedule(event_id).personal_time)

    port.create_personal_time(event_id, span=_span(), title="Dinner", description="with the crew")
    schedule = port.get_schedule(event_id)
    assert len(schedule.personal_time) == before + 1

    created = schedule.personal_time[-1]
    assert created.title == "Dinner"
    assert created.span.minutes == 60

    assert port.delete_personal_time(event_id, created.personal_time_id) is True
    assert len(port.get_schedule(event_id).personal_time) == before


def test_deleting_personal_time_is_idempotent(port: Any, event_id: str) -> None:
    """Documented: removing one already gone succeeds, so a lost response is safe."""
    port.create_personal_time(event_id, span=_span(2), title="Gym", description="d")
    block = port.get_schedule(event_id).personal_time[-1]
    assert port.delete_personal_time(event_id, block.personal_time_id) is True
    assert port.delete_personal_time(event_id, block.personal_time_id) is True


def test_update_personal_time_replaces_every_field(port: Any, event_id: str) -> None:
    port.create_personal_time(
        event_id, span=_span(3), title="Call", description="d", location="Lobby"
    )
    block = port.get_schedule(event_id).personal_time[-1]

    port.update_personal_time(
        event_id,
        block.personal_time_id,
        span=_span(4, minutes=30),
        title="Call (moved)",
        description="rescheduled",
    )
    updated = port.get_schedule(event_id).block_by_id(block.personal_time_id)
    assert updated is not None
    assert updated.title == "Call (moved)"
    assert updated.span.minutes == 30
    # A full replacement clears omitted optional fields.
    assert updated.location is None


def test_personal_time_rejects_non_five_minute_durations(port: Any, event_id: str) -> None:
    with pytest.raises(PersonalTimeFormatError):
        port.create_personal_time(event_id, span=_span(minutes=7), title="Odd", description="d")


# --------------------------------------------------------------------------- #
# The 409 window, which is the API's behaviour right now
# --------------------------------------------------------------------------- #
def test_reservations_closed_raises_operation_closed(recorded_dir: Path, event_id: str) -> None:
    closed = RecordedCatalogClient(recorded_dir, reservations_open=False)
    with pytest.raises(EventsOperationClosedError) as caught:
        closed.reserve_sessions(event_id, ["anything"])
    assert "8 October 2026" in caught.value.remedy

    with pytest.raises(EventsOperationClosedError):
        closed.cancel_reservation(event_id, "anything")


def test_favorites_still_work_while_reservations_are_closed(
    recorded_dir: Path, event_id: str
) -> None:
    """Explicitly documented: browsing and favoriting keep working."""
    closed = RecordedCatalogClient(recorded_dir, reservations_open=False)
    catalog = closed.list_sessions(event_id)
    result = closed.associate_favorites(event_id, [catalog.sessions[0].session_id])
    assert result.ok
