"""Tests for the live REST client, driven by a mock transport.

These exercise the behaviours that are easy to get wrong and expensive to get
wrong in production: the pagination walk, quota accounting for the two bulk
operations that bill per session, ``Retry-After`` handling, the two meanings of
403, and the rule that a token is never sent where none is needed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from replan.adapters.auth import TokenProvider, TokenSet, TokenStore
from replan.adapters.errors import (
    EventsApiError,
    EventsEdgeRefusedError,
    EventsNotRegisteredError,
    EventsOperationClosedError,
    EventsThrottledError,
    EventsValidationError,
)
from replan.adapters.rest import MAX_PAGES, RestCatalogClient
from replan.adapters.throttle import QuotaLedger, backoff_delay
from replan.config import RetryPolicy, Settings
from replan.domain import TimeSpan

EVENT = "reinvent2026"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class Recorder:
    """Captures every request the client makes."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def log(self, request: httpx.Request) -> None:
        self.requests.append(request)

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def authorized(self) -> list[bool]:
        return ["authorization" in request.headers for request in self.requests]


class FakeClock:
    """A clock that only moves when something sleeps.

    Both the retry logic and the quota ledger share it, so tests stay instant
    while quota windows still slide. A frozen clock would deadlock the ledger the
    moment a test crossed an operation's per-minute limit.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(seconds, 0.0)


def make_client(
    handler: Any,
    *,
    settings: Settings | None = None,
    tokens: TokenProvider | None = None,
    clock: FakeClock | None = None,
) -> RestCatalogClient:
    resolved = settings or Settings(retry=RetryPolicy(max_attempts=3, base_delay_seconds=0.0))
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url=resolved.api_base_url)
    fake = clock or FakeClock()
    return RestCatalogClient(
        resolved,
        client=http_client,
        tokens=tokens,
        ledger=QuotaLedger(resolved, clock=fake.time, sleeper=fake.sleep),
        sleeper=fake.sleep,
    )


def signed_in_provider(tmp_path: Any) -> TokenProvider:
    settings = Settings(state_dir=tmp_path)
    store = TokenStore(settings.token_path)
    store.save(
        TokenSet(
            access_token="test-access-token",
            expires_at=datetime.now(UTC) + timedelta(minutes=45),
            refresh_token="test-refresh-token",
        )
    )
    return TokenProvider(settings, store)


def json_response(payload: dict[str, Any], status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


EVENT_PAYLOAD = {
    "eventId": EVENT,
    "name": "re:Invent 2026",
    "eventType": "AWS Global Summit",
    "startDate": "2026-11-30T05:30:00.000-08:00",
    "endDate": "2026-12-04T23:59:00.000-08:00",
    "isOnline": False,
    "timezone": "America/Los_Angeles",
    "authenticationRequired": True,
}

PUBLIC_EVENT_PAYLOAD = {**EVENT_PAYLOAD, "eventId": "Summit-X", "authenticationRequired": False}


def session_payload(index: int) -> dict[str, Any]:
    return {
        "sessionId": f"S{index}",
        "title": f"Session {index}",
        "abbreviation": f"SVS{300 + index}",
        "level": "300 \u2013 Advanced",
        "type": "Breakout session",
        "venue": "Venetian",
        "room": "Level 2 | Hall C",
        "isReservable": True,
        "seatAvailability": "available",
        "sessionTime": {
            "date": "2026-12-01",
            "time": "10:00",
            "length": "60",
            "timezone": "America/Los_Angeles",
        },
    }


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #
def test_walk_follows_next_token_to_the_end() -> None:
    """A short page must not end the walk; only an absent nextToken may."""
    pages = [
        {"items": [session_payload(1)], "totalCount": 5, "nextToken": "t1"},
        # Deliberately short, and deliberately not last.
        {"items": [], "totalCount": 5, "nextToken": "t2"},
        {"items": [session_payload(2), session_payload(3)], "totalCount": 5, "nextToken": "t3"},
        {"items": [session_payload(4), session_payload(5)], "totalCount": 5},
    ]
    seen_tokens: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/events/{EVENT}":
            return json_response(EVENT_PAYLOAD)
        token = request.url.params.get("nextToken")
        seen_tokens.append(token)
        index = 0 if token is None else int(str(token)[1:])
        return json_response(pages[index])

    client = make_client(handler, tokens=_always_signed_in())
    catalog = client.list_sessions(EVENT)

    assert len(catalog) == 5
    assert catalog.total_count == 5
    assert catalog.is_complete
    assert seen_tokens == [None, "t1", "t2", "t3"]


def test_walk_refuses_to_loop_forever() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/events/{EVENT}":
            return json_response(EVENT_PAYLOAD)
        return json_response({"items": [session_payload(1)], "nextToken": "always"})

    client = make_client(handler, tokens=_always_signed_in())
    with pytest.raises(EventsApiError, match=f"after {MAX_PAGES} pages"):
        client.list_sessions(EVENT)


def test_include_abstracts_false_is_forwarded() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == f"/v1/events/{EVENT}":
            return json_response(EVENT_PAYLOAD)
        return json_response({"items": [session_payload(1)], "totalCount": 1})

    client = make_client(handler, tokens=_always_signed_in())
    client.list_sessions(EVENT, include_abstracts=False, locale="en-US")

    listing = [r for r in captured if r.url.path.endswith("/sessions")]
    assert listing[0].url.params.get("includeAbstracts") == "false"
    assert listing[0].url.params.get("locale") == "en-US"


# --------------------------------------------------------------------------- #
# Authentication discipline
# --------------------------------------------------------------------------- #
def test_list_events_never_sends_a_token(tmp_path: Any) -> None:
    """ListEvents and GetEvent need no credentials, so none are sent."""
    recorder = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.log(request)
        return json_response({"items": [EVENT_PAYLOAD]})

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    client.list_events()

    assert recorder.authorized() == [False]


def test_public_event_catalog_is_read_anonymously(tmp_path: Any) -> None:
    """A credential-free read of a public catalog is a real capability, not a fallback."""
    recorder = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.log(request)
        if request.url.path == "/v1/events/Summit-X":
            return json_response(PUBLIC_EVENT_PAYLOAD)
        return json_response({"items": [session_payload(1)], "totalCount": 1})

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    catalog = client.list_sessions("Summit-X")

    assert len(catalog) == 1
    assert not any(recorder.authorized()), "no request should carry a token"


def test_registration_required_event_sends_a_token(tmp_path: Any) -> None:
    recorder = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.log(request)
        if request.url.path == f"/v1/events/{EVENT}":
            return json_response(EVENT_PAYLOAD)
        return json_response({"items": [session_payload(1)], "totalCount": 1})

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    client.list_sessions(EVENT)

    sessions_requests = [
        request for request in recorder.requests if request.url.path.endswith("/sessions")
    ]
    assert sessions_requests
    for request in sessions_requests:
        assert request.headers["authorization"] == "Bearer test-access-token"


def test_schedule_requires_a_token(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-access-token"
        return json_response(
            {"schedule": {"reserved": ["S1"], "favorites": ["S2"], "personalTime": []}}
        )

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    schedule = client.get_schedule(EVENT)
    assert schedule.reserved_ids == {"S1"}
    assert schedule.favorite_ids == {"S2"}


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_429_waits_for_retry_after_then_succeeds(tmp_path: Any) -> None:
    calls = {"n": 0}
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/events/{EVENT}":
            return json_response(EVENT_PAYLOAD)
        calls["n"] += 1
        if calls["n"] == 1:
            return json_response({"message": "slow down"}, status=429, **{"Retry-After": "12"})
        return json_response({"items": [session_payload(1)], "totalCount": 1})

    client = make_client(handler, tokens=signed_in_provider(tmp_path), clock=clock)
    catalog = client.list_sessions(EVENT)

    assert len(catalog) == 1
    assert 12.0 in clock.slept, f"should have honoured Retry-After; slept {clock.slept}"


def test_429_that_never_clears_raises(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/events/{EVENT}":
            return json_response(EVENT_PAYLOAD)
        return json_response({"message": "slow down"}, status=429, **{"Retry-After": "3"})

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsThrottledError) as caught:
        client.list_sessions(EVENT)
    assert caught.value.retry_after_seconds == 3


def test_403_with_body_is_not_registered_and_is_not_retried(tmp_path: Any) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return json_response({"message": "not registered for this event"}, status=403)

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsNotRegisteredError) as caught:
        client.get_schedule(EVENT)
    assert attempts["n"] == 1, "a registration problem must not be retried"
    assert "Register for the event" in caught.value.remedy


def test_403_without_body_is_an_edge_refusal_and_is_retried(tmp_path: Any) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(403, text="<html>Forbidden</html>")

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsEdgeRefusedError):
        client.get_schedule(EVENT)
    assert attempts["n"] > 1, "an edge refusal should be retried after backoff"


def test_409_on_reserve_is_reported_as_closed(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"message": "not available"}, status=409)

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsOperationClosedError) as caught:
        client.reserve_sessions(EVENT, ["S1"])
    assert "8 October 2026" in caught.value.remedy


def test_400_is_never_retried(tmp_path: Any) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return json_response({"message": "bad nextToken"}, status=400)

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsValidationError):
        client.get_schedule(EVENT)
    assert attempts["n"] == 1


def test_500_on_a_read_is_retried(tmp_path: Any) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return json_response({"message": "boom"}, status=500)
        return json_response({"schedule": {"reserved": [], "favorites": [], "personalTime": []}})

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    schedule = client.get_schedule(EVENT)
    assert schedule.is_empty()
    assert calls["n"] == 3


def test_transport_failure_on_a_write_refuses_to_retry(tmp_path: Any) -> None:
    """No idempotency key exists, so a lost write must be reconciled, not resent."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection dropped")

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsApiError, match="Reconcile against GetSchedule"):
        client.create_personal_time(
            EVENT,
            span=TimeSpan(
                datetime(2026, 12, 2, 16, 0, tzinfo=UTC),
                datetime(2026, 12, 2, 17, 0, tzinfo=UTC),
            ),
            title="Dinner",
            description="team",
        )


def test_non_json_error_body_does_not_crash(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="<html>maintenance</html>")

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    with pytest.raises(EventsApiError):
        client.get_schedule(EVENT)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def test_bulk_reserve_sends_session_ids_and_reads_partial_failure(tmp_path: Any) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return json_response(
            {
                "result": {
                    "successful": ["S1"],
                    "failed": [
                        {"sessionId": "S2", "code": "sessionFull"},
                        {"sessionId": "S3", "code": "alreadyScheduled"},
                    ],
                }
            }
        )

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    result = client.reserve_sessions(EVENT, ["S1", "S2", "S3"])

    assert captured["body"] == {"sessionIds": ["S1", "S2", "S3"]}
    assert result.effective_successes == {"S1", "S3"}
    assert [f.session_id for f in result.real_failures] == ["S2"]


def test_empty_bulk_request_makes_no_call(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call should be made for an empty batch")

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    assert client.reserve_sessions(EVENT, []).successful == ()
    assert client.associate_favorites(EVENT, []).successful == ()


def test_single_removals_treat_404_as_already_done(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"message": "not found"}, status=404)

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    assert client.cancel_reservation(EVENT, "S1") is False
    assert client.disassociate_favorite(EVENT, "S1") is False
    # DeletePersonalTime is documented as idempotent: a missing entry still succeeds.
    assert client.delete_personal_time(EVENT, "pt-1") is False


def test_create_personal_time_sends_naive_utc(tmp_path: Any) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(204)

    client = make_client(handler, tokens=signed_in_provider(tmp_path))
    client.create_personal_time(
        EVENT,
        span=TimeSpan(
            datetime(2026, 12, 2, 16, 0, tzinfo=UTC),
            datetime(2026, 12, 2, 17, 0, tzinfo=UTC),
        ),
        title="Dinner",
        description="with the crew",
        location="Mandalay Bay",
    )
    body = captured["body"]
    assert body["startDateTime"] == "2026-12-02T16:00:00"
    assert body["endDateTime"] == "2026-12-02T17:00:00"
    assert not body["startDateTime"].endswith("Z")
    assert body["location"] == "Mandalay Bay"


# --------------------------------------------------------------------------- #
# Quota ledger
# --------------------------------------------------------------------------- #
def test_bulk_operations_are_charged_per_session() -> None:
    settings = Settings()
    ledger = QuotaLedger(settings, clock=lambda: 0.0, sleeper=lambda _s: None)

    # Published: these two count each session named, not each request.
    assert ledger.cost_of("ReserveSessions", 10) == 10
    assert ledger.cost_of("AssociateFavorites", 7) == 7
    # Everything else is one unit per call.
    assert ledger.cost_of("ListSessions", 10) == 1
    assert ledger.cost_of("DeletePersonalTime", 3) == 1


def test_quota_is_consumed_and_reported() -> None:
    settings = Settings()
    ledger = QuotaLedger(settings, clock=lambda: 0.0, sleeper=lambda _s: None)

    assert ledger.remaining("ReserveSessions") == 30
    ledger.acquire("ReserveSessions", ledger.cost_of("ReserveSessions", 12))
    assert ledger.remaining("ReserveSessions") == 18
    assert ledger.snapshot()["ReserveSessions"] == 18


def test_quota_blocks_until_the_window_slides() -> None:
    settings = Settings()
    now = {"t": 0.0}
    slept: list[float] = []

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    ledger = QuotaLedger(settings, clock=lambda: now["t"], sleeper=sleeper)
    ledger.acquire("GetSchedule", 60)  # whole minute's allowance
    assert ledger.remaining("GetSchedule") == 0

    ledger.acquire("GetSchedule", 1)
    assert slept, "should have waited for the window to slide"
    assert now["t"] >= 60.0


def test_batch_larger_than_the_quota_is_rejected_not_hung() -> None:
    settings = Settings()
    ledger = QuotaLedger(settings, clock=lambda: 0.0, sleeper=lambda _s: None)
    with pytest.raises(ValueError, match="Split the batch"):
        ledger.acquire("ReserveSessions", 31)


def test_throttle_note_fills_the_window_without_charging_quota() -> None:
    settings = Settings()
    ledger = QuotaLedger(settings, clock=lambda: 0.0, sleeper=lambda _s: None)
    wait = ledger.note_throttled("ListSessions", 17)
    assert wait == 17.0
    assert ledger.remaining("ListSessions") == 0


def test_backoff_is_bounded_and_jittered() -> None:
    import random

    policy = RetryPolicy(base_delay_seconds=0.5, max_delay_seconds=8.0)
    rng = random.Random(1234)
    for attempt in range(1, 8):
        delay = backoff_delay(attempt, policy, rng=rng)
        assert 0.0 <= delay <= policy.max_delay_seconds


def test_backoff_without_jitter_is_exponential() -> None:
    policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=100.0, jitter=False)
    assert backoff_delay(1, policy) == 1.0
    assert backoff_delay(2, policy) == 2.0
    assert backoff_delay(3, policy) == 4.0


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_probe_reports_reachability_without_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"items": [EVENT_PAYLOAD, PUBLIC_EVENT_PAYLOAD]})

    client = make_client(handler, tokens=_always_signed_in())
    report = client.probe()
    assert report["reachable"] is True
    assert report["events_visible"] == 2
    assert report["public_events"] == ["Summit-X"]
    assert report["event"]["requires_sign_in"] is True


def test_probe_reports_unreachable_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    client = make_client(handler, tokens=_always_signed_in())
    report = client.probe()
    assert report["reachable"] is False
    assert "error" in report


# --------------------------------------------------------------------------- #
def _always_signed_in() -> TokenProvider:
    class _Stub(TokenProvider):
        def __init__(self) -> None:
            self._settings = Settings()
            self._cached = TokenSet(
                access_token="test-access-token",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
                refresh_token="r",
            )
            self._store = TokenStore(Settings().token_path)

        def access_token(self) -> str:
            return "test-access-token"

        def peek(self) -> TokenSet | None:
            return self._cached

        def is_signed_in(self) -> bool:
            return True

    return _Stub()
