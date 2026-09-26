"""The live REST client for ``https://api.awsevents.com/v1``.

This is the real integration. What it handles, and why each piece exists:

* **Pagination that cannot stop early.** The docs are blunt: "page sizes can
  vary, and a short page does not mean the last page. Stop when ``nextToken`` is
  absent, never when a page looks short. A client that stops early silently
  misses sessions." The walk follows tokens only, and the resulting
  :class:`~replan.domain.Catalog` carries ``totalCount`` so completeness is
  checkable rather than assumed.
* **Quota-aware pacing** via :class:`~replan.adapters.throttle.QuotaLedger`,
  charging per session for the two bulk operations that bill that way.
* **Retry that knows which failures are retryable.** 429 honours
  ``Retry-After``; 500/503 back off with full jitter; 400/403-registration/404
  never retry, because retrying them cannot succeed.
* **Auth only where auth is needed.** ``ListEvents`` and ``GetEvent`` never send
  a token, and a public event's catalog is read anonymously, so the tool is
  useful before you have signed in at all.
* **Defensive body parsing.** "Do not assume the body is JSON" - the edge can
  return HTML or nothing, and that distinction is how the two meanings of 403 are
  told apart.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Self

import httpx

from replan.adapters.auth import NotSignedInError, TokenProvider
from replan.adapters.errors import (
    EventsApiError,
    EventsNotFoundError,
    EventsThrottledError,
    classify_response,
)
from replan.adapters.port import ProgressHook
from replan.adapters.throttle import QuotaLedger, backoff_delay
from replan.adapters.wire import (
    build_catalog,
    decode_bulk_result,
    decode_catalog_page,
    decode_event,
    decode_events,
    decode_schedule,
    decode_session,
    encode_personal_time,
)
from replan.config import Settings
from replan.domain import (
    AttendeeSchedule,
    BulkResult,
    Catalog,
    Event,
    Session,
    TimeSpan,
)

USER_AGENT = "replan/0.1 (+https://github.com/Binod231/AWS_REINVENT_CATELOG_HACKTHON)"

# A guard against a server that kept handing out tokens. The largest real catalog
# is a few thousand sessions at ~100 per page, so this is orders of magnitude of
# headroom while still terminating.
MAX_PAGES = 500


class RestCatalogClient:
    """Talks to the live AWS Events REST API."""

    surface_name = "live-rest"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        tokens: TokenProvider | None = None,
        ledger: QuotaLedger | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._tokens = tokens or TokenProvider(settings)
        self._ledger = ledger or QuotaLedger(settings)
        self._sleep = sleeper
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=False,
        )

    # --- lifecycle -------------------------------------------------------- #

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def quota_snapshot(self) -> dict[str, int]:
        return self._ledger.snapshot()

    @property
    def tokens(self) -> TokenProvider:
        return self._tokens

    # --- request plumbing ------------------------------------------------- #

    def _headers(self, *, authenticated: bool) -> dict[str, str]:
        if not authenticated:
            return {}
        return {"Authorization": f"Bearer {self._tokens.access_token()}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        authenticated: bool,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        quota_cost: int = 1,
        allow_404: bool = False,
    ) -> tuple[int, Any]:
        """Issue one request, with quota pacing and typed retries.

        Returns ``(status, payload)``. ``payload`` is ``None`` when the body was
        absent or not JSON, which the error classifier needs to tell an
        edge refusal apart from a registration problem.
        """
        url = self._settings.url(path)
        policy = self._settings.retry
        attempt = 0
        last_error: EventsApiError | None = None

        while attempt < policy.max_attempts:
            attempt += 1
            self._ledger.acquire(operation, quota_cost)

            try:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=self._headers(authenticated=authenticated),
                )
            except httpx.HTTPError as transport_error:
                # A dropped connection on a *write* is the dangerous case: we
                # cannot know whether it landed, and the API has no idempotency
                # key. Surface it rather than blindly re-sending.
                if method.upper() != "GET":
                    msg = (
                        f"{operation} failed in transport and its outcome is unknown. "
                        "Reconcile against GetSchedule before retrying; the API has "
                        "no idempotency key."
                    )
                    raise EventsApiError(msg, operation=operation) from transport_error
                last_error = EventsApiError(
                    f"transport error: {transport_error}", operation=operation
                )
                if attempt >= policy.max_attempts:
                    raise last_error from transport_error
                self._sleep(backoff_delay(attempt, policy, rng=self._rng))
                continue

            body_text = response.text or ""
            payload: Any | None
            try:
                payload = response.json() if body_text.strip() else None
            except ValueError:
                payload = None

            error = classify_response(
                status=response.status_code,
                body_text=body_text,
                payload=payload,
                headers=dict(response.headers),
                operation=operation,
            )

            if error is None:
                return response.status_code, payload

            if isinstance(error, EventsNotFoundError) and allow_404:
                return response.status_code, payload

            if isinstance(error, EventsThrottledError):
                wait = self._ledger.note_throttled(operation, error.retry_after_seconds)
                if attempt >= policy.max_attempts:
                    raise error
                self._sleep(wait)
                last_error = error
                continue

            if error.is_retryable and attempt < policy.max_attempts:
                # A 401 is retryable exactly once, after a refresh, which
                # access_token() performs on the next call.
                last_error = error
                self._sleep(backoff_delay(attempt, policy, rng=self._rng))
                continue

            raise error

        if last_error is not None:
            raise last_error
        msg = f"{operation} exhausted retries without a result"
        raise EventsApiError(msg, operation=operation)

    # --- events ----------------------------------------------------------- #

    def list_events(self, *, include_past: bool = False) -> tuple[Event, ...]:
        params = {"includePast": "true"} if include_past else None
        _, payload = self._request(
            "GET",
            "/events",
            operation="ListEvents",
            authenticated=False,
            params=params,
        )
        return decode_events(payload or {})

    def get_event(self, event_id: str) -> Event:
        _, payload = self._request(
            "GET",
            f"/events/{event_id}",
            operation="GetEvent",
            authenticated=False,
        )
        body = payload or {}
        if isinstance(body.get("event"), dict):
            body = body["event"]
        return decode_event(body)

    # --- catalog ---------------------------------------------------------- #

    def _needs_auth_for_catalog(self, event_id: str) -> bool:
        """Only send a token when the event actually requires one.

        Reading a public catalog anonymously is a genuine feature: it means the
        tool works, and the test suite runs, with no credentials at all.
        """
        try:
            event = self.get_event(event_id)
        except EventsApiError:
            return True
        return event.authentication_required

    def iter_session_pages(
        self,
        event_id: str,
        *,
        include_abstracts: bool = True,
        locale: str | None = None,
        authenticated: bool | None = None,
    ) -> Iterator[tuple[tuple[Session, ...], str | None, int | None, int]]:
        """Yield decoded pages, following ``nextToken`` to the end."""
        needs_auth = (
            self._needs_auth_for_catalog(event_id) if authenticated is None else authenticated
        )
        fallback_zone: str | None = None
        try:
            fallback_zone = self.get_event(event_id).timezone_name
        except EventsApiError:
            fallback_zone = None

        token: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {}
            if not include_abstracts:
                params["includeAbstracts"] = "false"
            if locale:
                params["locale"] = locale
            if token:
                params["nextToken"] = token

            _, payload = self._request(
                "GET",
                f"/events/{event_id}/sessions",
                operation="ListSessions",
                authenticated=needs_auth,
                params=params or None,
            )
            page = decode_catalog_page(payload or {}, fallback_timezone=fallback_zone)
            yield (page.sessions, page.next_token, page.total_count, page.skipped)

            pages += 1
            token = page.next_token
            # The single most important line in this client: stop on an absent
            # token, never on a short page.
            if not token:
                return
            if pages >= MAX_PAGES:
                msg = (
                    f"ListSessions still returned a nextToken after {MAX_PAGES} pages; "
                    "refusing to loop forever"
                )
                raise EventsApiError(msg, operation="ListSessions")

    def list_sessions(
        self,
        event_id: str,
        *,
        include_abstracts: bool = True,
        locale: str | None = None,
        progress: ProgressHook | None = None,
    ) -> Catalog:
        collected: list[Session] = []
        total: int | None = None
        skipped = 0

        pages = self.iter_session_pages(
            event_id, include_abstracts=include_abstracts, locale=locale
        )
        for page_number, (sessions, _token, page_total, page_skipped) in enumerate(pages, start=1):
            collected.extend(sessions)
            skipped += page_skipped
            if page_total is not None:
                total = page_total
            if progress is not None:
                progress(page_number, len(collected), total)

        event: Event | None = None
        try:
            event = self.get_event(event_id)
        except EventsApiError:
            event = None

        catalog = build_catalog(
            event_id=event_id,
            sessions=tuple(collected),
            event=event,
            source=self.surface_name,
            total_count=total,
            locale=locale or self._settings.locale,
        )
        if skipped:
            # Surfaced rather than swallowed: a silent skip is a silently wrong plan.
            catalog = catalog.model_copy(
                update={"source": f"{self.surface_name} ({skipped} skipped)"}
            )
        return catalog

    def get_session(self, event_id: str, session_id: str, *, locale: str | None = None) -> Session:
        params = {"locale": locale} if locale else None
        _, payload = self._request(
            "GET",
            f"/events/{event_id}/sessions/{session_id}",
            operation="GetSession",
            authenticated=self._needs_auth_for_catalog(event_id),
            params=params,
        )
        body = payload or {}
        if isinstance(body.get("session"), dict):
            body = body["session"]
        decoded = decode_session(body)
        if decoded is None:
            msg = f"GetSession returned a session without an id or title: {session_id!r}"
            raise EventsApiError(msg, operation="GetSession")
        return decoded

    # --- schedule --------------------------------------------------------- #

    def get_schedule(self, event_id: str) -> AttendeeSchedule:
        _, payload = self._request(
            "GET",
            f"/events/{event_id}/schedule",
            operation="GetSchedule",
            authenticated=True,
        )
        return decode_schedule(payload or {}, event_id=event_id)

    def reserve_sessions(self, event_id: str, session_ids: list[str]) -> BulkResult:
        if not session_ids:
            return BulkResult()
        _, payload = self._request(
            "POST",
            f"/events/{event_id}/reservations",
            operation="ReserveSessions",
            authenticated=True,
            json_body={"sessionIds": session_ids},
            quota_cost=self._ledger.cost_of("ReserveSessions", len(session_ids)),
        )
        return decode_bulk_result(payload or {})

    def cancel_reservation(self, event_id: str, session_id: str) -> bool:
        status, _ = self._request(
            "DELETE",
            f"/events/{event_id}/reservations/{session_id}",
            operation="CancelReservation",
            authenticated=True,
            allow_404=True,
        )
        # 404 means it was not reserved. Documented as safe to treat as done.
        return status != 404

    def associate_favorites(self, event_id: str, session_ids: list[str]) -> BulkResult:
        if not session_ids:
            return BulkResult()
        _, payload = self._request(
            "POST",
            f"/events/{event_id}/favorites",
            operation="AssociateFavorites",
            authenticated=True,
            json_body={"sessionIds": session_ids},
            quota_cost=self._ledger.cost_of("AssociateFavorites", len(session_ids)),
        )
        return decode_bulk_result(payload or {})

    def disassociate_favorite(self, event_id: str, session_id: str) -> bool:
        status, _ = self._request(
            "DELETE",
            f"/events/{event_id}/favorites/{session_id}",
            operation="DisassociateFavorite",
            authenticated=True,
            allow_404=True,
        )
        return status != 404

    # --- personal time ---------------------------------------------------- #

    def create_personal_time(
        self,
        event_id: str,
        *,
        span: TimeSpan,
        title: str,
        description: str,
        location: str | None = None,
    ) -> None:
        body = encode_personal_time(
            span=span, title=title, description=description, location=location
        )
        self._request(
            "POST",
            f"/events/{event_id}/personal-time",
            operation="CreatePersonalTime",
            authenticated=True,
            json_body=body,
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
        body = encode_personal_time(
            span=span, title=title, description=description, location=location
        )
        self._request(
            "PUT",
            f"/events/{event_id}/personal-time/{personal_time_id}",
            operation="UpdatePersonalTime",
            authenticated=True,
            json_body=body,
        )

    def delete_personal_time(self, event_id: str, personal_time_id: str) -> bool:
        status, _ = self._request(
            "DELETE",
            f"/events/{event_id}/personal-time/{personal_time_id}",
            operation="DeletePersonalTime",
            authenticated=True,
            allow_404=True,
        )
        # Documented as idempotent: removing one already gone still succeeds.
        return status != 404

    # --- diagnostics ------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        """Cheap reachability check for ``replan doctor``. Never sends a token."""
        report: dict[str, Any] = {"api_root": self._settings.api_root}
        try:
            events = self.list_events()
        except EventsApiError as error:
            report["reachable"] = False
            report["error"] = str(error)
            return report
        report["reachable"] = True
        report["events_visible"] = len(events)
        report["public_events"] = [
            event.event_id for event in events if not event.authentication_required
        ]
        target = next(
            (event for event in events if event.event_id == self._settings.event_id), None
        )
        if target is not None:
            report["event"] = {
                "event_id": target.event_id,
                "name": target.name,
                "requires_sign_in": target.authentication_required,
                "timezone": target.timezone_name,
                "days": [day.isoformat() for day in target.days],
            }
        try:
            report["signed_in"] = self._tokens.is_signed_in()
            current = self._tokens.peek()
            if current is not None:
                report["token"] = current.redacted()
                report["signed_in_as"] = current.signed_in_as
        except NotSignedInError:
            report["signed_in"] = False
        return report


__all__ = ["MAX_PAGES", "USER_AGENT", "ProgressHook", "RestCatalogClient"]
