"""Typed errors mapped from the API's documented status codes.

The published error table is unusually specific about *what to do*, and the
distinctions are load-bearing rather than cosmetic:

* ``401`` - sign in or refresh, then retry once.
* ``403`` - **two different causes told apart by the body.** JSON body: signed in
  but not registered for this event, so retrying never helps. No body: the edge
  refused us, so slow down and retry.
* ``404`` - on a removal, treat as already done.
* ``409`` - the operation is closed, not the request wrong. Reservations return
  this until reserved seating opens through the API on 8 October 2026.
* ``429`` - wait ``Retry-After`` seconds. A refused request spends no quota, so a
  batch too large for the remaining allowance can be shrunk and retried at once.
* ``500``/``503`` - safe to retry a read with backoff; a write must be reconciled
  against ``GetSchedule`` first, because the API has no idempotency key.

The docs also warn that "not every response comes from the API itself. The edge
can refuse a request with no body at all, or answer with an HTML error of its
own, so do not assume the body is JSON." :func:`classify_response` therefore
never assumes a parseable body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EventsApiError(Exception):
    """Base class for every failure returned by the Events API."""

    message: str
    status: int | None = None
    operation: str | None = None
    retry_after_seconds: int | None = None
    body_snippet: str | None = None

    def __str__(self) -> str:
        bits = [self.message]
        if self.status is not None:
            bits.append(f"(HTTP {self.status})")
        if self.operation:
            bits.append(f"during {self.operation}")
        return " ".join(bits)

    @property
    def is_retryable(self) -> bool:
        """Whether retrying this exact request could succeed without changes."""
        return False

    @property
    def remedy(self) -> str:
        """Plain-language next step, surfaced directly to the CLI user."""
        return "Inspect the message and adjust the request."


class EventsValidationError(EventsApiError):
    """400. A field is missing, too long, or malformed - including a foreign nextToken."""

    @property
    def remedy(self) -> str:
        return "Do not retry; fix the request. The message names the field."


class EventsAuthRequiredError(EventsApiError):
    """401. No valid token was sent, or it expired."""

    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def remedy(self) -> str:
        return "Run `replan login` to sign in with your AWS Builder ID, then retry."


class EventsNotRegisteredError(EventsApiError):
    """403 with a JSON body. Signed in, but not registered for this event."""

    @property
    def remedy(self) -> str:
        return (
            "Register for the event on its own registration site first. "
            "Signing in again will not help."
        )


class EventsEdgeRefusedError(EventsApiError):
    """403 with no body. The edge, not the API, refused this address."""

    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def remedy(self) -> str:
        return "Slow down and retry; the refusal came from the edge, not the API."


class EventsNotFoundError(EventsApiError):
    """404. The event, session, or personal time entry does not exist."""

    @property
    def remedy(self) -> str:
        return "Do not retry. For a removal, treat this as already done."


class EventsOperationClosedError(EventsApiError):
    """409. The operation is closed for now; the request itself was fine."""

    @property
    def remedy(self) -> str:
        return (
            "The operation is temporarily closed for every caller. Reservations open "
            "through the API on 8 October 2026; browsing and favorites keep working."
        )


class EventsThrottledError(EventsApiError):
    """429. This operation's per-minute quota is spent."""

    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def remedy(self) -> str:
        wait = self.retry_after_seconds
        if wait is None:
            return "Wait for the quota window to reset, then retry."
        return f"Wait {wait}s for the quota to reset, then retry."


class EventsServerError(EventsApiError):
    """500 or 503."""

    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def remedy(self) -> str:
        return (
            "Retry a read with backoff. Before retrying a write, reconcile against "
            "GetSchedule: the API has no idempotency key."
        )


_DEFAULT_MESSAGES: dict[int, str] = {
    400: "The API could not accept the request as sent",
    401: "Sign in to continue",
    403: "Access refused",
    404: "Not found",
    409: "That operation is not currently available",
    429: "Rate limit reached for this operation",
    500: "The Events service hit an internal error",
    503: "The Events service is temporarily unavailable",
}


def _extract_message(payload: Any, status: int) -> str:
    if isinstance(payload, dict):
        candidate = payload.get("message")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return _DEFAULT_MESSAGES.get(status, f"Unexpected response (HTTP {status})")


def _parse_retry_after(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        value = int(float(raw.strip()))
    except (TypeError, ValueError):
        return None
    return max(value, 0)


def classify_response(
    *,
    status: int,
    body_text: str,
    payload: Any,
    headers: dict[str, str] | None = None,
    operation: str | None = None,
) -> EventsApiError | None:
    """Map a raw HTTP response onto a typed error, or ``None`` when it succeeded.

    ``payload`` is the parsed JSON body, or ``None`` when the body was absent or
    not JSON. That distinction is what separates the two meanings of 403, so it
    is a parameter rather than something rediscovered here.
    """
    if 200 <= status < 300:
        return None

    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    retry_after = _parse_retry_after(lowered.get("retry-after"))
    message = _extract_message(payload, status)
    snippet = body_text[:400] if body_text else None

    shared: dict[str, Any] = {
        "status": status,
        "operation": operation,
        "retry_after_seconds": retry_after,
        "body_snippet": snippet,
    }

    if status == 400:
        return EventsValidationError(message, **shared)
    if status == 401:
        return EventsAuthRequiredError(message, **shared)
    if status == 403:
        # The documented discriminator: a JSON body means "not registered",
        # an empty or non-JSON body means the edge refused us.
        if payload is None:
            return EventsEdgeRefusedError("Refused at the edge rather than by the API", **shared)
        return EventsNotRegisteredError(message, **shared)
    if status == 404:
        return EventsNotFoundError(message, **shared)
    if status == 409:
        return EventsOperationClosedError(message, **shared)
    if status == 429:
        return EventsThrottledError(message, **shared)
    if status in (500, 503):
        return EventsServerError(message, **shared)
    return EventsApiError(message, **shared)


__all__ = [
    "EventsApiError",
    "EventsAuthRequiredError",
    "EventsEdgeRefusedError",
    "EventsNotFoundError",
    "EventsNotRegisteredError",
    "EventsOperationClosedError",
    "EventsServerError",
    "EventsThrottledError",
    "EventsValidationError",
    "classify_response",
]
