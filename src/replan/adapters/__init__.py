"""Adapters: the only layer that knows what the AWS Events API looks like.

Everything AWS-specific is confined here - JSON field names, the OAuth 2.0 +
PKCE sign-in flow, pagination tokens, per-operation quotas, the naive-UTC
personal-time format, and the partial-success semantics of the bulk operations.
The decision core imports :mod:`replan.domain` and never touches a payload.
"""

from __future__ import annotations

from replan.adapters.errors import (
    EventsApiError,
    EventsAuthRequiredError,
    EventsEdgeRefusedError,
    EventsNotFoundError,
    EventsNotRegisteredError,
    EventsOperationClosedError,
    EventsServerError,
    EventsThrottledError,
    EventsValidationError,
    classify_response,
)
from replan.adapters.wire import (
    decode_bulk_result,
    decode_catalog_page,
    decode_event,
    decode_schedule,
    decode_session,
    encode_personal_time,
)

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
    "decode_bulk_result",
    "decode_catalog_page",
    "decode_event",
    "decode_schedule",
    "decode_session",
    "encode_personal_time",
]
