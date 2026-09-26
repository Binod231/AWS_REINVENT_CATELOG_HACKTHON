"""A ``CatalogPort`` that drives AWS's **remote MCP server** at
``https://api.awsevents.com/mcp``.

This is the second of the API's two front doors, and it is not a thin alias for
the first. Differences that matter, straight from the developer guide:

* **Every** tool call requires sign-in, "even catalog reads" - unlike the REST
  API, where a public event's catalog is readable anonymously. So this client has
  no anonymous mode at all.
* Tools are named for the REST operations (``ListSessions``, ``GetSchedule``,
  ``ReserveSessions``, ...), one tool per operation.
* An operation can be closed while its tool is still listed, so a 409-equivalent
  refusal can arrive from a tool that appears available.
* Quotas are the same as the REST API's, so the same ledger applies.

Transport is streamable HTTP. The MCP Python SDK is an optional dependency
(``pip install 'replan[mcp]'``) and is imported lazily, so the REST path never
pays for it.

Tool results come back as JSON text content, which is handed to the *same*
decoders the REST client uses. That is deliberate: both surfaces return the same
shapes, so sharing the decoders means a bug can only be fixed once, and the
contract suite proves the two agree.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

from replan.adapters.auth import TokenProvider
from replan.adapters.errors import (
    EventsApiError,
    EventsAuthRequiredError,
    EventsNotFoundError,
    EventsOperationClosedError,
)
from replan.adapters.port import ProgressHook
from replan.adapters.throttle import QuotaLedger
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
    PersonalTime,
    Session,
    TimeSpan,
)

MCP_EXTRA_HINT = (
    "The MCP surface needs the Model Context Protocol SDK. "
    "Install it with: uv sync --extra mcp   (or pip install 'replan[mcp]')"
)

# Tool names as published in the developer guide's MCP tool table.
TOOL_LIST_EVENTS = "ListEvents"
TOOL_GET_EVENT = "GetEvent"
TOOL_LIST_SESSIONS = "ListSessions"
TOOL_GET_SESSION = "GetSession"
TOOL_GET_SCHEDULE = "GetSchedule"
TOOL_RESERVE = "ReserveSessions"
TOOL_CANCEL_RESERVATION = "CancelReservation"
TOOL_ASSOCIATE_FAVORITES = "AssociateFavorites"
TOOL_DISASSOCIATE_FAVORITE = "DisassociateFavorite"
TOOL_CREATE_PERSONAL_TIME = "CreatePersonalTime"
TOOL_UPDATE_PERSONAL_TIME = "UpdatePersonalTime"
TOOL_DELETE_PERSONAL_TIME = "DeletePersonalTime"

MAX_PAGES = 500


class McpUnavailableError(EventsApiError):
    """The MCP SDK is not installed, or the server could not be reached."""

    @property
    def remedy(self) -> str:
        return MCP_EXTRA_HINT


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async call from sync code.

    The CLI is synchronous by design - a scheduling tool is a batch job, not a
    server - so the async MCP session is driven from here. Refusing to nest inside
    a running loop keeps the failure obvious instead of mysterious.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    msg = (
        "the MCP client cannot be driven from inside a running event loop; "
        "call the async API directly instead"
    )
    raise McpUnavailableError(msg)


class RemoteMcpCatalogClient:
    """Speaks to AWS's remote MCP server, exposing the same port as REST."""

    surface_name = "live-mcp"

    def __init__(
        self,
        settings: Settings,
        *,
        tokens: TokenProvider | None = None,
        ledger: QuotaLedger | None = None,
    ) -> None:
        self._settings = settings
        self._tokens = tokens or TokenProvider(settings)
        self._ledger = ledger or QuotaLedger(settings)
        self._tool_names: tuple[str, ...] | None = None

    # --- session plumbing ------------------------------------------------- #

    async def _call_async(self, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as error:  # pragma: no cover - optional dependency
            raise McpUnavailableError(MCP_EXTRA_HINT) from error

        # Every MCP tool call is authenticated, including catalog reads. That is
        # the documented difference from the REST surface, and the reason this
        # client has no anonymous mode.
        token = self._tokens.access_token()

        # The transport takes headers via a pre-configured async client rather
        # than a `headers=` argument, so the bearer token is attached there.
        http_client = self._build_http_client({"Authorization": f"Bearer {token}"})

        async with streamable_http_client(
            self._settings.mcp_url, http_client=http_client
        ) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                if self._tool_names is None:
                    listing = await session.list_tools()
                    self._tool_names = tuple(item.name for item in listing.tools)
                result = await session.call_tool(tool, arguments)
                return self._unwrap(tool, result)

    def _build_http_client(self, headers: dict[str, str]) -> Any:
        """Create the async HTTP client the MCP transport will use.

        The SDK ships its own httpx fork; falling back to plain httpx keeps this
        working if that changes again.
        """
        try:
            import httpx2

            return httpx2.AsyncClient(
                headers=headers, timeout=self._settings.request_timeout_seconds
            )
        except ImportError:  # pragma: no cover - depends on SDK packaging
            import httpx

            return httpx.AsyncClient(
                headers=headers, timeout=self._settings.request_timeout_seconds
            )

    def _unwrap(self, tool: str, result: Any) -> Any:
        """Turn an MCP tool result into a payload the shared decoders accept.

        Attribute names are read snake_case first (SDK 2.x) then camelCase (1.x),
        because the SDK renamed them and pinning one spelling would silently treat
        an error result as a success on the other version.
        """
        if self._attr(result, "is_error", "isError"):
            raise self._classify_tool_error(tool, self._text_of(result))

        structured = self._attr(result, "structured_content", "structuredContent")
        if isinstance(structured, dict):
            # Some servers wrap the real payload; unwrap a lone "result" key.
            if set(structured) == {"result"} and isinstance(structured["result"], dict):
                return structured["result"]
            return structured

        text = self._text_of(result)
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"message": text}

    @staticmethod
    def _attr(source: Any, *names: str) -> Any:
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _text_of(result: Any) -> str:
        chunks: list[str] = []
        for item in getattr(result, "content", []) or []:
            value = getattr(item, "text", None)
            if isinstance(value, str):
                chunks.append(value)
        return "\n".join(chunks).strip()

    def _classify_tool_error(self, tool: str, text: str) -> EventsApiError:
        """Map a tool-level error onto the same typed errors REST produces.

        MCP hides the HTTP status, so the message is the only signal. An operation
        can be closed while its tool is still listed, which is exactly the
        reservations-before-8-October case, so that is detected explicitly rather
        than being reported as a generic failure.
        """
        lowered = text.lower()
        if "sign in" in lowered or "unauthor" in lowered or "401" in lowered:
            return EventsAuthRequiredError(text or "sign-in required", operation=tool)
        if "not available" in lowered or "409" in lowered or "closed" in lowered:
            return EventsOperationClosedError(
                text or "operation closed", status=409, operation=tool
            )
        if "not found" in lowered or "404" in lowered:
            return EventsNotFoundError(text or "not found", status=404, operation=tool)
        return EventsApiError(text or f"{tool} failed", operation=tool)

    def _call(self, tool: str, arguments: dict[str, Any], *, quota_cost: int = 1) -> Any:
        self._ledger.acquire(tool, quota_cost)
        return _run(self._call_async(tool, arguments))

    # --- capability discovery --------------------------------------------- #

    def available_tools(self) -> tuple[str, ...]:
        """Ask the server what it exposes. Used by ``replan doctor``."""
        if self._tool_names is None:
            self._call(TOOL_LIST_EVENTS, {})
        return self._tool_names or ()

    # --- events ----------------------------------------------------------- #

    def list_events(self, *, include_past: bool = False) -> tuple[Event, ...]:
        payload = self._call(TOOL_LIST_EVENTS, {"includePast": True} if include_past else {})
        return decode_events(payload if isinstance(payload, dict) else {})

    def get_event(self, event_id: str) -> Event:
        payload = self._call(TOOL_GET_EVENT, {"eventId": event_id})
        body = payload if isinstance(payload, dict) else {}
        if isinstance(body.get("event"), dict):
            body = body["event"]
        if not body.get("eventId"):
            msg = f"event {event_id!r} was not returned by {TOOL_GET_EVENT}"
            raise EventsNotFoundError(msg, status=404, operation=TOOL_GET_EVENT)
        return decode_event(body)

    # --- catalog ---------------------------------------------------------- #

    def list_sessions(
        self,
        event_id: str,
        *,
        include_abstracts: bool = True,
        locale: str | None = None,
        progress: ProgressHook | None = None,
    ) -> Catalog:
        event: Event | None = None
        try:
            event = self.get_event(event_id)
        except EventsApiError:
            event = None

        collected: list[Session] = []
        total: int | None = None
        token: str | None = None
        pages = 0

        while True:
            arguments: dict[str, Any] = {"eventId": event_id}
            if not include_abstracts:
                arguments["includeAbstracts"] = False
            if locale:
                arguments["locale"] = locale
            if token:
                arguments["nextToken"] = token

            payload = self._call(TOOL_LIST_SESSIONS, arguments)
            page = decode_catalog_page(
                payload if isinstance(payload, dict) else {},
                fallback_timezone=event.timezone_name if event else None,
            )
            collected.extend(page.sessions)
            if page.total_count is not None:
                total = page.total_count
            pages += 1
            if progress is not None:
                progress(pages, len(collected), total)

            token = page.next_token
            if not token:
                break
            if pages >= MAX_PAGES:
                msg = f"{TOOL_LIST_SESSIONS} kept returning a nextToken after {MAX_PAGES} pages"
                raise EventsApiError(msg, operation=TOOL_LIST_SESSIONS)

        return build_catalog(
            event_id=event_id,
            sessions=tuple(collected),
            event=event,
            source=self.surface_name,
            total_count=total,
            locale=locale or self._settings.locale,
        )

    def get_session(self, event_id: str, session_id: str, *, locale: str | None = None) -> Session:
        arguments: dict[str, Any] = {"eventId": event_id, "sessionId": session_id}
        if locale:
            arguments["locale"] = locale
        payload = self._call(TOOL_GET_SESSION, arguments)
        body = payload if isinstance(payload, dict) else {}
        if isinstance(body.get("session"), dict):
            body = body["session"]
        decoded = decode_session(body)
        if decoded is None:
            msg = f"session {session_id!r} was not returned by {TOOL_GET_SESSION}"
            raise EventsNotFoundError(msg, status=404, operation=TOOL_GET_SESSION)
        return decoded

    # --- schedule --------------------------------------------------------- #

    def get_schedule(self, event_id: str) -> AttendeeSchedule:
        payload = self._call(TOOL_GET_SCHEDULE, {"eventId": event_id})
        return decode_schedule(payload if isinstance(payload, dict) else {}, event_id=event_id)

    def reserve_sessions(self, event_id: str, session_ids: list[str]) -> BulkResult:
        if not session_ids:
            return BulkResult()
        payload = self._call(
            TOOL_RESERVE,
            {"eventId": event_id, "sessionIds": session_ids},
            quota_cost=self._ledger.cost_of(TOOL_RESERVE, len(session_ids)),
        )
        return decode_bulk_result(payload if isinstance(payload, dict) else {})

    def cancel_reservation(self, event_id: str, session_id: str) -> bool:
        try:
            self._call(TOOL_CANCEL_RESERVATION, {"eventId": event_id, "sessionId": session_id})
        except EventsNotFoundError:
            return False  # already gone, same as the REST 404
        return True

    def associate_favorites(self, event_id: str, session_ids: list[str]) -> BulkResult:
        if not session_ids:
            return BulkResult()
        payload = self._call(
            TOOL_ASSOCIATE_FAVORITES,
            {"eventId": event_id, "sessionIds": session_ids},
            quota_cost=self._ledger.cost_of(TOOL_ASSOCIATE_FAVORITES, len(session_ids)),
        )
        return decode_bulk_result(payload if isinstance(payload, dict) else {})

    def disassociate_favorite(self, event_id: str, session_id: str) -> bool:
        try:
            self._call(TOOL_DISASSOCIATE_FAVORITE, {"eventId": event_id, "sessionId": session_id})
        except EventsNotFoundError:
            return False
        return True

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
        self._call(TOOL_CREATE_PERSONAL_TIME, {"eventId": event_id, **body})

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
        self._call(
            TOOL_UPDATE_PERSONAL_TIME,
            {"eventId": event_id, "personalTimeId": personal_time_id, **body},
        )

    def delete_personal_time(self, event_id: str, personal_time_id: str) -> bool:
        try:
            self._call(
                TOOL_DELETE_PERSONAL_TIME,
                {"eventId": event_id, "personalTimeId": personal_time_id},
            )
        except EventsNotFoundError:
            # Documented as idempotent on the REST surface; keep that guarantee.
            return True
        return True

    def get_personal_time(self, event_id: str, personal_time_id: str) -> PersonalTime | None:
        return self.get_schedule(event_id).block_by_id(personal_time_id)

    # --- diagnostics ------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        report: dict[str, Any] = {"mcp_url": self._settings.mcp_url}
        if not self._tokens.is_signed_in():
            report["reachable"] = False
            report["signed_in"] = False
            report["error"] = (
                "the MCP server requires sign-in for every call, including catalog "
                "reads. Run `replan login` first."
            )
            return report
        try:
            events = self.list_events()
        except EventsApiError as error:
            report["reachable"] = False
            report["signed_in"] = True
            report["error"] = str(error)
            return report
        report["reachable"] = True
        report["signed_in"] = True
        report["events_visible"] = len(events)
        report["tools"] = list(self.available_tools())
        return report


__all__ = [
    "MAX_PAGES",
    "MCP_EXTRA_HINT",
    "McpUnavailableError",
    "RemoteMcpCatalogClient",
]
