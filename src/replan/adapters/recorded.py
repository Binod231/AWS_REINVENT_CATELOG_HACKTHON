"""A ``CatalogPort`` backed by committed recordings of real API responses.

Two jobs:

* **Credential-free demos and CI.** ``tests/fixtures/recorded/`` holds genuine
  payloads captured from events the API serves anonymously, so the whole product
  can be exercised by someone who has no re:Invent registration and no AWS
  account. The data is real; only the transport is not.
* **Writes with nothing at stake.** Schedule mutations are applied to in-memory
  state that reproduces the API's actual semantics, including the awkward parts:
  partial success, "already favorited" surfacing as a *failure*, 404-means-done
  on single removals, and reservations refusing with 409 until reserved seating
  opens through the API on 8 October 2026.

That last point is the reason this class models refusals rather than just
succeeding: code tested only against a happy path would fall over on the real
API's partial-failure responses.
"""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from replan.adapters.errors import (
    EventsNotFoundError,
    EventsOperationClosedError,
)
from replan.adapters.port import ProgressHook
from replan.adapters.wire import (
    build_catalog,
    decode_catalog_page,
    decode_event,
    decode_events,
    decode_session,
)
from replan.config import RESERVATIONS_OPEN_NOTE
from replan.domain import (
    REPLAN_BLOCK_MARKER,
    AttendeeSchedule,
    BulkFailure,
    BulkFailureCode,
    BulkResult,
    Catalog,
    Event,
    PersonalTime,
    Session,
    TimeSpan,
    now_utc,
    validate_personal_time_span,
)

DEFAULT_RECORDING_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "recorded"


class RecordedCatalogClient:
    """Replays recorded reads; simulates writes faithfully in memory."""

    surface_name = "recorded"

    def __init__(
        self,
        recording_dir: Path | None = None,
        *,
        reservations_open: bool = False,
        reference_now: datetime | None = None,
    ) -> None:
        self._dir = recording_dir or DEFAULT_RECORDING_DIR
        if not self._dir.exists():
            msg = (
                f"no recordings at {self._dir}. "
                "Run `uv run python scripts/record_fixtures.py` to capture them."
            )
            raise FileNotFoundError(msg)
        self._reservations_open = reservations_open
        self._now = reference_now or now_utc()

        self._reserved: set[str] = set()
        self._favorites: set[str] = set()
        self._blocks: dict[str, PersonalTime] = {}
        self._ids = itertools.count(1)
        self._catalog_cache: dict[str, Catalog] = {}

    # --- recording access ------------------------------------------------- #

    def _load(self, name: str) -> Any:
        path = self._dir / name
        if not path.exists():
            msg = f"recording {name!r} is missing from {self._dir}"
            raise EventsNotFoundError(msg, status=404)
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def recorded_event_id(self) -> str:
        return str(self._load("public_event.json")["eventId"])

    # --- reads ------------------------------------------------------------ #

    def list_events(self, *, include_past: bool = False) -> tuple[Event, ...]:
        # The recording was captured without includePast, so both answers are the
        # same set. Accepting the flag keeps the port signature honest.
        _ = include_past
        return decode_events(self._load("list_events.json"))

    def get_event(self, event_id: str) -> Event:
        for event in self.list_events():
            if event.event_id == event_id:
                return event
        payload = self._load("get_event.json")
        body = payload.get("event", payload)
        decoded = decode_event(body)
        if decoded.event_id == event_id:
            return decoded
        msg = f"event {event_id!r} is not in the recordings"
        raise EventsNotFoundError(msg, status=404)

    def list_sessions(
        self,
        event_id: str,
        *,
        include_abstracts: bool = True,
        locale: str | None = None,
        progress: ProgressHook | None = None,
    ) -> Catalog:
        cache_key = f"{event_id}:{include_abstracts}:{locale}"
        cached = self._catalog_cache.get(cache_key)
        if cached is not None:
            return cached

        recorded_id = self.recorded_event_id
        if event_id != recorded_id:
            msg = (
                f"only {recorded_id!r} is recorded; asked for {event_id!r}. "
                "Use --surface rest to read a different event live."
            )
            raise EventsNotFoundError(msg, status=404)

        event = self.get_event(event_id)
        page = decode_catalog_page(
            self._load("list_sessions_page.json"),
            fallback_timezone=event.timezone_name,
        )
        sessions = page.sessions
        if not include_abstracts:
            sessions = tuple(session.model_copy(update={"abstract": None}) for session in sessions)

        if progress is not None:
            progress(1, len(sessions), page.total_count)

        catalog = build_catalog(
            event_id=event_id,
            sessions=sessions,
            event=event,
            source=self.surface_name,
            total_count=page.total_count,
            locale=locale,
        )
        self._catalog_cache[cache_key] = catalog
        return catalog

    def get_session(self, event_id: str, session_id: str, *, locale: str | None = None) -> Session:
        catalog = self.list_sessions(event_id, locale=locale)
        found = catalog.by_id(session_id)
        if found is not None:
            return found
        payload = self._load("get_session.json")
        decoded = decode_session(payload.get("session", payload))
        if decoded is not None and decoded.session_id == session_id:
            return decoded
        msg = f"session {session_id!r} is not in the recordings"
        raise EventsNotFoundError(msg, status=404)

    # --- schedule --------------------------------------------------------- #

    def get_schedule(self, event_id: str) -> AttendeeSchedule:
        return AttendeeSchedule(
            event_id=event_id,
            reserved=tuple(sorted(self._reserved)),
            favorites=tuple(sorted(self._favorites)),
            personal_time=tuple(
                self._blocks[key] for key in sorted(self._blocks, key=self._block_sort_key)
            ),
            fetched_at=self._now,
        )

    def _block_sort_key(self, key: str) -> tuple[datetime, str]:
        return (self._blocks[key].span.start, key)

    def _known_session(self, event_id: str, session_id: str) -> Session | None:
        try:
            return self.list_sessions(event_id).by_id(session_id)
        except EventsNotFoundError:
            return None

    def reserve_sessions(self, event_id: str, session_ids: list[str]) -> BulkResult:
        if not self._reservations_open:
            # Exactly what the live API does today: a 409 for every caller.
            raise EventsOperationClosedError(
                RESERVATIONS_OPEN_NOTE, status=409, operation="ReserveSessions"
            )
        if not session_ids:
            return BulkResult()

        catalog = self.list_sessions(event_id)
        successful: list[str] = []
        failed: list[BulkFailure] = []

        for session_id in session_ids:
            session = catalog.by_id(session_id)
            if session is None:
                failed.append(BulkFailure(session_id=session_id, code=BulkFailureCode.OTHER))
                continue
            if session_id in self._reserved:
                failed.append(
                    BulkFailure(session_id=session_id, code=BulkFailureCode.ALREADY_SCHEDULED)
                )
                continue
            if session.is_reservable is False:
                failed.append(
                    BulkFailure(session_id=session_id, code=BulkFailureCode.SESSION_NOT_RESERVABLE)
                )
                continue
            if session.is_full:
                failed.append(BulkFailure(session_id=session_id, code=BulkFailureCode.SESSION_FULL))
                continue
            clashes = self._clashes_with_reserved(catalog, session)
            if clashes:
                failed.append(
                    BulkFailure(
                        session_id=session_id,
                        code=BulkFailureCode.SCHEDULE_CONFLICT,
                        conflicts_with=clashes,
                    )
                )
                continue
            self._reserved.add(session_id)
            successful.append(session_id)

        return BulkResult(successful=tuple(successful), failed=tuple(failed))

    def _clashes_with_reserved(self, catalog: Catalog, session: Session) -> tuple[str, ...]:
        if session.span is None:
            return ()
        clashing: list[str] = []
        for held_id in sorted(self._reserved):
            held = catalog.by_id(held_id)
            if held is not None and held.span is not None and held.span.overlaps(session.span):
                clashing.append(held_id)
        return tuple(clashing)

    def cancel_reservation(self, event_id: str, session_id: str) -> bool:
        if not self._reservations_open:
            raise EventsOperationClosedError(
                RESERVATIONS_OPEN_NOTE, status=409, operation="CancelReservation"
            )
        _ = event_id
        if session_id not in self._reserved:
            return False  # the API's 404: already gone
        self._reserved.discard(session_id)
        return True

    def associate_favorites(self, event_id: str, session_ids: list[str]) -> BulkResult:
        if not session_ids:
            return BulkResult()
        catalog = self.list_sessions(event_id)
        successful: list[str] = []
        failed: list[BulkFailure] = []
        for session_id in session_ids:
            if catalog.by_id(session_id) is None:
                failed.append(BulkFailure(session_id=session_id, code=BulkFailureCode.OTHER))
                continue
            if session_id in self._favorites:
                # The real API reports this as a failure, which is why
                # re-sending a bulk request is not a safe retry.
                failed.append(
                    BulkFailure(session_id=session_id, code=BulkFailureCode.ALREADY_FAVORITED)
                )
                continue
            self._favorites.add(session_id)
            successful.append(session_id)
        return BulkResult(successful=tuple(successful), failed=tuple(failed))

    def disassociate_favorite(self, event_id: str, session_id: str) -> bool:
        _ = event_id
        if session_id not in self._favorites:
            return False  # 404: not favorited
        self._favorites.discard(session_id)
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
        _ = event_id
        validate_personal_time_span(span)
        entry_id = f"pt-{next(self._ids)}"
        self._blocks[entry_id] = PersonalTime(
            personal_time_id=entry_id,
            title=title,
            description=description or title,
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
        _ = event_id
        if personal_time_id not in self._blocks:
            msg = f"personal time {personal_time_id!r} does not exist"
            raise EventsNotFoundError(msg, status=404)
        validate_personal_time_span(span)
        # A full replacement, as documented: omitted optional fields are cleared.
        self._blocks[personal_time_id] = PersonalTime(
            personal_time_id=personal_time_id,
            title=title,
            description=description or title,
            span=TimeSpan(span.start.astimezone(UTC), span.end.astimezone(UTC)),
            location=location,
        )

    def delete_personal_time(self, event_id: str, personal_time_id: str) -> bool:
        _ = event_id
        existed = personal_time_id in self._blocks
        self._blocks.pop(personal_time_id, None)
        # Documented as idempotent, so this returns True either way.
        return existed or True

    def get_personal_time(self, event_id: str, personal_time_id: str) -> PersonalTime | None:
        _ = event_id
        return self._blocks.get(personal_time_id)

    # --- test seams ------------------------------------------------------- #

    def seed(
        self,
        *,
        reserved: set[str] | None = None,
        favorites: set[str] | None = None,
        blocks: dict[str, PersonalTime] | None = None,
    ) -> None:
        """Preload upstream state, so diff and undo logic can be exercised."""
        if reserved is not None:
            self._reserved = set(reserved)
        if favorites is not None:
            self._favorites = set(favorites)
        if blocks is not None:
            self._blocks = dict(blocks)

    def open_reservations(self) -> None:
        """Simulate 8 October 2026, when ReserveSessions starts answering."""
        self._reservations_open = True

    def make_replan_block(
        self, *, span: TimeSpan, title: str, entry_id: str | None = None
    ) -> PersonalTime:
        key = entry_id or f"pt-{next(self._ids)}"
        block = PersonalTime(
            personal_time_id=key,
            title=title,
            description=f"Held by replan {REPLAN_BLOCK_MARKER}",
            span=span,
        )
        self._blocks[key] = block
        return block

    def probe(self) -> dict[str, Any]:
        events = self.list_events()
        return {
            "api_root": f"recordings at {self._dir}",
            "reachable": True,
            "events_visible": len(events),
            "public_events": [
                event.event_id for event in events if not event.authentication_required
            ],
            "recorded_event": self.recorded_event_id,
            "reservations_open": self._reservations_open,
            "signed_in": False,
        }


__all__ = ["DEFAULT_RECORDING_DIR", "RecordedCatalogClient"]
