"""The ``CatalogPort`` protocol: one interface over both AWS front doors.

Every operation the AWS Events API exposes appears here exactly once, named as
AWS names it. Three implementations satisfy it - the live REST API, the live
remote MCP server, and committed recordings - and a single contract test suite
runs against all of them, so they cannot drift apart.

The signatures keep the API's own asymmetries rather than smoothing them over,
because the asymmetries are real and callers must handle them:

* ``reserve_sessions`` and ``associate_favorites`` take *lists* and return a
  :class:`~replan.domain.BulkResult` that can report partial failure on a 200.
* ``cancel_reservation`` and ``disassociate_favorite`` take a single session and
  return nothing; a 404 means "already done".
* ``delete_personal_time`` is idempotent, while the other two writes are not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from replan.domain import (
    AttendeeSchedule,
    BulkResult,
    Catalog,
    Event,
    PersonalTime,
    Session,
    TimeSpan,
)

ProgressHook = Callable[[int, int, int | None], None]
"""``(pages_fetched, sessions_so_far, total_count)`` - so a multi-page walk of a
2,000-session catalog can show progress instead of appearing to hang."""


@runtime_checkable
class CatalogPort(Protocol):
    """What replan needs from an Events API surface."""

    @property
    def surface_name(self) -> str:
        """Short provenance label, printed on output so live and recorded data
        can never be confused for one another."""
        ...

    # --- reads that need no credentials for public events ----------------- #

    def list_events(self, *, include_past: bool = False) -> tuple[Event, ...]:
        """``ListEvents``. No credentials required, for any event."""
        ...

    def get_event(self, event_id: str) -> Event:
        """``GetEvent``. No credentials required, even after the event ended."""
        ...

    # --- catalog reads ---------------------------------------------------- #

    def list_sessions(
        self,
        event_id: str,
        *,
        include_abstracts: bool = True,
        locale: str | None = None,
        progress: ProgressHook | None = None,
    ) -> Catalog:
        """``ListSessions``, walked to completion.

        Implementations must follow ``nextToken`` until it is absent, never
        stopping on a short page, and must report ``totalCount`` on the result so
        callers can verify the walk finished.
        """
        ...

    def get_session(self, event_id: str, session_id: str, *, locale: str | None = None) -> Session:
        """``GetSession``."""
        ...

    # --- schedule --------------------------------------------------------- #

    def get_schedule(self, event_id: str) -> AttendeeSchedule:
        """``GetSchedule``. Requires a signed-in, registered attendee."""
        ...

    def reserve_sessions(self, event_id: str, session_ids: list[str]) -> BulkResult:
        """``ReserveSessions``. Partial success is normal; always read ``failed``."""
        ...

    def cancel_reservation(self, event_id: str, session_id: str) -> bool:
        """``CancelReservation``. ``False`` when it was not reserved (404)."""
        ...

    def associate_favorites(self, event_id: str, session_ids: list[str]) -> BulkResult:
        """``AssociateFavorites``. Partial success is normal."""
        ...

    def disassociate_favorite(self, event_id: str, session_id: str) -> bool:
        """``DisassociateFavorite``. ``False`` when it was not a favorite (404)."""
        ...

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
        """``CreatePersonalTime``.

        Not idempotent: re-sending adds a second entry. Callers that lost the
        response must reconcile against ``GetSchedule`` instead of retrying.
        """
        ...

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
        """``UpdatePersonalTime``. A full replacement: omitted optional fields are cleared."""
        ...

    def delete_personal_time(self, event_id: str, personal_time_id: str) -> bool:
        """``DeletePersonalTime``. Idempotent; deleting a missing entry succeeds."""
        ...


@runtime_checkable
class SupportsPersonalTimeRead(Protocol):
    """Optional capability: reading a single personal time entry back.

    The API has no GetPersonalTime operation, so this is satisfied only by
    implementations that can derive it from ``GetSchedule``.
    """

    def get_personal_time(self, event_id: str, personal_time_id: str) -> PersonalTime | None: ...


__all__ = ["CatalogPort", "ProgressHook", "SupportsPersonalTimeRead"]
