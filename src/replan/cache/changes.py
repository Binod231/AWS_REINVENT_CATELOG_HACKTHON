"""Detecting what moved in the catalog between two syncs.

The AWS Events API publishes no ``updated_at``, no ETag, and no change feed, so
the only way to know a session moved is to have kept the previous answer. That
makes this module the difference between a cache and something actually useful:
re:Invent reassigns rooms, shifts times, and fills seats continuously, and an
attendee whose plan depends on a 2:00 in Venetian genuinely needs to hear when it
becomes a 2:00 in Mandalay Bay.

Changes are classified rather than lumped together, because they demand different
reactions. A seat filling is routine and belongs to the watcher. A room moving can
break the travel feasibility of an entire afternoon and means the schedule must be
re-solved. Reporting both as "3 sessions changed" would bury the one that matters.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from replan.domain import Session

# Fields compared when deciding whether a session changed, and the bucket each
# one falls into. Anything not listed here is ignored on purpose: taxonomy churn
# ("areasOfInterest" gaining a label) is noise to a planner.
_TRACKED_FIELDS: tuple[tuple[str, str], ...] = (
    ("start", "rescheduled"),
    ("end", "rescheduled"),
    ("room", "moved"),
    ("venue", "moved"),
    ("seats", "seats"),
    ("reservable", "seats"),
    ("title", "edited"),
    ("level", "edited"),
    ("format", "edited"),
    ("speakers", "edited"),
    ("abstract", "edited"),
)


class ChangeKind(StrEnum):
    """What sort of change happened, ordered by how much it should worry you."""

    ADDED = "added"
    REMOVED = "removed"
    RESCHEDULED = "rescheduled"
    MOVED = "moved"
    EDITED = "edited"
    SEATS = "seats"

    @property
    def label(self) -> str:
        return {
            ChangeKind.ADDED: "new session",
            ChangeKind.REMOVED: "no longer in the catalog",
            ChangeKind.RESCHEDULED: "time changed",
            ChangeKind.MOVED: "room or venue changed",
            ChangeKind.EDITED: "details changed",
            ChangeKind.SEATS: "seat availability changed",
        }[self]

    @property
    def breaks_a_plan(self) -> bool:
        """Whether an existing solved agenda may no longer be valid.

        A time or room change can make a previously walkable pair of sessions
        impossible, and a removal leaves a hole. Those require re-solving. Seat
        movement and copy edits do not.
        """
        return self in {
            ChangeKind.ADDED,
            ChangeKind.REMOVED,
            ChangeKind.RESCHEDULED,
            ChangeKind.MOVED,
        }

    @property
    def severity(self) -> int:
        return {
            ChangeKind.REMOVED: 4,
            ChangeKind.RESCHEDULED: 3,
            ChangeKind.MOVED: 3,
            ChangeKind.ADDED: 2,
            ChangeKind.EDITED: 1,
            ChangeKind.SEATS: 0,
        }[self]


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One field that differs, with both values kept for the explanation."""

    field_name: str
    before: str | None
    after: str | None

    def describe(self) -> str:
        return f"{self.field_name}: {self.before or '-'} -> {self.after or '-'}"


@dataclass(frozen=True, slots=True)
class SessionChange:
    session_id: str
    code: str
    title: str
    kind: ChangeKind
    fields: tuple[FieldChange, ...] = ()

    def describe(self) -> str:
        if not self.fields:
            return f"{self.code} {self.title} - {self.kind.label}"
        detail = "; ".join(change.describe() for change in self.fields)
        return f"{self.code} {self.title} - {detail}"


@dataclass(frozen=True, slots=True)
class SyncDiff:
    """Everything that changed between the previous sync and this one."""

    event_id: str
    changes: tuple[SessionChange, ...] = ()
    unchanged: int = 0
    is_first_sync: bool = False
    total_now: int = 0
    total_before: int = 0
    _by_kind: dict[ChangeKind, tuple[SessionChange, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def of(self, kind: ChangeKind) -> tuple[SessionChange, ...]:
        return tuple(change for change in self.changes if change.kind is kind)

    @property
    def added(self) -> tuple[SessionChange, ...]:
        return self.of(ChangeKind.ADDED)

    @property
    def removed(self) -> tuple[SessionChange, ...]:
        return self.of(ChangeKind.REMOVED)

    @property
    def plan_breaking(self) -> tuple[SessionChange, ...]:
        """Changes that can invalidate an already-solved agenda."""
        return tuple(change for change in self.changes if change.kind.breaks_a_plan)

    @property
    def is_empty(self) -> bool:
        return not self.changes

    def counts(self) -> tuple[tuple[ChangeKind, int], ...]:
        tally: dict[ChangeKind, int] = {}
        for change in self.changes:
            tally[change.kind] = tally.get(change.kind, 0) + 1
        return tuple(sorted(tally.items(), key=lambda item: (-item[0].severity, item[0].value)))

    def summary(self) -> str:
        if self.is_first_sync:
            return f"first sync: {self.total_now} sessions cached"
        if self.is_empty:
            return f"no changes ({self.unchanged} sessions unchanged)"
        parts = [f"{count} {kind.value}" for kind, count in self.counts()]
        return ", ".join(parts)


def _snapshot(session: Session) -> dict[str, str | None]:
    """The comparable projection of a session."""
    return {
        "start": session.span.start.isoformat() if session.span else None,
        "end": session.span.end.isoformat() if session.span else None,
        "room": session.room.name,
        "venue": session.room.venue,
        "seats": session.seat_availability.value if session.seat_availability else None,
        "reservable": None if session.is_reservable is None else str(session.is_reservable),
        "title": session.title,
        "level": session.level_raw or session.level.label,
        "format": session.type_raw or session.format.value,
        "speakers": ", ".join(speaker.name for speaker in session.speakers) or None,
        # Abstracts are long; compare a digest so the DB stays small and a
        # one-character edit is still caught.
        "abstract": _digest(session.abstract) if session.abstract else None,
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def session_fingerprint(session: Session) -> str:
    """A stable hash over the tracked fields only.

    Used as a cheap equality check so an unchanged sync does no further work. It
    deliberately excludes taxonomy lists, which churn without affecting a plan.
    """
    snapshot = _snapshot(session)
    material = "\u0000".join(f"{key}={snapshot.get(key) or ''}" for key, _bucket in _TRACKED_FIELDS)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def compare_session(before: Session, after: Session) -> SessionChange | None:
    """Classify the difference between two versions of one session.

    When several buckets changed at once, the most severe wins: a session that
    moved room *and* filled up is reported as a move, because that is the part
    that can break the walk.
    """
    old, new = _snapshot(before), _snapshot(after)
    buckets: dict[str, list[FieldChange]] = {}

    for field_name, bucket in _TRACKED_FIELDS:
        if old.get(field_name) == new.get(field_name):
            continue
        buckets.setdefault(bucket, []).append(
            FieldChange(
                field_name=field_name,
                before=old.get(field_name),
                after=new.get(field_name),
            )
        )

    if not buckets:
        return None

    kind = max(
        (ChangeKind(bucket) for bucket in buckets),
        key=lambda candidate: candidate.severity,
    )
    changed_fields = tuple(change for bucket in buckets.values() for change in bucket)
    return SessionChange(
        session_id=after.session_id,
        code=after.code,
        title=after.title,
        kind=kind,
        fields=changed_fields,
    )


def diff_catalogs(
    event_id: str,
    before: dict[str, Session],
    after: dict[str, Session],
    *,
    is_first_sync: bool = False,
) -> SyncDiff:
    """Compute the full diff between two keyed session maps."""
    if is_first_sync or not before:
        return SyncDiff(
            event_id=event_id,
            changes=(),
            unchanged=0,
            is_first_sync=True,
            total_now=len(after),
            total_before=len(before),
        )

    changes: list[SessionChange] = []
    unchanged = 0

    for session_id, session in after.items():
        previous = before.get(session_id)
        if previous is None:
            changes.append(
                SessionChange(
                    session_id=session_id,
                    code=session.code,
                    title=session.title,
                    kind=ChangeKind.ADDED,
                )
            )
            continue
        difference = compare_session(previous, session)
        if difference is None:
            unchanged += 1
        else:
            changes.append(difference)

    for session_id, session in before.items():
        if session_id not in after:
            changes.append(
                SessionChange(
                    session_id=session_id,
                    code=session.code,
                    title=session.title,
                    kind=ChangeKind.REMOVED,
                )
            )

    changes.sort(key=lambda change: (-change.kind.severity, change.code))
    return SyncDiff(
        event_id=event_id,
        changes=tuple(changes),
        unchanged=unchanged,
        is_first_sync=False,
        total_now=len(after),
        total_before=len(before),
    )


__all__ = [
    "ChangeKind",
    "FieldChange",
    "SessionChange",
    "SyncDiff",
    "compare_session",
    "diff_catalogs",
    "session_fingerprint",
]
