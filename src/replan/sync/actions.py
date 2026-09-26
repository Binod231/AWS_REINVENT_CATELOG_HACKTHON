"""The vocabulary of a write-back: what we intend to do, and what happened.

Actions are ordered, individually reversible where the API allows it, and carry
enough detail to be shown to a user *before* anything is sent. That ordering is not
cosmetic - it is chosen so a failure part-way through leaves the account in a state
that is still coherent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from replan.domain import TimeSpan


class ActionKind(StrEnum):
    """One write against the Events API."""

    RESERVE = "reserve"
    CANCEL_RESERVATION = "cancel-reservation"
    ADD_FAVORITE = "add-favorite"
    REMOVE_FAVORITE = "remove-favorite"
    CREATE_PERSONAL_TIME = "create-personal-time"
    DELETE_PERSONAL_TIME = "delete-personal-time"
    UPDATE_PERSONAL_TIME = "update-personal-time"

    @property
    def operation(self) -> str:
        """The API operation name, for quota accounting and error messages."""
        return {
            ActionKind.RESERVE: "ReserveSessions",
            ActionKind.CANCEL_RESERVATION: "CancelReservation",
            ActionKind.ADD_FAVORITE: "AssociateFavorites",
            ActionKind.REMOVE_FAVORITE: "DisassociateFavorite",
            ActionKind.CREATE_PERSONAL_TIME: "CreatePersonalTime",
            ActionKind.DELETE_PERSONAL_TIME: "DeletePersonalTime",
            ActionKind.UPDATE_PERSONAL_TIME: "UpdatePersonalTime",
        }[self]

    @property
    def is_batched(self) -> bool:
        """Whether the API takes several sessions in one call.

        Both batched operations bill per session rather than per request, so a batch
        saves round trips and nothing else.
        """
        return self in {ActionKind.RESERVE, ActionKind.ADD_FAVORITE}

    @property
    def is_destructive(self) -> bool:
        """Whether this removes something the attendee currently has."""
        return self in {
            ActionKind.CANCEL_RESERVATION,
            ActionKind.REMOVE_FAVORITE,
            ActionKind.DELETE_PERSONAL_TIME,
        }

    @property
    def is_safe_to_retry(self) -> bool:
        """Whether re-sending after an unknown outcome is safe.

        Only the single removals and the idempotent delete qualify: a 404 means
        "already done". Re-sending a create would add a second entry, and re-sending
        a bulk request reports prior successes as failures.
        """
        return self in {
            ActionKind.CANCEL_RESERVATION,
            ActionKind.REMOVE_FAVORITE,
            ActionKind.DELETE_PERSONAL_TIME,
        }

    @property
    def label(self) -> str:
        return self.value.replace("-", " ")


class ActionPhase(IntEnum):
    """Execution order.

    Deliberate, and the reasoning matters:

    * Personal time that replan itself created is cleared *first*, so a stale block
      cannot make a reservation fail with ``scheduleConflict``.
    * Reservations go next, because seats are the only genuinely scarce resource
      here. If the run is going to be cut short by a quota or a network failure, the
      seats should already be held.
    * Favorites follow: cheap, never scarce, and harmless to defer.
    * New personal time is created after reservations, so a fresh block cannot
      conflict with a seat we were about to take.
    * Anything that removes something the attendee already had is last, so a failure
      earlier in the run never costs them something they wanted to keep.
    """

    CLEAR_OWN_BLOCKS = 10
    RESERVE = 20
    FAVORITES = 30
    CREATE_BLOCKS = 40
    PRUNE = 50


class ActionStatus(StrEnum):
    PLANNED = "planned"
    DONE = "done"
    ALREADY = "already"
    """The desired state already held. Counted as success, not failure."""

    FAILED = "failed"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"
    """The write was sent but its outcome could not be established. Needs reconciling."""

    @property
    def is_success(self) -> bool:
        return self in {ActionStatus.DONE, ActionStatus.ALREADY}

    @property
    def needs_attention(self) -> bool:
        return self in {ActionStatus.FAILED, ActionStatus.UNKNOWN}


@dataclass(frozen=True, slots=True)
class Action:
    """A single intended write."""

    kind: ActionKind
    session_ids: tuple[str, ...] = ()
    personal_time_id: str | None = None
    span: TimeSpan | None = None
    title: str = ""
    description: str = ""
    location: str | None = None
    reason: str = ""
    """Why this action is in the plan, shown in the dry run."""

    labels: tuple[str, ...] = ()
    """Human-readable names for the targets, so a preview is readable."""

    @property
    def phase(self) -> ActionPhase:
        return {
            ActionKind.DELETE_PERSONAL_TIME: ActionPhase.CLEAR_OWN_BLOCKS,
            ActionKind.RESERVE: ActionPhase.RESERVE,
            ActionKind.ADD_FAVORITE: ActionPhase.FAVORITES,
            ActionKind.CREATE_PERSONAL_TIME: ActionPhase.CREATE_BLOCKS,
            ActionKind.UPDATE_PERSONAL_TIME: ActionPhase.CREATE_BLOCKS,
            ActionKind.CANCEL_RESERVATION: ActionPhase.PRUNE,
            ActionKind.REMOVE_FAVORITE: ActionPhase.PRUNE,
        }[self.kind]

    @property
    def target_count(self) -> int:
        if self.kind.is_batched:
            return len(self.session_ids)
        return 1

    def describe(self) -> str:
        if self.kind.is_batched:
            names = ", ".join(self.labels or self.session_ids)
            return f"{self.kind.label}: {names}"
        if self.session_ids:
            name = self.labels[0] if self.labels else self.session_ids[0]
            return f"{self.kind.label}: {name}"
        when = f" {self.span}" if self.span else ""
        return f"{self.kind.label}: {self.title}{when}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "sessionIds": list(self.session_ids),
            "personalTimeId": self.personal_time_id,
            "start": self.span.start.isoformat() if self.span else None,
            "end": self.span.end.isoformat() if self.span else None,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "reason": self.reason,
            "labels": list(self.labels),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Action:
        span: TimeSpan | None = None
        if payload.get("start") and payload.get("end"):
            from datetime import datetime

            span = TimeSpan(
                datetime.fromisoformat(payload["start"]),
                datetime.fromisoformat(payload["end"]),
            )
        return cls(
            kind=ActionKind(payload["kind"]),
            session_ids=tuple(payload.get("sessionIds") or ()),
            personal_time_id=payload.get("personalTimeId"),
            span=span,
            title=payload.get("title") or "",
            description=payload.get("description") or "",
            location=payload.get("location"),
            reason=payload.get("reason") or "",
            labels=tuple(payload.get("labels") or ()),
        )


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """Everything a write-back run intends to do, in execution order."""

    event_id: str
    actions: tuple[Action, ...] = ()
    notes: tuple[str, ...] = ()
    """Things the user should know: skipped foreign blocks, closed operations."""

    @property
    def is_empty(self) -> bool:
        return not self.actions

    @property
    def ordered(self) -> tuple[Action, ...]:
        return tuple(sorted(self.actions, key=lambda item: (item.phase, item.kind.value)))

    @property
    def total_targets(self) -> int:
        return sum(action.target_count for action in self.actions)

    @property
    def destructive(self) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if action.kind.is_destructive)

    def of_kind(self, kind: ActionKind) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if action.kind is kind)

    def quota_cost(self) -> dict[str, int]:
        """Quota units per operation, so a preview can warn about throttling.

        The batched operations bill per session named, which is why this counts
        targets rather than calls.
        """
        cost: dict[str, int] = {}
        for action in self.actions:
            key = action.kind.operation
            cost[key] = cost.get(key, 0) + action.target_count
        return cost

    def summary(self) -> str:
        if self.is_empty:
            return "nothing to do; your schedule already matches the plan"
        counts: dict[str, int] = {}
        for action in self.actions:
            counts[action.kind.label] = counts.get(action.kind.label, 0) + action.target_count
        return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))


@dataclass(frozen=True, slots=True)
class ApplyOutcome:
    """What happened to one action."""

    action: Action
    status: ActionStatus
    detail: str = ""
    succeeded_ids: tuple[str, ...] = ()
    failed_ids: tuple[str, ...] = ()
    created_personal_time_id: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status.is_success

    def describe(self) -> str:
        return f"[{self.status.value}] {self.action.describe()}" + (
            f" - {self.detail}" if self.detail else ""
        )


@dataclass(frozen=True, slots=True)
class ApplyReport:
    """The result of a whole run."""

    event_id: str
    run_id: int | None
    dry_run: bool
    outcomes: tuple[ApplyOutcome, ...] = ()
    notes: tuple[str, ...] = ()
    verification: tuple[str, ...] = field(default_factory=tuple)
    verified: bool = False

    @property
    def succeeded(self) -> tuple[ApplyOutcome, ...]:
        return tuple(item for item in self.outcomes if item.is_success)

    @property
    def failed(self) -> tuple[ApplyOutcome, ...]:
        return tuple(item for item in self.outcomes if item.status is ActionStatus.FAILED)

    @property
    def unknown(self) -> tuple[ApplyOutcome, ...]:
        """Writes whose outcome could not be established.

        These are the dangerous ones. The API has no idempotency key, so they must
        be reconciled against GetSchedule rather than retried.
        """
        return tuple(item for item in self.outcomes if item.status is ActionStatus.UNKNOWN)

    @property
    def skipped(self) -> tuple[ApplyOutcome, ...]:
        return tuple(item for item in self.outcomes if item.status is ActionStatus.SKIPPED)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.unknown

    def summary(self) -> str:
        if self.dry_run:
            return f"dry run: {len(self.outcomes)} action(s) would be applied"
        parts = [f"{len(self.succeeded)} succeeded"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.unknown:
            parts.append(f"{len(self.unknown)} unknown")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)

    def to_json(self) -> str:
        return json.dumps(
            {
                "eventId": self.event_id,
                "runId": self.run_id,
                "dryRun": self.dry_run,
                "verified": self.verified,
                "summary": self.summary(),
                "outcomes": [
                    {
                        "kind": item.action.kind.value,
                        "status": item.status.value,
                        "detail": item.detail,
                        "succeeded": list(item.succeeded_ids),
                        "failed": list(item.failed_ids),
                    }
                    for item in self.outcomes
                ],
                "notes": list(self.notes),
                "verification": list(self.verification),
            },
            indent=2,
        )


__all__ = [
    "Action",
    "ActionKind",
    "ActionPhase",
    "ActionPlan",
    "ActionStatus",
    "ApplyOutcome",
    "ApplyReport",
]
