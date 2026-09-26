"""Writing a plan back to the attendee's real schedule.

This is the only part of replan that changes anything outside the local machine, so
it is the part built most defensively. Three facts from the API's own documentation
shape the whole design:

1. **There is no idempotency key.** "If you never learned the outcome of a write - a
   timeout, a dropped connection, a 500, or a 503 - do not re-send it as is." So
   every run diffs against ``GetSchedule`` before acting and verifies against it
   afterwards, and a write whose outcome is unknown is reported rather than retried.

2. **A 200 does not mean success.** ``ReserveSessions`` and ``AssociateFavorites``
   report per session, and a session that was *already* reserved or favorited comes
   back in ``failed``. Re-sending is therefore not a safe retry, and "already in the
   desired state" has to be folded into success rather than treated as an error.

3. **Removals are the safe direction.** Cancelling a reservation or removing a
   favorite that is already gone returns 404, which the docs say to treat as done.
   ``DeletePersonalTime`` is outright idempotent.

On top of that, two policies of our own: nothing is written without ``--apply``, and
replan will not delete a personal-time block it did not create unless forced. A tool
that books seats on your behalf should be conservative about unbooking them.
"""

from __future__ import annotations

from replan.sync.actions import (
    Action,
    ActionKind,
    ActionPlan,
    ActionStatus,
    ApplyOutcome,
    ApplyReport,
)
from replan.sync.apply import apply_plan, verify_against_schedule
from replan.sync.diff import DiffOptions, build_action_plan
from replan.sync.journal import Journal, JournalRun, UndoReport, undo_run

__all__ = [
    "Action",
    "ActionKind",
    "ActionPlan",
    "ActionStatus",
    "ApplyOutcome",
    "ApplyReport",
    "DiffOptions",
    "Journal",
    "JournalRun",
    "UndoReport",
    "apply_plan",
    "build_action_plan",
    "undo_run",
    "verify_against_schedule",
]
