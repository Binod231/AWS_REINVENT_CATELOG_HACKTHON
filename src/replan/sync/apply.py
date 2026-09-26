"""Executing an action plan against the live API, carefully.

The rules this file exists to obey, all from the API's own documentation:

* **A 200 is not success.** Bulk results are read per session, and codes meaning
  "already in the desired state" are folded into success. Anything else is a real
  failure and is reported as one.
* **An unknown outcome is not a failure, and must not be retried.** With no
  idempotency key, re-sending a create adds a second entry. Those writes get status
  ``UNKNOWN`` and the run tells the user to reconcile.
* **Closed operations are expected, not exceptional.** Reservations return 409 for
  every caller until reserved seating opens through the API on 8 October 2026. The
  run notes it, skips the reservations, and carries on with favorites - which do work.
* **404 on a removal means done.** The port already returns ``False`` for that, and
  that is treated as success.

Finally, ``verify_against_schedule`` re-reads ``GetSchedule`` after writing and
compares it with what was intended. That is the only honest way to confirm a write
landed, and the API explicitly recommends it.
"""

from __future__ import annotations

from typing import Any, assert_never

from replan.adapters.errors import (
    EventsApiError,
    EventsNotFoundError,
    EventsOperationClosedError,
)
from replan.domain import AttendeeSchedule, PersonalTime, TimeSpan
from replan.solver import Plan
from replan.sync.actions import (
    Action,
    ActionKind,
    ActionPlan,
    ActionStatus,
    ApplyOutcome,
    ApplyReport,
)
from replan.sync.journal import Journal


def execute_action(port: Any, event_id: str, action: Action) -> ApplyOutcome:
    """Perform one action. Never raises for an expected API refusal."""
    try:
        if action.kind is ActionKind.RESERVE:
            result = port.reserve_sessions(event_id, list(action.session_ids))
            return _from_bulk(action, result)

        if action.kind is ActionKind.ADD_FAVORITE:
            result = port.associate_favorites(event_id, list(action.session_ids))
            return _from_bulk(action, result)

        if action.kind is ActionKind.CANCEL_RESERVATION:
            statuses = [
                port.cancel_reservation(event_id, session_id) for session_id in action.session_ids
            ]
            # False means 404, which the docs say to treat as already done.
            done = all(statuses)
            return ApplyOutcome(
                action=action,
                status=ActionStatus.DONE if done else ActionStatus.ALREADY,
                detail="" if done else "was not reserved, so nothing to cancel",
                succeeded_ids=action.session_ids,
            )

        if action.kind is ActionKind.REMOVE_FAVORITE:
            statuses = [
                port.disassociate_favorite(event_id, session_id)
                for session_id in action.session_ids
            ]
            done = all(statuses)
            return ApplyOutcome(
                action=action,
                status=ActionStatus.DONE if done else ActionStatus.ALREADY,
                detail="" if done else "was not a favorite, so nothing to remove",
                succeeded_ids=action.session_ids,
            )

        if action.kind is ActionKind.CREATE_PERSONAL_TIME:
            if action.span is None:
                return ApplyOutcome(
                    action=action,
                    status=ActionStatus.FAILED,
                    detail="no time span given for the block",
                )
            port.create_personal_time(
                event_id,
                span=action.span,
                title=action.title,
                description=action.description,
                location=action.location,
            )
            # The id is not in the response, so it is resolved later by reading the
            # schedule back. Without that the block cannot be undone.
            return ApplyOutcome(action=action, status=ActionStatus.DONE)

        if action.kind is ActionKind.DELETE_PERSONAL_TIME:
            if action.personal_time_id is None:
                return ApplyOutcome(
                    action=action,
                    status=ActionStatus.FAILED,
                    detail="no personal time id to delete",
                )
            port.delete_personal_time(event_id, action.personal_time_id)
            # Documented as idempotent, so this is success either way.
            return ApplyOutcome(action=action, status=ActionStatus.DONE)

        if action.kind is ActionKind.UPDATE_PERSONAL_TIME:
            if action.personal_time_id is None or action.span is None:
                return ApplyOutcome(
                    action=action,
                    status=ActionStatus.FAILED,
                    detail="update needs both an id and a span",
                )
            port.update_personal_time(
                event_id,
                action.personal_time_id,
                span=action.span,
                title=action.title,
                description=action.description,
                location=action.location,
            )
            return ApplyOutcome(action=action, status=ActionStatus.DONE)

    except EventsOperationClosedError as error:
        return ApplyOutcome(
            action=action,
            status=ActionStatus.SKIPPED,
            detail=f"{error.message} {error.remedy}",
        )
    except EventsNotFoundError:
        # For a removal this is "already gone"; for anything else the target is
        # missing and there is nothing to do.
        return ApplyOutcome(
            action=action,
            status=ActionStatus.ALREADY,
            detail="the API reported it does not exist, so treating as already done",
        )
    except EventsApiError as error:
        unknown = (
            "Reconcile against GetSchedule" in str(error)
            or "outcome is unknown" in str(error).lower()
        )
        return ApplyOutcome(
            action=action,
            status=ActionStatus.UNKNOWN if unknown else ActionStatus.FAILED,
            detail=f"{error} - {error.remedy}",
        )

    # Every ActionKind is handled above, so the kind is narrowed to Never here.
    # Adding a kind without an executor becomes a type error rather than a silent
    # "unsupported action" at runtime.
    assert_never(action.kind)


def _from_bulk(action: Action, result: Any) -> ApplyOutcome:
    """Interpret a BulkResult.

    ``effective_successes`` folds in the "already reserved"/"already favorited"
    codes, because the end state is what the caller wanted. Everything left in
    ``real_failures`` is a genuine refusal worth reporting.
    """
    succeeded = tuple(sorted(result.effective_successes))
    failures = result.real_failures

    if not failures:
        return ApplyOutcome(action=action, status=ActionStatus.DONE, succeeded_ids=succeeded)

    detail = "; ".join(item.describe() for item in failures)
    partial = bool(succeeded)
    return ApplyOutcome(
        action=action,
        status=ActionStatus.FAILED,
        detail=("partial: " if partial else "") + detail,
        succeeded_ids=succeeded,
        failed_ids=tuple(item.session_id for item in failures),
    )


def _resolve_created_block_ids(
    port: Any,
    event_id: str,
    journal: Journal | None,
    run_id: int | None,
    created: list[tuple[int, Action]],
) -> list[str]:
    """Find the ids of blocks we just created, by reading the schedule back.

    ``CreatePersonalTime`` answers 204 with no body, so this is the only way to learn
    what id the new entry was given - and without the id, ``replan undo`` could not
    remove it.
    """
    if not created:
        return []
    notes: list[str] = []
    try:
        schedule = port.get_schedule(event_id)
    except EventsApiError as error:
        notes.append(f"could not read the schedule back to identify created blocks: {error}")
        return notes

    for seq, action in created:
        match = next(
            (
                block
                for block in schedule.personal_time
                if action.span is not None
                and block.span.start == action.span.start
                and block.span.end == action.span.end
                and block.title.strip().lower() == action.title.strip().lower()
            ),
            None,
        )
        if match is None:
            notes.append(
                f"created '{action.title}' but could not find it on the schedule "
                "afterwards, so it cannot be undone automatically"
            )
            continue
        if journal is not None and run_id is not None:
            journal.attach_personal_time_id(run_id, seq, match.personal_time_id)
    return notes


def verify_against_schedule(
    port: Any,
    event_id: str,
    plan: Plan,
    action_plan: ActionPlan,
) -> tuple[bool, tuple[str, ...]]:
    """Re-read the schedule and confirm the intended state now holds.

    The API recommends exactly this after a write, and it is the only claim of
    success worth making: the transport said 200, but the schedule is the truth.
    """
    try:
        schedule: AttendeeSchedule = port.get_schedule(event_id)
    except EventsApiError as error:
        return False, (f"could not verify: {error}",)

    problems: list[str] = []

    intended_reserved = {
        session_id
        for action in action_plan.of_kind(ActionKind.RESERVE)
        for session_id in action.session_ids
    }
    missing_reserved = intended_reserved - schedule.reserved_ids
    if missing_reserved:
        problems.append(
            f"{len(missing_reserved)} intended reservation(s) are not on your schedule: "
            + ", ".join(sorted(missing_reserved))
        )

    intended_favorites = {
        session_id
        for action in action_plan.of_kind(ActionKind.ADD_FAVORITE)
        for session_id in action.session_ids
    }
    missing_favorites = intended_favorites - schedule.favorite_ids
    if missing_favorites:
        problems.append(
            f"{len(missing_favorites)} intended favorite(s) are not on your schedule: "
            + ", ".join(sorted(missing_favorites))
        )

    intended_blocks = action_plan.of_kind(ActionKind.CREATE_PERSONAL_TIME)
    for action in intended_blocks:
        if action.span is None:
            continue
        found = any(
            block.span.start == action.span.start
            and block.span.end == action.span.end
            and block.title.strip().lower() == action.title.strip().lower()
            for block in schedule.personal_time
        )
        if not found:
            problems.append(f"personal time block '{action.title}' is not on your schedule")

    # Verify the plan's own intent, not merely the action list. A planned session that
    # needs a seat and still has none is worth reporting even if no reserve action was
    # generated for it, because that is the outcome the attendee actually cares about.
    unseated = [
        item.code for item in plan.needs_reservation if item.session_id not in schedule.reserved_ids
    ]
    if unseated:
        problems.append(
            f"{len(unseated)} planned session(s) still have no seat: " + ", ".join(sorted(unseated))
        )

    if problems:
        return False, tuple(problems)

    confirmed = [
        f"{len(schedule.reserved)} reservation(s), {len(schedule.favorites)} favorite(s), "
        f"{len(schedule.personal_time)} personal time block(s) on your schedule"
    ]
    return True, tuple(confirmed)


def apply_plan(
    port: Any,
    plan: Plan,
    action_plan: ActionPlan,
    *,
    dry_run: bool = True,
    journal: Journal | None = None,
    verify: bool = True,
    progress: Any = None,
) -> ApplyReport:
    """Execute an action plan. Dry run by default."""
    event_id = action_plan.event_id
    notes = list(action_plan.notes)
    outcomes: list[ApplyOutcome] = []

    run_id: int | None = None
    if journal is not None:
        run_id = journal.start_run(
            event_id,
            dry_run=dry_run,
            surface=getattr(port, "surface_name", None),
        )

    if dry_run:
        for action in action_plan.ordered:
            outcomes.append(
                ApplyOutcome(
                    action=action,
                    status=ActionStatus.PLANNED,
                    detail=action.reason,
                )
            )
        report = ApplyReport(
            event_id=event_id,
            run_id=run_id,
            dry_run=True,
            outcomes=tuple(outcomes),
            notes=tuple(notes),
        )
        if journal is not None and run_id is not None:
            for seq, outcome in enumerate(outcomes):
                journal.record(run_id, seq, outcome)
            journal.finish_run(run_id, report.summary())
        return report

    created_blocks: list[tuple[int, Action]] = []
    reservations_closed = False

    for seq, action in enumerate(action_plan.ordered):
        if reservations_closed and action.kind is ActionKind.RESERVE:
            # One 409 means the operation is closed for everyone; do not keep asking.
            outcomes.append(
                ApplyOutcome(
                    action=action,
                    status=ActionStatus.SKIPPED,
                    detail="reservations are closed; skipped without another call",
                )
            )
            continue

        if progress is not None:
            progress(seq + 1, len(action_plan.ordered), action)

        outcome = execute_action(port, event_id, action)
        outcomes.append(outcome)

        if (
            action.kind is ActionKind.RESERVE
            and outcome.status is ActionStatus.SKIPPED
            and "8 October 2026" in outcome.detail
        ):
            reservations_closed = True
            notes.append(
                "Reservations are not open through the API yet, so seats were not "
                "taken. Favorites and personal time were still applied."
            )

        if outcome.status is ActionStatus.DONE and action.kind is ActionKind.CREATE_PERSONAL_TIME:
            created_blocks.append((seq, action))

        if journal is not None and run_id is not None:
            journal.record(run_id, seq, outcome)

    notes.extend(_resolve_created_block_ids(port, event_id, journal, run_id, created_blocks))

    verified = False
    verification: tuple[str, ...] = ()
    if verify:
        verified, verification = verify_against_schedule(port, event_id, plan, action_plan)

    report = ApplyReport(
        event_id=event_id,
        run_id=run_id,
        dry_run=False,
        outcomes=tuple(outcomes),
        notes=tuple(notes),
        verification=verification,
        verified=verified,
    )

    if journal is not None and run_id is not None:
        journal.finish_run(run_id, report.summary())

    return report


def restore_block(span: TimeSpan, title: str, description: str) -> PersonalTime:
    """Build a PersonalTime for previewing a restore, without an upstream id yet."""
    return PersonalTime(
        personal_time_id="pending",
        title=title,
        description=description,
        span=span,
    )


__all__ = ["apply_plan", "execute_action", "restore_block", "verify_against_schedule"]
