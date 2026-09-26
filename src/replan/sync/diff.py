"""What needs to change, worked out by comparing the plan with live upstream state.

The API has no idempotency key, so the only safe way to write is to know exactly what
is already there. Every action below exists because ``GetSchedule`` said the desired
state does not hold yet.

Two asymmetries are deliberate and worth defending:

**Adding is default, removing is opt-in.** replan will happily favorite and reserve
what your plan calls for, but it will not un-favorite or cancel anything unless you
pass ``--prune``. The API gives favorites no metadata at all, so there is no way to
tell a favorite replan added from one you starred yourself on the mobile app. Guessing
wrong would silently delete your own work.

**Personal time is different, because we can sign it.** The API has no user metadata
field, so replan writes a marker into the block's description. Blocks carrying that
marker are ours to manage; blocks without it were made by hand and are never touched
without ``--force``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from replan.domain import (
    REPLAN_BLOCK_MARKER,
    AttendeeSchedule,
    Catalog,
    PersonalTime,
    TimeSpan,
    resolve_day,
    resolve_timezone,
    round_to_increment,
)
from replan.profiles import MealWindow, Profile
from replan.solver import Plan
from replan.sync.actions import Action, ActionKind, ActionPlan

# ReserveSessions and AssociateFavorites allow 30 sessions per minute. Batching at
# the quota means one full batch spends the whole minute's allowance, so we stay a
# little under it to leave room for the verification read that follows.
MAX_BATCH = 25


def block_description(title: str, note: str = "") -> str:
    """Build a personal-time description that records replan's authorship.

    The marker is visible to the attendee on purpose. A tool that puts blocks on
    your calendar should sign its work, not hide it.
    """
    body = note.strip() or f"{title.strip()} held by replan"
    return f"{body} {REPLAN_BLOCK_MARKER}"


@dataclass(frozen=True, slots=True)
class DiffOptions:
    """Switches that change how aggressive the write-back is."""

    favorite_plan_items: bool = True
    """Favorite everything in the plan. Favorites are interest only, never a seat,
    and they work today - reservations do not open through the API until 8 October
    2026, so this is the useful half in the meantime."""

    reserve_seats: bool = True
    create_personal_time: bool = True

    prune_favorites: bool = False
    """Remove favorites not in the plan. Off by default: the API cannot tell us which
    favorites replan added, so this would delete work you did by hand."""

    prune_reservations: bool = False
    """Cancel reservations not in the plan. Off by default for the same reason, and
    because a held seat is expensive to get back."""

    force_foreign_blocks: bool = False
    """Allow deleting personal-time blocks replan did not create."""

    include_meal_windows: bool = True


def _materialise_meal_windows(
    profile: Profile,
    catalog: Catalog,
    days: tuple[date, ...],
) -> tuple[tuple[MealWindow, TimeSpan], ...]:
    """Turn recurring meal windows into concrete spans for the event's days."""
    zone = resolve_timezone(catalog.event.timezone_name if catalog.event else None)
    calendar = list(days) or list(catalog.days)
    materialised: list[tuple[MealWindow, TimeSpan]] = []

    for window in profile.meal_windows:
        if not window.protect:
            continue
        wanted = calendar
        if window.days:
            resolved: list[date] = []
            for token in window.days:
                try:
                    resolved.append(resolve_day(token, calendar))
                except ValueError:
                    continue
            wanted = resolved
        for current in wanted:
            start = round_to_increment(datetime.combine(current, window.start, tzinfo=zone))
            end = round_to_increment(datetime.combine(current, window.end, tzinfo=zone), up=True)
            if end <= start:
                continue
            span = TimeSpan(start, end)
            # The API requires a whole number of 5-minute increments; rounding above
            # guarantees it, so validation here would be redundant.
            materialised.append((window, span))

    return tuple(materialised)


def _same_block(existing: PersonalTime, title: str, span: TimeSpan) -> bool:
    """Whether an existing block already represents this window.

    Compared on title and exact span. The API stores blocks to the minute, and both
    sides are rounded to the same 5-minute grid, so an exact comparison is safe.
    """
    return (
        existing.title.strip().lower() == title.strip().lower()
        and existing.span.start == span.start
        and existing.span.end == span.end
    )


def build_action_plan(
    plan: Plan,
    schedule: AttendeeSchedule,
    catalog: Catalog,
    profile: Profile,
    options: DiffOptions | None = None,
) -> ActionPlan:
    """Diff a solved plan against the attendee's live schedule."""
    settings = options or DiffOptions()
    actions: list[Action] = []
    notes: list[str] = []

    labels = {item.session_id: f"{item.code} {item.title[:48]}" for item in plan.items}

    # --- reservations ------------------------------------------------------ #
    if settings.reserve_seats:
        wanted = [
            item
            for item in plan.items
            if item.needs_reservation and item.session_id not in schedule.reserved_ids
        ]
        for chunk in _chunks([item.session_id for item in wanted], MAX_BATCH):
            actions.append(
                Action(
                    kind=ActionKind.RESERVE,
                    session_ids=tuple(chunk),
                    labels=tuple(labels.get(item, item) for item in chunk),
                    reason="in your plan and reservable, but no seat held yet",
                )
            )
        already = [
            item
            for item in plan.items
            if item.needs_reservation and item.session_id in schedule.reserved_ids
        ]
        if already:
            notes.append(
                f"{len(already)} planned session(s) already have a seat held; leaving them alone"
            )

    # --- favorites --------------------------------------------------------- #
    if settings.favorite_plan_items:
        missing = [
            item.session_id for item in plan.items if item.session_id not in schedule.favorite_ids
        ]
        for chunk in _chunks(missing, MAX_BATCH):
            actions.append(
                Action(
                    kind=ActionKind.ADD_FAVORITE,
                    session_ids=tuple(chunk),
                    labels=tuple(labels.get(item, item) for item in chunk),
                    reason="in your plan but not marked as a favorite",
                )
            )

    # --- personal time ----------------------------------------------------- #
    if settings.create_personal_time and settings.include_meal_windows:
        materialised = _materialise_meal_windows(profile, catalog, plan.days)
        for window, span in materialised:
            if any(_same_block(block, window.title, span) for block in schedule.personal_time):
                continue
            actions.append(
                Action(
                    kind=ActionKind.CREATE_PERSONAL_TIME,
                    span=span,
                    title=window.title,
                    description=block_description(window.title, window.description),
                    location=window.location,
                    reason="protected time from your profile that is not on your schedule yet",
                )
            )

        # Remove replan-authored blocks that no longer correspond to any window.
        wanted_pairs = {(window.title.strip().lower(), span) for window, span in materialised}
        for block in schedule.personal_time:
            if not block.created_by_replan:
                continue
            if (block.title.strip().lower(), block.span) in wanted_pairs:
                continue
            actions.append(
                Action(
                    kind=ActionKind.DELETE_PERSONAL_TIME,
                    personal_time_id=block.personal_time_id,
                    span=block.span,
                    title=block.title,
                    reason="replan created this block, and your profile no longer asks for it",
                )
            )

        foreign = [block for block in schedule.personal_time if not block.created_by_replan]
        if foreign:
            notes.append(
                f"{len(foreign)} personal time block(s) were not created by replan and "
                "will not be touched"
            )

    # --- pruning (opt-in) --------------------------------------------------- #
    planned_ids = {item.session_id for item in plan.items}

    if settings.prune_reservations:
        for session_id in sorted(schedule.reserved_ids - planned_ids):
            session = catalog.by_id(session_id)
            actions.append(
                Action(
                    kind=ActionKind.CANCEL_RESERVATION,
                    session_ids=(session_id,),
                    labels=(session.code if session else session_id,),
                    reason="you hold a seat but it is not in this plan",
                )
            )
    elif schedule.reserved_ids - planned_ids:
        extra = len(schedule.reserved_ids - planned_ids)
        notes.append(
            f"{extra} held reservation(s) are not in this plan. They were left alone; "
            "pass --prune-reservations to cancel them."
        )

    if settings.prune_favorites:
        for session_id in sorted(schedule.favorite_ids - planned_ids):
            session = catalog.by_id(session_id)
            actions.append(
                Action(
                    kind=ActionKind.REMOVE_FAVORITE,
                    session_ids=(session_id,),
                    labels=(session.code if session else session_id,),
                    reason="favorited but not in this plan",
                )
            )
    elif schedule.favorite_ids - planned_ids:
        extra = len(schedule.favorite_ids - planned_ids)
        notes.append(
            f"{extra} favorite(s) are not in this plan. Left alone, because the API "
            "cannot tell us which favorites replan added."
        )

    if settings.force_foreign_blocks:
        notes.append("--force is set: personal time blocks replan did not create may be deleted")

    return ActionPlan(event_id=plan.event_id, actions=tuple(actions), notes=tuple(notes))


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


__all__ = ["MAX_BATCH", "DiffOptions", "block_description", "build_action_plan"]
