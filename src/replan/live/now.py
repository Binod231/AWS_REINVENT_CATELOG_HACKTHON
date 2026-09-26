"""What should I do next, from here?

The re-solve is deliberately narrow. It considers only the rest of the current day,
only sessions you could physically reach from where you are standing, and it holds any
seat you already hold. Anything wider would be answering a question you did not ask.

Two things this does that the main solver cannot:

* **Reachability from a position.** The solver knows travel between sessions. It has
  no idea where *you* are. Standing in Mandalay Bay at 14:10 rules out a 14:30 in the
  Venetian, and that has nothing to do with the catalog.
* **Departure times, not durations.** "It is a 24 minute walk" requires arithmetic
  under time pressure. "Leave at 14:06" does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import IntEnum
from zoneinfo import ZoneInfo

from replan.domain import (
    AttendeeSchedule,
    Catalog,
    Room,
    Session,
    now_utc,
    resolve_day,
)
from replan.profiles import Profile, Scorer
from replan.solver import Plan, PlanItem, SolveRequest, solve
from replan.travel import Leg, TravelGraph

_CLOCK = re.compile(r"^(\d{1,2})[:.](\d{2})$")


class Urgency(IntEnum):
    """How soon you have to move. Ordered so the most pressing sorts first."""

    LEAVE_NOW = 4
    LEAVE_SOON = 3
    """Within ten minutes."""

    COMFORTABLE = 2
    STAY_PUT = 1
    """Same room as where you already are. No transfer at all."""

    AFTER_CURRENT = 0
    """You are in something else until then."""

    @property
    def label(self) -> str:
        return {
            Urgency.LEAVE_NOW: "leave now",
            Urgency.LEAVE_SOON: "leave soon",
            Urgency.COMFORTABLE: "you have time",
            Urgency.STAY_PUT: "stay where you are",
            Urgency.AFTER_CURRENT: "after your current session",
        }[self]

    @property
    def needs_movement(self) -> bool:
        return self in {Urgency.LEAVE_NOW, Urgency.LEAVE_SOON, Urgency.COMFORTABLE}


@dataclass(frozen=True, slots=True)
class Moment:
    """A resolved point in time, with the conference day it belongs to."""

    at: datetime
    day: date
    zone: ZoneInfo
    is_during_event: bool

    @property
    def clock(self) -> str:
        return f"{self.at.astimezone(self.zone):%a %H:%M}"


@dataclass(frozen=True, slots=True)
class NextMove:
    """One concrete instruction."""

    item: PlanItem
    depart_at: datetime | None
    travel_minutes: int
    leg: Leg | None
    urgency: Urgency
    minutes_until_departure: int | None
    zone: ZoneInfo

    @property
    def depart_clock(self) -> str | None:
        if self.depart_at is None:
            return None
        return f"{self.depart_at.astimezone(self.zone):%H:%M}"

    @property
    def start_clock(self) -> str:
        return f"{self.item.span.start.astimezone(self.zone):%H:%M}"

    @property
    def end_clock(self) -> str:
        return f"{self.item.span.end.astimezone(self.zone):%H:%M}"

    def describe(self) -> str:
        head = f"{self.start_clock} {self.item.code} - {self.item.title}"
        where = f" in {self.item.room_display}"
        if self.urgency is Urgency.STAY_PUT:
            return f"{head}{where} - same room, stay put"
        if self.depart_at is None or self.travel_minutes == 0:
            return f"{head}{where}"
        if self.minutes_until_departure is not None and self.minutes_until_departure <= 0:
            return f"{head}{where} - leave now ({self.travel_minutes} min)"
        return f"{head}{where} - leave {self.depart_clock} ({self.travel_minutes} min)"


@dataclass(frozen=True, slots=True)
class NowReport:
    """The answer to "what next?"."""

    moment: Moment
    location: Room | None
    in_progress: PlanItem | None
    moves: tuple[NextMove, ...]
    plan: Plan
    unreachable: tuple[tuple[Session, int], ...] = ()
    """Sessions ruled out purely because of where you are, with the travel needed."""

    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.moves

    @property
    def next_move(self) -> NextMove | None:
        return self.moves[0] if self.moves else None


def parse_moment(
    token: str | None,
    catalog: Catalog,
    *,
    reference: datetime | None = None,
) -> Moment:
    """Resolve ``--at`` into a concrete instant on a concrete conference day.

    Accepts ``now``, a bare clock time like ``14:10``, a weekday plus time like
    ``wed 14:10``, or a full ISO timestamp. A bare clock resolves against the event
    day you are currently in, or the first day if the event has not started - which is
    what someone typing ``--at 14:10`` in October actually means.
    """
    zone = catalog.zone
    days = catalog.days
    current = (reference or now_utc()).astimezone(zone)

    if not days:
        return Moment(at=current, day=current.date(), zone=zone, is_during_event=False)

    def day_for(value: date) -> date:
        return value

    text = (token or "now").strip().lower()

    if text in {"now", ""}:
        today = current.date()
        if today in days:
            return Moment(at=current, day=today, zone=zone, is_during_event=True)
        # Outside the event: answer for the first day that has not finished yet, so
        # the command is still useful before the conference starts.
        upcoming = next((item for item in days if item >= today), days[-1])
        anchor = datetime.combine(upcoming, current.timetz()).astimezone(zone)
        return Moment(at=anchor, day=upcoming, zone=zone, is_during_event=False)

    # Full ISO timestamp.
    try:
        parsed = datetime.fromisoformat(token or "")
    except ValueError:
        parsed = None
    if parsed is not None:
        aware = parsed if parsed.tzinfo else parsed.replace(tzinfo=zone)
        localised = aware.astimezone(zone)
        return Moment(
            at=localised,
            day=day_for(localised.date()),
            zone=zone,
            is_during_event=localised.date() in days,
        )

    # "<day> <clock>" or a bare clock.
    parts = text.split()
    clock_text = parts[-1]
    match = _CLOCK.match(clock_text)
    if match is None:
        msg = (
            f"cannot read a time from {token!r}. Try 'now', '14:10', 'wed 14:10', "
            "or an ISO timestamp."
        )
        raise ValueError(msg)

    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        msg = f"{clock_text!r} is not a valid time of day"
        raise ValueError(msg)

    if len(parts) > 1:
        chosen = resolve_day(" ".join(parts[:-1]), days)
    else:
        today = current.date()
        chosen = today if today in days else days[0]

    anchor = datetime(chosen.year, chosen.month, chosen.day, hour, minute, tzinfo=zone)
    return Moment(at=anchor, day=chosen, zone=zone, is_during_event=chosen in days)


def parse_location(token: str | None, catalog: Catalog) -> Room | None:
    """Resolve ``--location`` from a room string or a session code.

    Accepting a session code matters in practice: mid-conference you know you are in
    AIM301, not that you are in "Venetian | Level 2 | Titian 2205".
    """
    if not token or not token.strip():
        return None
    session = catalog.resolve(token.strip())
    if session is not None:
        return session.room
    return Room.parse(token)


def _current_session(
    moment: Moment,
    plan: Plan | None,
    schedule: AttendeeSchedule | None,
    catalog: Catalog,
) -> PlanItem | None:
    """Whatever you are sitting in right now, if anything."""
    if plan is not None:
        for item in plan.items:
            if item.span.contains(moment.at):
                return item
    if schedule is not None:
        for session_id in schedule.reserved_ids:
            session = catalog.by_id(session_id)
            if session is None or session.span is None:
                continue
            if session.span.contains(moment.at):
                return PlanItem(
                    session_id=session.session_id,
                    code=session.code,
                    title=session.title,
                    span=session.span,
                    room_display=session.room.display,
                    score=0.0,
                    already_reserved=True,
                    timezone_name=session.timezone_name,
                )
    return None


def _reachable(
    session: Session,
    *,
    from_room: Room | None,
    earliest: datetime,
    travel: TravelGraph,
    tolerance: int,
) -> tuple[bool, int]:
    """Whether you could still walk in, and the travel that would take."""
    if session.span is None:
        return False, 0
    if from_room is None:
        # No position given, so only the clock constrains us.
        return session.span.start >= earliest, 0

    leg = travel.leg(from_room, session.room)
    needed = leg.minutes + travel.entry_overhead_for(session)
    if needed > tolerance:
        return False, needed
    return session.span.start >= earliest + timedelta(minutes=needed), needed


def replan_now(
    catalog: Catalog,
    profile: Profile,
    travel: TravelGraph,
    *,
    at: str | None = None,
    location: str | None = None,
    schedule: AttendeeSchedule | None = None,
    existing_plan: Plan | None = None,
    reference: datetime | None = None,
    limit: int = 3,
    time_limit_seconds: float = 5.0,
) -> NowReport:
    """Re-solve the rest of today from where you are standing."""
    moment = parse_moment(at, catalog, reference=reference)
    here = parse_location(location, catalog)
    notes: list[str] = []

    if not moment.is_during_event:
        notes.append(
            f"{moment.at.astimezone(moment.zone):%Y-%m-%d} is not a day of this event; "
            f"showing {moment.day.isoformat()} instead"
        )

    in_progress = _current_session(moment, existing_plan, schedule, catalog)

    # If you are in a session, you cannot leave before it ends, and you will be leaving
    # from *its* room rather than wherever you were when you asked.
    earliest = moment.at
    if in_progress is not None:
        earliest = in_progress.span.end
        current_session = catalog.by_id(in_progress.session_id)
        if current_session is not None:
            here = current_session.room
        notes.append(
            f"you are in {in_progress.code} until "
            f"{in_progress.span.end.astimezone(moment.zone):%H:%M}"
        )

    tolerance = profile.walk_tolerance_minutes
    candidates: list[Session] = []
    unreachable: list[tuple[Session, int]] = []

    for session in catalog.sessions:
        if session.span is None or session.day != moment.day:
            continue
        if session.span.start < earliest:
            continue
        ok, needed = _reachable(
            session, from_room=here, earliest=earliest, travel=travel, tolerance=tolerance
        )
        if ok:
            candidates.append(session)
        elif needed:
            unreachable.append((session, needed))

    remainder = catalog.replace_sessions(tuple(candidates))

    # Seats you already hold for later today stay pinned: you paid for them with a
    # reservation, and dropping them silently would be the wrong call.
    held_later = frozenset(
        session_id
        for session_id in (schedule.reserved_ids if schedule else frozenset())
        if (found := catalog.by_id(session_id)) is not None
        and found.span is not None
        and found.span.start >= earliest
        and found.day == moment.day
    )
    if held_later:
        codes = sorted(
            catalog.require(session_id).code
            for session_id in held_later
            if catalog.by_id(session_id) is not None
        )
        notes.append(f"holding your existing reservation(s): {', '.join(codes)}")

    scorer = Scorer(profile, remainder)
    request = SolveRequest(
        catalog=remainder,
        profile=profile,
        scores=scorer.by_id(),
        travel=travel,
        schedule=schedule,
        days=frozenset({moment.day}),
        time_limit_seconds=time_limit_seconds,
    )
    plan = solve(request)

    moves: list[NextMove] = []
    for index, item in enumerate(sorted(plan.items, key=lambda entry: entry.span.start)):
        if index >= limit:
            break

        origin = here if index == 0 else None
        if index > 0:
            previous = plan.item_for(item.previous_session_id or "")
            if previous is not None:
                previous_session = catalog.by_id(previous.session_id)
                origin = previous_session.room if previous_session else None

        leg: Leg | None = None
        travel_minutes = 0
        depart_at: datetime | None = None
        minutes_until: int | None = None

        target = catalog.by_id(item.session_id)
        if origin is not None and target is not None:
            leg = travel.leg(origin, target.room)
            travel_minutes = leg.minutes + travel.entry_overhead_for(target)
            depart_at = item.span.start - timedelta(minutes=travel_minutes)
            minutes_until = int((depart_at - max(moment.at, earliest)).total_seconds() // 60)

        if in_progress is not None and index == 0:
            urgency = Urgency.AFTER_CURRENT
        elif travel_minutes == 0:
            # Same room. There is nothing to leave for, so urging someone to move
            # would be nonsense.
            urgency = Urgency.STAY_PUT
        elif minutes_until is None:
            urgency = Urgency.COMFORTABLE
        elif minutes_until <= 0:
            urgency = Urgency.LEAVE_NOW
        elif minutes_until <= 10:
            urgency = Urgency.LEAVE_SOON
        else:
            urgency = Urgency.COMFORTABLE

        moves.append(
            NextMove(
                item=item,
                depart_at=depart_at,
                travel_minutes=travel_minutes,
                leg=leg,
                urgency=urgency,
                minutes_until_departure=minutes_until,
                zone=moment.zone,
            )
        )

    if not moves:
        if here is not None and unreachable:
            notes.append(
                f"{len(unreachable)} session(s) remain today but none are reachable from "
                f"{here.display} in time"
            )
        else:
            notes.append("nothing left on the schedule for today")

    return NowReport(
        moment=moment,
        location=here,
        in_progress=in_progress,
        moves=tuple(moves),
        plan=plan,
        unreachable=tuple(sorted(unreachable, key=lambda pair: pair[1])[:5]),
        notes=tuple(notes),
    )


__all__ = [
    "Moment",
    "NextMove",
    "NowReport",
    "Urgency",
    "parse_location",
    "parse_moment",
    "replan_now",
]
