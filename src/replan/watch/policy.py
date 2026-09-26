"""Should this seat be taken?

A seat opening up is not the same as a seat being worth having. The policy exists so
an automated reservation can never make the schedule worse than leaving it alone, and
every refusal comes with a reason that reads as a sentence.

The checks, in the order they run and why that order:

1. **Has it already been handled?** Cheapest check, and it stops duplicate work.
2. **Is the band actually bookable?** ``walkUp`` means no reservation is taken at all,
   so there is nothing to grab.
3. **Has it started?** A seat in a session that began twenty minutes ago is worthless.
4. **Do you already hold it?** Reserving twice is refused by the API anyway.
5. **Does it clash with a seat you hold?** This is the important one. Taking a seat that
   overlaps something you already reserved trades a certainty for a conflict.
6. **Can you get there?** Non-overlapping is not the same as reachable. The session
   before it might be forty minutes away.
7. **Is it worth it?** A session that scores below your threshold is not worth an
   automated booking, even if it is free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from replan.domain import (
    AttendeeSchedule,
    Catalog,
    SeatAvailability,
    Session,
    now_utc,
)
from replan.profiles import Profile, Score
from replan.travel import TravelGraph
from replan.watch.store import WatchTarget


class Verdict(StrEnum):
    """What the watcher should do about an opening."""

    RESERVE = "reserve"
    """Take it."""

    NOTIFY_ONLY = "notify-only"
    """Tell the attendee, but do not book. Used when reserving is closed, or when the
    decision is one a human should make."""

    IGNORE = "ignore"
    """Nothing to do."""

    @property
    def should_notify(self) -> bool:
        return self in {Verdict.RESERVE, Verdict.NOTIFY_ONLY}


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of evaluating one opening."""

    verdict: Verdict
    reason: str
    session_id: str
    code: str
    score: float = 0.0
    conflicts_with: tuple[str, ...] = ()

    @property
    def should_reserve(self) -> bool:
        return self.verdict is Verdict.RESERVE

    def describe(self) -> str:
        return f"{self.code}: {self.verdict.value} - {self.reason}"


@dataclass(frozen=True, slots=True)
class WatchPolicy:
    """Rules for automated seat grabbing."""

    auto_reserve: bool = False
    """Off by default. Spending a reservation without being asked is a big step, so it
    is opt-in even though the watcher is useful without it."""

    minimum_score: float = 0.0
    """Skip openings scoring below this. Zero means "anything on the watch list"."""

    require_travel_feasible: bool = True
    allow_conflicts: bool = False
    reservations_open: bool = True
    """Set false to force notify-only, which is the situation until 8 October 2026."""

    def evaluate(
        self,
        session: Session,
        target: WatchTarget,
        *,
        schedule: AttendeeSchedule,
        catalog: Catalog,
        travel: TravelGraph,
        profile: Profile,
        score: Score | None = None,
        reference: datetime | None = None,
    ) -> Decision:
        """Decide what to do about a session whose seat band just improved."""
        code = session.code
        value = score.total if score is not None else 0.0

        if target.reserved_at is not None:
            return Decision(
                verdict=Verdict.IGNORE,
                reason="already reserved by a previous run",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        band = session.seat_availability
        if band is SeatAvailability.WALK_UP:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason="walk-up session, so there is no seat to reserve",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        if band is not None and not band.is_bookable:
            return Decision(
                verdict=Verdict.IGNORE,
                reason=f"still {band.value}",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        if session.is_reservable is False:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason="the catalog says this session takes no reservations",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        moment = reference or now_utc()
        if session.span is None:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason="no published time, so it cannot be checked against your schedule",
                session_id=session.session_id,
                code=code,
                score=value,
            )
        if session.span.start <= moment:
            return Decision(
                verdict=Verdict.IGNORE,
                reason="already started",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        if session.session_id in schedule.reserved_ids:
            return Decision(
                verdict=Verdict.IGNORE,
                reason="you already hold a seat",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        # --- the checks that protect the existing schedule ------------------ #
        held = [
            other
            for other in (catalog.by_id(item) for item in sorted(schedule.reserved_ids))
            if other is not None and other.span is not None
        ]

        clashing = [
            other.code
            for other in held
            if other.span is not None and other.span.overlaps(session.span)
        ]
        if clashing and not self.allow_conflicts:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason=(
                    "it overlaps a seat you already hold ("
                    + ", ".join(clashing)
                    + "), so replan will not trade a certainty for a clash"
                ),
                session_id=session.session_id,
                code=code,
                score=value,
                conflicts_with=tuple(clashing),
            )

        if self.require_travel_feasible:
            unreachable = self._first_unreachable(session, held, travel, profile)
            if unreachable is not None:
                other_code, detail = unreachable
                return Decision(
                    verdict=Verdict.NOTIFY_ONLY,
                    reason=f"you could not get there from {other_code}: {detail}",
                    session_id=session.session_id,
                    code=code,
                    score=value,
                    conflicts_with=(other_code,),
                )

        if value < self.minimum_score:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason=(
                    f"scores {value:.0f}, below your auto-reserve threshold of "
                    f"{self.minimum_score:.0f}"
                ),
                session_id=session.session_id,
                code=code,
                score=value,
            )

        if not self.reservations_open:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason=(
                    "a seat is free, but reservations do not open through the API until "
                    "8 October 2026 - book it in the app"
                ),
                session_id=session.session_id,
                code=code,
                score=value,
            )

        if not self.auto_reserve:
            return Decision(
                verdict=Verdict.NOTIFY_ONLY,
                reason="a seat is free; pass --auto-reserve to have replan take it",
                session_id=session.session_id,
                code=code,
                score=value,
            )

        return Decision(
            verdict=Verdict.RESERVE,
            reason=f"a seat is free and it scores {value:.0f}",
            session_id=session.session_id,
            code=code,
            score=value,
        )

    def _first_unreachable(
        self,
        session: Session,
        held: list[Session],
        travel: TravelGraph,
        profile: Profile,
    ) -> tuple[str, str] | None:
        """The first held session this one could not be combined with, if any.

        Only adjacent-in-time held sessions matter: a seat at 09:00 says nothing about
        whether you can reach a 16:00, and checking every pair would reject far too much.
        """
        if session.span is None:
            return None
        same_day = [other for other in held if other.span is not None and other.day == session.day]
        if not same_day:
            return None

        before = [
            other
            for other in same_day
            if other.span is not None and other.span.end <= session.span.start
        ]
        after = [
            other
            for other in same_day
            if other.span is not None and other.span.start >= session.span.end
        ]

        neighbours: list[Session] = []
        if before:
            neighbours.append(max(before, key=lambda item: item.span.start))  # type: ignore[union-attr]
        if after:
            neighbours.append(min(after, key=lambda item: item.span.start))  # type: ignore[union-attr]

        for neighbour in neighbours:
            result = travel.feasibility(neighbour, session, extra_buffer=profile.min_break_minutes)
            if not result.ok:
                return neighbour.code, result.reason
            if result.leg.minutes > profile.walk_tolerance_minutes:
                return (
                    neighbour.code,
                    f"{result.leg.minutes} min exceeds your "
                    f"{profile.walk_tolerance_minutes} min walk tolerance",
                )
        return None


def band_improved(previous: SeatAvailability | None, current: SeatAvailability | None) -> bool:
    """Whether the seat situation got better since last time.

    This is the entire detection mechanism, and it is a comparison rather than a
    threshold because the API publishes no seat count. Going from ``unavailable`` to
    ``limited`` is an opening; ``limited`` to ``veryLimited`` is not.
    """
    if current is None:
        return False
    if not current.is_bookable:
        return False
    if previous is None:
        # First sighting of a bookable seat still counts: the session was added to the
        # watch list precisely because someone wanted it.
        return True
    if not previous.is_bookable:
        return True
    # Both bookable: report only a genuine loosening, so a session hovering between
    # limited and veryLimited does not notify on every poll.
    return current.urgency < previous.urgency


__all__ = ["Decision", "Verdict", "WatchPolicy", "band_improved"]
