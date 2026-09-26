"""Work both solvers need: candidate selection, pair feasibility, topic universe.

Keeping this shared matters for more than tidiness. The CP-SAT model and the greedy
fallback must agree about what is *possible*, or the fallback would produce plans
the real solver considers illegal, and the recorded rejection reasons would
contradict each other.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime

from replan.domain import Session, TimeSpan
from replan.profiles import Score
from replan.solver.model import (
    Rejection,
    RejectionReason,
    SolveRequest,
)
from replan.travel import Leg

# Coverage reward per distinct topic, before the profile's coverage weight.
# Sized relative to the scorer's match points so that covering a new topic is
# worth roughly one solid topic match, which is the trade we want: take a slightly
# weaker session on a *new* subject over a marginally better duplicate.
COVERAGE_POINTS = 9.0

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
"""Sort sentinel. Timezone-aware so it compares cleanly with real session starts."""

# Topics appearing on almost everything carry no information, so rewarding them
# would just add noise to the objective.
_COVERAGE_MAX_FREQUENCY = 0.35

# A topic carried by a single session is the *most* distinctive thing on offer, not
# the least: attending it covers ground nothing else does. Only ubiquity is
# uninformative, so the filter is one-sided.
_COVERAGE_MIN_SESSIONS = 1

# ...but the frequency filter only means something once there are enough
# candidates for "appears on a third of them" to be a real signal. On a small pool
# the cap falls below the minimum and would discard every topic, leaving the
# coverage objective with nothing to reward.
_COVERAGE_FILTER_MIN_POOL = 20

# A small reward for attending anything at all, so that when nothing
# distinguishes two plans the fuller one wins. Without it, a profile that matches
# nothing scores every session at zero, and an empty agenda is genuinely optimal -
# technically correct and completely useless. Kept far below a single topic match
# (10 points) so it can never outvote a real preference, and deliberately below
# the cost of a long transfer, because crossing the Strip for a session you have
# no interest in is not an improvement.
ATTENDANCE_BONUS = 1.0

# Idle time between sessions is not simply bad. A short gap is a breather, a coffee
# and a chance to ask a follow-up question, so the first stretch is free. A very
# long gap is lunch or the expo floor, not dead weight, so the charge is capped.
# Charging linearly for every idle minute would make a plan with one three-hour
# afternoon gap look worse than skipping the afternoon entirely.
IDLE_FREE_MINUTES = 30
IDLE_CHARGED_CAP = 90


def charged_idle_minutes(idle_minutes: int) -> int:
    """Idle minutes the objective actually pays for."""
    return min(max(idle_minutes - IDLE_FREE_MINUTES, 0), IDLE_CHARGED_CAP)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A session in the model, with everything precomputed."""

    index: int
    session: Session
    score: Score
    day: date
    pinned: bool
    reserved: bool

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def span(self) -> TimeSpan:
        assert self.session.span is not None
        return self.session.span

    @property
    def value(self) -> float:
        return self.score.total

    @property
    def forced(self) -> bool:
        """Must be in the plan: you pinned it, or you already hold the seat."""
        return self.pinned or self.reserved


@dataclass(frozen=True, slots=True)
class Arc:
    """A legal transition from one candidate to the next within a day."""

    first: int
    second: int
    travel_minutes: int
    idle_minutes: int
    venue_changed: bool
    leg: Leg


@dataclass(frozen=True, slots=True)
class Incompatible:
    """A pair that cannot both be in the plan."""

    first: int
    second: int
    reason: RejectionReason
    detail: str


@dataclass(frozen=True, slots=True)
class Prepared:
    """The fully analysed problem, ready for either solver."""

    candidates: tuple[Candidate, ...]
    arcs: tuple[Arc, ...]
    incompatible: tuple[Incompatible, ...]
    by_day: dict[date, tuple[int, ...]]
    series_groups: dict[str, tuple[int, ...]]
    topic_sessions: dict[str, tuple[int, ...]]
    pre_rejections: tuple[Rejection, ...]
    protected: tuple[tuple[str, TimeSpan], ...]
    forced_indices: frozenset[int]

    @property
    def topics_available(self) -> int:
        return len(self.topic_sessions)

    def candidate(self, index: int) -> Candidate:
        return self.candidates[index]

    def arcs_by_day(self) -> dict[date, tuple[Arc, ...]]:
        grouped: dict[date, list[Arc]] = defaultdict(list)
        for arc in self.arcs:
            grouped[self.candidates[arc.first].day].append(arc)
        return {key: tuple(value) for key, value in grouped.items()}


def _rejection(session: Session, score: Score, reason: RejectionReason, detail: str) -> Rejection:
    return Rejection(
        session_id=session.session_id,
        code=session.code,
        title=session.title,
        score=score.total,
        reason=reason,
        detail=detail,
    )


def prepare(request: SolveRequest) -> Prepared:
    """Analyse the catalog into a solvable problem, recording what got dropped."""
    pre_rejections: list[Rejection] = []

    # --- vetoed by the profile ------------------------------------------- #
    for session in request.catalog.sessions:
        score = request.scores.get(session.session_id)
        if score is None:
            continue
        if not score.eligible:
            pre_rejections.append(
                _rejection(session, score, RejectionReason.VETOED, score.veto_detail)
            )

    eligible = request.eligible()
    pins = request.profile.pins(request.catalog)
    reserved = request.reserved_ids
    protected = request.protected_spans()

    # --- protected time --------------------------------------------------- #
    survivors: list[Session] = []
    for session in eligible:
        assert session.span is not None
        if session.session_id in pins or session.session_id in reserved:
            # You explicitly asked for this, so it wins over a meal window.
            survivors.append(session)
            continue
        clash = next(
            ((title, span) for title, span in protected if span.overlaps(session.span)),
            None,
        )
        if clash is not None:
            title, span = clash
            score = request.scores[session.session_id]
            pre_rejections.append(
                _rejection(
                    session,
                    score,
                    RejectionReason.PROTECTED_TIME,
                    f"overlaps {title} ({span.format_in(request.zone)})",
                )
            )
            continue
        survivors.append(session)

    # --- candidate cap ---------------------------------------------------- #
    per_day: dict[date, list[Session]] = defaultdict(list)
    for session in survivors:
        current = session.day
        assert current is not None
        per_day[current].append(session)

    chosen: list[Session] = []
    for current, sessions in per_day.items():
        ranked = sorted(
            sessions,
            key=lambda item: (
                -(1 if item.session_id in pins or item.session_id in reserved else 0),
                -request.scores[item.session_id].total,
                item.code,
            ),
        )
        keep = ranked[: request.candidates_per_day]
        dropped = ranked[request.candidates_per_day :]
        chosen.extend(keep)
        cutoff = request.scores[keep[-1].session_id].total if keep else 0.0
        for session in dropped:
            score = request.scores[session.session_id]
            pre_rejections.append(
                _rejection(
                    session,
                    score,
                    RejectionReason.BELOW_CUTOFF,
                    (
                        f"scored {score.total:.0f}, below the top "
                        f"{request.candidates_per_day} for "
                        f"{current.isoformat()} (cutoff {cutoff:.0f})"
                    ),
                )
            )
        _ = current

    # Every survivor has a span (unscheduled sessions were vetoed upstream), so
    # sorting on it is safe; the guard keeps the type checker honest.
    chosen.sort(key=lambda item: (item.span.start if item.span else _EPOCH, item.code))

    candidates = tuple(
        Candidate(
            index=index,
            session=session,
            score=request.scores[session.session_id],
            day=session.day,  # type: ignore[arg-type]
            pinned=session.session_id in pins,
            reserved=session.session_id in reserved,
        )
        for index, session in enumerate(chosen)
    )

    by_day: dict[date, list[int]] = defaultdict(list)
    for candidate in candidates:
        by_day[candidate.day].append(candidate.index)

    series_groups: dict[str, list[int]] = defaultdict(list)
    for candidate in candidates:
        series_groups[candidate.session.series].append(candidate.index)

    arcs, incompatible = _analyse_pairs(request, candidates, by_day)
    topic_sessions = _topic_universe(candidates)

    forced = frozenset(candidate.index for candidate in candidates if candidate.forced)

    return Prepared(
        candidates=candidates,
        arcs=arcs,
        incompatible=incompatible,
        by_day={key: tuple(value) for key, value in sorted(by_day.items())},
        series_groups={key: tuple(value) for key, value in series_groups.items() if len(value) > 1},
        topic_sessions=topic_sessions,
        pre_rejections=tuple(pre_rejections),
        protected=protected,
        forced_indices=forced,
    )


def _analyse_pairs(
    request: SolveRequest,
    candidates: tuple[Candidate, ...],
    by_day: dict[date, list[int]],
) -> tuple[tuple[Arc, ...], tuple[Incompatible, ...]]:
    """Classify every same-day pair as a legal transition or an impossibility."""
    graph = request.travel
    tolerance = request.profile.walk_tolerance_minutes
    min_break = request.profile.min_break_minutes

    arcs: list[Arc] = []
    blocked: list[Incompatible] = []

    for indices in by_day.values():
        ordered = sorted(indices, key=lambda index: candidates[index].span.start)
        for position, first_index in enumerate(ordered):
            first = candidates[first_index]
            for second_index in ordered[position + 1 :]:
                second = candidates[second_index]

                if first.span.overlaps(second.span):
                    blocked.append(
                        Incompatible(
                            first=first_index,
                            second=second_index,
                            reason=RejectionReason.TIME_CONFLICT,
                            detail=(
                                f"overlaps {first.session.code} "
                                f"({first.span.format_in(request.zone)})"
                            ),
                        )
                    )
                    continue

                gap = first.span.gap_before(second.span)
                if gap is None:
                    continue

                leg = graph.leg(first.session.room, second.session.room)
                needed = leg.minutes + graph.entry_overhead_for(second.session)

                if needed > tolerance:
                    blocked.append(
                        Incompatible(
                            first=first_index,
                            second=second_index,
                            reason=RejectionReason.WALK_TOO_FAR,
                            detail=(
                                f"{needed} min from {first.session.code} exceeds your "
                                f"{tolerance} min walk tolerance"
                            ),
                        )
                    )
                    continue

                if gap < needed + min_break:
                    # Phrase this by whichever constraint actually bites. Saying a
                    # transfer "needs 0 min plus a 15 min break" hides the fact
                    # that it is the break setting doing the blocking.
                    if needed == 0:
                        detail = (
                            f"only {gap} min after {first.session.code}, and you asked "
                            f"for a {min_break} min break between sessions"
                        )
                    elif min_break:
                        detail = (
                            f"only {gap} min after {first.session.code}: the transfer "
                            f"needs {needed} min ({leg.mode.label}) plus your "
                            f"{min_break} min break"
                        )
                    else:
                        detail = (
                            f"only {gap} min after {first.session.code}, but the "
                            f"transfer needs {needed} min ({leg.mode.label}: {leg.detail})"
                        )
                    blocked.append(
                        Incompatible(
                            first=first_index,
                            second=second_index,
                            reason=RejectionReason.TRAVEL_INFEASIBLE,
                            detail=detail,
                        )
                    )
                    continue

                arcs.append(
                    Arc(
                        first=first_index,
                        second=second_index,
                        travel_minutes=needed,
                        idle_minutes=max(gap - needed, 0),
                        venue_changed=(
                            first.session.room.known_venue != second.session.room.known_venue
                        ),
                        leg=leg,
                    )
                )

    return tuple(arcs), tuple(blocked)


def _topic_universe(candidates: tuple[Candidate, ...]) -> dict[str, tuple[int, ...]]:
    """Topics worth rewarding coverage of.

    Filtered on both ends. A topic on one session cannot represent breadth, and a
    topic on a third of the catalog is a label rather than a subject, so rewarding
    either would only add noise to the objective.
    """
    counts: dict[str, list[int]] = defaultdict(list)
    for candidate in candidates:
        for key in candidate.session.topic_keys:
            counts[key].append(candidate.index)

    total = len(candidates)
    ceiling = (
        total * _COVERAGE_MAX_FREQUENCY if total >= _COVERAGE_FILTER_MIN_POOL else float(total)
    )
    return {
        key: tuple(indices)
        for key, indices in sorted(counts.items())
        if _COVERAGE_MIN_SESSIONS <= len(indices) <= ceiling
    }


def build_rejection_for_unselected(
    request: SolveRequest,
    prepared: Prepared,
    index: int,
    selected: frozenset[int],
) -> Rejection:
    """Explain why a feasible candidate did not make the final plan.

    Preference order is deliberate: name a concrete blocking session if one
    exists, because "clashes with KEY002" is a real answer and "a better
    combination won" is only true when nothing specific ruled it out.
    """
    candidate = prepared.candidate(index)
    session = candidate.session
    score = candidate.score

    # Another showing of the same content already chosen.
    for sibling in prepared.series_groups.get(session.series, ()):
        if sibling != index and sibling in selected:
            other = prepared.candidate(sibling)
            return Rejection(
                session_id=session.session_id,
                code=session.code,
                title=session.title,
                score=score.total,
                reason=RejectionReason.DUPLICATE_CONTENT,
                detail=(
                    f"{other.session.code} at {other.span.format_in(request.zone)} is the "
                    f"showing of {session.series} already in the plan"
                ),
                conflicts_with=(other.session_id,),
            )

    # A concrete pairwise impossibility with something selected.
    for entry in prepared.incompatible:
        if entry.first == index and entry.second in selected:
            other = prepared.candidate(entry.second)
            return Rejection(
                session_id=session.session_id,
                code=session.code,
                title=session.title,
                score=score.total,
                reason=entry.reason,
                detail=_flip_detail(entry, other, request),
                conflicts_with=(other.session_id,),
            )
        if entry.second == index and entry.first in selected:
            return Rejection(
                session_id=session.session_id,
                code=session.code,
                title=session.title,
                score=score.total,
                reason=entry.reason,
                detail=entry.detail,
                conflicts_with=(prepared.candidate(entry.first).session_id,),
            )

    # The day hit its cap.
    limit = request.profile.max_sessions_per_day
    if limit is not None:
        same_day = sum(1 for other in prepared.by_day.get(candidate.day, ()) if other in selected)
        if same_day >= limit:
            return Rejection(
                session_id=session.session_id,
                code=session.code,
                title=session.title,
                score=score.total,
                reason=RejectionReason.DAY_FULL,
                detail=(
                    f"{candidate.day.isoformat()} already has {same_day} sessions, "
                    f"your max_sessions_per_day"
                ),
            )

    return Rejection(
        session_id=session.session_id,
        code=session.code,
        title=session.title,
        score=score.total,
        reason=RejectionReason.NOT_CHOSEN,
        detail=(
            f"feasible, but scored {score.total:.0f} and the chosen combination "
            "scored higher overall"
        ),
    )


def _flip_detail(entry: Incompatible, other: Candidate, request: SolveRequest) -> str:
    """Re-word a pair conflict from the perspective of the rejected session."""
    if entry.reason is RejectionReason.TIME_CONFLICT:
        return (
            f"overlaps {other.session.code} "
            f"({other.span.format_in(request.zone)}) which is in the plan"
        )
    return entry.detail


__all__ = [
    "COVERAGE_POINTS",
    "Arc",
    "Candidate",
    "Incompatible",
    "Prepared",
    "build_rejection_for_unselected",
    "prepare",
]
