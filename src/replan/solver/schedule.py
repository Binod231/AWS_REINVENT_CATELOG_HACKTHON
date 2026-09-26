"""The public entry point: pick a solver, run it, assemble a :class:`Plan`.

Assembly is where the travel legs get attached to the items, because that is a
property of the *final ordering* rather than of the optimisation. It is also where
every unselected candidate gets an explanation, which is the part that makes the
result arguable-with rather than take-it-or-leave-it.
"""

from __future__ import annotations

from datetime import timedelta

from replan.solver.common import build_rejection_for_unselected, prepare
from replan.solver.cpsat import SolverUnavailableError, solve_cpsat
from replan.solver.greedy import solve_greedy
from replan.solver.model import (
    ObjectiveBreakdown,
    Plan,
    PlanItem,
    Rejection,
    SolveRequest,
    SolveStats,
)


def solve(request: SolveRequest, *, prefer_cpsat: bool = True) -> Plan:
    """Solve and return a complete, explained plan."""
    prepared = prepare(request)

    selected: frozenset[int]
    breakdown: ObjectiveBreakdown
    stats: SolveStats

    if prefer_cpsat:
        try:
            selected, breakdown, stats, prepared = solve_cpsat(request, prepared)
        except SolverUnavailableError as error:
            selected, breakdown, stats, prepared = solve_greedy(
                request, prepared, fallback_reason=str(error)
            )
        else:
            # Fall back only when the solver genuinely failed. An empty plan that
            # CP-SAT *proved* optimal is a real answer - nothing on offer was worth
            # the cost of attending it - and greedy would not do better.
            if stats.status not in ("OPTIMAL", "FEASIBLE"):
                selected, breakdown, stats, prepared = solve_greedy(
                    request,
                    prepared,
                    fallback_reason=f"CP-SAT returned {stats.status}; fell back to greedy",
                )
    else:
        selected, breakdown, stats, prepared = solve_greedy(request, prepared)

    ordered = sorted(selected, key=lambda index: prepared.candidate(index).span.start)

    items: list[PlanItem] = []
    previous_by_day: dict[object, int] = {}
    graph = request.travel

    for index in ordered:
        candidate = prepared.candidate(index)
        session = candidate.session
        previous_index = previous_by_day.get(candidate.day)

        travel_minutes: int | None = None
        travel_mode: str | None = None
        travel_detail: str | None = None
        travel_confidence: str | None = None
        depart_by = None
        idle: int | None = None
        previous_id: str | None = None

        if previous_index is not None:
            previous = prepared.candidate(previous_index)
            leg = graph.leg(previous.session.room, session.room)
            needed = leg.minutes + graph.entry_overhead_for(session)
            gap = previous.span.gap_before(candidate.span)
            travel_minutes = needed
            travel_mode = leg.mode.value
            travel_detail = leg.detail
            travel_confidence = leg.confidence.label
            depart_by = candidate.span.start - _minutes(needed)
            idle = max((gap or 0) - needed, 0)
            previous_id = previous.session_id

        items.append(
            PlanItem(
                session_id=session.session_id,
                code=session.code,
                title=session.title,
                span=candidate.span,
                room_display=session.room.display,
                score=candidate.value,
                pinned=candidate.pinned,
                already_reserved=candidate.reserved,
                needs_reservation=session.needs_reservation,
                seat_band=(session.seat_availability.value if session.seat_availability else None),
                timezone_name=session.timezone_name,
                travel_minutes=travel_minutes,
                travel_mode=travel_mode,
                travel_detail=travel_detail,
                travel_confidence=travel_confidence,
                depart_previous_by=depart_by,
                idle_minutes=idle,
                previous_session_id=previous_id,
                top_reasons=candidate.score.top_reasons(2),
            )
        )
        previous_by_day[candidate.day] = index

    rejections: list[Rejection] = list(prepared.pre_rejections)
    for index in range(len(prepared.candidates)):
        if index in selected:
            continue
        rejections.append(build_rejection_for_unselected(request, prepared, index, selected))
    rejections.sort(key=lambda entry: -entry.score)

    warnings: list[str] = []
    unresolved_pins = request.profile.unresolved(request.profile.must_attend, request.catalog)
    if unresolved_pins:
        warnings.append(
            "must_attend entries not found in the catalog: " + ", ".join(unresolved_pins)
        )
    missing_pins = [
        request.catalog.require(session_id).code
        for session_id in sorted(request.profile.pins(request.catalog))
        if session_id not in {item.session_id for item in items}
    ]
    if missing_pins:
        warnings.append(
            "pinned sessions that could not be placed (unschedulable or mutually "
            "exclusive): " + ", ".join(missing_pins)
        )
    if not request.catalog.is_complete:
        warnings.append(
            f"the cached catalog holds {len(request.catalog)} of "
            f"{request.catalog.total_count} sessions, so this plan may be missing options"
        )

    return Plan(
        event_id=request.catalog.event_id,
        profile_name=request.profile.name,
        items=tuple(items),
        rejections=tuple(rejections),
        stats=stats,
        objective=breakdown,
        catalog_source=request.catalog.source,
        catalog_fetched_at=request.catalog.fetched_at,
        timezone_name=(request.catalog.event.timezone_name if request.catalog.event else None),
        goals=request.profile.goals,
        warnings=tuple(warnings),
    )


def _minutes(value: int) -> timedelta:
    return timedelta(minutes=value)


__all__ = ["solve"]
