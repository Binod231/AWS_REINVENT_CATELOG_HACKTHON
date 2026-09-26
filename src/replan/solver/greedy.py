"""The fallback solver.

Used when OR-Tools is not installed, or when CP-SAT cannot return a solution. It
takes candidates in descending value and keeps each one that does not break a hard
constraint against what is already held.

It is genuinely worse, and that is reported rather than hidden. Greedy cannot see
that declining a good session unlocks two better ones, and it cannot reason about
topic coverage globally, so it systematically over-picks near-duplicates. The
:class:`~replan.solver.model.SolveStats` it returns says which solver ran, so a
plan is never silently downgraded.

It is still a real solver: it enforces exactly the same hard constraints as the
CP-SAT model, using the same prepared analysis, so its plans are always legal.
"""

from __future__ import annotations

import time

from replan.solver.common import (
    ATTENDANCE_BONUS,
    COVERAGE_POINTS,
    Prepared,
    charged_idle_minutes,
    prepare,
)
from replan.solver.model import (
    ObjectiveBreakdown,
    SolveRequest,
    SolverKind,
    SolveStats,
)


def solve_greedy(
    request: SolveRequest,
    prepared: Prepared | None = None,
    *,
    fallback_reason: str | None = None,
) -> tuple[frozenset[int], ObjectiveBreakdown, SolveStats, Prepared]:
    work = prepared if prepared is not None else prepare(request)
    weights = request.profile.weights
    started = time.monotonic()

    blocked_pairs: set[tuple[int, int]] = set()
    for entry in work.incompatible:
        blocked_pairs.add((entry.first, entry.second))
        blocked_pairs.add((entry.second, entry.first))

    series_of = {
        index: work.candidate(index).session.series for index in range(len(work.candidates))
    }

    selected: set[int] = set()
    used_series: set[str] = set()
    per_day_count: dict[object, int] = {}
    limit = request.profile.max_sessions_per_day

    def conflicts(index: int) -> bool:
        return any((index, held) in blocked_pairs for held in selected)

    def take(index: int) -> None:
        selected.add(index)
        used_series.add(series_of[index])
        day = work.candidate(index).day
        per_day_count[day] = per_day_count.get(day, 0) + 1

    # Forced items first: pins and seats you already hold are not negotiable.
    for index in sorted(work.forced_indices):
        if conflicts(index):
            continue
        take(index)

    # A light coverage heuristic: a session on an already-covered topic is
    # discounted, which imitates the CP-SAT diminishing-returns behaviour without
    # any global reasoning.
    covered: set[str] = set()
    for index in selected:
        covered |= work.candidate(index).session.topic_keys

    remaining = sorted(
        (index for index in range(len(work.candidates)) if index not in selected),
        key=lambda index: (-work.candidate(index).value, work.candidate(index).session.code),
    )

    improved = True
    while improved:
        improved = False
        best_index: int | None = None
        best_value = float("-inf")

        for index in remaining:
            if index in selected:
                continue
            candidate = work.candidate(index)
            if series_of[index] in used_series:
                continue
            if conflicts(index):
                continue
            if limit is not None and per_day_count.get(candidate.day, 0) >= limit:
                continue

            new_topics = candidate.session.topic_keys - covered
            marginal = (
                candidate.value
                + ATTENDANCE_BONUS
                + len(new_topics & set(work.topic_sessions)) * COVERAGE_POINTS * weights.coverage
            )
            if marginal > best_value:
                best_value = marginal
                best_index = index

        if best_index is not None:
            take(best_index)
            covered |= work.candidate(best_index).session.topic_keys
            improved = True

    chosen = frozenset(selected)
    elapsed = time.monotonic() - started

    utility = sum(work.candidate(index).value for index in chosen)
    covered_topics = {
        topic
        for topic, indices in work.topic_sessions.items()
        if any(index in chosen for index in indices)
    }

    travel_cost = idle_cost = hop_cost = 0.0
    for arc in work.arcs:
        if arc.first not in chosen or arc.second not in chosen:
            continue
        # Only price genuinely consecutive pairs: nothing selected in between.
        between = any(
            index in chosen
            and work.candidate(arc.first).span.end
            <= work.candidate(index).span.start
            < work.candidate(arc.second).span.start
            for index in range(len(work.candidates))
            if index not in (arc.first, arc.second)
        )
        if between:
            continue
        travel_cost += arc.travel_minutes * weights.travel_penalty
        idle_cost += charged_idle_minutes(arc.idle_minutes) * weights.idle_penalty
        if arc.venue_changed:
            hop_cost += 10.0 * weights.venue_hop_penalty

    fatigue_cost = 0.0
    comfort = request.profile.max_consecutive_minutes
    for indices in work.by_day.values():
        minutes = sum(work.candidate(index).span.minutes for index in indices if index in chosen)
        fatigue_cost += max(minutes - comfort, 0) * weights.fatigue_penalty

    breakdown = ObjectiveBreakdown(
        utility=round(utility * weights.utility, 2),
        coverage=round(len(covered_topics) * COVERAGE_POINTS * weights.coverage, 2),
        travel_cost=round(travel_cost, 2),
        fatigue_cost=round(fatigue_cost, 2),
        hop_cost=round(hop_cost, 2),
        idle_cost=round(idle_cost, 2),
    )

    stats = SolveStats(
        solver=SolverKind.GREEDY,
        status="greedy-complete",
        optimal=False,
        wall_seconds=elapsed,
        candidates=len(work.candidates),
        considered=len(work.candidates),
        selected=len(chosen),
        topics_covered=len(covered_topics),
        topics_available=work.topics_available,
        fallback_reason=fallback_reason or "greedy solver used; result is not proven optimal",
    )

    return chosen, breakdown, stats, work


__all__ = ["solve_greedy"]
