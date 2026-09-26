"""The CP-SAT model.

The interesting modelling problem here is travel. Pricing the walk between
*consecutive* sessions is not a property of any single session, and it is not a
property of a pair either, because whether two chosen sessions are consecutive
depends on what else was chosen between them.

The formulation used is a successor-arc model. For each legal same-day transition
``i -> j`` there is a boolean ``succ[i][j]`` meaning "j is the next session after
i". Three families of constraint force those arcs to describe the real chain
rather than whatever is cheapest:

* an arc requires both endpoints be selected;
* each session has at most one successor and at most one predecessor;
* **the arc count per day is pinned to (selected - 1)**.

That last one is what makes it work. Without it the solver would simply set every
arc to zero and pay no travel cost at all. With it, and because a day's selected
sessions are mutually non-overlapping and therefore totally ordered in time, the
only assignment satisfying the degree limits with exactly n-1 arcs is the
time-ordered chain. Travel is then priced exactly.

Coverage uses the other useful trick: one boolean per topic, bounded above by the
number of selected sessions carrying it. Since covering a topic is a reward, the
solver sets it whenever it can, and because a topic can only be covered *once* the
second session on the same subject earns nothing. That is diminishing returns
without any non-linear terms.

Everything is scaled to integers, because CP-SAT objectives are integral.
"""

from __future__ import annotations

import time
from typing import Any

from replan.solver.common import (
    ATTENDANCE_BONUS,
    COVERAGE_POINTS,
    Prepared,
    charged_idle_minutes,
    prepare,
)
from replan.solver.model import ObjectiveBreakdown, SolveRequest, SolverKind, SolveStats

SCALE = 100
"""Fixed-point factor. Scores carry one or two meaningful decimals, so 100 keeps
the objective exact without inflating coefficients into slow territory."""


class SolverUnavailableError(RuntimeError):
    """OR-Tools is not installed."""


def _import_cpsat() -> Any:
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:  # pragma: no cover - depends on extras
        msg = (
            "OR-Tools is not installed. Install it with `uv sync --extra solver` "
            "(or `pip install 'replan[solver]'`), or accept the greedy fallback."
        )
        raise SolverUnavailableError(msg) from error
    return cp_model


def cpsat_available() -> bool:
    try:
        _import_cpsat()
    except SolverUnavailableError:
        return False
    return True


def solve_cpsat(
    request: SolveRequest,
    prepared: Prepared | None = None,
) -> tuple[frozenset[int], ObjectiveBreakdown, SolveStats, Prepared]:
    """Solve, returning the selected candidate indices and the objective parts."""
    cp_model = _import_cpsat()
    work = prepared if prepared is not None else prepare(request)
    weights = request.profile.weights
    started = time.monotonic()

    model = cp_model.CpModel()

    # --- decision variables ---------------------------------------------- #
    chosen = {
        candidate.index: model.new_bool_var(f"x_{candidate.index}") for candidate in work.candidates
    }

    # Pins and sessions you already hold a seat for are not optional.
    for index in work.forced_indices:
        model.add(chosen[index] == 1)

    # --- hard: mutually impossible pairs --------------------------------- #
    for entry in work.incompatible:
        model.add(chosen[entry.first] + chosen[entry.second] <= 1)

    # --- hard: one showing per piece of content --------------------------- #
    for indices in work.series_groups.values():
        model.add(sum(chosen[index] for index in indices) <= 1)

    # --- hard: sessions per day ------------------------------------------- #
    limit = request.profile.max_sessions_per_day
    if limit is not None:
        for indices in work.by_day.values():
            model.add(sum(chosen[index] for index in indices) <= limit)

    # --- successor arcs, so travel can be priced exactly ------------------ #
    arc_vars: dict[tuple[int, int], Any] = {}
    for arc in work.arcs:
        variable = model.new_bool_var(f"succ_{arc.first}_{arc.second}")
        arc_vars[(arc.first, arc.second)] = variable
        # An arc implies both endpoints.
        model.add_implication(variable, chosen[arc.first])
        model.add_implication(variable, chosen[arc.second])

    outgoing: dict[int, list[Any]] = {}
    incoming: dict[int, list[Any]] = {}
    for (first, second), variable in arc_vars.items():
        outgoing.setdefault(first, []).append(variable)
        incoming.setdefault(second, []).append(variable)

    for index, variables in outgoing.items():
        model.add(sum(variables) <= chosen[index])
    for index, variables in incoming.items():
        model.add(sum(variables) <= chosen[index])

    # The constraint that makes the whole thing honest: a day with n selected
    # sessions has exactly n-1 transitions, so the solver cannot dodge the cost.
    for current, indices in work.by_day.items():
        day_arcs = [
            variable
            for (first, second), variable in arc_vars.items()
            if work.candidate(first).day == current and work.candidate(second).day == current
        ]
        selected_count = sum(chosen[index] for index in indices)
        any_selected = model.new_bool_var(f"any_{current.isoformat()}")
        model.add(selected_count >= 1).only_enforce_if(any_selected)
        model.add(selected_count == 0).only_enforce_if(~any_selected)
        if day_arcs:
            model.add(sum(day_arcs) == selected_count - any_selected)
        else:
            # No legal transitions at all, so at most one session can be attended.
            model.add(selected_count <= 1)

    # --- objective -------------------------------------------------------- #
    terms: list[Any] = []

    utility_terms = [
        round((work.candidate(index).value * weights.utility + ATTENDANCE_BONUS) * SCALE)
        * chosen[index]
        for index in chosen
    ]
    terms.extend(utility_terms)

    coverage_vars: dict[str, Any] = {}
    for topic, indices in work.topic_sessions.items():
        covered = model.new_bool_var(f"cov_{topic}")
        # Bounded above by the selection, and rewarded, so it is set exactly when
        # at least one carrier is chosen - and only ever once.
        model.add(covered <= sum(chosen[index] for index in indices))
        coverage_vars[topic] = covered
        terms.append(round(COVERAGE_POINTS * weights.coverage * SCALE) * covered)

    for (first, second), variable in arc_vars.items():
        arc = next(item for item in work.arcs if item.first == first and item.second == second)
        cost = arc.travel_minutes * weights.travel_penalty
        cost += charged_idle_minutes(arc.idle_minutes) * weights.idle_penalty
        if arc.venue_changed:
            cost += 10.0 * weights.venue_hop_penalty
        terms.append(-round(cost * SCALE) * variable)

    # Fatigue: minutes of session time beyond what you said you can sustain.
    fatigue_vars: list[Any] = []
    comfort = request.profile.max_consecutive_minutes
    for current, indices in work.by_day.items():
        total_minutes = sum(work.candidate(index).span.minutes * chosen[index] for index in indices)
        excess = model.new_int_var(0, 24 * 60, f"excess_{current.isoformat()}")
        model.add(excess >= total_minutes - comfort)
        fatigue_vars.append(excess)
        terms.append(-round(weights.fatigue_penalty * SCALE) * excess)

    model.maximize(sum(terms))

    # --- solve ------------------------------------------------------------ #
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = request.time_limit_seconds
    solver.parameters.random_seed = request.random_seed
    # Determinism matters: the same profile and catalog must give the same plan,
    # or "why did this change?" becomes unanswerable.
    solver.parameters.num_workers = 1
    status = solver.solve(model)

    elapsed = time.monotonic() - started
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        stats = SolveStats(
            solver=SolverKind.CPSAT,
            status=status_name,
            optimal=False,
            wall_seconds=elapsed,
            candidates=len(work.candidates),
            considered=len(work.candidates),
            selected=0,
            topics_covered=0,
            topics_available=work.topics_available,
            fallback_reason=f"CP-SAT returned {status_name}",
        )
        return frozenset(), ObjectiveBreakdown(), stats, work

    selected = frozenset(index for index, variable in chosen.items() if solver.value(variable))

    # --- recover the objective in human units ----------------------------- #
    utility = sum(work.candidate(index).value for index in selected)
    covered_topics = [topic for topic, variable in coverage_vars.items() if solver.value(variable)]
    coverage = len(covered_topics) * COVERAGE_POINTS * weights.coverage

    travel_cost = idle_cost = hop_cost = 0.0
    for (first, second), variable in arc_vars.items():
        if not solver.value(variable):
            continue
        arc = next(item for item in work.arcs if item.first == first and item.second == second)
        travel_cost += arc.travel_minutes * weights.travel_penalty
        idle_cost += charged_idle_minutes(arc.idle_minutes) * weights.idle_penalty
        if arc.venue_changed:
            hop_cost += 10.0 * weights.venue_hop_penalty

    fatigue_cost = sum(solver.value(item) for item in fatigue_vars) * weights.fatigue_penalty

    breakdown = ObjectiveBreakdown(
        utility=round(utility * weights.utility, 2),
        coverage=round(coverage, 2),
        travel_cost=round(travel_cost, 2),
        fatigue_cost=round(fatigue_cost, 2),
        hop_cost=round(hop_cost, 2),
        idle_cost=round(idle_cost, 2),
    )

    stats = SolveStats(
        solver=SolverKind.CPSAT,
        status=status_name,
        optimal=status == cp_model.OPTIMAL,
        wall_seconds=elapsed,
        candidates=len(work.candidates),
        considered=len(work.candidates),
        selected=len(selected),
        topics_covered=len(covered_topics),
        topics_available=work.topics_available,
        fallback_reason=(
            None
            if status == cp_model.OPTIMAL
            else "hit the time limit; this is the best solution found so far"
        ),
    )

    return selected, breakdown, stats, work


__all__ = ["SCALE", "SolverUnavailableError", "cpsat_available", "solve_cpsat"]
