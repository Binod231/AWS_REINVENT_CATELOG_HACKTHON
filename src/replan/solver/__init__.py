"""Turning a ranked list of wants into a schedule you can actually walk.

Ranking is the easy half. The hard half is that the best sessions collide: they
overlap in time, they sit forty minutes apart on the Strip, they repeat content you
already have, and five of them are the same topic. Picking a set that is
simultaneously high-value, conflict-free, travel-feasible and varied is a
constrained optimisation problem, not a sort.

Two implementations satisfy one interface:

* :mod:`replan.solver.cpsat` - a CP-SAT model that solves it properly, including a
  successor-arc formulation so travel time between *consecutive* choices is priced
  rather than guessed.
* :mod:`replan.solver.greedy` - a weighted-interval fallback used when OR-Tools is
  not installed or the solver runs out of time. It is worse, and says so.

Both record *why* every session was left out, which is what makes
``replan why`` able to answer honestly.
"""

from __future__ import annotations

from replan.solver.greedy import solve_greedy
from replan.solver.model import (
    ObjectiveBreakdown,
    Plan,
    PlanItem,
    Rejection,
    RejectionReason,
    SolveRequest,
    SolverKind,
    SolveStats,
)
from replan.solver.schedule import solve

__all__ = [
    "ObjectiveBreakdown",
    "Plan",
    "PlanItem",
    "Rejection",
    "RejectionReason",
    "SolveRequest",
    "SolveStats",
    "SolverKind",
    "solve",
    "solve_greedy",
]
