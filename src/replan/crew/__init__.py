"""Crew mode: multi-attendee coordinated planning for re:Invent.

Enables an entire team to coordinate their conference schedules to maximize
union topic coverage across the company while preventing redundant attendance.
"""

from __future__ import annotations

from replan.crew.export import export_crew_ics, export_crew_markdown
from replan.crew.models import (
    CrewCoverageReport,
    CrewMember,
    CrewPlan,
    TopicCoverage,
)
from replan.crew.solver import (
    CrewSolveRequest,
    compute_crew_coverage,
    solve_crew,
)

__all__ = [
    "CrewCoverageReport",
    "CrewMember",
    "CrewPlan",
    "CrewSolveRequest",
    "TopicCoverage",
    "compute_crew_coverage",
    "export_crew_ics",
    "export_crew_markdown",
    "solve_crew",
]
