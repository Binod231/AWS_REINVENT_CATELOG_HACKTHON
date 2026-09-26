"""Answering "why?" and getting the plan out of the terminal.

Two jobs that sound cosmetic and are not.

**Explaining.** A planner that silently omits the session you were looking forward
to is a planner you stop trusting. Every decision the solver made is recorded, so
:mod:`replan.explain.explainer` can name the exact constraint that removed a
session - and, more usefully, say which line of your profile to change if you
disagree.

**Exporting.** A plan that lives in a terminal is a plan you do not have with you
at the conference. :mod:`replan.explain.exports` writes a real calendar file, and
its alarms fire at *departure* time rather than at the session start, because
knowing you have a 24-minute walk is useless if you find out when the session
begins.
"""

from __future__ import annotations

from replan.explain.explainer import (
    Explanation,
    Verdict,
    explain_session,
    explain_worst,
)
from replan.explain.exports import (
    export_day_sheets_html,
    export_ics,
    export_markdown,
)

__all__ = [
    "Explanation",
    "Verdict",
    "explain_session",
    "explain_worst",
    "export_day_sheets_html",
    "export_ics",
    "export_markdown",
]
