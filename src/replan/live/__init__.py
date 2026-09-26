"""Re-planning from where you are standing, right now.

An agenda built in November is a plan. By Wednesday afternoon it is a historical
document: a keynote overran, you skipped something to finish a conversation, the room
you wanted filled up. What you need at 14:10 on the show floor is not the whole week,
it is the next two moves and the time you have to stand up.

That is a different question from the one :mod:`replan.solver` answers, and it needs
two extra inputs the original solve never had: the current moment, and where you
physically are. Both change what is reachable. A 14:30 session across the Strip was
always in the catalog; whether it is available to you depends entirely on whether you
are in the Venetian or Mandalay Bay when you ask.
"""

from __future__ import annotations

from replan.live.now import (
    Moment,
    NextMove,
    NowReport,
    Urgency,
    parse_location,
    parse_moment,
    replan_now,
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
