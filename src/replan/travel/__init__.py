"""Travel feasibility across the re:Invent campus.

The dimension the API does not give you. ``venue`` and ``room`` arrive as free
text with no distances attached, so knowing whether a 13:30 in the Venetian can be
followed by a 14:00 in Mandalay Bay is left entirely to the client. It cannot.
"""

from __future__ import annotations

from replan.travel.graph import (
    DEFAULT_DATA_PATH,
    Confidence,
    Feasibility,
    Leg,
    Overheads,
    TravelGraph,
    TravelMode,
    VenueProfile,
    default_graph,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "Confidence",
    "Feasibility",
    "Leg",
    "Overheads",
    "TravelGraph",
    "TravelMode",
    "VenueProfile",
    "default_graph",
]
