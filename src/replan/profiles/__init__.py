"""What *you* want out of the conference, and how much each session delivers.

Split deliberately in two:

* :mod:`replan.profiles.schema` - the declarative profile. A YAML file the
  attendee owns, version-controllable, diffable, and shareable with colleagues.
* :mod:`replan.profiles.scorer` - a pure function from (profile, session) to a
  number plus the reasons behind it.

The scorer never touches the network and never makes a scheduling decision. It
answers "how much do I want this?" and nothing else. Whether a wanted session is
*reachable* is the travel graph's problem, and which wanted sessions actually fit
together is the solver's. Keeping those three apart is what makes each of them
testable, and what lets the explainer say precisely which stage rejected a
session.
"""

from __future__ import annotations

from replan.profiles.schema import (
    DEFAULT_PROFILE_YAML,
    PROFILE_FILENAME,
    MealWindow,
    ObjectiveWeights,
    Profile,
    ProfileError,
    discover_profile,
    load_profile,
    write_starter_profile,
)
from replan.profiles.scorer import (
    Factor,
    MatchWhere,
    Score,
    Scorer,
    VetoReason,
)

__all__ = [
    "DEFAULT_PROFILE_YAML",
    "PROFILE_FILENAME",
    "Factor",
    "MatchWhere",
    "MealWindow",
    "ObjectiveWeights",
    "Profile",
    "ProfileError",
    "Score",
    "Scorer",
    "VetoReason",
    "discover_profile",
    "load_profile",
    "write_starter_profile",
]
