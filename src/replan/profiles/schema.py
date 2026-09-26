"""The preference profile: a YAML file describing what you want from re:Invent.

Why a file rather than flags. A good agenda takes several iterations to tune, and
re-typing twenty options each time guarantees you never bother. A file can be
edited, diffed, committed, and handed to a colleague who wants your shortlist as
a starting point. It is also what makes crew mode possible: three profiles in, one
coverage plan out.

Two things are deliberately *not* in here. Session times, because those belong to
the catalog. And anything the solver decides, because a profile states preferences,
not conclusions.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from replan.domain import Catalog, Level, SessionFormat, slugify


class ProfileError(ValueError):
    """The profile could not be read, or asks for something contradictory."""


class MealWindow(BaseModel):
    """Time you want protected, which becomes a personal-time block upstream.

    Named "meal window" because that is the common case, but it is really any
    recurring commitment: lunch, a customer meeting, the gym, a daily standup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    start: time
    end: time
    days: tuple[str, ...] = ()
    """Day tokens (``tue``, ``2026-12-01``, ``day2``). Empty means every day."""

    description: str = ""
    location: str | None = None
    protect: bool = True
    """When true the solver may not schedule over it. When false it is advisory."""

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end <= self.start:
            msg = f"meal window {self.title!r} ends at or before it starts"
            raise ValueError(msg)
        return self

    @property
    def minutes(self) -> int:
        return (self.end.hour * 60 + self.end.minute) - (self.start.hour * 60 + self.start.minute)


class ObjectiveWeights(BaseModel):
    """How the solver trades competing goals off against each other.

    Exposed rather than hard-coded because reasonable people want different
    agendas from identical interests: a first-timer wants breadth and slack, a
    returning engineer wants depth and will sprint between rooms for it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    utility: float = Field(default=1.0, ge=0.0, le=10.0)
    """Weight on raw per-session desire."""

    coverage: float = Field(default=0.6, ge=0.0, le=10.0)
    """Reward for covering a *new* topic. This is what stops five near-identical
    Bedrock sessions from beating a balanced day."""

    travel_penalty: float = Field(default=0.4, ge=0.0, le=10.0)
    """Cost per minute of transfer between consecutive sessions."""

    fatigue_penalty: float = Field(default=0.3, ge=0.0, le=10.0)
    """Cost of stacking sessions back to back past your comfort limit."""

    venue_hop_penalty: float = Field(default=0.25, ge=0.0, le=10.0)
    """Cost per property change within a single day."""

    idle_penalty: float = Field(default=0.05, ge=0.0, le=10.0)
    """Mild cost for long dead gaps, so a plan does not spread thin pointlessly."""


class Profile(BaseModel):
    """A complete statement of preference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "me"
    goals: tuple[str, ...] = ()
    """Free text, for your own benefit and quoted back in explanations."""

    # --- what you want --------------------------------------------------- #
    topics: dict[str, float] = Field(default_factory=dict)
    """Keyword -> weight. Matched against topics, tracks, services, areas of
    interest, focus areas, titles and abstracts. Weights are relative."""

    avoid_topics: dict[str, float] = Field(default_factory=dict)
    """Keyword -> penalty weight."""

    levels: dict[int, float] = Field(default_factory=dict)
    """Level -> weight, e.g. ``{300: 1.0, 400: 0.8, 200: 0.3}``."""

    formats: dict[str, float] = Field(default_factory=dict)
    """Format slug -> weight, e.g. ``{chalk-talk: 1.0, breakout: 0.5}``."""

    speakers: tuple[str, ...] = ()
    """Names you will follow regardless of topic."""

    preferred_venues: tuple[str, ...] = ()
    """Properties you would rather stay in, e.g. because your hotel is there."""

    # --- hard constraints ------------------------------------------------ #
    must_attend: tuple[str, ...] = ()
    """Session codes or ids that the plan must include, if they can be reached."""

    never_attend: tuple[str, ...] = ()
    """Session codes or ids to exclude outright."""

    # --- shape of your day ----------------------------------------------- #
    day_start: time | None = None
    day_end: time | None = None
    max_sessions_per_day: int | None = Field(default=None, ge=1, le=20)
    max_consecutive_minutes: int = Field(default=240, ge=30, le=900)
    min_break_minutes: int = Field(default=0, ge=0, le=180)
    walk_tolerance_minutes: int = Field(default=45, ge=0, le=180)
    """Refuse transfers longer than this, even when the clock allows them."""

    meal_windows: tuple[MealWindow, ...] = ()
    include_walk_up: bool = True
    """Whether walk-up sessions (no reservation taken) are worth planning around."""

    include_unreservable: bool = True
    avoid_repeats: bool = False
    """Mildly prefer the first showing of repeated content."""

    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)

    # --- validation ------------------------------------------------------ #

    @field_validator("topics", "avoid_topics", "formats", mode="before")
    @classmethod
    def _coerce_weight_map(cls, value: object) -> object:
        """Accept a bare list as "all weight 1.0", which is how people write it first."""
        if value is None:
            return {}
        if isinstance(value, list | tuple):
            return {str(item): 1.0 for item in value if str(item).strip()}
        return value

    @field_validator("levels", mode="before")
    @classmethod
    def _coerce_levels(cls, value: object) -> object:
        if value is None:
            return {}
        if isinstance(value, list | tuple):
            return {int(item): 1.0 for item in value}
        if isinstance(value, dict):
            return {int(key): float(weight) for key, weight in value.items()}
        return value

    @field_validator(
        "goals", "speakers", "must_attend", "never_attend", "preferred_venues", mode="before"
    )
    @classmethod
    def _coerce_sequence(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value

    @model_validator(mode="after")
    def _check_coherent(self) -> Self:
        overlap = {item.upper() for item in self.must_attend} & {
            item.upper() for item in self.never_attend
        }
        if overlap:
            listed = ", ".join(sorted(overlap))
            msg = f"these sessions are in both must_attend and never_attend: {listed}"
            raise ValueError(msg)
        if self.day_start and self.day_end and self.day_end <= self.day_start:
            msg = "day_end must be after day_start"
            raise ValueError(msg)
        return self

    # --- derived --------------------------------------------------------- #

    @property
    def normalised_topics(self) -> dict[str, float]:
        return {slugify(key): float(weight) for key, weight in self.topics.items() if key.strip()}

    @property
    def normalised_avoid(self) -> dict[str, float]:
        return {
            slugify(key): float(weight) for key, weight in self.avoid_topics.items() if key.strip()
        }

    @property
    def level_weights(self) -> dict[Level, float]:
        resolved: dict[Level, float] = {}
        for raw, weight in self.levels.items():
            resolved[Level.parse(raw)] = float(weight)
        return resolved

    @property
    def format_weights(self) -> dict[SessionFormat, float]:
        return {SessionFormat.parse(key): float(weight) for key, weight in self.formats.items()}

    @property
    def speaker_needles(self) -> tuple[str, ...]:
        return tuple(item.strip().lower() for item in self.speakers if item.strip())

    def resolve_ids(self, tokens: tuple[str, ...], catalog: Catalog) -> frozenset[str]:
        """Turn codes or ids into concrete session ids present in this catalog."""
        found: set[str] = set()
        for token in tokens:
            match = catalog.resolve(token)
            if match is not None:
                found.add(match.session_id)
        return frozenset(found)

    def unresolved(self, tokens: tuple[str, ...], catalog: Catalog) -> tuple[str, ...]:
        """Tokens that match nothing, so a typo in ``must_attend`` is not silent."""
        return tuple(token for token in tokens if catalog.resolve(token) is None)

    def pins(self, catalog: Catalog) -> frozenset[str]:
        return self.resolve_ids(self.must_attend, catalog)

    def bans(self, catalog: Catalog) -> frozenset[str]:
        return self.resolve_ids(self.never_attend, catalog)

    # --- io --------------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, text: str) -> Profile:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as error:
            msg = f"profile is not valid YAML: {error}"
            raise ProfileError(msg) from error
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            msg = "profile must be a YAML mapping at the top level"
            raise ProfileError(msg)
        try:
            return cls.model_validate(raw)
        except ValueError as error:
            raise ProfileError(str(error)) from error

    def to_yaml(self) -> str:
        payload = self.model_dump(mode="json", exclude_defaults=True)
        return str(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def load_profile(path: Path | None) -> Profile:
    """Read a profile, or return sensible defaults when none is given."""
    if path is None:
        return Profile()
    if not path.exists():
        msg = f"profile not found: {path}. Create one with `replan profile init`."
        raise ProfileError(msg)
    return Profile.from_yaml(path.read_text(encoding="utf-8"))


DEFAULT_PROFILE_YAML = """\
# replan preference profile
#
# Everything is optional. Delete what you do not care about; the defaults are
# reasonable. Weights are relative to each other, not absolute, so {a: 2, b: 1}
# and {a: 4, b: 2} mean the same thing.
#
# Tune it, re-run `replan optimize`, and compare. That loop is the point of the
# file existing.

name: me

# For your own reference. Quoted back in explanations so a plan reminds you what
# you were trying to achieve.
goals:
  - Ship an agentic feature next quarter without a rewrite
  - Come back able to argue for or against Bedrock AgentCore
  - Meet two people working on the same problem

# Keyword -> weight. Matched against a session's topics, tracks, services, areas
# of interest, focus areas, title and abstract. Specific beats vague: "Bedrock
# AgentCore" is a better line than "AI".
topics:
  Bedrock AgentCore: 3.0
  agentic: 2.5
  Model Context Protocol: 2.5
  Amazon Bedrock: 2.0
  evaluation: 1.5
  AWS Lambda: 1.5
  observability: 1.2
  cost optimization: 1.0

# Things you would rather not sit through. Applied as a penalty, not a ban.
avoid_topics:
  blockchain: 2.0
  migration: 0.5

# Level -> weight. 100 foundational, 200 intermediate, 300 advanced, 400 expert.
levels:
  400: 1.0
  300: 1.0
  200: 0.4
  100: 0.1

# Format -> weight. Small-group formats are where you actually get answers, but
# they are also the ones that fill first.
formats:
  chalk-talk: 1.5
  code-talk: 1.3
  builders-session: 1.2
  workshop: 1.0
  breakout: 0.8
  keynote: 0.6
  lightning-talk: 0.4

# People worth following regardless of subject.
speakers:
  - Werner Vogels

# Properties you would rather stay in, e.g. where you are sleeping.
preferred_venues:
  - Venetian
  - Caesars Forum

# Codes the plan must include. The solver pins these and builds around them.
must_attend: []

# Codes to exclude outright.
never_attend: []

# The shape of a day you can actually sustain for five days.
day_start: "08:30"
day_end: "19:00"
max_sessions_per_day: 5
max_consecutive_minutes: 210
min_break_minutes: 15

# Refuse any transfer longer than this, even when the clock technically allows
# it. Raise it if you genuinely do not mind crossing the Strip mid-afternoon.
walk_tolerance_minutes: 40

# Time to protect. These become personal-time blocks on your real schedule when
# you run `replan apply`.
meal_windows:
  - title: Lunch
    start: "12:00"
    end: "13:00"
    description: Protected by replan
    protect: true

# Mildly prefer the first showing of repeated content, so you are not holding a
# seat in the last repeat when an earlier one would do.
avoid_repeats: true

# How the solver trades goals off. Raise coverage for breadth, raise
# travel_penalty if you hate walking, raise utility to chase your top hits
# regardless of the cost of getting to them.
weights:
  utility: 1.0
  coverage: 0.6
  travel_penalty: 0.4
  fatigue_penalty: 0.3
  venue_hop_penalty: 0.25
  idle_penalty: 0.05
"""


def write_starter_profile(path: Path, *, force: bool = False) -> Path:
    """Write the commented starter profile, refusing to clobber by default."""
    if path.exists() and not force:
        msg = f"{path} already exists. Pass --force to overwrite it."
        raise ProfileError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_PROFILE_YAML, encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_PROFILE_YAML",
    "MealWindow",
    "ObjectiveWeights",
    "Profile",
    "ProfileError",
    "load_profile",
    "write_starter_profile",
]
