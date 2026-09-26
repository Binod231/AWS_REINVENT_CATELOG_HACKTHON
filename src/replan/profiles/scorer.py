"""How much do I want this session, and why?

A pure function: no network, no clock, no randomness. Given a profile and a
session it returns a number and the complete list of contributions that produced
it, such that the contributions sum to the number. That property is enforced by a
test, and it is what makes ``replan why`` able to give a real answer instead of a
plausible-sounding one.

Two separations are load-bearing:

* **Desire is not urgency.** A very-limited chalk talk is not more *interesting*
  because it is nearly full; it is more *urgent*. Mixing the two would quietly
  distort a plan toward whatever happened to be filling up. ``Score.urgency`` is
  reported alongside the total, never inside it.
* **Wanting is not eligibility.** A banned session, or one with no published
  time, is vetoed outright rather than scored very low, so the explainer can say
  "excluded because..." rather than "ranked 1,847th".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from replan.domain import Catalog, Level, SeatAvailability, Session, slugify
from replan.profiles.schema import Profile

# Points awarded for a full-strength match on a weight-1.0 keyword, per place the
# match was found. Structured fields are trusted far more than prose: a session
# tagged with the "Amazon Bedrock" service really is about Bedrock, whereas one
# that merely mentions it in an abstract might not be.
_MATCH_POINTS: dict[str, float] = {
    "topic": 10.0,
    "service": 9.0,
    "track": 7.0,
    "area": 7.0,
    "focus": 6.0,
    "title": 6.0,
    "abstract": 2.5,
}

_LEVEL_POINTS = 12.0
_FORMAT_POINTS = 10.0
_SPEAKER_POINTS = 14.0
_VENUE_POINTS = 4.0
_REPEAT_PENALTY = 3.0
_WALKUP_PENALTY = 2.0
_UNRESERVABLE_PENALTY = 1.0
_PIN_BONUS = 1000.0
"""Large enough that a pinned session outranks anything unpinned, without being
infinite - the solver still needs to compare two pins against each other."""


class MatchWhere(StrEnum):
    """Where a keyword matched. Reported so a weak match reads as weak."""

    TOPIC = "topic"
    SERVICE = "service"
    TRACK = "track"
    AREA = "area"
    FOCUS = "focus"
    TITLE = "title"
    ABSTRACT = "abstract"

    @property
    def label(self) -> str:
        return {
            MatchWhere.TOPIC: "tagged topic",
            MatchWhere.SERVICE: "covers the service",
            MatchWhere.TRACK: "in the track",
            MatchWhere.AREA: "area of interest",
            MatchWhere.FOCUS: "focus area",
            MatchWhere.TITLE: "in the title",
            MatchWhere.ABSTRACT: "mentioned in the abstract",
        }[self]


class VetoReason(StrEnum):
    """Why a session is not a candidate at all."""

    BANNED = "banned"
    UNSCHEDULED = "unscheduled"
    OUTSIDE_DAY_WINDOW = "outside-day-window"
    TOO_LONG = "too-long"
    EXCLUDED_WALK_UP = "excluded-walk-up"
    EXCLUDED_UNRESERVABLE = "excluded-unreservable"

    def explain(self, session: Session, profile: Profile) -> str:
        if self is VetoReason.BANNED:
            return f"{session.code} is in never_attend"
        if self is VetoReason.UNSCHEDULED:
            return (
                f"{session.code} has no published time"
                f"{' (all-day entry)' if session.is_all_day else ''}, so it cannot be placed"
            )
        if self is VetoReason.OUTSIDE_DAY_WINDOW:
            window = f"{profile.day_start}-{profile.day_end}"
            return f"{session.code} falls outside your day window ({window})"
        if self is VetoReason.TOO_LONG:
            return (
                f"{session.code} runs {session.duration_minutes} min, longer than your "
                f"max_consecutive_minutes of {profile.max_consecutive_minutes}"
            )
        if self is VetoReason.EXCLUDED_WALK_UP:
            return f"{session.code} is walk-up and include_walk_up is false"
        return f"{session.code} takes no reservation and include_unreservable is false"


@dataclass(frozen=True, slots=True)
class Factor:
    """One contribution to a score."""

    name: str
    points: float
    detail: str

    def __post_init__(self) -> None:
        # Normalise to float so a mix of int and float contributions cannot make
        # the summed total change type.
        object.__setattr__(self, "points", float(self.points))

    def describe(self) -> str:
        sign = "+" if self.points >= 0 else ""
        return f"{sign}{self.points:.1f}  {self.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Score:
    """The verdict on one session."""

    session_id: str
    code: str
    total: float
    factors: tuple[Factor, ...]
    urgency: int
    """0-3, how soon a seat should be grabbed. Deliberately outside ``total``."""

    veto: VetoReason | None = None
    veto_detail: str = ""
    pinned: bool = False

    @property
    def eligible(self) -> bool:
        return self.veto is None

    @property
    def positive_factors(self) -> tuple[Factor, ...]:
        return tuple(item for item in self.factors if item.points > 0)

    @property
    def negative_factors(self) -> tuple[Factor, ...]:
        return tuple(item for item in self.factors if item.points < 0)

    def top_reasons(self, limit: int = 3) -> tuple[str, ...]:
        ranked = sorted(self.factors, key=lambda item: -abs(item.points))
        return tuple(item.detail for item in ranked[:limit])

    def describe(self) -> str:
        if self.veto is not None:
            return f"excluded: {self.veto_detail}"
        lines = [f"{self.code} scores {self.total:.1f}"]
        lines.extend(f"  {item.describe()}" for item in self.factors)
        return "\n".join(lines)


class Scorer:
    """Scores sessions against one profile.

    Constructed per profile per catalog so pins, bans and keyword sets are
    resolved once rather than on every session.
    """

    def __init__(self, profile: Profile, catalog: Catalog) -> None:
        self._profile = profile
        self._catalog = catalog
        self._pins = profile.pins(catalog)
        self._bans = profile.bans(catalog)
        self._topics = profile.normalised_topics
        self._avoid = profile.normalised_avoid
        self._levels = profile.level_weights
        self._formats = profile.format_weights
        self._speakers = profile.speaker_needles
        self._venues = tuple(slugify(item) for item in profile.preferred_venues)
        self._earliest_repeat = self._first_showings(catalog)

    @staticmethod
    def _first_showings(catalog: Catalog) -> dict[str, str]:
        """series -> session_id of its earliest scheduled occurrence."""
        best: dict[str, tuple[object, str]] = {}
        for session in catalog.sessions:
            if session.span is None:
                continue
            current = best.get(session.series)
            if current is None or session.span.start < current[0]:  # type: ignore[operator]
                best[session.series] = (session.span.start, session.session_id)
        return {series: value[1] for series, value in best.items()}

    @property
    def profile(self) -> Profile:
        return self._profile

    @property
    def pinned_ids(self) -> frozenset[str]:
        return self._pins

    @property
    def banned_ids(self) -> frozenset[str]:
        return self._bans

    # --- vetoes ----------------------------------------------------------- #

    def _veto(self, session: Session) -> VetoReason | None:
        if session.session_id in self._bans:
            return VetoReason.BANNED
        # A pinned session skips the soft exclusions: you said you want it.
        pinned = session.session_id in self._pins
        if session.span is None or session.is_all_day:
            return VetoReason.UNSCHEDULED
        if pinned:
            return None
        if (
            session.duration_minutes
            and session.duration_minutes > self._profile.max_consecutive_minutes
        ):
            return VetoReason.TOO_LONG
        if self._outside_window(session):
            return VetoReason.OUTSIDE_DAY_WINDOW
        if (
            not self._profile.include_walk_up
            and session.seat_availability is SeatAvailability.WALK_UP
        ):
            return VetoReason.EXCLUDED_WALK_UP
        if not self._profile.include_unreservable and session.is_reservable is False:
            return VetoReason.EXCLUDED_UNRESERVABLE
        return None

    def _outside_window(self, session: Session) -> bool:
        if session.span is None:
            return False
        start_limit, end_limit = self._profile.day_start, self._profile.day_end
        if start_limit is None and end_limit is None:
            return False
        local_start, local_end = session.span.local(session.zone)
        if start_limit is not None and local_start.time() < start_limit:
            return True
        return bool(end_limit is not None and local_end.time() > end_limit)

    # --- scoring ---------------------------------------------------------- #

    def score(self, session: Session) -> Score:
        veto = self._veto(session)
        pinned = session.session_id in self._pins
        urgency = session.scarcity

        if veto is not None:
            return Score(
                session_id=session.session_id,
                code=session.code,
                total=0.0,
                factors=(),
                urgency=urgency,
                veto=veto,
                veto_detail=veto.explain(session, self._profile),
                pinned=pinned,
            )

        factors: list[Factor] = []

        if pinned:
            factors.append(Factor("must attend", _PIN_BONUS, "you listed this in must_attend"))

        factors.extend(self._topic_factors(session))
        factors.extend(self._avoid_factors(session))

        level_factor = self._level_factor(session)
        if level_factor is not None:
            factors.append(level_factor)

        format_factor = self._format_factor(session)
        if format_factor is not None:
            factors.append(format_factor)

        factors.extend(self._speaker_factors(session))

        venue_factor = self._venue_factor(session)
        if venue_factor is not None:
            factors.append(venue_factor)

        repeat_factor = self._repeat_factor(session)
        if repeat_factor is not None:
            factors.append(repeat_factor)

        seat_factor = self._seat_factor(session)
        if seat_factor is not None:
            factors.append(seat_factor)

        # The invariant the explainer depends on: the parts sum to the whole.
        # float() because sum() of an empty list is int, and a Score whose total
        # is sometimes int and sometimes float is a trap for anything serialising
        # it or comparing types.
        total = float(sum(item.points for item in factors))

        return Score(
            session_id=session.session_id,
            code=session.code,
            total=round(total, 4),
            factors=tuple(factors),
            urgency=urgency,
            pinned=pinned,
        )

    # --- individual contributions ----------------------------------------- #

    def _match_places(self, session: Session, keyword: str) -> list[tuple[MatchWhere, str]]:
        """Every place a keyword appears, strongest signal first.

        Matching is substring-based on slugs, so "bedrock" hits "Amazon Bedrock"
        and "agent" hits "Bedrock AgentCore". That is the behaviour a user expects
        from a keyword list, and exact-match-only would make the feature useless
        against real catalog vocabulary.
        """
        needle = keyword
        if not needle:
            return []
        found: list[tuple[MatchWhere, str]] = []

        def scan(values: tuple[str, ...], where: MatchWhere) -> None:
            for value in values:
                if needle in slugify(value):
                    found.append((where, value))
                    return

        scan(session.topics, MatchWhere.TOPIC)
        scan(session.services, MatchWhere.SERVICE)
        scan(session.effective_tracks, MatchWhere.TRACK)
        scan(session.areas_of_interest, MatchWhere.AREA)
        scan(session.focus_areas, MatchWhere.FOCUS)
        if needle in slugify(session.title):
            found.append((MatchWhere.TITLE, session.title))
        if session.abstract and needle in slugify(session.abstract):
            found.append((MatchWhere.ABSTRACT, "abstract"))
        return found

    def _topic_factors(self, session: Session) -> list[Factor]:
        factors: list[Factor] = []
        for keyword, weight in sorted(self._topics.items()):
            places = self._match_places(session, keyword)
            if not places:
                continue
            # Only the strongest place counts, so a keyword repeated across five
            # fields cannot dominate the whole score.
            where, value = max(places, key=lambda item: _MATCH_POINTS[item[0].value])
            points = _MATCH_POINTS[where.value] * weight
            readable = keyword.replace("-", " ")
            factors.append(
                Factor(
                    f"topic: {readable}",
                    round(points, 4),
                    f"{where.label} ({value[:60]})",
                )
            )
        return factors

    def _avoid_factors(self, session: Session) -> list[Factor]:
        factors: list[Factor] = []
        for keyword, weight in sorted(self._avoid.items()):
            places = self._match_places(session, keyword)
            if not places:
                continue
            where, value = max(places, key=lambda item: _MATCH_POINTS[item[0].value])
            points = -_MATCH_POINTS[where.value] * weight
            readable = keyword.replace("-", " ")
            factors.append(
                Factor(
                    f"avoided topic: {readable}",
                    round(points, 4),
                    f"{where.label} ({value[:60]})",
                )
            )
        return factors

    def _level_factor(self, session: Session) -> Factor | None:
        if not self._levels:
            return None
        weight = self._levels.get(session.level)
        if weight is None:
            if session.level is Level.UNKNOWN:
                return None
            return Factor(
                "level",
                round(-_LEVEL_POINTS * 0.25, 4),
                f"{session.level_raw or session.level.label} is not in your levels",
            )
        return Factor(
            "level",
            round(_LEVEL_POINTS * weight, 4),
            f"{session.level_raw or session.level.label} matches your preference",
        )

    def _format_factor(self, session: Session) -> Factor | None:
        if not self._formats:
            return None
        weight = self._formats.get(session.format)
        if weight is None:
            return None
        return Factor(
            "format",
            round(_FORMAT_POINTS * weight, 4),
            f"{session.type_raw or session.format.label} is a format you favour",
        )

    def _speaker_factors(self, session: Session) -> list[Factor]:
        if not self._speakers:
            return []
        factors: list[Factor] = []
        for needle in self._speakers:
            for speaker in session.speakers:
                if needle in speaker.name.lower():
                    factors.append(
                        Factor("speaker", _SPEAKER_POINTS, f"{speaker.person} is speaking")
                    )
                    break
        return factors

    def _venue_factor(self, session: Session) -> Factor | None:
        if not self._venues:
            return None
        venue = session.room.known_venue
        label = session.room.venue or (venue.label if venue else "")
        haystack = slugify(label)
        if not haystack:
            return None
        for needle in self._venues:
            if needle and needle in haystack:
                return Factor("venue", _VENUE_POINTS, f"{label} is convenient for you")
        return None

    def _repeat_factor(self, session: Session) -> Factor | None:
        if not self._profile.avoid_repeats:
            return None
        first = self._earliest_repeat.get(session.series)
        if first is None or first == session.session_id:
            return None
        return Factor(
            "repeat",
            -_REPEAT_PENALTY,
            f"a later showing of {session.series}; an earlier one exists",
        )

    def _seat_factor(self, session: Session) -> Factor | None:
        """Small nudges for seat mechanics, never a proxy for interest."""
        if session.seat_availability is SeatAvailability.WALK_UP:
            return Factor(
                "seating",
                -_WALKUP_PENALTY,
                "walk-up, so attendance is not guaranteed by planning",
            )
        if session.is_reservable is False:
            return Factor("seating", -_UNRESERVABLE_PENALTY, "takes no reservation")
        return None

    # --- bulk --------------------------------------------------------------#

    def score_all(self, *, include_vetoed: bool = False) -> tuple[Score, ...]:
        scores = [self.score(session) for session in self._catalog.sessions]
        if not include_vetoed:
            scores = [item for item in scores if item.eligible]
        scores.sort(key=lambda item: (-item.total, item.code))
        return tuple(scores)

    def ranked(self, limit: int | None = None) -> tuple[Score, ...]:
        scores = self.score_all()
        return scores[:limit] if limit else scores

    def by_id(self) -> dict[str, Score]:
        return {score.session_id: score for score in self.score_all(include_vetoed=True)}


__all__ = ["Factor", "MatchWhere", "Score", "Scorer", "VetoReason"]
