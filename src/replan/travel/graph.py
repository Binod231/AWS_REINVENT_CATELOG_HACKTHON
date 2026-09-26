"""Can you actually get there in time?

The AWS Events API tells you where a session is, as free text, and nothing at all
about how far that is from anywhere else. re:Invent runs across roughly four miles
of Las Vegas Boulevard, so a schedule that looks fine in a list view can be
physically impossible: a 13:30 chalk talk in the Venetian and a 14:00 workshop in
Mandalay Bay do not overlap, and cannot both be attended.

This module supplies the missing dimension. It answers three questions:

* ``leg`` - how long does this specific transfer take, by what mode, and how
  confident are we?
* ``feasibility`` - given two scheduled sessions, is the connection makeable, and
  by what margin?
* ``depart_by`` - what time do I actually need to stand up?

Two design commitments worth stating. First, **confidence is part of the answer**:
a curated pair is reported differently from a zone fallback, and from a venue we
do not recognise at all, so the explainer never presents a guess as a measurement.
Second, **every number is overridable**, because an attendee who has walked the
route knows better than this file does.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Self

import yaml

from replan.domain import KnownVenue, Room, Session

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "venues.yaml"


class TravelMode(StrEnum):
    STAY = "stay"
    """Same room. The only free transfer."""

    WALK_INTERNAL = "walk-internal"
    """Room to room inside one property."""

    WALK = "walk"
    """On foot between properties."""

    SHUTTLE = "shuttle"
    """The re:Invent inter-venue shuttle, including the wait."""

    @property
    def label(self) -> str:
        return {
            TravelMode.STAY: "stay put",
            TravelMode.WALK_INTERNAL: "walk inside the building",
            TravelMode.WALK: "walk",
            TravelMode.SHUTTLE: "shuttle",
        }[self]


class Confidence(IntEnum):
    """How much to trust a travel estimate.

    Surfaced rather than hidden: a planner that presents a guessed number and a
    curated one identically is lying by omission.
    """

    CURATED = 3
    """A specific pair we have data for."""

    ZONE_DEFAULT = 2
    """Both venues recognised, but this pair was not listed."""

    SAME_BUILDING_ASSUMED = 1
    """Venue unknown; assumed one building, which is right for single-site events."""

    UNKNOWN = 0
    """Venue strings differ but neither is recognised. Pessimistic guess."""

    @property
    def label(self) -> str:
        return {
            Confidence.CURATED: "estimated from campus layout",
            Confidence.ZONE_DEFAULT: "approximated from venue zones",
            Confidence.SAME_BUILDING_ASSUMED: "assumed same building",
            Confidence.UNKNOWN: "unknown venues, pessimistic guess",
        }[self]


@dataclass(frozen=True, slots=True)
class Leg:
    """One transfer between two rooms."""

    minutes: int
    mode: TravelMode
    confidence: Confidence
    origin: Room
    destination: Room
    transit_minutes: int = 0
    overhead_minutes: int = 0
    detail: str = ""

    @property
    def is_free(self) -> bool:
        return self.minutes == 0

    def describe(self) -> str:
        if self.mode is TravelMode.STAY:
            return "same room, no transfer needed"
        where = f"{self.origin.display} -> {self.destination.display}"
        body = (
            f"{self.minutes} min by {self.mode.label} "
            f"({self.transit_minutes} min transit + {self.overhead_minutes} min "
            f"getting out and in)"
        )
        suffix = f" [{self.confidence.label}]"
        return f"{where}: {body}{suffix}"


@dataclass(frozen=True, slots=True)
class Feasibility:
    """Whether one session can follow another."""

    leg: Leg
    gap_minutes: int
    """Clear minutes between the first session ending and the second starting."""

    slack_minutes: int
    """Gap minus travel. Negative means impossible."""

    ok: bool
    reason: str

    @property
    def is_tight(self) -> bool:
        """Makeable, but with nothing to spare."""
        return self.ok and self.slack_minutes <= 5


@dataclass(frozen=True, slots=True)
class VenueProfile:
    venue: KnownVenue
    label: str
    zone: str
    internal_typical: int
    internal_max: int
    per_floor_change: int
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Overheads:
    room_exit: int = 3
    room_entry: int = 4
    shuttle_wait: int = 9
    reserved_seat_entry_bonus: int = 3

    @property
    def fixed(self) -> int:
        return self.room_exit + self.room_entry


@dataclass(frozen=True, slots=True)
class TravelGraph:
    """Travel times across the campus, loaded from YAML and cached in memory."""

    venues: dict[KnownVenue, VenueProfile]
    pairs: dict[tuple[KnownVenue, KnownVenue], dict[str, int]]
    zone_defaults: dict[tuple[str, str], dict[str, int]]
    unknown_same_building: int
    unknown_different_building: int
    overheads: Overheads
    source: str = "bundled"
    _cache: dict[tuple[str, str], Leg] = field(default_factory=dict, compare=False, repr=False)

    # --- loading ---------------------------------------------------------- #

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        overrides: Path | None = None,
    ) -> Self:
        """Load the bundled data, then apply an override file on top."""
        base_path = path or DEFAULT_DATA_PATH
        raw = cls._read(base_path)
        source = base_path.name
        if overrides is not None:
            if not overrides.exists():
                msg = f"travel override file not found: {overrides}"
                raise FileNotFoundError(msg)
            raw = cls._merge(raw, cls._read(overrides))
            source = f"{base_path.name} + {overrides.name}"
        return cls._build(raw, source=source)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            msg = f"{path} does not contain a YAML mapping"
            raise ValueError(msg)
        return loaded

    @staticmethod
    def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        """Shallow-merge per section, so an override can change one pair."""
        merged = dict(base)
        for key, value in patch.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                combined = dict(existing)
                combined.update(value)
                merged[key] = combined
            else:
                merged[key] = value
        return merged

    @classmethod
    def _build(cls, raw: dict[str, Any], *, source: str) -> Self:
        venues: dict[KnownVenue, VenueProfile] = {}
        for key, payload in (raw.get("venues") or {}).items():
            resolved = KnownVenue.parse(key)
            if resolved is None:
                continue
            internal = payload.get("internal") or {}
            venues[resolved] = VenueProfile(
                venue=resolved,
                label=str(payload.get("label") or resolved.label),
                zone=str(payload.get("zone") or "central"),
                internal_typical=int(internal.get("typical", 9)),
                internal_max=int(internal.get("max", 15)),
                per_floor_change=int(internal.get("per_floor_change", 2)),
                note=payload.get("note"),
            )

        pairs: dict[tuple[KnownVenue, KnownVenue], dict[str, int]] = {}
        for key, payload in (raw.get("pairs") or {}).items():
            left_raw, _, right_raw = str(key).partition("|")
            left, right = KnownVenue.parse(left_raw), KnownVenue.parse(right_raw)
            if left is None or right is None:
                continue
            times = {
                name: int(value)
                for name, value in (payload or {}).items()
                if isinstance(value, int | float)
            }
            pairs[cls._pair_key(left, right)] = times

        zone_defaults: dict[tuple[str, str], dict[str, int]] = {}
        for key, payload in (raw.get("zone_defaults") or {}).items():
            left_raw, _, right_raw = str(key).partition("|")
            zone_key = tuple(sorted((left_raw.strip(), right_raw.strip())))
            zone_defaults[(zone_key[0], zone_key[1])] = {
                name: int(value)
                for name, value in (payload or {}).items()
                if isinstance(value, int | float)
            }

        unknown = raw.get("unknown_venue") or {}
        overhead_payload = raw.get("overheads") or {}

        return cls(
            venues=venues,
            pairs=pairs,
            zone_defaults=zone_defaults,
            unknown_same_building=int(unknown.get("same_building", 10)),
            unknown_different_building=int(unknown.get("different_building", 30)),
            overheads=Overheads(
                room_exit=int(overhead_payload.get("room_exit", 3)),
                room_entry=int(overhead_payload.get("room_entry", 4)),
                shuttle_wait=int(overhead_payload.get("shuttle_wait", 9)),
                reserved_seat_entry_bonus=int(overhead_payload.get("reserved_seat_entry_bonus", 3)),
            ),
            source=source,
        )

    @staticmethod
    def _pair_key(left: KnownVenue, right: KnownVenue) -> tuple[KnownVenue, KnownVenue]:
        """Symmetric key. Travel time is assumed the same in both directions."""
        return (left, right) if left.value <= right.value else (right, left)

    # --- the core question ------------------------------------------------ #

    def leg(self, origin: Room, destination: Room) -> Leg:
        """How long to get from one room to another, and by what means.

        Memoised on room keys: the solver asks this for every ordered pair of
        candidate sessions, which on a real catalog is hundreds of thousands of
        calls against a few hundred distinct rooms.
        """
        cache_key = (origin.key, destination.key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        computed = self._compute_leg(origin, destination)
        self._cache[cache_key] = computed
        return computed

    def _compute_leg(self, origin: Room, destination: Room) -> Leg:
        if origin.key == destination.key:
            return Leg(
                minutes=0,
                mode=TravelMode.STAY,
                confidence=Confidence.CURATED,
                origin=origin,
                destination=destination,
                detail="same room",
            )

        left, right = origin.known_venue, destination.known_venue

        # Same recognised property: internal walk plus a floor-change penalty.
        if left is not None and left == right:
            return self._internal_leg(origin, destination, left)

        # Neither venue recognised. Single-site events (Summits) publish no venue
        # at all, and treating that as a cross-Strip journey would be absurd.
        if left is None and right is None:
            same_building = (origin.venue or "") == (destination.venue or "")
            transit = (
                self.unknown_same_building if same_building else self.unknown_different_building
            )
            return Leg(
                minutes=transit + self.overheads.fixed,
                mode=TravelMode.WALK_INTERNAL if same_building else TravelMode.WALK,
                confidence=(
                    Confidence.SAME_BUILDING_ASSUMED if same_building else Confidence.UNKNOWN
                ),
                origin=origin,
                destination=destination,
                transit_minutes=transit,
                overhead_minutes=self.overheads.fixed,
                detail=(
                    "venue not published; assumed one building"
                    if same_building
                    else "neither venue recognised; pessimistic estimate"
                ),
            )

        # One side recognised and the other not: cannot route it properly, so be
        # pessimistic rather than optimistic.
        if left is None or right is None:
            transit = self.unknown_different_building
            return Leg(
                minutes=transit + self.overheads.fixed,
                mode=TravelMode.WALK,
                confidence=Confidence.UNKNOWN,
                origin=origin,
                destination=destination,
                transit_minutes=transit,
                overhead_minutes=self.overheads.fixed,
                detail="only one venue recognised; pessimistic estimate",
            )

        return self._between_venues(origin, destination, left, right)

    def _internal_leg(self, origin: Room, destination: Room, venue: KnownVenue) -> Leg:
        profile = self.venues.get(venue)
        typical = profile.internal_typical if profile else 9
        per_floor = profile.per_floor_change if profile else 2

        floors = self._floor_change(origin, destination)
        transit = typical + (per_floor * floors)
        if profile is not None:
            # Never claim an internal transfer is worse than crossing the whole
            # property plus every floor.
            transit = min(transit, profile.internal_max + per_floor * floors)

        detail = f"inside {profile.label if profile else venue.label}"
        if floors:
            detail += f", {floors} floor change(s)"

        return Leg(
            minutes=transit + self.overheads.fixed,
            mode=TravelMode.WALK_INTERNAL,
            confidence=Confidence.CURATED,
            origin=origin,
            destination=destination,
            transit_minutes=transit,
            overhead_minutes=self.overheads.fixed,
            detail=detail,
        )

    @staticmethod
    def _floor_change(origin: Room, destination: Room) -> int:
        left, right = origin.floor_hint, destination.floor_hint
        if left is None or right is None:
            return 0
        try:
            return abs(int(left) - int(right))
        except ValueError:
            return 0

    def _between_venues(
        self,
        origin: Room,
        destination: Room,
        left: KnownVenue,
        right: KnownVenue,
    ) -> Leg:
        times, confidence = self._times_for(left, right)
        walk = times.get("walk")
        shuttle = times.get("shuttle")

        options: list[tuple[int, TravelMode, int]] = []
        if walk is not None:
            options.append((walk + self.overheads.fixed, TravelMode.WALK, walk))
        if shuttle is not None:
            total = shuttle + self.overheads.shuttle_wait + self.overheads.fixed
            options.append((total, TravelMode.SHUTTLE, shuttle))

        if not options:
            transit = self.unknown_different_building
            return Leg(
                minutes=transit + self.overheads.fixed,
                mode=TravelMode.WALK,
                confidence=Confidence.UNKNOWN,
                origin=origin,
                destination=destination,
                transit_minutes=transit,
                overhead_minutes=self.overheads.fixed,
                detail=f"no route data for {left.label} -> {right.label}",
            )

        # Pick whichever actually gets you there sooner. The shuttle wait is why
        # walking often wins between adjacent properties.
        total_minutes, mode, transit = min(options, key=lambda item: item[0])
        overhead = total_minutes - transit

        detail = f"{left.label} -> {right.label}"
        if mode is TravelMode.SHUTTLE:
            detail += f" ({transit} min ride + {self.overheads.shuttle_wait} min wait)"
        elif len(options) > 1:
            detail += " (faster than waiting for the shuttle)"

        return Leg(
            minutes=total_minutes,
            mode=mode,
            confidence=confidence,
            origin=origin,
            destination=destination,
            transit_minutes=transit,
            overhead_minutes=overhead,
            detail=detail,
        )

    def _times_for(self, left: KnownVenue, right: KnownVenue) -> tuple[dict[str, int], Confidence]:
        curated = self.pairs.get(self._pair_key(left, right))
        if curated:
            return curated, Confidence.CURATED

        left_zone = self.venues[left].zone if left in self.venues else "central"
        right_zone = self.venues[right].zone if right in self.venues else "central"
        zone_key = tuple(sorted((left_zone, right_zone)))
        fallback = self.zone_defaults.get((zone_key[0], zone_key[1]))
        if fallback:
            return fallback, Confidence.ZONE_DEFAULT
        return {}, Confidence.UNKNOWN

    def minutes(self, origin: Room, destination: Room) -> int:
        return self.leg(origin, destination).minutes

    # --- scheduling questions --------------------------------------------- #

    def feasibility(
        self,
        first: Session,
        second: Session,
        *,
        extra_buffer: int = 0,
    ) -> Feasibility:
        """Can ``second`` follow ``first``?

        Unscheduled sessions are reported as not-feasible with a clear reason
        rather than silently treated as compatible, because the catalog really does
        contain all-day and undated entries.
        """
        if first.span is None or second.span is None:
            return Feasibility(
                leg=self.leg(first.room, second.room),
                gap_minutes=0,
                slack_minutes=0,
                ok=False,
                reason="one of these sessions has no published time",
            )

        # Order them so the caller cannot get it wrong by passing them backwards.
        earlier, later = (
            (first, second) if first.span.start <= second.span.start else (second, first)
        )
        assert earlier.span is not None and later.span is not None
        gap = earlier.span.gap_before(later.span)
        leg = self.leg(earlier.room, later.room)

        if gap is None:
            # Times are rendered in each session's own zone. Formatting them in a
            # fixed zone would show a Dubai Summit session in Las Vegas time.
            return Feasibility(
                leg=leg,
                gap_minutes=0,
                slack_minutes=-leg.minutes,
                ok=False,
                reason=(
                    f"{earlier.code} and {later.code} overlap in time "
                    f"({earlier.span.format_in(earlier.zone)} vs "
                    f"{later.span.format_in(later.zone)})"
                ),
            )

        needed = leg.minutes + extra_buffer
        slack = gap - needed
        if slack < 0:
            return Feasibility(
                leg=leg,
                gap_minutes=gap,
                slack_minutes=slack,
                ok=False,
                reason=(
                    f"only {gap} min between {earlier.code} ending and {later.code} "
                    f"starting, but the transfer needs {needed} min "
                    f"({leg.mode.label}: {leg.detail})"
                ),
            )

        descriptor = "tight but makeable" if slack <= 5 else "comfortable"
        return Feasibility(
            leg=leg,
            gap_minutes=gap,
            slack_minutes=slack,
            ok=True,
            reason=(f"{descriptor}: {gap} min gap, {needed} min transfer, {slack} min spare"),
        )

    def depart_by(self, first: Session, second: Session) -> datetime | None:
        """The latest moment you can leave ``first`` and still make ``second``.

        This is the number an attendee actually wants: not "it is a 24 minute
        walk", but "stand up at 14:06".
        """
        if first.span is None or second.span is None:
            return None
        leg = self.leg(first.room, second.room)
        return second.span.start - timedelta(minutes=leg.minutes)

    def entry_overhead_for(self, session: Session) -> int:
        """Extra minutes to allow for getting into a specific session.

        Reserved-seat formats scan you in, and that queue is real.
        """
        if session.needs_reservation or session.format.is_hands_on:
            return self.overheads.reserved_seat_entry_bonus
        return 0

    # --- introspection ---------------------------------------------------- #

    def known_venue_count(self) -> int:
        return len(self.venues)

    def describe_source(self) -> str:
        return (
            f"{self.source}: {len(self.venues)} venues, {len(self.pairs)} curated "
            f"routes, {len(self.zone_defaults)} zone fallbacks"
        )

    def matrix(self) -> tuple[tuple[str, str, int, str], ...]:
        """Every curated route as ``(from, to, minutes, mode)``, for display."""
        rows: list[tuple[str, str, int, str]] = []
        for left, right in sorted(self.pairs, key=lambda item: (item[0].value, item[1].value)):
            origin = Room(venue=left.label, name="Unspecified")
            destination = Room(venue=right.label, name="Unspecified")
            leg = self.leg(origin, destination)
            rows.append((left.label, right.label, leg.minutes, leg.mode.value))
        return tuple(rows)


@functools.lru_cache(maxsize=4)
def default_graph(overrides: Path | None = None) -> TravelGraph:
    """The shared graph instance. Cached because loading parses YAML."""
    return TravelGraph.load(overrides=overrides)


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
