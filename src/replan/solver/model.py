"""The shapes a solve produces.

A :class:`Plan` is deliberately more than a list of sessions. It carries the
travel leg into each item, the objective broken into named parts, and a
:class:`Rejection` for every session that did not make it, with the reason. That
last part is the difference between a tool you trust and one you argue with: when
a plan omits the session you were looking forward to, you get told which
constraint removed it.

Plans serialise to JSON so they can be handed between commands - ``optimize``
writes one, ``why``, ``export`` and ``apply`` read it - and so a plan can be
diffed against a later one after the catalog shifts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from replan.domain import (
    REINVENT_TZ,
    AttendeeSchedule,
    Catalog,
    Session,
    TimeSpan,
    day_label,
    now_utc,
    resolve_timezone,
)
from replan.profiles import Profile, Score
from replan.travel import TravelGraph

PLAN_FORMAT_VERSION = 1


class SolverKind(StrEnum):
    CPSAT = "cp-sat"
    GREEDY = "greedy"

    @property
    def label(self) -> str:
        return {
            SolverKind.CPSAT: "CP-SAT constraint solver",
            SolverKind.GREEDY: "greedy weighted-interval fallback",
        }[self]

    @property
    def is_optimal_capable(self) -> bool:
        return self is SolverKind.CPSAT


class RejectionReason(StrEnum):
    """Why a session is not in the plan.

    Ordered from "your own settings did this" to "the solver made a judgement
    call", because that is the order a user wants to hear them in.
    """

    VETOED = "vetoed"
    """Excluded by the profile before scoring: banned, unscheduled, out of window."""

    PROTECTED_TIME = "protected-time"
    """Collides with a meal window or personal-time block you asked to keep."""

    BELOW_CUTOFF = "below-cutoff"
    """Scored too low to enter the candidate pool."""

    TIME_CONFLICT = "time-conflict"
    """Overlaps something that was selected."""

    TRAVEL_INFEASIBLE = "travel-infeasible"
    """No overlap, but not enough minutes to cross between the rooms."""

    WALK_TOO_FAR = "walk-too-far"
    """Reachable in time, but beyond your walk tolerance."""

    DUPLICATE_CONTENT = "duplicate-content"
    """Another showing of the same content is already in the plan."""

    DAY_FULL = "day-full"
    """Would exceed max_sessions_per_day."""

    NOT_CHOSEN = "not-chosen"
    """Feasible, but the solver found a better overall combination."""

    @property
    def label(self) -> str:
        return {
            RejectionReason.VETOED: "excluded by your profile",
            RejectionReason.PROTECTED_TIME: "clashes with time you protected",
            RejectionReason.BELOW_CUTOFF: "scored below the candidate cutoff",
            RejectionReason.TIME_CONFLICT: "overlaps a session in the plan",
            RejectionReason.TRAVEL_INFEASIBLE: "not enough time to get there",
            RejectionReason.WALK_TOO_FAR: "further than your walk tolerance",
            RejectionReason.DUPLICATE_CONTENT: "you already have this content",
            RejectionReason.DAY_FULL: "that day is already full",
            RejectionReason.NOT_CHOSEN: "a better combination won",
        }[self]

    @property
    def is_your_choice(self) -> bool:
        """Whether the user's own settings caused this, so it is fixable by editing."""
        return self in {
            RejectionReason.VETOED,
            RejectionReason.PROTECTED_TIME,
            RejectionReason.BELOW_CUTOFF,
            RejectionReason.WALK_TOO_FAR,
            RejectionReason.DAY_FULL,
        }


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One session in the plan, with how you get to it."""

    session_id: str
    code: str
    title: str
    span: TimeSpan
    room_display: str
    score: float
    pinned: bool = False
    already_reserved: bool = False
    needs_reservation: bool = False
    seat_band: str | None = None
    timezone_name: str | None = None

    # Filled for every item after the first of its day.
    travel_minutes: int | None = None
    travel_mode: str | None = None
    travel_detail: str | None = None
    travel_confidence: str | None = None
    depart_previous_by: datetime | None = None
    idle_minutes: int | None = None
    previous_session_id: str | None = None

    top_reasons: tuple[str, ...] = ()

    @property
    def zone(self) -> ZoneInfo:
        return resolve_timezone(self.timezone_name)

    @property
    def day(self) -> date:
        return self.span.day_in(self.zone)

    @property
    def when(self) -> str:
        return self.span.format_in(self.zone)

    @property
    def is_tight_connection(self) -> bool:
        return self.idle_minutes is not None and self.idle_minutes <= 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "code": self.code,
            "title": self.title,
            "start": self.span.start.isoformat(),
            "end": self.span.end.isoformat(),
            "room": self.room_display,
            "score": self.score,
            "pinned": self.pinned,
            "alreadyReserved": self.already_reserved,
            "needsReservation": self.needs_reservation,
            "seatBand": self.seat_band,
            "timezone": self.timezone_name,
            "travelMinutes": self.travel_minutes,
            "travelMode": self.travel_mode,
            "travelDetail": self.travel_detail,
            "travelConfidence": self.travel_confidence,
            "departPreviousBy": (
                self.depart_previous_by.isoformat() if self.depart_previous_by else None
            ),
            "idleMinutes": self.idle_minutes,
            "previousSessionId": self.previous_session_id,
            "topReasons": list(self.top_reasons),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanItem:
        return cls(
            session_id=str(payload["sessionId"]),
            code=str(payload.get("code") or payload["sessionId"]),
            title=str(payload.get("title") or ""),
            span=TimeSpan(
                datetime.fromisoformat(payload["start"]),
                datetime.fromisoformat(payload["end"]),
            ),
            room_display=str(payload.get("room") or ""),
            score=float(payload.get("score") or 0.0),
            pinned=bool(payload.get("pinned")),
            already_reserved=bool(payload.get("alreadyReserved")),
            needs_reservation=bool(payload.get("needsReservation")),
            seat_band=payload.get("seatBand"),
            timezone_name=payload.get("timezone"),
            travel_minutes=payload.get("travelMinutes"),
            travel_mode=payload.get("travelMode"),
            travel_detail=payload.get("travelDetail"),
            travel_confidence=payload.get("travelConfidence"),
            depart_previous_by=(
                datetime.fromisoformat(payload["departPreviousBy"])
                if payload.get("departPreviousBy")
                else None
            ),
            idle_minutes=payload.get("idleMinutes"),
            previous_session_id=payload.get("previousSessionId"),
            top_reasons=tuple(payload.get("topReasons") or ()),
        )


@dataclass(frozen=True, slots=True)
class Rejection:
    """A session that is not in the plan, and why."""

    session_id: str
    code: str
    title: str
    score: float
    reason: RejectionReason
    detail: str
    conflicts_with: tuple[str, ...] = ()

    def describe(self) -> str:
        return f"{self.code}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "code": self.code,
            "title": self.title,
            "score": self.score,
            "reason": self.reason.value,
            "detail": self.detail,
            "conflictsWith": list(self.conflicts_with),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Rejection:
        return cls(
            session_id=str(payload["sessionId"]),
            code=str(payload.get("code") or payload["sessionId"]),
            title=str(payload.get("title") or ""),
            score=float(payload.get("score") or 0.0),
            reason=RejectionReason(payload.get("reason") or RejectionReason.NOT_CHOSEN.value),
            detail=str(payload.get("detail") or ""),
            conflicts_with=tuple(payload.get("conflictsWith") or ()),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    """What the objective was made of, in human units rather than solver units."""

    utility: float = 0.0
    coverage: float = 0.0
    travel_cost: float = 0.0
    fatigue_cost: float = 0.0
    hop_cost: float = 0.0
    idle_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.utility
            + self.coverage
            - self.travel_cost
            - self.fatigue_cost
            - self.hop_cost
            - self.idle_cost
        )

    def rows(self) -> tuple[tuple[str, float], ...]:
        return (
            ("session value", self.utility),
            ("topic coverage", self.coverage),
            ("travel cost", -self.travel_cost),
            ("fatigue cost", -self.fatigue_cost),
            ("venue hop cost", -self.hop_cost),
            ("idle time cost", -self.idle_cost),
            ("total", self.total),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "utility": self.utility,
            "coverage": self.coverage,
            "travelCost": self.travel_cost,
            "fatigueCost": self.fatigue_cost,
            "hopCost": self.hop_cost,
            "idleCost": self.idle_cost,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObjectiveBreakdown:
        return cls(
            utility=float(payload.get("utility") or 0.0),
            coverage=float(payload.get("coverage") or 0.0),
            travel_cost=float(payload.get("travelCost") or 0.0),
            fatigue_cost=float(payload.get("fatigueCost") or 0.0),
            hop_cost=float(payload.get("hopCost") or 0.0),
            idle_cost=float(payload.get("idleCost") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class SolveStats:
    """How the solve went. Reported so a fallback is never mistaken for optimal."""

    solver: SolverKind
    status: str
    optimal: bool
    wall_seconds: float
    candidates: int
    considered: int
    selected: int
    topics_covered: int
    topics_available: int
    fallback_reason: str | None = None

    @property
    def coverage_ratio(self) -> float:
        if not self.topics_available:
            return 0.0
        return self.topics_covered / self.topics_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver.value,
            "status": self.status,
            "optimal": self.optimal,
            "wallSeconds": round(self.wall_seconds, 3),
            "candidates": self.candidates,
            "considered": self.considered,
            "selected": self.selected,
            "topicsCovered": self.topics_covered,
            "topicsAvailable": self.topics_available,
            "fallbackReason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SolveStats:
        return cls(
            solver=SolverKind(payload.get("solver") or SolverKind.GREEDY.value),
            status=str(payload.get("status") or "unknown"),
            optimal=bool(payload.get("optimal")),
            wall_seconds=float(payload.get("wallSeconds") or 0.0),
            candidates=int(payload.get("candidates") or 0),
            considered=int(payload.get("considered") or 0),
            selected=int(payload.get("selected") or 0),
            topics_covered=int(payload.get("topicsCovered") or 0),
            topics_available=int(payload.get("topicsAvailable") or 0),
            fallback_reason=payload.get("fallbackReason"),
        )


@dataclass(frozen=True, slots=True)
class Plan:
    """A solved agenda."""

    event_id: str
    profile_name: str
    items: tuple[PlanItem, ...]
    rejections: tuple[Rejection, ...]
    stats: SolveStats
    objective: ObjectiveBreakdown
    catalog_source: str = "unknown"
    catalog_fetched_at: datetime | None = None
    generated_at: datetime = field(default_factory=now_utc)
    timezone_name: str | None = None
    goals: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def zone(self) -> ZoneInfo:
        return resolve_timezone(self.timezone_name)

    @property
    def days(self) -> tuple[date, ...]:
        return tuple(sorted({item.day for item in self.items}))

    def by_day(self) -> tuple[tuple[date, tuple[PlanItem, ...]], ...]:
        grouped: dict[date, list[PlanItem]] = {}
        for item in self.items:
            grouped.setdefault(item.day, []).append(item)
        return tuple(
            (current, tuple(sorted(entries, key=lambda entry: entry.span.start)))
            for current, entries in sorted(grouped.items())
        )

    @property
    def total_score(self) -> float:
        return round(sum(item.score for item in self.items), 2)

    @property
    def total_minutes(self) -> int:
        return sum(item.span.minutes for item in self.items)

    @property
    def total_travel_minutes(self) -> int:
        return sum(item.travel_minutes or 0 for item in self.items)

    @property
    def needs_reservation(self) -> tuple[PlanItem, ...]:
        return tuple(
            item for item in self.items if item.needs_reservation and not item.already_reserved
        )

    @property
    def tight_connections(self) -> tuple[PlanItem, ...]:
        return tuple(item for item in self.items if item.is_tight_connection)

    def item_for(self, token: str) -> PlanItem | None:
        wanted = token.strip().upper()
        for item in self.items:
            if item.session_id == token or item.code.upper() == wanted:
                return item
        return None

    def rejection_for(self, token: str) -> Rejection | None:
        wanted = token.strip().upper()
        for entry in self.rejections:
            if entry.session_id == token or entry.code.upper() == wanted:
                return entry
        return None

    def worst_rejections(self, limit: int = 5) -> tuple[Rejection, ...]:
        """The highest-scoring sessions that did not make it.

        The most interesting question a plan can answer: what did I nearly get?
        """
        ranked = sorted(self.rejections, key=lambda entry: -entry.score)
        return tuple(ranked[:limit])

    def day_label_for(self, value: date) -> str:
        return day_label(value)

    # --- serialisation ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "formatVersion": PLAN_FORMAT_VERSION,
            "eventId": self.event_id,
            "profileName": self.profile_name,
            "generatedAt": self.generated_at.isoformat(),
            "timezone": self.timezone_name,
            "catalogSource": self.catalog_source,
            "catalogFetchedAt": (
                self.catalog_fetched_at.isoformat() if self.catalog_fetched_at else None
            ),
            "goals": list(self.goals),
            "warnings": list(self.warnings),
            "stats": self.stats.to_dict(),
            "objective": self.objective.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "rejections": [entry.to_dict() for entry in self.rejections],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Plan:
        version = int(payload.get("formatVersion") or 0)
        if version > PLAN_FORMAT_VERSION:
            msg = (
                f"this plan was written by a newer replan (format {version}); "
                f"this build understands up to {PLAN_FORMAT_VERSION}"
            )
            raise ValueError(msg)
        return cls(
            event_id=str(payload.get("eventId") or "unknown"),
            profile_name=str(payload.get("profileName") or "me"),
            items=tuple(PlanItem.from_dict(item) for item in payload.get("items") or ()),
            rejections=tuple(
                Rejection.from_dict(entry) for entry in payload.get("rejections") or ()
            ),
            stats=SolveStats.from_dict(payload.get("stats") or {}),
            objective=ObjectiveBreakdown.from_dict(payload.get("objective") or {}),
            catalog_source=str(payload.get("catalogSource") or "unknown"),
            catalog_fetched_at=(
                datetime.fromisoformat(payload["catalogFetchedAt"])
                if payload.get("catalogFetchedAt")
                else None
            ),
            generated_at=(
                datetime.fromisoformat(payload["generatedAt"])
                if payload.get("generatedAt")
                else now_utc()
            ),
            timezone_name=payload.get("timezone"),
            goals=tuple(payload.get("goals") or ()),
            warnings=tuple(payload.get("warnings") or ()),
        )

    @classmethod
    def read(cls, path: Path) -> Plan:
        if not path.exists():
            msg = f"plan not found: {path}. Create one with `replan optimize`."
            raise FileNotFoundError(msg)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """Everything a solver needs. Assembled once, passed to either implementation."""

    catalog: Catalog
    profile: Profile
    scores: dict[str, Score]
    travel: TravelGraph
    schedule: AttendeeSchedule | None = None
    days: frozenset[date] | None = None
    candidates_per_day: int = 45
    """Cap on how many sessions per day enter the model.

    The successor-arc formulation is quadratic in candidates per day, so an
    unbounded pool on a 2,000-session catalog would build millions of variables.
    Taking the top N by score keeps the model in the thousands, which solves in
    well under a second. Pins and existing reservations bypass the cap entirely,
    so nothing you explicitly asked for can be pruned.
    """

    time_limit_seconds: float = 10.0
    random_seed: int = 20261130
    respect_reservations: bool = True

    @property
    def zone(self) -> ZoneInfo:
        return self.catalog.zone if self.catalog.event else REINVENT_TZ

    @property
    def reserved_ids(self) -> frozenset[str]:
        if not self.respect_reservations or self.schedule is None:
            return frozenset()
        return self.schedule.reserved_ids

    def protected_spans(self) -> tuple[tuple[str, TimeSpan], ...]:
        """Meal windows plus existing personal-time blocks, as concrete spans.

        Both are treated identically by the solver: time the attendee already
        claimed. Personal time comes from ``GetSchedule``, so a block created in
        the official app is respected too.
        """
        spans: list[tuple[str, TimeSpan]] = []

        if self.schedule is not None:
            for block in self.schedule.personal_time:
                spans.append((block.title, block.span))

        catalog_days = sorted(self.days) if self.days else list(self.catalog.days)
        zone = self.zone
        for window in self.profile.meal_windows:
            if not window.protect:
                continue
            wanted_days = catalog_days
            if window.days:
                resolved: list[date] = []
                for token in window.days:
                    try:
                        from replan.domain import resolve_day

                        resolved.append(resolve_day(token, catalog_days))
                    except ValueError:
                        continue
                wanted_days = resolved
            for current in wanted_days:
                start = datetime.combine(current, window.start, tzinfo=zone)
                end = datetime.combine(current, window.end, tzinfo=zone)
                if end > start:
                    spans.append((window.title, TimeSpan(start, end)))
        return tuple(spans)

    def eligible(self) -> tuple[Session, ...]:
        """Sessions the scorer did not veto, restricted to the requested days."""
        result: list[Session] = []
        for session in self.catalog.sessions:
            score = self.scores.get(session.session_id)
            if score is None or not score.eligible:
                continue
            if session.span is None:
                continue
            if self.days is not None and session.day not in self.days:
                continue
            result.append(session)
        return tuple(result)


__all__ = [
    "PLAN_FORMAT_VERSION",
    "ObjectiveBreakdown",
    "Plan",
    "PlanItem",
    "Rejection",
    "RejectionReason",
    "SolveRequest",
    "SolveStats",
    "SolverKind",
]
