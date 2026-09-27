"""Domain models for crew mode: multi-attendee coordinated planning.

A crew is a team of colleagues attending re:Invent together. Instead of every
member individually browsing the catalog and ending up in the exact same room,
crew mode solves for a team schedule that maximizes topic coverage while
penalizing redundant attendance on regular breakout sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from replan.domain import now_utc
from replan.profiles import Profile
from replan.solver import Plan, PlanItem


@dataclass(frozen=True, slots=True)
class CrewMember:
    """An attendee in a crew."""

    name: str
    profile: Profile
    role: str | None = None
    profile_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TopicCoverage:
    """Coverage status of an individual topic across the crew."""

    topic: str
    weight: float
    covered_by: tuple[str, ...] = ()
    session_codes: tuple[str, ...] = ()

    @property
    def is_covered(self) -> bool:
        return len(self.covered_by) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "weight": self.weight,
            "covered": self.is_covered,
            "coveredBy": list(self.covered_by),
            "sessionCodes": list(self.session_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicCoverage:
        return cls(
            topic=str(data["topic"]),
            weight=float(data.get("weight", 1.0)),
            covered_by=tuple(data.get("coveredBy") or ()),
            session_codes=tuple(data.get("sessionCodes") or ()),
        )


@dataclass(frozen=True, slots=True)
class CrewCoverageReport:
    """How well the crew's combined schedule covers their collective goals."""

    total_topics: int
    covered_topics: int
    coverage_percentage: float
    topics: tuple[TopicCoverage, ...] = ()
    member_session_counts: dict[str, int] = field(default_factory=dict)
    redundancy_count: int = 0
    shared_sessions: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "totalTopics": self.total_topics,
            "coveredTopics": self.covered_topics,
            "coveragePercentage": round(self.coverage_percentage, 1),
            "topics": [topic.to_dict() for topic in self.topics],
            "memberSessionCounts": self.member_session_counts,
            "redundancyCount": self.redundancy_count,
            "sharedSessions": [
                {"sessionCode": code, "members": list(members)}
                for code, members in self.shared_sessions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrewCoverageReport:
        return cls(
            total_topics=int(data.get("totalTopics", 0)),
            covered_topics=int(data.get("coveredTopics", 0)),
            coverage_percentage=float(data.get("coveragePercentage", 0.0)),
            topics=tuple(TopicCoverage.from_dict(t) for t in data.get("topics", ())),
            member_session_counts=dict(data.get("memberSessionCounts") or {}),
            redundancy_count=int(data.get("redundancyCount", 0)),
            shared_sessions=tuple(
                (str(entry["sessionCode"]), tuple(entry.get("members", ())))
                for entry in data.get("sharedSessions", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class CrewPlan:
    """The collective plan for a team of attendees."""

    event_id: str
    members: dict[str, Plan]
    coverage: CrewCoverageReport
    created_at: datetime = field(default_factory=now_utc)
    format_version: int = 1

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.members.keys()))

    @property
    def total_sessions(self) -> int:
        return sum(len(p.items) for p in self.members.values())

    @property
    def unique_session_count(self) -> int:
        unique: set[str] = set()
        for p in self.members.values():
            for item in p.items:
                unique.add(item.session_id)
        return len(unique)

    def all_items(self) -> list[tuple[str, PlanItem]]:
        """All planned sessions across members, tagged with the member name."""
        entries: list[tuple[str, PlanItem]] = []
        for name, plan in self.members.items():
            for item in plan.items:
                entries.append((name, item))
        return sorted(entries, key=lambda pair: pair[1].span.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formatVersion": self.format_version,
            "eventId": self.event_id,
            "createdAt": self.created_at.isoformat(),
            "totalSessions": self.total_sessions,
            "uniqueSessions": self.unique_session_count,
            "coverage": self.coverage.to_dict(),
            "members": {name: plan.to_dict() for name, plan in self.members.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrewPlan:
        return cls(
            event_id=str(data["eventId"]),
            members={
                name: Plan.from_dict(plan_data)
                for name, plan_data in data.get("members", {}).items()
            },
            coverage=CrewCoverageReport.from_dict(data.get("coverage", {})),
            created_at=datetime.fromisoformat(data["createdAt"]),
            format_version=int(data.get("formatVersion", 1)),
        )


__all__ = [
    "CrewCoverageReport",
    "CrewMember",
    "CrewPlan",
    "TopicCoverage",
]
