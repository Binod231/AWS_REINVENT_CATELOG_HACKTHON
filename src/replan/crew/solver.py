"""Coordinated multi-attendee scheduling for crew mode.

Single-attendee scheduling optimizes for one person's wants.
Crew scheduling optimizes for an entire team attending re:Invent together:
1. Maximize individual utility and level alignment for each member.
2. Maximize the *union* of topics covered by the organization.
3. Minimize redundant attendance: two colleagues should not sit in the same
   200-level breakout when one could attend an architecture talk and the
   other an AI/ML deep dive (keynotes and pinned sessions are exempt).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from replan.crew.models import CrewCoverageReport, CrewMember, CrewPlan, TopicCoverage
from replan.domain import Catalog, SessionFormat
from replan.profiles import Scorer
from replan.solver import Plan, SolveRequest, solve
from replan.travel import TravelGraph


@dataclass(frozen=True, slots=True)
class CrewSolveRequest:
    """The inputs needed to coordinate a crew solve."""

    catalog: Catalog
    members: tuple[CrewMember, ...]
    travel: TravelGraph
    redundancy_penalty: float = 20.0
    days: frozenset[date] | None = None
    time_limit_seconds: float = 15.0
    prefer_cpsat: bool = True


def compute_crew_coverage(
    members: tuple[CrewMember, ...] | list[CrewMember],
    plans: dict[str, Plan],
    catalog: Catalog,
) -> CrewCoverageReport:
    """Analyze which topics across all member profiles were covered by the crew."""
    # Union of all weighted topics across member profiles
    all_topics: dict[str, float] = {}
    for m in members:
        for topic, weight in m.profile.topics.items():
            all_topics[topic] = max(all_topics.get(topic, 0.0), float(weight))

    session_owners: dict[str, list[str]] = {}
    session_codes_by_member: dict[str, set[str]] = {}

    for name, plan in plans.items():
        session_codes_by_member[name] = set()
        for item in plan.items:
            session_owners.setdefault(item.code, []).append(name)
            session_codes_by_member[name].add(item.code)

    topic_coverages: list[TopicCoverage] = []
    covered_topic_count = 0

    # Build scorers per member to use their topic matching
    scorers = {m.name: Scorer(m.profile, catalog) for m in members}

    for topic, weight in sorted(all_topics.items(), key=lambda pair: (-pair[1], pair[0])):
        covering_members: list[str] = []
        matching_sessions: list[str] = []

        for m in members:
            member_plan = plans.get(m.name)
            if member_plan is None:
                continue

            for item in member_plan.items:
                session = catalog.by_id(item.session_id)
                topic_lower = topic.lower()
                matches = False
                if session is not None:
                    matched_topics = [t.lower() for t in scorers[m.name].topics_matched(session)]
                    if topic_lower in matched_topics:
                        matches = True
                if not matches:
                    item_title = item.title.lower()
                    if (
                        topic_lower in item_title
                        or topic_lower in item.code.lower()
                        or any(topic_lower in r.lower() for r in item.top_reasons)
                    ):
                        matches = True

                if matches:
                    if m.name not in covering_members:
                        covering_members.append(m.name)
                    if item.code not in matching_sessions:
                        matching_sessions.append(item.code)

        is_covered = len(covering_members) > 0
        if is_covered:
            covered_topic_count += 1

        topic_coverages.append(
            TopicCoverage(
                topic=topic,
                weight=weight,
                covered_by=tuple(sorted(covering_members)),
                session_codes=tuple(sorted(matching_sessions)),
            )
        )

    shared: list[tuple[str, tuple[str, ...]]] = []
    redundancy_count = 0

    for code, owners in sorted(session_owners.items()):
        if len(owners) > 1:
            shared.append((code, tuple(sorted(owners))))
            is_keynote = False
            for p in plans.values():
                it = p.item_for(code)
                if it is not None and "keynote" in it.title.lower():
                    is_keynote = True
                    break
            if not is_keynote:
                redundancy_count += len(owners) - 1

    total_topics = len(all_topics)
    pct = (covered_topic_count / total_topics * 100.0) if total_topics > 0 else 100.0

    return CrewCoverageReport(
        total_topics=total_topics,
        covered_topics=covered_topic_count,
        coverage_percentage=pct,
        topics=tuple(topic_coverages),
        member_session_counts={name: len(p.items) for name, p in plans.items()},
        redundancy_count=redundancy_count,
        shared_sessions=tuple(shared),
    )


def solve_crew(request: CrewSolveRequest) -> CrewPlan:
    """Solve for a coordinated crew schedule."""
    if not request.members:
        msg = "At least one crew member profile is required."
        raise ValueError(msg)

    # Single-member optimization directly runs the single-attendee solver
    if len(request.members) == 1:
        member = request.members[0]
        scorer = Scorer(member.profile, request.catalog)
        single_req = SolveRequest(
            catalog=request.catalog,
            profile=member.profile,
            scores=scorer.by_id(),
            travel=request.travel,
            days=request.days,
            time_limit_seconds=request.time_limit_seconds,
        )
        plan = solve(single_req, prefer_cpsat=request.prefer_cpsat)
        single_plans: dict[str, Plan] = {member.name: plan}
        coverage = compute_crew_coverage(request.members, single_plans, request.catalog)
        return CrewPlan(
            event_id=request.catalog.event.event_id if request.catalog.event else "unknown",
            members=single_plans,
            coverage=coverage,
        )

    # Multi-member coordinated solve:
    plans: dict[str, Plan] = {}
    assigned_sessions: dict[str, list[str]] = {}  # session_id -> list of member names

    ordered_members = sorted(request.members, key=lambda m: (len(m.profile.must_attend), m.name))

    for member in ordered_members:
        scorer = Scorer(member.profile, request.catalog)
        base_scores = scorer.by_id()

        # Apply redundancy penalties to already-assigned sessions
        adjusted_scores = dict(base_scores)
        for s_id, score in base_scores.items():
            if not score.eligible:
                continue
            session = request.catalog.by_id(s_id)
            if session is None:
                continue

            is_keynote = (
                session.format is SessionFormat.KEYNOTE or "keynote" in session.title.lower()
            )
            is_pinned = (
                s_id in member.profile.must_attend or session.code in member.profile.must_attend
            )

            if not is_keynote and not is_pinned and s_id in assigned_sessions:
                num_existing = len(assigned_sessions[s_id])
                penalty = request.redundancy_penalty * num_existing
                new_total = max(0.1, score.total - penalty)
                # Replace total on Score dataclass
                adjusted_scores[s_id] = replace(score, total=new_total)

        member_req = SolveRequest(
            catalog=request.catalog,
            profile=member.profile,
            scores=adjusted_scores,
            travel=request.travel,
            days=request.days,
            time_limit_seconds=request.time_limit_seconds,
        )
        plan = solve(member_req, prefer_cpsat=request.prefer_cpsat)
        plans[member.name] = plan

        for item in plan.items:
            assigned_sessions.setdefault(item.session_id, []).append(member.name)

    coverage = compute_crew_coverage(request.members, plans, request.catalog)
    event_id = request.catalog.event.event_id if request.catalog.event else "unknown"

    return CrewPlan(
        event_id=event_id,
        members=plans,
        coverage=coverage,
    )


__all__ = [
    "CrewSolveRequest",
    "compute_crew_coverage",
    "solve_crew",
]
