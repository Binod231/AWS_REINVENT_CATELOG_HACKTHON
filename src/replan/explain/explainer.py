"""Plain-language answers to "why is this session in my plan, or not?"

The design goal is that every answer is *actionable*. Knowing a session was
dropped is mildly interesting; knowing it was dropped because a 51-minute transfer
exceeds your 45-minute walk tolerance, and that raising one line in your profile
would let it in, is useful. So each explanation carries a ``remedy``: the specific
change that would alter the outcome, or an honest statement that nothing you can
configure would.

Where the API's own wording exists, it is quoted rather than paraphrased. If AWS
says a session is ``"300 – Advanced"``, the explanation says that, not "level 3".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from replan.domain import Catalog, Session
from replan.profiles import Profile, Score, Scorer
from replan.solver import Plan, PlanItem, Rejection, RejectionReason
from replan.travel import TravelGraph


class Verdict(StrEnum):
    SCHEDULED = "scheduled"
    REJECTED = "rejected"
    UNKNOWN_SESSION = "unknown-session"

    @property
    def label(self) -> str:
        return {
            Verdict.SCHEDULED: "in your plan",
            Verdict.REJECTED: "not in your plan",
            Verdict.UNKNOWN_SESSION: "not found",
        }[self]


@dataclass(frozen=True, slots=True)
class Explanation:
    """Everything worth saying about one session's fate."""

    session_id: str
    code: str
    title: str
    verdict: Verdict
    headline: str
    facts: tuple[tuple[str, str], ...] = ()
    """Label/value pairs: when, where, level, seats, travel."""

    score_lines: tuple[str, ...] = ()
    """The scoring factors, strongest first."""

    remedy: str | None = None
    """The change that would flip this decision, when one exists."""

    alternatives: tuple[str, ...] = ()
    """Other showings of the same content, and what happened to each."""

    conflicts_with: tuple[str, ...] = ()

    def as_text(self) -> str:
        lines = [f"{self.code} - {self.headline}"]
        lines.extend(f"  {label}: {value}" for label, value in self.facts)
        if self.score_lines:
            lines.append("  scoring:")
            lines.extend(f"    {item}" for item in self.score_lines)
        if self.alternatives:
            lines.append("  other showings:")
            lines.extend(f"    {item}" for item in self.alternatives)
        if self.remedy:
            lines.append(f"  to change this: {self.remedy}")
        return "\n".join(lines)


def _remedy_for(rejection: Rejection, profile: Profile, session: Session | None) -> str:
    """The specific profile change that would let this session in.

    Returns ``str``, not ``str | None``, deliberately: the branches below cover
    every :class:`RejectionReason`, so adding a new member makes this function fall
    off the end and mypy reports a missing return. That turns "we forgot to explain
    the new rejection reason" from a silent gap into a build failure.
    """
    reason = rejection.reason

    if reason is RejectionReason.VETOED:
        detail = rejection.detail.lower()
        if "never_attend" in detail:
            return f"remove {rejection.code} from never_attend"
        if "no published time" in detail or "all-day" in detail:
            return (
                "nothing - the catalog publishes no start time for this entry, so it "
                "cannot be placed on a timeline"
            )
        if "day window" in detail:
            return f"widen day_start/day_end (currently {profile.day_start}-{profile.day_end})"
        if "max_consecutive_minutes" in detail:
            minutes = session.duration_minutes if session else None
            target = f" to at least {minutes}" if minutes else ""
            return f"raise max_consecutive_minutes{target}"
        if "walk-up" in detail:
            return "set include_walk_up: true"
        if "no reservation" in detail:
            return "set include_unreservable: true"
        if "levels" in detail:
            return "add this level to your levels map, or remove the levels filter"
        return "adjust the profile setting named in the reason above"

    if reason is RejectionReason.PROTECTED_TIME:
        return (
            "drop or move the meal window it collides with, set protect: false on it, "
            f"or add {rejection.code} to must_attend to override"
        )

    if reason is RejectionReason.BELOW_CUTOFF:
        return "raise --candidates, weight its topics higher, or add it to must_attend"

    if reason is RejectionReason.WALK_TOO_FAR:
        return f"raise walk_tolerance_minutes above its current {profile.walk_tolerance_minutes}"

    if reason is RejectionReason.TRAVEL_INFEASIBLE:
        if profile.min_break_minutes:
            return (
                f"lower min_break_minutes (currently {profile.min_break_minutes}), or "
                "drop the session it follows. The clock itself may still not allow it."
            )
        return (
            "nothing in the profile - there genuinely is not enough time to get "
            "between the rooms. Drop the session it follows, or look for a repeat."
        )

    if reason is RejectionReason.TIME_CONFLICT:
        return (
            "the two overlap, so only one is possible. Pin whichever you prefer with "
            "must_attend, or take a repeat showing if one exists."
        )

    if reason is RejectionReason.DUPLICATE_CONTENT:
        return (
            "this is the same content as a session already scheduled. Pin this "
            "showing with must_attend if you want this one instead."
        )

    if reason is RejectionReason.DAY_FULL:
        current = profile.max_sessions_per_day
        return f"raise max_sessions_per_day (currently {current})"

    if reason is RejectionReason.NOT_CHOSEN:
        return (
            "pin it with must_attend, or weight its topics higher. It was feasible; "
            "the solver simply found a better-scoring combination."
        )

    # Every RejectionReason is handled above, so `reason` is narrowed to Never here.
    # Adding a member without a remedy is therefore a type error, not a silent gap.
    assert_never(reason)


def _score_lines(score: Score | None) -> tuple[str, ...]:
    if score is None or not score.factors:
        return ()
    ranked = sorted(score.factors, key=lambda item: -abs(item.points))
    lines = [f"{item.points:+.1f}  {item.name} - {item.detail}" for item in ranked]
    lines.append(f"={score.total:.1f}  total")
    return tuple(lines)


def _session_facts(session: Session, catalog: Catalog) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    if session.span is not None:
        facts.append(("when", session.span.format_in(session.zone)))
    else:
        facts.append(("when", "no published time"))
    facts.append(("where", session.room.display))
    # Quote the API's own strings rather than our normalisation of them.
    if session.level_raw:
        facts.append(("level", session.level_raw))
    if session.type_raw:
        facts.append(("format", session.type_raw))
    if session.seat_availability is not None:
        facts.append(("seats", session.seat_availability.label))
    elif session.is_reservable is None:
        facts.append(("seats", "not published for this event"))
    if session.speakers:
        facts.append(("speakers", ", ".join(item.person for item in session.speakers[:4])))
    if session.services:
        facts.append(("services", ", ".join(session.services[:6])))
    repeats = catalog.repeats_of(session)
    if len(repeats) > 1:
        facts.append(("showings", f"{len(repeats)} of {session.series}"))
    return facts


def _alternatives(session: Session, catalog: Catalog, plan: Plan) -> tuple[str, ...]:
    """Other showings of the same content, and their fate in this plan."""
    siblings = [
        item for item in catalog.repeats_of(session) if item.session_id != session.session_id
    ]
    if not siblings:
        return ()
    lines: list[str] = []
    for sibling in sorted(
        siblings,
        key=lambda item: item.span.start if item.span else None,  # type: ignore[arg-type,return-value]
    ):
        when = sibling.span.format_in(sibling.zone) if sibling.span else "unscheduled"
        if plan.item_for(sibling.session_id) is not None:
            state = "[in your plan]"
        else:
            rejection = plan.rejection_for(sibling.session_id)
            state = f"({rejection.reason.label})" if rejection else "(not considered)"
        lines.append(f"{sibling.code} {when} {state}")
    return tuple(lines)


def _nearest_planned_leg(
    session: Session,
    plan: Plan,
    catalog: Catalog,
    travel: TravelGraph,
) -> str | None:
    """Describe the transfer from the closest scheduled session on the same day."""
    if session.span is None:
        return None
    same_day = [item for item in plan.items if item.day == session.day]
    if not same_day:
        return None

    reference = session.span.start

    def distance(item: PlanItem) -> float:
        return abs((item.span.start - reference).total_seconds())

    closest = min(same_day, key=distance)
    other = catalog.by_id(closest.session_id)
    if other is None:
        return None
    leg = travel.leg(other.room, session.room)
    if leg.minutes == 0:
        return f"{other.code} is in the same room"
    return (
        f"{leg.minutes} min {leg.mode.label} from {other.code} "
        f"({other.span.format_in(other.zone) if other.span else 'unscheduled'})"
    )


def explain_session(
    token: str,
    plan: Plan,
    catalog: Catalog,
    profile: Profile,
    travel: TravelGraph,
    scorer: Scorer | None = None,
) -> Explanation:
    """Explain one session's fate, by id or by code."""
    session = catalog.resolve(token)
    if session is None:
        return Explanation(
            session_id=token,
            code=token,
            title="",
            verdict=Verdict.UNKNOWN_SESSION,
            headline=f"no session in the {catalog.event_id} catalog matches {token!r}",
            remedy="check the code, or run `replan sync` if the catalog is stale",
        )

    resolved_scorer = scorer or Scorer(profile, catalog)
    score = resolved_scorer.score(session)
    facts = _session_facts(session, catalog)
    item = plan.item_for(session.session_id)

    if item is not None:
        if item.travel_minutes is not None and item.previous_session_id:
            previous = catalog.by_id(item.previous_session_id)
            previous_code = previous.code if previous else item.previous_session_id
            if item.travel_minutes == 0:
                facts.append(("getting there", f"same room as {previous_code}"))
            else:
                leave = (
                    f", leave by {item.depart_previous_by.astimezone(item.zone):%H:%M}"
                    if item.depart_previous_by
                    else ""
                )
                facts.append(
                    (
                        "getting there",
                        f"{item.travel_minutes} min {item.travel_mode} from {previous_code}{leave}",
                    )
                )
            if item.idle_minutes is not None:
                facts.append(("spare time", f"{item.idle_minutes} min"))
        else:
            facts.append(("getting there", "first session of the day"))

        if item.already_reserved:
            facts.append(("reservation", "you already hold a seat"))
        elif item.needs_reservation:
            facts.append(("reservation", "needs one - run `replan apply`"))

        headline = "in your plan"
        if item.pinned:
            headline = "in your plan, pinned by must_attend"
        elif item.is_tight_connection:
            headline = "in your plan, but the connection into it is tight"

        return Explanation(
            session_id=session.session_id,
            code=session.code,
            title=session.title,
            verdict=Verdict.SCHEDULED,
            headline=headline,
            facts=tuple(facts),
            score_lines=_score_lines(score),
            alternatives=_alternatives(session, catalog, plan),
        )

    rejection = plan.rejection_for(session.session_id)
    if rejection is None:
        return Explanation(
            session_id=session.session_id,
            code=session.code,
            title=session.title,
            verdict=Verdict.REJECTED,
            headline="not in your plan, and not recorded as considered",
            facts=tuple(facts),
            score_lines=_score_lines(score),
            remedy=(
                "this plan may predate the session appearing in the catalog; "
                "re-run `replan optimize`"
            ),
        )

    blocking: list[str] = []
    for other_id in rejection.conflicts_with:
        other = catalog.by_id(other_id)
        if other is None:
            continue
        when = other.span.format_in(other.zone) if other.span else "unscheduled"
        blocking.append(f"{other.code} {when} - {other.title}")

    # What it would actually cost to reach this session from the plan. Useful even
    # when the rejection was not about travel: it answers "could I swap something?"
    nearest = _nearest_planned_leg(session, plan, catalog, travel)
    if nearest is not None:
        facts.append(("nearest in plan", nearest))

    return Explanation(
        session_id=session.session_id,
        code=session.code,
        title=session.title,
        verdict=Verdict.REJECTED,
        headline=f"{rejection.reason.label}: {rejection.detail}",
        facts=tuple(facts),
        score_lines=_score_lines(score),
        remedy=_remedy_for(rejection, profile, session),
        alternatives=_alternatives(session, catalog, plan),
        conflicts_with=tuple(blocking),
    )


def explain_worst(
    plan: Plan,
    catalog: Catalog,
    profile: Profile,
    travel: TravelGraph,
    limit: int = 5,
    scorer: Scorer | None = None,
) -> tuple[Explanation, ...]:
    """Explain the highest-scoring sessions that did not make the plan.

    The most useful question a plan can answer, and the one a user asks first:
    what did I nearly get, and what would it cost me to have it?
    """
    resolved_scorer = scorer or Scorer(profile, catalog)
    return tuple(
        explain_session(rejection.session_id, plan, catalog, profile, travel, resolved_scorer)
        for rejection in plan.worst_rejections(limit)
    )


__all__ = ["Explanation", "Verdict", "explain_session", "explain_worst"]
