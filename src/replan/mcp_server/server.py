"""replan's own MCP server: the optimiser, exposed to a coding agent.

There are two MCP servers in this story and they do different jobs. The one AWS
publishes at ``api.awsevents.com/mcp`` is a transport - it hands an agent the same
twelve catalog operations the REST API has. This one is the reasoning layer: it answers
"which sessions, in what order, and can I physically get there", which is the part the
catalog cannot answer for you.

Three things shape the tool surface.

**No tool makes a slow call by accident.** Catalog reads come from the local SQLite
cache, so ``search_sessions`` is a millisecond FTS5 query rather than a few hundred
paginated requests against a per-minute quota. Refreshing the cache is
``sync_catalog``, which an agent calls on purpose.

**Writing is a two-step move that cannot be skipped.** ``apply_plan`` without
``confirm=True`` returns the dry-run diff and writes nothing, so the agent has
necessarily seen what would change before anything does. ``preview_plan`` is the same
diff under a read-only annotation, for clients that auto-approve read-only tools.

**Plans are passed by id, not by value.** ``optimize_agenda`` returns a plan id and a
compact summary; ``explain_decision``, ``coverage_report`` and ``apply_plan`` take that
id. Returning fifty sessions of JSON that the agent then has to send back would spend
the context window on bookkeeping.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from replan.adapters.auth import AuthError, NotSignedInError, TokenStore
from replan.adapters.errors import EventsApiError
from replan.adapters.factory import SurfaceChoice, surface_label
from replan.cache import SearchFilters
from replan.config import RESERVATIONS_OPEN_NOTE, Settings
from replan.domain import (
    Catalog,
    KnownVenue,
    Level,
    Room,
    SeatAvailability,
    Session,
    SessionFormat,
)
from replan.explain import explain_session, explain_worst
from replan.live import parse_location, replan_now
from replan.mcp_server.context import ReplanContext, StoredPlan, ToolInputError
from replan.profiles import ProfileError, Score
from replan.solver import Plan, PlanItem, SolveRequest, solve
from replan.sync import ActionPlan, DiffOptions, build_action_plan
from replan.sync import apply_plan as run_apply
from replan.travel import Confidence, Leg
from replan.watch import Watcher

SERVER_NAME = "replan"

INSTRUCTIONS = """\
replan turns the AWS Events catalog into a schedule you can actually walk.

Start with catalog_status. If nothing is cached, call sync_catalog once - every other
read works off that local copy, which is why they are fast.

The usual path: search_sessions or rank_sessions to explore, optimize_agenda to solve
for a conflict-free travel-feasible agenda, explain_decision to find out why something
did or did not make it, whats_next during the event itself.

Writing back to the attendee's real schedule goes through apply_plan, which is a dry
run unless you pass confirm=true. Read the diff it returns before you confirm it.

Two facts worth knowing up front. The catalog gives venue and room as free text with no
distances, so replan carries its own Las Vegas travel graph - that is what travel_time
queries and what makes a plan feasible rather than merely non-overlapping. And reserved
seating does not open through this API until 8 October 2026; until then reservations
return HTTP 409 for everyone, while favorites and personal time work normally.
"""

MAX_SEARCH_LIMIT = 100
MAX_RANK_LIMIT = 100


def _actionable[**P](fn: Callable[P, dict[str, Any]]) -> Callable[P, dict[str, Any]]:
    """Turn replan's own errors into tool errors that keep their remedy.

    An unhandled exception reaches the client as "Error executing tool X" with the
    message stripped, which is useless to an agent trying to recover. Everything
    replan raises already explains what to do about it, so it is worth preserving.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except (ToolInputError, ProfileError) as error:
            raise ToolError(str(error)) from error
        except EventsApiError as error:
            raise ToolError(f"{error} {error.remedy}") from error
        except AuthError as error:
            raise ToolError(f"{error} Run `replan login` in a terminal.") from error

    return wrapper


# --------------------------------------------------------------------------- #
# rendering
#
# Every tool returns plain JSON-able dicts. Compact on purpose: an agenda is read by a
# model with a budget, so each row carries what a decision needs and nothing else.
# --------------------------------------------------------------------------- #
def _clock(value: datetime | None) -> str | None:
    return value.strftime("%H:%M") if value else None


def _stamp(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _band(value: SeatAvailability | None) -> str | None:
    return value.value if value else None


def _session_row(session: Session) -> dict[str, Any]:
    # Rendered in the event's own timezone, never the caller's: a Dubai Summit session
    # shown in Pacific time would be wrong by eleven hours.
    local = session.span.local(session.zone) if session.span else None
    return {
        "code": session.code,
        "sessionId": session.session_id,
        "title": session.title,
        "day": session.day.isoformat() if session.day else None,
        "start": _clock(local[0]) if local else None,
        "end": _clock(local[1]) if local else None,
        "minutes": session.duration_minutes,
        "level": session.level.value,
        "format": session.format.value,
        "room": session.room.display if session.room else None,
        "venue": session.room.venue if session.room else None,
        "seats": _band(session.seat_availability),
        "reservable": session.is_reservable,
    }


def _score_row(score: Score, session: Session) -> dict[str, Any]:
    row = _session_row(session)
    row["score"] = round(score.total, 1)
    row["why"] = [factor.describe() for factor in score.factors[:4]]
    if score.veto is not None:
        row["vetoed"] = score.veto.value
        row["vetoDetail"] = score.veto_detail
    return row


def _plan_item_row(item: PlanItem) -> dict[str, Any]:
    start, end = item.span.local(item.zone)
    row: dict[str, Any] = {
        "code": item.code,
        "sessionId": item.session_id,
        "title": item.title,
        "day": item.day.isoformat(),
        "start": _clock(start),
        "end": _clock(end),
        "room": item.room_display,
        "score": round(item.score, 1),
        "seats": item.seat_band,
        "needsReservation": item.needs_reservation,
        "alreadyReserved": item.already_reserved,
    }
    if item.travel_minutes:
        row["travelMinutes"] = item.travel_minutes
        row["travelDetail"] = item.travel_detail
    if item.depart_previous_by is not None:
        row["departPreviousBy"] = _clock(item.depart_previous_by.astimezone(item.zone))
    if item.idle_minutes:
        row["idleMinutes"] = item.idle_minutes
    return row


def _plan_summary(stored: StoredPlan, *, include_items: bool = True) -> dict[str, Any]:
    plan = stored.plan
    stats = plan.stats
    summary: dict[str, Any] = {
        "planId": stored.plan_id,
        "eventId": plan.event_id,
        "profileName": plan.profile_name,
        "generatedAt": _stamp(plan.generated_at),
        "catalogSource": plan.catalog_source,
        "sessions": len(plan.items),
        "days": [day.isoformat() for day in plan.days],
        "totalScore": round(plan.total_score, 1),
        "totalTravelMinutes": plan.total_travel_minutes,
        "solver": stats.solver.value,
        "solverStatus": stats.status,
        "provenOptimal": stats.optimal,
        "solveSeconds": round(stats.wall_seconds, 2),
        "candidates": stats.candidates,
        "topicsCovered": f"{stats.topics_covered}/{stats.topics_available}",
        "objective": plan.objective.to_dict(),
        "needsReservation": [item.code for item in plan.needs_reservation],
        "tightConnections": [
            {"code": item.code, "travelMinutes": item.travel_minutes}
            for item in plan.tight_connections
        ],
        "warnings": list(plan.warnings),
    }
    if stats.fallback_reason:
        summary["fallbackReason"] = stats.fallback_reason
    if include_items:
        summary["byDay"] = [
            {
                "day": day.isoformat(),
                "label": plan.day_label_for(day),
                "items": [_plan_item_row(item) for item in items],
            }
            for day, items in plan.by_day()
        ]
    if stored.path is not None:
        summary["planFile"] = str(stored.path)
    return summary


def _explanation_row(explanation: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "code": explanation.code,
        "sessionId": explanation.session_id,
        "title": explanation.title,
        "verdict": explanation.verdict.value,
        "headline": explanation.headline,
        "facts": dict(explanation.facts),
    }
    if explanation.conflicts_with:
        row["blockedBy"] = list(explanation.conflicts_with)
    if explanation.alternatives:
        row["otherShowings"] = list(explanation.alternatives)
    if explanation.score_lines:
        row["scoring"] = list(explanation.score_lines)
    if explanation.remedy:
        row["toChangeThis"] = explanation.remedy
    return row


def _leg_row(leg: Leg) -> dict[str, Any]:
    return {
        "minutes": leg.minutes,
        "mode": leg.mode.value,
        "confidence": leg.confidence.value,
        "confidenceLabel": leg.confidence.label,
        "transitMinutes": leg.transit_minutes,
        "overheadMinutes": leg.overhead_minutes,
        "detail": leg.detail,
        "from": leg.origin.display if leg.origin else None,
        "to": leg.destination.display if leg.destination else None,
    }


def _action_row(action: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": action.kind.value,
        "describe": action.describe(),
        "targets": action.target_count,
        "reason": action.reason,
    }
    if action.session_ids:
        row["sessionIds"] = list(action.session_ids)
    if action.title:
        row["title"] = action.title
    return row


def _action_plan_row(action_plan: ActionPlan) -> dict[str, Any]:
    return {
        "summary": action_plan.summary(),
        "isEmpty": action_plan.is_empty,
        # Actions are batched, so "4 sessions" and "1 request" are both true and only
        # one of them is what a reader wants to know.
        "targets": action_plan.total_targets,
        "destructive": [_action_row(item) for item in action_plan.destructive],
        "actions": [_action_row(item) for item in action_plan.ordered],
        "quotaCost": action_plan.quota_cost(),
        "notes": list(action_plan.notes),
    }


def _diff_options(
    *,
    favorites: bool,
    reservations: bool,
    personal_time: bool,
    prune_favorites: bool,
    prune_reservations: bool,
) -> DiffOptions:
    return DiffOptions(
        favorite_plan_items=favorites,
        reserve_seats=reservations,
        create_personal_time=personal_time,
        prune_favorites=prune_favorites,
        prune_reservations=prune_reservations,
    )


def _as_room(token: str, context: ReplanContext, event_id: str) -> tuple[Room, str]:
    """Interpret a free-text token as either a session's room or a venue.

    Both readings are useful: "how far is it from my 2pm to my 3pm" and "how far is
    Mandalay Bay from the Venetian" are the same question asked two ways.
    """
    catalog = context.catalog(event_id)
    session = catalog.resolve(token)
    if session is not None and session.room is not None:
        return session.room, f"{session.code} ({session.room.display})"

    room = parse_location(token, catalog)
    if room is None:
        msg = (
            f"Could not read {token!r} as a place. Give a session code, a venue name "
            "like 'Mandalay Bay', or 'Venue | Room'."
        )
        raise ToolInputError(msg)
    return room, room.display


def _is_recorded(source: str) -> bool:
    """Whether a catalog came from committed sample data.

    Substring rather than prefix on purpose. The cache decorates the provenance it
    stores - a synced recording reads back as ``cache (recorded)`` - and a safety check
    that a decoration can defeat is not a safety check.
    """
    return "recorded" in source.lower()


def _signed_in(settings: Settings) -> tuple[bool, str | None]:
    """Local token check, so status never costs a network round trip."""
    token = TokenStore(settings.token_path).load()
    if token is None:
        return False, None
    return not token.is_expired, token.signed_in_as


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #
def build_server(context: ReplanContext | None = None) -> MCPServer:
    """Wire up every tool against one shared, lazily built context."""
    ctx = context if context is not None else ReplanContext()
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        title="replan - re:Invent agenda intelligence",
        instructions=INSTRUCTIONS,
        version="0.1.0",
    )

    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False)
    local_write = ToolAnnotations(read_only_hint=False, destructive_hint=False)
    remote_read = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)

    # --- orientation ------------------------------------------------------- #

    @server.tool(annotations=read_only)
    @_actionable
    def catalog_status(event_id: str | None = None) -> dict[str, Any]:
        """What replan currently knows: cached catalog, profile in use, sign-in, plans.

        Call this first. It costs nothing and it answers the two questions that make
        every other tool's output make sense - how fresh the catalog is, and whether a
        real profile is loaded or built-in defaults are standing in for one.
        """
        wanted = ctx.event_or_default(event_id)
        info = ctx.cache.info(wanted)
        signed_in, identity = _signed_in(ctx.settings)
        profile_source = ctx.profile_source

        status: dict[str, Any] = {
            "eventId": wanted,
            "cached": info is not None,
            "signedIn": signed_in,
            "signedInAs": identity,
            "surface": ctx.surface.value,
            "stateDir": str(ctx.settings.state_dir),
            "profileSource": str(profile_source) if profile_source else None,
            "profileName": ctx.profile.name,
            "topics": sorted(ctx.profile.normalised_topics),
            "cachedEvents": list(ctx.cache.cached_events()),
            "plans": list(ctx.known_plan_ids()),
            "travelGraph": ctx.travel.describe_source(),
            "reservationsNote": RESERVATIONS_OPEN_NOTE,
        }
        if info is not None:
            status["sessions"] = info.session_count
            status["catalogTotalCount"] = info.total_count
            status["catalogSource"] = info.source
            status["fetchedAt"] = _stamp(info.fetched_at)
            status["syncCount"] = info.sync_count
            status["lastSync"] = info.last_sync_summary
            status["fullTextSearch"] = ctx.cache.supports_full_text

        # A list, because a fresh install is missing both a catalog and a profile, and
        # being told about only one of them sends the agent round the loop twice.
        advice: list[str] = []
        if info is None:
            advice.append(f"Call sync_catalog to fetch the {wanted} catalog.")
        if profile_source is None:
            advice.append(
                "No profile found, so every session scores zero and a plan would be "
                "arbitrary. Run `replan profile init` and edit profile.yaml, or set "
                "REPLAN_PROFILE."
            )
        if _is_recorded(str(info.source) if info else ""):
            advice.append(
                "This catalog came from committed sample data, not the live API. "
                "Fine for exploring; replan will refuse to write a plan built from it "
                "to a real account."
            )
        status["advice"] = advice
        return status

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=True))
    @_actionable
    def sync_catalog(
        event_id: str | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        """Walk the live catalog and refresh the local cache. Slow on purpose, so explicit.

        This is the one tool that pages through the API. On a large event that is
        hundreds of requests against a 120-per-minute quota and takes minutes, which is
        exactly why the read tools never do it implicitly. Reports what changed since
        last time, flagging changes that would break an existing plan.

        surface: "rest" (live REST), "mcp" (live remote MCP), "recorded" (committed
        sample data, no credentials), or omit for automatic.
        """
        wanted = ctx.event_or_default(event_id)
        # None means "whatever this server was started with", not "guess again": a
        # server launched with --surface recorded must not silently go live.
        port = ctx.build_port(_surface_choice(surface) if surface else None)
        started = time.monotonic()
        catalog = port.list_sessions(wanted)

        diff = ctx.cache.save(catalog)
        ctx.forget_catalog(wanted)

        result: dict[str, Any] = {
            "eventId": wanted,
            "surface": surface_label(port),
            "sessions": len(catalog),
            "totalCount": catalog.total_count,
            "complete": catalog.is_complete,
            "elapsedSeconds": round(time.monotonic() - started, 1),
            "summary": diff.summary(),
            "firstSync": diff.is_first_sync,
        }
        breaking = diff.plan_breaking
        if breaking:
            result["planBreakingChanges"] = [change.describe() for change in breaking[:25]]
            result["advice"] = (
                "Some sessions moved, changed room, or were cancelled. Re-run "
                "optimize_agenda before applying anything."
            )
        return result

    # --- exploring --------------------------------------------------------- #

    @server.tool(annotations=read_only)
    @_actionable
    def search_sessions(
        query: str = "",
        level: str | None = None,
        session_format: str | None = None,
        day: str | None = None,
        venue: str | None = None,
        track: str | None = None,
        service: str | None = None,
        reservable_only: bool = False,
        bookable_only: bool = False,
        limit: int = 20,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Full-text search over the cached catalog, with filters.

        The Events API has no search or filter parameters at all - you fetch sessions
        and narrow them yourself. replan does that narrowing in SQLite FTS5 over the
        synced copy, so this is a local millisecond query rather than an API call.

        Leave query empty to filter only. day is YYYY-MM-DD. level is 100/200/300/400.
        bookable_only keeps sessions with a seat free right now; reservable_only keeps
        the ones that take a reservation at all.
        """
        wanted = ctx.event_or_default(event_id)
        ctx.catalog(wanted)  # fail early with the sync advice if nothing is cached
        filters = SearchFilters(
            levels=frozenset({Level.parse(level)}) if level else frozenset(),
            formats=(
                frozenset({SessionFormat.parse(session_format)}) if session_format else frozenset()
            ),
            days=frozenset({_parse_day(day)}) if day else frozenset(),
            venues=_parse_venue(venue),
            tracks=frozenset({track}) if track else frozenset(),
            services=frozenset({service}) if service else frozenset(),
            reservable_only=reservable_only,
            bookable_only=bookable_only,
        )
        capped = max(1, min(limit, MAX_SEARCH_LIMIT))
        hits = ctx.cache.search(wanted, query, filters=filters, limit=capped)
        scorer = ctx.scorer(wanted)

        rows: list[dict[str, Any]] = []
        for hit in hits:
            row = _session_row(hit.session)
            row["score"] = round(scorer.score(hit.session).total, 1)
            if hit.snippet:
                row["snippet"] = hit.snippet
            rows.append(row)

        return {
            "eventId": wanted,
            "query": query,
            "matched": len(rows),
            "truncated": len(rows) >= capped,
            "results": rows,
        }

    @server.tool(annotations=read_only)
    @_actionable
    def rank_sessions(
        limit: int = 15,
        day: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Rank the catalog by how well it matches the profile, with the reasons.

        This is preference, not scheduling: nothing here accounts for overlaps or for
        whether you could walk between two of them. optimize_agenda does that. Useful
        for sanity-checking the profile - if the top of this list is not what you
        wanted, fix profile.yaml before solving.
        """
        wanted = ctx.event_or_default(event_id)
        catalog = ctx.catalog(wanted)
        scorer = ctx.scorer(wanted)
        capped = max(1, min(limit, MAX_RANK_LIMIT))

        scores = scorer.ranked()
        rows: list[dict[str, Any]] = []
        for score in scores:
            session = catalog.by_id(score.session_id)
            if session is None:
                continue
            if day and (session.day is None or session.day.isoformat() != day):
                continue
            rows.append(_score_row(score, session))
            if len(rows) >= capped:
                break

        result: dict[str, Any] = {
            "eventId": wanted,
            "profileSource": str(ctx.profile_source) if ctx.profile_source else None,
            "ranked": len(rows),
            "results": rows,
        }
        if ctx.profile_source is None:
            result["warning"] = (
                "No profile loaded, so these scores are all zero and the order is "
                "arbitrary. Run `replan profile init`."
            )
        return result

    @server.tool(annotations=read_only)
    @_actionable
    def travel_time(
        origin: str,
        destination: str,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """How long it really takes to get between two places, or two sessions.

        The catalog gives venue and room as free text and publishes no distances, so a
        schedule can look perfectly valid and be unwalkable. replan carries a Las Vegas
        travel graph with door-to-door times, room-exit and room-entry overheads, and a
        confidence level on each figure.

        Accepts session codes on either side. Given two sessions it also reports
        whether the transfer is actually makeable in the gap between them.
        """
        wanted = ctx.event_or_default(event_id)
        catalog = ctx.catalog(wanted)
        first_room, first_label = _as_room(origin, ctx, wanted)
        second_room, second_label = _as_room(destination, ctx, wanted)
        leg = ctx.travel.leg(first_room, second_room)

        result: dict[str, Any] = {
            "origin": first_label,
            "destination": second_label,
            "leg": _leg_row(leg),
            "describe": leg.describe(),
            "walkToleranceMinutes": ctx.profile.walk_tolerance_minutes,
            "withinWalkTolerance": leg.minutes <= ctx.profile.walk_tolerance_minutes,
        }
        if leg.confidence is Confidence.UNKNOWN:
            # A guessed number presented like a measured one is the failure mode this
            # whole graph exists to avoid, so say it in words as well as a field.
            result["warning"] = (
                "Neither place is a venue replan has data for, so this is a "
                "deliberately pessimistic guess rather than a measurement. Known "
                "venues: " + ", ".join(sorted(item.label for item in KnownVenue))
            )

        first_session = catalog.resolve(origin)
        second_session = catalog.resolve(destination)
        if first_session is not None and second_session is not None:
            feasibility = ctx.travel.feasibility(
                first_session,
                second_session,
                extra_buffer=ctx.profile.min_break_minutes,
            )
            result["transfer"] = {
                "ok": feasibility.ok,
                "reason": feasibility.reason,
                "gapMinutes": feasibility.gap_minutes,
                "slackMinutes": feasibility.slack_minutes,
                "tight": feasibility.is_tight,
            }
        return result

    # --- solving ----------------------------------------------------------- #

    @server.tool(annotations=local_write)
    @_actionable
    def optimize_agenda(
        days: list[str] | None = None,
        candidates_per_day: int = 45,
        time_limit_seconds: float = 10.0,
        use_schedule: bool = False,
        greedy: bool = False,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Solve for a conflict-free, travel-feasible agenda. Returns a plan id.

        Hard constraints: no overlaps, every transfer physically makeable inside your
        walk tolerance, one showing per repeated session, protected time honoured, daily
        session cap. Soft objective: session value, breadth of topic coverage, minus
        travel, venue-hopping, idle-gap and fatigue costs.

        use_schedule reads the attendee's real schedule first, so held seats are pinned
        and existing personal time is respected. That needs sign-in. greedy forces the
        fallback heuristic instead of CP-SAT, which is only useful for comparison.

        Keep the returned planId: explain_decision, coverage_report and apply_plan all
        take it.
        """
        wanted = ctx.event_or_default(event_id)
        catalog = ctx.catalog(wanted)
        scorer = ctx.scorer(wanted)

        notes: list[str] = []
        attendee = None
        if use_schedule:
            try:
                attendee = ctx.build_port().get_schedule(wanted)
                notes.append(
                    f"pinned {len(attendee.reserved)} held seat(s) and "
                    f"{len(attendee.personal_time)} personal-time block(s)"
                )
            except (EventsApiError, AuthError) as error:
                # Worth solving anyway: a plan that ignores held seats is still a
                # useful answer, as long as nobody is told it accounted for them.
                notes.append(f"could not read your schedule ({error}); solved without it")

        selected_days = _parse_day_list(days, catalog)
        request = SolveRequest(
            catalog=catalog,
            profile=ctx.profile,
            scores=scorer.by_id(),
            travel=ctx.travel,
            schedule=attendee,
            days=selected_days,
            candidates_per_day=max(5, candidates_per_day),
            time_limit_seconds=max(0.5, time_limit_seconds),
        )
        plan = solve(request, prefer_cpsat=not greedy)
        stored = ctx.remember_plan(plan)

        result = _plan_summary(stored)
        result["notes"] = notes
        if ctx.profile_source is None:
            result["warning"] = (
                "Solved against a default profile, so every session scored zero and "
                "this plan is essentially arbitrary. Run `replan profile init`."
            )
        if not plan.items:
            result["advice"] = (
                "Nothing could be scheduled. Usually the profile is too restrictive: "
                "check avoid lists, level filters, walk tolerance and max sessions "
                "per day."
            )
        return result

    @server.tool(annotations=read_only)
    @_actionable
    def explain_decision(
        session: str | None = None,
        plan_id: str | None = None,
        worst: bool = False,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Why a session is, or is not, in a plan - and what to change to flip it.

        Every answer names the stage that decided: the profile scored it low, the
        travel graph found it unreachable, or the solver had something better in the
        slot. Pass worst=true instead of a session to get the highest-scoring sessions
        that did not make it, which is the fastest way to find a profile setting that
        is too strict.
        """
        stored = ctx.plan(plan_id)
        plan = stored.plan
        catalog = ctx.catalog(plan.event_id)
        scorer = ctx.scorer(plan.event_id)

        if worst or session is None:
            explanations = explain_worst(
                plan,
                catalog,
                ctx.profile,
                ctx.travel,
                limit=max(1, min(limit, 20)),
                scorer=scorer,
            )
            return {
                "planId": stored.plan_id,
                "mode": "worst",
                "explanations": [_explanation_row(item) for item in explanations],
            }

        explanation = explain_session(session, plan, catalog, ctx.profile, ctx.travel, scorer)
        return {
            "planId": stored.plan_id,
            "mode": "session",
            "explanation": _explanation_row(explanation),
        }

    @server.tool(annotations=read_only)
    @_actionable
    def coverage_report(plan_id: str | None = None) -> dict[str, Any]:
        """Which of your stated topics the plan covers, and which it missed.

        Two different numbers here. Per-topic coverage answers "did I get what I asked
        for", matched the same way the scorer matches, so it agrees with the scores.
        The breadth figure is the solver's own: how many distinct topics across the
        whole candidate pool the selection touched, which is what the coverage term in
        the objective was maximising.
        """
        stored = ctx.plan(plan_id)
        plan = stored.plan
        catalog = ctx.catalog(plan.event_id)
        scorer = ctx.scorer(plan.event_id)
        weights = ctx.profile.normalised_topics

        covered: dict[str, list[str]] = {topic: [] for topic in sorted(weights)}
        for item in plan.items:
            session = catalog.by_id(item.session_id)
            if session is None:
                continue
            for topic in scorer.topics_matched(session):
                covered.setdefault(topic, []).append(session.code)

        hit = {topic: codes for topic, codes in covered.items() if codes}
        missed = sorted(topic for topic, codes in covered.items() if not codes)

        result: dict[str, Any] = {
            "planId": stored.plan_id,
            "eventId": plan.event_id,
            "topicsRequested": len(weights),
            "topicsCovered": len(hit),
            "covered": {
                topic: {"weight": weights.get(topic, 0.0), "sessions": codes}
                for topic, codes in sorted(hit.items())
            },
            "missed": [{"topic": topic, "weight": weights.get(topic, 0.0)} for topic in missed],
            "solverBreadth": f"{plan.stats.topics_covered}/{plan.stats.topics_available}",
            "objective": plan.objective.to_dict(),
        }
        if missed:
            result["advice"] = (
                "For a missed topic, ask explain_decision about a session that carries "
                "it: either nothing in the catalog matches the keyword, or something "
                "outscored it in every slot."
            )
        if not weights:
            result["warning"] = "The profile lists no topics, so there is nothing to cover."
        return result

    @server.tool(annotations=read_only)
    @_actionable
    def whats_next(
        at: str | None = None,
        location: str | None = None,
        plan_id: str | None = None,
        limit: int = 3,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Standing in a hallway mid-conference: what now, and when to leave.

        Re-plans from where you physically are rather than from the start of the day,
        so a session that is technically next but twenty minutes away is ranked behind
        one you can reach. Every suggestion carries a departure time, not just a start
        time, because the start time is the one that makes you late.

        at accepts "now", a clock time like "14:20", or an ISO timestamp. location
        accepts a session code you just left, a venue, or "Venue | Room".
        """
        wanted = ctx.event_or_default(event_id)
        catalog = ctx.catalog(wanted)
        existing = None
        if plan_id:
            existing = ctx.plan(plan_id).plan

        report = replan_now(
            catalog,
            ctx.profile,
            ctx.travel,
            at=at,
            location=location,
            existing_plan=existing,
            reference=ctx.now(),
            limit=max(1, min(limit, 10)),
        )

        moves: list[dict[str, Any]] = []
        for move in report.moves:
            row = _plan_item_row(move.item)
            row["urgency"] = move.urgency.value
            row["departAt"] = move.depart_clock
            row["minutesUntilDeparture"] = move.minutes_until_departure
            row["travelMinutes"] = move.travel_minutes
            if move.leg is not None:
                row["travelDetail"] = move.leg.detail
            moves.append(row)

        return {
            "eventId": wanted,
            "at": _stamp(report.moment.at),
            "localTime": report.moment.at.astimezone(report.moment.zone).strftime("%a %H:%M"),
            "duringEvent": report.moment.is_during_event,
            "location": report.location.display if report.location else None,
            "inProgress": (_plan_item_row(report.in_progress) if report.in_progress else None),
            "moves": moves,
            "unreachable": [
                {"code": code, "reason": reason} for code, reason in report.unreachable
            ],
            "notes": list(report.notes),
        }

    # --- the attendee's real schedule -------------------------------------- #

    @server.tool(annotations=remote_read)
    @_actionable
    def my_schedule(event_id: str | None = None) -> dict[str, Any]:
        """Read the attendee's real schedule: reservations, favorites, personal time.

        Live and credentialed, unlike every other read here. Needs `replan login`
        first. Favorites come back as bare ids with no metadata, which is why replan
        cannot tell its own favorites from ones you starred by hand.
        """
        wanted = ctx.event_or_default(event_id)
        port = ctx.build_port()
        mine = port.get_schedule(wanted)

        catalog = None
        try:
            catalog = ctx.catalog(wanted)
        except ToolInputError:
            catalog = None

        def describe(ids: tuple[str, ...]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for identifier in ids:
                session = catalog.by_id(identifier) if catalog else None
                rows.append(
                    _session_row(session) if session else {"sessionId": identifier, "title": None}
                )
            return rows

        return {
            "eventId": wanted,
            "surface": surface_label(port),
            "fetchedAt": _stamp(mine.fetched_at),
            "reservedCount": len(mine.reserved),
            "favoriteCount": len(mine.favorites),
            "reserved": describe(mine.reserved),
            "favorites": describe(mine.favorites),
            "personalTime": [
                {
                    "id": block.personal_time_id,
                    "title": block.title,
                    "start": _stamp(block.span.start) if block.span else None,
                    "end": _stamp(block.span.end) if block.span else None,
                    "createdByReplan": not block.is_foreign,
                }
                for block in mine.personal_time
            ],
        }

    # --- writing back ------------------------------------------------------ #

    @server.tool(annotations=remote_read)
    @_actionable
    def preview_plan(
        plan_id: str | None = None,
        favorites: bool = True,
        reservations: bool = True,
        personal_time: bool = True,
        prune_favorites: bool = False,
        prune_reservations: bool = False,
    ) -> dict[str, Any]:
        """The exact diff apply_plan would make. Reads your schedule, writes nothing.

        Adding is the default and removing is opt-in, because the API returns favorites
        as bare ids with no metadata: replan genuinely cannot tell a favorite it added
        from one you starred yourself, so pruning is something you ask for explicitly.
        """
        stored, action_plan, notes = _build_diff(
            plan_id,
            favorites=favorites,
            reservations=reservations,
            personal_time=personal_time,
            prune_favorites=prune_favorites,
            prune_reservations=prune_reservations,
        )
        result = _action_plan_row(action_plan)
        result["planId"] = stored.plan_id
        result["eventId"] = stored.plan.event_id
        result["dryRun"] = True
        result["notes"] = [*result["notes"], *notes]
        result["toApply"] = (
            f"apply_plan(plan_id={stored.plan_id!r}, confirm=True) makes these changes."
        )
        return result

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        )
    )
    @_actionable
    def apply_plan(
        plan_id: str | None = None,
        confirm: bool = False,
        favorites: bool = True,
        reservations: bool = True,
        personal_time: bool = True,
        prune_favorites: bool = False,
        prune_reservations: bool = False,
    ) -> dict[str, Any]:
        """Write a plan to the attendee's real schedule. Dry run unless confirm=true.

        Without confirm this returns the same diff preview_plan does and changes
        nothing, so you always see what would happen before it does. With confirm it
        reserves seats, adds favorites and creates personal-time blocks, then re-reads
        the schedule to check the API agreed, and journals the run so `replan undo` can
        reverse it.

        Not idempotent, and that is the API's design rather than a shortcoming here:
        there is no idempotency key, and an already-held seat comes back as a failure
        rather than a no-op, so re-sending a batch is not a safe retry. Reservations
        return HTTP 409 for everyone until 8 October 2026; favorites and personal time
        work today.
        """
        stored, action_plan, notes = _build_diff(
            plan_id,
            favorites=favorites,
            reservations=reservations,
            personal_time=personal_time,
            prune_favorites=prune_favorites,
            prune_reservations=prune_reservations,
        )

        if not confirm:
            result = _action_plan_row(action_plan)
            result["planId"] = stored.plan_id
            result["dryRun"] = True
            result["applied"] = False
            result["notes"] = [*result["notes"], *notes]
            result["toApply"] = "Nothing was written. Pass confirm=true to apply this diff."
            return result

        if action_plan.is_empty:
            return {
                "planId": stored.plan_id,
                "dryRun": False,
                "applied": False,
                "summary": "Your schedule already matches this plan; nothing to do.",
            }

        port = ctx.build_port()
        label = surface_label(port)
        _refuse_recorded_plan_against_live(stored.plan, label)
        report = run_apply(
            port,
            stored.plan,
            action_plan,
            dry_run=False,
            journal=ctx.journal(),
            verify=True,
        )

        return {
            "planId": stored.plan_id,
            "eventId": report.event_id,
            "runId": report.run_id,
            "surface": label,
            "dryRun": False,
            "applied": True,
            "ok": report.ok,
            "summary": report.summary(),
            "targets": action_plan.total_targets,
            "succeeded": [item.describe() for item in report.succeeded],
            "failed": [item.describe() for item in report.failed],
            "skipped": [item.describe() for item in report.skipped],
            "verified": report.verified,
            "verification": list(report.verification),
            "notes": [*report.notes, *notes],
            "toUndo": f"`replan undo --run {report.run_id} --apply` reverses this.",
        }

    # --- watching ---------------------------------------------------------- #

    @server.tool(annotations=local_write)
    @_actionable
    def watch_session(
        sessions: list[str] | None = None,
        plan_id: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Add full sessions to the seat watch list. Local only; nothing is booked.

        The API publishes a seat band, never a count, and has no waitlist, so "a seat
        opened" is only knowable by comparing against what was seen last time. Adding a
        session records today's band as the baseline. `replan watch run` does the
        polling; this tool just maintains the list.

        Pass plan_id to watch every session in a plan that still needs a seat.
        """
        wanted = ctx.event_or_default(event_id)
        catalog = ctx.catalog(wanted)
        watcher = Watcher(
            None,
            ctx.watch_store(),
            event_id=wanted,
            catalog=catalog,
            profile=ctx.profile,
            travel=ctx.travel,
        )

        added: list[str] = []
        missing: list[str] = []
        if plan_id:
            added.extend(target.code for target in watcher.seed_from_plan(ctx.plan(plan_id).plan))
        for token in sessions or []:
            target = watcher.add(token)
            if target is None:
                missing.append(token)
            else:
                added.append(target.code)

        return {
            "eventId": wanted,
            "added": added,
            "notFound": missing,
            "watching": len(ctx.watch_store().pending(wanted)),
            "toRun": "`replan watch run` polls the list; add --auto-reserve to book.",
        }

    @server.tool(annotations=read_only)
    @_actionable
    def watch_status(event_id: str | None = None) -> dict[str, Any]:
        """The seat watch list and the last band seen for each entry."""
        wanted = ctx.event_or_default(event_id)
        targets = ctx.watch_store().list(wanted)
        return {
            "eventId": wanted,
            "watching": len(targets),
            "targets": [
                {
                    "code": target.code,
                    "sessionId": target.session_id,
                    "title": target.title,
                    "lastBand": _band(target.last_band),
                    "lastSeenAt": _stamp(target.last_seen_at),
                    "checks": target.checks,
                    "settled": target.is_settled,
                }
                for target in targets
            ],
        }

    # --- shared by preview and apply --------------------------------------- #

    def _build_diff(
        plan_id: str | None,
        *,
        favorites: bool,
        reservations: bool,
        personal_time: bool,
        prune_favorites: bool,
        prune_reservations: bool,
    ) -> tuple[StoredPlan, ActionPlan, list[str]]:
        stored = ctx.plan(plan_id)
        plan = stored.plan
        catalog = ctx.catalog(plan.event_id)
        notes: list[str] = []

        try:
            schedule = ctx.build_port().get_schedule(plan.event_id)
        except NotSignedInError as error:
            msg = (
                f"{error} A diff has to be computed against your real schedule, and "
                "guessing it empty would propose booking things you already hold."
            )
            raise ToolInputError(msg) from error

        action_plan = build_action_plan(
            plan,
            schedule,
            catalog,
            ctx.profile,
            _diff_options(
                favorites=favorites,
                reservations=reservations,
                personal_time=personal_time,
                prune_favorites=prune_favorites,
                prune_reservations=prune_reservations,
            ),
        )
        if _is_recorded(plan.catalog_source):
            notes.append("This plan was built from recorded sample data, not the live catalog.")
        return stored, action_plan, notes

    return server


def _parse_day(raw: str) -> date:
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as error:
        msg = f"{raw!r} is not a date. Use YYYY-MM-DD."
        raise ToolInputError(msg) from error


def _parse_venue(raw: str | None) -> frozenset[KnownVenue]:
    """Venue filters only work for venues the travel graph knows.

    Filtering on a venue replan has no data for would silently return nothing, so an
    unrecognised name is refused with the list of ones that work.
    """
    if not raw:
        return frozenset()
    found = KnownVenue.parse(raw)
    if found is None:
        offered = ", ".join(sorted(item.label for item in KnownVenue))
        msg = f"{raw!r} is not a venue replan has data for. Known venues: {offered}"
        raise ToolInputError(msg)
    return frozenset({found})


def _surface_choice(raw: str | None) -> SurfaceChoice:
    if raw is None:
        return SurfaceChoice.AUTO
    try:
        return SurfaceChoice(raw.strip().lower())
    except ValueError as error:
        valid = ", ".join(item.value for item in SurfaceChoice)
        msg = f"surface must be one of: {valid}"
        raise ToolInputError(msg) from error


def _parse_day_list(days: list[str] | None, catalog: Catalog) -> frozenset[date] | None:
    """Turn ISO day strings into dates, refusing days the catalog does not have.

    Solving for a day that is not in the catalog returns an empty plan, which reads as
    a broken optimiser rather than a typo. Better to say which days exist.
    """
    if not days:
        return None
    available = set(catalog.days)
    wanted: set[date] = set()
    for raw in days:
        parsed = _parse_day(raw)
        if parsed not in available:
            offered = ", ".join(day.isoformat() for day in sorted(available))
            msg = f"{raw} is not a day in this catalog. Available: {offered}"
            raise ToolInputError(msg)
        wanted.add(parsed)
    return frozenset(wanted)


def _refuse_recorded_plan_against_live(plan: Plan, surface: str) -> None:
    """Never write a plan built from sample data into somebody's real schedule.

    The session ids in the recordings belong to a different event, so this would
    either fail confusingly or book the wrong things. Nobody ever wants it.
    """
    if _is_recorded(plan.catalog_source) and surface.startswith("live"):
        msg = (
            "This plan was built from recorded sample data, so its session ids do not "
            "belong to the live event. Run sync_catalog against the live API and "
            "optimize_agenda again before applying."
        )
        raise ToolInputError(msg)


def serve(
    *,
    surface: str | None = None,
    profile_path: Path | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8930,
) -> None:
    """Run the server until the client disconnects.

    stdio is what an installed MCP client launches. streamable-http exists so the same
    server can run somewhere else and be shared, and it binds to loopback by default
    because it carries no authentication of its own.
    """
    context = ReplanContext(
        surface=_surface_choice(surface),
        profile_path=profile_path,
    )
    server = build_server(context)
    if transport == "stdio":
        server.run("stdio")
    elif transport == "streamable-http":
        server.run("streamable-http", host=host, port=port)
    elif transport == "sse":
        server.run("sse", host=host, port=port)
    else:
        msg = f"unknown transport {transport!r}; use stdio, streamable-http or sse"
        raise ToolInputError(msg)


__all__ = ["INSTRUCTIONS", "SERVER_NAME", "build_server", "serve"]
