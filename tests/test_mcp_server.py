"""Tests for replan's own MCP server.

Three things are worth protecting here, and they are all about an agent driving this
rather than a human.

**The write path cannot be taken blind.** ``apply_plan`` without ``confirm`` must return
a diff and change nothing, and it must refuse a plan built from recorded sample data
against a live account.

**Refusals keep their remedy.** An unhandled exception reaches the client as "Error
executing tool X" with the message stripped, which is useless. Everything replan raises
already says what to do, so the tests check the sentence survives.

**Plan ids resolve the way an agent uses them.** "latest", an explicit id, and an id
from a previous process all have to work, because an MCP client is free to restart the
server between turns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from replan.adapters.auth import NotSignedInError
from replan.adapters.factory import SurfaceChoice
from replan.adapters.recorded import RecordedCatalogClient
from replan.cache import CatalogCache
from replan.config import Settings
from replan.domain import Catalog
from replan.mcp_server import ReplanContext, build_server
from replan.mcp_server.context import PLAN_HISTORY_LIMIT, ToolInputError
from replan.solver import Plan, SolveRequest, solve

PROFILE_YAML = """\
name: test-profile
topics:
  agent: 5
  bedrock: 3
max_sessions_per_day: 3
min_break_minutes: 5
walk_tolerance_minutes: 45
"""


@pytest.fixture
def workspace(tmp_path: Path, recorded_dir: Path) -> dict[str, Any]:
    """A state dir with the recorded catalog synced into it, plus a real profile."""
    state = tmp_path / "state"
    state.mkdir()
    profile = tmp_path / "profile.yaml"
    profile.write_text(PROFILE_YAML, encoding="utf-8")

    port = RecordedCatalogClient(recorded_dir, reservations_open=True)
    event_id = port.recorded_event_id
    catalog = port.list_sessions(event_id)

    settings = Settings(state_dir=state, event_id=event_id)
    CatalogCache(settings.cache_path).save(catalog)

    return {
        "settings": settings,
        "profile": profile,
        "event_id": event_id,
        "catalog": catalog,
        "state": state,
    }


def make_context(workspace: dict[str, Any], **kwargs: Any) -> ReplanContext:
    return ReplanContext(
        workspace["settings"],
        surface=SurfaceChoice.RECORDED,
        profile_path=workspace["profile"],
        **kwargs,
    )


async def call(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    result = await server.call_tool(name, arguments)
    payload: dict[str, Any] = result.structured_content or {}
    return payload


@pytest.fixture
def server(workspace: dict[str, Any]) -> Any:
    return build_server(make_context(workspace))


# --------------------------------------------------------------------------- #
# The tool surface itself
# --------------------------------------------------------------------------- #
async def test_every_tool_is_annotated_and_documented(server: Any) -> None:
    """Annotations are how a client decides what to auto-approve, so none may be absent."""
    tools = await server.list_tools()
    assert len(tools) >= 12
    for tool in tools:
        assert tool.description, f"{tool.name} has no description"
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.annotations.read_only_hint is not None
        assert tool.output_schema is not None, f"{tool.name} returns no structured output"


async def test_only_apply_plan_is_marked_destructive(server: Any) -> None:
    """Exactly one tool can change somebody's real schedule, and it says so."""
    destructive = {
        tool.name
        for tool in await server.list_tools()
        if tool.annotations is not None and tool.annotations.destructive_hint
    }
    assert destructive == {"apply_plan"}


async def test_network_tools_are_flagged_open_world(server: Any) -> None:
    """A client should be able to tell which tools leave the machine."""
    open_world = {
        tool.name
        for tool in await server.list_tools()
        if tool.annotations is not None and tool.annotations.open_world_hint
    }
    assert open_world == {"sync_catalog", "my_schedule", "preview_plan", "apply_plan"}


async def test_apply_plan_is_not_marked_idempotent(server: Any) -> None:
    """The API has no idempotency key and reports an existing seat as a failure, so
    re-sending a batch is not a safe retry. Claiming otherwise would invite one."""
    tools = {tool.name: tool for tool in await server.list_tools()}
    annotations = tools["apply_plan"].annotations
    assert annotations is not None
    assert annotations.idempotent_hint is False


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
async def test_catalog_status_reports_the_cache_and_the_profile(
    server: Any, workspace: dict[str, Any]
) -> None:
    status = await call(server, "catalog_status")
    assert status["cached"] is True
    assert status["sessions"] == len(workspace["catalog"])
    assert status["profileName"] == "test-profile"
    assert status["profileSource"] == str(workspace["profile"])
    assert status["topics"] == ["agent", "bedrock"]
    assert "8 October 2026" in status["reservationsNote"]


async def test_catalog_status_warns_when_no_profile_was_found(
    workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A default profile scores every session zero, which looks like a broken optimiser
    rather than a missing file unless somebody says so."""
    monkeypatch.delenv("REPLAN_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path / "state")
    context = ReplanContext(
        workspace["settings"], surface=SurfaceChoice.RECORDED, profile_path=None
    )
    status = await call(build_server(context), "catalog_status")
    assert status["profileSource"] is None
    assert any("profile init" in line for line in status["advice"])


async def test_status_reports_every_problem_not_just_the_first(tmp_path: Path) -> None:
    """A fresh install is missing both a catalog and a profile, and being told about
    only one sends the agent round the loop twice."""
    settings = Settings(state_dir=tmp_path / "empty", event_id="reinvent2026")
    context = ReplanContext(settings, surface=SurfaceChoice.RECORDED)
    status = await call(build_server(context), "catalog_status")
    assert status["cached"] is False
    assert any("sync_catalog" in line for line in status["advice"])
    assert any("profile init" in line for line in status["advice"])


async def test_status_says_when_the_catalog_is_only_sample_data(server: Any) -> None:
    status = await call(server, "catalog_status")
    assert any("sample data" in line for line in status["advice"])


# --------------------------------------------------------------------------- #
# Reads never walk the live catalog
# --------------------------------------------------------------------------- #
async def test_search_refuses_with_a_remedy_when_nothing_is_cached(tmp_path: Path) -> None:
    """The refusal has to name the fix, because an agent cannot guess it."""
    settings = Settings(state_dir=tmp_path / "empty", event_id="reinvent2026")
    server = build_server(ReplanContext(settings, surface=SurfaceChoice.RECORDED))
    with pytest.raises(ToolError) as caught:
        await server.call_tool("search_sessions", {"query": "lambda"})
    message = str(caught.value)
    assert "sync_catalog" in message
    assert "quota" in message


async def test_search_finds_real_sessions(server: Any) -> None:
    result = await call(server, "search_sessions", query="AI", limit=5)
    assert result["matched"] > 0
    first = result["results"][0]
    assert first["code"]
    assert first["title"]
    assert "score" in first


async def test_search_rejects_a_bad_day(server: Any) -> None:
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        await server.call_tool("search_sessions", {"day": "next tuesday"})


async def test_search_rejects_an_unknown_venue_rather_than_returning_nothing(
    server: Any,
) -> None:
    """Silently matching zero sessions would look like the catalog was empty."""
    with pytest.raises(ToolError, match="Known venues"):
        await server.call_tool("search_sessions", {"venue": "Hotel California"})


async def test_rank_sessions_explains_each_score(server: Any) -> None:
    result = await call(server, "rank_sessions", limit=3)
    assert result["ranked"] >= 1
    assert all(row["why"] for row in result["results"])
    assert result["results"] == sorted(
        result["results"], key=lambda row: row["score"], reverse=True
    )


# --------------------------------------------------------------------------- #
# Travel, the thing the catalog cannot tell you
# --------------------------------------------------------------------------- #
async def test_travel_time_between_two_venues(server: Any) -> None:
    """The signature number: a 51-minute shuttle the catalog does not mention at all,
    which is over the profile's 45-minute tolerance."""
    result = await call(server, "travel_time", origin="Venetian", destination="Mandalay Bay")
    assert result["leg"]["minutes"] > 45
    assert result["leg"]["mode"] == "shuttle"
    assert result["walkToleranceMinutes"] == 45
    assert result["withinWalkTolerance"] is False


async def test_travel_time_between_two_sessions_reports_feasibility(
    server: Any, workspace: dict[str, Any]
) -> None:
    catalog: Catalog = workspace["catalog"]
    scheduled = [item for item in catalog.sessions if item.span is not None]
    first, second = scheduled[0], scheduled[-1]
    result = await call(server, "travel_time", origin=first.code, destination=second.code)
    assert "transfer" in result
    assert "gapMinutes" in result["transfer"]


async def test_travel_time_flags_a_guess_as_a_guess(server: Any) -> None:
    """Presenting a pessimistic default like a measured figure is the failure mode the
    travel graph exists to prevent, so the warning is part of the contract."""
    result = await call(server, "travel_time", origin="Narnia", destination="Venetian")
    assert result["leg"]["confidence"] == 0
    assert "pessimistic" in result["warning"]


async def test_two_unknown_venues_are_assumed_to_be_one_building(server: Any) -> None:
    """Right for a single-site Summit, which is exactly what the recorded catalog is:
    it publishes no venue field at all."""
    result = await call(server, "travel_time", origin="Hall A", destination="Hall B")
    assert result["leg"]["confidence"] == 1
    assert "warning" not in result


# --------------------------------------------------------------------------- #
# Solving, and passing plans around by id
# --------------------------------------------------------------------------- #
async def test_optimize_returns_a_plan_id_and_a_readable_agenda(server: Any) -> None:
    result = await call(server, "optimize_agenda", time_limit_seconds=5)
    assert result["planId"]
    assert result["sessions"] > 0
    assert result["byDay"]
    assert result["objective"]["total"] > 0
    day = result["byDay"][0]
    assert day["items"][0]["start"]
    assert day["items"][0]["code"]


async def test_optimize_rejects_a_day_the_catalog_does_not_have(server: Any) -> None:
    """Returning an empty plan would read as a broken optimiser rather than a typo."""
    with pytest.raises(ToolError, match="Available:"):
        await server.call_tool("optimize_agenda", {"days": ["2001-01-01"]})


async def test_latest_resolves_to_the_most_recent_plan(server: Any) -> None:
    first = await call(server, "optimize_agenda", time_limit_seconds=3)
    second = await call(server, "optimize_agenda", time_limit_seconds=3, candidates_per_day=20)
    assert first["planId"] != second["planId"]

    latest = await call(server, "coverage_report")
    assert latest["planId"] == second["planId"]


async def test_a_plan_survives_a_server_restart(workspace: dict[str, Any]) -> None:
    """An MCP client may restart the server between turns; an id that stopped resolving
    would be baffling."""
    first = build_server(make_context(workspace))
    plan_id = (await call(first, "optimize_agenda", time_limit_seconds=3))["planId"]

    fresh = build_server(make_context(workspace))
    again = await call(fresh, "coverage_report", plan_id=plan_id)
    assert again["planId"] == plan_id


async def test_unknown_plan_id_lists_the_ones_that_exist(server: Any) -> None:
    await call(server, "optimize_agenda", time_limit_seconds=3)
    with pytest.raises(ToolError, match="Known:"):
        await server.call_tool("coverage_report", {"plan_id": "not-a-plan"})


async def test_a_plan_id_cannot_escape_the_plans_directory(server: Any) -> None:
    """Plan ids become filenames, so a path separator has to be refused rather than
    resolved."""
    with pytest.raises(ToolError, match="not a valid plan id"):
        await server.call_tool("coverage_report", {"plan_id": "../../etc/passwd"})


async def test_coverage_report_before_any_plan_says_what_to_do(server: Any) -> None:
    with pytest.raises(ToolError, match="optimize_agenda"):
        await server.call_tool("coverage_report", {})


def solve_a_plan(context: ReplanContext) -> Plan:
    """A real plan, built the way the tool builds one."""
    catalog = context.catalog()
    return solve(
        SolveRequest(
            catalog=catalog,
            profile=context.profile,
            scores=context.scorer().by_id(),
            travel=context.travel,
            time_limit_seconds=3.0,
        )
    )


async def test_two_plans_solved_in_the_same_second_get_different_ids(server: Any) -> None:
    """Second resolution is fine for a filename and nowhere near enough for an agent
    that solves twice in one breath."""
    first = await call(server, "optimize_agenda", time_limit_seconds=2)
    second = await call(server, "optimize_agenda", time_limit_seconds=2)
    assert first["planId"] != second["planId"]


def test_plan_history_is_pruned(workspace: dict[str, Any]) -> None:
    """An agent iterating on a profile produces a lot of plans and none are worth
    unbounded disk."""
    context = make_context(workspace)
    plan = solve_a_plan(context)
    for index in range(PLAN_HISTORY_LIMIT + 5):
        context.remember_plan(plan, plan_id=f"plan-{index:03d}")
    assert len(list(context.plans_dir.glob("*.json"))) == PLAN_HISTORY_LIMIT


# --------------------------------------------------------------------------- #
# Explaining
# --------------------------------------------------------------------------- #
async def test_explain_says_why_something_is_in_the_plan(server: Any) -> None:
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    code = plan["byDay"][0]["items"][0]["code"]
    result = await call(server, "explain_decision", session=code, plan_id=plan["planId"])
    assert result["explanation"]["verdict"] == "scheduled"
    assert result["explanation"]["scoring"]


async def test_explain_worst_names_a_remedy_for_each_rejection(server: Any) -> None:
    """A planner that silently drops the session you wanted is one you stop trusting."""
    await call(server, "optimize_agenda", time_limit_seconds=5)
    result = await call(server, "explain_decision", worst=True, limit=3)
    assert result["mode"] == "worst"
    assert result["explanations"]
    assert all(item["headline"] for item in result["explanations"])


async def test_explain_an_unknown_session_is_an_answer_not_a_crash(server: Any) -> None:
    await call(server, "optimize_agenda", time_limit_seconds=3)
    result = await call(server, "explain_decision", session="NOPE999")
    assert result["explanation"]["verdict"] == "unknown-session"
    assert "replan sync" in result["explanation"]["toChangeThis"]


async def test_coverage_report_separates_your_topics_from_solver_breadth(server: Any) -> None:
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    result = await call(server, "coverage_report", plan_id=plan["planId"])
    assert result["topicsRequested"] == 2
    assert set(result["covered"]) | {item["topic"] for item in result["missed"]} == {
        "agent",
        "bedrock",
    }
    assert "/" in result["solverBreadth"]


# --------------------------------------------------------------------------- #
# whats_next
# --------------------------------------------------------------------------- #
async def test_whats_next_reports_departure_times(server: Any, workspace: dict[str, Any]) -> None:
    """The start time is not the actionable number; the time to leave is."""
    catalog: Catalog = workspace["catalog"]
    day = sorted(catalog.days)[0]
    result = await call(server, "whats_next", at=f"{day.isoformat()}T09:00:00+00:00", limit=3)
    assert result["duringEvent"] is True
    assert result["moves"]
    assert all("urgency" in move for move in result["moves"])


async def test_whats_next_outside_the_event_still_answers(
    server: Any, workspace: dict[str, Any]
) -> None:
    result = await call(server, "whats_next", at="2020-01-01T09:00:00+00:00")
    assert result["duringEvent"] is False


# --------------------------------------------------------------------------- #
# Writing back
# --------------------------------------------------------------------------- #
async def test_preview_writes_nothing_and_says_how_to_apply(server: Any) -> None:
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    preview = await call(server, "preview_plan", plan_id=plan["planId"])
    assert preview["dryRun"] is True
    assert preview["actions"]
    assert preview["targets"] >= 1
    assert "confirm=True" in preview["toApply"]


async def test_apply_without_confirm_is_a_dry_run(server: Any) -> None:
    """The two-step is the guard: you cannot write without having been shown the diff."""
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    result = await call(server, "apply_plan", plan_id=plan["planId"])
    assert result["applied"] is False
    assert result["dryRun"] is True
    assert "confirm=true" in result["toApply"]


async def test_apply_with_confirm_writes_and_verifies(server: Any) -> None:
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    result = await call(server, "apply_plan", plan_id=plan["planId"], confirm=True)
    assert result["applied"] is True
    assert result["ok"] is True
    assert result["verified"] is True
    assert result["runId"]
    assert "undo" in result["toUndo"]


async def test_applying_twice_finds_nothing_left_to_do(server: Any) -> None:
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    await call(server, "apply_plan", plan_id=plan["planId"], confirm=True)
    second = await call(server, "apply_plan", plan_id=plan["planId"], confirm=True)
    assert second["applied"] is False
    assert "already matches" in second["summary"]


async def test_pruning_is_opt_in(server: Any) -> None:
    """The API returns favorites as bare ids with no metadata, so replan cannot tell one
    it added from one you starred yourself. Removal has to be asked for."""
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    default = await call(server, "preview_plan", plan_id=plan["planId"])
    kinds = {action["kind"] for action in default["actions"]}
    assert not any("remove" in kind or "cancel" in kind for kind in kinds)


async def test_a_recorded_plan_is_never_written_to_a_live_account(
    workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session ids in a recording belong to a different event, so this would book
    the wrong things. Nobody ever wants it."""
    from replan.mcp_server import server as server_module

    context = make_context(workspace)
    server = build_server(context)
    plan = await call(server, "optimize_agenda", time_limit_seconds=3)

    class LivePort:
        surface_name = "live-rest"

        def get_schedule(self, event_id: str) -> Any:
            return RecordedCatalogClient().get_schedule(event_id)

    monkeypatch.setattr(context, "build_port", lambda *args, **kwargs: LivePort())
    assert server_module.surface_label(LivePort()) == "live-rest"

    with pytest.raises(ToolError, match="recorded sample data"):
        await server.call_tool("apply_plan", {"plan_id": plan["planId"], "confirm": True})


async def test_a_diff_is_never_computed_against_a_guessed_schedule(
    workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assuming an empty schedule would propose booking things you already hold."""
    context = make_context(workspace)
    server = build_server(context)
    plan = await call(server, "optimize_agenda", time_limit_seconds=3)

    class UnsignedPort:
        surface_name = "live-rest"

        def get_schedule(self, event_id: str) -> Any:
            raise NotSignedInError("not signed in; run `replan login`")

    monkeypatch.setattr(context, "build_port", lambda *args, **kwargs: UnsignedPort())
    with pytest.raises(ToolError, match="guessing it empty"):
        await server.call_tool("preview_plan", {"plan_id": plan["planId"]})


async def test_my_schedule_describes_ids_using_the_cached_catalog(server: Any) -> None:
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    await call(server, "apply_plan", plan_id=plan["planId"], confirm=True)
    mine = await call(server, "my_schedule")
    assert mine["favoriteCount"] >= 1
    assert all(row["title"] for row in mine["favorites"])


# --------------------------------------------------------------------------- #
# Watching
# --------------------------------------------------------------------------- #
async def test_watch_session_adds_and_reports(server: Any, workspace: dict[str, Any]) -> None:
    catalog: Catalog = workspace["catalog"]
    code = catalog.sessions[0].code
    added = await call(server, "watch_session", sessions=[code, "NOPE999"])
    assert added["added"] == [code]
    assert added["notFound"] == ["NOPE999"]

    status = await call(server, "watch_status")
    assert status["watching"] == 1
    assert status["targets"][0]["code"] == code


async def test_watch_seeded_from_a_plan_only_takes_what_needs_a_seat(server: Any) -> None:
    """This Summit publishes no isReservable flag, so there is nothing to wait for."""
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    result = await call(server, "watch_session", plan_id=plan["planId"])
    assert result["added"] == []


# --------------------------------------------------------------------------- #
# Syncing
# --------------------------------------------------------------------------- #
async def test_sync_catalog_refreshes_the_cache_and_reports_the_diff(
    server: Any, workspace: dict[str, Any]
) -> None:
    result = await call(server, "sync_catalog")
    assert result["sessions"] == len(workspace["catalog"])
    assert result["surface"] == "recorded"
    assert result["summary"]


async def test_sync_catalog_rejects_an_unknown_surface(server: Any) -> None:
    with pytest.raises(ToolError, match="surface must be one of"):
        await server.call_tool("sync_catalog", {"surface": "carrier-pigeon"})


# --------------------------------------------------------------------------- #
# The context itself
# --------------------------------------------------------------------------- #
def test_reloading_a_profile_forgets_derived_scores(workspace: dict[str, Any]) -> None:
    """Scores are memoised per event, so an edited profile has to invalidate them or the
    next plan would silently use the old weights."""
    context = make_context(workspace)
    before = context.scorer().profile.name
    assert before == "test-profile"

    workspace["profile"].write_text("name: edited\ntopics:\n  lambda: 9\n", encoding="utf-8")
    context.reload_profile()
    assert context.scorer().profile.name == "edited"


def test_a_bad_profile_is_reported_not_swallowed(workspace: dict[str, Any]) -> None:
    """Falling back to defaults on a typo would silently plan against the wrong
    preferences, which is worse than refusing."""
    workspace["profile"].write_text("walk_tolerance_minutes: -5\n", encoding="utf-8")
    context = make_context(workspace)
    with pytest.raises(ToolInputError):
        _ = context.profile


def test_a_list_of_topics_means_equal_weights(workspace: dict[str, Any]) -> None:
    """A convenience worth keeping: nobody wants to type ': 1' four times."""
    workspace["profile"].write_text("topics: [agent, bedrock]\n", encoding="utf-8")
    context = make_context(workspace)
    assert context.profile.topics == {"agent": 1.0, "bedrock": 1.0}


def test_remember_plan_survives_an_unwritable_state_dir(
    workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the plan because the disk said no would be a poor trade: the in-memory
    copy still answers every follow-up in this session."""
    context = make_context(workspace)
    plan = solve_a_plan(context)

    def refuse(self: Plan, path: Path) -> Path:
        raise OSError("read-only file system")

    monkeypatch.setattr(Plan, "write", refuse)
    stored = context.remember_plan(plan)
    assert stored.path is None
    assert context.plan(stored.plan_id).plan is plan


def test_reading_an_old_plan_does_not_make_it_the_latest(workspace: dict[str, Any]) -> None:
    """ "latest" has to mean the one we were just talking about, not the last one read."""
    context = make_context(workspace)
    plan = solve_a_plan(context)
    old = context.remember_plan(plan, plan_id="older")
    newest = context.remember_plan(plan, plan_id="newest")

    assert context.plan(old.plan_id).plan_id == "older"
    assert context.plan(None).plan_id == newest.plan_id


def test_ports_are_reused_across_tool_calls(workspace: dict[str, Any]) -> None:
    """The quota ledger lives on the client, so a fresh port per call would reset the
    pacing and walk straight through the published per-minute limits."""
    context = make_context(workspace)
    assert context.build_port() is context.build_port()
    context.close()
    assert context.build_port() is not None


def test_plan_ids_are_stable_and_sortable(workspace: dict[str, Any]) -> None:
    moment = datetime(2026, 11, 30, 17, 5, 9, tzinfo=UTC)
    context = make_context(workspace, clock=lambda: moment)
    assert context.new_plan_id("reinvent2026") == "reinvent2026-20261130-170509"


def test_event_id_falls_back_to_settings(workspace: dict[str, Any]) -> None:
    context = make_context(workspace)
    assert context.event_or_default(None) == workspace["event_id"]
    assert context.event_or_default("   ") == workspace["event_id"]
    assert context.event_or_default("other") == "other"


# --------------------------------------------------------------------------- #
# Content, not just structure
# --------------------------------------------------------------------------- #
async def test_results_are_json_serialisable(server: Any) -> None:
    """Structured output has to survive the transport, so nothing exotic may leak in."""
    plan = await call(server, "optimize_agenda", time_limit_seconds=5)
    for name, arguments in (
        ("catalog_status", {}),
        ("search_sessions", {"query": "AI", "limit": 3}),
        ("rank_sessions", {"limit": 3}),
        ("travel_time", {"origin": "Venetian", "destination": "Wynn"}),
        ("coverage_report", {"plan_id": plan["planId"]}),
        ("explain_decision", {"worst": True, "limit": 2}),
        ("watch_status", {}),
    ):
        result = await server.call_tool(name, arguments)
        json.dumps(result.structured_content)


async def test_times_are_rendered_in_the_events_own_timezone(
    server: Any, workspace: dict[str, Any]
) -> None:
    """A Dubai Summit session shown in Pacific time would be wrong by eleven hours."""
    catalog: Catalog = workspace["catalog"]
    scheduled = next(item for item in catalog.sessions if item.span is not None)
    result = await call(server, "search_sessions", query=scheduled.code, limit=5)
    row = next(item for item in result["results"] if item["code"] == scheduled.code)
    local_start, _ = scheduled.span.local(scheduled.zone)  # type: ignore[union-attr]
    assert row["start"] == local_start.strftime("%H:%M")
