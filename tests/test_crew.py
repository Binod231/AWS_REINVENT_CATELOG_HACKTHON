"""Tests for Task 13: Crew mode coordinated multi-attendee scheduling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from replan.adapters.wire import build_catalog, decode_session
from replan.cli.app import app
from replan.crew import (
    CrewMember,
    CrewPlan,
    CrewSolveRequest,
    export_crew_ics,
    export_crew_markdown,
    solve_crew,
)
from replan.domain import Catalog, Session
from replan.profiles import Profile, Scorer
from replan.solver import SolveRequest, solve
from replan.travel import default_graph

runner = CliRunner()

DAY = "2026-12-01"


def session_at(
    code: str,
    clock: str,
    *,
    minutes: int = 60,
    venue: str = "Venetian",
    room: str = "Level 2 | Hall C",
    topics: tuple[str, ...] = ("Bedrock",),
    title: str | None = None,
) -> Session:
    payload: dict[str, Any] = {
        "sessionId": code,
        "abbreviation": code,
        "title": title or f"Session {code}",
        "abstract": f"Abstract for {code}",
        "level": "300 – Advanced",
        "type": "Breakout session",
        "room": room,
        "venue": venue,
        "topics": list(topics),
        "services": ["Amazon Bedrock"],
        "sessionTime": {
            "date": DAY,
            "time": clock,
            "length": str(minutes),
            "timezone": "America/Los_Angeles",
        },
    }
    decoded = decode_session(payload)
    assert decoded is not None
    return decoded


@pytest.fixture
def catalog() -> Catalog:
    sessions = (
        session_at("AIM301", "10:00", topics=("agent", "bedrock"), title="Agentic AI"),
        session_at("AIM302", "11:30", topics=("bedrock", "rag"), title="Bedrock RAG"),
        session_at(
            "ARC301", "10:00", topics=("architecture", "serverless"), title="Modern Architecture"
        ),
        session_at(
            "ARC302", "13:00", topics=("serverless", "lambda"), title="Serverless at Scale"
        ),
        session_at("ARC303", "14:30", topics=("architecture", "cloud"), title="Cloud Architecture"),
        session_at(
            "COP301", "14:30", topics=("containers", "kubernetes"), title="Kubernetes Patterns"
        ),
        session_at("DEV301", "14:30", topics=("serverless", "devops"), title="DevOps at Scale"),
        session_at("KEY001", "08:30", topics=("keynote",), title="CEO Keynote"),
    )
    return build_catalog(event_id="reinvent2026", sessions=sessions, source="test")


def test_single_member_crew_matches_single_plan(catalog: Catalog) -> None:
    """When a crew has 1 member, solve_crew degrades cleanly to standard single-user solve."""
    profile = Profile(name="Solo", topics={"agent": 3.0, "bedrock": 2.0})
    member = CrewMember(name="Solo", profile=profile)
    graph = default_graph()

    crew_req = CrewSolveRequest(
        catalog=catalog,
        members=(member,),
        travel=graph,
    )
    crew_plan = solve_crew(crew_req)

    scorer = Scorer(profile, catalog)
    single_req = SolveRequest(
        catalog=catalog,
        profile=profile,
        scores=scorer.by_id(),
        travel=graph,
    )
    single_plan = solve(single_req)

    assert "Solo" in crew_plan.members
    assert len(crew_plan.members["Solo"].items) == len(single_plan.items)
    crew_codes = [item.code for item in crew_plan.members["Solo"].items]
    single_codes = [item.code for item in single_plan.items]
    assert crew_codes == single_codes


def test_two_member_crew_diversifies_coverage(catalog: Catalog) -> None:
    """When two members share topic interests, crew mode penalizes redundant attendance."""
    profile1 = Profile(name="Alice", topics={"agent": 3.0, "architecture": 3.0})
    profile2 = Profile(name="Bob", topics={"agent": 3.0, "serverless": 2.0})
    graph = default_graph()

    member1 = CrewMember(name="Alice", profile=profile1)
    member2 = CrewMember(name="Bob", profile=profile2)

    crew_req = CrewSolveRequest(
        catalog=catalog,
        members=(member1, member2),
        travel=graph,
        redundancy_penalty=50.0,
    )
    crew_plan = solve_crew(crew_req)

    alice_sessions = {item.session_id for item in crew_plan.members["Alice"].items}
    bob_sessions = {item.session_id for item in crew_plan.members["Bob"].items}

    # They should not be assigned to the exact same breakout session
    shared = alice_sessions.intersection(bob_sessions)
    assert len(shared) <= 1


def test_coverage_report_calculation(catalog: Catalog) -> None:
    """Topic coverage percentage and breakdown are accurately calculated."""
    p1 = Profile(name="P1", topics={"agent": 3.0, "lambda": 2.0})
    p2 = Profile(name="P2", topics={"bedrock": 3.0, "containers": 2.0})
    m1 = CrewMember(name="P1", profile=p1)
    m2 = CrewMember(name="P2", profile=p2)
    graph = default_graph()

    plan = solve_crew(CrewSolveRequest(catalog=catalog, members=(m1, m2), travel=graph))
    report = plan.coverage

    assert report.total_topics == 4
    assert 0.0 <= report.coverage_percentage <= 100.0
    assert len(report.topics) == 4
    assert "P1" in report.member_session_counts
    assert "P2" in report.member_session_counts


def test_crew_plan_serialization_roundtrip(catalog: Catalog) -> None:
    """CrewPlan serializes to dict/json and deserializes losslessly."""
    p1 = Profile(name="Alice", topics={"agent": 3.0})
    m1 = CrewMember(name="Alice", profile=p1)
    plan = solve_crew(CrewSolveRequest(catalog=catalog, members=(m1,), travel=default_graph()))

    payload = plan.to_dict()
    assert payload["eventId"] == plan.event_id
    assert "members" in payload
    assert "Alice" in payload["members"]

    restored = CrewPlan.from_dict(payload)
    assert restored.event_id == plan.event_id
    assert len(restored.members["Alice"].items) == len(plan.members["Alice"].items)
    assert restored.coverage.total_topics == plan.coverage.total_topics


def test_export_crew_ics_and_markdown(catalog: Catalog, tmp_path: Path) -> None:
    """Exporting ICS and Markdown creates valid files."""
    p1 = Profile(name="Alice", topics={"agent": 3.0})
    p2 = Profile(name="Bob", topics={"bedrock": 2.0})
    m1 = CrewMember(name="Alice", profile=p1)
    m2 = CrewMember(name="Bob", profile=p2)
    plan = solve_crew(CrewSolveRequest(catalog=catalog, members=(m1, m2), travel=default_graph()))

    ics_dir = tmp_path / "calendars"
    created_ics = export_crew_ics(plan, ics_dir, catalog=catalog)
    assert len(created_ics) == 2
    for ics in created_ics:
        assert ics.exists()
        assert ics.read_text(encoding="utf-8").startswith("BEGIN:VCALENDAR")

    md_path = tmp_path / "crew_report.md"
    export_crew_markdown(plan, md_path, catalog=catalog)
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Crew Coverage Plan" in content
    assert "Alice" in content
    assert "Bob" in content


def test_cli_crew_command(tmp_path: Path) -> None:
    """CLI replan crew command executes successfully with profile directory."""
    profiles_dir = Path("examples/crew")
    out_json = tmp_path / "crew.json"
    out_ics = tmp_path / "ics"
    out_md = tmp_path / "crew.md"

    result = runner.invoke(
        app,
        [
            "crew",
            "--profiles",
            str(profiles_dir),
            "--surface",
            "recorded",
            "-e",
            "Summit-Dubai-2026",
            "--out",
            str(out_json),
            "--ics-dir",
            str(out_ics),
            "--markdown",
            str(out_md),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "replan crew" in result.output
    assert "Crew Members" in result.output
    assert out_json.exists()
    assert out_md.exists()
    assert out_ics.exists()
    assert len(list(out_ics.glob("*.ics"))) == 3
