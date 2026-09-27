"""Exporting crew plans into shared calendars and markdown reports."""

from __future__ import annotations

from pathlib import Path

from replan.crew.models import CrewPlan
from replan.domain import Catalog
from replan.explain.exports import export_ics


def export_crew_ics(
    crew_plan: CrewPlan,
    output_dir: Path,
    *,
    catalog: Catalog | None = None,
) -> list[Path]:
    """Write an individual .ics calendar file for every member of the crew."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name, plan in crew_plan.members.items():
        # Sanitize member name for filesystem
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        ics_path = output_dir / f"{safe_name}.ics"
        export_ics(plan, ics_path, catalog=catalog)
        created.append(ics_path)
    return created


def export_crew_markdown(
    crew_plan: CrewPlan,
    path: Path,
    *,
    catalog: Catalog | None = None,
) -> Path:
    """Generate a shared Markdown agenda and topic coverage manifesto for the crew."""
    _ = catalog
    report = crew_plan.coverage
    lines: list[str] = [
        f"# {crew_plan.event_id} - Crew Coverage Plan",
        "",
        f"**Generated:** {crew_plan.created_at:%Y-%m-%d %H:%M UTC} | "
        f"**Members:** {len(crew_plan.members)} | "
        f"**Unique Sessions:** {crew_plan.unique_session_count} | "
        f"**Topic Coverage:** {report.coverage_percentage:.1f}%",
        "",
        "## Executive Summary",
        "",
        f"- **Collective Topic Coverage:** {report.covered_topics}/{report.total_topics} "
        f"({report.coverage_percentage:.1f}%) priority topics covered across the team.",
        f"- **Session Allocation:** {crew_plan.total_sessions} total seat assignments across "
        f"{crew_plan.unique_session_count} distinct sessions.",
        f"- **Redundant Sessions:** {report.redundancy_count} non-keynote duplicate attendances "
        "(minimized by coordinated solver).",
        "",
        "## Member Breakdown",
        "",
        "| Member | Sessions | Primary Topics |",
        "| :--- | :--- | :--- |",
    ]

    for name, plan in sorted(crew_plan.members.items()):
        member_topics = [t.topic for t in report.topics if name in t.covered_by]
        topic_summary = ", ".join(member_topics[:4]) if member_topics else "General"
        lines.append(f"| **{name}** | {len(plan.items)} | {topic_summary} |")

    lines.extend(
        [
            "",
            "## Topic Coverage Matrix",
            "",
            "| Priority Topic | Status | Covered By | Sessions |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    for topic in report.topics:
        status = "✅ Covered" if topic.is_covered else "⚠️ Uncovered"
        covered_by = ", ".join(topic.covered_by) if topic.covered_by else "-"
        sessions = ", ".join(topic.session_codes[:3]) if topic.session_codes else "-"
        lines.append(f"| {topic.topic} | {status} | {covered_by} | {sessions} |")

    if report.shared_sessions:
        lines.extend(
            [
                "",
                "## Shared Sessions (All-Hands / Keynotes)",
                "",
                "Sessions where multiple team members attend together:",
                "",
            ]
        )
        for code, members in report.shared_sessions:
            member_str = ", ".join(members)
            # Find item details from any plan
            title = ""
            when = ""
            where = ""
            for p in crew_plan.members.values():
                item = p.item_for(code)
                if item:
                    title = item.title
                    when = f"{item.span.start:%a %H:%M}-{item.span.end:%H:%M}"
                    where = item.room_display
                    break
            lines.append(f"- **{code}**: {title} ({when} at {where}) -> Attendees: *{member_str}*")

    lines.extend(
        [
            "",
            "## Complete Team Timeline",
            "",
        ]
    )

    # Chronological listing of all sessions
    all_sessions = crew_plan.all_items()
    current_day = None
    for member_name, item in all_sessions:
        if item.day != current_day:
            current_day = item.day
            lines.extend(
                [
                    f"### {item.day:%A, %B %d, %Y}",
                    "",
                    "| Time | Member | Session | Room | Title |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
        time_str = f"{item.span.start:%H:%M} - {item.span.end:%H:%M}"
        row = (
            f"| {time_str} | **{member_name}** | `{item.code}` | "
            f"{item.room_display} | {item.title} |"
        )
        lines.append(row)
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = [
    "export_crew_ics",
    "export_crew_markdown",
]
