"""Getting the plan out of the terminal and onto your phone.

The one idea worth stating: **alarms fire when you need to leave, not when the
session starts.** A calendar that pings you at 14:00 for a 14:00 session in another
property has told you something you can no longer act on. replan knows the transfer
takes 24 minutes, so the alarm goes at 13:36 and says so.

Three outputs:

* RFC 5545 ``.ics`` - imports into any calendar, with departure alarms and the
  travel note in each event's description.
* Markdown - for a README, a gist, or sharing with colleagues.
* Printable HTML day sheets - one page per day, because phone batteries die and
  conference wifi is conference wifi.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from icalendar import Alarm, Calendar, Event

from replan.domain import Catalog, day_label, now_utc
from replan.solver import Plan, PlanItem

PRODID = "-//replan//AWS re:Invent agenda//EN"

# A fallback nudge for the first session of a day, where there is no previous
# session to depart from.
FIRST_SESSION_LEAD_MINUTES = 20


def _uid(plan: Plan, item: PlanItem) -> str:
    """A stable UID, so re-importing updates events instead of duplicating them."""
    material = f"{plan.event_id}:{item.session_id}:{item.span.start.isoformat()}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{digest}@replan"


def _description(item: PlanItem, catalog: Catalog | None) -> str:
    parts: list[str] = []

    if item.travel_minutes is not None and item.travel_minutes > 0:
        leave = (
            f" Leave by {item.depart_previous_by.astimezone(item.zone):%H:%M}."
            if item.depart_previous_by
            else ""
        )
        parts.append(f"Getting here: {item.travel_minutes} min by {item.travel_mode}.{leave}")
        if item.travel_detail:
            parts.append(item.travel_detail)
        if item.idle_minutes is not None:
            parts.append(f"Spare time: {item.idle_minutes} min.")
    elif item.travel_minutes == 0:
        parts.append("Same room as your previous session.")

    if item.already_reserved:
        parts.append("Seat: already reserved.")
    elif item.needs_reservation:
        parts.append(f"Seat: needs a reservation ({item.seat_band or 'reservable'}).")
    elif item.seat_band:
        parts.append(f"Seat: {item.seat_band}.")

    if item.top_reasons:
        parts.append("Why this session: " + "; ".join(item.top_reasons) + ".")

    session = catalog.by_id(item.session_id) if catalog else None
    if session is not None:
        if session.speakers:
            parts.append("Speakers: " + ", ".join(s.name for s in session.speakers) + ".")
        if session.level_raw:
            parts.append(f"Level: {session.level_raw}.")
        if session.abstract:
            parts.append("")
            parts.append(session.abstract)

    parts.append("")
    parts.append("Planned by replan. Times shown in the event's local timezone.")
    return "\n".join(parts)


def export_ics(
    plan: Plan,
    path: Path,
    *,
    catalog: Catalog | None = None,
    departure_alarms: bool = True,
) -> Path:
    """Write the plan as an RFC 5545 calendar."""
    calendar = Calendar()
    calendar.add("prodid", PRODID)
    calendar.add("version", "2.0")
    calendar.add("calscale", "GREGORIAN")
    calendar.add("method", "PUBLISH")
    calendar.add("x-wr-calname", f"replan - {plan.event_id}")

    stamp = now_utc()

    for item in plan.items:
        event = Event()
        event.add("uid", _uid(plan, item))
        event.add("dtstamp", stamp)
        event.add("summary", f"{item.code} {item.title}")
        # Written in UTC. Calendar clients localise for the reader, and a fixed
        # offset would be wrong for anyone travelling.
        event.add("dtstart", item.span.start.astimezone(UTC))
        event.add("dtend", item.span.end.astimezone(UTC))
        event.add("location", item.room_display)
        event.add("description", _description(item, catalog))
        event.add("categories", ["re:Invent", "replan"])
        if item.already_reserved:
            event.add("status", "CONFIRMED")
        else:
            event.add("status", "TENTATIVE")

        if departure_alarms:
            alarm = Alarm()
            alarm.add("action", "DISPLAY")
            if item.travel_minutes and item.travel_minutes > 0:
                # The whole point: remind me when to stand up, not when it starts.
                alarm.add(
                    "description",
                    f"Leave now for {item.code} - {item.travel_minutes} min "
                    f"{item.travel_mode} to {item.room_display}",
                )
                alarm.add("trigger", timedelta(minutes=-item.travel_minutes))
            else:
                alarm.add("description", f"{item.code} starts soon - {item.room_display}")
                alarm.add("trigger", timedelta(minutes=-FIRST_SESSION_LEAD_MINUTES))
            event.add_component(alarm)

        calendar.add_component(event)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(calendar.to_ical())
    return path


def _travel_note(item: PlanItem) -> str:
    if item.travel_minutes is None:
        return "first session of the day"
    if item.travel_minutes == 0:
        return "same room"
    leave = (
        f", leave {item.depart_previous_by.astimezone(item.zone):%H:%M}"
        if item.depart_previous_by
        else ""
    )
    tight = " **tight**" if item.is_tight_connection else ""
    return f"{item.travel_minutes} min {item.travel_mode}{leave}{tight}"


def export_markdown(plan: Plan, path: Path, *, catalog: Catalog | None = None) -> Path:
    """Write the plan as Markdown."""
    lines: list[str] = [
        f"# {plan.event_id} agenda",
        "",
        f"Profile **{plan.profile_name}**. "
        f"{len(plan.items)} sessions, {plan.total_minutes // 60}h "
        f"{plan.total_minutes % 60}m of content, "
        f"{plan.total_travel_minutes} min in transit.",
        "",
    ]

    if plan.goals:
        lines.append("What this plan is for:")
        lines.append("")
        lines.extend(f"- {goal}" for goal in plan.goals)
        lines.append("")

    stats = plan.stats
    lines.append(
        f"Solved by {stats.solver.label} "
        f"({'proven optimal' if stats.optimal else stats.status}) in "
        f"{stats.wall_seconds:.2f}s, covering {stats.topics_covered} of "
        f"{stats.topics_available} topics."
    )
    lines.append("")

    if plan.warnings:
        lines.append("> [!WARNING]")
        for warning in plan.warnings:
            lines.append(f"> {warning}")
        lines.append("")

    for current, items in plan.by_day():
        lines.append(f"## {day_label(current).title()} {current.isoformat()}")
        lines.append("")
        lines.append("| Time | Code | Session | Level | Where | Seat | Getting there |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in items:
            local_start, local_end = item.span.local(item.zone)
            seat = (
                "held"
                if item.already_reserved
                else ("reserve" if item.needs_reservation else (item.seat_band or "-"))
            )
            title = item.title.replace("|", "\\|")
            # Quote the API's own level string when the catalog is to hand.
            session = catalog.by_id(item.session_id) if catalog else None
            level = (session.level_raw or "-") if session else "-"
            lines.append(
                f"| {local_start:%H:%M}-{local_end:%H:%M} | {item.code} | {title} "
                f"| {level} | {item.room_display} | {seat} | {_travel_note(item)} |"
            )
        lines.append("")

    needing = plan.needs_reservation
    if needing:
        lines.append("## Seats to reserve")
        lines.append("")
        lines.extend(f"- {item.code} {item.title}" for item in needing)
        lines.append("")
        lines.append("Run `replan apply --plan <plan.json>` to preview the writes.")
        lines.append("")

    worst = plan.worst_rejections(8)
    if worst:
        lines.append("## Highest-scoring sessions that did not fit")
        lines.append("")
        lines.append("| Code | Score | Why not |")
        lines.append("| --- | --- | --- |")
        for rejection in worst:
            detail = rejection.detail.replace("|", "\\|")
            lines.append(f"| {rejection.code} | {rejection.score:.0f} | {detail} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"Generated by replan from the {plan.catalog_source} catalog"
        + (
            f", fetched {plan.catalog_fetched_at:%Y-%m-%d %H:%M} UTC"
            if plan.catalog_fetched_at
            else ""
        )
        + "."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_DAY_SHEET_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ day_title }} - {{ event_id }}</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 1.5rem; color: #10202f; background: #fff;
    line-height: 1.45; font-size: 15px;
  }
  header { border-bottom: 3px solid #10202f; padding-bottom: .6rem; margin-bottom: 1.2rem; }
  h1 { margin: 0 0 .2rem; font-size: 1.5rem; letter-spacing: -.01em; }
  .meta { color: #5a6b7c; font-size: .85rem; }
  .slot {
    display: grid; grid-template-columns: 5.5rem 1fr; gap: 1rem;
    padding: .8rem 0; border-bottom: 1px solid #dfe6ec; page-break-inside: avoid;
  }
  .time { font-variant-numeric: tabular-nums; font-weight: 600; font-size: .95rem; }
  .time .end { display: block; font-weight: 400; color: #5a6b7c; font-size: .85rem; }
  .code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .8rem;
          background: #eef2f6; padding: .1rem .4rem; border-radius: 3px; }
  .title { font-weight: 600; margin: .25rem 0; }
  .where { color: #33475b; font-size: .9rem; }
  .move { margin-top: .35rem; font-size: .85rem; color: #5a6b7c; }
  .move.tight { color: #b3261e; font-weight: 600; }
  .badge { display: inline-block; font-size: .7rem; text-transform: uppercase;
           letter-spacing: .04em; padding: .1rem .4rem; border-radius: 3px;
           margin-left: .35rem; vertical-align: middle; }
  .badge.held { background: #d7f2e0; color: #0d5c2c; }
  .badge.reserve { background: #fdf0d2; color: #7a4f00; }
  .badge.pin { background: #eadcf7; color: #4b2170; }
  .why { font-size: .82rem; color: #6b7b8c; margin-top: .3rem; font-style: italic; }
  footer { margin-top: 1.5rem; font-size: .78rem; color: #77899a; }
  @media print {
    body { padding: 0; font-size: 11pt; }
    .slot { padding: .5rem 0; }
    footer { position: fixed; bottom: 0; }
  }
</style>
</head>
<body>
<header>
  <h1>{{ day_title }}</h1>
  <div class="meta">
    {{ event_id }} &middot; {{ items|length }} sessions &middot;
    {{ transit }} min in transit &middot; profile &ldquo;{{ profile_name }}&rdquo;
  </div>
</header>

{% for item in items %}
<div class="slot">
  <div class="time">
    {{ item.start }}
    <span class="end">{{ item.end }}</span>
  </div>
  <div>
    <span class="code">{{ item.code }}</span>
    {%- if item.pinned %}<span class="badge pin">pinned</span>{% endif %}
    {%- if item.held %}<span class="badge held">seat held</span>
    {%- elif item.needs_reservation %}<span class="badge reserve">reserve</span>{% endif %}
    <div class="title">{{ item.title }}</div>
    <div class="where">{{ item.room }}</div>
    {%- if item.move %}
    <div class="move{% if item.tight %} tight{% endif %}">{{ item.move }}</div>
    {%- endif %}
    {%- if item.why %}
    <div class="why">{{ item.why }}</div>
    {%- endif %}
  </div>
</div>
{% endfor %}

<footer>
  Generated by replan from the {{ catalog_source }} catalog.
  Travel times are estimates; alarms in the .ics export fire at departure time.
</footer>
</body>
</html>
"""


def export_day_sheets_html(plan: Plan, directory: Path) -> tuple[Path, ...]:
    """Write one printable HTML page per day, plus an index."""
    from jinja2 import Environment, select_autoescape

    environment = Environment(autoescape=select_autoescape(["html"]))
    template = environment.from_string(_DAY_SHEET_TEMPLATE)

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for current, items in plan.by_day():
        rendered_items = []
        for item in items:
            local_start, local_end = item.span.local(item.zone)
            note = _travel_note(item).replace("**", "")
            rendered_items.append(
                {
                    "start": f"{local_start:%H:%M}",
                    "end": f"{local_end:%H:%M}",
                    "code": item.code,
                    "title": item.title,
                    "room": item.room_display,
                    "move": "" if item.travel_minutes is None else note,
                    "tight": item.is_tight_connection,
                    "held": item.already_reserved,
                    "needs_reservation": item.needs_reservation and not item.already_reserved,
                    "pinned": item.pinned,
                    "why": "; ".join(item.top_reasons),
                }
            )

        html = template.render(
            day_title=f"{day_label(current).title()} {current.isoformat()}",
            event_id=plan.event_id,
            profile_name=plan.profile_name,
            items=rendered_items,
            transit=sum(item.travel_minutes or 0 for item in items),
            catalog_source=plan.catalog_source,
        )
        target = directory / f"{current.isoformat()}.html"
        target.write_text(html, encoding="utf-8")
        written.append(target)

    if written:
        index_lines = [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            f"<title>replan - {plan.event_id}</title>",
            "<style>body{font-family:system-ui,sans-serif;margin:3rem auto;max-width:34rem;"
            "padding:0 1.5rem;line-height:1.6}a{color:#0b6bcb}</style></head><body>",
            f"<h1>{plan.event_id} day sheets</h1><ul>",
        ]
        for target in written:
            index_lines.append(f"<li><a href='{target.name}'>{target.stem}</a></li>")
        index_lines.append("</ul></body></html>")
        index = directory / "index.html"
        index.write_text("\n".join(index_lines), encoding="utf-8")
        written.append(index)

    return tuple(written)


def _iso_week_safe(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "FIRST_SESSION_LEAD_MINUTES",
    "PRODID",
    "export_day_sheets_html",
    "export_ics",
    "export_markdown",
]
