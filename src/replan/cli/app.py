"""The ``replan`` Typer application.

Every command is a thin shell: parse arguments, build the requested surface, call
into the pure core, render the result. No business logic lives here.

Two conventions run through all of it:

* **Provenance is always printed.** You can never be unsure whether you are
  looking at the live API or a recording.
* **Writes are dry-run by default.** Anything that changes an attendee's real
  schedule needs an explicit ``--apply``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from replan import __version__
from replan.adapters.auth import AuthError, TokenProvider, first_free_callback_port
from replan.adapters.auth import login as run_login
from replan.adapters.auth import revoke as revoke_token
from replan.adapters.errors import EventsApiError
from replan.cache import CatalogCache, SearchFilters
from replan.cache.changes import ChangeKind
from replan.cli.factory import (
    SurfaceChoice,
    build_port,
    die,
    load_settings,
    print_provenance,
)
from replan.config import ALLOWED_CALLBACK_PORTS, RESERVATIONS_OPEN_NOTE
from replan.crew import (
    CrewMember,
    CrewSolveRequest,
    export_crew_ics,
    export_crew_markdown,
    solve_crew,
)
from replan.domain import (
    KnownVenue,
    Level,
    Room,
    SeatAvailability,
    SessionFormat,
    resolve_day,
)
from replan.explain import (
    Verdict,
    explain_session,
    explain_worst,
    export_day_sheets_html,
    export_ics,
    export_markdown,
)
from replan.live import Urgency, replan_now
from replan.mcp_server.context import ToolInputError
from replan.profiles import ProfileError, Scorer, load_profile, write_starter_profile
from replan.solver import Plan, SolveRequest, solve
from replan.sync import (
    ActionStatus,
    DiffOptions,
    Journal,
    apply_plan,
    build_action_plan,
    undo_run,
)
from replan.travel import default_graph
from replan.watch import Watcher, WatchPolicy, WatchStore, build_notifier
from replan.watch.watcher import DEFAULT_INTERVAL_SECONDS

console = Console()

err_console = Console(stderr=True)
"""For anything that must not touch stdout. `replan mcp serve` speaks JSON-RPC on
stdout, and a single stray character there corrupts the stream."""

app = typer.Typer(
    name="replan",
    help=(
        "Agenda intelligence for the AWS re:Invent session catalog.\n\n"
        "Reads the live Events API, scores it against your goals, solves for a "
        "schedule you can physically walk, explains every decision, and writes "
        "the result back to your attendee account."
    ),
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

catalog_app = typer.Typer(help="Read the session catalog.", no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")


# --------------------------------------------------------------------------- #
# shared options
# --------------------------------------------------------------------------- #
SurfaceOption = Annotated[
    SurfaceChoice,
    typer.Option(
        "--surface",
        "-s",
        help="Which front door to use: live REST, live MCP, or committed recordings.",
        case_sensitive=False,
    ),
]
EventOption = Annotated[
    str | None,
    typer.Option("--event", "-e", help="Event id, e.g. reinvent2026."),
]
LocaleOption = Annotated[
    str | None,
    typer.Option("--locale", help="BCP-47 tag for localized text, e.g. pt-BR."),
]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"replan {__version__}")
        raise typer.Exit(0)


@app.callback()
def root(
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the replan version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """replan - build a re:Invent agenda that is actually possible to walk."""


# --------------------------------------------------------------------------- #
# login / logout
# --------------------------------------------------------------------------- #
@app.command()
def login(
    port: Annotated[
        int | None,
        typer.Option(
            "--port", help=f"Loopback port. Must be one of {list(ALLOWED_CALLBACK_PORTS)}."
        ),
    ] = None,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Print the URL instead of opening a browser.")
    ] = False,
) -> None:
    """Sign in with your AWS Builder ID.

    Runs the OAuth 2.0 authorization code flow with PKCE on your own machine.
    The API registers only six loopback ports, so one of those must be free.
    """
    settings = load_settings()
    chosen = port
    if chosen is None:
        try:
            chosen = first_free_callback_port()
        except AuthError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
    if chosen not in ALLOWED_CALLBACK_PORTS:
        allowed = ", ".join(str(item) for item in ALLOWED_CALLBACK_PORTS)
        console.print(
            f"[red]Port {chosen} is not registered with the API.[/red]\n"
            f"[yellow]Only these work: {allowed}. Any other port fails with "
            f"redirect_mismatch.[/yellow]"
        )
        raise typer.Exit(1)

    import dataclasses

    settings = dataclasses.replace(settings, callback_port=chosen)
    console.print(f"Signing in via [cyan]{settings.redirect_uri}[/cyan]")

    def show_url(url: str) -> None:
        if no_browser:
            console.print("\nOpen this URL to sign in:\n")
            console.print(url)
        else:
            console.print("[dim]Opening your browser...[/dim]")

    try:
        tokens = run_login(settings, open_browser=not no_browser, on_url=show_url)
    except AuthError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    TokenProvider(settings).remember(tokens)
    who = tokens.signed_in_as
    console.print(
        f"[green]Signed in{f' as {who}' if who else ''}.[/green] "
        f"Access token valid for {tokens.expires_in_seconds // 60} minutes; "
        "it refreshes automatically for 30 days."
    )
    console.print(f"[dim]Token stored at {settings.token_path} (owner-only).[/dim]")


@app.command()
def logout() -> None:
    """Revoke your token and delete it locally.

    Deleting the local file alone is not enough: the token keeps working until it
    is revoked, so this does both.
    """
    settings = load_settings()
    provider = TokenProvider(settings)
    tokens = provider.peek()
    if tokens is None:
        console.print("Not signed in.")
        return
    revoked = revoke_token(settings, tokens)
    provider.forget()
    if revoked:
        console.print("[green]Token revoked and removed.[/green]")
    else:
        console.print(
            "[yellow]Local token removed, but revocation did not confirm.[/yellow]\n"
            "To be certain, end your Builder ID session at "
            "https://profile.aws.amazon.com"
        )


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
@app.command()
def doctor(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
) -> None:
    """Check what replan can reach right now, and what it would need to do more."""
    settings = load_settings(event_id=event)
    console.print(
        Panel.fit(
            f"[bold]replan {__version__}[/bold]\n"
            f"API root   {settings.api_root}\n"
            f"MCP server {settings.mcp_url}\n"
            f"Event      {settings.event_id}\n"
            f"State dir  {settings.state_dir}",
            title="configuration",
            border_style="cyan",
        )
    )

    try:
        port = build_port(settings, surface)
    except EventsApiError as error:
        die(console, error)
        return

    report: dict[str, Any] = port.probe() if hasattr(port, "probe") else {}
    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("result")

    table.add_row("surface", getattr(port, "surface_name", "unknown"))
    reachable = report.get("reachable")
    table.add_row(
        "reachable",
        "[green]yes[/green]" if reachable else f"[red]no[/red] {report.get('error', '')}",
    )
    if "events_visible" in report:
        table.add_row("events visible", str(report["events_visible"]))
    if report.get("public_events"):
        table.add_row("readable without sign-in", ", ".join(report["public_events"][:6]))
    signed_in = report.get("signed_in")
    table.add_row(
        "signed in",
        "[green]yes[/green]" if signed_in else "[yellow]no - run `replan login`[/yellow]",
    )
    if report.get("signed_in_as"):
        table.add_row("identity", str(report["signed_in_as"]))
    if report.get("token"):
        token_info = report["token"]
        table.add_row("token", f"expires in {token_info.get('expires_in_seconds', '?')}s")
    if report.get("tools"):
        table.add_row("mcp tools", str(len(report["tools"])))

    target = report.get("event")
    if isinstance(target, dict):
        table.add_row("event name", str(target.get("name")))
        table.add_row(
            "needs registration",
            "yes" if target.get("requires_sign_in") else "no",
        )
        table.add_row("timezone", str(target.get("timezone")))
        days = target.get("days") or []
        if days:
            table.add_row("days", f"{days[0]} .. {days[-1]} ({len(days)})")

    console.print(table)
    console.print(f"[dim]{RESERVATIONS_OPEN_NOTE}[/dim]")

    if hasattr(port, "close"):
        port.close()


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
@app.command()
def events(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    include_past: Annotated[
        bool, typer.Option("--include-past", help="Also list events that have ended.")
    ] = False,
) -> None:
    """List AWS events. Needs no credentials, for any event."""
    settings = load_settings()
    port = build_port(settings, surface)
    try:
        found = port.list_events(include_past=include_past)
    except EventsApiError as error:
        die(console, error)
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("event id")
    table.add_column("name")
    table.add_column("when")
    table.add_column("timezone")
    table.add_column("sign-in")

    for item in found:
        when = "-"
        if item.start_date and item.end_date:
            start = item.start_date.astimezone(item.zone).date()
            end = item.end_date.astimezone(item.zone).date()
            when = f"{start} .. {end}" if start != end else str(start)
        table.add_row(
            item.event_id,
            item.name,
            when,
            item.timezone_name or "-",
            "[yellow]required[/yellow]" if item.authentication_required else "[green]open[/green]",
        )

    print_provenance(console, port, settings)
    console.print(table)
    console.print(
        f"[dim]{len(found)} event(s). Catalogs marked 'open' are readable with no "
        "credentials at all.[/dim]"
    )
    if hasattr(port, "close"):
        port.close()


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
@catalog_app.command("stats")
def catalog_stats(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
    locale: LocaleOption = None,
) -> None:
    """Walk the whole catalog and summarise it."""
    settings = load_settings(event_id=event, locale=locale)
    port = build_port(settings, surface)

    def progress(pages: int, sessions: int, total: int | None) -> None:
        suffix = f"/{total}" if total else ""
        console.print(f"[dim]  page {pages}: {sessions}{suffix} sessions[/dim]")

    try:
        catalog = port.list_sessions(settings.event_id, locale=settings.locale, progress=progress)
    except EventsApiError as error:
        die(console, error)
        return

    print_provenance(console, port, settings)
    stats = catalog.stats()

    summary = Table.grid(padding=(0, 2))
    summary.add_row("sessions", str(stats.total))
    summary.add_row("scheduled", str(stats.scheduled))
    summary.add_row("unscheduled / all-day", str(stats.unscheduled))
    summary.add_row("distinct content (series)", str(stats.distinct_series))
    summary.add_row("repeat showings", str(stats.repeats))
    summary.add_row("reservable", str(stats.reservable))
    summary.add_row("distinct speakers", str(stats.distinct_speakers))
    summary.add_row("distinct rooms", str(stats.distinct_rooms))
    summary.add_row(
        "walk complete",
        "[green]yes[/green]" if catalog.is_complete else "[red]no[/red]",
    )
    console.print(Panel(summary, title=f"{settings.event_id} catalog", border_style="cyan"))

    _print_counts("by day", stats.by_day)
    _print_counts("by level", stats.by_level)
    _print_counts("by format", stats.by_format, limit=12)
    _print_counts("by track", stats.by_track, limit=12)
    _print_counts("by venue", stats.by_venue, limit=10)
    _print_counts("seat availability", stats.by_seat_availability)

    if hasattr(port, "close"):
        port.close()


def _print_counts(title: str, rows: tuple[tuple[Any, int], ...], *, limit: int = 20) -> None:
    if not rows:
        return
    table = Table(title=title, show_header=False, title_justify="left", box=None)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    for label, count in rows[:limit]:
        table.add_row(str(label), str(count))
    if len(rows) > limit:
        table.add_row("[dim]...[/dim]", f"[dim]+{len(rows) - limit} more[/dim]")
    console.print(table)


@catalog_app.command("show")
def catalog_show(
    session: Annotated[str, typer.Argument(help="Session id or code, e.g. SVS401.")],
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
    locale: LocaleOption = None,
) -> None:
    """Show one session in full."""
    settings = load_settings(event_id=event, locale=locale)
    port = build_port(settings, surface)
    try:
        found = port.get_session(settings.event_id, session, locale=settings.locale)
    except EventsApiError:
        try:
            catalog = port.list_sessions(settings.event_id, locale=settings.locale)
        except EventsApiError as error:
            die(console, error)
            return
        resolved = catalog.resolve(session)
        if resolved is None:
            console.print(f"[red]No session matches {session!r}.[/red]")
            raise typer.Exit(1) from None
        found = resolved

    print_provenance(console, port, settings)
    body = Table.grid(padding=(0, 2))
    body.add_row("code", found.code)
    body.add_row("level", found.level_raw or found.level.label)
    body.add_row("format", found.type_raw or found.format.label)
    # Shown in the *event's* timezone, not ours: a Dubai Summit session must not
    # be printed in Las Vegas time.
    body.add_row(
        "when",
        found.span.format_in(found.zone) if found.span else "not scheduled",
    )
    if found.timezone_name:
        body.add_row("timezone", found.timezone_name)
    body.add_row("where", found.room.display)
    if found.seat_availability:
        body.add_row("seats", found.seat_availability.label)
    elif found.is_reservable is None:
        body.add_row("seats", "[dim]not published for this event[/dim]")
    if found.speakers:
        body.add_row("speakers", "\n".join(s.name for s in found.speakers))
    for label, values in (
        ("tracks", found.effective_tracks),
        ("topics", found.topics),
        ("services", found.services),
        ("roles", found.roles),
        ("industries", found.industries),
    ):
        if values:
            body.add_row(label, ", ".join(values))

    console.print(Panel(body, title=found.title, border_style="cyan"))
    if found.abstract:
        console.print(Panel(found.abstract, title="abstract", border_style="dim"))

    if hasattr(port, "close"):
        port.close()


@catalog_app.command("list")
def catalog_list(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
    level: Annotated[
        int | None, typer.Option("--level", help="Filter by level: 100, 200, 300, 400.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show.")] = 25,
) -> None:
    """List sessions, newest catalog walk, optionally filtered by level."""
    settings = load_settings(event_id=event)
    port = build_port(settings, surface)
    try:
        catalog = port.list_sessions(settings.event_id, include_abstracts=False)
    except EventsApiError as error:
        die(console, error)
        return

    sessions = list(catalog.sessions)
    if level is not None:
        wanted = Level.parse(level)
        sessions = [item for item in sessions if item.level is wanted]

    # Unscheduled sessions sort last rather than crashing the comparison.
    sessions.sort(
        key=lambda item: (item.span is None, item.span.start if item.span else 0, item.code)
    )

    print_provenance(console, port, settings)
    table = Table(show_header=True, header_style="bold")
    table.add_column("code")
    table.add_column("when")
    table.add_column("level")
    table.add_column("format")
    table.add_column("title", overflow="fold")

    for item in sessions[:limit]:
        table.add_row(
            item.code,
            item.span.format_in(item.zone) if item.span else "-",
            str(int(item.level)) if item.level else "-",
            item.format.label,
            item.title,
        )
    console.print(table)
    console.print(f"[dim]showing {min(limit, len(sessions))} of {len(sessions)}[/dim]")

    if hasattr(port, "close"):
        port.close()


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
@app.command()
def schedule(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
) -> None:
    """Show your own schedule: reservations, favorites, personal time."""
    settings = load_settings(event_id=event)
    port = build_port(settings, surface)
    try:
        mine = port.get_schedule(settings.event_id)
    except EventsApiError as error:
        die(console, error)
        return

    print_provenance(console, port, settings)
    console.print(
        f"reserved [bold]{len(mine.reserved)}[/bold]   "
        f"favorites [bold]{len(mine.favorites)}[/bold]   "
        f"personal time [bold]{len(mine.personal_time)}[/bold]"
    )

    if mine.personal_time:
        table = Table(title="personal time", show_header=True, header_style="bold")
        table.add_column("id")
        table.add_column("when (UTC)")
        table.add_column("title")
        table.add_column("by replan")
        for block in mine.personal_time:
            table.add_row(
                block.personal_time_id,
                f"{block.span.start:%Y-%m-%d %H:%M} -> {block.span.end:%H:%M}",
                block.title,
                "yes" if block.created_by_replan else "no",
            )
        console.print(table)

    if hasattr(port, "close"):
        port.close()


# --------------------------------------------------------------------------- #
# sync / search / changes
# --------------------------------------------------------------------------- #
def _parse_days(tokens: list[str] | None, cache: CatalogCache, event_id: str) -> frozenset[Any]:
    """Resolve ``--day`` tokens against the days the catalog actually has."""
    if not tokens:
        return frozenset()
    from datetime import date as date_type

    available = [date_type.fromisoformat(value) for value in cache.distinct(event_id, "day")]
    if not available:
        msg = "no cached days yet; run `replan sync` first"
        raise typer.BadParameter(msg)
    resolved = set()
    for token in tokens:
        try:
            resolved.add(resolve_day(token, available))
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    return frozenset(resolved)


@app.command()
def sync(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
    locale: LocaleOption = None,
    include_abstracts: Annotated[
        bool,
        typer.Option(
            "--abstracts/--no-abstracts",
            help="Fetch abstracts. Skipping them makes the walk much lighter.",
        ),
    ] = True,
    show_changes: Annotated[
        int, typer.Option("--show", help="How many individual changes to list.")
    ] = 12,
) -> None:
    """Walk the live catalog and cache it locally, reporting what changed.

    The API publishes no change feed, no ETag and no updated-at, so the only way
    to know a session moved room is to have kept the previous sync. That is what
    this does, and why the diff is worth printing.
    """
    settings = load_settings(event_id=event, locale=locale)
    port = build_port(settings, surface)
    cache = CatalogCache(settings.cache_path)

    def progress(pages: int, sessions: int, total: int | None) -> None:
        suffix = f"/{total}" if total else ""
        console.print(f"[dim]  page {pages}: {sessions}{suffix} sessions[/dim]")

    try:
        catalog = port.list_sessions(
            settings.event_id,
            include_abstracts=include_abstracts,
            locale=settings.locale,
            progress=progress,
        )
    except EventsApiError as error:
        die(console, error)
        return

    if not catalog.is_complete:
        console.print(
            f"[yellow]Warning: fetched {len(catalog)} of {catalog.total_count} "
            "sessions. Refusing to trust an incomplete walk for planning.[/yellow]"
        )

    diff = cache.save(catalog)
    print_provenance(console, port, settings)
    console.print(
        f"Cached [bold]{len(catalog)}[/bold] sessions to {cache.path}  "
        f"[dim]({'FTS5' if cache.supports_full_text else 'LIKE fallback'})[/dim]"
    )
    console.print(f"[bold]{diff.summary()}[/bold]")

    if diff.changes:
        table = Table(show_header=True, header_style="bold")
        table.add_column("what")
        table.add_column("code")
        table.add_column("session", overflow="fold")
        table.add_column("detail", overflow="fold")
        for change in diff.changes[:show_changes]:
            colour = "red" if change.kind.breaks_a_plan else "dim"
            table.add_row(
                f"[{colour}]{change.kind.value}[/{colour}]",
                change.code,
                change.title,
                "; ".join(item.describe() for item in change.fields) or change.kind.label,
            )
        console.print(table)
        if len(diff.changes) > show_changes:
            console.print(
                f"[dim]+{len(diff.changes) - show_changes} more; see `replan changes`[/dim]"
            )

        breaking = diff.plan_breaking
        if breaking:
            console.print(
                f"[yellow]{len(breaking)} change(s) could invalidate an existing plan "
                "(time, room, added or removed). Re-run `replan optimize`.[/yellow]"
            )

    if hasattr(port, "close"):
        port.close()


@app.command()
def search(
    query: Annotated[str, typer.Argument(help='Words, or "an exact phrase".')] = "",
    event: EventOption = None,
    level: Annotated[
        list[int] | None, typer.Option("--level", "-l", help="100, 200, 300, 400. Repeatable.")
    ] = None,
    day: Annotated[
        list[str] | None,
        typer.Option("--day", "-d", help="tue, 2026-12-01, or day2. Repeatable."),
    ] = None,
    session_format: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help="workshop, chalk-talk, breakout... Repeatable."),
    ] = None,
    venue: Annotated[
        list[str] | None, typer.Option("--venue", help="venetian, wynn, mgm-grand... Repeatable.")
    ] = None,
    track: Annotated[list[str] | None, typer.Option("--track", help="Exact track name.")] = None,
    service: Annotated[
        list[str] | None, typer.Option("--service", help="e.g. 'AWS Lambda'.")
    ] = None,
    bookable: Annotated[
        bool, typer.Option("--bookable", help="Only sessions with a seat you could take now.")
    ] = False,
    scheduled: Annotated[
        bool, typer.Option("--scheduled", help="Exclude all-day and unscheduled entries.")
    ] = False,
    unique: Annotated[
        bool, typer.Option("--unique", help="Collapse repeat showings to the first one.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Results to show.")] = 20,
    show_abstract: Annotated[
        bool, typer.Option("--abstract", help="Include a slice of each abstract.")
    ] = False,
) -> None:
    """Search the cached catalog offline.

    The API deliberately does not search or filter, so this is a local index over
    everything a session carries: title, abstract, speakers, tracks, topics,
    services, roles and industries.
    """
    settings = load_settings(event_id=event)
    cache = CatalogCache(settings.cache_path)
    info = cache.info(settings.event_id)
    if info is None:
        console.print(
            f"[yellow]Nothing cached for {settings.event_id}.[/yellow] "
            "Run [bold]replan sync[/bold] first."
        )
        raise typer.Exit(1)

    filters = SearchFilters(
        levels=frozenset(Level.parse(item) for item in (level or [])),
        formats=frozenset(SessionFormat.parse(item) for item in (session_format or [])),
        days=_parse_days(day, cache, settings.event_id),
        venues=frozenset(
            found for item in (venue or []) if (found := KnownVenue.parse(item)) is not None
        ),
        tracks=frozenset(track or []),
        services=frozenset(service or []),
        bookable_only=bookable,
        scheduled_only=scheduled,
        exclude_repeats=unique,
    )

    hits = cache.search(settings.event_id, query, filters=filters, limit=limit)

    console.print(
        f"[dim]cache[/dim] {info.session_count} sessions, synced {info.age_human}  "
        f"[dim]event[/dim] {settings.event_id}"
    )
    if not hits:
        console.print("[yellow]No matches.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("code")
    table.add_column("when")
    table.add_column("lvl", justify="right")
    table.add_column("format")
    table.add_column("where", overflow="fold")
    table.add_column("seats")
    table.add_column("title", overflow="fold")

    for hit in hits:
        item = hit.session
        seats = "-"
        if item.seat_availability:
            colour = {
                SeatAvailability.AVAILABLE: "green",
                SeatAvailability.LIMITED: "yellow",
                SeatAvailability.VERY_LIMITED: "red",
                SeatAvailability.UNAVAILABLE: "red",
                SeatAvailability.WALK_UP: "cyan",
            }[item.seat_availability]
            seats = f"[{colour}]{item.seat_availability.value}[/{colour}]"
        table.add_row(
            item.code,
            item.span.format_in(item.zone) if item.span else "-",
            str(int(item.level)) if item.level else "-",
            item.format.label,
            item.room.display,
            seats,
            item.title,
        )
    console.print(table)

    if show_abstract:
        for hit in hits[:5]:
            if hit.session.abstract:
                console.print(
                    Panel(
                        hit.session.abstract[:400]
                        + ("..." if len(hit.session.abstract) > 400 else ""),
                        title=f"{hit.session.code} {hit.session.title}",
                        border_style="dim",
                    )
                )

    console.print(f"[dim]{len(hits)} result(s)[/dim]")


@app.command()
def changes(
    event: EventOption = None,
    syncs: Annotated[int, typer.Option("--syncs", help="How many recent syncs to cover.")] = 1,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show.")] = 40,
    breaking_only: Annotated[
        bool,
        typer.Option("--breaking", help="Only changes that can invalidate a plan."),
    ] = False,
) -> None:
    """Show what moved in the catalog, from the local sync history."""
    settings = load_settings(event_id=event)
    cache = CatalogCache(settings.cache_path)

    history = cache.sync_history(settings.event_id, limit=syncs)
    if not history:
        console.print(
            f"[yellow]No sync history for {settings.event_id}.[/yellow] "
            "Run [bold]replan sync[/bold] at least twice to see changes."
        )
        raise typer.Exit(1)

    console.print("[bold]recent syncs[/bold]")
    for when, summary in history:
        console.print(f"  [dim]{when[:19]}[/dim]  {summary}")

    rows = cache.recent_changes(settings.event_id, syncs=syncs, limit=limit)
    if breaking_only:
        rows = tuple(row for row in rows if ChangeKind(row[0]).breaks_a_plan)

    if not rows:
        console.print("\n[green]No catalog changes recorded in that window.[/green]")
        return

    table = Table(show_header=True, header_style="bold", title="catalog changes")
    table.add_column("what")
    table.add_column("code")
    table.add_column("session", overflow="fold")
    table.add_column("detail", overflow="fold")
    for kind, code, title, _session_id, details in rows:
        parsed = ChangeKind(kind)
        colour = "red" if parsed.breaks_a_plan else "dim"
        table.add_row(
            f"[{colour}]{kind}[/{colour}]",
            code or "-",
            title or "-",
            "; ".join(details) or parsed.label,
        )
    console.print(table)


@app.command("cache")
def cache_command(
    event: EventOption = None,
    clear: Annotated[bool, typer.Option("--clear", help="Delete cached sessions.")] = False,
    all_events: Annotated[
        bool, typer.Option("--all", help="With --clear, wipe every event.")
    ] = False,
) -> None:
    """Inspect or clear the local catalog cache."""
    settings = load_settings(event_id=event)
    store = CatalogCache(settings.cache_path)

    if clear:
        removed = store.clear(None if all_events else settings.event_id)
        scope = "all events" if all_events else settings.event_id
        console.print(f"[green]Cleared {removed} cached session(s) for {scope}.[/green]")
        return

    cached = store.cached_events()
    if not cached:
        console.print(f"Cache is empty. [dim]{store.path}[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("event")
    table.add_column("sessions", justify="right")
    table.add_column("complete")
    table.add_column("synced")
    table.add_column("source")
    table.add_column("last change summary", overflow="fold")

    for name in cached:
        info = store.info(name)
        if info is None:
            continue
        table.add_row(
            name,
            str(info.session_count),
            "[green]yes[/green]" if info.is_complete else "[red]no[/red]",
            info.age_human,
            info.source or "-",
            info.last_sync_summary or "-",
        )
    console.print(table)
    console.print(f"[dim]{store.path}[/dim]")


# --------------------------------------------------------------------------- #
# travel
# --------------------------------------------------------------------------- #
@app.command()
def travel(
    origin: Annotated[
        str | None,
        typer.Argument(help='Room, e.g. "Venetian | Level 2 | Hall C". Or a session code.'),
    ] = None,
    destination: Annotated[
        str | None, typer.Argument(help="Room or session code to travel to.")
    ] = None,
    event: EventOption = None,
    matrix: Annotated[
        bool, typer.Option("--matrix", help="Print every curated route instead.")
    ] = False,
    overrides: Annotated[
        Path | None,
        typer.Option("--travel-overrides", help="YAML file of your own travel times."),
    ] = None,
) -> None:
    """How long does it take to get from one room to another?

    Accepts room strings exactly as replan prints them, or session codes from the
    cached catalog - in which case it also answers whether the connection is
    actually makeable given the two session times.

    The Events API publishes no distances at all, so these are estimates from the
    campus layout. Override them with --travel-overrides if you know better.
    """
    graph = default_graph(overrides)

    if matrix:
        table = Table(show_header=True, header_style="bold", title="curated routes")
        table.add_column("from")
        table.add_column("to")
        table.add_column("minutes", justify="right")
        table.add_column("mode")
        for left, right, minutes, mode in graph.matrix():
            table.add_row(left, right, str(minutes), mode)
        console.print(table)
        console.print(f"[dim]{graph.describe_source()}[/dim]")
        console.print(
            "[dim]Door to door, including getting out of one room and into the "
            "next. Estimates, not measurements.[/dim]"
        )
        return

    if not origin or not destination:
        console.print(
            "[red]Give two rooms or two session codes, or use --matrix.[/red]\n"
            '[dim]e.g. replan travel "Venetian | Level 2 | Hall C" '
            '"Mandalay Bay | South Pacific F"[/dim]'
        )
        raise typer.Exit(2)

    settings = load_settings(event_id=event)
    store = CatalogCache(settings.cache_path)
    cached = store.load(settings.event_id)

    first = cached.resolve(origin) if cached else None
    second = cached.resolve(destination) if cached else None

    if first is not None and second is not None:
        # Both are real sessions: answer the question that actually matters.
        result = graph.feasibility(first, second)
        leg = result.leg
        body = Table.grid(padding=(0, 2))
        body.add_row("from", f"{first.code}  {first.room.display}")
        body.add_row(
            "",
            f"[dim]{first.span.format_in(first.zone) if first.span else 'unscheduled'}"
            f"  {first.title}[/dim]",
        )
        body.add_row("to", f"{second.code}  {second.room.display}")
        body.add_row(
            "",
            f"[dim]{second.span.format_in(second.zone) if second.span else 'unscheduled'}"
            f"  {second.title}[/dim]",
        )
        body.add_row("transfer", f"{leg.minutes} min by {leg.mode.label}")
        body.add_row(
            "breakdown", f"{leg.transit_minutes} transit + {leg.overhead_minutes} door-to-door"
        )
        body.add_row("gap", f"{result.gap_minutes} min")
        body.add_row("spare", f"{result.slack_minutes} min")
        body.add_row("confidence", leg.confidence.label)

        depart = graph.depart_by(first, second)
        if depart is not None:
            body.add_row("leave by", f"{depart.astimezone(first.zone):%a %H:%M}")

        verdict, colour = (
            ("makeable", "green")
            if result.ok and not result.is_tight
            else ("tight", "yellow")
            if result.ok
            else ("not possible", "red")
        )
        # The reason goes in the body, not the title: panel titles get truncated
        # to the terminal width and the reason is the most useful line here.
        body.add_row("verdict", f"[{colour}]{result.reason}[/{colour}]")
        console.print(
            Panel(
                body,
                title=f"[{colour}]{verdict}[/{colour}]: {first.code} -> {second.code}",
                border_style=colour,
            )
        )
        return

    # Fall back to treating the arguments as room strings.
    try:
        from_room = Room.parse(origin)
        to_room = Room.parse(destination)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(2) from error

    leg = graph.leg(from_room, to_room)
    body = Table.grid(padding=(0, 2))
    body.add_row("from", from_room.display)
    body.add_row("to", to_room.display)
    body.add_row("total", f"[bold]{leg.minutes} min[/bold]")
    body.add_row("mode", leg.mode.label)
    body.add_row("transit", f"{leg.transit_minutes} min")
    body.add_row("getting out and in", f"{leg.overhead_minutes} min")
    body.add_row("confidence", leg.confidence.label)
    if leg.detail:
        body.add_row("detail", leg.detail)
    console.print(Panel(body, title="travel estimate", border_style="cyan"))

    if first is None and second is None and cached is not None:
        console.print(
            "[dim]Tip: pass session codes instead of rooms to also check whether "
            "the connection is makeable in the time available.[/dim]"
        )


# --------------------------------------------------------------------------- #
# profile / rank
# --------------------------------------------------------------------------- #
profile_app = typer.Typer(help="Manage your preference profile.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")

ProfileOption = Annotated[
    Path | None,
    typer.Option("--profile", "-p", help="Path to your profile.yaml."),
]


def _require_cached_catalog(settings: Any, event_id: str | None = None) -> Any:
    """Load the cached catalog, or explain how to get one.

    ``event_id`` overrides the configured default, so a command driven by a plan
    file uses the event that plan was actually built for.
    """
    wanted = event_id or settings.event_id
    store = CatalogCache(settings.cache_path)
    cached = store.load(wanted)
    if cached is None or not len(cached):
        console.print(
            f"[yellow]Nothing cached for {wanted}.[/yellow] "
            f"Run [bold]replan sync --event {wanted}[/bold] first."
        )
        raise typer.Exit(1)
    return cached


@profile_app.command("init")
def profile_init(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the profile.")
    ] = Path("profile.yaml"),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a commented starter profile you can edit."""
    try:
        written = write_starter_profile(output, force=force)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    console.print(f"[green]Wrote {written}[/green]")
    console.print(
        "Edit the topics, levels and weights, then run "
        "[bold]replan rank --profile " + str(written) + "[/bold]"
    )


@profile_app.command("show")
def profile_show(
    profile_path: ProfileOption = None,
    event: EventOption = None,
) -> None:
    """Summarise a profile, and check it against the cached catalog.

    Validating pins here means a typo in ``must_attend`` is caught before it
    quietly disappears from a plan.
    """
    settings = load_settings(event_id=event)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    body = Table.grid(padding=(0, 2))
    body.add_row("name", current.name)
    if current.goals:
        body.add_row("goals", "\n".join(f"- {item}" for item in current.goals))
    if current.topics:
        ranked = sorted(current.topics.items(), key=lambda item: -item[1])
        body.add_row("topics", ", ".join(f"{key} ({weight:g})" for key, weight in ranked[:10]))
    if current.avoid_topics:
        body.add_row("avoiding", ", ".join(current.avoid_topics))
    if current.levels:
        body.add_row(
            "levels",
            ", ".join(f"{key}:{weight:g}" for key, weight in sorted(current.levels.items())),
        )
    if current.formats:
        body.add_row("formats", ", ".join(sorted(current.formats)))
    if current.speakers:
        body.add_row("speakers", ", ".join(current.speakers))
    body.add_row("day window", f"{current.day_start or '-'} to {current.day_end or '-'}")
    body.add_row("max sessions/day", str(current.max_sessions_per_day or "unlimited"))
    body.add_row("walk tolerance", f"{current.walk_tolerance_minutes} min")
    body.add_row("meal windows", str(len(current.meal_windows)))
    weights = current.weights
    body.add_row(
        "weights",
        f"utility {weights.utility:g}, coverage {weights.coverage:g}, "
        f"travel {weights.travel_penalty:g}, fatigue {weights.fatigue_penalty:g}, "
        f"hops {weights.venue_hop_penalty:g}",
    )
    console.print(Panel(body, title="profile", border_style="cyan"))

    store = CatalogCache(settings.cache_path)
    cached = store.load(settings.event_id)
    if cached is None:
        console.print("[dim]Sync a catalog to validate must_attend / never_attend.[/dim]")
        return

    for label, tokens in (
        ("must_attend", current.must_attend),
        ("never_attend", current.never_attend),
    ):
        if not tokens:
            continue
        missing = current.unresolved(tokens, cached)
        resolved = len(tokens) - len(missing)
        console.print(f"{label}: [green]{resolved} resolved[/green]", end="")
        if missing:
            console.print(f"  [red]{len(missing)} not found: {', '.join(missing)}[/red]")
        else:
            console.print()


@app.command()
def rank(
    profile_path: ProfileOption = None,
    event: EventOption = None,
    top: Annotated[int, typer.Option("--top", "-n", help="How many to show.")] = 20,
    explain: Annotated[
        bool, typer.Option("--explain", help="Show every scoring factor per session.")
    ] = False,
    show_vetoed: Annotated[
        bool, typer.Option("--vetoed", help="List excluded sessions and why instead.")
    ] = False,
) -> None:
    """Rank the cached catalog against your profile.

    Scores are additive and every contribution is recorded, so `--explain` shows
    exactly why a session placed where it did.
    """
    settings = load_settings(event_id=event)
    cached = _require_cached_catalog(settings)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    scorer = Scorer(current, cached)

    if show_vetoed:
        vetoed = [item for item in scorer.score_all(include_vetoed=True) if not item.eligible]
        table = Table(show_header=True, header_style="bold", title="excluded sessions")
        table.add_column("code")
        table.add_column("reason")
        table.add_column("detail", overflow="fold")
        for item in vetoed[:top]:
            table.add_row(item.code, item.veto.value if item.veto else "-", item.veto_detail)
        console.print(table)
        console.print(f"[dim]{len(vetoed)} excluded of {len(cached)} total[/dim]")
        return

    scores = scorer.ranked(top)
    if not scores:
        console.print("[yellow]No eligible sessions. Loosen the profile?[/yellow]")
        return

    lookup = {session.session_id: session for session in cached.sessions}
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("code")
    table.add_column("when")
    table.add_column("title", overflow="fold")
    table.add_column("why", overflow="fold")

    for index, item in enumerate(scores, start=1):
        session = lookup[item.session_id]
        marker = " [magenta]pin[/magenta]" if item.pinned else ""
        table.add_row(
            str(index),
            f"{item.total:.0f}{marker}",
            item.code,
            session.span.format_in(session.zone) if session.span else "-",
            session.title,
            "; ".join(item.top_reasons(2)),
        )
    console.print(table)

    eligible = len(scorer.score_all())
    console.print(
        f"[dim]{eligible} eligible of {len(cached)} cached. "
        f"Use --vetoed to see what was excluded.[/dim]"
    )

    if explain:
        for item in scores[:5]:
            session = lookup[item.session_id]
            detail = Table.grid(padding=(0, 2))
            for factor in sorted(item.factors, key=lambda entry: -abs(entry.points)):
                colour = "green" if factor.points >= 0 else "red"
                detail.add_row(
                    f"[{colour}]{factor.points:+.1f}[/{colour}]", factor.name, factor.detail
                )
            detail.add_row("[bold]=[/bold]", "[bold]total[/bold]", f"[bold]{item.total:.1f}[/bold]")
            console.print(Panel(detail, title=f"{item.code}  {session.title}", border_style="dim"))


# --------------------------------------------------------------------------- #
# optimize
# --------------------------------------------------------------------------- #
@app.command()
def optimize(
    profile_path: ProfileOption = None,
    event: EventOption = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the plan JSON here.")
    ] = None,
    day: Annotated[
        list[str] | None, typer.Option("--day", "-d", help="Only solve these days.")
    ] = None,
    surface: SurfaceOption = SurfaceChoice.AUTO,
    use_schedule: Annotated[
        bool,
        typer.Option(
            "--schedule/--no-schedule",
            help="Fetch your real schedule so held seats are pinned and personal time respected.",
        ),
    ] = False,
    candidates_per_day: Annotated[
        int, typer.Option("--candidates", help="Candidate pool size per day.")
    ] = 45,
    time_limit: Annotated[
        float, typer.Option("--time-limit", help="Solver time budget in seconds.")
    ] = 10.0,
    greedy: Annotated[
        bool, typer.Option("--greedy", help="Force the greedy fallback, for comparison.")
    ] = False,
    overrides: Annotated[
        Path | None, typer.Option("--travel-overrides", help="Your own travel times.")
    ] = None,
) -> None:
    """Solve for a conflict-free, travel-feasible agenda.

    Hard constraints: no overlaps, every transfer physically makeable within your
    walk tolerance, one showing per piece of content, protected time honoured, and
    your daily session cap. Soft objective: session value, breadth of topic
    coverage, and penalties for travel, venue hopping, idle gaps and fatigue.
    """
    settings = load_settings(event_id=event)
    cached = _require_cached_catalog(settings)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    store = CatalogCache(settings.cache_path)
    days = _parse_days(day, store, settings.event_id) or None

    attendee = None
    if use_schedule:
        port = build_port(settings, surface)
        try:
            attendee = port.get_schedule(settings.event_id)
            console.print(
                f"[dim]your schedule: {len(attendee.reserved)} reserved, "
                f"{len(attendee.personal_time)} personal time block(s)[/dim]"
            )
        except EventsApiError as error:
            console.print(f"[yellow]Could not read your schedule: {error}[/yellow]")
            console.print(f"[yellow]{error.remedy}[/yellow]")
        finally:
            if hasattr(port, "close"):
                port.close()

    scorer = Scorer(current, cached)
    request = SolveRequest(
        catalog=cached,
        profile=current,
        scores=scorer.by_id(),
        travel=default_graph(overrides),
        schedule=attendee,
        days=days,
        candidates_per_day=candidates_per_day,
        time_limit_seconds=time_limit,
    )

    plan = solve(request, prefer_cpsat=not greedy)
    _render_plan(plan)

    if out is not None:
        plan.write(out)
        console.print(f"\n[green]Wrote {out}[/green]")
        console.print(
            f"[dim]Next: replan why --plan {out} --worst   |   replan export --plan {out}[/dim]"
        )


def _render_plan(plan: Any) -> None:
    """Print a solved plan: the day-by-day agenda, then how it was scored."""
    stats = plan.stats
    quality = (
        "[green]proven optimal[/green]" if stats.optimal else f"[yellow]{stats.status}[/yellow]"
    )
    console.print(
        f"[dim]solver[/dim] {stats.solver.label}  {quality}  "
        f"[dim]{stats.wall_seconds:.2f}s, {stats.candidates} candidates[/dim]"
    )
    if stats.fallback_reason:
        console.print(f"[yellow]{stats.fallback_reason}[/yellow]")
    for warning in plan.warnings:
        console.print(f"[yellow]warning: {warning}[/yellow]")

    if not plan.items:
        console.print("[yellow]No sessions could be scheduled. Loosen the profile?[/yellow]")
        return

    for current_day, items in plan.by_day():
        table = Table(
            show_header=True,
            header_style="bold",
            title=f"{plan.day_label_for(current_day)} {current_day.isoformat()}",
            title_justify="left",
        )
        table.add_column("when")
        table.add_column("code")
        table.add_column("session", overflow="fold")
        table.add_column("where", overflow="fold")
        table.add_column("seat")
        table.add_column("getting there", overflow="fold")

        for item in items:
            move = "-"
            if item.travel_minutes == 0:
                # "leave 14:50" would be nonsense for a zero-minute transfer.
                move = "[dim]same room[/dim]"
            elif item.travel_minutes is not None:
                colour = "red" if item.is_tight_connection else "dim"
                leave = (
                    f", leave {item.depart_previous_by.astimezone(item.zone):%H:%M}"
                    if item.depart_previous_by
                    else ""
                )
                move = f"[{colour}]{item.travel_minutes} min {item.travel_mode}{leave}[/{colour}]"
            seat = "-"
            if item.already_reserved:
                seat = "[green]held[/green]"
            elif item.needs_reservation:
                seat = f"[yellow]reserve[/yellow] {item.seat_band or ''}".strip()
            elif item.seat_band:
                seat = item.seat_band

            marker = " [magenta]pin[/magenta]" if item.pinned else ""
            table.add_row(
                item.when,
                item.code + marker,
                item.title,
                item.room_display,
                seat,
                move,
            )
        console.print(table)

    summary = Table.grid(padding=(0, 2))
    summary.add_row("sessions", str(len(plan.items)))
    summary.add_row("session time", f"{plan.total_minutes // 60}h {plan.total_minutes % 60}m")
    summary.add_row("time in transit", f"{plan.total_travel_minutes} min")
    summary.add_row(
        "topic coverage",
        f"{stats.topics_covered}/{stats.topics_available} ({stats.coverage_ratio:.0%})",
    )
    seats_needed = len(plan.needs_reservation)
    if seats_needed:
        summary.add_row("seats to reserve", str(seats_needed))
    tight = len(plan.tight_connections)
    if tight:
        summary.add_row("tight connections", f"[yellow]{tight}[/yellow]")
    console.print(Panel(summary, title="plan summary", border_style="cyan"))

    objective = Table(show_header=False, box=None, title="objective", title_justify="left")
    objective.add_column(style="cyan")
    objective.add_column(justify="right")
    for label, value in plan.objective.rows():
        style = "bold" if label == "total" else ""
        objective.add_row(f"[{style}]{label}[/{style}]" if style else label, f"{value:+.1f}")
    console.print(objective)


# --------------------------------------------------------------------------- #
# why / export
# --------------------------------------------------------------------------- #
PlanOption = Annotated[Path, typer.Option("--plan", help="Plan JSON written by `replan optimize`.")]


def _load_plan(path: Path) -> Any:
    try:
        return Plan.read(path)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


def _render_explanation(explanation: Any) -> None:
    colour = {
        Verdict.SCHEDULED: "green",
        Verdict.REJECTED: "yellow",
        Verdict.UNKNOWN_SESSION: "red",
    }[explanation.verdict]

    body = Table.grid(padding=(0, 2))
    body.add_row("verdict", f"[{colour}]{explanation.headline}[/{colour}]")
    for label, value in explanation.facts:
        body.add_row(label, value)
    if explanation.conflicts_with:
        body.add_row("blocked by", "\n".join(explanation.conflicts_with))
    if explanation.alternatives:
        body.add_row("other showings", "\n".join(explanation.alternatives))
    if explanation.score_lines:
        body.add_row("scoring", "\n".join(explanation.score_lines))
    if explanation.remedy:
        body.add_row("to change this", f"[cyan]{explanation.remedy}[/cyan]")

    title = f"{explanation.code}"
    if explanation.title:
        title += f"  {explanation.title}"
    console.print(Panel(body, title=title, border_style=colour))


@app.command()
def why(
    session: Annotated[str | None, typer.Argument(help="Session code or id to explain.")] = None,
    plan_path: PlanOption = Path("plan.json"),
    profile_path: ProfileOption = None,
    event: EventOption = None,
    worst: Annotated[
        bool,
        typer.Option("--worst", help="Explain the best sessions that did not make it."),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many, with --worst.")] = 3,
    overrides: Annotated[
        Path | None, typer.Option("--travel-overrides", help="Your own travel times.")
    ] = None,
) -> None:
    """Explain why a session is, or is not, in your plan.

    Every answer says what to change if you disagree. That is the point: a planner
    that silently drops the session you were looking forward to is one you stop
    trusting.
    """
    settings = load_settings(event_id=event)
    plan = _load_plan(plan_path)
    cached = _require_cached_catalog(settings, plan.event_id)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    graph = default_graph(overrides)
    scorer = Scorer(current, cached)

    if worst or session is None:
        explanations = explain_worst(plan, cached, current, graph, limit, scorer)
        if not explanations:
            console.print("[green]Nothing was left out.[/green]")
            return
        console.print(
            f"[bold]The {len(explanations)} highest-scoring sessions that did not "
            "make your plan[/bold]\n"
        )
        for explanation in explanations:
            _render_explanation(explanation)
        return

    _render_explanation(explain_session(session, plan, cached, current, graph, scorer))


@app.command()
def export(
    plan_path: PlanOption = Path("plan.json"),
    event: EventOption = None,
    ics: Annotated[
        Path | None, typer.Option("--ics", help="Write an RFC 5545 calendar here.")
    ] = None,
    markdown: Annotated[
        Path | None, typer.Option("--markdown", help="Write a Markdown agenda here.")
    ] = None,
    html_dir: Annotated[
        Path | None,
        typer.Option("--html-dir", help="Write printable day sheets into this directory."),
    ] = None,
    departure_alarms: Annotated[
        bool,
        typer.Option(
            "--departure-alarms/--start-alarms",
            help="Fire calendar alarms when you must leave, not when the session starts.",
        ),
    ] = True,
) -> None:
    """Export the plan as a calendar, Markdown, or printable day sheets.

    The calendar's alarms default to *departure* time. Being told at 14:00 that a
    14:00 session is starting in another property is information you can no longer
    act on.
    """
    settings = load_settings(event_id=event)
    plan = _load_plan(plan_path)

    store = CatalogCache(settings.cache_path)
    cached = store.load(plan.event_id) or store.load(settings.event_id)

    if ics is None and markdown is None and html_dir is None:
        # Sensible defaults beside the plan, so the command is useful bare.
        base = plan_path.parent
        ics = base / f"{plan.event_id}.ics"
        markdown = base / f"{plan.event_id}-agenda.md"
        html_dir = base / "daysheets"

    if ics is not None:
        written = export_ics(plan, ics, catalog=cached, departure_alarms=departure_alarms)
        alarm_note = "alarms at departure time" if departure_alarms else "alarms at session start"
        console.print(f"[green]Calendar[/green] {written}  [dim]({alarm_note})[/dim]")

    if markdown is not None:
        written = export_markdown(plan, markdown, catalog=cached)
        console.print(f"[green]Markdown[/green] {written}")

    if html_dir is not None:
        pages = export_day_sheets_html(plan, html_dir)
        console.print(
            f"[green]Day sheets[/green] {len(pages)} file(s) in {html_dir}"
            f"  [dim](open {html_dir / 'index.html'})[/dim]"
        )

    if cached is None:
        console.print(
            "[yellow]No cached catalog for this plan, so abstracts and speakers were "
            "omitted from the calendar. Run `replan sync` and export again.[/yellow]"
        )


# --------------------------------------------------------------------------- #
# apply / undo / history
# --------------------------------------------------------------------------- #
@app.command()
def apply(
    plan_path: PlanOption = Path("plan.json"),
    profile_path: ProfileOption = None,
    event: EventOption = None,
    surface: SurfaceOption = SurfaceChoice.AUTO,
    do_apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Actually write to your schedule. Without this, nothing is sent.",
        ),
    ] = False,
    reserve: Annotated[
        bool, typer.Option("--reserve/--no-reserve", help="Take seats for planned sessions.")
    ] = True,
    favorites: Annotated[
        bool, typer.Option("--favorites/--no-favorites", help="Mark planned sessions as favorites.")
    ] = True,
    personal_time: Annotated[
        bool,
        typer.Option(
            "--personal-time/--no-personal-time",
            help="Create the protected time windows from your profile.",
        ),
    ] = True,
    prune_favorites: Annotated[
        bool,
        typer.Option("--prune-favorites", help="Remove favorites that are not in the plan."),
    ] = False,
    prune_reservations: Annotated[
        bool,
        typer.Option("--prune-reservations", help="Cancel reservations not in the plan."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Allow touching personal time replan did not create."),
    ] = False,
) -> None:
    """Write the plan to your real schedule. Dry run unless you pass --apply.

    Every run diffs against GetSchedule first and verifies against it afterwards,
    because the API has no idempotency key: if a write's outcome is unknown, the only
    safe move is to read the truth rather than send it again.
    """
    settings = load_settings(event_id=event)
    plan = _load_plan(plan_path)
    cached = _require_cached_catalog(settings, plan.event_id)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    port = build_port(settings, surface)
    label = getattr(port, "surface_name", "unknown")
    style = "green" if label.startswith("live") else "yellow"
    # The plan's own event, not the configured default: a plan built for one event
    # must never be applied to another without saying so.
    console.print(
        f"[dim]source[/dim] [{style}]{label}[/{style}]  [dim]event[/dim] {plan.event_id}",
        highlight=False,
    )

    try:
        schedule = port.get_schedule(plan.event_id)
    except EventsApiError as error:
        die(console, error)
        return

    options = DiffOptions(
        favorite_plan_items=favorites,
        reserve_seats=reserve,
        create_personal_time=personal_time,
        prune_favorites=prune_favorites,
        prune_reservations=prune_reservations,
        force_foreign_blocks=force,
    )
    action_plan = build_action_plan(plan, schedule, cached, current, options)

    console.print(
        f"[dim]your schedule now:[/dim] {len(schedule.reserved)} reserved, "
        f"{len(schedule.favorites)} favorites, {len(schedule.personal_time)} blocks"
    )
    console.print(f"[bold]{action_plan.summary()}[/bold]")

    for note in action_plan.notes:
        console.print(f"[dim]note: {note}[/dim]")

    if action_plan.is_empty:
        return

    table = Table(show_header=True, header_style="bold", title="planned writes")
    table.add_column("#", justify="right")
    table.add_column("action")
    table.add_column("targets", overflow="fold")
    table.add_column("why", overflow="fold")
    # Spans are rendered in the event's own timezone. A Dubai lunch window shown in
    # Las Vegas time reads as 01:00 and looks like a bug.
    event_zone = cached.zone
    for index, action in enumerate(action_plan.ordered, start=1):
        colour = "red" if action.kind.is_destructive else "cyan"
        if action.labels:
            targets = ", ".join(action.labels)
        else:
            when = f" {action.span.format_in(event_zone)}" if action.span else ""
            targets = f"{action.title}{when}"
        table.add_row(
            str(index),
            f"[{colour}]{action.kind.label}[/{colour}]",
            targets,
            action.reason,
        )
    console.print(table)

    quota = action_plan.quota_cost()
    console.print(
        "[dim]quota cost: "
        + ", ".join(f"{operation} {units}" for operation, units in sorted(quota.items()))
        + ". ReserveSessions and AssociateFavorites bill per session, 30 per minute.[/dim]"
    )

    if action_plan.destructive and do_apply:
        console.print(
            f"[yellow]{len(action_plan.destructive)} action(s) will remove something "
            "you currently have.[/yellow]"
        )

    journal = Journal(settings.journal_path)
    report = apply_plan(
        port,
        plan,
        action_plan,
        dry_run=not do_apply,
        journal=journal,
        verify=do_apply,
    )

    if not do_apply:
        console.print(
            "\n[yellow]Dry run - nothing was sent.[/yellow] "
            "Re-run with [bold]--apply[/bold] to write these changes."
        )
        return

    console.print(f"\n[bold]{report.summary()}[/bold]")
    for outcome in report.outcomes:
        if outcome.status is ActionStatus.DONE:
            continue
        colour = {
            ActionStatus.ALREADY: "dim",
            ActionStatus.SKIPPED: "yellow",
            ActionStatus.FAILED: "red",
            ActionStatus.UNKNOWN: "red",
        }.get(outcome.status, "white")
        console.print(f"[{colour}]{outcome.describe()}[/{colour}]")

    for note in report.notes:
        console.print(f"[dim]note: {note}[/dim]")

    if report.unknown:
        console.print(
            "[red]Some writes have an unknown outcome. The API has no idempotency "
            "key, so do not re-run blindly: check `replan schedule` and apply again "
            "to send only what is still missing.[/red]"
        )

    if report.verified:
        console.print(f"[green]Verified against GetSchedule:[/green] {report.verification[0]}")
    else:
        for problem in report.verification:
            console.print(f"[yellow]verification: {problem}[/yellow]")

    if report.run_id is not None:
        console.print(
            f"\n[dim]Journalled as run {report.run_id}. "
            f"Undo with: replan undo --run {report.run_id}[/dim]"
        )


@app.command()
def undo(
    run: Annotated[
        int | None, typer.Option("--run", help="Run id to undo. Defaults to the most recent.")
    ] = None,
    event: EventOption = None,
    surface: SurfaceOption = SurfaceChoice.AUTO,
    do_apply: Annotated[bool, typer.Option("--apply", help="Actually reverse the writes.")] = False,
) -> None:
    """Reverse a previous `replan apply`.

    The API keeps no history of its own, so this replays replan's local journal
    backwards. Actions are reversed newest first, which keeps the intermediate states
    as coherent as the forward run's were.
    """
    settings = load_settings(event_id=event)
    journal = Journal(settings.journal_path)

    target = journal.run(run) if run is not None else journal.latest_run(settings.event_id)
    if target is None:
        console.print(
            "[yellow]No applied run found to undo.[/yellow] "
            "See `replan history` for what is recorded."
        )
        raise typer.Exit(1)

    if target.dry_run:
        console.print(
            f"[yellow]Run {target.run_id} was a dry run; there is nothing to undo.[/yellow]"
        )
        raise typer.Exit(1)

    if target.undone_at is not None:
        console.print(
            f"[yellow]Run {target.run_id} was already undone at "
            f"{target.undone_at:%Y-%m-%d %H:%M} UTC.[/yellow]"
        )

    reversible = target.reversible_entries
    console.print(
        f"Run [bold]{target.run_id}[/bold] on {target.event_id}, "
        f"started {target.started_at:%Y-%m-%d %H:%M} UTC. {target.summary or ''}"
    )
    if not reversible:
        console.print("[yellow]Nothing in that run can be reversed.[/yellow]")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold", title="would reverse")
    table.add_column("original")
    table.add_column("targets", overflow="fold")
    for entry in reversible:
        table.add_row(
            entry.action.kind.label, ", ".join(entry.action.labels or entry.succeeded_ids)
        )
    console.print(table)

    port = build_port(settings, surface)
    report = undo_run(journal, target, port, dry_run=not do_apply)

    if not do_apply:
        console.print(
            f"\n[yellow]Dry run - nothing was sent.[/yellow] {report.reversed_count} "
            "action(s) would be reversed. Re-run with [bold]--apply[/bold]."
        )
        return

    console.print(f"\n[bold]{report.summary()}[/bold]")
    for failure in report.failed:
        console.print(f"[red]{failure}[/red]")
    for skip in report.skipped:
        console.print(f"[dim]{skip}[/dim]")


@app.command()
def history(
    event: EventOption = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Runs to show.")] = 10,
) -> None:
    """Show the local history of write-backs."""
    settings = load_settings(event_id=event)
    journal = Journal(settings.journal_path)
    runs = journal.history(settings.event_id, limit=limit)

    if not runs:
        console.print(f"No write history yet. [dim]{journal.path}[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("run", justify="right")
    table.add_column("when")
    table.add_column("mode")
    table.add_column("surface")
    table.add_column("actions", justify="right")
    table.add_column("result", overflow="fold")

    for item in runs:
        mode = "[dim]dry run[/dim]" if item.dry_run else "[green]applied[/green]"
        if item.undone_at is not None:
            mode += " [yellow](undone)[/yellow]"
        table.add_row(
            str(item.run_id),
            f"{item.started_at:%Y-%m-%d %H:%M}",
            mode,
            item.surface or "-",
            str(len(item.entries)),
            item.summary or "-",
        )
    console.print(table)
    console.print(f"[dim]{journal.path}[/dim]")


# --------------------------------------------------------------------------- #
# now
# --------------------------------------------------------------------------- #
@app.command()
def now(
    profile_path: ProfileOption = None,
    event: EventOption = None,
    at: Annotated[
        str,
        typer.Option("--at", help="'now', '14:10', 'wed 14:10', or an ISO timestamp."),
    ] = "now",
    location: Annotated[
        str | None,
        typer.Option(
            "--location",
            "-l",
            help="Where you are: a room string, or the code of the session you are in.",
        ),
    ] = None,
    plan_path: Annotated[
        Path | None,
        typer.Option("--plan", help="Existing plan, used to detect what you are in now."),
    ] = None,
    surface: SurfaceOption = SurfaceChoice.AUTO,
    use_schedule: Annotated[
        bool,
        typer.Option(
            "--schedule/--no-schedule",
            help="Read your real schedule so held seats are kept.",
        ),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many moves to show.")] = 3,
    overrides: Annotated[
        Path | None, typer.Option("--travel-overrides", help="Your own travel times.")
    ] = None,
) -> None:
    """What should I do next, from where I am standing?

    Re-solves only the rest of today, only sessions you could physically reach from
    your current position, and reports departure times rather than durations. "Leave
    at 14:06" is actionable under time pressure; "it is a 24 minute walk" is arithmetic.
    """
    settings = load_settings(event_id=event)
    cached = _require_cached_catalog(settings)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    existing = _load_plan(plan_path) if plan_path is not None else None

    attendee = None
    if use_schedule:
        port = build_port(settings, surface)
        try:
            attendee = port.get_schedule(settings.event_id)
        except EventsApiError as error:
            console.print(f"[yellow]Could not read your schedule: {error}[/yellow]")
        finally:
            if hasattr(port, "close"):
                port.close()

    try:
        report = replan_now(
            cached,
            current,
            default_graph(overrides),
            at=at,
            location=location,
            schedule=attendee,
            existing_plan=existing,
            limit=limit,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(2) from error

    header = f"[bold]{report.moment.clock}[/bold]"
    if report.location is not None:
        header += f"  you are at [cyan]{report.location.display}[/cyan]"
    console.print(header)

    for note in report.notes:
        console.print(f"[dim]{note}[/dim]")

    if report.in_progress is not None:
        item = report.in_progress
        console.print(
            f"\n[yellow]now:[/yellow] {item.code} - {item.title}\n"
            f"      {item.room_display}, until "
            f"{item.span.end.astimezone(report.moment.zone):%H:%M}"
        )

    if report.is_empty:
        console.print("\n[yellow]Nothing further to plan today.[/yellow]")
        if report.unreachable:
            table = Table(
                show_header=True,
                header_style="bold",
                title="today's remaining sessions, and what they would take",
            )
            table.add_column("code")
            table.add_column("needs", justify="right")
            table.add_column("session", overflow="fold")
            for session, minutes in report.unreachable:
                table.add_row(session.code, f"{minutes} min", session.title)
            console.print(table)
        return

    console.print()
    table = Table(show_header=True, header_style="bold", title="next moves")
    table.add_column("when")
    table.add_column("action")
    table.add_column("code")
    table.add_column("session", overflow="fold")
    table.add_column("where", overflow="fold")

    for move in report.moves:
        colour = {
            Urgency.LEAVE_NOW: "red",
            Urgency.LEAVE_SOON: "yellow",
            Urgency.COMFORTABLE: "green",
            Urgency.STAY_PUT: "cyan",
            Urgency.AFTER_CURRENT: "cyan",
        }[move.urgency]

        if move.urgency.needs_movement and move.travel_minutes and move.depart_clock:
            action = (
                f"[{colour}]{move.urgency.label}[/{colour}] {move.depart_clock}"
                f" ({move.travel_minutes} min)"
            )
        else:
            action = f"[{colour}]{move.urgency.label}[/{colour}]"

        seat = ""
        if move.item.already_reserved:
            seat = " [green]seat held[/green]"
        elif move.item.needs_reservation:
            seat = " [yellow]needs a seat[/yellow]"

        table.add_row(
            f"{move.start_clock}-{move.end_clock}",
            action,
            move.item.code,
            move.item.title + seat,
            move.item.room_display,
        )
    console.print(table)

    leading = report.next_move
    if leading is not None and leading.urgency is Urgency.LEAVE_NOW:
        console.print(
            f"\n[red bold]Leave now[/red bold] for {leading.item.code} in "
            f"{leading.item.room_display} ({leading.travel_minutes} min)."
        )

    if report.unreachable:
        console.print(
            f"\n[dim]{len(report.unreachable)} session(s) were ruled out only because of "
            "where you are. Move first and ask again, or raise walk_tolerance_minutes.[/dim]"
        )


# --------------------------------------------------------------------------- #
# watch
# --------------------------------------------------------------------------- #
watch_app = typer.Typer(help="Watch full sessions for a seat opening up.", no_args_is_help=True)
app.add_typer(watch_app, name="watch")


def _watch_store(settings: Any) -> WatchStore:
    return WatchStore(settings.state_dir / "watch.db")


@watch_app.command("add")
def watch_add(
    sessions: Annotated[
        list[str] | None, typer.Argument(help="Session codes or ids to watch.")
    ] = None,
    plan_path: Annotated[
        Path | None,
        typer.Option("--plan", help="Watch every session in this plan that needs a seat."),
    ] = None,
    event: EventOption = None,
    profile_path: ProfileOption = None,
) -> None:
    """Add sessions to the watch list.

    The natural source is a plan: the sessions worth watching are exactly the ones the
    optimiser wanted and could not book.
    """
    settings = load_settings(event_id=event)
    event_id = settings.event_id
    plan = None
    if plan_path is not None:
        plan = _load_plan(plan_path)
        event_id = plan.event_id

    cached = _require_cached_catalog(settings, event_id)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    store = _watch_store(settings)
    watcher = Watcher(
        None,
        store,
        event_id=event_id,
        catalog=cached,
        profile=current,
        travel=default_graph(),
    )

    added: list[str] = []
    missing: list[str] = []

    if plan is not None:
        for seeded in watcher.seed_from_plan(plan):
            added.append(seeded.code)
        if not added:
            # Worth explaining rather than shrugging: on an event that publishes no
            # isReservable flag there is no seat to wait for in the first place.
            reservable = sum(1 for item in plan.items if item.needs_reservation)
            if reservable == 0:
                console.print(
                    f"[yellow]No session in this plan takes reservations, so there is "
                    f"nothing to watch.[/yellow]\n[dim]{plan.event_id} does not publish "
                    "seat availability; only re:Invent-style events do.[/dim]"
                )
            else:
                console.print(
                    "[green]Every reservable session in this plan already has a seat.[/green]"
                )

    for token in sessions or []:
        resolved = watcher.add(token)
        if resolved is None:
            missing.append(token)
        else:
            added.append(resolved.code)

    if added:
        console.print(f"[green]Watching {len(added)} session(s):[/green] {', '.join(added)}")
    if missing:
        console.print(f"[red]Not found in the catalog:[/red] {', '.join(missing)}")
    if not added and not missing and plan is None:
        console.print(
            "[yellow]Nothing added.[/yellow] Pass session codes, or --plan to watch "
            "everything in a plan that still needs a seat."
        )


@watch_app.command("list")
def watch_list(event: EventOption = None) -> None:
    """Show the watch list."""
    settings = load_settings(event_id=event)
    store = _watch_store(settings)
    targets = store.list(settings.event_id)

    if not targets:
        console.print(f"Watch list is empty. [dim]{store.path}[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("code")
    table.add_column("session", overflow="fold")
    table.add_column("last seen")
    table.add_column("checks", justify="right")
    table.add_column("state")

    for target in targets:
        band = target.last_band.value if target.last_band else "-"
        if target.reserved_at is not None:
            state = "[green]reserved[/green]"
        elif target.notified_at is not None:
            state = "[yellow]notified[/yellow]"
        else:
            state = "[dim]watching[/dim]"
        table.add_row(target.code, target.title, band, str(target.checks), state)

    console.print(table)
    console.print(f"[dim]{store.path}[/dim]")


@watch_app.command("remove")
def watch_remove(
    sessions: Annotated[list[str], typer.Argument(help="Session codes or ids to stop watching.")],
    event: EventOption = None,
    clear_all: Annotated[bool, typer.Option("--all", help="Clear the whole list.")] = False,
) -> None:
    """Stop watching sessions."""
    settings = load_settings(event_id=event)
    store = _watch_store(settings)

    if clear_all:
        removed = store.clear(settings.event_id)
        console.print(f"[green]Removed {removed} target(s).[/green]")
        return

    cached = _require_cached_catalog(settings)
    for token in sessions:
        session = cached.resolve(token)
        session_id = session.session_id if session else token
        if store.remove(settings.event_id, session_id):
            console.print(f"[green]Stopped watching {token}.[/green]")
        else:
            console.print(f"[yellow]{token} was not on the watch list.[/yellow]")


@watch_app.command("run")
def watch_run(
    event: EventOption = None,
    profile_path: ProfileOption = None,
    surface: SurfaceOption = SurfaceChoice.AUTO,
    once: Annotated[bool, typer.Option("--once", help="Poll a single time and exit.")] = False,
    iterations: Annotated[
        int | None, typer.Option("--iterations", help="Stop after this many polls.")
    ] = None,
    interval: Annotated[
        int, typer.Option("--interval", help="Seconds between polls.")
    ] = DEFAULT_INTERVAL_SECONDS,
    auto_reserve: Annotated[
        bool,
        typer.Option(
            "--auto-reserve",
            help="Take the seat automatically. Off by default: spending a reservation "
            "unasked is a big step.",
        ),
    ] = False,
    minimum_score: Annotated[
        float, typer.Option("--min-score", help="Only auto-reserve above this score.")
    ] = 0.0,
    slack: Annotated[
        bool, typer.Option("--slack/--no-slack", help="Also post to REPLAN_SLACK_WEBHOOK_URL.")
    ] = True,
    overrides: Annotated[
        Path | None, typer.Option("--travel-overrides", help="Your own travel times.")
    ] = None,
) -> None:
    """Poll the watch list and act when a seat opens.

    Seat availability is published as a band, not a count, so an opening is detected as
    a transition: the band moving back off `unavailable` since the last check.
    """
    settings = load_settings(event_id=event)
    cached = _require_cached_catalog(settings)
    try:
        current = load_profile(profile_path)
    except ProfileError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    store = _watch_store(settings)
    if not store.pending(settings.event_id):
        console.print(
            "[yellow]Nothing to watch.[/yellow] Add sessions with "
            "[bold]replan watch add --plan plan.json[/bold]."
        )
        raise typer.Exit(1)

    port = build_port(settings, surface)
    print_provenance(console, port, settings)

    notifier = build_notifier(
        console=True,
        slack_webhook_url=settings.slack_webhook_url if slack else None,
        sns_topic_arn=settings.sns_topic_arn,
        aws_region=settings.aws_region,
        writer=console.print,
    )

    policy = WatchPolicy(auto_reserve=auto_reserve, minimum_score=minimum_score)
    watcher = Watcher(
        port,
        store,
        event_id=settings.event_id,
        catalog=cached,
        profile=current,
        travel=default_graph(overrides),
        policy=policy,
        notifier=notifier,
        interval_seconds=interval,
    )

    if auto_reserve:
        console.print(
            "[yellow]--auto-reserve is on: replan will take seats on your behalf when "
            "the policy allows it.[/yellow]"
        )
    console.print(
        f"[dim]polling about every {watcher.interval_seconds}s, with jitter. Ctrl-C to stop.[/dim]"
    )

    rounds = 1 if once else iterations

    def report(result: Any) -> None:
        console.print(f"[dim]{result.at:%H:%M:%S}  {result.summary()}[/dim]")
        for problem in result.errors:
            console.print(f"[red]{problem}[/red]")
        for note in result.skipped:
            console.print(f"[dim]{note}[/dim]")

    try:
        results = watcher.run(iterations=rounds, on_result=report)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
        return
    finally:
        if hasattr(port, "close"):
            port.close()

    total_openings = sum(len(item.openings) for item in results)
    total_reserved = sum(len(item.reserved) for item in results)
    console.print(
        f"\n[bold]{len(results)} poll(s): {total_openings} opening(s), "
        f"{total_reserved} reserved.[/bold]"
    )
    if watcher.reservations_closed:
        console.print(f"[yellow]{RESERVATIONS_OPEN_NOTE}[/yellow]")


# --------------------------------------------------------------------------- #
# mcp
# --------------------------------------------------------------------------- #
mcp_app = typer.Typer(
    help="Expose replan to a coding agent over MCP.",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve(
    surface: SurfaceOption = SurfaceChoice.AUTO,
    profile_path: ProfileOption = None,
    transport: Annotated[
        str,
        typer.Option("--transport", help="stdio (default), streamable-http, or sse."),
    ] = "stdio",
    host: Annotated[str, typer.Option("--host", help="Bind address for HTTP transports.")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option("--port", help="Port for HTTP transports.")] = 8930,
) -> None:
    """Run replan as an MCP server so Kiro or Claude can plan an agenda in chat.

    Normally you do not run this by hand: an MCP client launches it over stdio. See
    examples/mcp/kiro-mcp.json for a config you can drop into .kiro/settings/mcp.json.

    Nothing is printed on stdout, because stdout is the protocol channel.
    """
    from replan.mcp_server import serve

    if transport == "stdio":
        # Every diagnostic goes to stderr. A single stray character on stdout
        # corrupts the JSON-RPC stream and the client drops the connection.
        err_console.print(f"[dim]replan MCP server on stdio (surface {surface.value})[/dim]")
    else:
        console.print(
            f"[yellow]replan MCP server on {transport} at {host}:{port}[/yellow]\n"
            "[yellow]This transport has no authentication of its own, so keep it on "
            "loopback unless you put something in front of it.[/yellow]"
        )

    try:
        serve(
            surface=surface.value,
            profile_path=profile_path,
            transport=transport,
            host=host,
            port=port,
        )
    except ToolInputError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


@mcp_app.command("tools")
def mcp_tools() -> None:
    """List the tools replan exposes, and which of them can change anything."""
    import asyncio

    from replan.mcp_server import build_server

    async def collect() -> list[Any]:
        return list(await build_server().list_tools())

    try:
        tools = asyncio.run(collect())
    except ToolInputError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    table = Table(show_header=True, header_style="bold")
    table.add_column("tool")
    table.add_column("effect")
    table.add_column("what it does", overflow="fold")

    for tool in sorted(tools, key=lambda item: item.name):
        notes = tool.annotations
        if notes is not None and notes.destructive_hint:
            effect = "[red]writes[/red]"
        elif notes is not None and notes.read_only_hint:
            effect = "[green]read-only[/green]"
        else:
            effect = "[yellow]local write[/yellow]"
        if notes is not None and notes.open_world_hint:
            effect += " [dim]+network[/dim]"
        first_line = (tool.description or "").strip().splitlines()
        table.add_row(tool.name, effect, first_line[0] if first_line else "-")

    console.print(table)
    console.print(f"[dim]{len(tools)} tools. Launch with `replan mcp serve`.[/dim]")


# --------------------------------------------------------------------------- #
# crew
# --------------------------------------------------------------------------- #
@app.command("crew")
def crew_cmd(
    profiles_path: Annotated[
        Path,
        typer.Option(
            "--profiles",
            "-p",
            help="Directory of attendee profiles or a single YAML profile.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Path to write the collective crew plan as JSON.",
        ),
    ] = None,
    ics_dir: Annotated[
        Path | None,
        typer.Option(
            "--ics-dir",
            help="Directory to write per-member .ics calendar files.",
        ),
    ] = None,
    markdown: Annotated[
        Path | None,
        typer.Option(
            "--markdown",
            "-m",
            help="Path to write a shared Markdown coverage report.",
        ),
    ] = None,
    surface: SurfaceOption = SurfaceChoice.AUTO,
    event: EventOption = None,
    day: Annotated[
        list[str] | None,
        typer.Option("--day", "-d", help="Only solve these days."),
    ] = None,
    redundancy_penalty: Annotated[
        float,
        typer.Option(
            "--redundancy-penalty",
            help="Utility penalty for assigning two members to the same session.",
        ),
    ] = 20.0,
    greedy: Annotated[
        bool,
        typer.Option("--greedy", help="Force the greedy interval solver instead of CP-SAT."),
    ] = False,
    time_limit: Annotated[
        float,
        typer.Option("--time-limit", help="CP-SAT time limit per member in seconds."),
    ] = 15.0,
) -> None:
    """Solve for a coordinated crew schedule across multiple attendee profiles.

    Maximizes collective topic coverage for an engineering team while minimizing
    redundant attendance on breakout sessions (keynotes and must-attends are exempt).
    """
    if profiles_path.is_dir():
        profile_files = sorted(
            p for p in profiles_path.iterdir() if p.suffix in (".yaml", ".yml")
        )
        if not profile_files:
            console.print(f"[red]No YAML profiles found in {profiles_path}[/red]")
            raise typer.Exit(1)
    else:
        profile_files = [profiles_path]

    members: list[CrewMember] = []
    for p in profile_files:
        try:
            profile = load_profile(p)
            name = profile.name if profile.name and profile.name != "me" else p.stem.capitalize()
            role = profile.goals[0] if profile.goals else None
            members.append(CrewMember(name=name, profile=profile, role=role, profile_path=p))
        except ProfileError as err:
            console.print(f"[red]Error in {p.name}: {err}[/red]")
            raise typer.Exit(1) from err

    settings = load_settings(event_id=event)
    cache = CatalogCache(settings.cache_path)
    days = _parse_days(day, cache, settings.event_id) or None
    catalog = cache.load(settings.event_id)

    if catalog is None or not len(catalog):
        port = build_port(settings, choice=surface)
        try:
            print_provenance(console, port, settings)
            with console.status("Fetching catalog..."):
                catalog = port.list_sessions(settings.event_id)
            cache.save(catalog)
        finally:
            if hasattr(port, "close"):
                port.close()
    else:
        msg = f"[dim]using cached catalog for {settings.event_id} ({len(catalog)} sessions)[/dim]"
        console.print(msg)

    graph = default_graph()
    request = CrewSolveRequest(
        catalog=catalog,
        members=tuple(members),
        travel=graph,
        redundancy_penalty=redundancy_penalty,
        days=days,
        time_limit_seconds=time_limit,
        prefer_cpsat=not greedy,
    )

    with console.status("[bold green]Solving coordinated crew schedule...[/bold green]"):
        crew_plan = solve_crew(request)

    # Render results
    report = crew_plan.coverage
    pct_style = "green" if report.coverage_percentage >= 70 else "yellow"
    console.print(
        f"\n[bold]replan crew[/bold]  [dim]event[/dim] {crew_plan.event_id}  "
        f"[dim]members[/dim] {len(crew_plan.members)}  "
        f"[dim]unique sessions[/dim] {crew_plan.unique_session_count}  "
        f"[{pct_style}]{report.coverage_percentage:.1f}% topic coverage[/{pct_style}]"
    )

    member_table = Table(show_header=True, header_style="bold", title="Crew Members")
    member_table.add_column("member")
    member_table.add_column("role / primary focus", overflow="fold")
    member_table.add_column("sessions", justify="right")
    member_table.add_column("topics covered", overflow="fold")

    for member_name, plan in sorted(crew_plan.members.items()):
        member_topics = [t.topic for t in report.topics if member_name in t.covered_by]
        topic_summary = ", ".join(member_topics[:4]) if member_topics else "-"
        matching = next((m for m in members if m.name == member_name), None)
        role_str = matching.role if (matching and matching.role) else "-"
        member_table.add_row(
            f"[bold]{member_name}[/bold]",
            role_str,
            str(len(plan.items)),
            topic_summary,
        )
    console.print(member_table)

    cov_table = Table(show_header=True, header_style="bold", title="Topic Coverage Matrix")
    cov_table.add_column("topic")
    cov_table.add_column("priority", justify="right")
    cov_table.add_column("status")
    cov_table.add_column("covered by")
    cov_table.add_column("sessions", overflow="fold")

    for topic in report.topics[:12]:
        status_str = "[green]covered[/green]" if topic.is_covered else "[dim]uncovered[/dim]"
        who = ", ".join(topic.covered_by) if topic.covered_by else "-"
        codes = ", ".join(topic.session_codes[:3]) if topic.session_codes else "-"
        cov_table.add_row(
            topic.topic,
            f"{topic.weight:.1f}",
            status_str,
            who,
            codes,
        )
    console.print(cov_table)

    if report.shared_sessions:
        shared_table = Table(
            show_header=True, header_style="bold", title="Shared Sessions (All-Hands)"
        )
        shared_table.add_column("session code")
        shared_table.add_column("title", overflow="fold")
        shared_table.add_column("attendees")
        for code, attendees in report.shared_sessions:
            item_title = ""
            for member_plan in crew_plan.members.values():
                found_item = member_plan.item_for(code)
                if found_item:
                    item_title = found_item.title
                    break
            shared_table.add_row(code, item_title, ", ".join(attendees))
        console.print(shared_table)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(crew_plan.to_dict(), indent=2), encoding="utf-8")
        console.print(f"\n[green]Wrote crew plan JSON to {out}[/green]")

    if ics_dir is not None:
        created = export_crew_ics(crew_plan, ics_dir, catalog=catalog)
        console.print(f"[green]Exported {len(created)} calendar(s) to {ics_dir}[/green]")

    if markdown is not None:
        export_crew_markdown(crew_plan, markdown, catalog=catalog)
        console.print(f"[green]Wrote crew markdown report to {markdown}[/green]")


def main() -> None:
    """Console-script entry point.

    Expected failures - not signed in, throttled, operation closed, not
    registered - are reported as guidance rather than a traceback. A stack trace
    would be noise here: the API's own error table already says what to do, and
    :attr:`EventsApiError.remedy` carries that advice.
    """
    try:
        app()
    except (EventsApiError, AuthError) as error:
        console.print(f"[red]{error}[/red]")
        remedy = getattr(error, "remedy", None)
        if remedy:
            console.print(f"[yellow]{remedy}[/yellow]")
        # sys.exit, not typer.Exit: we are outside Typer's runtime here, where
        # typer.Exit would surface as an unhandled exception.
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(130)


__all__ = ["app", "main"]
