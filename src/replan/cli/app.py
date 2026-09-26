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
from replan.domain import (
    KnownVenue,
    Level,
    Room,
    SeatAvailability,
    SessionFormat,
    resolve_day,
)
from replan.profiles import ProfileError, Scorer, load_profile, write_starter_profile
from replan.solver import SolveRequest, solve
from replan.travel import default_graph

console = Console()

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


def _require_cached_catalog(settings: Any) -> Any:
    """Load the cached catalog, or explain how to get one."""
    store = CatalogCache(settings.cache_path)
    cached = store.load(settings.event_id)
    if cached is None or not len(cached):
        console.print(
            f"[yellow]Nothing cached for {settings.event_id}.[/yellow] "
            "Run [bold]replan sync[/bold] first."
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
