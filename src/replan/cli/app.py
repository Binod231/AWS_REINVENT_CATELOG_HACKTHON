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
from replan.cli.factory import (
    SurfaceChoice,
    build_port,
    die,
    load_settings,
    print_provenance,
)
from replan.config import ALLOWED_CALLBACK_PORTS, RESERVATIONS_OPEN_NOTE
from replan.domain import Level

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


def main() -> None:
    """Console-script entry point.

    Expected failures - not signed in, throttled, event closed, not registered -
    are reported as guidance rather than a traceback. A stack trace here would
    just be noise: the API's own error table already says what to do, and
    :attr:`EventsApiError.remedy` carries that advice.
    """
    try:
        app()
    except (EventsApiError, AuthError) as error:
        console.print(f"[red]{error}[/red]")
        remedy = getattr(error, "remedy", None)
        if remedy:
            console.print(f"[yellow]{remedy}[/yellow]")
        # sys.exit, not typer.Exit: we are outside Typer's runtime here, so
        # typer.Exit would surface as an unhandled exception.
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(130)


__all__ = ["app", "main"]
