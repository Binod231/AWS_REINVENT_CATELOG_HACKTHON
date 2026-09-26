"""Build the surface the user asked for, and report which one they got.

One rule governs this module: the user must always be able to tell live data from
a recording. Every command prints its provenance, and ``--surface`` is the only
way to change it. A planner that books real seats has no business being vague
about where its data came from.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any

import typer
from rich.console import Console

from replan.adapters.errors import EventsApiError
from replan.adapters.recorded import RecordedCatalogClient
from replan.adapters.rest import RestCatalogClient
from replan.config import Settings, Surface


class SurfaceChoice(StrEnum):
    """What ``--surface`` accepts."""

    REST = "rest"
    MCP = "mcp"
    RECORDED = "recorded"
    AUTO = "auto"


def load_settings(
    *,
    event_id: str | None = None,
    locale: str | None = None,
) -> Settings:
    try:
        settings = Settings.from_env()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    updates: dict[str, Any] = {}
    if event_id:
        updates["event_id"] = event_id
    if locale:
        updates["locale"] = locale
    return dataclasses.replace(settings, **updates) if updates else settings


def build_port(
    settings: Settings,
    choice: SurfaceChoice = SurfaceChoice.AUTO,
) -> Any:
    """Return a ``CatalogPort``.

    ``auto`` prefers the live REST API and falls back to recordings only when the
    network is unreachable, so the default experience is the real thing.
    """
    if choice is SurfaceChoice.RECORDED:
        return RecordedCatalogClient()
    if choice is SurfaceChoice.MCP:
        from replan.adapters.remote_mcp import RemoteMcpCatalogClient

        return RemoteMcpCatalogClient(settings)
    if choice is SurfaceChoice.REST:
        return RestCatalogClient(settings)

    # auto
    if settings.surface is Surface.RECORDED:
        return RecordedCatalogClient()
    if settings.surface is Surface.MCP:
        from replan.adapters.remote_mcp import RemoteMcpCatalogClient

        return RemoteMcpCatalogClient(settings)

    live = RestCatalogClient(settings)
    probe = live.probe()
    if probe.get("reachable"):
        return live
    live.close()
    return RecordedCatalogClient()


def print_provenance(console: Console, port: Any, settings: Settings) -> None:
    label = getattr(port, "surface_name", "unknown")
    style = "green" if label.startswith("live") else "yellow"
    console.print(
        f"[dim]source[/dim] [{style}]{label}[/{style}]  [dim]event[/dim] {settings.event_id}",
        highlight=False,
    )


def die(console: Console, error: EventsApiError) -> None:
    """Report an API failure the way the docs say to act on it, then exit 1."""
    console.print(f"[red]{error}[/red]")
    console.print(f"[yellow]{error.remedy}[/yellow]")
    raise typer.Exit(1)


__all__ = ["SurfaceChoice", "build_port", "die", "load_settings", "print_provenance"]
