"""CLI-side wiring: settings from flags, and how to report a surface or a failure.

One rule governs this module: the user must always be able to tell live data from
a recording. Every command prints its provenance, and ``--surface`` is the only
way to change it. A planner that books real seats has no business being vague
about where its data came from.

The surface factory itself lives in :mod:`replan.adapters.factory`, because the MCP
server and the scheduled Lambda need it too. It is re-exported here so callers keep
importing it from one place.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import typer
from rich.console import Console

from replan.adapters.errors import EventsApiError
from replan.adapters.factory import SurfaceChoice, build_port, surface_label
from replan.config import Settings


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


def print_provenance(console: Console, port: Any, settings: Settings) -> None:
    label = surface_label(port)
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


__all__ = [
    "SurfaceChoice",
    "build_port",
    "die",
    "load_settings",
    "print_provenance",
    "surface_label",
]
