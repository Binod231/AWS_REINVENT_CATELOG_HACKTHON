"""The ``replan`` Typer application.

Subcommands are registered here as each capability lands. Every command is a
thin shell: parse arguments, build the requested adapter, call into the pure
core, render the result. No business logic lives in this layer.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from replan import __version__

console = Console()

app = typer.Typer(
    name="replan",
    help=(
        "Agenda intelligence for the AWS re:Invent 2026 session catalog.\n\n"
        "Reads the catalog, scores it against your goals, solves for a "
        "travel-feasible schedule, explains every decision, and writes the "
        "result back to your attendee account."
    ),
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


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


def main() -> None:
    """Console-script entry point."""
    app()


__all__ = ["app", "main"]
