"""Pick a front door and hand back a ``CatalogPort``.

This lives with the adapters rather than the CLI because more than one caller needs
it: the ``replan`` command, the MCP server replan exposes to coding agents, and the
scheduled Lambda all have to answer the same question - live REST, live remote MCP,
or committed recordings?

``auto`` prefers the real thing and falls back to recordings only when the network is
unreachable, so nobody demos against a fixture by accident.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from replan.adapters.recorded import RecordedCatalogClient
from replan.adapters.rest import RestCatalogClient
from replan.config import Settings, Surface


class SurfaceChoice(StrEnum):
    """What ``--surface`` accepts."""

    REST = "rest"
    MCP = "mcp"
    RECORDED = "recorded"
    AUTO = "auto"


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


def surface_label(port: Any) -> str:
    """How a port describes itself, for provenance lines."""
    return str(getattr(port, "surface_name", "unknown"))


__all__ = ["SurfaceChoice", "build_port", "surface_label"]
