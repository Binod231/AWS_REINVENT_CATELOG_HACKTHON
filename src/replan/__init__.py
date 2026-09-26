"""replan - agenda intelligence for the AWS re:Invent 2026 session catalog.

The package is organised as a hexagon:

* ``replan.domain``   - pure data model, no I/O, no framework imports.
* ``replan.adapters`` - the *only* modules that know the shape of the AWS
  Event Catalog API. Swapping REST for the remote MCP server, or for recorded
  fixtures, is a change of one class.
* ``replan.travel``, ``replan.profiles``, ``replan.solver``, ``replan.explain``
  - the decision core. Deterministic, network-free, unit-testable.
* ``replan.cli``, ``replan.mcp_server`` - driving interfaces.

Nothing in the core imports an adapter; adapters are injected. That is what
lets the whole product be built and demonstrated before the upstream API
contract is known.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
