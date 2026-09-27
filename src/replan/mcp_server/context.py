"""Shared state for replan's MCP tools.

An MCP tool call is stateless, but almost every one of replan's tools needs the same
three expensive things: the session catalog, the scored profile, and the travel graph.
Rebuilding those per call would make a conversation with the agent feel like wading
through treacle, so they are built once, lazily, and memoised here.

Two rules shape this module.

**Reads come from the local cache, never from a surprise network walk.** ``ListSessions``
is paginated and re:Invent publishes thousands of sessions, so a "quick" catalog read is
minutes of requests against a 120-per-minute quota. An agent that triggered that by
asking an innocent question would be a bad citizen on someone else's API. So the tools
read the SQLite cache that ``replan sync`` fills, and refuse with an actionable message
when it is empty. Refreshing is its own tool, called on purpose.

**Plans outlive the call that made them.** ``optimize_agenda`` returns a plan id rather
than fifty sessions of JSON, and later tools take that id. Plans are written to disk as
well as held in memory, because an MCP client is free to restart the server between
turns and an id that stopped resolving would be baffling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from replan.adapters.factory import SurfaceChoice, build_port
from replan.cache import CatalogCache
from replan.config import Settings
from replan.domain import Catalog, now_utc
from replan.profiles import Profile, ProfileError, Scorer, discover_profile
from replan.solver import Plan
from replan.sync import Journal
from replan.travel import TravelGraph, default_graph
from replan.watch import WatchStore

PLAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
LATEST = "latest"
PLAN_HISTORY_LIMIT = 40
"""How many plan files to keep. An agent iterating on a profile produces a lot of
them, and none of them are worth unbounded disk."""


class ToolInputError(ValueError):
    """The caller asked for something that cannot be done, with a way to fix it."""


@dataclass(frozen=True, slots=True)
class StoredPlan:
    """A plan and the id the agent refers to it by."""

    plan_id: str
    plan: Plan
    path: Path | None = None


class ReplanContext:
    """Lazily built, memoised state shared by every tool."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        surface: SurfaceChoice = SurfaceChoice.AUTO,
        profile_path: Path | None = None,
        clock: Any = now_utc,
    ) -> None:
        self._settings = settings or Settings.from_env()
        self._surface = surface
        self._profile_path_hint = profile_path
        self._clock = clock

        self._catalogs: dict[str, Catalog] = {}
        self._scorers: dict[str, Scorer] = {}
        self._profile: Profile | None = None
        self._profile_source: Path | None = None
        self._travel: TravelGraph | None = None
        self._plans: dict[str, Plan] = {}
        self._latest_id: str | None = None
        self._ports: dict[SurfaceChoice, Any] = {}

    # --- basics ------------------------------------------------------------ #

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def surface(self) -> SurfaceChoice:
        return self._surface

    @property
    def default_event_id(self) -> str:
        return self._settings.event_id

    def event_or_default(self, event_id: str | None) -> str:
        return (event_id or self._settings.event_id).strip() or self._settings.event_id

    def settings_for(self, event_id: str) -> Settings:
        if event_id == self._settings.event_id:
            return self._settings
        return replace(self._settings, event_id=event_id)

    @property
    def plans_dir(self) -> Path:
        return self._settings.state_dir / "plans"

    # --- profile ----------------------------------------------------------- #

    @property
    def profile(self) -> Profile:
        if self._profile is None:
            self._load_profile()
        if self._profile is None:  # pragma: no cover - _load_profile always sets it
            msg = "profile failed to load"
            raise ToolInputError(msg)
        return self._profile

    @property
    def profile_source(self) -> Path | None:
        """Where the profile came from, or ``None`` for built-in defaults.

        Reported by every tool that scores anything. A plan built from defaults scores
        every session at zero, which looks like a bug unless you are told.
        """
        if self._profile is None:
            self._load_profile()
        return self._profile_source

    def _load_profile(self) -> None:
        try:
            self._profile, self._profile_source = discover_profile(
                self._profile_path_hint,
                search=(self._settings.state_dir,),
            )
        except ProfileError as error:
            raise ToolInputError(str(error)) from error

    def reload_profile(self, path: Path | None = None) -> Path | None:
        """Re-read the profile from disk, forgetting every score derived from it."""
        if path is not None:
            self._profile_path_hint = path
        self._profile = None
        self._profile_source = None
        self._scorers.clear()
        return self.profile_source

    # --- travel ------------------------------------------------------------ #

    @property
    def travel(self) -> TravelGraph:
        if self._travel is None:
            self._travel = default_graph()
        return self._travel

    # --- catalog ----------------------------------------------------------- #

    @property
    def cache(self) -> CatalogCache:
        return CatalogCache(self._settings.cache_path)

    def catalog(self, event_id: str | None = None) -> Catalog:
        """The cached catalog, or an error explaining how to get one."""
        wanted = self.event_or_default(event_id)
        found = self._catalogs.get(wanted)
        if found is not None:
            return found

        loaded = self.cache.load(wanted)
        if loaded is None or not len(loaded):
            msg = (
                f"No catalog cached for {wanted}. Call sync_catalog first, or run "
                f"`replan sync --event {wanted}` in a terminal. Reads go through the "
                "local cache on purpose: walking the live catalog is hundreds of "
                "paginated requests against a per-minute quota."
            )
            raise ToolInputError(msg)
        self._catalogs[wanted] = loaded
        return loaded

    def scorer(self, event_id: str | None = None) -> Scorer:
        wanted = self.event_or_default(event_id)
        found = self._scorers.get(wanted)
        if found is None:
            found = Scorer(self.profile, self.catalog(wanted))
            self._scorers[wanted] = found
        return found

    def forget_catalog(self, event_id: str | None = None) -> None:
        """Drop memoised state after a sync, so the next read sees new data."""
        wanted = self.event_or_default(event_id)
        self._catalogs.pop(wanted, None)
        self._scorers.pop(wanted, None)

    def resolve_session(self, token: str, event_id: str | None = None) -> Any:
        """A session by code or id, or an error naming what was tried."""
        catalog = self.catalog(event_id)
        found = catalog.resolve(token)
        if found is None:
            msg = (
                f"No session matching {token!r} in {self.event_or_default(event_id)}. "
                "Codes look like SVS401; ids are opaque. search_sessions finds either."
            )
            raise ToolInputError(msg)
        return found

    # --- ports and local stores -------------------------------------------- #

    def build_port(self, surface: SurfaceChoice | None = None) -> Any:
        """A port, reused across tool calls.

        Reuse is not just an optimisation. The quota ledger that paces requests lives
        on the client, so a fresh one per tool call would reset the pacing and let a
        chatty agent walk straight through the published per-minute limits. It also
        keeps the recorded surface coherent: writes made by one tool call are visible
        to the next, which is what makes a credential-free demo behave like the real
        thing.
        """
        choice = surface or self._surface
        found = self._ports.get(choice)
        if found is None:
            found = build_port(self._settings, choice)
            self._ports[choice] = found
        return found

    def close(self) -> None:
        """Release every port. Safe to call more than once."""
        for port in self._ports.values():
            closer = getattr(port, "close", None)
            if callable(closer):
                closer()
        self._ports.clear()

    def journal(self) -> Journal:
        return Journal(self._settings.journal_path)

    def watch_store(self) -> WatchStore:
        return WatchStore(self._settings.state_dir / "watch.db")

    # --- plans ------------------------------------------------------------- #

    def new_plan_id(self, event_id: str) -> str:
        """A readable id that cannot collide.

        Second resolution is plenty for a human reading a filename, and not nearly
        enough for an agent that solves twice in one breath - so a suffix is added when
        the obvious name is taken.
        """
        stamp = self._clock().strftime("%Y%m%d-%H%M%S")
        base = f"{event_id}-{stamp}"
        taken = set(self._plans) | {path.stem for path in self.plans_dir.glob("*.json")}
        if base not in taken:
            return base
        for suffix in range(2, 100):
            candidate = f"{base}-{suffix}"
            if candidate not in taken:
                return candidate
        return f"{base}-{len(taken)}"

    def remember_plan(self, plan: Plan, plan_id: str | None = None) -> StoredPlan:
        """Store a plan in memory and on disk, and prune old ones."""
        identifier = plan_id or self.new_plan_id(plan.event_id)
        self._plans[identifier] = plan
        self._latest_id = identifier
        path: Path | None = None
        try:
            path = plan.write(self.plans_dir / f"{identifier}.json")
            self._prune_plans()
        except OSError:
            # An unwritable state directory is not a reason to lose the plan: the
            # in-memory copy still answers every follow-up in this session.
            path = None
        return StoredPlan(plan_id=identifier, plan=plan, path=path)

    def _prune_plans(self) -> None:
        files = sorted(
            self.plans_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for stale in files[PLAN_HISTORY_LIMIT:]:
            stale.unlink(missing_ok=True)

    def plan(self, plan_id: str | None) -> StoredPlan:
        """Resolve a plan id, ``latest``, or nothing at all.

        ``None`` and ``latest`` both mean "the one we were just talking about", which
        is how an agent naturally refers to it.
        """
        wanted = (plan_id or LATEST).strip() or LATEST
        if wanted == LATEST:
            return self._latest_plan()

        if not PLAN_ID_PATTERN.match(wanted):
            # Plan ids become filenames, so anything with a path separator in it is
            # refused rather than resolved.
            msg = (
                f"{wanted!r} is not a valid plan id. Use the id returned by "
                "optimize_agenda, or 'latest'."
            )
            raise ToolInputError(msg)

        in_memory = self._plans.get(wanted)
        if in_memory is not None:
            return StoredPlan(plan_id=wanted, plan=in_memory)

        path = self.plans_dir / f"{wanted}.json"
        if path.exists():
            return self._read_plan(wanted, path)

        msg = (
            f"No plan called {wanted!r}. Call optimize_agenda to build one, or pass "
            "'latest'. Known: " + (", ".join(self.known_plan_ids()) or "none yet")
        )
        raise ToolInputError(msg)

    def _latest_plan(self) -> StoredPlan:
        # Tracked explicitly rather than taken from insertion order: reading an old
        # plan by id also caches it, and that must not make it "the latest".
        if self._latest_id is not None:
            return StoredPlan(plan_id=self._latest_id, plan=self._plans[self._latest_id])
        files = sorted(
            self.plans_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not files:
            msg = "No plan yet. Call optimize_agenda first."
            raise ToolInputError(msg)
        return self._read_plan(files[0].stem, files[0])

    def _read_plan(self, plan_id: str, path: Path) -> StoredPlan:
        try:
            plan = Plan.read(path)
        except (OSError, ValueError) as error:
            msg = f"could not read plan {plan_id!r}: {error}"
            raise ToolInputError(msg) from error
        self._plans[plan_id] = plan
        return StoredPlan(plan_id=plan_id, plan=plan, path=path)

    def known_plan_ids(self) -> tuple[str, ...]:
        on_disk = {path.stem for path in self.plans_dir.glob("*.json")}
        return tuple(sorted(on_disk | set(self._plans)))

    def now(self) -> datetime:
        stamp: datetime = self._clock()
        return stamp


__all__ = [
    "LATEST",
    "PLAN_HISTORY_LIMIT",
    "ReplanContext",
    "StoredPlan",
    "ToolInputError",
]
