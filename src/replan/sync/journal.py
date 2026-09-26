"""A local record of every write, so it can be undone.

The Events API offers no history: ``GetSchedule`` tells you what your schedule is now,
never what it was or who changed it. So if replan is going to reserve twenty seats on
your behalf, it has to remember what it did, or "put it back how it was" is impossible.

That is this module. Each run is journalled with its actions and their outcomes, and
``undo_run`` reverses the successful ones in reverse order.

One thing the journal cannot do is guess. ``CreatePersonalTime`` returns 204 with no
body, so the new entry's id is simply not in the response. The only way to learn it is
to read ``GetSchedule`` back and match on title and span - which the apply step does,
recording the id here. Without that, created blocks would be un-undoable.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from replan.domain import now_utc
from replan.sync.actions import Action, ActionKind, ActionStatus, ApplyOutcome

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    dry_run     INTEGER NOT NULL DEFAULT 0,
    surface     TEXT,
    summary     TEXT,
    undone_at   TEXT
);

CREATE TABLE IF NOT EXISTS entries (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq                    INTEGER NOT NULL,
    kind                   TEXT NOT NULL,
    payload                TEXT NOT NULL,
    status                 TEXT NOT NULL,
    detail                 TEXT,
    succeeded_ids          TEXT,
    created_personal_time_id TEXT,
    reversed_at            TEXT
);

CREATE INDEX IF NOT EXISTS idx_entries_run ON entries(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_runs_event ON runs(event_id, id DESC);
"""


@dataclass(frozen=True, slots=True)
class JournalEntry:
    entry_id: int
    seq: int
    action: Action
    status: ActionStatus
    detail: str
    succeeded_ids: tuple[str, ...]
    created_personal_time_id: str | None
    reversed_at: datetime | None

    @property
    def is_reversible(self) -> bool:
        """Whether undo can meaningfully reverse this entry."""
        if self.reversed_at is not None:
            return False
        if not self.status.is_success:
            return False
        if self.action.kind is ActionKind.CREATE_PERSONAL_TIME:
            # Only if we managed to learn the id it was given.
            return self.created_personal_time_id is not None
        return True


@dataclass(frozen=True, slots=True)
class JournalRun:
    run_id: int
    event_id: str
    started_at: datetime
    finished_at: datetime | None
    dry_run: bool
    surface: str | None
    summary: str | None
    undone_at: datetime | None
    entries: tuple[JournalEntry, ...] = ()

    @property
    def reversible_entries(self) -> tuple[JournalEntry, ...]:
        # Reverse order: undo the last thing first, so intermediate states stay
        # coherent the same way the forward run did.
        return tuple(
            sorted(
                (entry for entry in self.entries if entry.is_reversible),
                key=lambda entry: -entry.seq,
            )
        )


@dataclass(frozen=True, slots=True)
class UndoReport:
    run_id: int
    reversed_count: int = 0
    failed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{self.reversed_count} action(s) reversed"]
        if self.failed:
            parts.append(f"{len(self.failed)} could not be reversed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)


class Journal:
    """SQLite-backed write history."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    # --- recording --------------------------------------------------------- #

    def start_run(self, event_id: str, *, dry_run: bool, surface: str | None = None) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO runs (event_id, started_at, dry_run, surface) VALUES (?,?,?,?)",
                (event_id, now_utc().isoformat(), int(dry_run), surface),
            )
            run_id = cursor.lastrowid
        if run_id is None:  # pragma: no cover - sqlite always supplies one
            msg = "could not create a journal run"
            raise RuntimeError(msg)
        return int(run_id)

    def record(self, run_id: int, seq: int, outcome: ApplyOutcome) -> None:
        """Record one outcome.

        Written immediately rather than batched at the end, so a crash mid-run still
        leaves a journal that ``undo`` can act on. That is the whole point of having
        one.
        """
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO entries (
                    run_id, seq, kind, payload, status, detail, succeeded_ids,
                    created_personal_time_id
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    seq,
                    outcome.action.kind.value,
                    json.dumps(outcome.action.to_payload()),
                    outcome.status.value,
                    outcome.detail,
                    json.dumps(list(outcome.succeeded_ids)),
                    outcome.created_personal_time_id,
                ),
            )

    def attach_personal_time_id(self, run_id: int, seq: int, personal_time_id: str) -> None:
        """Record the id a created block turned out to have.

        Needed because CreatePersonalTime returns no body, so the id can only be
        discovered by reading the schedule back.
        """
        with self._transaction() as connection:
            connection.execute(
                "UPDATE entries SET created_personal_time_id = ? WHERE run_id = ? AND seq = ?",
                (personal_time_id, run_id, seq),
            )

    def finish_run(self, run_id: int, summary: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE runs SET finished_at = ?, summary = ? WHERE id = ?",
                (now_utc().isoformat(), summary, run_id),
            )

    def mark_reversed(self, entry_id: int) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE entries SET reversed_at = ? WHERE id = ?",
                (now_utc().isoformat(), entry_id),
            )

    def mark_run_undone(self, run_id: int) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE runs SET undone_at = ? WHERE id = ?",
                (now_utc().isoformat(), run_id),
            )

    # --- reading ----------------------------------------------------------- #

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def _entries_for(self, connection: sqlite3.Connection, run_id: int) -> tuple[JournalEntry, ...]:
        rows = connection.execute(
            "SELECT * FROM entries WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        entries: list[JournalEntry] = []
        for row in rows:
            try:
                action = Action.from_payload(json.loads(row["payload"]))
            except (ValueError, KeyError):
                continue
            try:
                succeeded = tuple(json.loads(row["succeeded_ids"] or "[]"))
            except ValueError:
                succeeded = ()
            entries.append(
                JournalEntry(
                    entry_id=int(row["id"]),
                    seq=int(row["seq"]),
                    action=action,
                    status=ActionStatus(row["status"]),
                    detail=row["detail"] or "",
                    succeeded_ids=succeeded,
                    created_personal_time_id=row["created_personal_time_id"],
                    reversed_at=self._parse_datetime(row["reversed_at"]),
                )
            )
        return tuple(entries)

    def run(self, run_id: int) -> JournalRun | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            entries = self._entries_for(connection, run_id)
        return JournalRun(
            run_id=int(row["id"]),
            event_id=row["event_id"],
            started_at=self._parse_datetime(row["started_at"]) or now_utc(),
            finished_at=self._parse_datetime(row["finished_at"]),
            dry_run=bool(row["dry_run"]),
            surface=row["surface"],
            summary=row["summary"],
            undone_at=self._parse_datetime(row["undone_at"]),
            entries=entries,
        )

    def latest_run(
        self, event_id: str | None = None, *, applied_only: bool = True
    ) -> JournalRun | None:
        clauses = []
        params: list[Any] = []
        if event_id:
            clauses.append("event_id = ?")
            params.append(event_id)
        if applied_only:
            clauses.append("dry_run = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT id FROM runs {where} ORDER BY id DESC LIMIT 1",  # noqa: S608
                params,
            ).fetchone()
        return self.run(int(row["id"])) if row else None

    def history(self, event_id: str | None = None, limit: int = 20) -> tuple[JournalRun, ...]:
        where = "WHERE event_id = ?" if event_id else ""
        params: list[Any] = [event_id] if event_id else []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT id FROM runs {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                [*params, limit],
            ).fetchall()
        runs = [self.run(int(row["id"])) for row in rows]
        return tuple(item for item in runs if item is not None)


def _reverse_of(entry: JournalEntry) -> Action | None:
    """The action that undoes a journalled one."""
    action = entry.action
    kind = action.kind

    if kind is ActionKind.RESERVE:
        # Only cancel what actually succeeded.
        ids = entry.succeeded_ids or action.session_ids
        if not ids:
            return None
        return Action(
            kind=ActionKind.CANCEL_RESERVATION,
            session_ids=ids,
            labels=action.labels,
            reason="undoing a reservation replan made",
        )

    if kind is ActionKind.ADD_FAVORITE:
        ids = entry.succeeded_ids or action.session_ids
        if not ids:
            return None
        return Action(
            kind=ActionKind.REMOVE_FAVORITE,
            session_ids=ids,
            labels=action.labels,
            reason="undoing a favorite replan added",
        )

    if kind is ActionKind.CREATE_PERSONAL_TIME:
        if entry.created_personal_time_id is None:
            return None
        return Action(
            kind=ActionKind.DELETE_PERSONAL_TIME,
            personal_time_id=entry.created_personal_time_id,
            title=action.title,
            span=action.span,
            reason="undoing a personal time block replan created",
        )

    if kind is ActionKind.DELETE_PERSONAL_TIME:
        if action.span is None:
            return None
        return Action(
            kind=ActionKind.CREATE_PERSONAL_TIME,
            span=action.span,
            title=action.title,
            description=action.description or f"{action.title} restored by replan undo",
            location=action.location,
            reason="restoring a personal time block replan deleted",
        )

    if kind is ActionKind.CANCEL_RESERVATION:
        return Action(
            kind=ActionKind.RESERVE,
            session_ids=action.session_ids,
            labels=action.labels,
            reason="restoring a reservation replan cancelled",
        )

    if kind is ActionKind.REMOVE_FAVORITE:
        return Action(
            kind=ActionKind.ADD_FAVORITE,
            session_ids=action.session_ids,
            labels=action.labels,
            reason="restoring a favorite replan removed",
        )

    # UpdatePersonalTime is a full replacement and we do not journal the previous
    # values, so it cannot be reversed faithfully. Saying so is better than
    # pretending.
    return None


def undo_run(
    journal: Journal,
    run: JournalRun,
    port: Any,
    *,
    dry_run: bool = True,
) -> UndoReport:
    """Reverse a journalled run, newest action first."""
    from replan.sync.apply import execute_action

    reversed_count = 0
    failed: list[str] = []
    skipped: list[str] = []

    for entry in run.reversible_entries:
        reverse = _reverse_of(entry)
        if reverse is None:
            skipped.append(f"{entry.action.kind.label} cannot be reversed automatically")
            continue

        if dry_run:
            reversed_count += 1
            continue

        outcome = execute_action(port, run.event_id, reverse)
        if outcome.is_success:
            journal.mark_reversed(entry.entry_id)
            reversed_count += 1
        else:
            failed.append(f"{reverse.describe()}: {outcome.detail}")

    if not dry_run and not failed:
        journal.mark_run_undone(run.run_id)

    return UndoReport(
        run_id=run.run_id,
        reversed_count=reversed_count,
        failed=tuple(failed),
        skipped=tuple(skipped),
    )


__all__ = [
    "SCHEMA_VERSION",
    "Journal",
    "JournalEntry",
    "JournalRun",
    "UndoReport",
    "undo_run",
]
