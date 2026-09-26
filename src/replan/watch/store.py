"""The watch list, and the last seat band seen for each entry.

Persistence is the whole mechanism. Because the API exposes no seat count and no
waitlist, "a seat opened" is definitionally a *change* since last time - which is
unknowable without a record of last time. This table is that record.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from replan.domain import SeatAvailability, Session, now_utc

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS targets (
    event_id      TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    code          TEXT,
    title         TEXT,
    priority      INTEGER NOT NULL DEFAULT 0,
    added_at      TEXT NOT NULL,
    last_band     TEXT,
    last_seen_at  TEXT,
    notified_at   TEXT,
    reserved_at   TEXT,
    checks        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (event_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_targets_event ON targets(event_id, priority DESC);
"""


@dataclass(frozen=True, slots=True)
class WatchTarget:
    """One session being watched."""

    event_id: str
    session_id: str
    code: str
    title: str
    priority: int = 0
    added_at: datetime | None = None
    last_band: SeatAvailability | None = None
    last_seen_at: datetime | None = None
    notified_at: datetime | None = None
    reserved_at: datetime | None = None
    checks: int = 0

    @property
    def is_settled(self) -> bool:
        """Whether there is nothing left to do for this target."""
        return self.reserved_at is not None

    @property
    def has_been_seen(self) -> bool:
        return self.last_band is not None or self.last_seen_at is not None

    def describe(self) -> str:
        band = self.last_band.value if self.last_band else "unknown"
        state = "reserved" if self.is_settled else band
        return f"{self.code} [{state}] {self.title}"


class WatchStore:
    """SQLite-backed watch list."""

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

    # --- mutation ---------------------------------------------------------- #

    def add(self, event_id: str, session: Session, *, priority: int = 0) -> WatchTarget:
        """Add or refresh a target.

        The current band is recorded on insert so the first poll has a baseline to
        compare against. Without it the first check would look like a transition.
        """
        band = session.seat_availability
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO targets (
                    event_id, session_id, code, title, priority, added_at,
                    last_band, last_seen_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id, session_id) DO UPDATE SET
                    priority = excluded.priority,
                    code = excluded.code,
                    title = excluded.title
                """,
                (
                    event_id,
                    session.session_id,
                    session.code,
                    session.title,
                    priority,
                    now_utc().isoformat(),
                    band.value if band else None,
                    now_utc().isoformat(),
                ),
            )
        target = self.get(event_id, session.session_id)
        if target is None:  # pragma: no cover - just inserted
            msg = f"failed to add {session.session_id} to the watch list"
            raise RuntimeError(msg)
        return target

    def remove(self, event_id: str, session_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM targets WHERE event_id = ? AND session_id = ?",
                (event_id, session_id),
            )
            return cursor.rowcount > 0

    def clear(self, event_id: str | None = None) -> int:
        with self._transaction() as connection:
            if event_id is None:
                cursor = connection.execute("DELETE FROM targets")
            else:
                cursor = connection.execute("DELETE FROM targets WHERE event_id = ?", (event_id,))
            return int(cursor.rowcount)

    def record_band(self, event_id: str, session_id: str, band: SeatAvailability | None) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE targets
                SET last_band = ?, last_seen_at = ?, checks = checks + 1
                WHERE event_id = ? AND session_id = ?
                """,
                (band.value if band else None, now_utc().isoformat(), event_id, session_id),
            )

    def mark_notified(self, event_id: str, session_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE targets SET notified_at = ? WHERE event_id = ? AND session_id = ?",
                (now_utc().isoformat(), event_id, session_id),
            )

    def mark_reserved(self, event_id: str, session_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE targets SET reserved_at = ? WHERE event_id = ? AND session_id = ?",
                (now_utc().isoformat(), event_id, session_id),
            )

    # --- reading ----------------------------------------------------------- #

    @staticmethod
    def _parse(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _row_to_target(self, row: sqlite3.Row) -> WatchTarget:
        return WatchTarget(
            event_id=row["event_id"],
            session_id=row["session_id"],
            code=row["code"] or row["session_id"],
            title=row["title"] or "",
            priority=int(row["priority"] or 0),
            added_at=self._parse(row["added_at"]),
            last_band=SeatAvailability.parse(row["last_band"]),
            last_seen_at=self._parse(row["last_seen_at"]),
            notified_at=self._parse(row["notified_at"]),
            reserved_at=self._parse(row["reserved_at"]),
            checks=int(row["checks"] or 0),
        )

    def get(self, event_id: str, session_id: str) -> WatchTarget | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM targets WHERE event_id = ? AND session_id = ?",
                (event_id, session_id),
            ).fetchone()
        return self._row_to_target(row) if row else None

    def list(self, event_id: str, *, include_settled: bool = True) -> tuple[WatchTarget, ...]:
        clause = "" if include_settled else " AND reserved_at IS NULL"
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM targets WHERE event_id = ?"  # noqa: S608 - fixed clause
                + clause
                + " ORDER BY priority DESC, code",
                (event_id,),
            ).fetchall()
        return tuple(self._row_to_target(row) for row in rows)

    def pending(self, event_id: str) -> tuple[WatchTarget, ...]:
        """Targets still worth polling."""
        return self.list(event_id, include_settled=False)

    def events(self) -> tuple[str, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT DISTINCT event_id FROM targets ORDER BY event_id"
            ).fetchall()
        return tuple(row["event_id"] for row in rows)


__all__ = ["SCHEMA_VERSION", "WatchStore", "WatchTarget"]
