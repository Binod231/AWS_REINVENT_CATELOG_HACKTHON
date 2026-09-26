"""SQLite-backed catalog cache with FTS5 search.

Design notes worth stating, because each is a deliberate trade:

* **Sessions are stored as their serialised domain form and re-validated on load.**
  That guarantees a cached read and a live read produce identical objects, and it
  means a decoder improvement applies to already-cached data for free.
* **Filterable columns are denormalised alongside.** Level, day, venue, format and
  seat band are extracted into real columns so a filtered search is an index scan
  rather than 2,000 JSON parses.
* **FTS5 is a separate table kept in step manually**, not an external-content
  table. External content would be tidier, but it couples the index's lifetime to
  row ids in a way that makes the whole-event replace below harder to reason about.
* **A sync replaces an event's rows inside one transaction.** The catalog is a
  snapshot, not a stream; a half-applied sync would be worse than no sync.
* **Graceful degradation to LIKE** when a Python build lacks FTS5, because being
  slower is much better than not running.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from replan.cache.changes import SyncDiff, diff_catalogs, session_fingerprint
from replan.domain import (
    Catalog,
    Event,
    KnownVenue,
    Level,
    SeatAvailability,
    Session,
    SessionFormat,
    day_label,
    now_utc,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalogs (
    event_id      TEXT PRIMARY KEY,
    fetched_at    TEXT,
    source        TEXT,
    total_count   INTEGER,
    session_count INTEGER,
    locale        TEXT,
    event_json    TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    event_id          TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    code              TEXT,
    series            TEXT,
    title             TEXT NOT NULL,
    level             INTEGER NOT NULL,
    format            TEXT NOT NULL,
    venue             TEXT,
    known_venue       TEXT,
    room_key          TEXT,
    start_utc         TEXT,
    end_utc           TEXT,
    day               TEXT,
    day_label         TEXT,
    duration_minutes  INTEGER,
    is_all_day        INTEGER NOT NULL DEFAULT 0,
    is_reservable     INTEGER,
    seat_availability TEXT,
    fingerprint       TEXT NOT NULL,
    payload           TEXT NOT NULL,
    PRIMARY KEY (event_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_day   ON sessions(event_id, day);
CREATE INDEX IF NOT EXISTS idx_sessions_level ON sessions(event_id, level);
CREATE INDEX IF NOT EXISTS idx_sessions_venue ON sessions(event_id, known_venue);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(event_id, start_utc);
CREATE INDEX IF NOT EXISTS idx_sessions_series ON sessions(event_id, series);

CREATE TABLE IF NOT EXISTS sync_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL,
    synced_at  TEXT NOT NULL,
    source     TEXT,
    total_now  INTEGER,
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS session_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sync_id     INTEGER NOT NULL REFERENCES sync_log(id) ON DELETE CASCADE,
    event_id    TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    code        TEXT,
    title       TEXT,
    kind        TEXT NOT NULL,
    fields_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_changes_event ON session_changes(event_id, sync_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    event_id UNINDEXED,
    session_id UNINDEXED,
    code,
    title,
    abstract,
    speakers,
    tracks,
    topics,
    services,
    roles,
    industries,
    room,
    tokenize='porter unicode61'
);
"""

_FTS_SPECIALS = re.compile(r'["\'()*:^-]')
_PHRASE = re.compile(r'"([^"]+)"')


def build_fts_query(text: str) -> str:
    """Turn human input into a safe FTS5 MATCH expression.

    User input is never passed through raw. FTS5 has its own operator syntax, and
    an unbalanced quote or a stray colon raises ``sqlite3.OperationalError`` rather
    than returning nothing - a search box that crashes on an apostrophe is not
    acceptable.

    Quoted phrases are honoured, bare words are ANDed, and every word gets a
    prefix wildcard so "bedrock agent" matches "Bedrock Agents".
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    terms: list[str] = []
    for phrase in _PHRASE.findall(cleaned):
        safe = _FTS_SPECIALS.sub(" ", phrase).strip()
        if safe:
            terms.append(f'"{safe}"')
    remainder = _PHRASE.sub(" ", cleaned)

    for word in remainder.split():
        safe = _FTS_SPECIALS.sub("", word).strip()
        if not safe:
            continue
        # Prefix search, so "lambda" also finds "lambda's" and "serverless" finds
        # "serverlessly". Porter stemming handles most of the rest.
        terms.append(f'"{safe}"*')

    return " AND ".join(terms)


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Structured narrowing, applied as SQL rather than in Python."""

    levels: frozenset[Level] = frozenset()
    formats: frozenset[SessionFormat] = frozenset()
    days: frozenset[date] = frozenset()
    venues: frozenset[KnownVenue] = frozenset()
    tracks: frozenset[str] = frozenset()
    services: frozenset[str] = frozenset()
    seat_bands: frozenset[SeatAvailability] = frozenset()
    reservable_only: bool = False
    bookable_only: bool = False
    scheduled_only: bool = False
    exclude_repeats: bool = False

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.levels,
                self.formats,
                self.days,
                self.venues,
                self.tracks,
                self.services,
                self.seat_bands,
                self.reservable_only,
                self.bookable_only,
                self.scheduled_only,
                self.exclude_repeats,
            )
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    session: Session
    score: float
    """Relevance. Lower is better (SQLite bm25 returns negative values)."""

    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class CacheInfo:
    event_id: str
    session_count: int
    total_count: int | None
    source: str | None
    fetched_at: datetime | None
    locale: str | None
    last_sync_summary: str | None = None
    sync_count: int = 0

    @property
    def is_complete(self) -> bool:
        return self.total_count is None or self.session_count >= self.total_count

    @property
    def age_seconds(self) -> float | None:
        if self.fetched_at is None:
            return None
        return (now_utc() - self.fetched_at).total_seconds()

    @property
    def age_human(self) -> str:
        age = self.age_seconds
        if age is None:
            return "unknown"
        if age < 90:
            return f"{int(age)}s ago"
        if age < 5400:
            return f"{int(age // 60)}m ago"
        if age < 172800:
            return f"{int(age // 3600)}h ago"
        return f"{int(age // 86400)}d ago"


class CatalogCache:
    """The local catalog database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fts_available = True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def supports_full_text(self) -> bool:
        return self._fts_available

    # --- connection ------------------------------------------------------- #

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
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

    def _initialise(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)
            try:
                connection.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError:
                # No FTS5 in this build. Search degrades to LIKE, which is slower
                # but still correct, and the CLI says so.
                self._fts_available = False
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # --- writing ---------------------------------------------------------- #

    def save(self, catalog: Catalog) -> SyncDiff:
        """Replace an event's cached sessions, returning what changed.

        The whole event is replaced in one transaction. A catalog is a snapshot,
        and a partially-applied sync would leave the planner reasoning about a
        mixture of two different days' data.
        """
        previous = self._load_sessions_map(catalog.event_id)
        current = {session.session_id: session for session in catalog.sessions}
        difference = diff_catalogs(
            catalog.event_id,
            previous,
            current,
            is_first_sync=not previous,
        )

        with self._transaction() as connection:
            connection.execute("DELETE FROM sessions WHERE event_id = ?", (catalog.event_id,))
            if self._fts_available:
                connection.execute(
                    "DELETE FROM sessions_fts WHERE event_id = ?", (catalog.event_id,)
                )

            connection.executemany(
                """
                INSERT INTO sessions (
                    event_id, session_id, code, series, title, level, format,
                    venue, known_venue, room_key, start_utc, end_utc, day,
                    day_label, duration_minutes, is_all_day, is_reservable,
                    seat_availability, fingerprint, payload
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [self._session_row(catalog.event_id, item) for item in catalog.sessions],
            )

            if self._fts_available:
                connection.executemany(
                    """
                    INSERT INTO sessions_fts (
                        event_id, session_id, code, title, abstract, speakers,
                        tracks, topics, services, roles, industries, room
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [self._fts_row(catalog.event_id, item) for item in catalog.sessions],
                )

            connection.execute(
                """
                INSERT INTO catalogs (
                    event_id, fetched_at, source, total_count, session_count,
                    locale, event_json
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                    fetched_at=excluded.fetched_at,
                    source=excluded.source,
                    total_count=excluded.total_count,
                    session_count=excluded.session_count,
                    locale=excluded.locale,
                    event_json=excluded.event_json
                """,
                (
                    catalog.event_id,
                    (catalog.fetched_at or now_utc()).isoformat(),
                    catalog.source,
                    catalog.total_count,
                    len(catalog.sessions),
                    catalog.locale,
                    catalog.event.model_dump_json() if catalog.event else None,
                ),
            )

            cursor = connection.execute(
                """
                INSERT INTO sync_log (event_id, synced_at, source, total_now, summary)
                VALUES (?,?,?,?,?)
                """,
                (
                    catalog.event_id,
                    now_utc().isoformat(),
                    catalog.source,
                    len(catalog.sessions),
                    difference.summary(),
                ),
            )
            sync_id = cursor.lastrowid

            if difference.changes:
                connection.executemany(
                    """
                    INSERT INTO session_changes (
                        sync_id, event_id, session_id, code, title, kind, fields_json
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    [
                        (
                            sync_id,
                            catalog.event_id,
                            change.session_id,
                            change.code,
                            change.title,
                            change.kind.value,
                            json.dumps(
                                [
                                    {
                                        "field": item.field_name,
                                        "before": item.before,
                                        "after": item.after,
                                    }
                                    for item in change.fields
                                ]
                            ),
                        )
                        for change in difference.changes
                    ],
                )

        return difference

    @staticmethod
    def _session_row(event_id: str, session: Session) -> tuple[Any, ...]:
        span = session.span
        current_day = session.day
        return (
            event_id,
            session.session_id,
            session.code,
            session.series,
            session.title,
            int(session.level),
            session.format.value,
            session.room.venue,
            session.room.known_venue.value if session.room.known_venue else None,
            session.room.key,
            span.start.isoformat() if span else None,
            span.end.isoformat() if span else None,
            current_day.isoformat() if current_day else None,
            day_label(current_day) if current_day else None,
            session.duration_minutes,
            1 if session.is_all_day else 0,
            None if session.is_reservable is None else int(session.is_reservable),
            session.seat_availability.value if session.seat_availability else None,
            session_fingerprint(session),
            session.model_dump_json(),
        )

    @staticmethod
    def _fts_row(event_id: str, session: Session) -> tuple[Any, ...]:
        return (
            event_id,
            session.session_id,
            session.code,
            session.title,
            session.abstract or "",
            " ".join(speaker.name for speaker in session.speakers),
            " ".join(session.effective_tracks),
            " ".join(session.topics),
            " ".join(session.services),
            " ".join(session.roles),
            " ".join(session.industries),
            session.room.display,
        )

    # --- reading ---------------------------------------------------------- #

    def _load_sessions_map(self, event_id: str) -> dict[str, Session]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id, payload FROM sessions WHERE event_id = ?",
                (event_id,),
            ).fetchall()
        result: dict[str, Session] = {}
        for row in rows:
            restored = self._revive(row["payload"])
            if restored is not None:
                result[row["session_id"]] = restored
        return result

    @staticmethod
    def _revive(payload: str) -> Session | None:
        """Rebuild a Session, tolerating a row written by an older schema."""
        try:
            return Session.model_validate_json(payload)
        except ValueError:
            return None

    def load(self, event_id: str) -> Catalog | None:
        """Rebuild the cached catalog, or ``None`` when nothing is cached."""
        with closing(self._connect()) as connection:
            header = connection.execute(
                "SELECT * FROM catalogs WHERE event_id = ?", (event_id,)
            ).fetchone()
            if header is None:
                return None
            rows = connection.execute(
                "SELECT payload FROM sessions WHERE event_id = ? ORDER BY session_id",
                (event_id,),
            ).fetchall()

        sessions = tuple(item for item in (self._revive(row["payload"]) for row in rows) if item)
        event: Event | None = None
        if header["event_json"]:
            try:
                event = Event.model_validate_json(header["event_json"])
            except ValueError:
                event = None

        return Catalog(
            event_id=event_id,
            sessions=sessions,
            event=event,
            fetched_at=(
                datetime.fromisoformat(header["fetched_at"]) if header["fetched_at"] else None
            ),
            source=f"cache ({header['source']})" if header["source"] else "cache",
            total_count=header["total_count"],
            locale=header["locale"],
        )

    def info(self, event_id: str) -> CacheInfo | None:
        with closing(self._connect()) as connection:
            header = connection.execute(
                "SELECT * FROM catalogs WHERE event_id = ?", (event_id,)
            ).fetchone()
            if header is None:
                return None
            log = connection.execute(
                """
                SELECT summary, COUNT(*) OVER () AS total
                FROM sync_log WHERE event_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (event_id,),
            ).fetchone()

        return CacheInfo(
            event_id=event_id,
            session_count=header["session_count"] or 0,
            total_count=header["total_count"],
            source=header["source"],
            fetched_at=(
                datetime.fromisoformat(header["fetched_at"]) if header["fetched_at"] else None
            ),
            locale=header["locale"],
            last_sync_summary=log["summary"] if log else None,
            sync_count=log["total"] if log else 0,
        )

    def cached_events(self) -> tuple[str, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT event_id FROM catalogs ORDER BY event_id").fetchall()
        return tuple(row["event_id"] for row in rows)

    # --- search ----------------------------------------------------------- #

    def search(
        self,
        event_id: str,
        query: str = "",
        *,
        filters: SearchFilters | None = None,
        limit: int = 25,
    ) -> tuple[SearchHit, ...]:
        """Full-text search with structured filters.

        With an empty query this is a pure filter listing ordered by start time,
        which is what ``--day tue --level 300`` should do.
        """
        narrowing = filters or SearchFilters()
        clauses, params = self._filter_sql(event_id, narrowing)

        match_expression = build_fts_query(query)
        use_fts = bool(match_expression) and self._fts_available

        if use_fts:
            sql = f"""
                SELECT s.payload AS payload, bm25(sessions_fts) AS score
                FROM sessions_fts
                JOIN sessions s
                  ON s.event_id = sessions_fts.event_id
                 AND s.session_id = sessions_fts.session_id
                WHERE sessions_fts MATCH ?
                  AND {" AND ".join(clauses)}
                ORDER BY score
                LIMIT ?
            """
            bindings: list[Any] = [match_expression, *params, limit]
        elif query.strip():
            # FTS5 unavailable: fall back to LIKE across the important columns.
            needle = f"%{query.strip().lower()}%"
            sql = f"""
                SELECT s.payload AS payload, 0.0 AS score
                FROM sessions s
                WHERE {" AND ".join(clauses)}
                  AND (lower(s.title) LIKE ? OR lower(s.code) LIKE ?)
                ORDER BY s.start_utc IS NULL, s.start_utc, s.code
                LIMIT ?
            """
            bindings = [*params, needle, needle, limit]
        else:
            sql = f"""
                SELECT s.payload AS payload, 0.0 AS score
                FROM sessions s
                WHERE {" AND ".join(clauses)}
                ORDER BY s.start_utc IS NULL, s.start_utc, s.code
                LIMIT ?
            """
            bindings = [*params, limit]

        with closing(self._connect()) as connection:
            try:
                rows = connection.execute(sql, bindings).fetchall()
            except sqlite3.OperationalError:
                # A malformed MATCH should degrade, never crash a search.
                return self.search(event_id, "", filters=narrowing, limit=limit)

        hits: list[SearchHit] = []
        for row in rows:
            restored = self._revive(row["payload"])
            if restored is None:
                continue
            hits.append(SearchHit(session=restored, score=float(row["score"])))

        # Post-filter the taxonomy lists, which live inside the JSON payload
        # rather than in columns. Cheap, because the SQL already narrowed hard.
        if narrowing.tracks or narrowing.services:
            hits = [hit for hit in hits if self._matches_taxonomy(hit.session, narrowing)]
        return tuple(hits)

    @staticmethod
    def _matches_taxonomy(session: Session, filters: SearchFilters) -> bool:
        if filters.tracks:
            wanted = {item.lower() for item in filters.tracks}
            have = {item.lower() for item in session.effective_tracks}
            if not (wanted & have):
                return False
        if filters.services:
            wanted = {item.lower() for item in filters.services}
            have = {item.lower() for item in session.services}
            if not (wanted & have):
                return False
        return True

    @staticmethod
    def _filter_sql(event_id: str, filters: SearchFilters) -> tuple[list[str], list[Any]]:
        clauses = ["s.event_id = ?"]
        params: list[Any] = [event_id]

        if filters.levels:
            placeholders = ",".join("?" for _ in filters.levels)
            clauses.append(f"s.level IN ({placeholders})")
            params.extend(int(level) for level in sorted(filters.levels))
        if filters.formats:
            placeholders = ",".join("?" for _ in filters.formats)
            clauses.append(f"s.format IN ({placeholders})")
            params.extend(sorted(item.value for item in filters.formats))
        if filters.days:
            placeholders = ",".join("?" for _ in filters.days)
            clauses.append(f"s.day IN ({placeholders})")
            params.extend(item.isoformat() for item in sorted(filters.days))
        if filters.venues:
            placeholders = ",".join("?" for _ in filters.venues)
            clauses.append(f"s.known_venue IN ({placeholders})")
            params.extend(sorted(item.value for item in filters.venues))
        if filters.seat_bands:
            placeholders = ",".join("?" for _ in filters.seat_bands)
            clauses.append(f"s.seat_availability IN ({placeholders})")
            params.extend(sorted(item.value for item in filters.seat_bands))
        if filters.reservable_only:
            clauses.append("s.is_reservable = 1")
        if filters.bookable_only:
            # Bookable means reservable and not full. walkUp is excluded because
            # there is no seat to take.
            clauses.append(
                "s.is_reservable = 1 AND "
                "(s.seat_availability IS NULL OR s.seat_availability IN "
                "('available','limited','veryLimited'))"
            )
        if filters.scheduled_only:
            clauses.append("s.start_utc IS NOT NULL AND s.is_all_day = 0")
        if filters.exclude_repeats:
            # Keep the earliest showing of each piece of content.
            clauses.append(
                "s.start_utc = (SELECT MIN(t.start_utc) FROM sessions t "
                "WHERE t.event_id = s.event_id AND t.series = s.series)"
            )
        return clauses, params

    # --- change history --------------------------------------------------- #

    def recent_changes(
        self, event_id: str, *, syncs: int = 1, limit: int = 100
    ) -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
        """``(kind, code, title, session_id, field descriptions)`` from recent syncs."""
        with closing(self._connect()) as connection:
            sync_ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM sync_log WHERE event_id = ? ORDER BY id DESC LIMIT ?",
                    (event_id, syncs),
                ).fetchall()
            ]
            if not sync_ids:
                return ()
            placeholders = ",".join("?" for _ in sync_ids)
            rows = connection.execute(
                f"""
                SELECT kind, code, title, session_id, fields_json
                FROM session_changes
                WHERE event_id = ? AND sync_id IN ({placeholders})
                ORDER BY id DESC LIMIT ?
                """,
                (event_id, *sync_ids, limit),
            ).fetchall()

        result = []
        for row in rows:
            descriptions: tuple[str, ...] = ()
            if row["fields_json"]:
                try:
                    parsed = json.loads(row["fields_json"])
                    descriptions = tuple(
                        f"{item['field']}: {item.get('before') or '-'} -> "
                        f"{item.get('after') or '-'}"
                        for item in parsed
                    )
                except (ValueError, KeyError, TypeError):
                    descriptions = ()
            result.append((row["kind"], row["code"], row["title"], row["session_id"], descriptions))
        return tuple(result)

    def sync_history(self, event_id: str, limit: int = 10) -> tuple[tuple[str, str], ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT synced_at, summary FROM sync_log WHERE event_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (event_id, limit),
            ).fetchall()
        return tuple((row["synced_at"], row["summary"] or "") for row in rows)

    # --- maintenance ------------------------------------------------------ #

    def clear(self, event_id: str | None = None) -> int:
        with self._transaction() as connection:
            if event_id is None:
                removed = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                for table in ("sessions", "catalogs", "session_changes", "sync_log"):
                    connection.execute(f"DELETE FROM {table}")
                if self._fts_available:
                    connection.execute("DELETE FROM sessions_fts")
            else:
                removed = connection.execute(
                    "SELECT COUNT(*) FROM sessions WHERE event_id = ?", (event_id,)
                ).fetchone()[0]
                for table in ("sessions", "catalogs", "session_changes", "sync_log"):
                    connection.execute(f"DELETE FROM {table} WHERE event_id = ?", (event_id,))
                if self._fts_available:
                    connection.execute("DELETE FROM sessions_fts WHERE event_id = ?", (event_id,))
        return int(removed)

    def distinct(self, event_id: str, column: str) -> tuple[str, ...]:
        """Distinct values of a whitelisted column, for shell completion and help."""
        allowed = {"day", "day_label", "format", "known_venue", "venue", "seat_availability"}
        if column not in allowed:
            msg = f"{column!r} is not a filterable column"
            raise ValueError(msg)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT DISTINCT {column} AS value FROM sessions "
                "WHERE event_id = ? AND value IS NOT NULL ORDER BY value",
                (event_id,),
            ).fetchall()
        return tuple(str(row["value"]) for row in rows)


__all__ = [
    "SCHEMA_VERSION",
    "CacheInfo",
    "CatalogCache",
    "SearchFilters",
    "SearchHit",
    "build_fts_query",
]
