"""Catalog and attendee-state entities, mapped onto the live AWS Events API.

Field names follow the API's own vocabulary so a reader can hold the OpenAPI
description and this file side by side. Three decisions are worth calling out,
because each is forced by the real contract rather than chosen:

* **Almost everything is optional.** The API states that "a field is present only
  when the event provides it", and a live Summit catalog really does omit
  ``venue``, ``seatAvailability``, ``isReservable`` and ``isAllDaySession``. Only
  ``sessionId`` and ``title`` are guaranteed, so only those are required here.
* **Seat fullness is a band, not a count.** ``SeatAvailability`` is
  ``available | limited | veryLimited | unavailable | walkUp``. No number of
  free seats exists anywhere in the API, so nothing downstream may assume one.
* **Raw upstream strings are preserved.** ``type_raw`` and ``level_raw`` sit
  beside the parsed enums so the explainer can quote what AWS actually said.

Every model is frozen. The solver hashes thousands of these into sets and reuses
them across candidate schedules, and the explainer quotes the exact object that
produced a decision; mutable sessions would make that unanswerable.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Self
from zoneinfo import ZoneInfo

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from replan.domain.enums import (
    CODE_PREFIX_TRACKS,
    BulkFailureCode,
    KnownVenue,
    Level,
    SeatAvailability,
    SessionFormat,
)
from replan.domain.timespan import (
    REINVENT_TZ,
    TimeSpan,
    day_label,
    resolve_timezone,
)

_NON_WORD = re.compile(r"[^a-z0-9]+")
_CODE_PREFIX = re.compile(r"^([A-Z]{2,4})")
_REPEAT_SUFFIX = re.compile(r"-R(\d*)$", re.IGNORECASE)
_SPEAKER_SPLIT = re.compile(r"\s*,\s*")


def slugify(text: str) -> str:
    return _NON_WORD.sub("-", text.strip().lower()).strip("-")


class Frozen(BaseModel):
    """Immutable base. ``extra="forbid"`` guards our own construction sites;
    adapters are responsible for discarding unknown upstream keys deliberately
    rather than letting them silently flow through.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
class EventAddress(Frozen):
    city: str | None = None
    country: str | None = None
    state: str | None = None
    street: str | None = None
    postal_code: str | None = None


class Event(Frozen):
    """An AWS event. ``ListEvents``/``GetEvent`` need no credentials."""

    event_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    event_type: str | None = None
    start_date: AwareDatetime | None = None
    end_date: AwareDatetime | None = None
    is_online: bool = False
    timezone_name: str | None = None
    timezone_abbreviation: str | None = None
    time_format: str | None = None
    address: EventAddress | None = None
    supported_language_codes: tuple[str, ...] = ()
    authentication_required: bool = False

    @property
    def zone(self) -> ZoneInfo:
        """The event's own timezone, defaulting to re:Invent's."""
        return resolve_timezone(self.timezone_name)

    @property
    def is_reinvent(self) -> bool:
        return "reinvent" in self.event_id.lower()

    @property
    def days(self) -> tuple[date, ...]:
        if self.start_date is None or self.end_date is None:
            return ()
        zone = self.zone
        first = self.start_date.astimezone(zone).date()
        last = self.end_date.astimezone(zone).date()
        if last < first:
            return (first,)
        return tuple(first + timedelta(days=offset) for offset in range((last - first).days + 1))

    def __str__(self) -> str:
        return f"{self.name} ({self.event_id})"


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class Speaker(Frozen):
    """A presenter.

    The API gives a single ``name`` string, and live data packs role and employer
    into it: ``"Fatiha Bouabache, Solutions Architect, AWS"``. We keep the raw
    value and expose a best-effort split, because searching for "AWS" or for a
    job title is genuinely useful and guessing wrong here costs nothing.
    """

    name: str = Field(min_length=1)

    @property
    def parts(self) -> tuple[str, str | None, str | None]:
        chunks = [chunk for chunk in _SPEAKER_SPLIT.split(self.name.strip()) if chunk]
        if not chunks:
            return (self.name, None, None)
        person = chunks[0]
        title = chunks[1] if len(chunks) > 2 else None
        company = chunks[-1] if len(chunks) > 1 else None
        return (person, title, company)

    @property
    def person(self) -> str:
        return self.parts[0]

    @property
    def company(self) -> str | None:
        return self.parts[2]

    @property
    def is_aws(self) -> bool:
        return "aws" in self.name.lower() or "amazon" in self.name.lower()

    def __str__(self) -> str:
        return self.name


class Room(Frozen):
    """Where a session happens.

    ``venue`` is free text from the API and may be absent; ``known_venue`` is our
    recognition of it for travel purposes. Keeping both means a Summit in one
    building and re:Invent across six properties use the same type.
    """

    venue: str | None = None
    name: str | None = None
    known_venue: KnownVenue | None = None

    @model_validator(mode="after")
    def _resolve_venue(self) -> Self:
        if self.known_venue is None:
            # Room names sometimes carry the property when `venue` is absent.
            resolved = KnownVenue.parse(self.venue) or KnownVenue.parse(self.name)
            if resolved is not None:
                object.__setattr__(self, "known_venue", resolved)
        return self

    @property
    def key(self) -> str:
        venue_part = (
            self.known_venue.value if self.known_venue else slugify(self.venue or "unknown")
        )
        return f"{venue_part}/{slugify(self.name or 'unspecified')}"

    @property
    def display(self) -> str:
        parts = [part for part in (self.venue, self.name) if part]
        return " | ".join(parts) if parts else "Location TBD"

    @property
    def floor_hint(self) -> str | None:
        """A coarse in-building zone, used for intra-property walking penalties.

        Real values look like ``"Level 2 | Hall C"`` or ``"Floor 0, Women of the
        Cloud Theatre"``, so we pull the first level/floor token we can find.
        """
        if not self.name:
            return None
        match = re.search(r"\b(?:level|floor|lvl|fl)\.?\s*(\d+)", self.name, re.IGNORECASE)
        return match.group(1) if match else None

    @classmethod
    def parse(cls, text: str) -> Room:
        """Read back the string form this class prints.

        Used by the CLI so ``replan travel`` and ``replan now --location`` accept
        exactly what the rest of the tool displays.
        """
        parts = [chunk.strip() for chunk in text.split("|") if chunk.strip()]
        if not parts:
            msg = f"cannot parse a room from {text!r}"
            raise ValueError(msg)
        if len(parts) == 1:
            single = parts[0]
            if KnownVenue.parse(single) is not None:
                return cls(venue=single, name=None)
            return cls(venue=None, name=single)
        return cls(venue=parts[0], name=" | ".join(parts[1:]))

    def __str__(self) -> str:
        return self.display


class Session(Frozen):
    """One scheduled occurrence of catalog content.

    ``span`` is ``None`` for all-day and unscheduled entries, which the catalog
    does contain. The optimiser simply cannot place those, and says so, rather
    than inventing a time for them.
    """

    session_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abbreviation: str | None = None
    abstract: str | None = None
    level: Level = Level.UNKNOWN
    level_raw: str | None = None
    format: SessionFormat = SessionFormat.OTHER
    type_raw: str | None = None
    room: Room = Room()
    span: TimeSpan | None = None
    timezone_name: str | None = None
    is_all_day: bool = False
    is_reservable: bool | None = None
    seat_availability: SeatAvailability | None = None

    # Taxonomy lists, all straight from the API.
    tracks: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()
    areas_of_interest: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    segments: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    customer_personas: tuple[str, ...] = ()
    experiences: tuple[str, ...] = ()
    additional_activities: tuple[str, ...] = ()
    focus_areas: tuple[str, ...] = ()
    speakers: tuple[Speaker, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @field_validator(
        "tracks",
        "topics",
        "industries",
        "areas_of_interest",
        "roles",
        "services",
        "segments",
        "features",
        "customer_personas",
        "experiences",
        "additional_activities",
        "focus_areas",
        mode="before",
    )
    @classmethod
    def _coerce_str_sequence(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list | tuple):
            return tuple(str(item) for item in value if item is not None and str(item).strip())
        return value

    # --- derived -------------------------------------------------------- #

    @property
    def zone(self) -> ZoneInfo:
        return resolve_timezone(self.timezone_name)

    @property
    def code(self) -> str:
        """Public session code, falling back to the id when absent."""
        return self.abbreviation or self.session_id

    @property
    def series(self) -> str:
        """Identifier shared by every repeat of the same content.

        re:Invent publishes repeats as separate catalog entries with codes like
        ``SVS401-R1``. Stripping the suffix lets the optimiser treat repeats as
        interchangeable and escape a clash by taking a later showing - a move
        humans routinely miss when hand-building an agenda.
        """
        return _REPEAT_SUFFIX.sub("", self.code).upper()

    @property
    def repeat_index(self) -> int | None:
        match = _REPEAT_SUFFIX.search(self.code)
        if not match:
            return None
        return int(match.group(1)) if match.group(1) else 1

    @property
    def is_repeat(self) -> bool:
        return self.repeat_index is not None

    @property
    def code_prefix(self) -> str | None:
        match = _CODE_PREFIX.match(self.code.upper())
        return match.group(1) if match else None

    @property
    def effective_tracks(self) -> tuple[str, ...]:
        """Tracks as published, or inferred from the code prefix when empty."""
        if self.tracks:
            return self.tracks
        prefix = self.code_prefix
        inferred = CODE_PREFIX_TRACKS.get(prefix or "")
        return (inferred,) if inferred else ()

    @property
    def day(self) -> date | None:
        return self.span.day_in(self.zone) if self.span else None

    @property
    def day_label(self) -> str | None:
        current = self.day
        return day_label(current) if current else None

    @property
    def duration_minutes(self) -> int | None:
        return self.span.minutes if self.span else None

    @property
    def is_scheduled(self) -> bool:
        """Whether this session can be placed on a timeline at all."""
        return self.span is not None and not self.is_all_day

    @property
    def needs_reservation(self) -> bool:
        if self.is_reservable is False:
            return False
        if self.seat_availability is SeatAvailability.WALK_UP:
            return False
        return bool(self.is_reservable)

    @property
    def is_full(self) -> bool:
        return self.seat_availability is SeatAvailability.UNAVAILABLE

    @property
    def is_bookable_now(self) -> bool:
        if not self.needs_reservation:
            return False
        return self.seat_availability is None or self.seat_availability.is_bookable

    @property
    def scarcity(self) -> int:
        """0-3, how urgently a seat should be grabbed. Drives watcher ordering."""
        if self.seat_availability is not None:
            return self.seat_availability.urgency
        return 2 if self.format.is_small_group else 1

    @property
    def topic_keys(self) -> frozenset[str]:
        """Normalised concept set for coverage accounting.

        Deliberately wide: the API spreads meaning across ``topics``, ``tracks``,
        ``services``, ``areasOfInterest`` and ``focusAreas``, and a planner that
        only read ``topics`` would think a Bedrock session and a SageMaker
        session were the same thing.
        """
        return frozenset(
            slugify(item)
            for item in (
                *self.topics,
                *self.effective_tracks,
                *self.services,
                *self.areas_of_interest,
                *self.focus_areas,
            )
            if item.strip()
        )

    @property
    def search_blob(self) -> str:
        """Everything the full-text index should see."""
        return " ".join(
            part
            for part in (
                self.code,
                self.title,
                self.abstract or "",
                " ".join(speaker.name for speaker in self.speakers),
                " ".join(self.effective_tracks),
                " ".join(self.topics),
                " ".join(self.services),
                " ".join(self.areas_of_interest),
                " ".join(self.roles),
                " ".join(self.industries),
                " ".join(self.focus_areas),
                self.level_raw or "",
                self.type_raw or "",
                self.room.display,
            )
            if part
        )

    def __str__(self) -> str:
        return f"{self.code} {self.title}"


# --------------------------------------------------------------------------- #
# Attendee schedule
# --------------------------------------------------------------------------- #
class PersonalTime(Frozen):
    """A block of the attendee's own time.

    ``title`` and ``description`` are both required by the API, and
    ``UpdatePersonalTime`` is a full replacement that clears any optional field
    you omit, so this model carries everything needed to re-send it intact.
    """

    personal_time_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    span: TimeSpan
    location: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @property
    def created_by_replan(self) -> bool:
        """Whether replan authored this block.

        The write-back engine will delete blocks it created, but refuses to touch
        one the attendee made by hand unless forced. We cannot store metadata
        upstream, so authorship is recorded in the description with a marker.
        """
        return REPLAN_BLOCK_MARKER in (self.description or "")

    def day_in(self, zone: ZoneInfo = REINVENT_TZ) -> date:
        return self.span.day_in(zone)


REPLAN_BLOCK_MARKER = "[replan]"
"""Marker written into a personal-time description to record authorship.

The API has no user metadata field, so this is the only way to distinguish a
block replan created from one the attendee made by hand. It is visible to the
attendee on purpose: a tool that books time on your calendar should sign its work.
"""


class AttendeeSchedule(Frozen):
    """``GetSchedule``: reserved session ids, favorite session ids, personal time.

    Note what is *not* here, because it constrains the whole product: no waitlist,
    no reservation timestamps, no seat or confirmation numbers. The API returns
    bare id lists, so replan keeps its own local journal to know what it did.
    """

    event_id: str = Field(min_length=1)
    reserved: tuple[str, ...] = ()
    favorites: tuple[str, ...] = ()
    personal_time: tuple[PersonalTime, ...] = ()
    fetched_at: AwareDatetime | None = None

    @property
    def reserved_ids(self) -> frozenset[str]:
        return frozenset(self.reserved)

    @property
    def favorite_ids(self) -> frozenset[str]:
        return frozenset(self.favorites)

    @property
    def replan_blocks(self) -> tuple[PersonalTime, ...]:
        return tuple(block for block in self.personal_time if block.created_by_replan)

    @property
    def manual_blocks(self) -> tuple[PersonalTime, ...]:
        return tuple(block for block in self.personal_time if not block.created_by_replan)

    def block_by_id(self, personal_time_id: str) -> PersonalTime | None:
        for block in self.personal_time:
            if block.personal_time_id == personal_time_id:
                return block
        return None

    def is_empty(self) -> bool:
        return not (self.reserved or self.favorites or self.personal_time)


# --------------------------------------------------------------------------- #
# Bulk results
# --------------------------------------------------------------------------- #
class BulkFailure(Frozen):
    """One session a bulk request was refused for."""

    session_id: str = Field(min_length=1)
    code: BulkFailureCode = BulkFailureCode.OTHER
    code_raw: str | None = None
    conflicts_with: tuple[str, ...] = ()

    @property
    def is_already_done(self) -> bool:
        return self.code.is_already_done

    def describe(self) -> str:
        base = f"{self.session_id}: {self.code.explanation}"
        if self.conflicts_with:
            base += f" (clashes with {', '.join(self.conflicts_with)})"
        return base


class BulkResult(Frozen):
    """Outcome of ``ReserveSessions`` or ``AssociateFavorites``.

    The API is explicit that a 200 does not mean everything worked, and that
    prior successes are reported as failures on a retry. So ``failed`` is always
    read, and ``effective_successes`` folds in the "already done" codes to give
    the answer callers actually want: which sessions are now in the desired state.
    """

    successful: tuple[str, ...] = ()
    failed: tuple[BulkFailure, ...] = ()

    @property
    def effective_successes(self) -> frozenset[str]:
        return frozenset(self.successful) | {
            failure.session_id for failure in self.failed if failure.is_already_done
        }

    @property
    def real_failures(self) -> tuple[BulkFailure, ...]:
        return tuple(failure for failure in self.failed if not failure.is_already_done)

    @property
    def ok(self) -> bool:
        return not self.real_failures

    def failure_for(self, session_id: str) -> BulkFailure | None:
        for failure in self.failed:
            if failure.session_id == session_id:
                return failure
        return None


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
class CatalogStats(Frozen):
    total: int
    scheduled: int
    unscheduled: int
    by_day: tuple[tuple[str, int], ...]
    by_level: tuple[tuple[str, int], ...]
    by_format: tuple[tuple[str, int], ...]
    by_track: tuple[tuple[str, int], ...]
    by_venue: tuple[tuple[str, int], ...]
    by_seat_availability: tuple[tuple[str, int], ...]
    reservable: int
    repeats: int
    distinct_series: int
    distinct_speakers: int
    distinct_rooms: int


class Catalog(BaseModel):
    """An immutable view of one event's sessions at a point in time.

    ``source`` is not decoration. A tool that reserves real seats must never leave
    the user unsure whether they are looking at live data or a recorded fixture,
    so provenance travels with the data and is printed on every command.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    event_id: str = Field(min_length=1)
    sessions: tuple[Session, ...] = ()
    event: Event | None = None
    fetched_at: AwareDatetime | None = None
    source: str = "unknown"
    total_count: int | None = None
    locale: str | None = None

    _by_id: dict[str, Session] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _context: object, /) -> None:
        self._by_id.update({session.session_id: session for session in self.sessions})

    def __len__(self) -> int:
        return len(self.sessions)

    def iter_sessions(self) -> Iterator[Session]:
        return iter(self.sessions)

    @property
    def zone(self) -> ZoneInfo:
        return self.event.zone if self.event else REINVENT_TZ

    @property
    def is_complete(self) -> bool:
        """Whether we hold every session the API says the catalog has.

        The docs warn that a short page does not mean the last page; this is the
        assertion that a full walk actually finished.
        """
        return self.total_count is None or len(self.sessions) >= self.total_count

    def by_id(self, session_id: str) -> Session | None:
        return self._by_id.get(session_id)

    def require(self, session_id: str) -> Session:
        found = self._by_id.get(session_id)
        if found is None:
            msg = f"session {session_id!r} is not in this catalog"
            raise KeyError(msg)
        return found

    def by_code(self, code: str) -> tuple[Session, ...]:
        wanted = code.strip().upper()
        return tuple(
            session
            for session in self.sessions
            if session.code.upper() == wanted or session.series == wanted
        )

    def resolve(self, token: str) -> Session | None:
        """Look something up the way a human would type it: id, then code."""
        direct = self.by_id(token)
        if direct is not None:
            return direct
        matches = self.by_code(token)
        return matches[0] if matches else None

    @property
    def scheduled(self) -> tuple[Session, ...]:
        return tuple(session for session in self.sessions if session.is_scheduled)

    @property
    def days(self) -> tuple[date, ...]:
        return tuple(sorted({day for session in self.sessions if (day := session.day)}))

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(
            sorted({track for session in self.sessions for track in session.effective_tracks})
        )

    @property
    def services(self) -> tuple[str, ...]:
        return tuple(sorted({item for session in self.sessions for item in session.services}))

    @property
    def rooms(self) -> tuple[Room, ...]:
        return tuple(sorted({session.room for session in self.sessions}, key=lambda r: r.key))

    @property
    def known_venues(self) -> tuple[KnownVenue, ...]:
        return tuple(sorted({v for session in self.sessions if (v := session.room.known_venue)}))

    def repeats_of(self, session: Session) -> tuple[Session, ...]:
        """Every occurrence of the same content, including ``session`` itself."""
        return tuple(candidate for candidate in self.sessions if candidate.series == session.series)

    def replace_sessions(self, sessions: tuple[Session, ...]) -> Catalog:
        return Catalog(
            event_id=self.event_id,
            sessions=sessions,
            event=self.event,
            fetched_at=self.fetched_at,
            source=self.source,
            total_count=self.total_count,
            locale=self.locale,
        )

    def stats(self) -> CatalogStats:
        by_day: Counter[str] = Counter()
        by_level: Counter[str] = Counter()
        by_format: Counter[str] = Counter()
        by_track: Counter[str] = Counter()
        by_venue: Counter[str] = Counter()
        by_seats: Counter[str] = Counter()
        speakers: set[str] = set()
        series: set[str] = set()
        rooms: set[str] = set()
        reservable = repeats = scheduled = 0

        for session in self.sessions:
            current_day = session.day
            if current_day is not None:
                by_day[f"{day_label(current_day)} {current_day.isoformat()}"] += 1
            by_level[session.level.label] += 1
            by_format[session.format.label] += 1
            by_venue[
                session.room.known_venue.label
                if session.room.known_venue
                else (session.room.venue or "unspecified")
            ] += 1
            by_seats[
                session.seat_availability.value if session.seat_availability else "not published"
            ] += 1
            for track in session.effective_tracks:
                by_track[track] += 1
            for speaker in session.speakers:
                speakers.add(speaker.person)
            series.add(session.series)
            rooms.add(session.room.key)
            reservable += 1 if session.is_reservable else 0
            repeats += 1 if session.is_repeat else 0
            scheduled += 1 if session.is_scheduled else 0

        return CatalogStats(
            total=len(self.sessions),
            scheduled=scheduled,
            unscheduled=len(self.sessions) - scheduled,
            by_day=tuple(sorted(by_day.items())),
            by_level=tuple(sorted(by_level.items())),
            by_format=tuple(by_format.most_common()),
            by_track=tuple(by_track.most_common()),
            by_venue=tuple(by_venue.most_common()),
            by_seat_availability=tuple(by_seats.most_common()),
            reservable=reservable,
            repeats=repeats,
            distinct_series=len(series),
            distinct_speakers=len(speakers),
            distinct_rooms=len(rooms),
        )


__all__ = [
    "REPLAN_BLOCK_MARKER",
    "AttendeeSchedule",
    "BulkFailure",
    "BulkResult",
    "Catalog",
    "CatalogStats",
    "Event",
    "EventAddress",
    "Frozen",
    "PersonalTime",
    "Room",
    "Session",
    "Speaker",
    "slugify",
]
