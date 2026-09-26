"""Closed vocabularies, derived from the live AWS Events API.

Every value here was checked against the published OpenAPI 3.1 description at
``https://api.awsevents.com/v1/openapi.json`` and against real payloads from the
public event catalogs the API serves anonymously.

Two rules shape this module:

1. **Parse, never trust.** The API documents that "a field is present only when
   the event provides it. Treat any single field as optional, and read
   defensively." Observed reality backs that up: one Summit catalog omits
   ``venue``, ``seatAvailability``, ``isReservable`` and ``isAllDaySession``
   entirely, and spells levels with an en dash (``"100 – Foundational"``) plus a
   literal ``"No Level"``.
2. **Never lose the original.** Each parsed enum keeps the upstream string on the
   model beside it, so the explainer can quote exactly what AWS said rather than
   our normalisation of it.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum, unique


@unique
class Level(IntEnum):
    """Content level.

    ``UNKNOWN`` exists because the live catalog really does return ``"No Level"``
    for keynotes and expo content. Ordering it at 0 keeps ``>=`` filters honest.
    """

    UNKNOWN = 0
    FOUNDATIONAL = 100
    INTERMEDIATE = 200
    ADVANCED = 300
    EXPERT = 400

    @property
    def label(self) -> str:
        if self is Level.UNKNOWN:
            return "No level"
        return f"L{self.value} {self.name.title()}"

    @classmethod
    def parse(cls, raw: object) -> Level:
        """Accept every spelling the API has been observed to use.

        Handles ``"300 – Advanced"`` (en dash), ``"300 - Advanced"`` (hyphen),
        ``"Level 300"``, ``300``, ``"400"``, and ``"No Level"``.
        """
        if raw is None:
            return cls.UNKNOWN
        if isinstance(raw, Level):
            return raw
        if isinstance(raw, int):
            try:
                return cls(raw)
            except ValueError:
                return cls.UNKNOWN
        text = str(raw).strip().lower()
        if not text:
            return cls.UNKNOWN
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            try:
                return cls(int(digits[:3]))
            except ValueError:
                return cls.UNKNOWN
        for member in cls:
            if member is not cls.UNKNOWN and member.name.lower() in text:
                return member
        return cls.UNKNOWN


@unique
class SessionFormat(StrEnum):
    """Normalised delivery format, mapped from the API's free-text ``type``.

    The optimiser treats formats differently: a workshop is a multi-hour seated
    commitment, a chalk talk is small and fills instantly, a keynote is
    effectively unmissable and never reservable. Unrecognised types degrade to
    ``OTHER`` rather than raising, because the API adds types as the event
    platform grows.
    """

    KEYNOTE = "keynote"
    INNOVATION_TALK = "innovation-talk"
    LEADERSHIP_SESSION = "leadership-session"
    BREAKOUT = "breakout"
    CHALK_TALK = "chalk-talk"
    WORKSHOP = "workshop"
    BUILDERS_SESSION = "builders-session"
    CODE_TALK = "code-talk"
    LIGHTNING_TALK = "lightning-talk"
    DEV_CHAT = "dev-chat"
    GAME_DAY = "game-day"
    JAM = "jam"
    BOOTCAMP = "bootcamp"
    EXAM_PREP = "exam-prep"
    CERTIFICATION_EXAM = "certification-exam"
    GAMIFIED_LEARNING = "gamified-learning"
    STARTUP_TALK = "startup-talk"
    LIGHTNING_DEMO = "lightning-demo"
    SOCIAL = "social"
    EXPO = "expo"
    MEAL = "meal"
    OTHER = "other"

    @property
    def label(self) -> str:
        return self.value.replace("-", " ").title()

    @property
    def is_hands_on(self) -> bool:
        """You are at a keyboard; you cannot half-attend or slip out early."""
        return self in {
            SessionFormat.WORKSHOP,
            SessionFormat.BUILDERS_SESSION,
            SessionFormat.GAME_DAY,
            SessionFormat.JAM,
            SessionFormat.BOOTCAMP,
        }

    @property
    def is_small_group(self) -> bool:
        """Low-capacity formats: the ones that go to ``veryLimited`` in minutes."""
        return self in {
            SessionFormat.CHALK_TALK,
            SessionFormat.WORKSHOP,
            SessionFormat.BUILDERS_SESSION,
            SessionFormat.CODE_TALK,
            SessionFormat.DEV_CHAT,
        }

    @property
    def is_unmissable(self) -> bool:
        """Never schedule over these unless explicitly told to."""
        return self in {SessionFormat.KEYNOTE, SessionFormat.INNOVATION_TALK}

    @classmethod
    def parse(cls, raw: object) -> SessionFormat:
        if raw is None:
            return cls.OTHER
        if isinstance(raw, SessionFormat):
            return raw
        text = str(raw).strip().lower()
        if not text:
            return cls.OTHER
        squashed = text.replace("'", "").replace("’", "")
        squashed = "-".join(part for part in squashed.replace("_", " ").split() if part)
        # Values on the left were all observed live or are documented examples.
        aliases: dict[str, SessionFormat] = {
            "breakout-session": cls.BREAKOUT,
            "session": cls.BREAKOUT,
            "chalk-talk": cls.CHALK_TALK,
            "chalktalk": cls.CHALK_TALK,
            "code-talk": cls.CODE_TALK,
            "workshop": cls.WORKSHOP,
            "hands-on-lab": cls.WORKSHOP,
            "lab": cls.WORKSHOP,
            "builders-session": cls.BUILDERS_SESSION,
            "builders-sessions": cls.BUILDERS_SESSION,
            "builder-session": cls.BUILDERS_SESSION,
            "keynote": cls.KEYNOTE,
            "innovation-talk": cls.INNOVATION_TALK,
            "leadership-session": cls.LEADERSHIP_SESSION,
            "lightning-talk": cls.LIGHTNING_TALK,
            "lightning-demo": cls.LIGHTNING_DEMO,
            "dev-chat": cls.DEV_CHAT,
            "devchat": cls.DEV_CHAT,
            "game-day": cls.GAME_DAY,
            "gameday": cls.GAME_DAY,
            "jam": cls.JAM,
            "bootcamp": cls.BOOTCAMP,
            "exam-prep": cls.EXAM_PREP,
            "certification-exam": cls.CERTIFICATION_EXAM,
            "gamified-learning": cls.GAMIFIED_LEARNING,
            "startup-talk": cls.STARTUP_TALK,
            "expo": cls.EXPO,
            "meal": cls.MEAL,
            "social": cls.SOCIAL,
        }
        if squashed in aliases:
            return aliases[squashed]
        try:
            return cls(squashed)
        except ValueError:
            return cls.OTHER


@unique
class SeatAvailability(StrEnum):
    """``SeatAvailability`` from the API: a band, never a seat count.

    This is the single most important modelling constraint the real API imposes
    on a planner. There is no "37 seats left" anywhere in the contract, so any
    logic that wants a number has to work from these five bands instead.
    """

    AVAILABLE = "available"
    LIMITED = "limited"
    VERY_LIMITED = "veryLimited"
    UNAVAILABLE = "unavailable"
    WALK_UP = "walkUp"

    @property
    def label(self) -> str:
        return {
            SeatAvailability.AVAILABLE: "seats available",
            SeatAvailability.LIMITED: "limited seats",
            SeatAvailability.VERY_LIMITED: "very limited seats",
            SeatAvailability.UNAVAILABLE: "full",
            SeatAvailability.WALK_UP: "walk-up, no reservation",
        }[self]

    @property
    def is_bookable(self) -> bool:
        """Whether asking for a seat right now could plausibly succeed."""
        return self in {
            SeatAvailability.AVAILABLE,
            SeatAvailability.LIMITED,
            SeatAvailability.VERY_LIMITED,
        }

    @property
    def needs_no_reservation(self) -> bool:
        return self is SeatAvailability.WALK_UP

    @property
    def urgency(self) -> int:
        """0 = no rush, 3 = book it now. Drives watcher priority ordering."""
        return {
            SeatAvailability.WALK_UP: 0,
            SeatAvailability.AVAILABLE: 1,
            SeatAvailability.LIMITED: 2,
            SeatAvailability.VERY_LIMITED: 3,
            SeatAvailability.UNAVAILABLE: 3,
        }[self]

    @classmethod
    def parse(cls, raw: object) -> SeatAvailability | None:
        """``None`` when the event does not publish availability at all."""
        if raw is None:
            return None
        if isinstance(raw, SeatAvailability):
            return raw
        text = str(raw).strip()
        if not text:
            return None
        for member in cls:
            if member.value.lower() == text.lower():
                return member
        collapsed = text.lower().replace(" ", "").replace("-", "").replace("_", "")
        for member in cls:
            if member.value.lower() == collapsed:
                return member
        return None


@unique
class BulkFailureCode(StrEnum):
    """Per-session refusal reason from ``ReserveSessions``/``AssociateFavorites``.

    The API warns that this list grows, and that a client switching exhaustively
    on today's values would break later, so ``parse`` folds anything unknown into
    ``OTHER`` and the caller always has a defined branch.
    """

    SESSION_NOT_RESERVABLE = "sessionNotReservable"
    SCHEDULE_CONFLICT = "scheduleConflict"
    ALREADY_SCHEDULED = "alreadyScheduled"
    SESSION_FULL = "sessionFull"
    INSUFFICIENT_ACCESS = "insufficientAccess"
    TIME_PASSED = "timePassed"
    ALREADY_FAVORITED = "alreadyFavorited"
    NOT_FAVORITED = "notFavorited"
    OTHER = "other"

    @property
    def is_already_done(self) -> bool:
        """Refusals that mean the desired end state already holds.

        These are the codes the write-back engine treats as success. The API is
        explicit that re-sending a bulk request is not a safe retry precisely
        because prior successes come back as these failures.
        """
        return self in {
            BulkFailureCode.ALREADY_SCHEDULED,
            BulkFailureCode.ALREADY_FAVORITED,
            BulkFailureCode.NOT_FAVORITED,
        }

    @property
    def is_retryable_later(self) -> bool:
        """Worth watching for: the answer could change without us doing anything."""
        return self in {BulkFailureCode.SESSION_FULL, BulkFailureCode.SCHEDULE_CONFLICT}

    @property
    def explanation(self) -> str:
        return {
            BulkFailureCode.SESSION_NOT_RESERVABLE: "this session does not take reservations",
            BulkFailureCode.SCHEDULE_CONFLICT: "it clashes with something already on your schedule",
            BulkFailureCode.ALREADY_SCHEDULED: "it was already on your schedule",
            BulkFailureCode.SESSION_FULL: "the session is full",
            BulkFailureCode.INSUFFICIENT_ACCESS: "your registration does not include this session",
            BulkFailureCode.TIME_PASSED: "the session has already started or finished",
            BulkFailureCode.ALREADY_FAVORITED: "it was already a favorite",
            BulkFailureCode.NOT_FAVORITED: "it was not a favorite to begin with",
            BulkFailureCode.OTHER: "the API refused it without a specific reason",
        }[self]

    @classmethod
    def parse(cls, raw: object) -> BulkFailureCode:
        if isinstance(raw, BulkFailureCode):
            return raw
        text = str(raw or "").strip()
        for member in cls:
            if member.value == text:
                return member
        collapsed = text.lower().replace("_", "").replace("-", "")
        for member in cls:
            if member.value.lower() == collapsed:
                return member
        return cls.OTHER


@unique
class KnownVenue(StrEnum):
    """re:Invent campus properties, as vertices of the travel graph.

    The API's ``venue`` field is free text and is sometimes absent altogether, so
    this is a *recognition* layer rather than a schema constraint: a venue we
    know gets real walking and shuttle times, and anything else falls back to a
    conservative estimate. That keeps the planner correct at Summits too, where
    there is only one building and no ``venue`` field at all.
    """

    VENETIAN = "venetian"
    ENCORE = "encore"
    WYNN = "wynn"
    CAESARS_FORUM = "caesars-forum"
    MGM_GRAND = "mgm-grand"
    MANDALAY_BAY = "mandalay-bay"
    ARIA = "aria"
    BELLAGIO = "bellagio"
    LUXOR = "luxor"

    @property
    def label(self) -> str:
        overrides = {
            KnownVenue.VENETIAN: "Venetian",
            KnownVenue.CAESARS_FORUM: "Caesars Forum",
            KnownVenue.MGM_GRAND: "MGM Grand",
            KnownVenue.MANDALAY_BAY: "Mandalay Bay",
        }
        return overrides.get(self, self.value.replace("-", " ").title())

    @classmethod
    def parse(cls, raw: object) -> KnownVenue | None:
        """Best-effort recognition. ``None`` means "not a property we have data for"."""
        if raw is None:
            return None
        if isinstance(raw, KnownVenue):
            return raw
        text = str(raw).strip().lower().replace("'", "")
        if not text:
            return None
        # Substring matching, because the API returns things like
        # "The Venetian Las Vegas" and "Mandalay Bay Convention Center".
        needles: tuple[tuple[str, KnownVenue], ...] = (
            ("venetian", cls.VENETIAN),
            ("palazzo", cls.VENETIAN),
            ("sands", cls.VENETIAN),
            ("encore", cls.ENCORE),
            ("wynn", cls.WYNN),
            ("forum", cls.CAESARS_FORUM),
            ("caesar", cls.CAESARS_FORUM),
            ("mgm", cls.MGM_GRAND),
            ("mandalay", cls.MANDALAY_BAY),
            ("aria", cls.ARIA),
            ("bellagio", cls.BELLAGIO),
            ("luxor", cls.LUXOR),
        )
        for needle, venue in needles:
            if needle in text:
                return venue
        return None


# re:Invent's long-standing session-code prefixes, used to group repeats and to
# infer a track when the catalog leaves ``tracks`` empty.
CODE_PREFIX_TRACKS: dict[str, str] = {
    "AIM": "AI/ML",
    "ANT": "Analytics",
    "ARC": "Architecture",
    "BIZ": "Business Applications",
    "CMP": "Compute",
    "CON": "Containers",
    "DAT": "Databases",
    "DEV": "Developer Tools",
    "DOP": "DevOps & Developer Productivity",
    "EUC": "End User Computing",
    "FSI": "Financial Services",
    "GAM": "Games",
    "GAI": "Generative AI",
    "HYB": "Hybrid Cloud & Edge",
    "IMP": "Innovation & Transformation",
    "INV": "Innovation & Transformation",
    "KEY": "Keynotes",
    "MAM": "Migration & Modernization",
    "NET": "Networking & Content Delivery",
    "OPN": "Open Source",
    "PEX": "Partner Experience",
    "PUS": "Public Sector",
    "SEC": "Security, Identity & Compliance",
    "STG": "Storage",
    "STP": "Startups",
    "SUS": "Sustainability",
    "SVS": "Serverless & Event-Driven",
    "XNT": "Startups",
}

__all__ = [
    "CODE_PREFIX_TRACKS",
    "BulkFailureCode",
    "KnownVenue",
    "Level",
    "SeatAvailability",
    "SessionFormat",
]
