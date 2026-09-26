"""The poll loop.

Runs the same way in two places, which is the point of keeping it free of I/O
assumptions: as a local daemon on a laptop, and as a Lambda handler behind EventBridge
Scheduler. ``poll_once`` does all the work; ``run`` just calls it on a timer.

Pacing is not an afterthought. ``GetSession`` allows 120 requests a minute, and this
process is meant to stay up for days, so it goes through the same
:class:`~replan.adapters.throttle.QuotaLedger` as the rest of replan, honours
``Retry-After``, and jitters its sleep. A hundred attendees running this would otherwise
all wake on the minute boundary and hammer the same endpoint together.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from replan.adapters.auth import AuthError
from replan.adapters.errors import (
    EventsApiError,
    EventsNotFoundError,
    EventsOperationClosedError,
    EventsThrottledError,
)
from replan.domain import AttendeeSchedule, Catalog, SeatAvailability, Session, now_utc
from replan.profiles import Profile, Scorer
from replan.travel import TravelGraph
from replan.watch.notify import (
    Notification,
    NotificationKind,
    Notifier,
    reserved_notification,
    seat_open_notification,
)
from replan.watch.policy import Decision, Verdict, WatchPolicy, band_improved
from replan.watch.store import WatchStore, WatchTarget

DEFAULT_INTERVAL_SECONDS = 90
MIN_INTERVAL_SECONDS = 20
"""A floor on politeness. Nothing about this problem needs sub-20-second polling, and
the quota would not survive it across a watch list of any size."""

JITTER_FRACTION = 0.25


def _open_on_first_sight(target: WatchTarget, current: SeatAvailability | None) -> bool:
    """Whether a seat was already bookable the very first time we looked.

    ``band_improved`` deliberately reports nothing here, because the band did not
    change. But a target added while its seat happened to be free would then be watched
    forever in silence, which is precisely backwards. Report it once, on the first sweep
    only, so the attendee hears about a seat they can take right now.
    """
    return target.checks == 0 and current is not None and current.is_bookable


@dataclass(frozen=True, slots=True)
class PollResult:
    """What one sweep of the watch list found."""

    checked: int = 0
    openings: tuple[
        Decision,
        ...,
    ] = ()
    reserved: tuple[str, ...] = ()
    notified: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    at: datetime = field(default_factory=now_utc)
    reservations_closed: bool = False

    @property
    def found_anything(self) -> bool:
        return bool(self.openings)

    def summary(self) -> str:
        parts = [f"checked {self.checked}"]
        if self.openings:
            parts.append(f"{len(self.openings)} opening(s)")
        if self.reserved:
            parts.append(f"{len(self.reserved)} reserved")
        if self.notified:
            parts.append(f"{len(self.notified)} notified")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return ", ".join(parts)


class Watcher:
    """Polls a watch list and acts on openings."""

    def __init__(
        self,
        port: Any,
        store: WatchStore,
        *,
        event_id: str,
        catalog: Catalog,
        profile: Profile,
        travel: TravelGraph,
        policy: WatchPolicy | None = None,
        notifier: Notifier | None = None,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        sleeper: Any = time.sleep,
        clock: Any = now_utc,
        rng: random.Random | None = None,
    ) -> None:
        self._port = port
        self._store = store
        self._event_id = event_id
        self._catalog = catalog
        self._profile = profile
        self._travel = travel
        self._policy = policy or WatchPolicy()
        self._notifier = notifier
        self._interval = max(interval_seconds, MIN_INTERVAL_SECONDS)
        self._sleep = sleeper
        self._clock = clock
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography
        self._scorer = Scorer(profile, catalog)
        self._reservations_closed = not self._policy.reservations_open

    @property
    def interval_seconds(self) -> int:
        return self._interval

    @property
    def reservations_closed(self) -> bool:
        return self._reservations_closed

    # --- seeding ----------------------------------------------------------- #

    def seed_from_plan(self, plan: Any, *, only_unbooked: bool = True) -> tuple[WatchTarget, ...]:
        """Add the plan's sessions that still need a seat.

        This is the natural entry point: the sessions worth watching are exactly the
        ones the optimiser wanted and could not book.
        """
        added: list[WatchTarget] = []
        for item in plan.items:
            if only_unbooked and item.already_reserved:
                continue
            if not item.needs_reservation:
                continue
            session = self._catalog.by_id(item.session_id)
            if session is None:
                continue
            added.append(self._store.add(self._event_id, session, priority=int(item.score)))
        return tuple(added)

    def add(self, token: str, *, priority: int = 0) -> WatchTarget | None:
        session = self._catalog.resolve(token)
        if session is None:
            return None
        return self._store.add(self._event_id, session, priority=priority)

    # --- one sweep --------------------------------------------------------- #

    def poll_once(self, *, schedule: AttendeeSchedule | None = None) -> PollResult:
        """Check every pending target once."""
        targets = self._store.pending(self._event_id)
        if not targets:
            return PollResult(checked=0, at=self._clock())

        current_schedule = schedule
        degraded_notes: list[str] = []
        blind = False

        if current_schedule is None:
            try:
                current_schedule = self._port.get_schedule(self._event_id)
            except (EventsApiError, AuthError) as error:
                # Without the schedule we cannot check for clashes, so auto-reserving
                # would risk trading a seat you hold for one you do not want. Degrade to
                # notify-only rather than either guessing or giving up: knowing a seat
                # freed up is still worth a phone buzzing.
                current_schedule = AttendeeSchedule(event_id=self._event_id)
                blind = True
                degraded_notes.append(
                    f"could not read your schedule ({error}); watching in notify-only "
                    "mode, with no conflict checking"
                )

        openings: list[Decision] = []
        reserved: list[str] = []
        notified: list[str] = []
        errors: list[str] = []
        skipped: list[str] = []
        checked = 0

        for target in targets:
            try:
                session = self._port.get_session(self._event_id, target.session_id)
            except EventsNotFoundError:
                skipped.append(f"{target.code} is no longer in the catalog")
                self._store.remove(self._event_id, target.session_id)
                continue
            except EventsThrottledError as error:
                wait = error.retry_after_seconds or 30
                errors.append(f"throttled; waiting {wait}s")
                self._sleep(wait)
                continue
            except EventsApiError as error:
                errors.append(f"{target.code}: {error}")
                continue

            checked += 1
            previous = target.last_band
            current = session.seat_availability
            self._store.record_band(self._event_id, target.session_id, current)

            if not band_improved(previous, current) and not _open_on_first_sight(target, current):
                continue

            # Flying blind means never booking: a reservation we cannot check against
            # the existing schedule is a reservation we should not make.
            policy = self._policy if not blind else replace(self._policy, auto_reserve=False)
            decision = policy.evaluate(
                session,
                target,
                schedule=current_schedule,
                catalog=self._catalog,
                travel=self._travel,
                profile=self._profile,
                score=self._scorer.score(session),
                reference=self._clock(),
            )

            if decision.verdict is Verdict.IGNORE:
                continue

            openings.append(decision)

            took_it = False
            if decision.should_reserve:
                took_it, note = self._try_reserve(session)
                if took_it:
                    reserved.append(session.code)
                    self._store.mark_reserved(self._event_id, target.session_id)
                else:
                    errors.append(note) if note else None

            self._notify(session, decision, reserved=took_it, previous=previous)
            notified.append(session.code)
            self._store.mark_notified(self._event_id, target.session_id)

        return PollResult(
            checked=checked,
            openings=tuple(openings),
            reserved=tuple(reserved),
            notified=tuple(notified),
            errors=tuple(errors),
            skipped=(*degraded_notes, *skipped),
            at=self._clock(),
            reservations_closed=self._reservations_closed,
        )

    def _try_reserve(self, session: Session) -> tuple[bool, str]:
        """Attempt the reservation, tolerating the documented refusals."""
        try:
            result = self._port.reserve_sessions(self._event_id, [session.session_id])
        except EventsOperationClosedError:
            # Latches, so the rest of the sweep does not keep asking a closed door.
            self._reservations_closed = True
            return False, (
                f"{session.code}: a seat is free but reservations are closed through "
                "the API until 8 October 2026"
            )
        except EventsApiError as error:
            return False, f"{session.code}: {error}"

        if session.session_id in result.effective_successes:
            return True, ""
        failure = result.failure_for(session.session_id)
        detail = failure.describe() if failure else "the API refused it"
        return False, f"{session.code}: {detail}"

    def _notify(
        self,
        session: Session,
        decision: Decision,
        *,
        reserved: bool,
        previous: SeatAvailability | None,
    ) -> None:
        if self._notifier is None or not decision.verdict.should_notify:
            return
        if reserved:
            self._notifier.send(reserved_notification(session, event_id=self._event_id))
            return
        was = previous.value if previous else "unseen"
        now = session.seat_availability.value if session.seat_availability else "unknown"
        change = f"{was} -> {now}" if was != now else f"{now} already, on the first check"
        self._notifier.send(
            seat_open_notification(
                session,
                event_id=self._event_id,
                reason=f"{change}. {decision.reason}",
            )
        )

    # --- the loop ---------------------------------------------------------- #

    def _next_sleep(self) -> float:
        """Interval with jitter, so concurrent watchers spread out."""
        spread = self._interval * JITTER_FRACTION
        return max(
            MIN_INTERVAL_SECONDS,
            self._interval + self._rng.uniform(-spread, spread),
        )

    def run(
        self,
        *,
        iterations: int | None = None,
        announce: bool = True,
        on_result: Any = None,
    ) -> tuple[PollResult, ...]:
        """Poll until the watch list is empty or ``iterations`` is reached."""
        if announce and self._notifier is not None:
            pending = len(self._store.pending(self._event_id))
            self._notifier.send(
                Notification(
                    kind=NotificationKind.WATCHER_STARTED,
                    title=f"Watching {pending} session(s) for {self._event_id}",
                    body=(
                        f"checking about every {self._interval}s"
                        + (
                            "; reservations are closed so this is notify-only"
                            if self._reservations_closed
                            else ""
                        )
                    ),
                    event_id=self._event_id,
                )
            )

        results: list[PollResult] = []
        count = 0
        while iterations is None or count < iterations:
            result = self.poll_once()
            results.append(result)
            if on_result is not None:
                on_result(result)

            count += 1
            if not self._store.pending(self._event_id):
                if announce and self._notifier is not None:
                    self._notifier.send(
                        Notification(
                            kind=NotificationKind.WATCHER_STOPPED,
                            title="Nothing left to watch",
                            body="every session on the watch list is settled",
                            event_id=self._event_id,
                        )
                    )
                break
            if iterations is not None and count >= iterations:
                break
            self._sleep(self._next_sleep())

        return tuple(results)


def lambda_handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
    """Entry point for a scheduled Lambda.

    Assembles everything from configuration and performs exactly one sweep, because the
    schedule, not the process, owns the cadence. Reads the catalog from the local cache
    when one is available in the bundle, and falls back to a live walk otherwise.
    """
    from replan.adapters.rest import RestCatalogClient
    from replan.cache import CatalogCache
    from replan.config import Settings
    from replan.profiles import load_profile
    from replan.travel import default_graph
    from replan.watch.notify import build_notifier

    settings = Settings.from_env()
    event_id = str(event.get("eventId") or settings.event_id)

    port = RestCatalogClient(settings)
    cache = CatalogCache(settings.cache_path)
    catalog = cache.load(event_id)
    if catalog is None:
        catalog = port.list_sessions(event_id)
        cache.save(catalog)

    profile_path = event.get("profilePath")
    from pathlib import Path

    profile = load_profile(Path(profile_path) if profile_path else None)

    notifier = build_notifier(
        console=bool(event.get("console", False)),
        slack_webhook_url=settings.slack_webhook_url,
        sns_topic_arn=settings.sns_topic_arn,
        aws_region=settings.aws_region,
    )

    policy = WatchPolicy(
        auto_reserve=bool(event.get("autoReserve", False)),
        minimum_score=float(event.get("minimumScore", 0.0)),
        reservations_open=bool(event.get("reservationsOpen", True)),
    )

    watcher = Watcher(
        port,
        WatchStore(settings.state_dir / "watch.db"),
        event_id=event_id,
        catalog=catalog,
        profile=profile,
        travel=default_graph(),
        policy=policy,
        notifier=notifier,
    )

    result = watcher.poll_once()
    return {
        "eventId": event_id,
        "checked": result.checked,
        "openings": [item.describe() for item in result.openings],
        "reserved": list(result.reserved),
        "notified": list(result.notified),
        "errors": list(result.errors),
        "summary": result.summary(),
    }


__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "JITTER_FRACTION",
    "MIN_INTERVAL_SECONDS",
    "PollResult",
    "Watcher",
    "lambda_handler",
]
