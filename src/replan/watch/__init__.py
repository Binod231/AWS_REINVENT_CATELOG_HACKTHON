"""Watching for seats that open up, and taking them.

The catalog does not tell you how many seats are left. ``seatAvailability`` is a band -
``available``, ``limited``, ``veryLimited``, ``unavailable``, ``walkUp`` - so there is
no number to poll and no waitlist to join. The only signal that a seat freed up is the
band moving back off ``unavailable``, and the only way to see that is to have looked
before.

So this is a transition detector, not a counter. It remembers the band it last saw for
each watched session and acts when the band improves.

Three things it is careful about:

* **Politeness.** ``GetSession`` allows 120 requests a minute and the watcher is meant
  to run for days. It paces itself through the same quota ledger as everything else,
  honours ``Retry-After``, and adds jitter so a hundred attendees running this do not
  synchronise into a spike on the minute boundary.
* **Not making things worse.** An open seat is only worth taking if it does not clash
  with something you already hold and you can physically reach it. The policy checks
  both before spending a reservation.
* **Being useful today.** Reservations do not open through the API until 8 October
  2026. Until then the watcher degrades to notify-only rather than failing, because
  knowing a seat freed up is still worth a phone buzzing.
"""

from __future__ import annotations

from replan.watch.notify import (
    ConsoleNotifier,
    MultiNotifier,
    Notification,
    NotificationKind,
    Notifier,
    SlackNotifier,
    SnsNotifier,
    build_notifier,
)
from replan.watch.policy import Decision, Verdict, WatchPolicy
from replan.watch.store import WatchStore, WatchTarget
from replan.watch.watcher import PollResult, Watcher, lambda_handler

__all__ = [
    "ConsoleNotifier",
    "Decision",
    "MultiNotifier",
    "Notification",
    "NotificationKind",
    "Notifier",
    "PollResult",
    "SlackNotifier",
    "SnsNotifier",
    "Verdict",
    "WatchPolicy",
    "WatchStore",
    "WatchTarget",
    "Watcher",
    "build_notifier",
    "lambda_handler",
]
