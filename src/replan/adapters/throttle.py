"""Client-side quota accounting, so we stay inside the published limits.

The API publishes a per-operation, per-attendee, per-minute quota. Two details
make a naive limiter wrong, and both are handled here:

1. ``ReserveSessions`` and ``AssociateFavorites`` charge **per session named in
   the request**, not per request. A 30-session batch spends the entire minute's
   allowance in one call. :meth:`QuotaLedger.acquire` therefore takes a cost.
2. A refused request spends **no** quota. So on a 429 the right move is not
   blanket backoff: shrink the batch to what is left and retry immediately. That
   is what :meth:`QuotaLedger.remaining` is for.

Being a good citizen is not only politeness here. Going over the quota earns a
429 that would stall a seat watcher at exactly the moment a seat opens.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable

from replan.config import PER_SESSION_QUOTA_OPERATIONS, RetryPolicy, Settings

_WINDOW_SECONDS = 60.0


class QuotaLedger:
    """A sliding-window counter per operation.

    Thread-safe because the seat watcher polls concurrently with whatever the
    user is doing in the foreground.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._sleep = sleeper
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, operation: str, now: float) -> deque[float]:
        window = self._events.setdefault(operation, deque())
        cutoff = now - _WINDOW_SECONDS
        while window and window[0] <= cutoff:
            window.popleft()
        return window

    def remaining(self, operation: str) -> int:
        """How much of this operation's allowance is left in the current window."""
        with self._lock:
            now = self._clock()
            window = self._prune(operation, now)
            return max(self._settings.quota_for(operation) - len(window), 0)

    def cost_of(self, operation: str, session_count: int) -> int:
        """Quota units a call will spend."""
        if operation in PER_SESSION_QUOTA_OPERATIONS:
            return max(session_count, 1)
        return 1

    def acquire(self, operation: str, cost: int = 1) -> float:
        """Block until ``cost`` units are available, then record them.

        Returns the seconds spent waiting, which the CLI surfaces so a user never
        wonders why a bulk apply paused.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                window = self._prune(operation, now)
                quota = self._settings.quota_for(operation)
                if cost > quota:
                    msg = (
                        f"{operation} allows {quota} units per minute; "
                        f"a single call cannot spend {cost}. Split the batch."
                    )
                    raise ValueError(msg)
                if len(window) + cost <= quota:
                    for _ in range(cost):
                        window.append(now)
                    return waited
                # Wait until the oldest entry leaves the window.
                sleep_for = max(window[0] + _WINDOW_SECONDS - now, 0.05)
            self._sleep(sleep_for)
            waited += sleep_for

    def note_throttled(self, operation: str, retry_after_seconds: int | None) -> float:
        """Record that the API throttled us, and report how long to wait.

        A 429 spends no quota, so nothing is charged. We fill the local window
        instead, which stops us from immediately trying again and being refused.
        """
        with self._lock:
            now = self._clock()
            window = self._prune(operation, now)
            quota = self._settings.quota_for(operation)
            while len(window) < quota:
                window.append(now)
        return float(retry_after_seconds if retry_after_seconds is not None else 5)

    def snapshot(self) -> dict[str, int]:
        """Remaining allowance per operation we have touched. For ``replan doctor``."""
        with self._lock:
            now = self._clock()
            return {
                operation: max(
                    self._settings.quota_for(operation) - len(self._prune(operation, now)), 0
                )
                for operation in sorted(self._events)
            }


_DEFAULT_RNG = random.Random()  # noqa: S311 - jitter timing, not cryptography


def backoff_delay(
    attempt: int,
    policy: RetryPolicy,
    *,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with full jitter.

    Full jitter rather than equal jitter because every attendee's watcher wakes
    on the same minute boundary; spreading retries across the whole interval is
    what keeps a crowd of clients from synchronising into a spike.
    """
    # 2.0 rather than 2: an int base makes the result Any under strict typing,
    # because a negative exponent would yield a float.
    exponential: float = policy.base_delay_seconds * (2.0 ** max(attempt - 1, 0))
    capped: float = min(exponential, policy.max_delay_seconds)
    if not policy.jitter:
        return capped
    source = rng if rng is not None else _DEFAULT_RNG
    return float(source.uniform(0.0, capped))


__all__ = ["QuotaLedger", "backoff_delay"]
