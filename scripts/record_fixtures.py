#!/usr/bin/env python3
"""Record real AWS Events API responses as committed test fixtures.

Four of the events the API currently serves do not require registration, and
their catalogs are readable with **no credentials at all**. That is what makes
this project testable against genuine data: the fixtures below are real API
payloads, not invented ones, so the decoder is verified against the shapes AWS
actually emits rather than the shapes we imagined.

Only anonymous, public reads are recorded. Nothing here touches an attendee
schedule, and no token is ever sent, so there is nothing to redact.

Usage:
    uv run python scripts/record_fixtures.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://api.awsevents.com/v1"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "recorded"
TIMEOUT_SECONDS = 30
USER_AGENT = (
    "replan-fixture-recorder/0.1 (+https://github.com/Binod231/AWS_REINVENT_CATELOG_HACKTHON)"
)


def fetch(path: str) -> tuple[int, Any]:
    url = f"{BASE_URL}{path}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"_nonJsonBody": raw[:500]}


def write(name: str, payload: Any) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    target = FIXTURE_DIR / name
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    size_kb = target.stat().st_size / 1024
    print(f"  wrote {target.relative_to(Path.cwd())} ({size_kb:.1f} KiB)")


def main() -> int:
    print("Recording public AWS Events API responses (no credentials sent)\n")

    status, events = fetch("/events")
    if status != 200:
        print(f"ListEvents failed with HTTP {status}", file=sys.stderr)
        return 1
    write("list_events.json", events)

    public = [event for event in events.get("items", []) if not event.get("authenticationRequired")]
    if not public:
        print("No public events available to record right now.", file=sys.stderr)
        return 1

    # Pick the public event with the largest catalog: the richest decoder test.
    best: tuple[int, str, Any] | None = None
    for event in public:
        event_id = event["eventId"]
        status, page = fetch(f"/events/{event_id}/sessions")
        if status != 200:
            print(f"  skipped {event_id}: HTTP {status}")
            continue
        count = int(page.get("totalCount") or len(page.get("items", [])))
        print(f"  {event_id}: {count} sessions")
        if best is None or count > best[0]:
            best = (count, event_id, page)

    if best is None:
        print("Could not read any public catalog.", file=sys.stderr)
        return 1

    _, event_id, page = best
    print(f"\nUsing {event_id} as the catalog fixture")
    write("public_event.json", {"eventId": event_id})
    write("list_sessions_page.json", page)

    status, single = fetch(f"/events/{event_id}")
    if status == 200:
        write("get_event.json", single)

    first = (page.get("items") or [{}])[0]
    if first.get("sessionId"):
        status, session = fetch(f"/events/{event_id}/sessions/{first['sessionId']}")
        if status == 200:
            write("get_session.json", session)

    # Record the authenticated-endpoint refusal too: the 401 shape is part of the
    # contract our error classifier has to handle, and it costs nothing to capture.
    status, denied = fetch("/events/reinvent2026/sessions")
    write("unauthenticated_401.json", {"status": status, "body": denied})

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
