"""Shared fixtures.

``tests/fixtures/recorded/`` holds genuine AWS Events API responses captured by
``scripts/record_fixtures.py`` from events that need no credentials. Testing the
decoder against real payloads is the point: it catches the things a hand-written
fixture never would, like levels spelled with an en dash and whole fields
missing from a Summit catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

RECORDED = Path(__file__).parent / "fixtures" / "recorded"


def _load(name: str) -> Any:
    return json.loads((RECORDED / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def recorded_dir() -> Path:
    return RECORDED


@pytest.fixture(scope="session")
def list_events_payload() -> dict[str, Any]:
    return _load("list_events.json")


@pytest.fixture(scope="session")
def get_event_payload() -> dict[str, Any]:
    return _load("get_event.json")


@pytest.fixture(scope="session")
def list_sessions_payload() -> dict[str, Any]:
    return _load("list_sessions_page.json")


@pytest.fixture(scope="session")
def get_session_payload() -> dict[str, Any]:
    return _load("get_session.json")


@pytest.fixture(scope="session")
def public_event_id() -> str:
    return str(_load("public_event.json")["eventId"])


@pytest.fixture(scope="session")
def unauthenticated_payload() -> dict[str, Any]:
    return _load("unauthenticated_401.json")
