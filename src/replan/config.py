"""Configuration, all of it overridable by environment variable.

Defaults are the real published values for the AWS Events API, so replan works
with no configuration at all. Every constant here was taken from the official
developer guide rather than guessed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from platformdirs import user_state_dir

# --------------------------------------------------------------------------- #
# Published endpoints and values (AWS Events API developer guide)
# --------------------------------------------------------------------------- #
DEFAULT_API_BASE_URL = "https://api.awsevents.com"
DEFAULT_API_VERSION = "v1"
DEFAULT_MCP_URL = "https://api.awsevents.com/mcp"

OAUTH_AUTHORIZE_URL = "https://oauth.awsevents.com/oauth2/authorize"
OAUTH_TOKEN_URL = "https://oauth.awsevents.com/oauth2/token"  # noqa: S105 - a URL, not a secret
OAUTH_REVOKE_URL = "https://oauth.awsevents.com/oauth2/revoke"
OAUTH_LOGOUT_URL = "https://oauth.awsevents.com/logout"
BUILDER_ID_LOGOUT_URL = "https://idp.awsevents.com/oidc/logout"

# Public, shared, and deliberately not a secret: the flow is protected by PKCE,
# so there is no client secret to hold.
OAUTH_CLIENT_ID = "7vmom55m1qstvq8i71ph127bfq"
OAUTH_SCOPES = ("openid", "email", "events/access")
OAUTH_IDENTITY_PROVIDER = "AWSBuilderID"

# Only these six loopback ports are registered, and only at /callback. The
# redirect URI is matched in full with no wildcards, so anything else fails
# without the API saying the callback was the cause.
ALLOWED_CALLBACK_PORTS = (8484, 8485, 8486, 8487, 8488, 8489)
CALLBACK_PATH = "/callback"

ACCESS_TOKEN_LIFETIME_SECONDS = 60 * 60
REFRESH_TOKEN_LIFETIME_SECONDS = 30 * 24 * 60 * 60

REINVENT_EVENT_ID = "reinvent2026"

# Reserved seating opens on 6 October 2026, but not through this API until
# 8 October 2026. Until then ReserveSessions and CancelReservation return 409 to
# every caller. Surfacing this as a date rather than a mystery error is the
# difference between a confusing tool and a clear one.
RESERVATIONS_OPEN_NOTE = (
    "Reserved seating opens through the Events API on 8 October 2026. "
    "Until then reserve and cancel return HTTP 409 for everyone; "
    "browsing and favorites work normally."
)


class Surface(StrEnum):
    """Which front door to talk to."""

    REST = "rest"
    """The live REST API at api.awsevents.com."""

    MCP = "mcp"
    """The live remote MCP server at api.awsevents.com/mcp."""

    RECORDED = "recorded"
    """Committed real responses, for tests and credential-free demos."""


# --------------------------------------------------------------------------- #
# Per-operation quotas, requests per minute (published table)
# --------------------------------------------------------------------------- #
OPERATION_QUOTAS: dict[str, int] = {
    "GetSession": 120,
    "ListSessions": 120,
    "GetSchedule": 60,
    "ReserveSessions": 30,
    "CancelReservation": 30,
    "AssociateFavorites": 30,
    "DisassociateFavorite": 30,
    "CreatePersonalTime": 30,
    "UpdatePersonalTime": 30,
    "DeletePersonalTime": 30,
}

# These two count each *session named in the request*, not each request, so a
# batch is no cheaper than one call per session. The limiter charges accordingly.
PER_SESSION_QUOTA_OPERATIONS = frozenset({"ReserveSessions", "AssociateFavorites"})

# ListEvents and GetEvent have no published quota. We still pace them, politely.
UNLISTED_OPERATION_QUOTA = 60


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() or None if value else None


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def default_state_dir() -> Path:
    override = _env("REPLAN_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_state_dir("replan", appauthor=False))


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Backoff for retryable failures.

    Full jitter, because a fleet of attendees all running a watcher on the minute
    boundary is exactly the thundering herd the API's quotas are there to stop.
    """

    max_attempts: int = 4
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    jitter: bool = True


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything replan needs to reach the API."""

    api_base_url: str = DEFAULT_API_BASE_URL
    api_version: str = DEFAULT_API_VERSION
    mcp_url: str = DEFAULT_MCP_URL
    event_id: str = REINVENT_EVENT_ID
    surface: Surface = Surface.REST
    locale: str | None = None

    state_dir: Path = field(default_factory=default_state_dir)

    callback_port: int = ALLOWED_CALLBACK_PORTS[0]
    client_id: str = OAUTH_CLIENT_ID

    request_timeout_seconds: float = 30.0
    max_requests_per_minute: int = 0
    """Global politeness cap. 0 means "use the published per-operation quotas"."""

    retry: RetryPolicy = field(default_factory=RetryPolicy)

    slack_webhook_url: str | None = None
    sns_topic_arn: str | None = None
    aws_region: str | None = None

    @property
    def api_root(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/{self.api_version.strip('/')}"

    @property
    def redirect_uri(self) -> str:
        """Must be byte-identical in the authorize request and the token exchange."""
        return f"http://localhost:{self.callback_port}{CALLBACK_PATH}"

    @property
    def token_path(self) -> Path:
        return self.state_dir / "token.json"

    @property
    def cache_path(self) -> Path:
        return self.state_dir / "catalog.db"

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "journal.db"

    def quota_for(self, operation: str) -> int:
        if self.max_requests_per_minute > 0:
            return self.max_requests_per_minute
        return OPERATION_QUOTAS.get(operation, UNLISTED_OPERATION_QUOTA)

    def url(self, path: str) -> str:
        return f"{self.api_root}/{path.lstrip('/')}"

    @classmethod
    def from_env(cls) -> Settings:
        port = _env_int("REPLAN_CALLBACK_PORT", ALLOWED_CALLBACK_PORTS[0])
        if port not in ALLOWED_CALLBACK_PORTS:
            allowed = ", ".join(str(item) for item in ALLOWED_CALLBACK_PORTS)
            msg = (
                f"REPLAN_CALLBACK_PORT={port} is not registered with the API. "
                f"Only these loopback ports work: {allowed}."
            )
            raise ValueError(msg)

        surface_raw = (_env("REPLAN_SURFACE") or _env("REPLAN_SOURCE") or "rest").lower()
        try:
            surface = Surface(surface_raw)
        except ValueError:
            valid = ", ".join(item.value for item in Surface)
            msg = f"REPLAN_SURFACE={surface_raw!r} is not one of: {valid}"
            raise ValueError(msg) from None

        return cls(
            api_base_url=_env("REPLAN_API_BASE_URL") or DEFAULT_API_BASE_URL,
            api_version=_env("REPLAN_API_VERSION") or DEFAULT_API_VERSION,
            mcp_url=_env("REPLAN_MCP_URL") or DEFAULT_MCP_URL,
            event_id=_env("REPLAN_EVENT_ID") or REINVENT_EVENT_ID,
            surface=surface,
            locale=_env("REPLAN_LOCALE"),
            state_dir=default_state_dir(),
            callback_port=port,
            client_id=_env("REPLAN_CLIENT_ID") or OAUTH_CLIENT_ID,
            request_timeout_seconds=_env_float("REPLAN_TIMEOUT_SECONDS", 30.0),
            max_requests_per_minute=_env_int("REPLAN_MAX_REQUESTS_PER_MINUTE", 0),
            slack_webhook_url=_env("REPLAN_SLACK_WEBHOOK_URL"),
            sns_topic_arn=_env("REPLAN_SNS_TOPIC_ARN"),
            aws_region=_env("AWS_REGION"),
        )


__all__ = [
    "ACCESS_TOKEN_LIFETIME_SECONDS",
    "ALLOWED_CALLBACK_PORTS",
    "BUILDER_ID_LOGOUT_URL",
    "CALLBACK_PATH",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_MCP_URL",
    "OAUTH_AUTHORIZE_URL",
    "OAUTH_CLIENT_ID",
    "OAUTH_IDENTITY_PROVIDER",
    "OAUTH_LOGOUT_URL",
    "OAUTH_REVOKE_URL",
    "OAUTH_SCOPES",
    "OAUTH_TOKEN_URL",
    "OPERATION_QUOTAS",
    "PER_SESSION_QUOTA_OPERATIONS",
    "REFRESH_TOKEN_LIFETIME_SECONDS",
    "REINVENT_EVENT_ID",
    "RESERVATIONS_OPEN_NOTE",
    "RetryPolicy",
    "Settings",
    "Surface",
    "default_state_dir",
]
