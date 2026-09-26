"""AWS Builder ID sign-in: OAuth 2.0 authorization code flow with PKCE.

The API offers no hosted option and no client secret. An application runs on the
attendee's own machine, listens on one of six registered loopback ports, and
exchanges a code for a bearer token. This module implements exactly that, and the
details it is careful about are the ones the developer guide warns cost you an
error that "does not say which value was wrong":

* ``redirect_uri`` is matched in full, with no wildcards, and the authorization
  request and token exchange must send the **identical string**. It is therefore
  built once, in :attr:`replan.config.Settings.redirect_uri`, and never rebuilt.
* The code challenge must be the SHA-256 digest of the verifier, base64url
  encoded **without padding**. Hex or standard base64 fails.
* ``code_challenge_method=S256`` must be sent explicitly.
* ``identity_provider=AWSBuilderID`` must be sent, or the attendee gets a pointless
  extra page asking which provider to use when Builder ID is the only choice.
* A verifier may never be reused between attempts.

Token handling: the file is written with owner-only permissions, and no token,
code, or verifier is ever logged or included in an exception message.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import socket
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx

from replan.config import (
    ALLOWED_CALLBACK_PORTS,
    CALLBACK_PATH,
    OAUTH_AUTHORIZE_URL,
    OAUTH_IDENTITY_PROVIDER,
    OAUTH_REVOKE_URL,
    OAUTH_SCOPES,
    OAUTH_TOKEN_URL,
    Settings,
)

_VERIFIER_BYTES: Final = 64  # -> 86 base64url chars, inside the 43-128 range
_REFRESH_SKEW_SECONDS: Final = 120
_CALLBACK_TIMEOUT_SECONDS: Final = 300


class AuthError(RuntimeError):
    """Sign-in failed. Never carries a token, code, or verifier."""


class NotSignedInError(AuthError):
    """No usable token is stored, and none could be obtained silently."""


# --------------------------------------------------------------------------- #
# PKCE
# --------------------------------------------------------------------------- #
def _b64url(raw: bytes) -> str:
    """base64url with padding stripped, as the spec and this API require."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@dataclass(frozen=True, slots=True)
class PkcePair:
    verifier: str
    challenge: str
    state: str

    @classmethod
    def generate(cls) -> PkcePair:
        # token_urlsafe draws from `secrets`, a cryptographically secure source,
        # and is fresh on every call: reusing a verifier fails the exchange.
        verifier = secrets.token_urlsafe(_VERIFIER_BYTES)
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return cls(verifier=verifier, challenge=_b64url(digest), state=secrets.token_urlsafe(24))


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TokenSet:
    """The tokens returned by the token endpoint.

    Only ``access_token`` is ever sent to the Events API. The ID token describes
    who signed in and is explicitly *not* accepted in its place, so it is kept
    only to show the signed-in identity.
    """

    access_token: str
    expires_at: datetime
    refresh_token: str | None = None
    id_token: str | None = None
    scope: str | None = None

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at

    @property
    def expires_in_seconds(self) -> int:
        return max(int((self.expires_at - datetime.now(UTC)).total_seconds()), 0)

    @property
    def needs_refresh(self) -> bool:
        """True a little before real expiry, so a long call cannot expire mid-flight."""
        return datetime.now(UTC) >= self.expires_at - timedelta(seconds=_REFRESH_SKEW_SECONDS)

    def redacted(self) -> dict[str, Any]:
        """A safe-to-print summary. Used by ``replan doctor``."""
        return {
            "access_token": f"<{len(self.access_token)} chars>",
            "expires_at": self.expires_at.isoformat(),
            "expires_in_seconds": self.expires_in_seconds,
            "has_refresh_token": self.refresh_token is not None,
            "scope": self.scope,
        }

    @property
    def signed_in_as(self) -> str | None:
        """Read the email claim out of the ID token, for display only.

        The JWT signature is deliberately *not* verified here: this value is never
        used for an authorisation decision, only to show the attendee which
        account they are signed in as.
        """
        if not self.id_token:
            return None
        parts = self.id_token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        try:
            claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        except (ValueError, json.JSONDecodeError):
            return None
        email = claims.get("email")
        return str(email) if isinstance(email, str) else None

    @classmethod
    def from_response(
        cls, payload: dict[str, Any], *, previous: TokenSet | None = None
    ) -> TokenSet:
        access = payload.get("access_token")
        if not isinstance(access, str) or not access:
            msg = "token endpoint returned no access_token"
            raise AuthError(msg)
        lifetime = payload.get("expires_in")
        seconds = (
            int(lifetime)
            if isinstance(lifetime, int | float | str) and str(lifetime).isdigit()
            else 3600
        )
        # A refresh response may omit the refresh token, meaning "keep the one
        # you have". Dropping it would silently cut the 30-day window short.
        refresh = payload.get("refresh_token")
        if not isinstance(refresh, str) or not refresh:
            refresh = previous.refresh_token if previous else None
        identity = payload.get("id_token")
        if not isinstance(identity, str) or not identity:
            identity = previous.id_token if previous else None
        scope = payload.get("scope")
        return cls(
            access_token=access,
            expires_at=datetime.now(UTC) + timedelta(seconds=seconds),
            refresh_token=refresh,
            id_token=identity,
            scope=scope if isinstance(scope, str) else None,
        )


class TokenStore:
    """Persists tokens on the attendee's machine with owner-only permissions.

    The developer guide notes that deleting a client's configuration does not
    invalidate a token it already holds, so ``clear`` is paired with a real
    revocation in :func:`logout` rather than being treated as sign-out by itself.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> TokenSet | None:
        if not self._path.exists():
            return None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return TokenSet(
                access_token=raw["access_token"],
                expires_at=datetime.fromisoformat(raw["expires_at"]),
                refresh_token=raw.get("refresh_token"),
                id_token=raw.get("id_token"),
                scope=raw.get("scope"),
            )
        except (OSError, ValueError, KeyError):
            # A corrupt token file must not be fatal; treat it as "not signed in".
            return None

    def save(self, tokens: TokenSet) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "access_token": tokens.access_token,
            "expires_at": tokens.expires_at.isoformat(),
            "refresh_token": tokens.refresh_token,
            "id_token": tokens.id_token,
            "scope": tokens.scope,
        }
        # Create with 0600 from the outset rather than widening then narrowing,
        # so the token is never briefly world-readable.
        self._path.touch(mode=0o600, exist_ok=True)
        self._path.chmod(0o600)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> bool:
        if not self._path.exists():
            return False
        self._path.unlink()
        return True


# --------------------------------------------------------------------------- #
# Loopback callback server
# --------------------------------------------------------------------------- #
class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handles the single redirect back from the browser.

    ``result`` is a dict shared with the caller rather than a return value,
    because ``BaseHTTPRequestHandler`` is instantiated by the server, not by us.
    """

    result: dict[str, str] = {}
    expected_path: str = CALLBACK_PATH

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != self.expected_path:
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        for key in ("code", "state", "error", "error_description"):
            value = params.get(key, [""])[0]
            if value:
                self.result[key] = value

        ok = "code" in self.result and "error" not in self.result
        body = _CALLBACK_PAGE_OK if ok else _CALLBACK_PAGE_FAIL
        encoded = body.encode("utf-8")
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: Any) -> None:
        """Silence the default stderr logging; the query string holds the code."""


_CALLBACK_PAGE_OK = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Signed in</title>
<style>body{font-family:system-ui,-apple-system,sans-serif;margin:4rem auto;max-width:34rem;
padding:0 1.5rem;color:#0f1b2d;line-height:1.6}h1{font-size:1.4rem;margin-bottom:.5rem}
code{background:#eef2f7;padding:.15rem .4rem;border-radius:.25rem}</style></head>
<body><h1>Signed in to the AWS Events API</h1>
<p>replan has your access token. You can close this tab and return to your terminal.</p>
<p>To sign out later, run <code>replan logout</code>.</p></body></html>
"""

_CALLBACK_PAGE_FAIL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sign-in failed</title>
<style>body{font-family:system-ui,-apple-system,sans-serif;margin:4rem auto;max-width:34rem;
padding:0 1.5rem;color:#0f1b2d;line-height:1.6}h1{font-size:1.4rem;margin-bottom:.5rem}</style>
</head><body><h1>Sign-in did not complete</h1>
<p>Return to your terminal; replan printed the reason there.</p></body></html>
"""


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
        return True


def first_free_callback_port() -> int:
    """Pick a registered port that is actually free.

    Only the six registered ports can ever work, so if all are busy we say so
    plainly rather than falling back to a random port that is guaranteed to fail
    with an opaque ``redirect_mismatch``.
    """
    for port in ALLOWED_CALLBACK_PORTS:
        if port_is_free(port):
            return port
    allowed = ", ".join(str(item) for item in ALLOWED_CALLBACK_PORTS)
    msg = (
        f"All registered callback ports are in use ({allowed}). "
        "Free one of them and try again; no other port can complete sign-in."
    )
    raise AuthError(msg)


# --------------------------------------------------------------------------- #
# The flow
# --------------------------------------------------------------------------- #
def build_authorize_url(settings: Settings, pkce: PkcePair) -> str:
    query = {
        "response_type": "code",
        "client_id": settings.client_id,
        "redirect_uri": settings.redirect_uri,
        "scope": " ".join(OAUTH_SCOPES),
        "identity_provider": OAUTH_IDENTITY_PROVIDER,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "state": pkce.state,
    }
    return f"{OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(query, quote_via=urllib.parse.quote)}"


def _post_token(settings: Settings, form: dict[str, str]) -> dict[str, Any]:
    with httpx.Client(timeout=settings.request_timeout_seconds) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        # OAuth error codes are safe to surface; the form values are not.
        code = payload.get("error") or f"HTTP {response.status_code}"
        detail = payload.get("error_description") or ""
        msg = f"token endpoint refused the exchange: {code} {detail}".strip()
        raise AuthError(msg)
    if not isinstance(payload, dict):
        msg = "token endpoint returned a non-object body"
        raise AuthError(msg)
    return payload


def login(
    settings: Settings,
    *,
    open_browser: bool = True,
    timeout_seconds: int = _CALLBACK_TIMEOUT_SECONDS,
    on_url: Any = None,
) -> TokenSet:
    """Run the interactive sign-in and return a fresh token set.

    Binds to 127.0.0.1 only. The server handles exactly one request and then
    stops, so nothing is left listening after sign-in.
    """
    pkce = PkcePair.generate()
    authorize_url = build_authorize_url(settings, pkce)
    result: dict[str, str] = {}

    class BoundCallbackHandler(_CallbackHandler):
        pass

    BoundCallbackHandler.result = result
    BoundCallbackHandler.expected_path = urllib.parse.urlparse(settings.redirect_uri).path
    handler = BoundCallbackHandler

    try:
        server = http.server.HTTPServer(("127.0.0.1", settings.callback_port), handler)
    except OSError as error:
        msg = (
            f"cannot listen on 127.0.0.1:{settings.callback_port}. "
            "That port must be free, and only the registered ports can be used."
        )
        raise AuthError(msg) from error

    server.timeout = timeout_seconds
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    if on_url is not None:
        on_url(authorize_url)
    if open_browser:
        webbrowser.open(authorize_url)

    thread.join(timeout=timeout_seconds + 1)
    server.server_close()

    if not result:
        msg = (
            f"no callback arrived within {timeout_seconds}s. "
            "Open the sign-in URL manually if your browser did not launch."
        )
        raise AuthError(msg)
    if "error" in result:
        detail = result.get("error_description", "")
        hint = ""
        if result["error"] == "redirect_mismatch":
            hint = " The redirect URI must match exactly, including port and the /callback path."
        msg = f"authorization failed: {result['error']} {detail}{hint}".strip()
        raise AuthError(msg)
    if result.get("state") != pkce.state:
        # Mismatched state means this response is not ours. Refusing it is what
        # stops an attacker from injecting their own authorization code.
        msg = "state mismatch on the OAuth callback; sign-in aborted"
        raise AuthError(msg)

    payload = _post_token(
        settings,
        {
            "grant_type": "authorization_code",
            "client_id": settings.client_id,
            "redirect_uri": settings.redirect_uri,
            "code": result["code"],
            "code_verifier": pkce.verifier,
        },
    )
    return TokenSet.from_response(payload)


def refresh(settings: Settings, tokens: TokenSet) -> TokenSet:
    """Exchange a refresh token for a new access token."""
    if not tokens.refresh_token:
        msg = "no refresh token stored; run `replan login`"
        raise NotSignedInError(msg)
    payload = _post_token(
        settings,
        {
            "grant_type": "refresh_token",
            "client_id": settings.client_id,
            "refresh_token": tokens.refresh_token,
        },
    )
    return TokenSet.from_response(payload, previous=tokens)


def revoke(settings: Settings, tokens: TokenSet) -> bool:
    """Revoke the refresh token at the revocation endpoint.

    This is the part that actually signs the attendee out. Deleting the local
    file alone leaves a working token behind, which the guide calls out
    explicitly.
    """
    target = tokens.refresh_token
    if not target:
        return False
    try:
        with httpx.Client(timeout=settings.request_timeout_seconds) as client:
            response = client.post(
                OAUTH_REVOKE_URL,
                data={"token": target, "client_id": settings.client_id},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.HTTPError:
        return False
    return response.status_code < 400


class TokenProvider:
    """Supplies a valid bearer token, refreshing transparently.

    Used by the REST client so no call site has to think about expiry. Read-only
    operations on public events never touch this at all.
    """

    def __init__(self, settings: Settings, store: TokenStore | None = None) -> None:
        self._settings = settings
        self._store = store or TokenStore(settings.token_path)
        self._cached: TokenSet | None = None

    @property
    def store(self) -> TokenStore:
        return self._store

    def peek(self) -> TokenSet | None:
        """Current token without refreshing. For status reporting."""
        if self._cached is None:
            self._cached = self._store.load()
        return self._cached

    def is_signed_in(self) -> bool:
        tokens = self.peek()
        return tokens is not None and (not tokens.is_expired or tokens.refresh_token is not None)

    def access_token(self) -> str:
        tokens = self.peek()
        if tokens is None:
            msg = "not signed in; run `replan login`"
            raise NotSignedInError(msg)
        if tokens.needs_refresh:
            if not tokens.refresh_token:
                msg = "access token expired and no refresh token is stored; run `replan login`"
                raise NotSignedInError(msg)
            tokens = refresh(self._settings, tokens)
            self._store.save(tokens)
            self._cached = tokens
        return tokens.access_token

    def remember(self, tokens: TokenSet) -> None:
        self._store.save(tokens)
        self._cached = tokens

    def forget(self) -> bool:
        self._cached = None
        return self._store.clear()


__all__ = [
    "AuthError",
    "NotSignedInError",
    "PkcePair",
    "TokenProvider",
    "TokenSet",
    "TokenStore",
    "build_authorize_url",
    "first_free_callback_port",
    "login",
    "port_is_free",
    "refresh",
    "revoke",
]
