"""Telling someone a seat opened.

A watcher that finds a seat and says nothing is useless, so notification is a first-class
part of this rather than a print statement. Three sinks, behind one protocol: the console
for a local run, a Slack webhook for a team channel, and SNS for anything running on
Lambda where there is no terminal to print to.

Two rules throughout: a notifier never raises, because failing to send a message must not
kill a watcher that has been running for two days, and no secret is ever logged - the
webhook URL and topic ARN are referenced but never echoed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from replan.domain import REINVENT_TZ, Session, now_utc


class NotificationKind(StrEnum):
    SEAT_OPEN = "seat-open"
    RESERVED = "reserved"
    NEEDS_ATTENTION = "needs-attention"
    WATCHER_STARTED = "watcher-started"
    WATCHER_STOPPED = "watcher-stopped"
    ERROR = "error"

    @property
    def emoji(self) -> str:
        return {
            NotificationKind.SEAT_OPEN: "\N{BELL}",
            NotificationKind.RESERVED: "\N{WHITE HEAVY CHECK MARK}",
            NotificationKind.NEEDS_ATTENTION: "\N{WARNING SIGN}",
            NotificationKind.WATCHER_STARTED: "\N{EYES}",
            NotificationKind.WATCHER_STOPPED: "\N{OCTAGONAL SIGN}",
            NotificationKind.ERROR: "\N{CROSS MARK}",
        }[self]


@dataclass(frozen=True, slots=True)
class Notification:
    """One thing worth telling the attendee."""

    kind: NotificationKind
    title: str
    body: str = ""
    session: Session | None = None
    event_id: str | None = None
    at: datetime = field(default_factory=now_utc)
    zone: ZoneInfo = REINVENT_TZ

    def as_text(self) -> str:
        lines = [f"{self.kind.emoji} {self.title}"]
        if self.session is not None and self.session.span is not None:
            lines.append(
                f"   {self.session.span.format_in(self.session.zone)}  {self.session.room.display}"
            )
        if self.body:
            lines.append(f"   {self.body}")
        return "\n".join(lines)

    def as_slack_blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{self.kind.emoji} *{self.title}*"},
            }
        ]
        details: list[str] = []
        if self.session is not None:
            session = self.session
            if session.span is not None:
                details.append(f"*When* {session.span.format_in(session.zone)}")
            details.append(f"*Where* {session.room.display}")
            if session.level_raw:
                details.append(f"*Level* {session.level_raw}")
            if session.seat_availability is not None:
                details.append(f"*Seats* {session.seat_availability.label}")
        if self.body:
            details.append(self.body)
        if details:
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(details)}}
            )
        return blocks


@runtime_checkable
class Notifier(Protocol):
    """Somewhere a notification can go."""

    @property
    def name(self) -> str: ...

    def send(self, notification: Notification) -> bool:
        """Deliver it. Returns success, and never raises."""
        ...


class ConsoleNotifier:
    """Writes to stdout. The default, and the only one that needs no configuration."""

    name = "console"

    def __init__(self, writer: Any = None) -> None:
        self._writer = writer

    def send(self, notification: Notification) -> bool:
        text = notification.as_text()
        if self._writer is not None:
            self._writer(text)
        else:
            print(text)  # noqa: T201 - this notifier's entire job
        return True


class SlackNotifier:
    """Posts to an incoming webhook.

    The URL is a secret: it is held, used, and never logged or included in an error
    message.
    """

    name = "slack"

    def __init__(self, webhook_url: str, *, timeout: float = 10.0) -> None:
        self._url = webhook_url
        self._timeout = timeout
        self.last_error: str | None = None

    def send(self, notification: Notification) -> bool:
        import httpx

        payload = {
            "text": notification.title,
            "blocks": notification.as_slack_blocks(),
        }
        try:
            response = httpx.post(self._url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as error:
            # Deliberately reports the exception type only. The URL is embedded in
            # many httpx error messages, and it is a credential.
            self.last_error = f"slack delivery failed: {type(error).__name__}"
            return False
        if response.status_code >= 400:
            self.last_error = f"slack returned HTTP {response.status_code}"
            return False
        self.last_error = None
        return True


class SnsNotifier:
    """Publishes to an SNS topic, for a watcher running on Lambda."""

    name = "sns"

    def __init__(self, topic_arn: str, *, region: str | None = None) -> None:
        self._topic_arn = topic_arn
        self._region = region
        self.last_error: str | None = None

    def send(self, notification: Notification) -> bool:
        try:
            import boto3
        except ImportError:
            self.last_error = "boto3 is not installed; install it with `uv sync --extra aws`"
            return False
        try:
            client = boto3.client("sns", region_name=self._region)
            client.publish(
                TopicArn=self._topic_arn,
                Subject=notification.title[:99],
                Message=notification.as_text(),
                MessageAttributes={
                    "kind": {"DataType": "String", "StringValue": notification.kind.value}
                },
            )
        except Exception as error:  # a notifier must never crash a long-running watcher
            self.last_error = f"sns publish failed: {type(error).__name__}"
            return False
        self.last_error = None
        return True


class MultiNotifier:
    """Fans out to several sinks.

    Succeeds if *any* sink accepted the message. One dead webhook should not mean the
    attendee hears nothing when the console was right there.
    """

    name = "multi"

    def __init__(self, *notifiers: Notifier) -> None:
        self._notifiers = tuple(notifiers)

    @property
    def sinks(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._notifiers)

    def send(self, notification: Notification) -> bool:
        results = [item.send(notification) for item in self._notifiers]
        return any(results) if results else False


def build_notifier(
    *,
    console: bool = True,
    slack_webhook_url: str | None = None,
    sns_topic_arn: str | None = None,
    aws_region: str | None = None,
    writer: Any = None,
) -> Notifier:
    """Assemble the notifier from whatever is configured."""
    sinks: list[Notifier] = []
    if console:
        sinks.append(ConsoleNotifier(writer))
    if slack_webhook_url:
        sinks.append(SlackNotifier(slack_webhook_url))
    if sns_topic_arn:
        sinks.append(SnsNotifier(sns_topic_arn, region=aws_region))
    if len(sinks) == 1:
        return sinks[0]
    return MultiNotifier(*sinks)


def seat_open_notification(session: Session, *, event_id: str, reason: str) -> Notification:
    return Notification(
        kind=NotificationKind.SEAT_OPEN,
        title=f"Seat available: {session.code} {session.title}",
        body=reason,
        session=session,
        event_id=event_id,
    )


def reserved_notification(session: Session, *, event_id: str) -> Notification:
    return Notification(
        kind=NotificationKind.RESERVED,
        title=f"Reserved: {session.code} {session.title}",
        body="replan took the seat for you. Cancel it in the app if you changed your mind.",
        session=session,
        event_id=event_id,
    )


def as_json(notification: Notification) -> str:
    return json.dumps(
        {
            "kind": notification.kind.value,
            "title": notification.title,
            "body": notification.body,
            "eventId": notification.event_id,
            "sessionId": notification.session.session_id if notification.session else None,
            "at": notification.at.isoformat(),
        }
    )


__all__ = [
    "ConsoleNotifier",
    "MultiNotifier",
    "Notification",
    "NotificationKind",
    "Notifier",
    "SlackNotifier",
    "SnsNotifier",
    "as_json",
    "build_notifier",
    "reserved_notification",
    "seat_open_notification",
]
