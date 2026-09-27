"""AWS Lambda entry point for serverless seat watcher and status probes.

Runs on an EventBridge schedule (e.g. every 2 minutes) or via Function URL.
Connects directly to the live AWS Events API (https://api.awsevents.com/v1) to monitor
watched sessions, detect seat band transitions, publish Amazon SNS alerts, and optionally
auto-reserve newly opened seats.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from replan.adapters.rest import RestCatalogClient
from replan.config import Settings
from replan.domain import SeatAvailability

logger = logging.getLogger(__name__)


def _get_watched_sessions(event: dict[str, Any]) -> list[str]:
    """Extract session codes/ids from invocation payload or environment."""
    if "sessions" in event and isinstance(event["sessions"], list):
        return [str(s) for s in event["sessions"]]

    env_sessions = os.environ.get("REPLAN_WATCH_SESSIONS", "")
    if env_sessions.strip():
        return [s.strip() for s in env_sessions.split(",") if s.strip()]

    return []


def _publish_sns_alert(
    topic_arn: str,
    subject: str,
    message: str,
) -> bool:
    """Publish seat opening alert to Amazon SNS."""
    try:
        import boto3

        client = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "us-west-2"))
        client.publish(
            TopicArn=topic_arn,
            Subject=subject[:100],
            Message=message,
        )
        return True
    except Exception as err:
        logger.warning("Failed to publish SNS notification: %s", err)
        return False


def lambda_handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
    """AWS Lambda entry point for seat watcher and status endpoint."""
    # Check if this is an HTTP probe via Function URL or API Gateway
    http_method = event.get("requestContext", {}).get("http", {}).get("method") or event.get(
        "httpMethod"
    )
    raw_path = event.get("rawPath") or event.get("path", "")

    settings = Settings.from_env()
    client = RestCatalogClient(settings)

    try:
        # Route 1: Health / Reachability check
        if http_method == "GET" and "/health" in raw_path:
            probe_result = client.probe()
            return {
                "statusCode": 200 if probe_result.get("reachable") else 503,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(probe_result),
            }

        # Route 2: Scheduled watch poll cycle
        watched_sessions = _get_watched_sessions(event)
        event_id = event.get("event_id") or settings.event_id

        if not watched_sessions:
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "status": "idle",
                        "event_id": event_id,
                        "message": "No sessions configured to watch. Set REPLAN_WATCH_SESSIONS.",
                        "checked_count": 0,
                        "openings_found": 0,
                    }
                ),
            }

        openings: list[dict[str, Any]] = []
        checked: list[dict[str, Any]] = []

        auto_reserve = event.get("auto_reserve") or (
            os.environ.get("REPLAN_AUTO_RESERVE", "").lower() in ("true", "1", "yes")
        )
        sns_arn = os.environ.get("REPLAN_SNS_TOPIC_ARN") or settings.sns_topic_arn

        for session_token in watched_sessions:
            try:
                session = client.get_session(event_id, session_token)
                availability = session.seat_availability
                is_open = availability is not None and availability not in (
                    SeatAvailability.UNAVAILABLE,
                    SeatAvailability.WALK_UP,
                )

                checked_info = {
                    "session_id": session.session_id,
                    "code": session.code,
                    "title": session.title,
                    "availability": availability.value if availability else "unknown",
                    "open": is_open,
                }
                checked.append(checked_info)

                if is_open:
                    openings.append(checked_info)
                    if sns_arn:
                        has_venue = session.room and session.room.venue
                        venue_str = session.room.venue if has_venue else "Unknown"
                        room_str = session.room.display if session.room else "Unknown"
                        _publish_sns_alert(
                            topic_arn=sns_arn,
                            subject=f"Seat Opened: {session.code} ({session.title})",
                            message=(
                                f"A seat has opened up for {session.code}: {session.title}!\n\n"
                                f"Venue: {venue_str}\n"
                                f"Room: {room_str}\n"
                                f"Seats status: {availability.value if availability else 'open'}\n"
                            ),
                        )

                    # Auto-reserve if attendee has active credentials
                    if auto_reserve:
                        try:
                            client.reserve_sessions(event_id, [session.session_id])
                            checked_info["reserved"] = True
                        except Exception as reserve_err:
                            checked_info["reservation_error"] = str(reserve_err)

            except Exception as item_err:
                checked.append(
                    {
                        "session_token": session_token,
                        "error": str(item_err),
                    }
                )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "status": "completed",
                    "event_id": event_id,
                    "checked_count": len(checked),
                    "openings_found": len(openings),
                    "openings": openings,
                    "details": checked,
                }
            ),
        }

    finally:
        client.close()


__all__ = ["lambda_handler"]
