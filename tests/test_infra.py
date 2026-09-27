"""Unit tests for Task 14: AWS Serverless deployment infrastructure and Lambda handler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from replan.infra.cdk_stack import ReplanServerlessStack, ReplanStackConfig
from replan.infra.handler import _get_watched_sessions, _publish_sns_alert, lambda_handler


def test_cdk_stack_template_synthesis() -> None:
    """ReplanServerlessStack generates valid CloudFormation JSON with required resources."""
    config = ReplanStackConfig(
        stack_name="replan-test",
        event_id="reinvent2026",
        schedule_expression="rate(1 minute)",
        notification_email="attendee@example.com",
    )
    stack = ReplanServerlessStack(config)
    template = stack.to_dict()

    assert template["AWSTemplateFormatVersion"] == "2010-09-09"
    resources = template["Resources"]

    # Verify Core Serverless Resources
    assert "ReplanStateTable" in resources
    assert resources["ReplanStateTable"]["Type"] == "AWS::DynamoDB::Table"
    assert resources["ReplanStateTable"]["Properties"]["BillingMode"] == "PAY_PER_REQUEST"

    assert "ReplanTokenSecret" in resources
    assert resources["ReplanTokenSecret"]["Type"] == "AWS::SecretsManager::Secret"

    assert "ReplanSeatAlertsTopic" in resources
    assert resources["ReplanSeatAlertsTopic"]["Type"] == "AWS::SNS::Topic"

    assert "ReplanWatcherFunction" in resources
    assert resources["ReplanWatcherFunction"]["Type"] == "AWS::Lambda::Function"
    assert (
        resources["ReplanWatcherFunction"]["Properties"]["Environment"]["Variables"][
            "REPLAN_EVENT_ID"
        ]
        == "reinvent2026"
    )

    assert "ReplanWatcherScheduleRule" in resources
    assert resources["ReplanWatcherScheduleRule"]["Type"] == "AWS::Events::Rule"
    assert (
        resources["ReplanWatcherScheduleRule"]["Properties"]["ScheduleExpression"]
        == "rate(1 minute)"
    )

    assert "ReplanFunctionUrl" in resources
    assert resources["ReplanFunctionUrl"]["Type"] == "AWS::Lambda::Url"

    assert "ReplanEmailSubscription" in resources
    assert (
        resources["ReplanEmailSubscription"]["Properties"]["Endpoint"]
        == "attendee@example.com"
    )

    # Verify Outputs
    outputs = template["Outputs"]
    assert "FunctionUrl" in outputs
    assert "TopicArn" in outputs
    assert "TableName" in outputs

    # Verify JSON serialization
    raw_json = stack.to_json()
    assert len(raw_json) > 500
    parsed = json.loads(raw_json)
    assert parsed["AWSTemplateFormatVersion"] == "2010-09-09"


def test_get_watched_sessions_extraction() -> None:
    """_get_watched_sessions parses from payload and environment variables."""
    # From payload
    assert _get_watched_sessions({"sessions": ["AIM101", "CMP202"]}) == ["AIM101", "CMP202"]

    # From environment
    with patch.dict("os.environ", {"REPLAN_WATCH_SESSIONS": "SEC301, DAT305"}):
        assert _get_watched_sessions({}) == ["SEC301", "DAT305"]

    # Empty
    with patch.dict("os.environ", {"REPLAN_WATCH_SESSIONS": ""}):
        assert _get_watched_sessions({}) == []


def test_publish_sns_alert_success_and_failure() -> None:
    """_publish_sns_alert sends messages via boto3 and handles exceptions gracefully."""
    mock_boto = MagicMock()
    mock_sns = MagicMock()
    mock_boto.client.return_value = mock_sns

    with patch.dict("sys.modules", {"boto3": mock_boto}):
        success = _publish_sns_alert("arn:aws:sns:topic", "Test Subject", "Test Body")
        assert success is True
        mock_sns.publish.assert_called_once_with(
            TopicArn="arn:aws:sns:topic",
            Subject="Test Subject",
            Message="Test Body",
        )

    # When exception raised
    mock_sns.publish.side_effect = RuntimeError("SNS error")
    with patch.dict("sys.modules", {"boto3": mock_boto}):
        failed = _publish_sns_alert("arn:aws:sns:topic", "Test", "Body")
        assert failed is False


def test_lambda_handler_health_check() -> None:
    """lambda_handler responds to GET /health with live probe status."""
    event: dict[str, Any] = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/health",
    }
    response = lambda_handler(event)
    assert response["statusCode"] in (200, 503)
    assert "application/json" in response["headers"]["Content-Type"]
    body = json.loads(response["body"])
    assert "api_root" in body


def test_lambda_handler_idle_when_no_sessions() -> None:
    """lambda_handler returns idle status when no watch targets exist."""
    event: dict[str, Any] = {"sessions": []}
    with patch.dict("os.environ", {"REPLAN_WATCH_SESSIONS": ""}):
        response = lambda_handler(event)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "idle"
        assert body["checked_count"] == 0


def test_lambda_handler_polls_live_session() -> None:
    """lambda_handler polls a live session against public Summit-Dubai-2026 event."""
    event: dict[str, Any] = {
        "event_id": "Summit-Dubai-2026",
        "sessions": ["AIM101"],
    }
    response = lambda_handler(event)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["status"] == "completed"
    assert body["checked_count"] == 1
    details = body["details"]
    assert len(details) == 1
    assert details[0]["code"] == "AIM101"
