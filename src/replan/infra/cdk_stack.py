"""AWS CloudFormation & CDK stack definition for replan serverless seat watcher.

Provisions:
- DynamoDB Table for watch state, idempotency tokens, and audit journals
- AWS Secrets Manager Secret for secure token storage (AWS Builder ID Bearer Token)
- Amazon SNS Topic for real-time seat opening alerts (SMS, Email, Slack)
- AWS Lambda Function running the watcher cycle against the live AWS Events API
- Amazon EventBridge Scheduler Rule triggering the Lambda every 2 minutes
- AWS Lambda Function URL for HTTPS health and status probes
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplanStackConfig:
    """Deployment configuration for replan serverless stack."""

    stack_name: str = "replan-serverless"
    api_base_url: str = "https://api.awsevents.com"
    event_id: str = "reinvent2026"
    schedule_expression: str = "rate(2 minutes)"
    memory_size: int = 256
    timeout_seconds: int = 60
    notification_email: str | None = None


class ReplanServerlessStack:
    """Generates AWS CloudFormation resources for replan serverless infrastructure."""

    def __init__(self, config: ReplanStackConfig | None = None) -> None:
        self.config = config or ReplanStackConfig()

    def to_dict(self) -> dict[str, Any]:
        """Produce the complete CloudFormation template specification as a dict."""
        resources: dict[str, Any] = {
            # 1. DynamoDB Table for watch store and sync audit journal
            "ReplanStateTable": {
                "Type": "AWS::DynamoDB::Table",
                "Properties": {
                    "TableName": f"{self.config.stack_name}-state",
                    "BillingMode": "PAY_PER_REQUEST",
                    "AttributeDefinitions": [
                        {"AttributeName": "PK", "AttributeType": "S"},
                        {"AttributeName": "SK", "AttributeType": "S"},
                    ],
                    "KeySchema": [
                        {"AttributeName": "PK", "KeyType": "HASH"},
                        {"AttributeName": "SK", "KeyType": "RANGE"},
                    ],
                    "TimeToLiveSpecification": {
                        "AttributeName": "ttl",
                        "Enabled": True,
                    },
                    "Tags": [
                        {"Key": "Project", "Value": "replan"},
                        {"Key": "Event", "Value": self.config.event_id},
                    ],
                },
            },
            # 2. Secrets Manager Secret for AWS Builder ID attendee token
            "ReplanTokenSecret": {
                "Type": "AWS::SecretsManager::Secret",
                "Properties": {
                    "Name": f"{self.config.stack_name}-attendee-token",
                    "Description": "OAuth token for AWS Events API attendee account",
                    "Tags": [{"Key": "Project", "Value": "replan"}],
                },
            },
            # 3. Amazon SNS Topic for Seat Opening Notifications
            "ReplanSeatAlertsTopic": {
                "Type": "AWS::SNS::Topic",
                "Properties": {
                    "TopicName": f"{self.config.stack_name}-seat-alerts",
                    "DisplayName": "re:Plan 2026 Seat Alerts",
                },
            },
            # 4. IAM Role for Lambda Function
            "ReplanWatcherRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": f"{self.config.stack_name}-watcher-role",
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    },
                    "ManagedPolicyArns": [
                        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                    ],
                    "Policies": [
                        {
                            "PolicyName": "ReplanPermissions",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "dynamodb:GetItem",
                                            "dynamodb:PutItem",
                                            "dynamodb:UpdateItem",
                                            "dynamodb:Query",
                                        ],
                                        "Resource": {"Fn::GetAtt": ["ReplanStateTable", "Arn"]},
                                    },
                                    {
                                        "Effect": "Allow",
                                        "Action": ["secretsmanager:GetSecretValue"],
                                        "Resource": {"Ref": "ReplanTokenSecret"},
                                    },
                                    {
                                        "Effect": "Allow",
                                        "Action": ["sns:Publish"],
                                        "Resource": {"Ref": "ReplanSeatAlertsTopic"},
                                    },
                                ],
                            },
                        }
                    ],
                },
            },
            # 5. AWS Lambda Function
            "ReplanWatcherFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": f"{self.config.stack_name}-watcher",
                    "Runtime": "python3.12",
                    "Handler": "infra.handler.lambda_handler",
                    "MemorySize": self.config.memory_size,
                    "Timeout": self.config.timeout_seconds,
                    "Role": {"Fn::GetAtt": ["ReplanWatcherRole", "Arn"]},
                    "Environment": {
                        "Variables": {
                            "REPLAN_API_BASE_URL": self.config.api_base_url,
                            "REPLAN_EVENT_ID": self.config.event_id,
                            "REPLAN_DYNAMO_TABLE": {"Ref": "ReplanStateTable"},
                            "REPLAN_SNS_TOPIC_ARN": {"Ref": "ReplanSeatAlertsTopic"},
                        }
                    },
                    "Code": {
                        "ZipFile": (
                            "# AWS Lambda Code Placeholder\n"
                            "from infra.handler import lambda_handler\n"
                        )
                    },
                },
            },
            # 6. Lambda Function URL (HTTPS endpoint for status / health checks)
            "ReplanFunctionUrl": {
                "Type": "AWS::Lambda::Url",
                "Properties": {
                    "AuthType": "NONE",
                    "TargetFunctionArn": {"Fn::GetAtt": ["ReplanWatcherFunction", "Arn"]},
                },
            },
            "ReplanFunctionUrlPermission": {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "FunctionName": {"Ref": "ReplanWatcherFunction"},
                    "Action": "lambda:InvokeFunctionUrl",
                    "Principal": "*",
                    "FunctionUrlAuthType": "NONE",
                },
            },
            # 7. EventBridge Scheduler Rule (Polls every 2 minutes)
            "ReplanWatcherScheduleRule": {
                "Type": "AWS::Events::Rule",
                "Properties": {
                    "Name": f"{self.config.stack_name}-schedule",
                    "Description": "Periodic trigger for replan seat watcher",
                    "ScheduleExpression": self.config.schedule_expression,
                    "State": "ENABLED",
                    "Targets": [
                        {
                            "Arn": {"Fn::GetAtt": ["ReplanWatcherFunction", "Arn"]},
                            "Id": "ReplanWatcherTarget",
                        }
                    ],
                },
            },
            "ReplanSchedulePermission": {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "FunctionName": {"Ref": "ReplanWatcherFunction"},
                    "Action": "lambda:InvokeFunction",
                    "Principal": "events.amazonaws.com",
                    "SourceArn": {"Fn::GetAtt": ["ReplanWatcherScheduleRule", "Arn"]},
                },
            },
        }

        # Optional Email Subscription for SNS
        if self.config.notification_email:
            resources["ReplanEmailSubscription"] = {
                "Type": "AWS::SNS::Subscription",
                "Properties": {
                    "Protocol": "email",
                    "TopicArn": {"Ref": "ReplanSeatAlertsTopic"},
                    "Endpoint": self.config.notification_email,
                },
            }

        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "re:Plan 2026 Serverless Seat Watcher and Agenda Intelligence Stack",
            "Resources": resources,
            "Outputs": {
                "FunctionUrl": {
                    "Description": "HTTPS URL for status checks and probe invocations",
                    "Value": {"Fn::GetAtt": ["ReplanFunctionUrl", "FunctionUrl"]},
                },
                "TopicArn": {
                    "Description": "Amazon SNS Topic ARN for seat alerts",
                    "Value": {"Ref": "ReplanSeatAlertsTopic"},
                },
                "TableName": {
                    "Description": "DynamoDB State and Journal Table",
                    "Value": {"Ref": "ReplanStateTable"},
                },
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize template to formatted CloudFormation JSON."""
        return json.dumps(self.to_dict(), indent=indent)


__all__ = ["ReplanServerlessStack", "ReplanStackConfig"]
