"""Bedrock client factory (scaffold)."""

import boto3
from botocore.config import Config


def get_bedrock_client(region_name: str | None = None):
    cfg = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=60)
    if region_name:
        return boto3.client("bedrock-runtime", region_name=region_name, config=cfg)
    return boto3.client("bedrock-runtime", config=cfg)


