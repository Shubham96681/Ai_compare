"""
Configuration module for model evaluation framework.
Define models, pricing, and evaluation settings here.
"""

from typing import Dict, Any

# AWS Configuration
AWS_PROFILE = "bells-dev"  # Used if AWS_ACCESS_KEY_ID not set
AWS_REGION = "us-east-2"

# AWS Credentials (if using direct credentials instead of profile)
# Leave as None to use AWS_PROFILE or default credentials
# TODO: Replace with your AWS credentials
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID_HERE"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY_HERE"

# Model IDs for Bedrock
MODELS: Dict[str, Dict[str, Any]] = {
    "claude-sonnet": {
        "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "provider": "anthropic",
        "name": "Claude 3.7 Sonnet",
    },
    "llama-3-2-11b": {
        "model_id": "us.meta.llama3-2-11b-instruct-v1:0",
        "provider": "meta",
        "name": "Llama 3.2 11B Instruct",
    },
    # Add more models here as needed
    # "claude-haiku": {
    #     "model_id": "us.anthropic.claude-3-5-haiku-20241022-v2:0",
    #     "provider": "anthropic",
    #     "name": "Claude 3.5 Haiku",
    # },
}

# Pricing per 1K tokens (input/output) in USD
# Source: AWS Bedrock pricing as of 2025
PRICING: Dict[str, Dict[str, float]] = {
    "claude-sonnet": {"input": 0.008, "output": 0.024},
    "llama-3-2-11b": {"input": 0.0006, "output": 0.0008},
    # Add pricing for other models
}

# Evaluation Settings
EVAL_SETTINGS = {
    "max_tokens": 1500,
    "temperature": 0.7,
    "max_retries": 2,
    "json_retry_prompt_prefix": "Return ONLY valid JSON. No extra text.\n",
    # Enhanced prompt prefix for Llama models to encourage JSON output
    "llama_json_prompt_suffix": "\n\nIMPORTANT: Return ONLY valid JSON. No markdown formatting, no code blocks, no explanations. Just the raw JSON array or object.",
}

# Prompt Loading Settings
PROMPT_SETTINGS = {
    "local_path": "20251001T000153731Z_e9c5e90710a8738a.json",  # Default: included sample file
    "s3_bucket": None,   # Set to bucket name if using S3
    "s3_key": None,      # Set to S3 key/prefix if using S3
    "csv_column": "prompt",  # Column name in CSV containing prompts
}

