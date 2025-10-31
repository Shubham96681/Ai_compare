"""Evaluator core: runs prompts against Bedrock models and collects metrics."""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from src.utils.bedrock_client import get_bedrock_client
from src.utils.timing import Stopwatch
from src.utils.json_utils import is_valid_json
from src.tokenizers import count_tokens
from src.model_registry import ModelRegistry


class BedrockEvaluator:
    """Evaluates prompts against Bedrock models and collects performance metrics."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        region_name: Optional[str] = None,
        max_retries: int = 3
    ):
        self.model_registry = model_registry
        self.region_name = region_name or model_registry.region_name
        self.bedrock_client = get_bedrock_client(self.region_name)
        self.max_retries = max_retries
    
    def evaluate_prompt(
        self,
        prompt: str,
        model: Dict[str, Any],
        prompt_id: Optional[int] = None,
        expected_json: bool = False,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt against a model.
        
        Args:
            prompt: The prompt text to evaluate
            model: Model configuration dictionary
            prompt_id: Optional prompt identifier
            expected_json: Whether JSON response is expected
            run_id: Optional run identifier for grouping
        
        Returns:
            Dictionary with evaluation metrics
        """
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        
        model_name = model.get("name", "unknown")
        model_id = model.get("bedrock_model_id", "unknown")
        provider = model.get("provider", "").lower()
        tokenizer_type = model.get("tokenizer", "heuristic")
        
        # Count input tokens
        input_tokens = count_tokens(tokenizer_type, prompt)
        
        # Prepare generation parameters
        gen_params = self.model_registry.get_generation_params(model)
        
        # Initialize metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "model_name": model_name,
            "model_id": model_id,
            "prompt_id": prompt_id,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "latency_ms": 0,
            "json_valid": False,
            "error": None,
            "status": "success",
            "cost_usd_input": 0.0,
            "cost_usd_output": 0.0,
            "cost_usd_total": 0.0,
        }
        
        # Make API call with timing
        try:
            with Stopwatch() as timer:
                response_text, output_tokens_actual = self._invoke_model(
                    prompt, model, provider, gen_params
                )
            
            metrics["latency_ms"] = timer.elapsed_ms
            metrics["output_tokens"] = output_tokens_actual
            metrics["response"] = response_text[:500] if len(response_text) > 500 else response_text  # Truncate for storage
            
            # Validate JSON if expected
            if expected_json:
                is_valid, _ = is_valid_json(response_text)
                metrics["json_valid"] = is_valid
            
            # Calculate costs
            pricing = self.model_registry.get_model_pricing(model)
            input_cost = (input_tokens / 1000.0) * pricing["input_per_1k_tokens_usd"]
            output_cost = (output_tokens_actual / 1000.0) * pricing["output_per_1k_tokens_usd"]
            
            metrics["cost_usd_input"] = round(input_cost, 6)
            metrics["cost_usd_output"] = round(output_cost, 6)
            metrics["cost_usd_total"] = round(input_cost + output_cost, 6)
            
        except Exception as e:
            metrics["status"] = "error"
            metrics["error"] = str(e)
            metrics["latency_ms"] = timer.elapsed_ms if 'timer' in locals() else 0
        
        return metrics
    
    def _invoke_model(
        self,
        prompt: str,
        model: Dict[str, Any],
        provider: str,
        gen_params: Dict[str, Any]
    ) -> Tuple[str, int]:
        """
        Invoke Bedrock model and return response text and token count.
        
        Returns:
            Tuple of (response_text, output_tokens)
        """
        model_id = model.get("bedrock_model_id")
        tokenizer_type = model.get("tokenizer", "heuristic")
        
        # Use Converse API for Anthropic Claude models
        if provider == "anthropic" or "claude" in model_id.lower():
            return self._invoke_converse(prompt, model_id, gen_params, tokenizer_type)
        
        # Use InvokeModel for other models (Llama, Titan, etc.)
        return self._invoke_model_direct(prompt, model_id, provider, gen_params, tokenizer_type)
    
    def _invoke_converse(
        self,
        prompt: str,
        model_id: str,
        gen_params: Dict[str, Any],
        tokenizer_type: str
    ) -> Tuple[str, int]:
        """Invoke Anthropic Claude using Converse API."""
        try:
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "maxTokens": gen_params.get("max_tokens", 512),
                "temperature": gen_params.get("temperature", 0.2),
                "topP": gen_params.get("top_p", 0.95)
            }
            
            response = self.bedrock_client.converse(
                modelId=model_id,
                messages=body["messages"],
                inferenceConfig={
                    "maxTokens": body["maxTokens"],
                    "temperature": body["temperature"],
                    "topP": body["topP"]
                }
            )
            
            # Extract response text
            content = response.get("output", {}).get("message", {}).get("content", [])
            response_text = ""
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    response_text += item["text"]
                elif isinstance(item, str):
                    response_text += item
            
            # Get actual token usage if available
            usage = response.get("usage", {})
            output_tokens = usage.get("outputTokens", 0)
            
            # If not available, estimate
            if output_tokens == 0:
                output_tokens = count_tokens(tokenizer_type, response_text)
            
            return response_text, output_tokens
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_msg}")
        except Exception as e:
            raise Exception(f"Failed to invoke model: {str(e)}")
    
    def _invoke_model_direct(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        gen_params: Dict[str, Any],
        tokenizer_type: str
    ) -> Tuple[str, int]:
        """Invoke model using InvokeModel API (for non-Claude models)."""
        try:
            # Prepare request body based on provider
            if provider == "meta" or "llama" in model_id.lower():
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": gen_params.get("max_tokens", 512),
                    "temperature": gen_params.get("temperature", 0.2),
                    "top_p": gen_params.get("top_p", 0.9)
                })
            elif provider == "amazon" or "titan" in model_id.lower():
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": gen_params.get("max_tokens", 512),
                        "temperature": gen_params.get("temperature", 0.2),
                        "topP": gen_params.get("top_p", 0.9)
                    }
                })
            else:
                # Generic format
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens": gen_params.get("max_tokens", 512),
                    "temperature": gen_params.get("temperature", 0.2),
                    "top_p": gen_params.get("top_p", 0.9)
                })
            
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            
            # Extract text based on provider
            if provider == "meta" or "llama" in model_id.lower():
                response_text = response_body.get("generation", "")
            elif provider == "amazon" or "titan" in model_id.lower():
                response_text = response_body.get("results", [{}])[0].get("outputText", "")
            else:
                response_text = response_body.get("completion", "") or response_body.get("generated_text", "")
            
            # Estimate output tokens
            output_tokens = count_tokens(tokenizer_type, response_text)
            
            return response_text, output_tokens
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_msg}")
        except Exception as e:
            raise Exception(f"Failed to invoke model: {str(e)}")
    
    def evaluate_prompts_batch(
        self,
        prompts_df,
        models: List[Dict[str, Any]],
        run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompts against multiple models.
        
        Args:
            prompts_df: DataFrame with columns: prompt_id, prompt, expected_json (optional)
            models: List of model configurations
            run_id: Optional run identifier
        
        Returns:
            List of metrics dictionaries
        """
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        
        all_metrics = []
        
        for _, row in prompts_df.iterrows():
            prompt_id = row.get("prompt_id", None)
            prompt = row.get("prompt", "")
            expected_json = bool(row.get("expected_json", False))
            
            if not prompt:
                continue
            
            # Evaluate against each model
            for model in models:
                metrics = self.evaluate_prompt(
                    prompt=prompt,
                    model=model,
                    prompt_id=prompt_id,
                    expected_json=expected_json,
                    run_id=run_id
                )
                all_metrics.append(metrics)
        
        return all_metrics
