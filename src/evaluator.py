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
        timer = None
        try:
            with Stopwatch() as timer:
                response_text, output_tokens_actual, input_tokens_actual = self._invoke_model(
                    prompt, model, provider, gen_params
                )
            
            metrics["latency_ms"] = timer.elapsed_ms
            metrics["output_tokens"] = output_tokens_actual
            
            # Use actual input tokens from API if available, otherwise use estimated
            if input_tokens_actual > 0:
                metrics["input_tokens"] = input_tokens_actual
                input_tokens = input_tokens_actual  # Use actual for cost calculation
            
            # Store full response for dashboard display (can be long, but needed for JSON output viewing)
            metrics["response"] = response_text  # Store full response
            
            # Always validate JSON to provide useful information
            # Set json_valid based on whether JSON was expected and the actual validation result
            is_valid, _ = is_valid_json(response_text)
            if expected_json:
                # If JSON was expected, use the validation result directly
                metrics["json_valid"] = is_valid
            else:
                # If JSON wasn't expected, always validate but use None to indicate "not applicable"
                # This way we still show useful info (if response happens to be valid JSON)
                metrics["json_valid"] = None  # None = not checked/not applicable
            
            # Calculate costs using actual token counts from API
            pricing = self.model_registry.get_model_pricing(model)
            input_cost = (input_tokens / 1000.0) * pricing["input_per_1k_tokens_usd"]
            output_cost = (output_tokens_actual / 1000.0) * pricing["output_per_1k_tokens_usd"]
            
            metrics["cost_usd_input"] = round(input_cost, 6)
            metrics["cost_usd_output"] = round(output_cost, 6)
            metrics["cost_usd_total"] = round(input_cost + output_cost, 6)
            
        except Exception as e:
            metrics["status"] = "error"
            metrics["error"] = str(e)
            metrics["latency_ms"] = timer.elapsed_ms if timer is not None and 'timer' in locals() else 0
            # Make sure input tokens are captured even on error
            if metrics["input_tokens"] == 0:
                metrics["input_tokens"] = input_tokens
        
        return metrics
    
    def _invoke_model(
        self,
        prompt: str,
        model: Dict[str, Any],
        provider: str,
        gen_params: Dict[str, Any]
    ) -> Tuple[str, int, int]:
        """
        Invoke Bedrock model and return response text and token count.
        
        Returns:
            Tuple of (response_text, output_tokens)
        """
        model_id = model.get("bedrock_model_id")
        tokenizer_type = model.get("tokenizer", "heuristic")
        use_inference_profile = model.get("use_inference_profile", False)
        
        # Use Converse API for Anthropic Claude models
        if provider == "anthropic" or "claude" in model_id.lower():
            return self._invoke_converse(prompt, model_id, gen_params, tokenizer_type, use_inference_profile)
        
        # Use InvokeModel for other models (Llama, Titan, etc.)
        return self._invoke_model_direct(prompt, model_id, provider, gen_params, tokenizer_type, use_inference_profile)
    
    def _invoke_converse(
        self,
        prompt: str,
        model_id: str,
        gen_params: Dict[str, Any],
        tokenizer_type: str,
        use_inference_profile: bool = False
    ) -> Tuple[str, int]:
        """Invoke Anthropic Claude using Converse API."""
        # Build model ID variants - prioritize inference profiles if needed
        model_id_without_suffix = model_id.rsplit(":", 1)[0] if ":" in model_id else model_id
        
        # Start with inference profile variants if needed, then try direct model IDs
        model_id_variants = []
        
        if use_inference_profile or "us." in model_id or "global." in model_id:
            # Already an inference profile ID - use as-is first
            model_id_variants.append(model_id)
        else:
            # Try inference profile first (for us-east-2 region)
            if self.region_name == "us-east-2":
                model_id_variants.append(f"us.{model_id}")
                model_id_variants.append(f"us.{model_id_without_suffix}:0")
            # Then try direct model IDs
            model_id_variants.extend([
                model_id,  # Original model ID
                model_id_without_suffix,  # Without version suffix
            ])
        
        model_id_variants = list(dict.fromkeys(model_id_variants))  # Remove duplicates while preserving order
        
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
        
        last_error = None
        for variant_id in model_id_variants:
            try:
                response = self.bedrock_client.converse(
                    modelId=variant_id,
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
                
                # Get actual token usage from Converse API response
                usage = response.get("usage", {})
                input_tokens_api = usage.get("inputTokens", 0)  # Actual input tokens from API
                output_tokens = usage.get("outputTokens", 0)    # Actual output tokens from API
                
                # If output tokens not available, estimate
                if output_tokens == 0:
                    output_tokens = count_tokens(tokenizer_type, response_text)
                
                # Return: (response_text, output_tokens, input_tokens)
                return response_text, output_tokens, input_tokens_api
                
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_msg = e.response.get("Error", {}).get("Message", str(e))
                last_error = f"Bedrock API error ({error_code}): {error_msg} (tried model ID: {variant_id})"
                
                # If error mentions inference profile, try inference profile variants
                if "inference profile" in error_msg.lower() or "on-demand throughput" in error_msg.lower():
                    if not variant_id.startswith("us.") and not variant_id.startswith("global."):
                        # Try with inference profile prefix
                        inference_profile_id = f"us.{variant_id}"
                        if inference_profile_id not in model_id_variants:
                            model_id_variants.append(inference_profile_id)
                    continue
                
                # If it's not a ValidationException, don't try other variants
                if error_code != "ValidationException":
                    raise Exception(last_error)
            except Exception as e:
                error_str = str(e)
                last_error = f"Failed to invoke model with ID '{variant_id}': {error_str}"
                
                # Check if error mentions inference profile
                if "inference profile" in error_str.lower() or "on-demand throughput" in error_str.lower():
                    if not variant_id.startswith("us.") and not variant_id.startswith("global."):
                        inference_profile_id = f"us.{variant_id}"
                        if inference_profile_id not in model_id_variants:
                            model_id_variants.append(inference_profile_id)
                    continue
                
                # Continue to next variant only if it's a validation error
                if "ValidationException" not in error_str:
                    raise Exception(last_error)
        
        # If all variants failed, raise the last error with all attempted IDs
        attempted_ids = ", ".join([f"'{id}'" for id in model_id_variants])
        raise Exception(f"{last_error} All attempted model IDs: {attempted_ids}")
    
    def _invoke_model_direct(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        gen_params: Dict[str, Any],
        tokenizer_type: str,
        use_inference_profile: bool = False
    ) -> Tuple[str, int, int]:
        """Invoke model using InvokeModel API (for non-Claude models)."""
        # Build model ID variants - prioritize inference profiles if needed
        model_id_without_suffix = model_id.rsplit(":", 1)[0] if ":" in model_id else model_id
        model_id_variants = []
        
        if use_inference_profile or "us." in model_id or "global." in model_id:
            # Already an inference profile ID - use as-is first
            model_id_variants.append(model_id)
        else:
            # Try inference profile first (for us-east-2 region)
            if self.region_name == "us-east-2":
                model_id_variants.append(f"us.{model_id}")
                model_id_variants.append(f"us.{model_id_without_suffix}:0")
            # Then try direct model IDs
            model_id_variants.append(model_id)
            # For Meta Llama models, try different formats
            if provider == "meta" or "llama" in model_id.lower():
                if model_id_without_suffix != model_id:
                    model_id_variants.append(model_id_without_suffix)
                if ":" not in model_id:
                    model_id_variants.append(f"{model_id}:0")
        
        model_id_variants = list(dict.fromkeys(model_id_variants))  # Remove duplicates
        
        last_error = None
        for variant_id in model_id_variants:
            try:
                return self._try_invoke_model_direct(
                    prompt, variant_id, provider, gen_params, tokenizer_type
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_msg = e.response.get("Error", {}).get("Message", str(e))
                last_error = f"Bedrock API error ({error_code}): {error_msg} (tried model ID: {variant_id})"
                
                # If error mentions inference profile, try inference profile variants
                if "inference profile" in error_msg.lower() or "on-demand throughput" in error_msg.lower():
                    if not variant_id.startswith("us.") and not variant_id.startswith("global."):
                        inference_profile_id = f"us.{variant_id}"
                        if inference_profile_id not in model_id_variants:
                            model_id_variants.append(inference_profile_id)
                    continue
                
                if error_code != "ValidationException":
                    raise Exception(last_error)
            except Exception as e:
                error_str = str(e)
                last_error = f"Failed to invoke model with ID '{variant_id}': {error_str}"
                
                # Check if error mentions inference profile
                if "inference profile" in error_str.lower() or "on-demand throughput" in error_str.lower():
                    if not variant_id.startswith("us.") and not variant_id.startswith("global."):
                        inference_profile_id = f"us.{variant_id}"
                        if inference_profile_id not in model_id_variants:
                            model_id_variants.append(inference_profile_id)
                    continue
                
                if "ValidationException" not in error_str:
                    raise Exception(last_error)
        
        # If all variants failed
        attempted_ids = ", ".join([f"'{id}'" for id in model_id_variants])
        raise Exception(f"{last_error} All attempted model IDs: {attempted_ids}")
    
    def _try_invoke_model_direct(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        gen_params: Dict[str, Any],
        tokenizer_type: str
    ) -> Tuple[str, int, int]:
        """Helper method to try invoking a model with a specific model ID."""
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
                # Check for token usage in Meta Llama response
                # Meta models may return: usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                usage = response_body.get("usage", {})
                output_tokens = usage.get("completion_tokens") or usage.get("generation_tokens") or usage.get("output_tokens") or 0
            elif provider == "amazon" or "titan" in model_id.lower():
                result = response_body.get("results", [{}])[0] if response_body.get("results") else {}
                response_text = result.get("outputText", "")
                # Check for token usage in Amazon Titan response
                # Titan may return: usage.inputTextTokenCount, usage.results[0].tokenCount
                usage = result.get("usage", {})
                output_tokens = usage.get("tokenCount") or usage.get("outputTokenCount") or 0
            else:
                response_text = response_body.get("completion", "") or response_body.get("generated_text", "")
                # Check for generic token usage fields
                usage = response_body.get("usage", {})
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("generation_tokens") or 0
            
            # If no token usage found in API response, estimate
            if output_tokens == 0:
                output_tokens = count_tokens(tokenizer_type, response_text)
            
            # Try to get actual input tokens using CountTokens API
            input_tokens_api = self._get_actual_input_tokens(prompt, model_id, provider)
            
            # Return: (response_text, output_tokens, input_tokens)
            return response_text, output_tokens, input_tokens_api
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_msg}")
        except Exception as e:
            raise Exception(f"Failed to invoke model: {str(e)}")
    
    def _get_actual_input_tokens(
        self,
        prompt: str,
        model_id: str,
        provider: str
    ) -> int:
        """
        Get actual input token count using AWS Bedrock CountTokens API.
        
        This API returns the exact token count that would be charged,
        providing accurate cost calculation.
        
        Returns:
            Actual input token count, or 0 if API is not available/unsupported
        """
        try:
            # CountTokens API is available for supported models
            # Format request body based on provider
            if provider == "anthropic":
                # For Anthropic, use Converse API format
                body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ]
                }
            elif provider == "meta" or "llama" in model_id.lower():
                body = {"prompt": prompt}
            elif provider == "amazon" or "titan" in model_id.lower():
                body = {"inputText": prompt}
            else:
                body = {"prompt": prompt}
            
            # Call CountTokens API if available
            # Note: CountTokens API may not be available in all boto3 versions/regions
            if hasattr(self.bedrock_client, 'count_tokens'):
                response = self.bedrock_client.count_tokens(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json"
                )
                
                # Parse response - handle both dict and readable stream
                body_data = response.get("body", {})
                if hasattr(body_data, "read"):
                    response_body = json.loads(body_data.read())
                elif isinstance(body_data, dict):
                    response_body = body_data
                elif isinstance(body_data, str):
                    response_body = json.loads(body_data)
                else:
                    response_body = {}
                
                # Extract token count - CountTokens API returns totalTokens
                total_tokens = response_body.get("totalTokens", 0) or \
                              response_body.get("inputTokenCount", 0) or \
                              response_body.get("tokenCount", 0)
                
                if total_tokens > 0:
                    return int(total_tokens)
            
            # If CountTokens API is not available, return 0 to use estimation
            return 0
            
        except AttributeError:
            # Method doesn't exist in this boto3 version
            return 0
        except ClientError as e:
            # API error - CountTokens may not be available for this model/region
            return 0
        except Exception as e:
            # Any other error - fall back to estimation
            return 0
    
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
