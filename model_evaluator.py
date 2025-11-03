"""
Model evaluator module - handles invoking models and collecting metrics.
"""

import boto3
import json
import re
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from config import MODELS, PRICING, EVAL_SETTINGS, AWS_PROFILE, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


class ModelEvaluator:
    """Evaluates a single model with multiple prompts and collects metrics."""
    
    def __init__(self, model_key: str):
        """
        Initialize evaluator for a specific model.
        
        Args:
            model_key: Key from MODELS config dictionary
        """
        if model_key not in MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODELS.keys())}")
        
        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.model_id = self.model_config["model_id"]
        self.provider = self.model_config["provider"]
        
        # Initialize Bedrock client
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            # Use explicit credentials
            session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
        else:
            # Use profile or default credentials
            session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        self.bedrock_client = session.client("bedrock-runtime")
        
        # Metrics storage
        self.results: List[Dict[str, Any]] = []
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on token usage."""
        if self.model_key not in PRICING:
            return 0.0
        
        pricing = PRICING[self.model_key]
        cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
        return round(cost, 6)
    
    def clean_json_output(self, raw: str) -> str:
        """
        Extract JSON substring from text if present.
        Enhanced version that handles Llama responses with explanatory text.
        """
        if not raw or not raw.strip():
            return raw
        
        # First try direct parsing
        try:
            json.loads(raw.strip())
            return raw.strip()
        except:
            pass
        
        # Strategy 1: Look for JSON in markdown code blocks
        code_block_patterns = [
            r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```',  # JSON in markdown code blocks
            r'```\s*(\[.*?\]|\{.*?\})\s*```',  # JSON in code blocks without json tag
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                cleaned = match.group(1) if match.lastindex else match.group(0)
                try:
                    json.loads(cleaned)
                    return cleaned
                except:
                    continue
        
        # Strategy 2: Find first occurrence of [ or { and try to extract balanced JSON
        # This handles cases like "Here is the JSON response: [{...}]"
        json_start_pattern = r'(\[|\{)'
        match = re.search(json_start_pattern, raw)
        if match:
            start_pos = match.start()
            start_char = match.group(1)
            end_char = ']' if start_char == '[' else '}'
            
            # Extract from start position to end of string
            candidate = raw[start_pos:]
            
            # Try to find balanced brackets
            depth = 0
            end_pos = -1
            for i, char in enumerate(candidate):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                extracted = candidate[:end_pos]
                try:
                    json.loads(extracted)
                    return extracted
                except:
                    pass
        
        # Strategy 3: Try to find JSON after common prefixes
        # Llama often says things like "Here is the JSON response:" or "Response:"
        prefix_patterns = [
            r'(?:Here is|Here\'s|The response is|Response|JSON response|Answer)[:\s]*(\[|\{)',
            r'(?:json|JSON)[:\s]*(\[|\{)',
        ]
        
        for pattern in prefix_patterns:
            match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
            if match:
                start_pos = match.end() - 1  # Position of [ or {
                start_char = raw[start_pos]
                end_char = ']' if start_char == '[' else '}'
                
                candidate = raw[start_pos:]
                depth = 0
                end_pos = -1
                for i, char in enumerate(candidate):
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    extracted = candidate[:end_pos]
                    try:
                        json.loads(extracted)
                        return extracted
                    except:
                        continue
        
        # Strategy 4: Try simple regex patterns (fallback)
        simple_patterns = [
            r'(\[.*?\]|\{.*?\})',  # Non-greedy match
        ]
        
        for pattern in simple_patterns:
            matches = list(re.finditer(pattern, raw, re.DOTALL))
            # Try matches from longest to shortest (more likely to be complete)
            for match in sorted(matches, key=lambda m: m.end() - m.start(), reverse=True):
                extracted = match.group(1) if match.lastindex else match.group(0)
                try:
                    json.loads(extracted)
                    return extracted
                except:
                    continue
        
        # If no valid JSON found, return original
        return raw
    
    def is_valid_json(self, text: str, try_cleaning: bool = True) -> Tuple[bool, str]:
        """
        Check if text contains valid JSON, optionally trying to clean it first.
        
        Args:
            text: Text to validate
            try_cleaning: Whether to attempt JSON extraction/cleaning first
            
        Returns:
            Tuple of (is_valid, cleaned_text)
        """
        if not text or not text.strip():
            return False, text
        
        # First try direct validation
        try:
            json.loads(text.strip())
            return True, text.strip()
        except (json.JSONDecodeError, TypeError):
            pass
        
        # If direct validation fails and cleaning is enabled, try cleaning
        if try_cleaning:
            cleaned = self.clean_json_output(text)
            if cleaned != text:  # Only try if cleaning changed something
                try:
                    json.loads(cleaned)
                    return True, cleaned
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return False, text
    
    def invoke_model(self, prompt: str, retry_on_invalid_json: bool = True) -> Tuple[str, float, bool, int, bool]:
        """
        Invoke the model with a prompt.
        
        Args:
            prompt: Input prompt text
            retry_on_invalid_json: Whether to retry if JSON is invalid
            
        Returns:
            Tuple of (output_text, latency_ms, is_valid_json, retry_count, json_was_cleaned)
        """
        retries = 0
        output = ""
        valid_json = False
        latency_ms = None
        json_was_cleaned = False
        
        # For Llama models, enhance initial prompt to encourage JSON output
        current_prompt = prompt
        prompt_lower = prompt.lower()
        if self.provider == "meta" and not ("json" in prompt_lower and ("return" in prompt_lower or "format" in prompt_lower or "output" in prompt_lower)):
            # Add JSON instruction suffix for better results if prompt doesn't already ask for JSON
            llama_suffix = EVAL_SETTINGS.get("llama_json_prompt_suffix", "")
            if llama_suffix and not prompt.endswith(llama_suffix):
                current_prompt = prompt + llama_suffix
        
        while retries <= EVAL_SETTINGS["max_retries"]:
            # Reset cleaning flag for each attempt
            attempt_json_cleaned = False
            start_time = time.time()
            
            try:
                # Prepare payload based on provider
                if self.provider == "anthropic":
                    payload = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{"role": "user", "content": [{"type": "text", "text": current_prompt}]}],
                        "max_tokens": EVAL_SETTINGS["max_tokens"],
                        "temperature": EVAL_SETTINGS["temperature"],
                    }
                elif self.provider == "meta":
                    payload = {
                        "prompt": current_prompt,
                        "max_gen_len": EVAL_SETTINGS["max_tokens"],
                        "temperature": EVAL_SETTINGS["temperature"],
                    }
                else:
                    # Generic payload - may need customization for other providers
                    payload = {
                        "prompt": current_prompt,
                        "max_tokens": EVAL_SETTINGS["max_tokens"],
                        "temperature": EVAL_SETTINGS["temperature"],
                    }
                
                # Invoke model
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload),
                )
                
                latency_ms = (time.time() - start_time) * 1000
                body_json = json.loads(response["body"].read())
                
                # Extract output based on provider
                if self.provider == "anthropic":
                    output = body_json.get("content", [{}])[0].get("text", "")
                    # Get actual token counts if available
                    input_tokens_actual = body_json.get("usage", {}).get("input_tokens", None)
                    output_tokens_actual = body_json.get("usage", {}).get("output_tokens", None)
                elif self.provider == "meta":
                    output = body_json.get("generation") or body_json.get("output_text") or str(body_json)
                    input_tokens_actual = body_json.get("prompt_token_count", None)
                    output_tokens_actual = body_json.get("generation_token_count", None)
                else:
                    output = body_json.get("generation") or body_json.get("output_text") or str(body_json)
                    input_tokens_actual = None
                    output_tokens_actual = None
                
                output = output.strip()
                
                # Validate JSON with cleaning (especially important for Llama models)
                # Llama models sometimes wrap JSON in markdown or add extra text
                valid_json, cleaned_output = self.is_valid_json(output, try_cleaning=True)
                
                # Use cleaned output if it's valid, otherwise use original
                if valid_json and cleaned_output != output:
                    output = cleaned_output
                    attempt_json_cleaned = True
                    json_was_cleaned = True  # Track that cleaning was needed
                
                # Retry if JSON is invalid and retries enabled
                if not valid_json and retry_on_invalid_json and retries < EVAL_SETTINGS["max_retries"]:
                    retries += 1
                    print(f"    {self.model_key}: Invalid JSON, retrying ({retries}/{EVAL_SETTINGS['max_retries']})...")
                    
                    # Enhanced retry prompt - more specific for Llama
                    if self.provider == "meta":
                        current_prompt = (
                            "IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, "
                            "no explanations, no text before or after. Just the raw JSON array or object.\n\n"
                            + prompt
                        )
                    else:
                        current_prompt = EVAL_SETTINGS["json_retry_prompt_prefix"] + prompt
                    continue
                
                break
                
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000 if latency_ms is None else latency_ms
                output = f"Error: {str(e)}"
                break
        
        return output, latency_ms, valid_json, retries, json_was_cleaned
    
    def evaluate_prompts(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate model on a list of prompts.
        
        Args:
            prompts: List of prompt dictionaries with 'prompt' and optional 'index' keys
            
        Returns:
            List of result dictionaries with all metrics
        """
        print(f"\nEvaluating {self.model_config['name']} ({self.model_key}) on {len(prompts)} prompts...")
        
        latencies = []
        self.results = []
        
        for prompt_data in prompts:
            prompt_text = prompt_data["prompt"]
            prompt_index = prompt_data.get("index", len(self.results) + 1)
            
            print(f"  [{prompt_index}] Processing prompt...", end="", flush=True)
            
            # Invoke model
            output, latency_ms, valid_json, retries, json_was_cleaned = self.invoke_model(prompt_text)
            
            # Calculate metrics
            input_tokens = self.estimate_tokens(prompt_text)
            output_tokens = self.estimate_tokens(output)
            cost = self.calculate_cost(input_tokens, output_tokens)
            
            latencies.append(latency_ms)
            
            result = {
                "prompt_index": prompt_index,
                "model_key": self.model_key,
                "model_name": self.model_config["name"],
                "prompt_snippet": prompt_text[:100].replace("\n", " "),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(latency_ms, 2) if latency_ms else None,
                "cost_usd": cost,
                "valid_json": valid_json,
                "retries": retries,
                "json_was_cleaned": json_was_cleaned,  # Track if JSON extraction was needed
                "output": output[:500],  # Truncate for CSV
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add metadata if present
            if "metadata" in prompt_data:
                result["metadata"] = json.dumps(prompt_data["metadata"])
            
            self.results.append(result)
            
            status = "✓" if valid_json else "✗"
            print(f" {status} ({latency_ms:.0f}ms)")
        
        return self.results
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics (p50, p95, averages, etc.) for this model."""
        if not self.results:
            return {}
        
        latencies = [r["latency_ms"] for r in self.results if r["latency_ms"] is not None]
        costs = [r["cost_usd"] for r in self.results]
        valid_json_count = sum(1 for r in self.results if r["valid_json"])
        total_input_tokens = sum(r["input_tokens"] for r in self.results)
        total_output_tokens = sum(r["output_tokens"] for r in self.results)
        
        if not latencies:
            return {}
        
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        return {
            "model_key": self.model_key,
            "model_name": self.model_config["name"],
            "total_prompts": len(self.results),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p50_latency_ms": round(latencies_sorted[n // 2], 2),
            "p95_latency_ms": round(latencies_sorted[int(n * 0.95)], 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_usd": round(statistics.mean(costs), 6),
            "valid_json_rate": round(valid_json_count / len(self.results), 4),
            "valid_json_count": valid_json_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_input_tokens": round(statistics.mean([r["input_tokens"] for r in self.results]), 1),
            "avg_output_tokens": round(statistics.mean([r["output_tokens"] for r in self.results]), 1),
        }

