"""Master Model Evaluator - Supports OpenAI ChatGPT as reference model."""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import time

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.utils.timing import Stopwatch
from src.tokenizers import count_tokens


class MasterModelEvaluator:
    """Evaluates prompts against master/reference models (e.g., ChatGPT)."""
    
    def __init__(self, model_type: str = "chatgpt", api_key: Optional[str] = None):
        """
        Initialize master model evaluator.
        
        Args:
            model_type: Type of master model ("chatgpt", "gpt-4", etc.)
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        """
        self.model_type = model_type.lower()
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found! Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed! Please install it with: pip install openai"
            )
        
        # Strip any whitespace from the API key
        self.api_key = self.api_key.strip()
        self.client = OpenAI(api_key=self.api_key)
        
        # Map model types to OpenAI model IDs
        self.model_map = {
            "chatgpt": "gpt-3.5-turbo",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-4o": "gpt-4o",
        }
    
    def evaluate_prompt(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt against the master model.
        
        Args:
            prompt: The prompt text to evaluate
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            model_id: Optional specific model ID (overrides model_type)
        
        Returns:
            Dictionary with evaluation metrics including response
        """
        # Determine model ID
        if model_id:
            openai_model_id = model_id
        else:
            openai_model_id = self.model_map.get(self.model_type, "gpt-3.5-turbo")
        
        # Count input tokens
        input_tokens = count_tokens("gpt2", prompt)  # Use GPT-2 tokenizer as approximation
        
        # Initialize metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": f"Master: {openai_model_id}",
            "model_id": openai_model_id,
            "input_prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "latency_ms": 0,
            "response": "",
            "status": "success",
            "error": None,
        }
        
        # Make API call with timing
        timer = None
        try:
            with Stopwatch() as timer:
                response = self.client.chat.completions.create(
                    model=openai_model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            
            metrics["latency_ms"] = timer.elapsed_ms
            
            # Extract response
            response_text = response.choices[0].message.content or ""
            metrics["response"] = response_text
            
            # Get token usage from API response
            if response.usage:
                metrics["input_tokens"] = response.usage.prompt_tokens
                metrics["output_tokens"] = response.usage.completion_tokens
            
            # Calculate cost (approximate pricing for GPT-3.5-turbo and GPT-4)
            cost = self._calculate_cost(openai_model_id, metrics["input_tokens"], metrics["output_tokens"])
            metrics["cost_usd_total"] = cost
            
        except Exception as e:
            metrics["status"] = "error"
            metrics["error"] = str(e)
            metrics["latency_ms"] = timer.elapsed_ms if timer is not None else 0
        
        return metrics
    
    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on OpenAI pricing (as of 2024)."""
        # Pricing per 1K tokens (approximate, update as needed)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
        }
        
        model_pricing = pricing.get(model_id, pricing["gpt-3.5-turbo"])
        input_cost = (input_tokens / 1000.0) * model_pricing["input"]
        output_cost = (output_tokens / 1000.0) * model_pricing["output"]
        
        return round(input_cost + output_cost, 6)

