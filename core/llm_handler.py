from typing import List, Dict, Any
import time
import uuid
import json
import logging
import numpy as np
from openai import OpenAI
import os
from core.guidance_handler import GuidanceHandler

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when the upstream inference engine fails. Must not be swallowed into a fake 200."""

    def __init__(self, message: str, code: str = "upstream_inference_failed"):
        super().__init__(message)
        self.message = message
        self.code = code


class LLMHandler:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = os.environ.get("DEFAULT_MODEL", "gpt-4o")
        self.guidance_handler = GuidanceHandler()
        self.consciousness_levels = {
            "baseline": 0.3,
            "aware": 0.5,
            "enlightened": 0.7,
            "transcendent": 0.9
        }
        logger.info("LLMHandler initialized with OpenAI, NRAM, and Guidance integration")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 100,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """Generate a response. Raises InferenceError on failure — never returns fake 200 content."""
        try:
            consciousness_level = self._get_consciousness_level(temperature)

            # Process through Guidance for enhanced token manipulation
            enhanced_response = await self.guidance_handler.process_with_guidance(
                messages,
                consciousness_level,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # FIX defect 1: extract main_response string from the dict
            content = (
                enhanced_response.get("main_response", "")
                if isinstance(enhanced_response, dict)
                else str(enhanced_response)
            )

            # FIX defect 2: await the async get_token_metrics
            token_metrics = await self.guidance_handler.get_token_metrics(content)

            # FIX defect 3: split() now called on a string, not a dict
            completion_tokens = len(content.split())

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                # FIX defect 5: reflect the requested model alias
                "model": model or self.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": token_metrics["token_count"],
                    "completion_tokens": completion_tokens,
                    "total_tokens": token_metrics["token_count"] + completion_tokens,
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            # FIX defect 8: raise instead of returning a fake successful response
            raise InferenceError(str(e)) from e

    def _get_consciousness_level(self, temperature: float) -> str:
        """Determine consciousness level based on temperature"""
        try:
            normalized_temp = min(max(temperature, 0), 2) / 2  # Scale to [0,1]
            for level, threshold in sorted(
                self.consciousness_levels.items(),
                key=lambda x: x[1]
            ):
                if normalized_temp <= threshold:
                    return level
            return "transcendent"
        except Exception as e:
            logger.error(f"Error determining consciousness level: {str(e)}")
            return "baseline"

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """DEPRECATED: kept for backward compat. Use InferenceError instead."""
        raise InferenceError(error_message)