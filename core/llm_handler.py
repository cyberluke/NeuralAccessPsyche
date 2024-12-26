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

class LLMHandler:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
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
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        try:
            # Get consciousness level based on temperature
            consciousness_level = self._get_consciousness_level(temperature)

            # Process through Guidance for enhanced token manipulation
            enhanced_response = await self.guidance_handler.process_with_guidance(
                messages,
                consciousness_level,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Get token metrics
            token_metrics = self.guidance_handler.get_token_metrics(enhanced_response)

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"nram-enhanced-{self.model}",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": enhanced_response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": token_metrics["token_count"],
                    "completion_tokens": len(enhanced_response.split()),
                    "total_tokens": token_metrics["token_count"] + len(enhanced_response.split())
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return self._create_error_response(str(e))

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
        """Create an error response in the expected format"""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"nram-enhanced-{self.model}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Neural processing error: {error_message}"
                },
                "finish_reason": "error"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }