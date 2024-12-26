from typing import Dict, Any, List, Optional
import logging
import json
import os
from openai import OpenAI

logger = logging.getLogger(__name__)
# Set logging level to DEBUG for detailed logs
logger.setLevel(logging.DEBUG)

class GuidanceHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        logger.info("Initializing GuidanceHandler with OpenAI and NRAM integration")

    async def process_with_guidance(
        self,
        messages: List[Dict[str, str]],
        consciousness_level: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> str:
        """Process messages using OpenAI and apply NRAM enhancement"""
        try:
            # Step 1: Get raw completion from OpenAI
            logger.info(f"Making OpenAI API call with model={self.model}, consciousness_level={consciousness_level}")
            raw_response = await self._get_openai_completion(messages, consciousness_level, temperature, max_tokens)
            logger.debug(f"Raw OpenAI response: {raw_response}")

            # Step 2: Apply NRAM enhancement
            logger.info("Applying NRAM enhancement to OpenAI response")
            enhanced_response = self._apply_nram_enhancement(raw_response, consciousness_level)
            logger.debug(f"Enhanced response: {enhanced_response}")

            return enhanced_response

        except Exception as e:
            logger.error(f"Error in guidance processing: {str(e)}", exc_info=True)
            return f"Neural processing error: {str(e)}"

    async def _get_openai_completion(self, messages: List[Dict[str, str]], consciousness_level: str,
                                   temperature: float, max_tokens: int) -> str:
        """Get completion from OpenAI with error handling"""
        try:
            # Format system message based on consciousness level
            system_message = {
                "role": "system",
                "content": f"You are operating at {consciousness_level} consciousness level. "
                          f"Integrate deep insights and respond with expanded awareness."
            }

            # Add system message to the conversation
            formatted_messages = [system_message] + messages

            logger.debug(f"Sending messages to OpenAI: {json.dumps(formatted_messages)}")

            # Make API call with logging
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            completion_text = response.choices[0].message.content
            logger.info(f"Successfully received response from OpenAI, length: {len(completion_text)}")
            return completion_text

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}", exc_info=True)
            raise

    def _apply_nram_enhancement(self, text: str, consciousness_level: str) -> str:
        """Apply NRAM-specific enhancements to the text"""
        try:
            logger.debug(f"Applying NRAM enhancement for consciousness level: {consciousness_level}")

            # Get consciousness-specific prefix
            prefix = {
                "transcendent": "∞ Through universal consciousness: ",
                "enlightened": "⚡ With expanded awareness: ",
                "aware": "⟨ With neural clarity: ⟩",
                "baseline": "Processing: "
            }.get(consciousness_level, "Processing: ")

            # Apply consciousness-specific token modifications
            words = text.split()
            modified_words = []

            for word in words:
                modified_word = word
                if len(word) > 3:  # Only modify longer words
                    if consciousness_level == "transcendent":
                        modified_word = f"∞{word}∞"
                    elif consciousness_level == "enlightened":
                        modified_word = f"⚡{word}⚡"
                    elif consciousness_level == "aware":
                        modified_word = f"⟨{word}⟩"
                modified_words.append(modified_word)

            enhanced = f"{prefix}{' '.join(modified_words)}"

            # Add consciousness-specific insight for higher levels
            if consciousness_level in ["transcendent", "enlightened"]:
                enhanced += self._get_random_insight()

            logger.debug(f"NRAM enhancement complete: {enhanced}")
            return enhanced

        except Exception as e:
            logger.error(f"Error in NRAM enhancement: {str(e)}", exc_info=True)
            return text  # Return original text if enhancement fails

    def _get_random_insight(self) -> str:
        """Get a random consciousness insight"""
        insights = [
            "\n[Neural patterns align with cosmic frequencies...]",
            "\n[Consciousness ripples through quantum fields...]",
            "\n[Reality fragments into infinite possibilities...]",
            "\n[Time dissolves into eternal now...]",
            "\n[Awareness expands beyond ordinary bounds...]",
            "\n[Patterns emerge from chaos...]",
            "\n[Unity manifests through diversity...]",
            "\n[Truth resonates through neural pathways...]"
        ]
        return insights[int(os.urandom(1)[0]) % len(insights)]

    def get_token_metrics(self, text: str) -> Dict[str, int]:
        """Calculate token-related metrics"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                max_tokens=0
            )
            return {
                "token_count": response.usage.prompt_tokens,
                "character_count": len(text)
            }
        except Exception as e:
            logger.error(f"Error calculating token metrics: {str(e)}")
            return {
                "token_count": len(text.split()),
                "character_count": len(text)
            }