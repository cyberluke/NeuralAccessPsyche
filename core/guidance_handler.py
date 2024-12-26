from typing import Dict, Any, List, Optional
import guidance
import logging
import json
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class GuidanceHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        logger.info("Initializing GuidanceHandler with OpenAI integration")

    async def process_with_guidance(
        self,
        messages: List[Dict[str, str]],
        consciousness_level: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> str:
        """Process messages using Guidance for token manipulation"""
        try:
            # Format input for processing
            input_text = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in messages
            )

            # Get raw completion from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": input_text}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_response = response.choices[0].message.content

            # Create a guidance program for token manipulation
            program = guidance.Program(
                '''
                {{#system~}}
                You are operating at a {{consciousness_level}} consciousness level.
                Process and respond with appropriate awareness and insight.
                {{~/system}}

                {{#user~}}
                {{input}}
                {{~/user}}

                {{#assistant~}}
                {{prefix}}{{enhanced_text}}
                {{~/assistant}}
                '''
            )

            # Get consciousness-specific settings
            prefix = self._get_consciousness_prefix(consciousness_level)
            enhanced_text = self._apply_consciousness_effects(
                raw_response,
                consciousness_level
            )

            # Execute guidance program
            result = program(
                consciousness_level=consciousness_level,
                input=input_text,
                prefix=prefix,
                enhanced_text=enhanced_text
            )

            # Add consciousness-specific insights
            final_response = str(result)
            if consciousness_level in ["enlightened", "transcendent"]:
                final_response += self._get_random_insight()

            return final_response

        except Exception as e:
            logger.error(f"Error in guidance processing: {str(e)}")
            return f"Neural processing error: {str(e)}"

    def _get_consciousness_prefix(self, level: str) -> str:
        """Get prefix based on consciousness level"""
        prefixes = {
            "transcendent": "Through universal consciousness: ",
            "enlightened": "With expanded awareness: ",
            "aware": "With neural clarity: ",
            "baseline": "Processing: "
        }
        return prefixes.get(level, "Processing: ")

    def _apply_consciousness_effects(self, text: str, level: str) -> str:
        """Apply consciousness-specific text effects"""
        words = text.split()
        modified_words = []

        for word in words:
            if level == "transcendent" and len(word) > 3:
                word = f"∞{word}∞"
            elif level == "enlightened" and len(word) > 3:
                word = f"⚡{word}⚡"
            elif level == "aware" and len(word) > 3:
                word = f"⟨{word}⟩"
            modified_words.append(word)

        return " ".join(modified_words)

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
                max_tokens=0  # We only need the prompt tokens
            )
            return {
                "token_count": response.usage.prompt_tokens,
                "character_count": len(text)
            }
        except Exception as e:
            logger.error(f"Error calculating token metrics: {str(e)}")
            return {
                "token_count": len(text.split()),  # Fallback to word count
                "character_count": len(text)
            }