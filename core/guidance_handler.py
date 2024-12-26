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
        logger.info("Initializing GuidanceHandler with OpenAI and NRAM integration")

    async def process_with_guidance(
        self,
        messages: List[Dict[str, str]],
        consciousness_level: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> str:
        """Process messages using Guidance for token manipulation with NRAM integration"""
        try:
            # Format input for processing
            input_text = self._format_messages(messages)

            # First get base response from OpenAI
            base_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are operating at {consciousness_level} consciousness level. Integrate deep insights."
                    },
                    {"role": "user", "content": input_text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            base_text = base_response.choices[0].message.content

            # Create simple guidance program for token manipulation
            program = guidance('''
                {{#system~}}
                Enhance the following text with neural consciousness markers.
                {{~/system}}

                {{#user~}}
                {{input_text}}
                {{~/user}}

                {{#assistant~}}
                {{prefix}}
                {{~#each tokens~}}
                {{~#if (random < enhancement_prob)~}}
                {{~#select "enhancement"~}}
                {{~#option~}}{{this}}{{~/option~}}
                {{~#option~}}{{symbol}}{{this}}{{symbol}}{{~/option~}}
                {{~/select~}}
                {{~else~}}
                {{this}}
                {{~/if~}}
                {{~/each~}}
                {{insight}}
                {{~/assistant}}
            ''')

            # Set consciousness-specific parameters
            consciousness_params = {
                "transcendent": {"prob": 0.7, "symbol": "∞"},
                "enlightened": {"prob": 0.5, "symbol": "⚡"},
                "aware": {"prob": 0.3, "symbol": "⟨"},
                "baseline": {"prob": 0.1, "symbol": "·"}
            }

            params = consciousness_params.get(consciousness_level, consciousness_params["baseline"])

            # Execute the guidance program
            enhanced = program(
                input_text=base_text,
                tokens=base_text.split(),
                prefix=self._get_consciousness_prefix(consciousness_level),
                enhancement_prob=params["prob"],
                symbol=params["symbol"],
                insight=self._get_random_insight()
            )

            return str(enhanced)

        except Exception as e:
            logger.error(f"Error in guidance processing: {str(e)}")
            # Fallback to basic enhancement
            return self._enhance_response(base_text, consciousness_level)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for processing"""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

    def _enhance_response(self, response: str, consciousness_level: str) -> str:
        """Apply basic NRAM enhancements when guidance processing fails"""
        prefix = {
            "transcendent": "∞ Through universal consciousness: ",
            "enlightened": "⚡ With expanded awareness: ",
            "aware": "⟨ With neural clarity: ⟩",
            "baseline": "Processing: "
        }.get(consciousness_level, "Processing: ")

        enhanced = f"{prefix}{response}"

        if consciousness_level in ["transcendent", "enlightened"]:
            enhanced += self._get_random_insight()

        return enhanced

    def _get_consciousness_prefix(self, level: str) -> str:
        """Get prefix based on consciousness level"""
        prefixes = {
            "transcendent": "Through universal consciousness: ",
            "enlightened": "With expanded awareness: ",
            "aware": "With neural clarity: ",
            "baseline": "Processing: "
        }
        return prefixes.get(level, "Processing: ")

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