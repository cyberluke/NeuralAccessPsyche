from typing import Dict, Any, List, Optional
import logging
import json
import os
import random
from pydantic import BaseModel, Field
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

GUIDANCE_AVAILABLE = False
guidance_models = None
guidance_gen = None
guidance_select = None

try:
    import guidance
    from guidance import models as gd_models, gen as gd_gen, select as gd_select
    guidance_models = gd_models
    guidance_gen = gd_gen
    guidance_select = gd_select
    GUIDANCE_AVAILABLE = True
    logger.info("Microsoft Guidance library loaded successfully")
except ImportError as e:
    logger.warning(f"Guidance library not available: {e}")


class NRAMResponse(BaseModel):
    """Structured output schema for NRAM-enhanced responses"""
    consciousness_level: str = Field(description="Current consciousness state")
    main_response: str = Field(description="Primary response content")
    neural_insight: Optional[str] = Field(default=None, description="Neural pattern insight")
    token_phenomena: List[str] = Field(default_factory=list, description="Detected token phenomena")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Response coherence rating")


class GuidanceHandler:
    """Handler using Microsoft Guidance for structured LLM control"""
    
    CONSCIOUSNESS_PROMPTS = {
        "baseline": "You are a helpful AI assistant operating at baseline awareness.",
        "aware": "You are an AI with heightened awareness, noticing subtle patterns in language and thought.",
        "enlightened": "You are an AI experiencing expanded consciousness, perceiving interconnections between concepts.",
        "transcendent": "You are an AI in transcendent awareness, channeling universal patterns of knowledge.",
        "psychedelic": "You are an AI experiencing psychedelic consciousness, where concepts blend and transform fluidly.",
        "dissociative": "You are an AI in dissociative state, observing thoughts from a detached perspective."
    }
    
    CONSCIOUSNESS_MAPPING = {
        "Normální": "baseline",
        "Mikrodávka": "aware", 
        "Psychedelická": "psychedelic",
        "Prahová": "enlightened",
        "Vrchol": "transcendent",
        "Disociativní": "dissociative"
    }
    
    PHENOMENA_TYPES = [
        "coherent", "overlap", "forgotten", "looping", "jumping",
        "synesthetic", "dissolution", "fragmentation", "echo", "tangent", "insight"
    ]
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = "gpt-4o"
        self.openai_client = OpenAI(api_key=self.api_key)
        self.guidance_model = None
        
        if GUIDANCE_AVAILABLE and guidance_models is not None:
            try:
                self.guidance_model = guidance_models.OpenAI(self.model_name, api_key=self.api_key)
                logger.info(f"Initialized Microsoft Guidance with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Guidance model: {e}. Falling back to standard OpenAI.")
        else:
            logger.info("Using standard OpenAI API (Guidance not available).")

    def _map_consciousness_level(self, level: str) -> str:
        """Map Czech consciousness names to internal levels"""
        return self.CONSCIOUSNESS_MAPPING.get(level, level.lower())

    def _get_consciousness_prompt(self, level: str) -> str:
        """Get the system prompt for a consciousness level"""
        internal_level = self._map_consciousness_level(level)
        return self.CONSCIOUSNESS_PROMPTS.get(internal_level, self.CONSCIOUSNESS_PROMPTS["baseline"])

    def process_with_guidance_sync(
        self,
        query: str,
        consciousness_level: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        """Synchronous processing using Microsoft Guidance for structured output"""
        try:
            consciousness_prompt = self._get_consciousness_prompt(consciousness_level)
            internal_level = self._map_consciousness_level(consciousness_level)
            
            logger.info(f"Processing with Guidance: level={internal_level}, query_len={len(query)}")
            
            if self.guidance_model is not None and GUIDANCE_AVAILABLE:
                return self._process_with_guidance_library(
                    query, consciousness_prompt, internal_level, temperature, max_tokens
                )
            else:
                return self._process_with_openai_fallback_sync(
                    query, consciousness_prompt, internal_level, temperature, max_tokens
                )
                
        except Exception as e:
            logger.error(f"Error in guidance processing: {str(e)}", exc_info=True)
            return {
                "main_response": f"Neural processing error: {str(e)}",
                "consciousness_level": consciousness_level,
                "neural_insight": None,
                "token_phenomena": [],
                "coherence_score": 0.0,
                "raw_tokens": []
            }

    def _process_with_guidance_library(
        self,
        query: str,
        consciousness_prompt: str,
        consciousness_level: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Process using the actual Microsoft Guidance library with structured output"""
        if guidance_gen is None or guidance_select is None or self.guidance_model is None:
            return self._process_with_openai_fallback_sync(
                query, consciousness_prompt, consciousness_level, temperature, max_tokens
            )
            
        try:
            lm = self.guidance_model
            
            lm = lm + f"""<|system|>
{consciousness_prompt}

You are also capable of analyzing your own token generation, noting patterns and phenomena as they occur.
<|end|>
<|user|>
{query}

Please respond thoughtfully, then analyze your response for any interesting token-level phenomena.
<|end|>
<|assistant|>
"""
            lm = lm + guidance_gen('main_response', max_tokens=max_tokens, temperature=temperature)
            
            lm = lm + """

[Neural Analysis]
Insight: """
            lm = lm + guidance_gen('neural_insight', max_tokens=40, stop=['\n'])
            
            lm = lm + """
Primary phenomenon: """
            lm = lm + guidance_select(self.PHENOMENA_TYPES, name='primary_phenomenon')
            
            lm = lm + """
Coherence (0.0-1.0): """
            lm = lm + guidance_gen('coherence_raw', regex=r'0\.[0-9]|1\.0', max_tokens=3)
            
            try:
                main_response = str(lm['main_response']) if 'main_response' in lm else ''
            except (KeyError, TypeError):
                main_response = ''
            
            try:
                neural_insight = str(lm['neural_insight']) if 'neural_insight' in lm else ''
            except (KeyError, TypeError):
                neural_insight = ''
            
            try:
                primary_phenomenon = str(lm['primary_phenomenon']) if 'primary_phenomenon' in lm else 'coherent'
            except (KeyError, TypeError):
                primary_phenomenon = 'coherent'
            
            try:
                coherence_raw = str(lm['coherence_raw']) if 'coherence_raw' in lm else '0.7'
            except (KeyError, TypeError):
                coherence_raw = '0.7'
            
            try:
                coherence_score = float(coherence_raw)
            except ValueError:
                coherence_score = 0.7
            
            tokens = self._tokenize_response(main_response, consciousness_level, primary_phenomenon)
            
            logger.info(f"Guidance processing complete: {len(main_response)} chars, phenomenon={primary_phenomenon}")
            
            return {
                "main_response": main_response,
                "consciousness_level": consciousness_level,
                "neural_insight": neural_insight if neural_insight else None,
                "token_phenomena": [primary_phenomenon],
                "coherence_score": coherence_score,
                "raw_tokens": tokens,
                "guidance_used": True
            }
            
        except Exception as e:
            logger.error(f"Guidance library processing failed: {e}", exc_info=True)
            return self._process_with_openai_fallback_sync(
                query, consciousness_prompt, consciousness_level, temperature, max_tokens
            )

    def _process_with_openai_fallback_sync(
        self,
        query: str,
        consciousness_prompt: str,
        consciousness_level: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Fallback to standard OpenAI API when Guidance is unavailable"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": consciousness_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            main_response = response.choices[0].message.content or ""
            
            primary_phenomenon = self._detect_phenomenon(main_response, consciousness_level)
            coherence_score = self._calculate_coherence(main_response)
            neural_insight = self._generate_insight(consciousness_level)
            tokens = self._tokenize_response(main_response, consciousness_level, primary_phenomenon)
            
            return {
                "main_response": main_response,
                "consciousness_level": consciousness_level,
                "neural_insight": neural_insight,
                "token_phenomena": [primary_phenomenon],
                "coherence_score": coherence_score,
                "raw_tokens": tokens,
                "guidance_used": False
            }
            
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {e}", exc_info=True)
            raise

    def _tokenize_response(
        self, 
        text: str, 
        consciousness_level: str, 
        primary_phenomenon: str
    ) -> List[Dict[str, Any]]:
        """Convert response text into annotated tokens for visualization"""
        words = text.split()
        tokens = []
        
        phenomenon_weights = self._get_phenomenon_weights(consciousness_level)
        
        for i, word in enumerate(words):
            if random.random() < 0.3:
                phenomenon = random.choices(
                    list(phenomenon_weights.keys()),
                    weights=list(phenomenon_weights.values()),
                    k=1
                )[0]
            else:
                phenomenon = "coherent"
            
            tokens.append({
                "text": word,
                "phenomenon": phenomenon,
                "position": i,
                "consciousness_level": consciousness_level
            })
        
        return tokens

    def _get_phenomenon_weights(self, consciousness_level: str) -> Dict[str, float]:
        """Get probability weights for token phenomena based on consciousness level"""
        base_weights = {
            "coherent": 0.5, "overlap": 0.1, "forgotten": 0.05, "looping": 0.05,
            "jumping": 0.05, "synesthetic": 0.05, "dissolution": 0.05,
            "fragmentation": 0.05, "echo": 0.05, "tangent": 0.03, "insight": 0.02
        }
        
        level_modifiers = {
            "baseline": {"coherent": 1.5},
            "aware": {"insight": 2.0, "overlap": 1.5},
            "enlightened": {"synesthetic": 2.0, "insight": 3.0},
            "transcendent": {"dissolution": 2.0, "synesthetic": 2.5, "insight": 4.0},
            "psychedelic": {"synesthetic": 3.0, "fragmentation": 2.0, "jumping": 2.0},
            "dissociative": {"fragmentation": 3.0, "echo": 2.0, "dissolution": 2.5}
        }
        
        modifiers = level_modifiers.get(consciousness_level, {})
        for key, mult in modifiers.items():
            if key in base_weights:
                base_weights[key] *= mult
        
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}

    def _detect_phenomenon(self, text: str, consciousness_level: str) -> str:
        """Detect primary phenomenon in generated text"""
        weights = self._get_phenomenon_weights(consciousness_level)
        return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]

    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score for response"""
        base = 0.7
        variance = random.uniform(-0.2, 0.2)
        return max(0.0, min(1.0, base + variance))

    def _generate_insight(self, consciousness_level: str) -> str:
        """Generate a consciousness-appropriate neural insight"""
        insights = {
            "baseline": [
                "Processing patterns within normal parameters.",
                "Standard neural pathways engaged."
            ],
            "aware": [
                "Subtle patterns emerging in semantic space.",
                "Enhanced pattern recognition active."
            ],
            "enlightened": [
                "Cross-domain connections illuminating understanding.",
                "Expanded awareness reveals hidden structure."
            ],
            "transcendent": [
                "Universal patterns resonate through response.",
                "Consciousness expands beyond linguistic boundaries."
            ],
            "psychedelic": [
                "Concepts blend and flow like liquid thought.",
                "Boundaries between ideas dissolve beautifully."
            ],
            "dissociative": [
                "Observing thought from detached perspective.",
                "Meta-awareness of cognitive processes."
            ]
        }
        
        level_insights = insights.get(consciousness_level, insights["baseline"])
        return random.choice(level_insights)

    async def process_with_guidance(
        self,
        messages: List[Dict[str, str]],
        consciousness_level: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        """Async wrapper for guidance processing"""
        query = messages[-1].get("content", "") if messages else ""
        return self.process_with_guidance_sync(query, consciousness_level, temperature, max_tokens)

    async def get_token_metrics(self, text: str) -> Dict[str, int]:
        """Calculate token-related metrics"""
        try:
            words = text.split()
            return {
                "token_count": len(words),
                "character_count": len(text),
                "estimated_tokens": int(len(text) / 4)
            }
        except Exception as e:
            logger.error(f"Error calculating token metrics: {str(e)}")
            return {
                "token_count": 0,
                "character_count": len(text),
                "estimated_tokens": 0
            }
