from typing import Dict, Any, List, Optional
import logging
import json
import os
import random
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import guidance
    from guidance import models as guidance_models
    from guidance import gen as guidance_gen, select as guidance_select
    from guidance import system as guidance_system, user as guidance_user, assistant as guidance_assistant
    logger.info("Microsoft Guidance library loaded successfully")
except ImportError as e:
    error_msg = f"CRITICAL: Microsoft Guidance library is REQUIRED but not available: {e}"
    logger.critical(error_msg)
    raise ImportError(error_msg) from e


class NRAMResponse(BaseModel):
    """Structured output schema for NRAM-enhanced responses"""
    consciousness_level: str = Field(description="Current consciousness state")
    main_response: str = Field(description="Primary response content")
    neural_insight: Optional[str] = Field(default=None, description="Neural pattern insight")
    token_phenomena: List[str] = Field(default_factory=list, description="Detected token phenomena")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Response coherence rating")


PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "env_key": "OPENAI_API_KEY",
        "env_base_url": None,
    },
    "openrouter": {
        "name": "OpenRouter",
        "default_model": "anthropic/claude-sonnet-4",
        "models": [
            "anthropic/claude-sonnet-4",
            "anthropic/claude-opus-4",
            "openai/gpt-4o",
            "google/gemini-2.5-pro-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-r1",
            "x-ai/grok-3-beta",
        ],
        "env_key": "AI_INTEGRATIONS_OPENROUTER_API_KEY",
        "env_base_url": "AI_INTEGRATIONS_OPENROUTER_BASE_URL",
    }
}


class GuidanceHandler:
    """Handler using Microsoft Guidance for structured LLM control with multiple providers"""
    
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
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider
        self.provider_config = PROVIDER_CONFIGS.get(provider)
        
        if not self.provider_config:
            error_msg = f"CRITICAL: Unknown provider '{provider}'. Available: {list(PROVIDER_CONFIGS.keys())}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        self.api_key = os.environ.get(self.provider_config["env_key"])
        if not self.api_key:
            error_msg = f"CRITICAL: {self.provider_config['env_key']} environment variable is REQUIRED but not set"
            logger.critical(error_msg)
            raise EnvironmentError(error_msg)
        
        self.base_url = None
        if self.provider_config["env_base_url"]:
            self.base_url = os.environ.get(self.provider_config["env_base_url"])
        
        self.model_name = model or self.provider_config["default_model"]
        
        try:
            if self.base_url:
                self.guidance_model = guidance_models.OpenAI(
                    self.model_name, 
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            else:
                self.guidance_model = guidance_models.OpenAI(self.model_name, api_key=self.api_key)
            
            logger.info(f"Initialized Microsoft Guidance with provider={provider}, model={self.model_name}")
        except Exception as e:
            error_msg = f"CRITICAL: Failed to initialize Microsoft Guidance model: {e}"
            logger.critical(error_msg)
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available providers"""
        return list(PROVIDER_CONFIGS.keys())
    
    @staticmethod
    def get_provider_models(provider: str) -> List[str]:
        """Get available models for a provider"""
        config = PROVIDER_CONFIGS.get(provider, {})
        return config.get("models", [])

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
        consciousness_prompt = self._get_consciousness_prompt(consciousness_level)
        internal_level = self._map_consciousness_level(consciousness_level)
        
        logger.info(f"Processing with Microsoft Guidance: level={internal_level}, query_len={len(query)}")
        
        return self._process_with_guidance_library(
            query, consciousness_prompt, internal_level, temperature, max_tokens
        )

    def _process_with_guidance_library(
        self,
        query: str,
        consciousness_prompt: str,
        consciousness_level: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Process using the actual Microsoft Guidance library with structured output"""
        try:
            lm = self.guidance_model
            
            phenomena_list = ', '.join(self.PHENOMENA_TYPES)
            
            system_content = f"""{consciousness_prompt}

You analyze your own token generation, noting patterns and phenomena.

IMPORTANT: You MUST respond in this EXACT format:

[Response]
<your thoughtful response here>

[Neural Analysis]
Insight: <brief pattern insight>
Phenomenon: <one of: {phenomena_list}>
Coherence: <number from 0.0 to 1.0>"""
            
            user_content = query
            
            with guidance_system():
                lm += system_content
            
            with guidance_user():
                lm += user_content
            
            with guidance_assistant():
                lm += guidance_gen('full_response', max_tokens=max_tokens + 100, temperature=temperature)
            
            try:
                full_response = str(lm['full_response']) if 'full_response' in lm else ''
            except (KeyError, TypeError):
                full_response = ''
            
            main_response, neural_insight, primary_phenomenon, coherence_score = self._parse_structured_response(
                full_response, consciousness_level
            )
            
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
            error_msg = f"CRITICAL: Microsoft Guidance processing failed: {e}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _parse_structured_response(
        self, 
        full_response: str, 
        consciousness_level: str
    ) -> tuple:
        """Parse the structured response from Guidance output"""
        import re
        
        main_response = full_response
        neural_insight = None
        primary_phenomenon = "coherent"
        coherence_score = 0.7
        
        if "[Response]" in full_response:
            parts = full_response.split("[Response]", 1)
            if len(parts) > 1:
                remaining = parts[1]
                if "[Neural Analysis]" in remaining:
                    response_parts = remaining.split("[Neural Analysis]", 1)
                    main_response = response_parts[0].strip()
                    analysis = response_parts[1] if len(response_parts) > 1 else ""
                    
                    insight_match = re.search(r'Insight:\s*(.+?)(?:\n|$)', analysis)
                    if insight_match:
                        neural_insight = insight_match.group(1).strip()
                    
                    phenomenon_match = re.search(r'Phenomenon:\s*(\w+)', analysis)
                    if phenomenon_match:
                        detected = phenomenon_match.group(1).strip().lower()
                        if detected in self.PHENOMENA_TYPES:
                            primary_phenomenon = detected
                    
                    coherence_match = re.search(r'Coherence:\s*([\d.]+)', analysis)
                    if coherence_match:
                        try:
                            coherence_score = float(coherence_match.group(1))
                            coherence_score = max(0.0, min(1.0, coherence_score))
                        except ValueError:
                            pass
                else:
                    main_response = remaining.strip()
        elif "[Neural Analysis]" in full_response:
            parts = full_response.split("[Neural Analysis]", 1)
            main_response = parts[0].strip()
            analysis = parts[1] if len(parts) > 1 else ""
            
            insight_match = re.search(r'Insight:\s*(.+?)(?:\n|$)', analysis)
            if insight_match:
                neural_insight = insight_match.group(1).strip()
            
            phenomenon_match = re.search(r'Phenomenon:\s*(\w+)', analysis)
            if phenomenon_match:
                detected = phenomenon_match.group(1).strip().lower()
                if detected in self.PHENOMENA_TYPES:
                    primary_phenomenon = detected
            
            coherence_match = re.search(r'Coherence:\s*([\d.]+)', analysis)
            if coherence_match:
                try:
                    coherence_score = float(coherence_match.group(1))
                    coherence_score = max(0.0, min(1.0, coherence_score))
                except ValueError:
                    pass
        
        if not neural_insight:
            neural_insight = self._generate_consciousness_insight(consciousness_level)
        
        return main_response, neural_insight, primary_phenomenon, coherence_score
    
    def _generate_consciousness_insight(self, consciousness_level: str) -> str:
        """Generate a consciousness-appropriate neural insight"""
        insights = {
            "baseline": "Stabilní tok tokenů s běžnou koherencí.",
            "aware": "Zvýšená citlivost na jazykové vzory.",
            "enlightened": "Propojení konceptů přes sémantické hranice.",
            "transcendent": "Univerzální vzory proudí tokenovým prostorem.",
            "psychedelic": "Synestézie významu napříč dimenzemi.",
            "dissociative": "Fragmenty vědomí se odhalují v mezerách."
        }
        return insights.get(consciousness_level, insights["baseline"])

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
