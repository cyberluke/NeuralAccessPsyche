import logging
import json
import numpy as np
from typing import Dict, Any, List
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class NRAMConfigSuggester:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.performance_history = []
        self.current_config = {
            "memory_size": 1024,
            "entropy_factor": 0.3,
            "consciousness_levels": {
                "baseline": 0.3,
                "aware": 0.5,
                "enlightened": 0.7,
                "transcendent": 0.9
            }
        }

    async def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current NRAM performance metrics"""
        try:
            # Add metrics to history
            self.performance_history.append(metrics)
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)

            # Calculate performance indicators
            avg_pattern_intensity = np.mean([m["pattern_intensity"] for m in self.performance_history[-10:]])
            state_stability = np.std([m["state_value"] for m in self.performance_history[-10:]])
            consciousness_distribution = self._analyze_consciousness_distribution()

            performance_data = {
                "avg_pattern_intensity": float(avg_pattern_intensity),
                "state_stability": float(state_stability),
                "consciousness_distribution": consciousness_distribution,
                "current_config": self.current_config
            }

            # Get AI suggestions
            suggestions = await self._get_ai_suggestions(performance_data)

            return {
                "performance_metrics": performance_data,
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {"error": str(e)}

    def _analyze_consciousness_distribution(self) -> Dict[str, float]:
        """Analyze distribution of consciousness levels"""
        try:
            recent_states = self.performance_history[-20:]
            total = len(recent_states)
            if total == 0:
                return {"baseline": 0.25, "aware": 0.25, "enlightened": 0.25, "transcendent": 0.25}

            distribution = {
                "baseline": 0,
                "aware": 0,
                "enlightened": 0,
                "transcendent": 0
            }

            for state in recent_states:
                level = state.get("consciousness_level", "baseline")
                distribution[level] = distribution.get(level, 0) + 1

            return {k: v/total for k, v in distribution.items()}
        except Exception as e:
            logger.error(f"Error analyzing consciousness distribution: {str(e)}")
            return {"baseline": 0.25, "aware": 0.25, "enlightened": 0.25, "transcendent": 0.25}

    async def _get_ai_suggestions(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered configuration suggestions"""
        try:
            prompt = {
                "role": "system",
                "content": """You are an expert NRAM configuration advisor. Analyze the provided performance metrics 
                and suggest optimal configurations. Focus on these aspects:
                1. Memory size and entropy factor
                2. Consciousness level thresholds
                3. Pattern recognition parameters
                Provide specific numerical suggestions with explanations.
                Response should be in JSON format with 'suggestions' and 'explanations' keys."""
            }

            # Add performance data to prompt
            user_prompt = {
                "role": "user",
                "content": f"Based on these performance metrics, suggest optimal NRAM configurations:\n{json.dumps(performance_data, indent=2)}"
            }

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[prompt, user_prompt],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            suggestions = json.loads(response.choices[0].message.content)
            return suggestions
        except Exception as e:
            logger.error(f"Error getting AI suggestions: {str(e)}")
            return {
                "suggestions": {"error": "Failed to generate suggestions"},
                "explanations": {"error": str(e)}
            }

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update current NRAM configuration"""
        try:
            # Validate new configuration
            if "memory_size" in new_config:
                self.current_config["memory_size"] = max(256, min(4096, new_config["memory_size"]))
            if "entropy_factor" in new_config:
                self.current_config["entropy_factor"] = max(0.1, min(0.9, new_config["entropy_factor"]))
            if "consciousness_levels" in new_config:
                for level, value in new_config["consciousness_levels"].items():
                    if level in self.current_config["consciousness_levels"]:
                        self.current_config["consciousness_levels"][level] = max(0.1, min(0.9, value))

            return self.current_config
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return {"error": str(e)}
