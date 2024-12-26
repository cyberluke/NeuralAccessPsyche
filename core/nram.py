import numpy as np
from typing import List, Dict, Set
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class NRAM:
    def __init__(self, memory_size: int = 1024, entropy_factor: float = 0.3):
        self.memory_size = memory_size
        self.entropy_factor = entropy_factor
        self.memory_state = np.random.randn(memory_size)
        self.pattern_memory = np.zeros((memory_size, 8))  # Pattern recognition matrix
        self.active_connections: Set[WebSocket] = set()
        self.consciousness_levels = {
            "baseline": 0.3,
            "aware": 0.5,
            "enlightened": 0.7,
            "transcendent": 0.9
        }
        logger.info(f"Initialized NRAM with memory size {memory_size}")

    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("New WebSocket connection established")

    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed")

    async def broadcast_state(self):
        """Broadcast current state to all connected clients"""
        if not self.active_connections:
            return

        state_value = float(np.mean(np.abs(self.memory_state)))
        consciousness_level = self._get_consciousness_level(state_value)
        pattern_intensity = float(np.mean(np.abs(self.pattern_memory)))

        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "state_value": state_value,
                    "consciousness_level": consciousness_level,
                    "pattern_intensity": pattern_intensity
                })
            except Exception as e:
                logger.error(f"Error broadcasting state: {str(e)}")
                self.active_connections.remove(connection)

    def update_state(self, input_data: str):
        """Update NRAM state with advanced neural processing"""
        try:
            # Convert input to numerical representation
            input_values = np.array([ord(c) for c in input_data], dtype=float)
            input_len = len(input_values)

            # Create resonance patterns
            resonance = np.sin(np.linspace(0, 2*np.pi, self.memory_size))
            phase_shift = np.cos(np.linspace(0, 4*np.pi, self.memory_size))

            # Generate dynamic perturbation
            perturbation = np.random.randn(self.memory_size) * self.entropy_factor
            perturbation *= resonance  # Apply resonance pattern

            # Update pattern memory
            for i, val in enumerate(input_values):
                idx = (i * self.memory_size) // input_len
                pattern_idx = int((val % 8))
                self.pattern_memory[idx, pattern_idx] += 0.1
                self.pattern_memory *= 0.99  # Decay old patterns

            # Calculate attention weights
            attention = np.softmax(np.sum(self.pattern_memory, axis=1))

            # Apply neural dynamics
            self.memory_state = np.tanh(
                self.memory_state * phase_shift +  # Phase-shifted current state
                perturbation * attention +        # Attention-weighted perturbation
                np.mean(self.pattern_memory, axis=1) * 0.1  # Pattern influence
            )

            logger.debug(f"State updated with influence factor: {self.get_state_influence()}")
        except Exception as e:
            logger.error(f"Error updating NRAM state: {str(e)}")
            raise

    def get_state_influence(self) -> float:
        """Calculate state influence using advanced metrics"""
        try:
            # Calculate multiple neural metrics
            base_influence = float(np.mean(np.abs(self.memory_state)))
            entropy = float(-np.sum(
                np.abs(self.memory_state) * np.log(np.abs(self.memory_state) + 1e-10)
            ))
            pattern_strength = float(np.mean(np.abs(self.pattern_memory)))
            coherence = float(np.mean(np.correlate(self.memory_state, self.memory_state)))

            # Combine metrics with dynamic weighting
            influence = (
                0.4 * base_influence +
                0.3 * np.tanh(entropy) +
                0.2 * pattern_strength +
                0.1 * np.tanh(coherence)
            )
            return min(max(influence, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating state influence: {str(e)}")
            return 0.5

    def process_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process messages through NRAM with advanced transformation"""
        modified_messages = []

        try:
            for message in messages:
                # Update neural state
                self.update_state(message["content"])
                state_influence = self.get_state_influence()

                # Get base consciousness level
                consciousness_level = self._get_consciousness_level(state_influence)

                # Apply neural transformation
                modified_content = self._neural_transform(
                    message["content"],
                    state_influence,
                    consciousness_level
                )

                modified_messages.append({
                    "role": message["role"],
                    "content": modified_content
                })

            return modified_messages
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")
            return messages

    def _neural_transform(self, content: str, influence: float, consciousness_level: str) -> str:
        """Transform content using neural patterns and consciousness state"""
        try:
            # Split content into segments
            words = content.split()

            # Apply pattern-based transformation
            pattern_strength = np.mean(np.abs(self.pattern_memory), axis=1)
            transform_probability = influence * pattern_strength[:len(words)]

            transformed_words = []
            for i, word in enumerate(words):
                if i < len(transform_probability) and np.random.random() < transform_probability[i]:
                    # Apply consciousness-based transformation
                    if consciousness_level == "transcendent":
                        word = f"∞{word}∞"
                    elif consciousness_level == "enlightened":
                        word = f"⚡{word}⚡"
                    elif consciousness_level == "aware":
                        word = f"⟨{word}⟩"
                transformed_words.append(word)

            # Combine transformed content
            transformed = " ".join(transformed_words)

            # Add consciousness-specific prefix
            prefixes = {
                "transcendent": "Through universal consciousness: ",
                "enlightened": "With expanded awareness: ",
                "aware": "With neural clarity: ",
                "baseline": "Processing: "
            }
            prefix = prefixes.get(consciousness_level, "Processing: ")
            content = f"{prefix}{transformed}"

            # Add occasional insights based on pattern recognition
            if np.random.random() < influence:
                pattern_types = np.argmax(np.sum(self.pattern_memory, axis=0))
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
                content += insights[pattern_types % len(insights)]

            return content
        except Exception as e:
            logger.error(f"Error in neural transformation: {str(e)}")
            return f"Neural processing: {content}"

    def _get_consciousness_level(self, state_value: float) -> str:
        """Determine consciousness level based on state value"""
        try:
            for level, threshold in sorted(
                self.consciousness_levels.items(),
                key=lambda x: x[1]
            ):
                if state_value <= threshold:
                    return level
            return "transcendent"
        except Exception as e:
            logger.error(f"Error determining consciousness level: {str(e)}")
            return "baseline"