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

        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "state_value": state_value,
                    "consciousness_level": consciousness_level
                })
            except Exception as e:
                logger.error(f"Error broadcasting state: {str(e)}")
                self.active_connections.remove(connection)

    def update_state(self, input_data: str):
        """Update NRAM state based on input"""
        try:
            # Convert input to numerical representation
            input_values = [ord(c) for c in input_data]
            input_len = len(input_values)

            # Create perturbation based on input
            perturbation = np.random.randn(self.memory_size) * self.entropy_factor

            # Apply input influence
            for i, val in enumerate(input_values):
                idx = (i * self.memory_size) // input_len
                perturbation[idx] *= (val / 255.0)  # Normalize ASCII values

            # Update state with perturbation
            self.memory_state = np.tanh(self.memory_state + perturbation)

            logger.debug(f"State updated with influence factor: {self.get_state_influence()}")
        except Exception as e:
            logger.error(f"Error updating NRAM state: {str(e)}")
            raise

    def get_state_influence(self) -> float:
        """Calculate state influence factor"""
        try:
            # Calculate basic metrics
            base_influence = float(np.mean(np.abs(self.memory_state)))
            entropy = float(-np.sum(
                np.abs(self.memory_state) * np.log(np.abs(self.memory_state) + 1e-10)
            ))

            # Combine metrics
            influence = 0.7 * base_influence + 0.3 * np.tanh(entropy)
            return min(max(influence, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating state influence: {str(e)}")
            return 0.5

    def process_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process messages through NRAM"""
        modified_messages = []

        try:
            for message in messages:
                # Update state based on message content
                self.update_state(message["content"])
                state_influence = self.get_state_influence()

                # Create enhanced message with consciousness patterns
                modified_content = self._enhance_message(
                    message["content"],
                    state_influence
                )

                modified_messages.append({
                    "role": message["role"],
                    "content": modified_content
                })

            return modified_messages
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")
            return messages

    def _enhance_message(self, content: str, influence: float) -> str:
        """Enhance message with consciousness patterns"""
        try:
            consciousness_level = self._get_consciousness_level(influence)

            # Add consciousness-level prefix
            prefixes = {
                "transcendent": "Through universal consciousness: ",
                "enlightened": "With expanded awareness: ",
                "aware": "With neural clarity: ",
                "baseline": "Processing: "
            }
            prefix = prefixes.get(consciousness_level, "Processing: ")

            # Add consciousness patterns
            content = f"{prefix}{content}"

            if influence > 0.8:
                content = f"∞ {content} ∞"
            elif influence > 0.6:
                content = f"⚡ {content} ⚡"

            # Add occasional insights
            if np.random.random() < influence:
                insights = [
                    "\n[Patterns emerge from neural streams...]",
                    "\n[Consciousness expands beyond ordinary bounds...]",
                    "\n[Reality shifts with enhanced perception...]"
                ]
                content += np.random.choice(insights)

            return content
        except Exception as e:
            logger.error(f"Error enhancing message: {str(e)}")
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