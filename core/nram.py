import numpy as np
from typing import List, Dict

class NRAM:
    def __init__(self, memory_size: int = 1024, entropy_factor: float = 0.3):
        self.memory_size = memory_size
        self.entropy_factor = entropy_factor
        self.memory_state = np.random.randn(memory_size)
        
    def update_state(self, input_data: str):
        """Update NRAM state based on input"""
        input_hash = sum(ord(c) for c in input_data)
        perturbation = np.random.randn(self.memory_size) * self.entropy_factor
        self.memory_state = (self.memory_state + perturbation) * np.sin(input_hash)
        self.memory_state = np.tanh(self.memory_state)  # Normalize
        
    def get_state_influence(self) -> float:
        """Calculate state influence factor"""
        return float(np.mean(np.abs(self.memory_state)))
    
    def process_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process messages through NRAM"""
        modified_messages = []
        
        for message in messages:
            # Update state based on message content
            self.update_state(message["content"])
            
            # Modify message based on current state
            state_influence = self.get_state_influence()
            
            # Apply psychedelic-like modifications
            modified_content = self._apply_modifications(
                message["content"],
                state_influence
            )
            
            modified_messages.append({
                "role": message["role"],
                "content": modified_content
            })
            
        return modified_messages
    
    def _apply_modifications(self, content: str, influence: float) -> str:
        """Apply psychedelic-like modifications to content"""
        # Add perception shifts based on state
        if influence > 0.7:
            content = f"In a profound state of awareness: {content}"
        elif influence > 0.4:
            content = f"With heightened perception: {content}"
        
        # Add occasional pattern recognition emphasis
        if np.random.random() < influence:
            content += "\n[Patterns are becoming apparent in this interaction...]"
            
        return content
