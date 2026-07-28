"""
Multi-vector controller for combining multiple activation vectors.

This module provides sophisticated control over multiple activation vectors,
including weighted combination, phase-based activation, and conflict resolution.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.steering.activation_addition import ActivationVector

logger = logging.getLogger(__name__)


class CombinationMode(Enum):
    """How to combine multiple vectors."""
    ADDITIVE = "additive"  # Simple weighted sum
    INTERPOLATION = "interpolation"  # Spherical interpolation
    PROJECTED = "projected"  # Project onto orthogonal subspace


@dataclass
class VectorConfig:
    """Configuration for a single vector in multi-vector control."""
    vector: ActivationVector
    weight: float = 1.0
    active_phases: Optional[List[str]] = None  # None = always active
    priority: int = 0  # Higher priority vectors applied first


@dataclass
class MultiVectorState:
    """Current state of multi-vector controller."""
    active_vectors: List[str] = field(default_factory=list)
    total_weight: float = 0.0
    phase: Optional[str] = None
    combination_mode: CombinationMode = CombinationMode.ADDITIVE


class MultiVectorController:
    """
    Advanced controller for managing multiple activation vectors.
    
    Features:
    - Weighted combination of multiple vectors
    - Phase-based activation (different vectors for different phases)
    - Conflict resolution (orthogonalization)
    - Dynamic weight adjustment
    """
    
    def __init__(
        self,
        device: str = "cuda",
        combination_mode: CombinationMode = CombinationMode.ADDITIVE
    ):
        self.device = device
        self.combination_mode = combination_mode
        self.vector_configs: Dict[str, VectorConfig] = {}
        self.current_phase: Optional[str] = None
        self.global_weight: float = 1.0
        
    def add_vector(
        self,
        name: str,
        vector: ActivationVector,
        weight: float = 1.0,
        active_phases: Optional[List[str]] = None,
        priority: int = 0
    ) -> None:
        """Add a vector with configuration."""
        config = VectorConfig(
            vector=vector,
            weight=weight,
            active_phases=active_phases,
            priority=priority
        )
        self.vector_configs[name] = config
        logger.info(f"Added vector '{name}' with weight {weight}")
    
    def remove_vector(self, name: str) -> None:
        """Remove a vector."""
        if name in self.vector_configs:
            del self.vector_configs[name]
            logger.info(f"Removed vector '{name}'")
    
    def set_phase(self, phase: Optional[str]) -> None:
        """Set current generation phase."""
        self.current_phase = phase
    
    def set_global_weight(self, weight: float) -> None:
        """Set global weight multiplier for all vectors."""
        self.global_weight = weight
    
    def get_active_vectors(self) -> List[Tuple[str, VectorConfig]]:
        """Get vectors active for current phase, sorted by priority."""
        active = []
        for name, config in self.vector_configs.items():
            # Check if vector is active in current phase
            if config.active_phases is None:
                active.append((name, config))
            elif self.current_phase in config.active_phases:
                active.append((name, config))
        
        # Sort by priority (descending)
        active.sort(key=lambda x: x[1].priority, reverse=True)
        return active
    
    def combine_vectors(
        self,
        hidden_dim: int,
        layer: int
    ) -> Optional[torch.Tensor]:
        """
        Combine all active vectors for a given layer.
        
        Returns combined steering vector or None if no vectors active.
        """
        active = self.get_active_vectors()
        if not active:
            return None
        
        # Filter vectors for this layer
        layer_vectors = [
            (name, config) for name, config in active
            if config.vector.layer == layer
        ]
        
        if not layer_vectors:
            return None
        
        if self.combination_mode == CombinationMode.ADDITIVE:
            return self._combine_additive(layer_vectors, hidden_dim)
        elif self.combination_mode == CombinationMode.INTERPOLATION:
            return self._combine_interpolation(layer_vectors, hidden_dim)
        elif self.combination_mode == CombinationMode.PROJECTED:
            return self._combine_projected(layer_vectors, hidden_dim)
        else:
            raise ValueError(f"Unknown combination mode: {self.combination_mode}")
    
    def _combine_additive(
        self,
        vectors: List[Tuple[str, VectorConfig]],
        hidden_dim: int
    ) -> torch.Tensor:
        """Simple weighted sum of vectors."""
        combined = torch.zeros(hidden_dim, device=self.device)
        
        for name, config in vectors:
            weight = config.weight * self.global_weight
            combined += config.vector.vector * weight
        
        return combined
    
    def _combine_interpolation(
        self,
        vectors: List[Tuple[str, VectorConfig]],
        hidden_dim: int
    ) -> torch.Tensor:
        """Spherical linear interpolation (SLERP) between vectors."""
        if len(vectors) == 1:
            name, config = vectors[0]
            return config.vector.vector * config.weight * self.global_weight
        
        # Normalize all vectors
        normalized = []
        weights = []
        for name, config in vectors:
            vec = config.vector.vector
            norm = vec.norm()
            if norm > 1e-8:
                normalized.append(vec / norm)
                weights.append(config.weight)
        
        if not normalized:
            return torch.zeros(hidden_dim, device=self.device)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight < 1e-8:
            return torch.zeros(hidden_dim, device=self.device)
        
        weights = [w / total_weight for w in weights]
        
        # SLERP between first two vectors, then mix in others
        result = normalized[0]
        for i in range(1, len(normalized)):
            result = self._slerp(result, normalized[i], weights[i])
        
        return result * self.global_weight
    
    def _slerp(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Spherical linear interpolation between two vectors."""
        # Compute cosine similarity
        cos_omega = torch.dot(v1, v2)
        cos_omega = torch.clamp(cos_omega, -1.0, 1.0)
        
        # If vectors are very close, use linear interpolation
        if cos_omega > 0.999:
            return (1 - t) * v1 + t * v2
        
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        
        return (torch.sin((1 - t) * omega) / sin_omega) * v1 + \
               (torch.sin(t * omega) / sin_omega) * v2
    
    def _combine_projected(
        self,
        vectors: List[Tuple[str, VectorConfig]],
        hidden_dim: int
    ) -> torch.Tensor:
        """Combine vectors with orthogonalization to reduce interference."""
        if len(vectors) == 1:
            name, config = vectors[0]
            return config.vector.vector * config.weight * self.global_weight
        
        # Sort by priority
        vectors_sorted = sorted(vectors, key=lambda x: x[1].priority, reverse=True)
        
        # Gram-Schmidt orthogonalization
        orthogonal = []
        for name, config in vectors_sorted:
            vec = config.vector.vector.clone()
            
            # Subtract projection onto all previous vectors
            for prev_vec in orthogonal:
                projection = torch.dot(vec, prev_vec) / torch.dot(prev_vec, prev_vec)
                vec = vec - projection * prev_vec
            
            # Normalize
            norm = vec.norm()
            if norm > 1e-8:
                orthogonal.append(vec / norm)
        
        # Combine orthogonalized vectors with weights
        combined = torch.zeros(hidden_dim, device=self.device)
        for i, (name, config) in enumerate(vectors_sorted):
            if i < len(orthogonal):
                weight = config.weight * self.global_weight
                combined += orthogonal[i] * weight
        
        return combined
    
    def apply_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        """
        Apply combined steering to hidden states.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
            layer: Current layer index
        
        Returns:
            Modified hidden states
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        combined = self.combine_vectors(hidden_dim, layer)
        if combined is None:
            return hidden_states
        
        # Broadcast to match hidden states shape
        steering = combined.unsqueeze(0).unsqueeze(0)
        return hidden_states + steering
    
    def get_status(self) -> Dict:
        """Get current controller status."""
        active = self.get_active_vectors()
        
        return {
            "device": self.device,
            "combination_mode": self.combination_mode.value,
            "current_phase": self.current_phase,
            "global_weight": self.global_weight,
            "total_vectors": len(self.vector_configs),
            "active_vectors": len(active),
            "vectors": {
                name: {
                    "weight": config.weight,
                    "layer": config.vector.layer,
                    "priority": config.priority,
                    "active_phases": config.active_phases,
                    "is_active": any(n == name for n, _ in active)
                }
                for name, config in self.vector_configs.items()
            }
        }
    
    def create_preset(self, name: str) -> Dict:
        """
        Create a preset configuration for common use cases.
        
        Presets:
        - "visionary": High novelty, high concreteness, human focus
        - "analytical": High concreteness, low novelty, technical focus
        - "creative": High novelty, moderate concreteness
        """
        presets = {
            "visionary": {
                "novelty": 1.2,
                "concreteness": 1.0,
                "human_focus": 1.1,
                "visionary": 1.3
            },
            "analytical": {
                "novelty": 0.5,
                "concreteness": 1.3,
                "human_focus": 0.7,
                "technical": 1.2
            },
            "creative": {
                "novelty": 1.5,
                "concreteness": 0.8,
                "human_focus": 1.0
            }
        }
        
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        
        return presets[name]
    
    def apply_preset(self, preset_name: str) -> None:
        """Apply a preset configuration."""
        preset = self.create_preset(preset_name)
        
        for vector_name, weight in preset.items():
            if vector_name in self.vector_configs:
                self.vector_configs[vector_name].weight = weight
        
        logger.info(f"Applied preset '{preset_name}'")
