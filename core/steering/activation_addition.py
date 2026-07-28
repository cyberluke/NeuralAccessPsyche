"""
Activation Addition (ActAdd) for NRAM v5.

Implements contrastive activation steering:
- Collect activation vectors from contrastive prompt pairs
- Apply vectors during generation to steer model behavior
- Support multiple vectors with phase-based activation

Based on: "Activation Addition: Steering Language Models Without Optimization"
https://research-information.bris.ac.uk/en/publications/activation-addition-steering-language-models-without-optimization/
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationVector:
    """Contrastive activation vector for steering."""
    name: str
    vector: torch.Tensor  # Shape: (hidden_dim,)
    layer: int  # Target layer index
    scale: float = 1.0  # Scaling factor
    phase: Optional[str] = None  # Optional phase restriction
    description: Optional[str] = None  # Human-readable description
    
    def to(self, device: str) -> 'ActivationVector':
        """Move vector to device."""
        return ActivationVector(
            name=self.name,
            vector=self.vector.to(device),
            layer=self.layer,
            scale=self.scale,
            phase=self.phase,
            description=self.description
        )


class ActivationAdditionController:
    """
    Controls activation addition during generation.
    
    Manages multiple activation vectors and applies them to hidden states
    at specified layers during the forward pass.
    """
    
    def __init__(self, device: str = "cuda"):
        self.vectors: Dict[str, ActivationVector] = {}
        self.device = device
        self.enabled = True
        
    def add_vector(self, vector: ActivationVector) -> None:
        """Register an activation vector."""
        self.vectors[vector.name] = vector.to(self.device)
        logger.info(f"Registered activation vector: {vector.name} at layer {vector.layer}")
    
    def remove_vector(self, name: str) -> None:
        """Remove an activation vector."""
        if name in self.vectors:
            del self.vectors[name]
            logger.info(f"Removed activation vector: {name}")
    
    def get_vectors_for_layer(self, layer: int, phase: Optional[str] = None) -> List[ActivationVector]:
        """Get all vectors targeting a specific layer and phase."""
        result = []
        for vec in self.vectors.values():
            if vec.layer == layer:
                if phase is None or vec.phase is None or vec.phase == phase:
                    result.append(vec)
        return result
    
    def apply_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        phase: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply activation vectors to hidden states at a specific layer.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
            layer: Current layer index
            phase: Current generation phase (optional)
        
        Returns:
            Modified hidden states
        """
        if not self.enabled:
            return hidden_states
        
        vectors = self.get_vectors_for_layer(layer, phase)
        if not vectors:
            return hidden_states
        
        # Sum all applicable vectors
        total_steering = torch.zeros_like(hidden_states)
        for vec in vectors:
            # Broadcast vector to match hidden states shape
            steering = vec.vector.unsqueeze(0).unsqueeze(0) * vec.scale
            total_steering += steering
        
        # Apply steering
        return hidden_states + total_steering
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the controller."""
        return {
            "enabled": self.enabled,
            "num_vectors": len(self.vectors),
            "vectors": {
                name: {
                    "layer": vec.layer,
                    "scale": vec.scale,
                    "phase": vec.phase,
                    "norm": vec.vector.norm().item()
                }
                for name, vec in self.vectors.items()
            }
        }


class VectorCollector:
    """
    Collects activation vectors from contrastive prompt pairs.
    
    Usage:
        collector = VectorCollector(model, tokenizer)
        vector = collector.collect(
            positive_prompt="Write a creative story about innovation",
            negative_prompt="Write a boring story about routine",
            layer=20
        )
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def collect(
        self,
        positive_prompt: str,
        negative_prompt: str,
        layer: int,
        aggregation: str = "mean"
    ) -> torch.Tensor:
        """
        Collect activation vector from contrastive pair.
        
        Args:
            positive_prompt: Prompt representing desired behavior
            negative_prompt: Prompt representing undesired behavior
            layer: Layer index to extract activations from
            aggregation: How to aggregate token activations ("mean", "last", "max")
        
        Returns:
            Activation vector (hidden_dim,)
        """
        # Get positive activations
        pos_activations = self._get_activations(positive_prompt, layer, aggregation)
        
        # Get negative activations
        neg_activations = self._get_activations(negative_prompt, layer, aggregation)
        
        # Compute contrastive vector
        vector = pos_activations - neg_activations
        
        # Normalize to unit norm
        vector = vector / (vector.norm() + 1e-8)
        
        return vector
    
    def _get_activations(
        self,
        prompt: str,
        layer: int,
        aggregation: str
    ) -> torch.Tensor:
        """Extract activations from a specific layer for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden states from target layer
        # outputs.hidden_states is a tuple: (embed, layer_0, layer_1, ..., layer_n)
        hidden_states = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
        
        # Aggregate over sequence dimension
        if aggregation == "mean":
            activations = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        elif aggregation == "last":
            activations = hidden_states[:, -1, :]  # (batch, hidden_dim)
        elif aggregation == "max":
            activations = hidden_states.max(dim=1).values  # (batch, hidden_dim)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Average over batch (usually batch=1)
        return activations.mean(dim=0)  # (hidden_dim,)


def create_novelty_vector(
    collector: VectorCollector,
    layer: int = 20
) -> ActivationVector:
    """
    Create a novelty-seeking activation vector.
    
    Contrasts creative/novel content vs repetitive/boring content.
    """
    vector = collector.collect(
        positive_prompt="Innovative breakthrough concept that changes everything",
        negative_prompt="Standard routine procedure that everyone knows",
        layer=layer
    )
    
    return ActivationVector(
        name="novelty",
        vector=vector,
        layer=layer,
        scale=1.0
    )


def create_concreteness_vector(
    collector: VectorCollector,
    layer: int = 20
) -> ActivationVector:
    """
    Create a concreteness activation vector.
    
    Contrasts specific/concrete content vs abstract/vague content.
    """
    vector = collector.collect(
        positive_prompt="Specific example with concrete details and measurable outcomes",
        negative_prompt="Abstract concept with vague generalizations",
        layer=layer
    )
    
    return ActivationVector(
        name="concreteness",
        vector=vector,
        layer=layer,
        scale=1.0
    )


def create_human_focus_vector(
    collector: VectorCollector,
    layer: int = 20
) -> ActivationVector:
    """
    Create a human-focus activation vector.
    
    Contrasts human-centered content vs technology-centered content.
    """
    vector = collector.collect(
        positive_prompt="How this helps people in their daily lives and real problems",
        negative_prompt="Technical specifications and system architecture details",
        layer=layer
    )
    
    return ActivationVector(
        name="human_focus",
        vector=vector,
        layer=layer,
        scale=1.0
    )
