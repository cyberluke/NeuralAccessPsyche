"""
Multi-vector representation control for NRAM.

Combines multiple activation vectors simultaneously with configurable
combination modes:
- sum: simple weighted sum
- normalized_sum: sum then normalize to unit norm
- orthogonalized_sum: Gram-Schmidt orthogonalization before sum
- norm_budgeted_sum: clamp total intervention norm to budget

All operations execute on live hidden states during forward pass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CombinationMode(str, Enum):
    """How to combine multiple vectors."""
    SUM = "sum"
    NORMALIZED_SUM = "normalized_sum"
    ORTHOGONALIZED_SUM = "orthogonalized_sum"
    NORM_BUDGETED_SUM = "norm_budgeted_sum"


@dataclass
class VectorEntry:
    """A single vector with coefficient."""
    vector_id: str
    coefficient: float
    vector: torch.Tensor  # (hidden_dim,)
    
    def validate(self) -> None:
        """Validate the entry."""
        if self.vector.dim() != 1:
            raise ValueError(f"Vector must be 1D, got shape {self.vector.shape}")


@dataclass
class CombinationResult:
    """Result of combining multiple vectors."""
    combined_vector: torch.Tensor  # (hidden_dim,)
    mode: CombinationMode
    individual_norms: List[float]
    combined_norm: float
    clamped: bool
    original_norm: Optional[float] = None


class MultiVectorController:
    """
    Controller for combining multiple activation vectors.
    
    Manages a set of vectors with coefficients and combines them
    according to the specified mode. The combined vector is then
    applied to hidden states during forward pass.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._entries: Dict[str, VectorEntry] = {}
        self._mode = CombinationMode.SUM
        self._norm_budget: Optional[float] = None
        self._per_layer_norm_clamp: Optional[float] = None
    
    def set_mode(self, mode: CombinationMode) -> None:
        """Set the combination mode."""
        self._mode = mode
        logger.info(f"Set combination mode to {mode.value}")
    
    def set_norm_budget(self, budget: Optional[float]) -> None:
        """Set total intervention norm budget."""
        self._norm_budget = budget
    
    def set_per_layer_norm_clamp(self, clamp: Optional[float]) -> None:
        """Set per-layer norm clamp."""
        self._per_layer_norm_clamp = clamp
    
    def add_vector(
        self,
        vector_id: str,
        coefficient: float,
        vector: torch.Tensor,
    ) -> None:
        """Add a vector with coefficient."""
        entry = VectorEntry(
            vector_id=vector_id,
            coefficient=coefficient,
            vector=vector.to(self.device),
        )
        entry.validate()
        self._entries[vector_id] = entry
        logger.info(f"Added vector '{vector_id}' with coefficient {coefficient}")
    
    def remove_vector(self, vector_id: str) -> None:
        """Remove a vector."""
        if vector_id in self._entries:
            del self._entries[vector_id]
            logger.info(f"Removed vector '{vector_id}'")
    
    def clear_vectors(self) -> None:
        """Remove all vectors."""
        self._entries.clear()
    
    def combine(self) -> Optional[CombinationResult]:
        """
        Combine all vectors according to the current mode.
        
        Returns:
            CombinationResult with the combined vector, or None if no vectors
        """
        if not self._entries:
            return None
        
        # Sort by vector_id for deterministic ordering
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.vector_id)
        
        # Collect vectors and coefficients
        vectors = [e.vector for e in sorted_entries]
        coefficients = [e.coefficient for e in sorted_entries]
        individual_norms = [float(v.norm().item()) for v in vectors]
        
        # Combine based on mode
        if self._mode == CombinationMode.SUM:
            combined = self._combine_sum(vectors, coefficients)
        elif self._mode == CombinationMode.NORMALIZED_SUM:
            combined = self._combine_normalized_sum(vectors, coefficients)
        elif self._mode == CombinationMode.ORTHOGONALIZED_SUM:
            combined = self._combine_orthogonalized_sum(vectors, coefficients)
        elif self._mode == CombinationMode.NORM_BUDGETED_SUM:
            combined = self._combine_norm_budgeted_sum(vectors, coefficients)
        else:
            raise ValueError(f"Unknown combination mode: {self._mode}")
        
        combined_norm = float(combined.norm().item())
        original_norm = combined_norm
        clamped = False
        
        # Apply per-layer norm clamp
        if self._per_layer_norm_clamp is not None and combined_norm > self._per_layer_norm_clamp:
            scale = self._per_layer_norm_clamp / combined_norm
            combined = combined * scale
            combined_norm = float(combined.norm().item())
            clamped = True
        
        # Apply total norm budget
        if self._norm_budget is not None and combined_norm > self._norm_budget:
            scale = self._norm_budget / combined_norm
            combined = combined * scale
            combined_norm = float(combined.norm().item())
            clamped = True
        
        return CombinationResult(
            combined_vector=combined,
            mode=self._mode,
            individual_norms=individual_norms,
            combined_norm=combined_norm,
            clamped=clamped,
            original_norm=original_norm,
        )
    
    def _combine_sum(
        self,
        vectors: List[torch.Tensor],
        coefficients: List[float],
    ) -> torch.Tensor:
        """Simple weighted sum."""
        result = torch.zeros_like(vectors[0])
        for vec, coeff in zip(vectors, coefficients):
            result = result + coeff * vec
        return result
    
    def _combine_normalized_sum(
        self,
        vectors: List[torch.Tensor],
        coefficients: List[float],
    ) -> torch.Tensor:
        """Sum then normalize to unit norm."""
        summed = self._combine_sum(vectors, coefficients)
        norm = summed.norm()
        if norm > 0:
            return summed / norm
        return summed
    
    def _combine_orthogonalized_sum(
        self,
        vectors: List[torch.Tensor],
        coefficients: List[float],
    ) -> torch.Tensor:
        """Gram-Schmidt orthogonalization before sum."""
        if not vectors:
            return torch.zeros(1, device=self.device)
        
        # Stack vectors: (n_vectors, hidden_dim)
        stacked = torch.stack(vectors, dim=0)
        
        # Gram-Schmidt orthogonalization
        orthogonal = []
        for i in range(stacked.shape[0]):
            v = stacked[i]
            for u in orthogonal:
                # Project out component along u
                proj = torch.dot(v, u) / torch.dot(u, u)
                v = v - proj * u
            norm = v.norm()
            if norm > 1e-8:
                orthogonal.append(v / norm)
        
        if not orthogonal:
            return torch.zeros_like(vectors[0])
        
        # Sum orthogonalized vectors with coefficients
        result = torch.zeros_like(vectors[0])
        for i, (ortho_vec, coeff) in enumerate(zip(orthogonal, coefficients)):
            if i < len(coefficients):
                result = result + coeff * ortho_vec
        
        return result
    
    def _combine_norm_budgeted_sum(
        self,
        vectors: List[torch.Tensor],
        coefficients: List[float],
    ) -> torch.Tensor:
        """Sum with norm budget clamping."""
        summed = self._combine_sum(vectors, coefficients)
        
        if self._norm_budget is not None:
            norm = summed.norm()
            if norm > self._norm_budget:
                scale = self._norm_budget / norm
                summed = summed * scale
        
        return summed
    
    def apply_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        strength: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[CombinationResult]]:
        """
        Combine vectors and apply to hidden states.
        
        Args:
            hidden_states: Current hidden states (batch, seq_len, hidden_dim)
            strength: Overall scaling factor
        
        Returns:
            Tuple of (modified_hidden_states, combination_result)
        """
        result = self.combine()
        if result is None:
            return hidden_states, None
        
        # Broadcast combined vector to hidden states shape
        # (hidden_dim,) -> (1, 1, hidden_dim)
        vector_expanded = result.combined_vector.unsqueeze(0).unsqueeze(0)
        vector_expanded = vector_expanded.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Apply: h' = h + strength * combined_vector
        delta = strength * vector_expanded
        modified = hidden_states + delta
        
        return modified, result
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the controller."""
        return {
            "mode": self._mode.value,
            "norm_budget": self._norm_budget,
            "per_layer_norm_clamp": self._per_layer_norm_clamp,
            "vectors": {
                vid: {
                    "coefficient": entry.coefficient,
                    "norm": float(entry.vector.norm().item()),
                }
                for vid, entry in self._entries.items()
            },
        }
