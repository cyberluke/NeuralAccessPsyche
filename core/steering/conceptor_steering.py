"""
Conceptor steering for NRAM v5.

Conceptors are matrix operators that can softly project activations into
subspaces, allowing smooth blending of multiple concepts without interference.

Based on: "Conceptors: An Overview" by H. Jaeger
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConceptorConfig:
    """Configuration for a conceptor."""
    name: str
    aperture: float = 2.0  # Controls softness of projection (0=hard, ∞=identity)
    layer: int = 0
    weight: float = 1.0
    active_phases: Optional[List[str]] = None


class Conceptor:
    """
    Conceptor matrix operator for soft subspace projection.
    
    A conceptor C with aperture α is defined as:
    C = R @ (R^T @ R + α^{-2} @ I)^{-1} @ R^T
    
    Where R is the pattern matrix (columns are pattern vectors).
    
    For single vector v, this simplifies to:
    C = v @ v^T / (v^T @ v + α^{-2})
    """
    
    def __init__(
        self,
        patterns: torch.Tensor,
        aperture: float = 2.0,
        device: str = "cuda"
    ):
        """
        Initialize conceptor.
        
        Args:
            patterns: Pattern matrix of shape (hidden_dim, num_patterns)
            aperture: Softness parameter (0=hard projection, ∞=identity)
            device: Device to use
        """
        self.device = device
        self.aperture = aperture
        self.patterns = patterns.to(device)
        
        # Compute conceptor matrix
        self.matrix = self._compute_conceptor_matrix()
    
    def _compute_conceptor_matrix(self) -> torch.Tensor:
        """Compute the conceptor matrix C."""
        hidden_dim = self.patterns.shape[0]
        
        # R^T @ R
        RtR = self.patterns.T @ self.patterns
        
        # Add regularization: R^T @ R + α^{-2} @ I
        reg = (self.aperture ** -2) * torch.eye(
            RtR.shape[0],
            device=self.device,
            dtype=RtR.dtype
        )
        
        # (R^T @ R + α^{-2} @ I)^{-1}
        inv = torch.linalg.inv(RtR + reg)
        
        # C = R @ (R^T @ R + α^{-2} @ I)^{-1} @ R^T
        C = self.patterns @ inv @ self.patterns.T
        
        return C
    
    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states through conceptor.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Projected hidden states
        """
        # Apply conceptor: h' = C @ h
        # Need to handle batch and sequence dimensions
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape to (batch_size * seq_len, hidden_dim)
        h_flat = hidden_states.reshape(-1, hidden_dim)
        
        # Apply conceptor: h' = h @ C^T (since C is symmetric, C^T = C)
        h_projected = h_flat @ self.matrix.T
        
        # Reshape back
        return h_projected.reshape(batch_size, seq_len, hidden_dim)
    
    def get_load_metric(self) -> float:
        """
        Compute load metric (trace of C / hidden_dim).
        
        Indicates how much of the space is used (0=empty, 1=full).
        """
        return self.matrix.trace().item() / self.matrix.shape[0]


class ConceptorController:
    """
    Controller for managing multiple conceptors.
    
    Supports:
    - Boolean operations (AND, OR, NOT)
    - Blending multiple conceptors
    - Phase-based activation
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.conceptors: Dict[str, Tuple[Conceptor, ConceptorConfig]] = {}
        self.current_phase: Optional[str] = None
    
    def add_conceptor(
        self,
        name: str,
        patterns: torch.Tensor,
        aperture: float = 2.0,
        layer: int = 0,
        weight: float = 1.0,
        active_phases: Optional[List[str]] = None
    ) -> None:
        """Add a conceptor."""
        config = ConceptorConfig(
            name=name,
            aperture=aperture,
            layer=layer,
            weight=weight,
            active_phases=active_phases
        )
        conceptor = Conceptor(patterns, aperture, device=self.device)
        self.conceptors[name] = (conceptor, config)
        logger.info(f"Added conceptor '{name}' with aperture {aperture}")
    
    def remove_conceptor(self, name: str) -> None:
        """Remove a conceptor."""
        if name in self.conceptors:
            del self.conceptors[name]
            logger.info(f"Removed conceptor '{name}'")
    
    def set_phase(self, phase: Optional[str]) -> None:
        """Set current generation phase."""
        self.current_phase = phase
    
    def get_active_conceptors(self) -> List[Tuple[str, Conceptor, ConceptorConfig]]:
        """Get conceptors active for current phase."""
        active = []
        for name, (conceptor, config) in self.conceptors.items():
            if config.active_phases is None or self.current_phase in config.active_phases:
                active.append((name, conceptor, config))
        return active
    
    def blend_conceptors(
        self,
        layer: int,
        mode: str = "weighted_sum"
    ) -> Optional[torch.Tensor]:
        """
        Blend multiple conceptors for a given layer.
        
        Args:
            layer: Target layer
            mode: Blending mode ("weighted_sum", "and", "or")
        
        Returns:
            Blended conceptor matrix or None
        """
        active = [
            (name, conceptor, config)
            for name, conceptor, config in self.get_active_conceptors()
            if config.layer == layer
        ]
        
        if not active:
            return None
        
        if mode == "weighted_sum":
            return self._blend_weighted_sum(active)
        elif mode == "and":
            return self._blend_and(active)
        elif mode == "or":
            return self._blend_or(active)
        else:
            raise ValueError(f"Unknown blending mode: {mode}")
    
    def _blend_weighted_sum(
        self,
        conceptors: List[Tuple[str, Conceptor, ConceptorConfig]]
    ) -> torch.Tensor:
        """Weighted sum of conceptor matrices."""
        matrices = []
        weights = []
        
        for name, conceptor, config in conceptors:
            matrices.append(conceptor.matrix)
            weights.append(config.weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight < 1e-8:
            return matrices[0]
        
        weights = [w / total_weight for w in weights]
        
        # Weighted sum
        blended = sum(w * m for w, m in zip(weights, matrices))
        return blended
    
    def _blend_and(
        self,
        conceptors: List[Tuple[str, Conceptor, ConceptorConfig]]
    ) -> torch.Tensor:
        """
        AND operation: intersection of subspaces.
        
        C_AND = C1 @ C2 @ ... @ Cn
        """
        result = conceptors[0][1].matrix
        for _, conceptor, _ in conceptors[1:]:
            result = result @ conceptor.matrix
        return result
    
    def _blend_or(
        self,
        conceptors: List[Tuple[str, Conceptor, ConceptorConfig]]
    ) -> torch.Tensor:
        """
        OR operation: union of subspaces.
        
        C_OR = I - (I - C1) @ (I - C2) @ ... @ (I - Cn)
        """
        hidden_dim = conceptors[0][1].matrix.shape[0]
        I = torch.eye(hidden_dim, device=self.device)
        
        result = I
        for _, conceptor, _ in conceptors:
            result = result @ (I - conceptor.matrix)
        
        return I - result
    
    def apply_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        blend_mode: str = "weighted_sum"
    ) -> torch.Tensor:
        """
        Apply blended conceptors to hidden states.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
            layer: Current layer
            blend_mode: How to blend conceptors
        
        Returns:
            Modified hidden states
        """
        blended = self.blend_conceptors(layer, blend_mode)
        if blended is None:
            return hidden_states
        
        # Apply blended conceptor
        batch_size, seq_len, hidden_dim = hidden_states.shape
        h_flat = hidden_states.reshape(-1, hidden_dim)
        h_projected = h_flat @ blended.T
        return h_projected.reshape(batch_size, seq_len, hidden_dim)
    
    def create_conceptor_from_vectors(
        self,
        name: str,
        vectors: List[torch.Tensor],
        aperture: float = 2.0,
        layer: int = 0,
        weight: float = 1.0,
        active_phases: Optional[List[str]] = None
    ) -> None:
        """
        Create conceptor from list of activation vectors.
        
        Args:
            name: Conceptor name
            vectors: List of activation vectors
            aperture: Softness parameter
            layer: Target layer
            weight: Blending weight
            active_phases: Phases when conceptor is active
        """
        # Stack vectors into pattern matrix
        patterns = torch.stack(vectors, dim=1)
        self.add_conceptor(name, patterns, aperture, layer, weight, active_phases)
    
    def get_status(self) -> Dict:
        """Get current controller status."""
        active = self.get_active_conceptors()
        
        return {
            "device": self.device,
            "current_phase": self.current_phase,
            "total_conceptors": len(self.conceptors),
            "active_conceptors": len(active),
            "conceptors": {
                name: {
                    "aperture": config.aperture,
                    "layer": config.layer,
                    "weight": config.weight,
                    "load_metric": conceptor.get_load_metric(),
                    "is_active": any(n == name for n, _, _ in active)
                }
                for name, (conceptor, config) in self.conceptors.items()
            }
        }


class LowRankSubspaceSteering:
    """
    Low-rank subspace steering for efficient concept representation.
    
    Instead of full conceptor matrices, uses low-rank approximations
    for memory efficiency and faster computation.
    """
    
    def __init__(
        self,
        rank: int = 64,
        device: str = "cuda"
    ):
        self.rank = rank
        self.device = device
        self.subspaces: Dict[str, Dict] = {}
    
    def add_subspace(
        self,
        name: str,
        basis_vectors: torch.Tensor,
        layer: int = 0,
        weight: float = 1.0
    ) -> None:
        """
        Add a low-rank subspace.
        
        Args:
            name: Subspace name
            basis_vectors: Orthonormal basis of shape (hidden_dim, rank)
            layer: Target layer
            weight: Projection weight
        """
        # Ensure basis is orthonormal
        Q, R = torch.linalg.qr(basis_vectors.to(self.device))
        
        self.subspaces[name] = {
            "basis": Q[:, :self.rank],
            "layer": layer,
            "weight": weight
        }
        
        logger.info(f"Added subspace '{name}' with rank {self.rank}")
    
    def project_to_subspace(
        self,
        hidden_states: torch.Tensor,
        subspace_name: str
    ) -> torch.Tensor:
        """
        Project hidden states onto a subspace.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
            subspace_name: Name of subspace to project onto
        
        Returns:
            Projected hidden states
        """
        if subspace_name not in self.subspaces:
            raise ValueError(f"Unknown subspace: {subspace_name}")
        
        subspace = self.subspaces[subspace_name]
        basis = subspace["basis"]
        weight = subspace["weight"]
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        h_flat = hidden_states.reshape(-1, hidden_dim)
        
        # Project: h' = B @ B^T @ h
        # where B is the orthonormal basis
        coords = h_flat @ basis  # (batch*seq, rank)
        h_projected = coords @ basis.T  # (batch*seq, hidden_dim)
        
        # Apply weight
        h_projected = h_projected * weight
        
        return h_projected.reshape(batch_size, seq_len, hidden_dim)
    
    def blend_subspaces(
        self,
        hidden_states: torch.Tensor,
        subspace_names: List[str],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Blend multiple subspace projections.
        
        Args:
            hidden_states: Input hidden states
            subspace_names: List of subspace names
            weights: Optional weights for each subspace
        
        Returns:
            Blended projection
        """
        if weights is None:
            weights = [1.0] * len(subspace_names)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight < 1e-8:
            return hidden_states
        
        weights = [w / total_weight for w in weights]
        
        # Project onto each subspace and blend
        result = torch.zeros_like(hidden_states)
        for name, weight in zip(subspace_names, weights):
            projected = self.project_to_subspace(hidden_states, name)
            result += projected * weight
        
        return result
