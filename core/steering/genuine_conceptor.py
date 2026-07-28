"""
Genuine conceptor steering for NRAM v5.

Implements real low-rank conceptor operators from collected activations.
NOT renamed vector addition.

Formulation:
  R = E[h h^T]
  C = R (R + aperture^{-2} I)^{-1}

For memory efficiency uses low-rank SVD/eigendecomposition.

Supports:
  - Positive conceptor
  - Negative conceptor
  - C AND B
  - C OR B
  - NOT C

Runtime forms:
  h' = h + alpha * C(h)
  h' = h - beta * C_negative(h)
  h' = h + alpha * C_positive(h) - beta * C_negative(h)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class ConceptorArtifact:
    """A trained conceptor with its metadata."""
    conceptor_id: str
    aperture: float
    rank: int
    singular_values: torch.Tensor  # Top-k singular values
    U: torch.Tensor  # Left singular vectors (hidden_dim, rank)
    eigenvalues: torch.Tensor  # eigenvalues of R in the reduced basis
    explained_energy: float
    layer_name: str
    sample_count: int
    model_hash: str
    dataset_hash: str


class GenuineConceptor:
    """
    Genuine conceptor operator.
    
    Uses low-rank SVD for memory efficiency:
      1. Collect activation patterns H (hidden_dim, num_samples)
      2. Compute R = H H^T / n (covariance)
      3. SVD: H = U S V^T
      4. R = U S^2 V^T V U^T / n = U diag(s^2/n) U^T
      5. C = U diag(s^2/n / (s^2/n + alpha^{-2})) U^T
    
    Apply: h' = C h = U diag(...) U^T h
    """
    
    def __init__(self, artifact: ConceptorArtifact, device: str = "cuda"):
        self.artifact = artifact
        self.device = device
        # Pre-compute diagonal for fast application
        s2 = artifact.singular_values ** 2
        inv_ap2 = artifact.aperture ** -2
        self._diag = s2 / (s2 + inv_ap2)
    
    @classmethod
    def from_activations(
        cls,
        activations: torch.Tensor,  # (num_samples, hidden_dim)
        aperture: float,
        conceptor_id: str,
        layer_name: str,
        max_rank: Optional[int] = None,
        energy_threshold: float = 0.99,
        model_hash: str = "",
        dataset_hash: str = "",
    ) -> "GenuineConceptor":
        """
        Train a conceptor from collected activations.
        
        Uses SVD for numerical stability and memory efficiency.
        """
        activations = activations.float()
        num_samples, hidden_dim = activations.shape
        
        # Center activations
        mean = activations.mean(dim=0, keepdim=True)
        centered = activations - mean
        
        # SVD of centered activations
        # For efficiency, compute SVD of the smaller matrix
        if num_samples < hidden_dim:
            # Compute H^T H first (num_samples, num_samples)
            HtH = centered @ centered.T / num_samples
            eigenvalues, V = torch.linalg.eigh(HtH)
            # Sort descending
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            V = V[:, idx]
            # U = H V S^{-1}
            S = torch.sqrt(torch.clamp(eigenvalues, min=0))
            U = centered.T @ V / (S.unsqueeze(0) + 1e-10)
        else:
            # Compute H H^T first (hidden_dim, hidden_dim)
            HHt = centered.T @ centered / num_samples
            eigenvalues, U = torch.linalg.eigh(HHt)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            U = U[:, idx]
            S = torch.sqrt(torch.clamp(eigenvalues, min=0))
        
        # Determine rank from energy threshold
        total_energy = eigenvalues.sum()
        cumulative_energy = torch.cumsum(eigenvalues, dim=0) / total_energy
        if max_rank is None:
            max_rank = len(eigenvalues)
        rank = min(
            max_rank,
            int((cumulative_energy >= energy_threshold).float().argmax().item()) + 1
        )
        rank = max(rank, 1)
        
        # Truncate to rank
        U_k = U[:, :rank].contiguous()
        S_k = S[:rank].contiguous()
        eigenvalues_k = eigenvalues[:rank].contiguous()
        explained_energy = cumulative_energy[rank - 1].item()
        
        artifact = ConceptorArtifact(
            conceptor_id=conceptor_id,
            aperture=aperture,
            rank=rank,
            singular_values=S_k,
            U=U_k,
            eigenvalues=eigenvalues_k,
            explained_energy=explained_energy,
            layer_name=layer_name,
            sample_count=num_samples,
            model_hash=model_hash,
            dataset_hash=dataset_hash,
        )
        
        return cls(artifact, device=str(activations.device))
    
    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply conceptor to hidden states.
        
        h' = C h = U diag(s^2/(s^2 + alpha^{-2})) U^T h
        
        Args:
            hidden_states: (batch, seq, hidden_dim) or (batch, hidden_dim)
        
        Returns:
            Projected hidden states (same shape)
        """
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch, seq, dim = hidden_states.shape
            h = hidden_states.reshape(-1, dim)
        else:
            h = hidden_states
        
        # Project to low-rank space: U^T h -> (rank, batch*seq)
        projected = self.artifact.U.T @ h.T  # (rank, batch*seq)
        
        # Apply diagonal: diag * projected
        scaled = self._diag.unsqueeze(1) * projected  # (rank, batch*seq)
        
        # Project back: U scaled -> (hidden_dim, batch*seq)
        result = self.artifact.U @ scaled  # (hidden_dim, batch*seq)
        
        return result.T.reshape(original_shape)
    
    def apply_with_strength(
        self, hidden_states: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        Apply conceptor with strength: h' = h + alpha * C(h)
        
        Zero alpha is exact no-op.
        """
        if abs(alpha) < 1e-10:
            return hidden_states
        Ch = self.apply(hidden_states)
        return hidden_states + alpha * Ch
    
    def get_load_metric(self) -> float:
        """Load metric: trace(C) / hidden_dim."""
        return self._diag.sum().item() / self.artifact.U.shape[0]


class ConceptorComposition:
    """
    Boolean composition of conceptors.
    
    Supports: C AND B, C OR B, NOT C
    """
    
    @staticmethod
    def and_op(C1: GenuineConceptor, C2: GenuineConceptor) -> torch.Tensor:
        """C AND B = C1 @ C2 (approximate for low-rank)."""
        # Full matrix product in low-rank basis
        # C1 = U1 D1 U1^T, C2 = U2 D2 U2^T
        # C1 C2 = U1 D1 U1^T U2 D2 U2^T
        U1, D1 = C1.artifact.U, C1._diag
        U2, D2 = C2.artifact.U, C2._diag
        # Compute in the joint basis
        M = U1.T @ U2  # (r1, r2)
        result_diag = D1.unsqueeze(1) * M * D2.unsqueeze(0)
        return U1 @ result_diag @ U2.T
    
    @staticmethod
    def or_op(C1: GenuineConceptor, C2: GenuineConceptor) -> torch.Tensor:
        """C OR B = C1 + C2 - C1 @ C2."""
        U1, D1 = C1.artifact.U, C1._diag
        U2, D2 = C2.artifact.U, C2._diag
        # C1 + C2
        sum_matrix = U1 @ torch.diag(D1) @ U1.T + U2 @ torch.diag(D2) @ U2.T
        # C1 @ C2
        M = U1.T @ U2
        prod_diag = D1.unsqueeze(1) * M * D2.unsqueeze(0)
        prod_matrix = U1 @ prod_diag @ U2.T
        return sum_matrix - prod_matrix
    
    @staticmethod
    def not_op(C: GenuineConceptor) -> torch.Tensor:
        """NOT C = I - C."""
        U, D = C.artifact.U, C._diag
        # I - U diag(D) U^T
        dim = U.shape[0]
        identity = torch.eye(dim, device=U.device, dtype=U.dtype)
        C_matrix = U @ torch.diag(D) @ U.T
        return identity - C_matrix


class ConceptorSteeringController:
    """
    Controller for conceptor-based steering during generation.
    
    Supports:
    - Positive conceptor: h' = h + alpha * C_pos(h)
    - Negative conceptor: h' = h - beta * C_neg(h)
    - Combined: h' = h + alpha * C_pos(h) - beta * C_neg(h)
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.conceptors: Dict[str, GenuineConceptor] = {}
        self.enabled = True
        self._invocation_count = 0
    
    def register_conceptor(self, conceptor: GenuineConceptor) -> None:
        """Register a trained conceptor."""
        self.conceptors[conceptor.artifact.conceptor_id] = conceptor
        logger.info(
            f"Registered conceptor: {conceptor.artifact.conceptor_id} "
            f"(rank={conceptor.artifact.rank}, aperture={conceptor.artifact.aperture})"
        )
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        positive_id: Optional[str] = None,
        negative_id: Optional[str] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply conceptor steering.
        
        h' = h + alpha * C_pos(h) - beta * C_neg(h)
        """
        if not self.enabled:
            return hidden_states
        
        self._invocation_count += 1
        result = hidden_states
        
        if positive_id and positive_id in self.conceptors:
            C_pos = self.conceptors[positive_id]
            if layer_name is None or C_pos.artifact.layer_name == layer_name:
                result = C_pos.apply_with_strength(result, alpha)
        
        if negative_id and negative_id in self.conceptors:
            C_neg = self.conceptors[negative_id]
            if layer_name is None or C_neg.artifact.layer_name == layer_name:
                result = result - beta * C_neg.apply(result)
        
        return result
    
    def reset(self) -> None:
        """Reset invocation count."""
        self._invocation_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.enabled,
            "num_conceptors": len(self.conceptors),
            "conceptors": {
                k: {
                    "rank": v.artifact.rank,
                    "aperture": v.artifact.aperture,
                    "layer": v.artifact.layer_name,
                    "explained_energy": v.artifact.explained_energy,
                    "load_metric": v.get_load_metric(),
                }
                for k, v in self.conceptors.items()
            },
            "invocation_count": self._invocation_count,
        }
