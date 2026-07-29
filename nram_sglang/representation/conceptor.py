"""
Conceptor steering runtime for NRAM.

Implements the conceptor operator:
    C = U diag(s² / (s² + aperture⁻²)) Uᵀ

Where:
- U, s come from SVD of activation patterns (low-rank representation)
- aperture controls how tightly the conceptor admits patterns
- Application is memory-efficient: no dense hidden_size × hidden_size matrix

The conceptor projects hidden states onto a learned subspace, enabling
multi-dimensional concept steering (e.g., novelty + coherence + concreteness
as a single subspace rather than a single vector).

Theory:
- aperture → 0: C → I (identity, passes everything)
- aperture → ∞: C → 0 (zero, blocks everything)
- Finite aperture: smooth interpolation based on singular values

Artifact format:
- basis_vectors: (rank, hidden_dim) - left singular vectors U
- singular_values: (rank,) - singular values s
- aperture: float - conceptor aperture
- metadata: model_hash, tokenizer_hash, layer, dataset_hash, rank, hidden_size, dtype, creation_commit
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class ConceptorArtifact:
    """
    Loaded conceptor artifact.
    
    Contains low-rank basis vectors and singular values for efficient
    conceptor projection, plus metadata proving provenance.
    """
    conceptor_id: str
    basis_vectors: torch.Tensor  # (rank, hidden_dim)
    singular_values: torch.Tensor  # (rank,)
    aperture: float
    layer: int
    model_hash: str
    tokenizer_hash: str
    dataset_hash: Optional[str]
    rank: int
    hidden_size: int
    dtype: str
    creation_commit: Optional[str]
    artifact_hash: str
    description: Optional[str] = None
    
    def to_device(self, device: str) -> "ConceptorArtifact":
        """Move tensors to specified device."""
        return ConceptorArtifact(
            conceptor_id=self.conceptor_id,
            basis_vectors=self.basis_vectors.to(device),
            singular_values=self.singular_values.to(device),
            aperture=self.aperture,
            layer=self.layer,
            model_hash=self.model_hash,
            tokenizer_hash=self.tokenizer_hash,
            dataset_hash=self.dataset_hash,
            rank=self.rank,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            creation_commit=self.creation_commit,
            artifact_hash=self.artifact_hash,
            description=self.description,
        )
    
    def compute_weights(self, aperture_override: Optional[float] = None) -> torch.Tensor:
        """
        Compute conceptor weights: s² / (s² + aperture⁻²)
        
        Args:
            aperture_override: Override aperture (for testing sensitivity)
        
        Returns:
            Weights tensor of shape (rank,)
        """
        aperture = aperture_override if aperture_override is not None else self.aperture
        sv_squared = self.singular_values ** 2
        
        if aperture <= 0:
            # aperture → 0: identity (all weights = 1)
            return torch.ones_like(self.singular_values)
        
        if aperture == float('inf'):
            # aperture → ∞: zero (all weights = 0)
            return torch.zeros_like(self.singular_values)
        
        aperture_sq_inv = 1.0 / (aperture ** 2)
        weights = sv_squared / (sv_squared + aperture_sq_inv)
        return weights


class ConceptorRuntime:
    """
    Runtime for conceptor interventions.
    
    Loads conceptor artifacts and provides them to hooks for application
    during the forward pass. Validates artifacts against model/tokenizer
    hashes to ensure compatibility.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._conceptors: Dict[str, ConceptorArtifact] = {}
    
    def load_from_safetensors(
        self,
        artifact_path: str,
        expected_model_hash: Optional[str] = None,
        expected_tokenizer_hash: Optional[str] = None,
    ) -> ConceptorArtifact:
        """
        Load a conceptor from a safetensors artifact.
        
        The artifact must contain:
        - basis_vectors: (rank, hidden_dim) tensor
        - singular_values: (rank,) tensor
        - metadata JSON with aperture, layer, model_hash, etc.
        
        Args:
            artifact_path: Path to .safetensors file
            expected_model_hash: Expected model hash (for validation)
            expected_tokenizer_hash: Expected tokenizer hash (for validation)
        
        Returns:
            Loaded ConceptorArtifact
        
        Raises:
            ValueError: If artifact is invalid or incompatible
        """
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors is required to load conceptor artifacts")
        
        # Extract tensors and metadata using safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            # Check for required tensors
            keys = f.keys()
            if "basis_vectors" not in keys:
                raise ValueError(f"Artifact missing 'basis_vectors' key: {artifact_path}")
            if "singular_values" not in keys:
                raise ValueError(f"Artifact missing 'singular_values' key: {artifact_path}")
            
            basis = f.get_tensor("basis_vectors")
            svs = f.get_tensor("singular_values")
            
            if not isinstance(basis, torch.Tensor) or not isinstance(svs, torch.Tensor):
                raise ValueError(f"basis_vectors and singular_values must be tensors")
            
            if basis.dim() != 2:
                raise ValueError(f"basis_vectors must be 2D (rank, hidden_dim), got {basis.shape}")
            if svs.dim() != 1:
                raise ValueError(f"singular_values must be 1D (rank,), got {svs.shape}")
            if basis.shape[0] != svs.shape[0]:
                raise ValueError(
                    f"Rank mismatch: basis_vectors has {basis.shape[0]} rows, "
                    f"singular_values has {svs.shape[0]} elements"
                )
            
            # Extract metadata from file-level metadata
            metadata_dict = f.metadata() or {}
            metadata_str = metadata_dict.get("metadata", "{}")
            
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else {}
            except json.JSONDecodeError:
                metadata = {}
        
        # Validate required metadata
        required_fields = ["model_hash", "tokenizer_hash", "layer", "aperture"]
        for f in required_fields:
            if f not in metadata:
                raise ValueError(f"Artifact metadata missing '{f}': {artifact_path}")
        
        # Validate hashes
        if expected_model_hash and metadata["model_hash"] != expected_model_hash:
            raise ValueError(
                f"Model hash mismatch: artifact={metadata['model_hash']}, "
                f"expected={expected_model_hash}"
            )
        if expected_tokenizer_hash and metadata["tokenizer_hash"] != expected_tokenizer_hash:
            raise ValueError(
                f"Tokenizer hash mismatch: artifact={metadata['tokenizer_hash']}, "
                f"expected={expected_tokenizer_hash}"
            )
        
        rank = int(basis.shape[0])
        hidden_size = int(basis.shape[1])
        
        # Compute artifact hash
        artifact_hash = self._compute_artifact_hash(basis, svs, metadata)
        
        artifact = ConceptorArtifact(
            conceptor_id=path.stem,
            basis_vectors=basis.to(self.device),
            singular_values=svs.to(self.device),
            aperture=float(metadata["aperture"]),
            layer=int(metadata["layer"]),
            model_hash=metadata["model_hash"],
            tokenizer_hash=metadata["tokenizer_hash"],
            dataset_hash=metadata.get("dataset_hash"),
            rank=rank,
            hidden_size=hidden_size,
            dtype=str(basis.dtype),
            creation_commit=metadata.get("creation_commit"),
            artifact_hash=artifact_hash,
            description=metadata.get("description"),
        )
        
        self._conceptors[artifact.conceptor_id] = artifact
        
        logger.info(
            f"Loaded conceptor '{artifact.conceptor_id}' at layer {artifact.layer}, "
            f"rank={rank}, aperture={artifact.aperture}, device={self.device}"
        )
        
        return artifact
    
    def load_from_tensors(
        self,
        conceptor_id: str,
        basis_vectors: torch.Tensor,
        singular_values: torch.Tensor,
        aperture: float,
        layer: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConceptorArtifact:
        """
        Load a conceptor directly from tensors (for testing or programmatic creation).
        
        Args:
            conceptor_id: Unique identifier
            basis_vectors: (rank, hidden_dim) tensor
            singular_values: (rank,) tensor
            aperture: Conceptor aperture
            layer: Target layer index
            metadata: Optional metadata dict
        
        Returns:
            ConceptorArtifact
        """
        metadata = metadata or {}
        rank = int(basis_vectors.shape[0])
        hidden_size = int(basis_vectors.shape[1])
        
        artifact_hash = self._compute_artifact_hash(basis_vectors, singular_values, metadata)
        
        artifact = ConceptorArtifact(
            conceptor_id=conceptor_id,
            basis_vectors=basis_vectors.to(self.device),
            singular_values=singular_values.to(self.device),
            aperture=aperture,
            layer=layer,
            model_hash=metadata.get("model_hash", "unknown"),
            tokenizer_hash=metadata.get("tokenizer_hash", "unknown"),
            dataset_hash=metadata.get("dataset_hash"),
            rank=rank,
            hidden_size=hidden_size,
            dtype=str(basis_vectors.dtype),
            creation_commit=metadata.get("creation_commit"),
            artifact_hash=artifact_hash,
            description=metadata.get("description"),
        )
        
        self._conceptors[conceptor_id] = artifact
        return artifact
    
    def get_conceptor(self, conceptor_id: str) -> Optional[ConceptorArtifact]:
        """Get a loaded conceptor by ID."""
        return self._conceptors.get(conceptor_id)
    
    def apply_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        conceptor_id: str,
        strength: float,
        aperture_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply conceptor projection to hidden states.
        
        Computes: h' = h + strength * (C(h) - h)
        Where C(h) = U diag(w) Uᵀ h and w = s² / (s² + aperture⁻²)
        
        Args:
            hidden_states: Current hidden states (num_tokens, hidden_dim) or
                          (batch, seq_len, hidden_dim)
            conceptor_id: ID of the conceptor to apply
            strength: Scaling factor for the projection delta
            aperture_override: Override aperture (for sensitivity testing)
        
        Returns:
            Tuple of (modified_hidden_states, delta_norm)
        """
        artifact = self._conceptors.get(conceptor_id)
        if artifact is None:
            logger.warning(f"Conceptor not found: {conceptor_id}")
            return hidden_states, 0.0
        
        # Ensure on correct device/dtype
        basis = artifact.basis_vectors.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        svs = artifact.singular_values.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        # Compute conceptor weights
        sv_squared = svs ** 2
        aperture = aperture_override if aperture_override is not None else artifact.aperture
        
        if aperture <= 0:
            # Identity: no change
            return hidden_states, 0.0
        
        if aperture == float('inf'):
            # Zero projection: delta = -hidden_states
            delta = strength * (torch.zeros_like(hidden_states) - hidden_states)
            delta_norm = float(delta.norm().item())
            return hidden_states + delta, delta_norm
        
        aperture_sq_inv = 1.0 / (aperture ** 2)
        weights = sv_squared / (sv_squared + aperture_sq_inv)
        
        # Project hidden states onto basis
        # hidden_states: (num_tokens, hidden_dim) or (batch, seq_len, hidden_dim)
        # basis: (rank, hidden_dim)
        # projection: (num_tokens, rank) or (batch, seq_len, rank)
        projection = torch.matmul(hidden_states, basis.T)
        
        # Apply conceptor weights
        if projection.dim() == 2:
            weighted_projection = projection * weights.unsqueeze(0)
        else:
            weighted_projection = projection * weights.unsqueeze(0).unsqueeze(0)
        
        # Project back
        reconcepted = torch.matmul(weighted_projection, basis)
        
        # Delta is the difference
        delta = strength * (reconcepted - hidden_states)
        delta_norm = float(delta.norm().item())
        
        return hidden_states + delta, delta_norm
    
    def fit_from_activations(
        self,
        conceptor_id: str,
        activations: torch.Tensor,
        aperture: float,
        layer: int,
        rank: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConceptorArtifact:
        """
        Fit a conceptor from activation data using SVD.
        
        Args:
            conceptor_id: Unique identifier
            activations: (num_samples, hidden_dim) activation matrix
            aperture: Conceptor aperture
            layer: Target layer index
            rank: Maximum rank (None = full rank)
            metadata: Optional metadata
        
        Returns:
            Fitted ConceptorArtifact
        """
        # Center activations
        mean = activations.mean(dim=0, keepdim=True)
        centered = activations - mean
        
        # Compute SVD
        if rank is None or rank >= min(centered.shape):
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        else:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
        
        # Basis vectors are rows of Vh (principal components)
        basis_vectors = Vh
        singular_values = S
        
        return self.load_from_tensors(
            conceptor_id=conceptor_id,
            basis_vectors=basis_vectors,
            singular_values=singular_values,
            aperture=aperture,
            layer=layer,
            metadata=metadata,
        )
    
    def _compute_artifact_hash(
        self,
        basis: torch.Tensor,
        svs: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> str:
        """Compute hash of artifact for integrity verification."""
        hasher = hashlib.sha256()
        hasher.update(basis.cpu().numpy().tobytes())
        hasher.update(svs.cpu().numpy().tobytes())
        hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()[:16]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of loaded conceptors."""
        return {
            "loaded_conceptors": len(self._conceptors),
            "conceptors": {
                cid: {
                    "layer": c.layer,
                    "rank": c.rank,
                    "hidden_size": c.hidden_size,
                    "aperture": c.aperture,
                    "artifact_hash": c.artifact_hash,
                }
                for cid, c in self._conceptors.items()
            },
        }


# Global runtime instance
conceptor_runtime = ConceptorRuntime()
