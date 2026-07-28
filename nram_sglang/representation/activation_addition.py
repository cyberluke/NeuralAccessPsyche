"""
Activation Addition runtime for NRAM.

Implements: h' = h + alpha * v

Where:
- h is the hidden state at a specific layer
- alpha is the strength (can be positive, negative, or zero)
- v is the activation vector (contrastive direction)

This module loads vector artifacts and applies them during the forward pass
via hooks. It validates artifacts and ensures correct device/dtype.

Critical: The vector must be a real contrastive direction collected from
actual model activations, not a hand-crafted logit bias.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ActivationVectorArtifact:
    """
    Loaded activation vector artifact.
    
    Contains the vector tensor and metadata proving it was collected
    from real model activations with proper contrastive pairs.
    """
    vector_id: str
    vector: torch.Tensor  # Shape: (hidden_dim,)
    layer: int
    model_hash: str
    tokenizer_hash: str
    pooling: str  # "mean", "last", "max"
    sample_count: int
    dataset_hash: Optional[str]
    normalization: str  # "none", "l2", "unit_norm"
    vector_norm: float
    artifact_hash: str
    description: Optional[str] = None
    
    def to_device(self, device: str) -> "ActivationVectorArtifact":
        """Move vector to specified device."""
        return ActivationVectorArtifact(
            vector_id=self.vector_id,
            vector=self.vector.to(device),
            layer=self.layer,
            model_hash=self.model_hash,
            tokenizer_hash=self.tokenizer_hash,
            pooling=self.pooling,
            sample_count=self.sample_count,
            dataset_hash=self.dataset_hash,
            normalization=self.normalization,
            vector_norm=self.vector_norm,
            artifact_hash=self.artifact_hash,
            description=self.description,
        )


class ActivationAdditionRuntime:
    """
    Runtime for activation addition interventions.
    
    Loads vector artifacts and provides them to hooks for application
    during the forward pass. Validates artifacts against model/tokenizer
    hashes to ensure compatibility.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._vectors: Dict[str, ActivationVectorArtifact] = {}
    
    def load_vector_from_safetensors(
        self,
        artifact_path: str,
        expected_model_hash: Optional[str] = None,
        expected_tokenizer_hash: Optional[str] = None,
    ) -> ActivationVectorArtifact:
        """
        Load an activation vector from a safetensors artifact.
        
        The artifact must contain:
        - vector tensor (hidden_dim,)
        - metadata JSON with model_hash, tokenizer_hash, layer, etc.
        
        Args:
            artifact_path: Path to .safetensors file
            expected_model_hash: Expected model hash (for validation)
            expected_tokenizer_hash: Expected tokenizer hash (for validation)
        
        Returns:
            Loaded ActivationVectorArtifact
        
        Raises:
            ValueError: If artifact is invalid or incompatible
        """
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        try:
            from safetensors.torch import load_file
            data = load_file(str(path))
        except ImportError:
            raise ImportError("safetensors is required to load activation vectors")
        
        # Extract vector
        if "vector" not in data:
            raise ValueError(f"Artifact missing 'vector' key: {artifact_path}")
        
        vector = data["vector"]
        if not isinstance(vector, torch.Tensor):
            raise ValueError(f"Vector is not a tensor: {type(vector)}")
        
        if vector.dim() != 1:
            raise ValueError(f"Vector must be 1D, got shape {vector.shape}")
        
        # Extract metadata
        metadata_str = data.get("metadata", "{}")
        if isinstance(metadata_str, torch.Tensor):
            metadata_str = metadata_str.item()
        
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else {}
        except json.JSONDecodeError:
            metadata = {}
        
        # Validate required metadata
        required_fields = ["model_hash", "tokenizer_hash", "layer", "pooling"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Artifact metadata missing '{field}': {artifact_path}")
        
        # Validate hashes if expected
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
        
        # Compute artifact hash
        artifact_hash = self._compute_artifact_hash(vector, metadata)
        
        # Compute vector norm
        vector_norm = float(vector.norm().item())
        
        # Create artifact
        artifact = ActivationVectorArtifact(
            vector_id=path.stem,
            vector=vector.to(self.device),
            layer=int(metadata["layer"]),
            model_hash=metadata["model_hash"],
            tokenizer_hash=metadata["tokenizer_hash"],
            pooling=metadata["pooling"],
            sample_count=int(metadata.get("sample_count", 0)),
            dataset_hash=metadata.get("dataset_hash"),
            normalization=metadata.get("normalization", "none"),
            vector_norm=vector_norm,
            artifact_hash=artifact_hash,
            description=metadata.get("description"),
        )
        
        # Store
        self._vectors[artifact.vector_id] = artifact
        
        logger.info(
            f"Loaded activation vector '{artifact.vector_id}' at layer {artifact.layer}, "
            f"norm={vector_norm:.4f}, device={self.device}"
        )
        
        return artifact
    
    def get_vector(self, vector_id: str) -> Optional[ActivationVectorArtifact]:
        """Get a loaded vector by ID."""
        return self._vectors.get(vector_id)
    
    def load_from_tensors(
        self,
        vector_id: str,
        vector: torch.Tensor,
        layer: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActivationVectorArtifact:
        """
        Load an activation vector directly from tensors (for testing or programmatic use).
        
        Args:
            vector_id: Unique identifier for the vector
            vector: 1D tensor of shape (hidden_dim,)
            layer: Target layer index
            metadata: Optional metadata dict with model_hash, tokenizer_hash, pooling, etc.
        
        Returns:
            ActivationVectorArtifact
        """
        metadata = metadata or {}
        if vector.dim() != 1:
            raise ValueError(f"Vector must be 1D (hidden_dim,), got shape {vector.shape}")
        
        vector_norm = float(vector.norm().item())
        artifact_hash = self._compute_artifact_hash(vector, metadata)
        
        artifact = ActivationVectorArtifact(
            vector_id=vector_id,
            vector=vector.to(self.device),
            layer=layer,
            model_hash=metadata.get("model_hash", "unknown"),
            tokenizer_hash=metadata.get("tokenizer_hash", "unknown"),
            pooling=metadata.get("pooling", "mean"),
            sample_count=int(metadata.get("sample_count", 0)),
            dataset_hash=metadata.get("dataset_hash"),
            normalization=metadata.get("normalization", "none"),
            vector_norm=vector_norm,
            artifact_hash=artifact_hash,
            description=metadata.get("description"),
        )
        
        self._vectors[vector_id] = artifact
        logger.info(
            f"Loaded activation vector '{vector_id}' at layer {layer}, "
            f"norm={vector_norm:.4f}, device={self.device}"
        )
        return artifact
    
    def apply_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        vector_id: str,
        strength: float,
    ) -> torch.Tensor:
        """
        Apply activation addition to hidden states.
        
        Computes: h' = h + alpha * v
        
        Supports both 2D (num_tokens, hidden_dim) from SGLang and
        3D (batch, seq_len, hidden_dim) from other runners.
        
        Args:
            hidden_states: Current hidden states
            vector_id: ID of the vector to apply
            strength: Alpha (scaling factor)
        
        Returns:
            Modified hidden states
        """
        artifact = self._vectors.get(vector_id)
        if artifact is None:
            logger.warning(f"Vector not found: {vector_id}")
            return hidden_states
        
        # Ensure vector is on same device/dtype
        vector = artifact.vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Broadcast: (hidden_dim,) -> match hidden states rank
        if hidden_states.dim() == 2:
            vector_expanded = vector.unsqueeze(0)  # (1, hidden_dim)
        elif hidden_states.dim() == 3:
            vector_expanded = vector.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        else:
            logger.error(f"Unexpected hidden_states rank {hidden_states.dim()}")
            return hidden_states
        
        # Apply: h' = h + alpha * v
        delta = strength * vector_expanded
        modified = hidden_states + delta
        
        return modified
    
    def _compute_artifact_hash(self, vector: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Compute a hash of the artifact for integrity verification."""
        hasher = hashlib.sha256()
        
        # Hash vector data
        vector_bytes = vector.cpu().numpy().tobytes()
        hasher.update(vector_bytes)
        
        # Hash metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        hasher.update(metadata_str.encode("utf-8"))
        
        return hasher.hexdigest()[:16]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of loaded vectors."""
        return {
            "loaded_vectors": len(self._vectors),
            "vectors": {
                vid: {
                    "layer": vec.layer,
                    "norm": vec.vector_norm,
                    "model_hash": vec.model_hash,
                    "artifact_hash": vec.artifact_hash,
                }
                for vid, vec in self._vectors.items()
            },
        }


# Global runtime instance
activation_addition_runtime = ActivationAdditionRuntime()
