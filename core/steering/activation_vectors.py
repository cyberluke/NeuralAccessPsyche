"""
Activation Addition vectors for NRAM v5.

This module provides pre-defined activation vectors for common steering tasks.
Vectors are loaded from disk or created on-demand.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional

from core.steering.activation_addition import ActivationVector

logger = logging.getLogger(__name__)


class VectorRegistry:
    """
    Registry for managing activation vectors.
    
    Loads vectors from disk and provides them for steering.
    """
    
    def __init__(self, vectors_dir: Path = Path("data/activation_vectors")):
        self.vectors_dir = vectors_dir
        self.vectors: Dict[str, ActivationVector] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load vector metadata from disk."""
        metadata_path = self.vectors_dir / "vectors_metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Vector metadata not found at {metadata_path}")
            return
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load vectors for each name and layer
        for name, info in metadata.get("vectors", {}).items():
            for layer_info in info["layers"]:
                layer = layer_info["layer"]
                vector_path = Path(layer_info["file"])
                
                if vector_path.exists():
                    vector = torch.load(vector_path, map_location="cpu")
                    key = f"{name}_layer{layer}"
                    self.vectors[key] = ActivationVector(
                        name=name,
                        vector=vector,
                        layer=layer,
                        scale=1.0,
                        description=info.get("description", "")
                    )
        
        logger.info(f"Loaded {len(self.vectors)} vectors from {self.vectors_dir}")
    
    def get_vector(
        self,
        name: str,
        layer: Optional[int] = None,
        device: str = "cpu"
    ) -> Optional[ActivationVector]:
        """
        Get a vector by name and optionally layer.
        
        If layer is None, returns the vector from the layer with highest norm.
        """
        if layer is not None:
            key = f"{name}_layer{layer}"
            if key in self.vectors:
                return self.vectors[key].to(device)
            return None
        
        # Find best layer (highest norm)
        best_vector = None
        best_norm = -1.0
        
        for key, vector in self.vectors.items():
            if vector.name == name:
                norm = vector.vector.norm().item()
                if norm > best_norm:
                    best_norm = norm
                    best_vector = vector
        
        if best_vector:
            return best_vector.to(device)
        return None
    
    def list_vectors(self) -> Dict[str, list]:
        """List all available vectors grouped by name."""
        result = {}
        for key, vector in self.vectors.items():
            if vector.name not in result:
                result[vector.name] = []
            result[vector.name].append({
                "layer": vector.layer,
                "norm": vector.vector.norm().item(),
                "description": vector.description
            })
        return result


def create_novelty_vector(
    registry: VectorRegistry,
    layer: Optional[int] = None,
    device: str = "cpu"
) -> Optional[ActivationVector]:
    """
    Create novelty steering vector.
    
    This vector steers toward novel, original content and away from
    paraphrasing existing ideas.
    """
    return registry.get_vector("novelty_vs_paraphrase", layer, device)


def create_concreteness_vector(
    registry: VectorRegistry,
    layer: Optional[int] = None,
    device: str = "cpu"
) -> Optional[ActivationVector]:
    """
    Create concreteness steering vector.
    
    This vector steers toward concrete, specific content with measurable
    details and away from vague generalities.
    """
    return registry.get_vector("concreteness", layer, device)


def create_human_focus_vector(
    registry: VectorRegistry,
    layer: Optional[int] = None,
    device: str = "cpu"
) -> Optional[ActivationVector]:
    """
    Create human-focus steering vector.
    
    This vector steers toward human-centered content and away from
    purely technical descriptions.
    """
    return registry.get_vector("human_focus", layer, device)


def create_visionary_vector(
    registry: VectorRegistry,
    layer: Optional[int] = None,
    device: str = "cpu"
) -> Optional[ActivationVector]:
    """
    Create visionary steering vector.
    
    This vector steers toward long-term, transformative thinking and
    away from short-term tactical improvements.
    """
    return registry.get_vector("visionary", layer, device)


def create_anti_corporate_vector(
    registry: VectorRegistry,
    layer: Optional[int] = None,
    device: str = "cpu"
) -> Optional[ActivationVector]:
    """
    Create anti-corporate jargon steering vector.
    
    This vector steers toward clear, honest language and away from
    corporate buzzwords and marketing speak.
    """
    return registry.get_vector("anti_corporate", layer, device)


def get_recommended_layers(vector_name: str, registry: VectorRegistry) -> list:
    """
    Get recommended layers for a vector based on norm analysis.
    
    Returns layers sorted by vector norm (strongest first).
    """
    vectors = registry.list_vectors()
    
    if vector_name not in vectors:
        return []
    
    # Sort by norm descending
    sorted_layers = sorted(
        vectors[vector_name],
        key=lambda x: x["norm"],
        reverse=True
    )
    
    return [info["layer"] for info in sorted_layers[:5]]
