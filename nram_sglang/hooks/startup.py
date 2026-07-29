"""
Startup script for NRAM v5 hooks.

This module is called by SGLang after the model is loaded to register
forward hooks on all decoder layers for hidden-state interventions.

It also loads activation vector and conceptor artifacts from
artifacts/nram_vectors/ so they are available during inference.

Usage:
    python -m nram_sglang.hooks.startup --model-path /path/to/model
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from nram_sglang.hooks.factory import register_nram_hooks_on_model
from nram_sglang.representation.activation_addition import activation_addition_runtime
from nram_sglang.representation.conceptor import conceptor_runtime

logger = logging.getLogger(__name__)


def register_hooks_on_loaded_model(model: torch.nn.Module) -> int:
    """
    Register NRAM hooks on an already-loaded model.
    
    Args:
        model: The loaded transformer model
    
    Returns:
        Number of hooks registered
    """
    count = register_nram_hooks_on_model(model)
    logger.info(f"NRAM v5: Registered {count} forward hooks on model")
    return count


def _detect_device() -> str:
    """Detect the best available device for tensor operations."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_nram_artifacts(artifact_dir: str = "artifacts/nram_vectors") -> dict:
    """
    Load activation vector and conceptor artifacts from disk.
    
    Scans the artifact directory for .safetensors files and loads them
    into the global runtime instances. This must be called at startup
    before any requests are processed.
    
    Args:
        artifact_dir: Path to directory containing .safetensors artifacts
    
    Returns:
        Dict with counts of loaded vectors and conceptors
    """
    artifact_path = Path(artifact_dir)
    if not artifact_path.exists():
        logger.warning(f"Artifact directory not found: {artifact_dir}")
        return {"vectors_loaded": 0, "conceptors_loaded": 0}
    
    # Reconfigure runtimes to use the detected device
    device = _detect_device()
    activation_addition_runtime.device = device
    conceptor_runtime.device = device
    
    vectors_loaded = 0
    conceptors_loaded = 0
    
    # Load activation vectors
    for vector_file in sorted(artifact_path.glob("*.safetensors")):
        try:
            vector_file_str = str(vector_file)
            # Try loading as activation vector first
            try:
                activation_addition_runtime.load_vector_from_safetensors(vector_file_str)
                vectors_loaded += 1
                logger.info(f"Loaded activation vector: {vector_file.name}")
                continue
            except (ValueError, KeyError):
                pass
            
            # Try loading as conceptor
            try:
                conceptor_runtime.load_from_safetensors(vector_file_str)
                conceptors_loaded += 1
                logger.info(f"Loaded conceptor: {vector_file.name}")
                continue
            except (ValueError, KeyError):
                pass
            
            logger.warning(f"Could not load artifact: {vector_file.name}")
        except Exception as e:
            logger.error(f"Error loading artifact {vector_file.name}: {e}")
    
    logger.info(f"NRAM v5: Loaded {vectors_loaded} vectors, {conceptors_loaded} conceptors on device={device}")
    return {"vectors_loaded": vectors_loaded, "conceptors_loaded": conceptors_loaded}


def main():
    """Register hooks on a model loaded from disk and load artifacts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Register NRAM hooks on a model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/nram_vectors",
                       help="Path to NRAM artifact directory")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    count = register_hooks_on_loaded_model(model)
    logger.info(f"Successfully registered {count} hooks")
    
    # Load NRAM artifacts
    artifact_stats = load_nram_artifacts(args.artifact_dir)
    logger.info(f"Loaded artifacts: {artifact_stats}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
