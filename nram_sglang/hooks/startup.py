"""
Startup script for NRAM v5 hooks.

This module is called by SGLang after the model is loaded to register
forward hooks on all decoder layers for hidden-state interventions.

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


def main():
    """Register hooks on a model loaded from disk."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Register NRAM hooks on a model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
