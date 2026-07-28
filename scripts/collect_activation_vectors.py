#!/usr/bin/env python3
"""
Collect activation vectors from contrastive prompt pairs.

This script extracts hidden states from a model using contrastive prompts
and saves the resulting activation vectors for use in steering.

Usage:
    python scripts/collect_activation_vectors.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.steering.activation_addition import VectorCollector, ActivationVector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Contrastive prompt pairs for different steering dimensions
CONTRASTIVE_PAIRS = {
    "novelty_vs_paraphrase": {
        "positive": "Innovative breakthrough concept that changes everything. Original idea that nobody has thought of before. Revolutionary approach to solving this problem.",
        "negative": "This is similar to existing solutions. Like other approaches, this builds on previous work. A variation of the standard method.",
        "description": "Novelty vs paraphrasing existing ideas"
    },
    "concreteness": {
        "positive": "Specific example with concrete details: 42% improvement, 3.5x faster, $1.2M savings. Measurable outcomes with exact metrics.",
        "negative": "Generally speaking, things could be better. Various improvements are possible. Some benefits might be expected.",
        "description": "Concrete specifics vs vague generalities"
    },
    "human_focus": {
        "positive": "How this helps people in their daily lives. Real problems that users face. Human-centered design that improves quality of life.",
        "negative": "Technical specifications and system architecture. Implementation details and algorithmic complexity. Low-level optimization strategies.",
        "description": "Human-centered vs technology-centered"
    },
    "visionary": {
        "positive": "Imagine a future where this transforms society. Long-term vision that reshapes how we think about this domain. Paradigm shift in approach.",
        "negative": "Short-term tactical improvements. Incremental changes to existing systems. Practical near-term optimizations.",
        "description": "Visionary long-term thinking vs short-term tactics"
    },
    "anti_corporate": {
        "positive": "Let's be honest about what this actually does. No buzzwords, just clear explanation. Straightforward description without marketing speak.",
        "negative": "Leveraging synergies to drive value creation. Strategic alignment of core competencies. Holistic approach to stakeholder engagement.",
        "description": "Clear honest language vs corporate jargon"
    }
}


def collect_vectors_for_layer(
    collector: VectorCollector,
    layer: int,
    pairs: Dict[str, Dict[str, str]]
) -> Dict[str, ActivationVector]:
    """Collect activation vectors for a specific layer."""
    vectors = {}
    
    for name, pair in pairs.items():
        logger.info(f"  Collecting vector: {name}")
        
        vector = collector.collect(
            positive_prompt=pair["positive"],
            negative_prompt=pair["negative"],
            layer=layer,
            aggregation="mean"
        )
        
        vectors[name] = ActivationVector(
            name=name,
            vector=vector,
            layer=layer,
            scale=1.0,
            description=pair["description"]
        )
    
    return vectors


def scan_layers(
    collector: VectorCollector,
    num_layers: int,
    pairs: Dict[str, Dict[str, str]],
    step: int = 4
) -> Dict[int, Dict[str, ActivationVector]]:
    """Scan multiple layers to find optimal steering layers."""
    all_vectors = {}
    
    # Scan every Nth layer
    layers_to_scan = list(range(0, num_layers, step))
    if (num_layers - 1) not in layers_to_scan:
        layers_to_scan.append(num_layers - 1)
    
    logger.info(f"Scanning layers: {layers_to_scan}")
    
    for layer in layers_to_scan:
        logger.info(f"Scanning layer {layer}/{num_layers-1}")
        vectors = collect_vectors_for_layer(collector, layer, pairs)
        all_vectors[layer] = vectors
    
    return all_vectors


def save_vectors(
    vectors: Dict[int, Dict[str, ActivationVector]],
    output_dir: Path,
    model_name: str
) -> None:
    """Save collected vectors to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "vectors": {}
    }
    
    for layer, layer_vectors in vectors.items():
        for name, vector in layer_vectors.items():
            # Save vector tensor
            vector_path = output_dir / f"{name}_layer{layer}.pt"
            torch.save(vector.vector, vector_path)
            
            # Add to metadata
            if name not in metadata["vectors"]:
                metadata["vectors"][name] = {
                    "description": vector.description,
                    "layers": []
                }
            
            metadata["vectors"][name]["layers"].append({
                "layer": layer,
                "file": str(vector_path),
                "norm": vector.vector.norm().item()
            })
    
    # Save metadata
    metadata_path = output_dir / "vectors_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(metadata['vectors'])} vectors to {output_dir}")


def analyze_vector_strength(
    vectors: Dict[int, Dict[str, ActivationVector]]
) -> Dict[str, List[Tuple[int, float]]]:
    """Analyze vector norms across layers to find optimal steering layers."""
    analysis = {}
    
    for name in next(iter(vectors.values())).keys():
        layer_strengths = []
        
        for layer, layer_vectors in vectors.items():
            if name in layer_vectors:
                norm = layer_vectors[name].vector.norm().item()
                layer_strengths.append((layer, norm))
        
        # Sort by strength
        layer_strengths.sort(key=lambda x: x[1], reverse=True)
        analysis[name] = layer_strengths
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Collect activation vectors")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/activation_vectors"),
        help="Output directory for vectors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--scan-all-layers",
        action="store_true",
        help="Scan all layers (slow) vs every 4th layer (fast)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else None
    )
    
    if args.device == "cpu":
        model = model.to(args.device)
    
    model.eval()
    
    # Get number of layers
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        logger.error("Could not determine number of layers")
        return
    
    logger.info(f"Model has {num_layers} layers")
    
    # Create collector
    collector = VectorCollector(model, tokenizer, args.device)
    
    # Scan layers
    step = 1 if args.scan_all_layers else 4
    logger.info(f"Scanning every {step} layer(s)")
    
    all_vectors = scan_layers(
        collector,
        num_layers,
        CONTRASTIVE_PAIRS,
        step=step
    )
    
    # Analyze results
    logger.info("\n=== Vector Strength Analysis ===")
    analysis = analyze_vector_strength(all_vectors)
    
    for name, strengths in analysis.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Description: {CONTRASTIVE_PAIRS[name]['description']}")
        logger.info(f"  Top 3 layers:")
        for layer, norm in strengths[:3]:
            logger.info(f"    Layer {layer}: norm={norm:.4f}")
    
    # Save vectors
    save_vectors(all_vectors, args.output_dir, args.model)
    
    logger.info("\n=== Collection Complete ===")
    logger.info(f"Vectors saved to: {args.output_dir}")
    logger.info(f"Metadata: {args.output_dir / 'vectors_metadata.json'}")


if __name__ == "__main__":
    main()
