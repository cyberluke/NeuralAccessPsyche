"""
Create deterministic NRAM activation vector and conceptor artifacts.

These are fixture artifacts for runtime verification. They use a fixed
seed so that every run produces identical tensors, enabling reproducible
tests. The vectors are NOT collected from real model activations — they
are deterministic fixtures that prove the wiring is functional.

For production use, replace these with vectors collected via
scripts/collect_activation_vectors.py from real model activations.

Artifact format (safetensors):
  - vector: (hidden_dim,) float32 tensor
  - metadata: JSON string with model_hash, tokenizer_hash, layer, etc.

Conceptor artifact format (safetensors):
  - basis_vectors: (rank, hidden_dim) float32 tensor
  - singular_values: (rank,) float32 tensor
  - metadata: JSON string with aperture, layer, model_hash, etc.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

# Qwen3-14B hidden dimension
HIDDEN_DIM = 5120
# Default model/tokenizer hashes (match what the runtime expects)
MODEL_HASH = "qwen3-14b-awq"
TOKENIZER_HASH = "qwen3-14b-awq"
# Default layer for interventions (mid-network layer)
DEFAULT_LAYER = 20
# Conceptor rank
CONCEPTOR_RANK = 16


def compute_model_hash(model_path: str) -> str:
    """Compute a short hash for model identification."""
    return hashlib.sha256(model_path.encode()).hexdigest()[:16]


def create_activation_vector_artifact(
    output_path: Path,
    vector_id: str,
    hidden_dim: int = HIDDEN_DIM,
    layer: int = DEFAULT_LAYER,
    seed: int = 42,
    description: str = "Deterministic fixture vector for runtime verification",
) -> Path:
    """Create a real safetensors activation vector artifact."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors is required: pip install safetensors")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(hidden_dim, generator=rng, dtype=torch.float32)
    # L2 normalize so norm is 1.0
    vector = vector / vector.norm()

    metadata = {
        "model_hash": MODEL_HASH,
        "tokenizer_hash": TOKENIZER_HASH,
        "layer": layer,
        "pooling": "mean",
        "sample_count": 1000,
        "dataset_hash": hashlib.sha256(b"fixture_dataset_v1").hexdigest()[:16],
        "normalization": "l2",
        "description": description,
        "vector_id": vector_id,
        "creation_commit": "fixture",
    }

    metadata_str = json.dumps(metadata, sort_keys=True)
    data = {"vector": vector}
    save_file(data, str(output_path), metadata={"metadata": metadata_str})
    print(f"Created activation vector artifact: {output_path}")
    print(f"  vector_id={vector_id}, layer={layer}, norm={vector.norm().item():.4f}")
    return output_path


def create_conceptor_artifact(
    output_path: Path,
    conceptor_id: str,
    hidden_dim: int = HIDDEN_DIM,
    rank: int = CONCEPTOR_RANK,
    layer: int = DEFAULT_LAYER,
    aperture: float = 2.0,
    seed: int = 123,
    description: str = "Deterministic fixture conceptor for runtime verification",
) -> Path:
    """Create a real safetensors conceptor artifact."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors is required: pip install safetensors")

    rng = torch.Generator(device="cpu").manual_seed(seed)

    # Create orthonormal basis via QR decomposition of random matrix
    random_matrix = torch.randn(hidden_dim, rank, generator=rng, dtype=torch.float32)
    basis, _ = torch.linalg.qr(random_matrix)
    basis_vectors = basis[:, :rank].T  # (rank, hidden_dim)

    # Singular values: decreasing, positive
    singular_values = torch.linspace(3.0, 0.5, rank, dtype=torch.float32)

    metadata = {
        "model_hash": MODEL_HASH,
        "tokenizer_hash": TOKENIZER_HASH,
        "layer": layer,
        "aperture": aperture,
        "rank": rank,
        "hidden_size": hidden_dim,
        "dtype": "float32",
        "dataset_hash": hashlib.sha256(b"fixture_conceptor_dataset_v1").hexdigest()[:16],
        "description": description,
        "conceptor_id": conceptor_id,
        "creation_commit": "fixture",
    }

    metadata_str = json.dumps(metadata, sort_keys=True)
    data = {
        "basis_vectors": basis_vectors,
        "singular_values": singular_values,
    }
    save_file(data, str(output_path), metadata={"metadata": metadata_str})
    print(f"Created conceptor artifact: {output_path}")
    print(f"  conceptor_id={conceptor_id}, layer={layer}, rank={rank}, aperture={aperture}")
    return output_path


def main():
    artifact_dir = Path("artifacts/nram_vectors")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create activation vector artifacts for different NRAM profiles
    vectors = {
        "novelty_vs_paraphrase": {"seed": 42, "layer": 20, "description": "Novelty direction (fixture)"},
        "coherence_direction": {"seed": 43, "layer": 20, "description": "Coherence direction (fixture)"},
        "creativity_direction": {"seed": 44, "layer": 15, "description": "Creativity direction (fixture)"},
    }

    for vid, cfg in vectors.items():
        create_activation_vector_artifact(
            output_path=artifact_dir / f"{vid}.safetensors",
            vector_id=vid,
            layer=cfg["layer"],
            seed=cfg["seed"],
            description=cfg["description"],
        )

    # Create conceptor artifacts
    conceptors = {
        "novelty_coherence": {"seed": 123, "aperture": 2.0, "layer": 20},
        "creative_subspace": {"seed": 456, "aperture": 1.5, "layer": 15},
    }

    for cid, cfg in conceptors.items():
        create_conceptor_artifact(
            output_path=artifact_dir / f"{cid}.safetensors",
            conceptor_id=cid,
            layer=cfg["layer"],
            aperture=cfg["aperture"],
            seed=cfg["seed"],
        )

    print(f"\nAll artifacts created in {artifact_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
