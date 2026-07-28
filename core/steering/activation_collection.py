"""
Activation collection pipeline for NRAM v5.

Collects hidden-state activations from contrastive datasets at specified
model layers, producing vector artifacts with immutable metadata.

Axes:
- novelty_vs_paraphrase
- concrete_vs_generic
- product_mechanism_vs_feature_list
- human_need_vs_technology
- coherent_vision_vs_fragmentation
- evidence_vs_hallucination
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrastive dataset schema
# ---------------------------------------------------------------------------

CONTRASTIVE_AXES = [
    "novelty_vs_paraphrase",
    "concrete_vs_generic",
    "product_mechanism_vs_feature_list",
    "human_need_vs_technology",
    "coherent_vision_vs_fragmentation",
    "evidence_vs_hallucination",
]


@dataclass
class ContrastivePair:
    """One positive/negative pair for a contrastive axis."""
    pair_id: str
    axis: str
    positive_text: str
    negative_text: str
    source_dataset: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationVectorMetadata:
    """Immutable metadata for a stored activation vector artifact."""
    vector_id: str
    axis: str
    model_hash: str
    tokenizer_hash: str
    layer_name: str
    layer_index: int
    pooling: str
    sample_count: int
    normalization: str
    vector_norm: float
    dataset_hash: str
    artifact_hash: str
    dtype: str = "float32"
    hidden_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "axis": self.axis,
            "model_hash": self.model_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "pooling": self.pooling,
            "sample_count": self.sample_count,
            "normalization": self.normalization,
            "vector_norm": self.vector_norm,
            "dataset_hash": self.dataset_hash,
            "artifact_hash": self.artifact_hash,
            "dtype": self.dtype,
            "hidden_dim": self.hidden_dim,
        }


# ---------------------------------------------------------------------------
# Layer index mapping
# ---------------------------------------------------------------------------

def compute_candidate_layers(total_layers: int) -> List[int]:
    """Map 25/50/70/85% depth percentages to actual layer indices."""
    percentages = [0.25, 0.50, 0.70, 0.85]
    return [min(int(p * total_layers), total_layers - 1) for p in percentages]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    Collects hidden-state activations from contrastive prompt pairs.

    Uses float32 accumulation, deterministic ordering, and explicit
    normalization. Stores vectors in safetensors format.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _register_capture_hook(self, layer_index: int) -> None:
        """Register a forward hook to capture hidden states at a layer."""
        target = self._get_layer_module(layer_index)
        if target is None:
            raise ValueError(f"Layer {layer_index} not found in model")

        def hook_fn(module, input_args, output):
            # Handle tuple outputs (common in transformer layers)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._captured[f"layer_{layer_index}"] = hidden.detach().cpu().float()

        handle = target.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _get_layer_module(self, layer_index: int) -> Optional[nn.Module]:
        """Get the module for a given layer index (Qwen3 topology)."""
        # Try common patterns
        patterns = [
            ("model.layers", lambda m, i: getattr(m.model.layers, str(i))),
            ("transformer.h", lambda m, i: getattr(m.transformer.h, str(i))),
        ]
        for pattern, accessor in patterns:
            parts = pattern.split(".")
            base = self.model
            valid = True
            for part in parts:
                if hasattr(base, part):
                    base = getattr(base, part)
                else:
                    valid = False
                    break
            if valid:
                try:
                    return accessor(self.model, layer_index)
                except (AttributeError, IndexError):
                    pass
        return None

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    @torch.no_grad()
    def collect_vector(
        self,
        pairs: List[ContrastivePair],
        axis: str,
        layer_index: int,
        layer_name: str,
        pooling: str = "mean",
        normalization: str = "l2",
    ) -> Tuple[torch.Tensor, ActivationVectorMetadata]:
        """
        Collect a contrastive activation vector from paired examples.

        Returns (vector, metadata).
        """
        axis_pairs = [p for p in pairs if p.axis == axis]
        if not axis_pairs:
            raise ValueError(f"No pairs for axis {axis}")

        # Register hook
        self._register_capture_hook(layer_index)

        try:
            positive_activations = []
            negative_activations = []

            for pair in axis_pairs:
                # Positive
                pos_act = self._extract_activation(pair.positive_text, layer_index, pooling)
                positive_activations.append(pos_act)

                # Negative
                neg_act = self._extract_activation(pair.negative_text, layer_index, pooling)
                negative_activations.append(neg_act)

            # Stack and compute mean difference in float32
            pos_stack = torch.stack(positive_activations, dim=0).float()
            neg_stack = torch.stack(negative_activations, dim=0).float()

            mean_pos = pos_stack.mean(dim=0)
            mean_neg = neg_stack.mean(dim=0)
            vector = mean_pos - mean_neg

            # Normalize
            if normalization == "l2":
                norm = vector.norm()
                if norm > 1e-8:
                    vector = vector / norm
            elif normalization == "none":
                pass
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

            # Compute metadata hashes
            model_hash = self._compute_model_hash()
            tokenizer_hash = self._compute_tokenizer_hash()
            dataset_hash = self._compute_dataset_hash(axis_pairs)

            hidden_dim = vector.shape[-1]
            vector_norm = vector.norm().item()

            # Create artifact hash from vector content
            vector_bytes = vector.cpu().numpy().tobytes()
            artifact_hash = hashlib.sha256(vector_bytes).hexdigest()

            vector_id = f"{axis}_layer{layer_index}_{pooling}"

            metadata = ActivationVectorMetadata(
                vector_id=vector_id,
                axis=axis,
                model_hash=model_hash,
                tokenizer_hash=tokenizer_hash,
                layer_name=layer_name,
                layer_index=layer_index,
                pooling=pooling,
                sample_count=len(axis_pairs),
                normalization=normalization,
                vector_norm=vector_norm,
                dataset_hash=dataset_hash,
                artifact_hash=artifact_hash,
                dtype="float32",
                hidden_dim=hidden_dim,
            )

            return vector, metadata

        finally:
            self._remove_hooks()

    def _extract_activation(
        self, text: str, layer_index: int, pooling: str
    ) -> torch.Tensor:
        """Extract activation from a single text at a given layer."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            _ = self.model(**inputs)

        captured = self._captured.get(f"layer_{layer_index}")
        if captured is None:
            raise RuntimeError(f"No activation captured for layer {layer_index}")

        # Pool over sequence dimension
        if pooling == "mean":
            return captured.mean(dim=1).squeeze(0)
        elif pooling == "last":
            return captured[:, -1, :].squeeze(0)
        elif pooling == "max":
            return captured.max(dim=1).values.squeeze(0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def _compute_model_hash(self) -> str:
        """Compute a hash of the model's state dict."""
        h = hashlib.sha256()
        for name, param in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(param.cpu().numpy().tobytes()[:1024])
        return h.hexdigest()[:16]

    def _compute_tokenizer_hash(self) -> str:
        """Compute a hash identifying the tokenizer."""
        vocab = getattr(self.tokenizer, "get_vocab", None)
        if callable(vocab):
            vocab_str = json.dumps(sorted(vocab().keys())[:100])
        else:
            vocab_str = "unknown"
        return hashlib.sha256(vocab_str.encode()).hexdigest()[:16]

    def _compute_dataset_hash(self, pairs: List[ContrastivePair]) -> str:
        """Compute a hash of the dataset content."""
        h = hashlib.sha256()
        for p in sorted(pairs, key=lambda x: x.pair_id):
            h.update(p.pair_id.encode())
            h.update(p.positive_text.encode())
            h.update(p.negative_text.encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------

class VectorArtifactRegistry:
    """
    Registry for activation vector artifacts.

    Stores vectors in safetensors format with immutable metadata.
    Validates artifacts on load.
    """

    def __init__(self, artifacts_dir: str = "artifacts/nram_vectors"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, ActivationVectorMetadata] = {}

    def store(
        self,
        vector: torch.Tensor,
        metadata: ActivationVectorMetadata,
    ) -> Path:
        """Store a vector artifact with metadata."""
        # Validate
        if torch.isnan(vector).any() or torch.isinf(vector).any():
            raise ValueError("Vector contains NaN or Inf values")

        # Save vector as safetensors
        try:
            from safetensors.torch import save_file
            tensor_path = self.artifacts_dir / f"{metadata.vector_id}.safetensors"
            save_file({"vector": vector.cpu().float()}, str(tensor_path))
        except ImportError:
            # Fallback to torch format
            tensor_path = self.artifacts_dir / f"{metadata.vector_id}.pt"
            torch.save({"vector": vector.cpu().float()}, str(tensor_path))

        # Save metadata
        meta_path = self.artifacts_dir / f"{metadata.vector_id}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        self._index[metadata.vector_id] = metadata
        logger.info(f"Stored vector artifact: {metadata.vector_id}")
        return tensor_path

    def load(
        self,
        vector_id: str,
        expected_model_hash: Optional[str] = None,
        expected_layer_name: Optional[str] = None,
        expected_hidden_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ActivationVectorMetadata]:
        """
        Load a vector artifact with validation.

        Raises on:
        - Wrong model hash
        - Incompatible dimensions
        - Incorrect dtype
        - NaN/Inf values
        - Unknown layer name
        - Corrupted artifact hash
        """
        if vector_id not in self._index:
            # Try loading from disk
            meta_path = self.artifacts_dir / f"{vector_id}.meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Vector artifact not found: {vector_id}")
            with open(meta_path) as f:
                data = json.load(f)
            metadata = ActivationVectorMetadata(**data)
            self._index[vector_id] = metadata
        else:
            metadata = self._index[vector_id]

        # Validate model hash
        if expected_model_hash and metadata.model_hash != expected_model_hash:
            raise ValueError(
                f"Model hash mismatch: expected {expected_model_hash}, "
                f"got {metadata.model_hash}"
            )

        # Validate layer name
        if expected_layer_name and metadata.layer_name != expected_layer_name:
            raise ValueError(
                f"Layer name mismatch: expected {expected_layer_name}, "
                f"got {metadata.layer_name}"
            )

        # Validate hidden dim
        if expected_hidden_dim and metadata.hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch: expected {expected_hidden_dim}, "
                f"got {metadata.hidden_dim}"
            )

        # Load tensor
        tensor_path_safetensors = self.artifacts_dir / f"{vector_id}.safetensors"
        tensor_path_pt = self.artifacts_dir / f"{vector_id}.pt"

        if tensor_path_safetensors.exists():
            from safetensors.torch import load_file
            data = load_file(str(tensor_path_safetensors))
            vector = data["vector"]
        elif tensor_path_pt.exists():
            data = torch.load(str(tensor_path_pt), map_location="cpu")
            vector = data["vector"]
        else:
            raise FileNotFoundError(f"Vector tensor not found for: {vector_id}")

        # Validate dtype
        if vector.dtype != torch.float32:
            raise ValueError(f"Expected float32, got {vector.dtype}")

        # Validate no NaN/Inf
        if torch.isnan(vector).any() or torch.isinf(vector).any():
            raise ValueError("Loaded vector contains NaN or Inf")

        # Validate artifact hash
        vector_bytes = vector.cpu().numpy().tobytes()
        computed_hash = hashlib.sha256(vector_bytes).hexdigest()
        if computed_hash != metadata.artifact_hash:
            raise ValueError(
                f"Artifact hash mismatch for {vector_id}: "
                f"expected {metadata.artifact_hash}, got {computed_hash}"
            )

        return vector, metadata

    def list_artifacts(self) -> List[ActivationVectorMetadata]:
        """List all registered artifacts."""
        return list(self._index.values())
