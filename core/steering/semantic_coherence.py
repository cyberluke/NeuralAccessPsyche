"""
Semantic coherence measurement for NRAM v5.

Measures:
- Adjacent-chunk cosine similarity (local coherence)
- Similarity to prompt/topic (global coherence)
- Rolling topic drift (coherence degradation)

Uses embedding model (Qwen3-Embedding-0.6B or sentence-transformers/all-MiniLM-L6-v2).

This module provides a more sophisticated coherence measurement than the
trigram-based approach in processor.py. It uses a separate embedding model
to compute semantic similarity between text chunks.

R5 Enhancements:
- Explicit model path and revision
- Startup preload or warm-up command
- Readiness state distinct from process health
- No hidden network download during run
- Model/tokenizer hashes in manifest
- Typed unavailable state if missing
- CUDA/XPU/CPU backend recorded
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CoherenceBackend(str, Enum):
    """Backend device for semantic coherence."""
    CUDA = "cuda"
    XPU = "xpu"
    CPU = "cpu"
    UNAVAILABLE = "unavailable"


class CoherenceReadiness(str, Enum):
    """Readiness state of semantic coherence runtime."""
    READY = "ready"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"
    NOT_INITIALIZED = "not_initialized"


@dataclass
class CoherenceModelManifest:
    """Manifest for semantic coherence model."""
    model_name: str
    model_revision: str
    model_path: Optional[str]
    model_sha256: Optional[str]
    tokenizer_sha256: Optional[str]
    backend: CoherenceBackend
    dtype: str
    hidden_size: int
    max_length: int
    loaded_at: float
    config_hash: str


@dataclass
class SemanticCoherenceResult:
    """Result from semantic coherence measurement."""
    adjacent_similarity: float
    prompt_similarity: float
    topic_drift: float
    num_chunks: int
    available: bool
    latency_ms: float


class SemanticCoherenceMeasurer:
    """Measure semantic coherence of generated text using embeddings.

    R5: Includes explicit readiness state, model manifest, and backend tracking.
    """

    # Default model configuration
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_MODEL_REVISION = "main"
    FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    FALLBACK_MODEL_REVISION = "main"

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_revision: Optional[str] = None,
        device: Optional[str] = None,
        preload: bool = True,
    ):
        """
        Initialize semantic coherence measurer.

        Args:
            model_name: Name or path of the embedding model (default: Qwen3-Embedding-0.6B)
            model_revision: Model revision (default: main)
            device: Device to use (default: auto-detect)
            preload: Whether to preload model at initialization (default: True)
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model_revision = model_revision or self.DEFAULT_MODEL_REVISION
        self._readiness = CoherenceReadiness.NOT_INITIALIZED
        self._manifest: Optional[CoherenceModelManifest] = None
        self._backend = CoherenceBackend.UNAVAILABLE
        self._load_start_time: Optional[float] = None

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        self.device = device

        # Model and tokenizer (loaded lazily or at preload)
        self.tokenizer: Any = None
        self.model: Any = None
        self._available = False

        # Preload if requested
        if preload:
            self._load_model()

    @property
    def readiness(self) -> CoherenceReadiness:
        """Current readiness state."""
        return self._readiness

    @property
    def available(self) -> bool:
        """Whether the embedding model is loaded and available."""
        return self._available and self._readiness == CoherenceReadiness.READY

    @property
    def backend(self) -> CoherenceBackend:
        """Backend device."""
        return self._backend

    @property
    def manifest(self) -> Optional[CoherenceModelManifest]:
        """Model manifest (if loaded)."""
        return self._manifest

    def _load_model(self) -> bool:
        """Load the embedding model. Returns True if successful."""
        self._readiness = CoherenceReadiness.LOADING
        self._load_start_time = time.monotonic()

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading semantic coherence model: {self.model_name} (revision: {self.model_revision})")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.model_revision,
            )

            # Load model
            dtype = torch.float16 if self.device in ("cuda", "xpu") else torch.float32
            self.model = AutoModel.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                torch_dtype=dtype,
            ).to(self.device)
            self.model.eval()

            # Determine backend
            if self.device == "cuda":
                self._backend = CoherenceBackend.CUDA
            elif self.device == "xpu":
                self._backend = CoherenceBackend.XPU
            else:
                self._backend = CoherenceBackend.CPU

            # Compute model and tokenizer hashes
            model_sha256 = self._compute_model_hash()
            tokenizer_sha256 = self._compute_tokenizer_hash()

            # Get model config
            hidden_size = getattr(self.model.config, "hidden_size", 0)
            max_length = getattr(self.model.config, "max_position_embeddings", 512)

            # Create manifest
            config_str = json.dumps({
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "device": self.device,
                "dtype": str(dtype),
                "hidden_size": hidden_size,
                "max_length": max_length,
            }, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

            self._manifest = CoherenceModelManifest(
                model_name=self.model_name,
                model_revision=self.model_revision,
                model_path=getattr(self.model, "name_or_path", None),
                model_sha256=model_sha256,
                tokenizer_sha256=tokenizer_sha256,
                backend=self._backend,
                dtype=str(dtype),
                hidden_size=hidden_size,
                max_length=max_length,
                loaded_at=time.time(),
                config_hash=config_hash,
            )

            self._available = True
            self._readiness = CoherenceReadiness.READY

            load_time_ms = (time.monotonic() - self._load_start_time) * 1000
            logger.info(
                f"SemanticCoherenceMeasurer loaded {self.model_name} on {self.device} "
                f"(backend={self._backend.value}, hidden_size={hidden_size}, "
                f"load_time={load_time_ms:.1f}ms, config_hash={config_hash})"
            )
            return True

        except Exception as e:
            logger.warning(f"SemanticCoherenceMeasurer failed to load {self.model_name}: {e}")

            # Try fallback model
            if self.model_name != self.FALLBACK_MODEL_NAME:
                logger.info(f"Trying fallback model: {self.FALLBACK_MODEL_NAME}")
                self.model_name = self.FALLBACK_MODEL_NAME
                self.model_revision = self.FALLBACK_MODEL_REVISION
                return self._load_model()

            # All models failed
            self._available = False
            self._readiness = CoherenceReadiness.UNAVAILABLE
            self._backend = CoherenceBackend.UNAVAILABLE
            self.tokenizer = None
            self.model = None
            return False

    def _compute_model_hash(self) -> Optional[str]:
        """Compute SHA-256 hash of model config."""
        try:
            config = self.model.config.to_dict()
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()
        except Exception:
            return None

    def _compute_tokenizer_hash(self) -> Optional[str]:
        """Compute SHA-256 hash of tokenizer config."""
        try:
            config = self.tokenizer.init_kwargs
            config_str = json.dumps(config, sort_keys=True, default=str)
            return hashlib.sha256(config_str.encode()).hexdigest()
        except Exception:
            return None

    def warmup(self, sample_text: str = "Hello, world!") -> bool:
        """Warm up the model with a sample inference.

        Args:
            sample_text: Text to use for warmup (default: "Hello, world!")

        Returns:
            True if warmup succeeded, False otherwise
        """
        if not self._available:
            return False

        try:
            _ = self.embed_text(sample_text)
            logger.info(f"SemanticCoherenceMeasurer warmup successful on device {self.device}")
            return True
        except Exception as e:
            logger.warning(f"SemanticCoherenceMeasurer warmup failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status."""
        status = {
            "readiness": self._readiness.value,
            "available": self._available,
            "backend": self._backend.value,
            "device": self.device,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
        }

        if self._manifest is not None:
            status["manifest"] = {
                "model_path": self._manifest.model_path,
                "model_sha256": self._manifest.model_sha256,
                "tokenizer_sha256": self._manifest.tokenizer_sha256,
                "dtype": self._manifest.dtype,
                "hidden_size": self._manifest.hidden_size,
                "max_length": self._manifest.max_length,
                "loaded_at": self._manifest.loaded_at,
                "config_hash": self._manifest.config_hash,
            }

        return status

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Compute embedding for text using mean pooling.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding tensor of shape (1, hidden_dim)
        """
        if not self._available:
            raise RuntimeError(
                f"SemanticCoherenceMeasurer not available (readiness={self._readiness.value})"
            )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Use mean pooling over the sequence dimension
        embedding = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embedding, p=2, dim=1)

    def measure_adjacent_similarity(self, text_chunks: List[str]) -> float:
        """
        Measure cosine similarity between adjacent chunks (local coherence).

        Args:
            text_chunks: List of text chunks to compare

        Returns:
            Average cosine similarity between adjacent chunks (0.0 to 1.0)
        """
        if not self._available:
            return 1.0  # Assume coherent if model not available

        if len(text_chunks) < 2:
            return 1.0

        embeddings = [self.embed_text(chunk) for chunk in text_chunks]
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = F.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim.item())
        return sum(similarities) / len(similarities)

    def measure_prompt_similarity(self, text_chunks: List[str], prompt: str) -> float:
        """
        Measure similarity of chunks to the original prompt (global coherence).

        Args:
            text_chunks: List of text chunks to compare
            prompt: Original prompt text

        Returns:
            Average cosine similarity to prompt (0.0 to 1.0)
        """
        if not self._available:
            return 1.0

        prompt_emb = self.embed_text(prompt)
        chunk_embs = [self.embed_text(chunk) for chunk in text_chunks]
        similarities = [
            F.cosine_similarity(prompt_emb, emb).item() for emb in chunk_embs
        ]
        return sum(similarities) / len(similarities) if similarities else 1.0

    def measure_topic_drift(self, text_chunks: List[str], window_size: int = 3) -> float:
        """
        Measure rolling topic drift using sliding window (coherence degradation).

        Args:
            text_chunks: List of text chunks to analyze
            window_size: Size of the sliding window

        Returns:
            Average topic drift (0.0 = no drift, 1.0 = complete drift)
        """
        if not self._available:
            return 0.0

        if len(text_chunks) < window_size:
            return 0.0

        embeddings = [self.embed_text(chunk) for chunk in text_chunks]
        drifts = []
        for i in range(len(embeddings) - window_size):
            window_start = embeddings[i]
            window_end = embeddings[i + window_size]
            drift = 1.0 - F.cosine_similarity(window_start, window_end).item()
            drifts.append(drift)
        return sum(drifts) / len(drifts) if drifts else 0.0

    def measure_all(
        self,
        text: str,
        prompt: str,
        chunk_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Measure all coherence metrics for the given text.

        Args:
            text: Generated text to analyze
            prompt: Original prompt text
            chunk_size: Number of words per chunk

        Returns:
            Dictionary with all coherence metrics and manifest info
        """
        start_time = time.monotonic()

        if not self._available:
            return {
                "adjacent_similarity": 1.0,
                "prompt_similarity": 1.0,
                "topic_drift": 0.0,
                "num_chunks": 0,
                "available": False,
                "readiness": self._readiness.value,
                "backend": self._backend.value,
                "latency_ms": 0.0,
            }

        # Split text into chunks by words
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        if not chunks:
            return {
                "adjacent_similarity": 1.0,
                "prompt_similarity": 1.0,
                "topic_drift": 0.0,
                "num_chunks": 0,
                "available": True,
                "readiness": self._readiness.value,
                "backend": self._backend.value,
                "latency_ms": (time.monotonic() - start_time) * 1000,
            }

        result = {
            "adjacent_similarity": self.measure_adjacent_similarity(chunks),
            "prompt_similarity": self.measure_prompt_similarity(chunks, prompt),
            "topic_drift": self.measure_topic_drift(chunks),
            "num_chunks": len(chunks),
            "available": True,
            "readiness": self._readiness.value,
            "backend": self._backend.value,
            "latency_ms": (time.monotonic() - start_time) * 1000,
        }

        # Include manifest hash if available
        if self._manifest is not None:
            result["model_config_hash"] = self._manifest.config_hash

        return result
