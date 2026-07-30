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
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SemanticCoherenceMeasurer:
    """Measure semantic coherence of generated text using embeddings."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
    ):
        """
        Initialize semantic coherence measurer.

        Args:
            model_name: Name or path of the embedding model
            device: Device to use (default: "cuda")
        """
        self.device = device
        self.model_name = model_name

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(device)
            self.model.eval()
            self._available = True
            logger.info(f"SemanticCoherenceMeasurer loaded {model_name} on {device}")
        except Exception as e:
            logger.warning(f"SemanticCoherenceMeasurer failed to load: {e}")
            self._available = False
            self.tokenizer = None
            self.model = None

    @property
    def available(self) -> bool:
        """Whether the embedding model is loaded and available."""
        return self._available

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
            raise RuntimeError("SemanticCoherenceMeasurer not available")

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
    ) -> Dict[str, float]:
        """
        Measure all coherence metrics for the given text.

        Args:
            text: Generated text to analyze
            prompt: Original prompt text
            chunk_size: Number of words per chunk

        Returns:
            Dictionary with all coherence metrics
        """
        if not self._available:
            return {
                "adjacent_similarity": 1.0,
                "prompt_similarity": 1.0,
                "topic_drift": 0.0,
                "num_chunks": 0,
                "available": False,
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
            }

        return {
            "adjacent_similarity": self.measure_adjacent_similarity(chunks),
            "prompt_similarity": self.measure_prompt_similarity(chunks, prompt),
            "topic_drift": self.measure_topic_drift(chunks),
            "num_chunks": len(chunks),
            "available": True,
        }
