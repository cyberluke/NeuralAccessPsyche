"""
Semantic/evidence closed-loop steering for NRAM v5.

Block-level evaluation using decoded text or embeddings.

Generation chunks (16-32 tokens):
  generate chunk
  → decode text
  → semantic/evidence evaluation
  → update next chunk configuration
  → continue generation

Measures:
- similarity to source
- similarity to previous generated sections
- source n-gram overlap
- forbidden-frame similarity
- evidence support
- contradiction risk
- semantic novelty

Controller actions:
- change latent-vector strength
- activate source blocker
- change entropy target
- activate an anti-paraphrase conceptor
- retry the last chunk
- switch branch-search policy
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SemanticEvaluation:
    """Evaluation of a generated chunk."""
    chunk_index: int
    text: str
    token_count: int
    source_similarity: float = 0.0
    previous_similarity: float = 0.0
    ngram_overlap: float = 0.0
    forbidden_similarity: float = 0.0
    evidence_support: float = 0.0
    contradiction_risk: float = 0.0
    semantic_novelty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "source_similarity": self.source_similarity,
            "previous_similarity": self.previous_similarity,
            "ngram_overlap": self.ngram_overlap,
            "forbidden_similarity": self.forbidden_similarity,
            "evidence_support": self.evidence_support,
            "contradiction_risk": self.contradiction_risk,
            "semantic_novelty": self.semantic_novelty,
        }


@dataclass
class SemanticControllerAction:
    """Action taken by the semantic controller."""
    action_type: str
    parameter: str
    value: float
    reason: str


@dataclass
class SemanticLoopConfig:
    """Configuration for semantic closed loop."""
    enabled: bool = False
    chunk_size: int = 24  # tokens per chunk
    source_text: Optional[str] = None
    forbidden_phrases: List[str] = field(default_factory=list)
    target_source_similarity: float = 0.3
    target_novelty: float = 0.7
    max_contradiction_risk: float = 0.3
    max_forbidden_similarity: float = 0.1
    retry_on_violation: bool = True
    max_retries: int = 2


class SemanticClosedLoopController:
    """
    Semantic closed-loop controller.
    
    Evaluates generated chunks and adjusts configuration for next chunks.
    Preserves exact chunk boundaries and controller decisions.
    """
    
    def __init__(self, config: SemanticLoopConfig, embedding_model: Optional[Any] = None):
        self.config = config
        self.embedding_model = embedding_model
        self._chunk_index = 0
        self._evaluations: List[SemanticEvaluation] = []
        self._actions: List[SemanticControllerAction] = []
        self._generated_texts: List[str] = []
        self._source_embedding: Optional[Any] = None
    
    def begin_request(self, source_text: Optional[str] = None) -> None:
        """Begin a new request."""
        self._chunk_index = 0
        self._evaluations.clear()
        self._actions.clear()
        self._generated_texts.clear()
        
        if source_text and self.embedding_model:
            self._source_embedding = self._compute_embedding(source_text)
    
    def end_request(self) -> None:
        """End request."""
        pass
    
    def evaluate_chunk(
        self,
        chunk_text: str,
        token_count: int,
    ) -> SemanticEvaluation:
        """
        Evaluate a generated chunk.
        
        Returns evaluation with all metrics.
        """
        evaluation = SemanticEvaluation(
            chunk_index=self._chunk_index,
            text=chunk_text,
            token_count=token_count,
        )
        
        # Compute metrics
        if self.embedding_model and self._source_embedding is not None:
            chunk_embedding = self._compute_embedding(chunk_text)
            evaluation.source_similarity = self._cosine_similarity(
                chunk_embedding, self._source_embedding
            )
        
        if self._generated_texts and self.embedding_model:
            previous_text = " ".join(self._generated_texts[-3:])
            previous_embedding = self._compute_embedding(previous_text)
            chunk_embedding = self._compute_embedding(chunk_text)
            evaluation.previous_similarity = self._cosine_similarity(
                chunk_embedding, previous_embedding
            )
        
        # N-gram overlap with source
        if self.config.source_text:
            evaluation.ngram_overlap = self._compute_ngram_overlap(
                chunk_text, self.config.source_text
            )
        
        # Forbidden phrase similarity
        if self.config.forbidden_phrases:
            evaluation.forbidden_similarity = self._check_forbidden_phrases(
                chunk_text, self.config.forbidden_phrases
            )
        
        # Semantic novelty (inverse of similarity to previous)
        evaluation.semantic_novelty = 1.0 - evaluation.previous_similarity
        
        self._evaluations.append(evaluation)
        self._generated_texts.append(chunk_text)
        self._chunk_index += 1
        
        return evaluation
    
    def decide_actions(
        self,
        evaluation: SemanticEvaluation,
    ) -> List[SemanticControllerAction]:
        """
        Decide controller actions based on evaluation.
        
        Returns list of actions to apply to next chunk.
        """
        actions: List[SemanticControllerAction] = []
        
        # Check source similarity
        if evaluation.source_similarity > self.config.target_source_similarity * 1.5:
            actions.append(SemanticControllerAction(
                action_type="reduce_source_similarity",
                parameter="source_blocker_strength",
                value=0.5,
                reason=f"Source similarity too high: {evaluation.source_similarity:.3f}",
            ))
        
        # Check novelty
        if evaluation.semantic_novelty < self.config.target_novelty * 0.5:
            actions.append(SemanticControllerAction(
                action_type="increase_novelty",
                parameter="novelty_vector_strength",
                value=0.3,
                reason=f"Novelty too low: {evaluation.semantic_novelty:.3f}",
            ))
        
        # Check contradiction risk (placeholder)
        if evaluation.contradiction_risk > self.config.max_contradiction_risk:
            actions.append(SemanticControllerAction(
                action_type="reduce_contradiction",
                parameter="coherence_vector_strength",
                value=0.4,
                reason=f"Contradiction risk too high: {evaluation.contradiction_risk:.3f}",
            ))
        
        # Check forbidden phrases
        if evaluation.forbidden_similarity > self.config.max_forbidden_similarity:
            actions.append(SemanticControllerAction(
                action_type="activate_anti_paraphrase",
                parameter="anti_paraphrase_conceptor_strength",
                value=0.6,
                reason=f"Forbidden similarity too high: {evaluation.forbidden_similarity:.3f}",
            ))
        
        self._actions.extend(actions)
        return actions
    
    def should_retry_chunk(self, evaluation: SemanticEvaluation) -> bool:
        """Check if chunk should be retried due to violations."""
        if not self.config.retry_on_violation:
            return False
        
        violations = 0
        if evaluation.forbidden_similarity > self.config.max_forbidden_similarity:
            violations += 1
        if evaluation.contradiction_risk > self.config.max_contradiction_risk:
            violations += 1
        
        return violations > 0
    
    def _compute_embedding(self, text: str) -> Any:
        """Compute embedding for text."""
        if self.embedding_model is None:
            return None
        # Placeholder for actual embedding computation
        return self.embedding_model.encode(text)
    
    def _cosine_similarity(self, emb1: Any, emb2: Any) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def _compute_ngram_overlap(self, text1: str, text2: str, n: int = 3) -> float:
        """Compute n-gram overlap between two texts."""
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union)
    
    def _check_forbidden_phrases(self, text: str, forbidden: List[str]) -> float:
        """Check for forbidden phrases in text."""
        text_lower = text.lower()
        matches = sum(1 for phrase in forbidden if phrase.lower() in text_lower)
        return min(1.0, matches / max(1, len(forbidden)))
    
    def get_telemetry(self) -> List[Dict[str, Any]]:
        """Get all telemetry."""
        return [e.to_dict() for e in self._evaluations]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.config.enabled,
            "chunk_index": self._chunk_index,
            "num_evaluations": len(self._evaluations),
            "num_actions": len(self._actions),
            "last_evaluation": self._evaluations[-1].to_dict() if self._evaluations else None,
        }
