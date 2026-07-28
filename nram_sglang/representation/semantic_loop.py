"""
Semantic closed-loop orchestration for NRAM.

Implements chunk-level feedback that operates on generated text,
not hidden states. The loop:
    1. Generate chunk
    2. Compute semantic/evidence score
    3. Apply controller decision
    4. Continue, retry, redirect, or stop
    5. Preserve complete decision trace

This runs at the orchestration layer (API/engine), not inside the
forward hook. It uses a pinned embedding model or deterministic
evaluator and records exact revision.

Each iteration records:
- generated text hash
- score components
- selected action
- updated config
- token count
- latency
- termination reason
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticAction(str, Enum):
    """Actions the semantic controller can take."""
    CONTINUE = "continue"
    RETRY = "retry"
    REDIRECT = "redirect"
    STOP = "stop"


@dataclass
class SemanticScore:
    """Score components from semantic evaluation."""
    novelty: float = 0.0
    coherence: float = 0.0
    source_distance: float = 0.0
    factual_preservation: float = 0.0
    overall: float = 0.0
    evaluator_revision: str = ""
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class SemanticIteration:
    """Record of a single semantic loop iteration."""
    iteration: int
    generated_text_hash: str
    generated_text: str
    token_count: int
    score: SemanticScore
    action: SemanticAction
    updated_config: Dict[str, Any]
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SemanticLoopConfig:
    """Configuration for semantic closed-loop."""
    enabled: bool = False
    target_novelty: float = 0.7
    target_coherence: float = 0.8
    max_iterations: int = 3
    chunk_size_tokens: int = 64
    evaluator_revision: str = "nram-semantic-eval-v1"
    novelty_weight: float = 0.4
    coherence_weight: float = 0.4
    source_distance_weight: float = 0.2
    retry_threshold: float = 0.3
    redirect_threshold: float = 0.5
    stop_threshold: float = 0.2


class SemanticClosedLoopController:
    """
    Controller for semantic closed-loop orchestration.
    
    Operates on generated text chunks, computing semantic scores
    and deciding whether to continue, retry, redirect, or stop.
    Maintains complete decision trace for auditability.
    """
    
    def __init__(
        self,
        config: SemanticLoopConfig,
        evaluator: Optional[Callable[[str, str], SemanticScore]] = None,
    ):
        self.config = config
        self.evaluator = evaluator or self._default_evaluator
        self.iterations: List[SemanticIteration] = []
        self._iteration_count: int = 0
        self._terminated: bool = False
        self._termination_reason: str = ""
    
    def evaluate_chunk(
        self,
        generated_text: str,
        source_text: str = "",
    ) -> SemanticScore:
        """
        Evaluate a generated chunk.
        
        Args:
            generated_text: The generated text to evaluate
            source_text: Original source text (for source_distance)
        
        Returns:
            SemanticScore with component scores
        """
        score = self.evaluator(generated_text, source_text)
        score.evaluator_revision = self.config.evaluator_revision
        
        # Compute overall weighted score
        score.overall = (
            self.config.novelty_weight * score.novelty
            + self.config.coherence_weight * score.coherence
            + self.config.source_distance_weight * score.source_distance
        )
        
        return score
    
    def decide_action(self, score: SemanticScore) -> SemanticAction:
        """
        Decide action based on score.
        
        Args:
            score: Semantic score from evaluation
        
        Returns:
            Selected action
        """
        if self._iteration_count >= self.config.max_iterations:
            return SemanticAction.STOP
        
        if score.overall < self.config.stop_threshold:
            return SemanticAction.STOP
        
        if score.overall < self.config.retry_threshold:
            return SemanticAction.RETRY
        
        if score.overall < self.config.redirect_threshold:
            return SemanticAction.REDIRECT
        
        return SemanticAction.CONTINUE
    
    def record_iteration(
        self,
        generated_text: str,
        token_count: int,
        score: SemanticScore,
        action: SemanticAction,
        updated_config: Dict[str, Any],
        latency_ms: float,
    ) -> SemanticIteration:
        """Record an iteration in the decision trace."""
        text_hash = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()[:16]
        
        iteration = SemanticIteration(
            iteration=self._iteration_count,
            generated_text_hash=text_hash,
            generated_text=generated_text,
            token_count=token_count,
            score=score,
            action=action,
            updated_config=updated_config,
            latency_ms=latency_ms,
        )
        
        self.iterations.append(iteration)
        self._iteration_count += 1
        
        if action == SemanticAction.STOP:
            self._terminated = True
            self._termination_reason = f"controller_decided_stop_at_iteration_{self._iteration_count}"
        
        logger.info(
            f"Semantic loop iteration {iteration.iteration}: "
            f"score={score.overall:.4f}, action={action.value}"
        )
        
        return iteration
    
    def is_terminated(self) -> bool:
        """Check if the loop has terminated."""
        return self._terminated
    
    def get_termination_reason(self) -> str:
        """Get reason for termination."""
        return self._termination_reason
    
    def get_decision_trace(self) -> List[Dict[str, Any]]:
        """Get complete decision trace for audit."""
        return [
            {
                "iteration": it.iteration,
                "text_hash": it.generated_text_hash,
                "token_count": it.token_count,
                "score_overall": it.score.overall,
                "score_components": {
                    "novelty": it.score.novelty,
                    "coherence": it.score.coherence,
                    "source_distance": it.score.source_distance,
                },
                "action": it.action.value,
                "latency_ms": it.latency_ms,
                "timestamp": it.timestamp,
            }
            for it in self.iterations
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "enabled": self.config.enabled,
            "iteration_count": self._iteration_count,
            "terminated": self._terminated,
            "termination_reason": self._termination_reason,
            "last_score": self.iterations[-1].score.overall if self.iterations else None,
            "decision_trace_len": len(self.iterations),
        }
    
    def _default_evaluator(
        self,
        generated_text: str,
        source_text: str,
    ) -> SemanticScore:
        """
        Default mechanical evaluator for functional testing.
        
        This is NOT a scientific evaluator. It provides deterministic
        scores for testing the control loop mechanics. For production
        use, replace with a pinned embedding model.
        """
        # Novelty: inverse of n-gram overlap with source
        novelty = 0.5  # Default neutral
        if source_text:
            gen_tokens = set(generated_text.lower().split())
            src_tokens = set(source_text.lower().split())
            if gen_tokens and src_tokens:
                overlap = len(gen_tokens & src_tokens) / max(len(gen_tokens), 1)
                novelty = 1.0 - overlap
        
        # Coherence: based on text length and structure
        coherence = min(1.0, len(generated_text) / 200.0)
        
        # Source distance: same as novelty for default
        source_distance = novelty
        
        return SemanticScore(
            novelty=novelty,
            coherence=coherence,
            source_distance=source_distance,
            factual_preservation=0.5,
            components={
                "novelty": novelty,
                "coherence": coherence,
                "source_distance": source_distance,
                "factual_preservation": 0.5,
            },
        )
