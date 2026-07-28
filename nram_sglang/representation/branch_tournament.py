"""
Branch-and-tournament generation for NRAM.

Generates genuinely distinct branches with controlled variation,
preserves every branch output, scores every branch with the same
documented objective, selects winner from scores, and returns
winner + optional branch evidence.

Each branch uses:
- Different random seed
- Different intervention genome (strength, vectors)
- Or different temperature/sampling parameters

The scorer uses the same documented objective for all branches.
Partial branch failure is handled without silently selecting
unevaluated candidates.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BranchStatus(str, Enum):
    """Status of a branch generation."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SCORED = "scored"


@dataclass
class BranchConfig:
    """Configuration for a single branch."""
    branch_id: str
    seed: Optional[int] = None
    temperature: float = 0.8
    intervention_genome: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 256


@dataclass
class BranchResult:
    """Result from a single branch generation."""
    branch_id: str
    status: BranchStatus
    generated_text: str = ""
    text_hash: str = ""
    token_count: int = 0
    score: Optional[float] = None
    score_components: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TournamentConfig:
    """Configuration for branch-and-tournament."""
    num_branches: int = 4
    scorer_objective: str = "novelty_weighted"
    max_tokens_per_branch: int = 256
    base_temperature: float = 0.8
    temperature_increment: float = 0.1
    base_seed: Optional[int] = None
    seed_increment: int = 1
    include_branch_evidence: bool = True
    handle_partial_failure: bool = True


class BranchAndTournamentGenerator:
    """
    Generator that creates multiple distinct branches and selects the best.
    
    Each branch is a genuine model generation with different:
    - Random seed
    - Temperature
    - Intervention parameters
    
    All branches are scored with the same objective function.
    The winner is selected from scored branches.
    """
    
    def __init__(
        self,
        config: TournamentConfig,
        generation_fn: Optional[Callable[[BranchConfig], BranchResult]] = None,
        scoring_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    ):
        self.config = config
        self.generation_fn = generation_fn or self._default_generation_fn
        self.scoring_fn = scoring_fn or self._default_scoring_fn
        self.branches: List[BranchResult] = []
        self.winner: Optional[BranchResult] = None
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
    
    def create_branch_configs(self) -> List[BranchConfig]:
        """Create configurations for all branches."""
        configs = []
        for i in range(self.config.num_branches):
            seed = (
                (self.config.base_seed or 42) + i * self.config.seed_increment
                if self.config.base_seed is not None
                else None
            )
            temperature = self.config.base_temperature + i * self.config.temperature_increment
            
            branch_config = BranchConfig(
                branch_id=f"branch_{i}",
                seed=seed,
                temperature=temperature,
                max_tokens=self.config.max_tokens_per_branch,
                intervention_genome={
                    "branch_index": i,
                    "temperature": temperature,
                },
            )
            configs.append(branch_config)
        
        return configs
    
    def generate_branch(self, config: BranchConfig) -> BranchResult:
        """Generate a single branch."""
        start_time = time.time()
        
        try:
            result = self.generation_fn(config)
            result.latency_ms = (time.time() - start_time) * 1000
            
            if result.status == BranchStatus.COMPLETED:
                result.text_hash = hashlib.sha256(
                    result.generated_text.encode("utf-8")
                ).hexdigest()[:16]
            
            self._total_tokens += result.token_count
            self._total_latency_ms += result.latency_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Branch {config.branch_id} failed: {e}")
            return BranchResult(
                branch_id=config.branch_id,
                status=BranchStatus.FAILED,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    def score_branch(self, result: BranchResult) -> BranchResult:
        """Score a completed branch."""
        if result.status != BranchStatus.COMPLETED:
            return result
        
        try:
            components = self.scoring_fn(result.generated_text)
            # Overall score is weighted sum
            score = sum(components.values()) / max(len(components), 1)
            result.score = score
            result.score_components = components
            result.status = BranchStatus.SCORED
        except Exception as e:
            logger.error(f"Scoring branch {result.branch_id} failed: {e}")
            result.status = BranchStatus.FAILED
            result.error = f"scoring_failed: {e}"
        
        return result
    
    def run_tournament(self, prompt: str = "") -> Optional[BranchResult]:
        """
        Run the full branch-and-tournament process.
        
        1. Generate all branches
        2. Score all completed branches
        3. Select winner from scored branches
        4. Handle partial failures
        
        Args:
            prompt: The generation prompt
        
        Returns:
            Winning BranchResult, or None if all branches failed
        """
        branch_configs = self.create_branch_configs()
        self.branches = []
        
        # Generate all branches
        for config in branch_configs:
            result = self.generate_branch(config)
            self.branches.append(result)
        
        # Score completed branches
        scored_branches = []
        for result in self.branches:
            if result.status == BranchStatus.COMPLETED:
                scored = self.score_branch(result)
                if scored.status == BranchStatus.SCORED:
                    scored_branches.append(scored)
        
        # Handle partial failure
        if not scored_branches:
            if self.config.handle_partial_failure:
                logger.warning("All branches failed or unscored")
                return None
            else:
                raise RuntimeError("All branches failed and partial failure not allowed")
        
        # Select winner
        self.winner = max(scored_branches, key=lambda b: b.score or 0.0)
        
        logger.info(
            f"Tournament winner: {self.winner.branch_id} "
            f"with score {self.winner.score:.4f}"
        )
        
        return self.winner
    
    def get_tournament_result(self) -> Dict[str, Any]:
        """Get complete tournament result for API response."""
        return {
            "winner": {
                "branch_id": self.winner.branch_id if self.winner else None,
                "text": self.winner.generated_text if self.winner else "",
                "score": self.winner.score if self.winner else None,
                "score_components": self.winner.score_components if self.winner else {},
                "token_count": self.winner.token_count if self.winner else 0,
                "latency_ms": self.winner.latency_ms if self.winner else 0.0,
            } if self.winner else None,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "status": b.status.value,
                    "text_hash": b.text_hash,
                    "token_count": b.token_count,
                    "score": b.score,
                    "score_components": b.score_components,
                    "latency_ms": b.latency_ms,
                    "error": b.error,
                }
                for b in self.branches
            ] if self.config.include_branch_evidence else [],
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "scorer_objective": self.config.scorer_objective,
            "num_branches": self.config.num_branches,
            "num_scored": sum(1 for b in self.branches if b.status == BranchStatus.SCORED),
            "num_failed": sum(1 for b in self.branches if b.status == BranchStatus.FAILED),
        }
    
    def _default_generation_fn(self, config: BranchConfig) -> BranchResult:
        """
        Default generation function (placeholder).
        
        In production, this is replaced by actual model generation
        via the SGLang engine with branch-specific parameters.
        """
        return BranchResult(
            branch_id=config.branch_id,
            status=BranchStatus.COMPLETED,
            generated_text=f"[branch_{config.branch_id}_generated_text]",
            token_count=config.max_tokens,
        )
    
    def _default_scoring_fn(self, text: str) -> Dict[str, float]:
        """
        Default scoring function (mechanical).
        
        In production, replaced with pinned embedding model evaluation.
        """
        # Mechanical scoring for functional testing
        length_score = min(1.0, len(text) / 500.0)
        uniqueness = len(set(text.lower().split())) / max(len(text.split()), 1)
        
        return {
            "length": length_score,
            "uniqueness": uniqueness,
        }
