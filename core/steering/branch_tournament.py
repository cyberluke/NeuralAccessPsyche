"""
Branch-and-tournament generation for NRAM v5.

Explicit generation orchestration:
1. Snapshot current generation state
2. Generate N candidate branches
3. Score each branch
4. Choose winner through deterministic weighted policy
5. Continue from selected branch

Scorer dimensions:
- novelty
- coherence
- source distance
- concrete mechanism
- evidence preservation
- factual risk
- cliche risk
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BranchScorerWeights:
    """Weights for branch scoring dimensions."""
    novelty: float = 0.15
    coherence: float = 0.20
    source_distance: float = 0.10
    concrete_mechanism: float = 0.15
    evidence_preservation: float = 0.15
    factual_risk: float = -0.15  # Negative: penalize risk
    cliche_risk: float = -0.10   # Negative: penalize cliches
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "novelty": self.novelty,
            "coherence": self.coherence,
            "source_distance": self.source_distance,
            "concrete_mechanism": self.concrete_mechanism,
            "evidence_preservation": self.evidence_preservation,
            "factual_risk": self.factual_risk,
            "cliche_risk": self.cliche_risk,
        }


@dataclass
class BranchCandidate:
    """A candidate branch in the tournament."""
    branch_id: str
    text: str
    token_ids: List[int]
    scores: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    seed: Optional[int] = None
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentResult:
    """Result of a branch tournament."""
    tournament_id: str
    junction_step: int
    num_branches: int
    branches: List[BranchCandidate]
    winner_id: str
    scorer_weights: Dict[str, float]
    tie_breaker: str = "deterministic_seed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "junction_step": self.junction_step,
            "num_branches": self.num_branches,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "text": b.text,
                    "scores": b.scores,
                    "weighted_score": b.weighted_score,
                }
                for b in self.branches
            ],
            "winner_id": self.winner_id,
            "scorer_weights": self.scorer_weights,
        }


@dataclass
class BranchSearchConfig:
    """Configuration for branch-and-tournament search."""
    enabled: bool = False
    num_branches: int = 4
    junction_steps: List[int] = field(default_factory=list)  # Steps to trigger tournament
    branch_length: int = 32  # Tokens per branch
    scorer_weights: BranchScorerWeights = field(default_factory=BranchScorerWeights)
    single_trajectory_mode: bool = False  # If True, skip branching
    base_seed: int = 42


class BranchTournamentEngine:
    """
    Branch-and-tournament generation engine.
    
    Generates multiple candidate branches, scores them, and selects
    the winner through deterministic weighted policy.
    """
    
    def __init__(
        self,
        config: BranchSearchConfig,
        scorer_fn: Optional[Callable[[str, Dict[str, Any]], Dict[str, float]]] = None,
        generator_fn: Optional[Callable] = None,
    ):
        self.config = config
        self._scorer_fn = scorer_fn
        self._generator_fn = generator_fn
        self._tournaments: List[TournamentResult] = []
        self._tournament_counter = 0
    
    def run_tournament(
        self,
        prefix_text: str,
        prefix_token_ids: List[int],
        junction_step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TournamentResult:
        """
        Run a branch tournament.
        
        Args:
            prefix_text: Text prefix for all branches
            prefix_token_ids: Token IDs for the prefix
            junction_step: Current generation step
            context: Additional context for scoring
        
        Returns:
            TournamentResult with winner selected
        """
        if self.config.single_trajectory_mode:
            # Single trajectory: just generate one branch
            return self._single_trajectory(prefix_text, prefix_token_ids, junction_step)
        
        self._tournament_counter += 1
        tournament_id = f"tournament_{self._tournament_counter}"
        
        # Generate branches
        branches: List[BranchCandidate] = []
        for i in range(self.config.num_branches):
            branch_seed = self.config.base_seed + i + junction_step
            branch = self._generate_branch(
                prefix_text, prefix_token_ids, branch_seed, i
            )
            branches.append(branch)
        
        # Score branches
        for branch in branches:
            branch.scores = self._score_branch(branch, context or {})
            branch.weighted_score = self._compute_weighted_score(branch.scores)
        
        # Select winner (deterministic tie-breaking)
        winner = self._select_winner(branches)
        
        result = TournamentResult(
            tournament_id=tournament_id,
            junction_step=junction_step,
            num_branches=len(branches),
            branches=branches,
            winner_id=winner.branch_id,
            scorer_weights=self.config.scorer_weights.to_dict(),
        )
        
        self._tournaments.append(result)
        logger.info(
            f"Tournament {tournament_id}: winner={winner.branch_id} "
            f"score={winner.weighted_score:.3f}"
        )
        
        return result
    
    def _generate_branch(
        self,
        prefix_text: str,
        prefix_token_ids: List[int],
        seed: int,
        branch_index: int,
    ) -> BranchCandidate:
        """Generate a single branch."""
        branch_id = f"branch_{branch_index}"
        
        if self._generator_fn is not None:
            # Use provided generator
            text, token_ids = self._generator_fn(
                prefix_text, prefix_token_ids, seed, self.config.branch_length
            )
        else:
            # Placeholder: return empty branch
            text = ""
            token_ids = []
        
        return BranchCandidate(
            branch_id=branch_id,
            text=text,
            token_ids=token_ids,
            seed=seed,
            sampling_params={
                "temperature": 0.8,
                "max_tokens": self.config.branch_length,
            },
        )
    
    def _score_branch(
        self,
        branch: BranchCandidate,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Score a branch on all dimensions."""
        if self._scorer_fn is not None:
            return self._scorer_fn(branch.text, context)
        
        # Default heuristic scoring
        scores: Dict[str, float] = {}
        
        # Novelty: inverse of n-gram overlap with context
        source = context.get("source_text", "")
        if source:
            overlap = self._ngram_overlap(branch.text, source)
            scores["novelty"] = 1.0 - overlap
        else:
            scores["novelty"] = 0.5
        
        # Coherence: simple length-based heuristic
        scores["coherence"] = min(1.0, len(branch.text) / 100.0)
        
        # Source distance
        scores["source_distance"] = scores.get("novelty", 0.5)
        
        # Concrete mechanism (placeholder)
        scores["concrete_mechanism"] = 0.5
        
        # Evidence preservation (placeholder)
        scores["evidence_preservation"] = 0.5
        
        # Factual risk (placeholder)
        scores["factual_risk"] = 0.2
        
        # Cliche risk (placeholder)
        scores["cliche_risk"] = 0.1
        
        return scores
    
    def _compute_weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted score from dimension scores."""
        weights = self.config.scorer_weights
        total = 0.0
        for dim, weight in weights.to_dict().items():
            total += weight * scores.get(dim, 0.0)
        return total
    
    def _select_winner(self, branches: List[BranchCandidate]) -> BranchCandidate:
        """Select winner with deterministic tie-breaking."""
        if not branches:
            raise ValueError("No branches to select from")
        
        # Sort by weighted score descending, then by branch_id for determinism
        sorted_branches = sorted(
            branches,
            key=lambda b: (-b.weighted_score, b.branch_id),
        )
        return sorted_branches[0]
    
    def _single_trajectory(
        self,
        prefix_text: str,
        prefix_token_ids: List[int],
        junction_step: int,
    ) -> TournamentResult:
        """Single trajectory mode: just one branch."""
        self._tournament_counter += 1
        tournament_id = f"tournament_{self._tournament_counter}"
        
        branch = BranchCandidate(
            branch_id="branch_0",
            text="",
            token_ids=[],
            seed=self.config.base_seed,
        )
        
        return TournamentResult(
            tournament_id=tournament_id,
            junction_step=junction_step,
            num_branches=1,
            branches=[branch],
            winner_id="branch_0",
            scorer_weights=self.config.scorer_weights.to_dict(),
        )
    
    def _ngram_overlap(self, text1: str, text2: str, n: int = 3) -> float:
        """Compute n-gram overlap."""
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
    
    def get_tournaments(self) -> List[TournamentResult]:
        """Get all tournament results."""
        return self._tournaments
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.config.enabled,
            "num_branches": self.config.num_branches,
            "num_tournaments": len(self._tournaments),
            "single_trajectory_mode": self.config.single_trajectory_mode,
            "scorer_weights": self.config.scorer_weights.to_dict(),
        }
