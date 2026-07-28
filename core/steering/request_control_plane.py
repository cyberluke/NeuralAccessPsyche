"""
Unified request-scoped control plane for NRAM v5.

Integrates all steering layers into a single canonical request configuration:

  Evidence and concept plane
  → structural constraints
  → base/expert/anti distributions
  → logit control
  → representation hooks during model forward
  → sampling
  → latent feedback
  → block-level semantic feedback
  → optional branch tournament
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.steering.representation_config import (
    NRAMRepresentationConfig,
    RepresentationRequestState,
)
from core.steering.latent_closed_loop import ClosedLoopConfig, LatentClosedLoopController
from core.steering.semantic_closed_loop import SemanticLoopConfig, SemanticClosedLoopController
from core.steering.branch_tournament import BranchSearchConfig, BranchTournamentEngine

logger = logging.getLogger(__name__)


@dataclass
class StructuralControlConfig:
    """Configuration for structural controls (phrase constraints, etc.)."""
    enabled: bool = True
    forbidden_phrases: List[str] = field(default_factory=list)
    source_ngram_block: bool = True
    ngram_size: int = 5


@dataclass
class LogitControlConfig:
    """Configuration for logit-level controls."""
    enabled: bool = True
    positive_token_ids: List[int] = field(default_factory=list)
    negative_token_ids: List[int] = field(default_factory=list)
    forbidden_token_ids: List[int] = field(default_factory=list)
    positive_bias: float = 0.0
    negative_bias: float = 0.0
    repetition_penalty: float = 0.0
    corporate_jargon_penalty: float = 0.0
    entropy_target: float = 3.0
    entropy_kp: float = 0.5


@dataclass
class DExpertsConfig:
    """Configuration for DExperts steering."""
    enabled: bool = False
    expert_name: Optional[str] = None
    alpha: float = 1.0
    beta: float = 0.5
    expert_model_path: Optional[str] = None
    anti_expert_model_path: Optional[str] = None


@dataclass
class TelemetryConfig:
    """Configuration for telemetry."""
    level: str = "summary"  # "none", "summary", "detailed"
    include_hidden_states: bool = False
    include_probe_scores: bool = True
    include_branch_data: bool = True


@dataclass
class NRAMRequestConfig:
    """
    Canonical request configuration for NRAM v5.
    
    Contains all steering layer configurations.
    """
    request_id: str = ""
    structural: StructuralControlConfig = field(default_factory=StructuralControlConfig)
    logits: LogitControlConfig = field(default_factory=LogitControlConfig)
    representation: NRAMRepresentationConfig = field(default_factory=NRAMRepresentationConfig)
    semantic_loop: SemanticLoopConfig = field(default_factory=SemanticLoopConfig)
    search: BranchSearchConfig = field(default_factory=BranchSearchConfig)
    dexperts: DExpertsConfig = field(default_factory=DExpertsConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    latent_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"nram-{uuid.uuid4().hex[:12]}"


class NRAMRequestControlPlane:
    """
    Unified request-scoped control plane.
    
    Manages the lifecycle of all steering components for a single request.
    Ensures proper ordering and state isolation.
    """
    
    def __init__(self, config: NRAMRequestConfig):
        self.config = config
        self._representation_state = RepresentationRequestState(
            request_id=config.request_id,
            config=config.representation,
        )
        self._latent_loop = LatentClosedLoopController(config.latent_loop)
        self._semantic_loop = SemanticClosedLoopController(config.semantic_loop)
        self._branch_engine = BranchTournamentEngine(config.search)
        self._step = 0
        self._active = False
    
    def begin_request(self) -> None:
        """Begin processing a request. Reset all state."""
        self._representation_state.reset()
        self._latent_loop.begin_request(self.config.request_id)
        self._semantic_loop.begin_request(
            source_text=self.config.semantic_loop.source_text
        )
        self._step = 0
        self._active = True
        logger.info(f"Begin request: {self.config.request_id}")
    
    def end_request(self) -> None:
        """End processing a request. Clean up state."""
        self._latent_loop.end_request()
        self._semantic_loop.end_request()
        self._active = False
        logger.info(f"End request: {self.config.request_id}")
    
    def advance_step(self) -> None:
        """Advance to next generation step."""
        self._step += 1
        self._representation_state.advance_step()
        self._latent_loop.advance_step()
    
    @property
    def current_step(self) -> int:
        return self._step
    
    @property
    def representation_state(self) -> RepresentationRequestState:
        return self._representation_state
    
    @property
    def latent_loop(self) -> LatentClosedLoopController:
        return self._latent_loop
    
    @property
    def semantic_loop(self) -> SemanticClosedLoopController:
        return self._semantic_loop
    
    @property
    def branch_engine(self) -> BranchTournamentEngine:
        return self._branch_engine
    
    def should_run_tournament(self) -> bool:
        """Check if a branch tournament should run at current step."""
        if not self.config.search.enabled:
            return False
        return self._step in self.config.search.junction_steps
    
    def get_active_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active capabilities."""
        return {
            "structural": {
                "available": True,
                "configured": self.config.structural.enabled,
                "active": self.config.structural.enabled and self._active,
                "runtime_backend": "logit_processor",
            },
            "logit_control": {
                "available": True,
                "configured": self.config.logits.enabled,
                "active": self.config.logits.enabled and self._active,
                "runtime_backend": "logit_processor",
            },
            "representation": {
                "available": True,
                "configured": self.config.representation.enabled,
                "active": self.config.representation.enabled and self._active,
                "runtime_backend": "forward_hook",
            },
            "latent_closed_loop": {
                "available": True,
                "configured": self.config.latent_loop.enabled,
                "active": self.config.latent_loop.enabled and self._active,
                "runtime_backend": "probe_hook",
            },
            "semantic_closed_loop": {
                "available": True,
                "configured": self.config.semantic_loop.enabled,
                "active": self.config.semantic_loop.enabled and self._active,
                "runtime_backend": "embedding_model",
            },
            "branch_tournament": {
                "available": True,
                "configured": self.config.search.enabled,
                "active": self.config.search.enabled and self._active,
                "runtime_backend": "generation_orchestrator",
            },
            "dexperts": {
                "available": True,
                "configured": self.config.dexperts.enabled,
                "active": self.config.dexperts.enabled and self._active,
                "runtime_backend": "multi_model_inference",
            },
        }
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get telemetry summary for the request."""
        return {
            "request_id": self.config.request_id,
            "step": self._step,
            "representation": {
                "hook_invocations": self._representation_state.hook_invocation_count,
                "telemetry_events": len(self._representation_state.telemetry_events),
            },
            "latent_loop": self._latent_loop.get_status(),
            "semantic_loop": self._semantic_loop.get_status(),
            "branch_tournament": self._branch_engine.get_status(),
        }


def build_request_config_from_nram_opts(
    nram_opts: Dict[str, Any],
) -> NRAMRequestConfig:
    """
    Build NRAMRequestConfig from NRAM options dict.
    
    This is the bridge between the public API and the internal control plane.
    """
    config = NRAMRequestConfig()
    
    # Representation control
    rep_opts = nram_opts.get("representation", {})
    if rep_opts.get("enabled", False):
        config.representation.enabled = True
        config.representation.telemetry_level = rep_opts.get("telemetry_level", "summary")
    
    # Latent closed loop
    loop_opts = nram_opts.get("latent_loop", {})
    if loop_opts.get("enabled", False):
        config.latent_loop.enabled = True
    
    # Semantic closed loop
    sem_opts = nram_opts.get("semantic_loop", {})
    if sem_opts.get("enabled", False):
        config.semantic_loop.enabled = True
        config.semantic_loop.source_text = sem_opts.get("source_text")
        config.semantic_loop.chunk_size = sem_opts.get("chunk_size", 24)
    
    # Branch tournament
    search_opts = nram_opts.get("branch_search", {})
    if search_opts.get("enabled", False):
        config.search.enabled = True
        config.search.num_branches = search_opts.get("num_branches", 4)
    
    # DExperts
    dexperts_opts = nram_opts.get("dexperts", {})
    if dexperts_opts.get("enabled", False):
        config.dexperts.enabled = True
        config.dexperts.expert_name = dexperts_opts.get("expert_name")
        config.dexperts.alpha = dexperts_opts.get("alpha", 1.0)
        config.dexperts.beta = dexperts_opts.get("beta", 0.5)
    
    # Telemetry
    tel_opts = nram_opts.get("telemetry", {})
    config.telemetry.level = tel_opts.get("level", "summary")
    
    return config
