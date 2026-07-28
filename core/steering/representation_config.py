"""
Request-scoped representation control configuration for NRAM v5.

Defines typed configuration objects that travel with generation requests
and control hidden-state interventions during model forward passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from enum import Enum


class InterventionKind(str, Enum):
    """Types of representation interventions."""
    ACTIVATION_ADDITION = "activation_addition"
    CONCEPTOR = "conceptor"
    PROBE = "probe"
    CLOSED_LOOP = "closed_loop"
    REFT = "reft"


class TokenScope(str, Enum):
    """When to apply the intervention."""
    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"


class CombinationMode(str, Enum):
    """How to combine multiple vectors."""
    SUM = "sum"
    NORMALIZED_SUM = "normalized_sum"
    ORTHOGONALIZED_SUM = "orthogonalized_sum"
    NORM_BUDGETED_SUM = "norm_budgeted_sum"


@dataclass
class RepresentationIntervention:
    """Single representation intervention specification."""
    kind: InterventionKind
    intervention_id: str
    layer_name: str
    strength: float
    vector_or_operator_id: str
    token_scope: TokenScope = TokenScope.BOTH
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    
    def is_active_at_step(self, step: int, phase: str) -> bool:
        """Check if intervention is active at given step and phase."""
        # Check step bounds
        if self.start_step is not None and step < self.start_step:
            return False
        if self.end_step is not None and step > self.end_step:
            return False
        
        # Check phase
        if self.token_scope == TokenScope.PREFILL and phase != "prefill":
            return False
        if self.token_scope == TokenScope.DECODE and phase != "decode":
            return False
        
        return True


@dataclass
class NRAMRepresentationConfig:
    """Request-scoped representation control configuration."""
    enabled: bool = False
    request_id: Optional[str] = None
    interventions: List[RepresentationIntervention] = field(default_factory=list)
    telemetry_level: str = "summary"  # "none", "summary", "detailed"
    combination_mode: CombinationMode = CombinationMode.SUM
    norm_budget: Optional[float] = None
    orthogonalize: bool = False
    
    def get_interventions_for_layer(self, layer_name: str) -> List[RepresentationIntervention]:
        """Get all interventions targeting a specific layer."""
        return [iv for iv in self.interventions if iv.layer_name == layer_name]
    
    def get_interventions_for_step(
        self, step: int, phase: str
    ) -> List[RepresentationIntervention]:
        """Get all interventions active at given step and phase."""
        return [
            iv for iv in self.interventions
            if iv.is_active_at_step(step, phase)
        ]


@dataclass
class RepresentationRequestState:
    """Mutable per-request state for representation control."""
    request_id: str
    config: NRAMRepresentationConfig
    step_counter: int = 0
    hook_invocation_count: int = 0
    telemetry_events: List[dict] = field(default_factory=list)
    controller_state: dict = field(default_factory=dict)
    
    def reset(self) -> None:
        """Reset state for new request."""
        self.step_counter = 0
        self.hook_invocation_count = 0
        self.telemetry_events.clear()
        self.controller_state.clear()
    
    def record_telemetry(self, event: dict) -> None:
        """Record a telemetry event."""
        if self.config.telemetry_level != "none":
            self.telemetry_events.append(event)
    
    def advance_step(self) -> None:
        """Advance to next generation step."""
        self.step_counter += 1


@dataclass
class RepresentationTelemetry:
    """Telemetry from a representation intervention."""
    request_id: str
    decode_step: int
    layer_name: str
    intervention_id: str
    alpha: float
    hidden_norm_before: float
    hidden_norm_after: float
    delta_norm: float
    vector_norm: float
    cosine_hidden_vector: float
    hook_invocation_count: int
    prefill_or_decode: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "decode_step": self.decode_step,
            "layer_name": self.layer_name,
            "intervention_id": self.intervention_id,
            "alpha": self.alpha,
            "hidden_norm_before": self.hidden_norm_before,
            "hidden_norm_after": self.hidden_norm_after,
            "delta_norm": self.delta_norm,
            "vector_norm": self.vector_norm,
            "cosine_hidden_vector": self.cosine_hidden_vector,
            "hook_invocation_count": self.hook_invocation_count,
            "prefill_or_decode": self.prefill_or_decode,
        }
