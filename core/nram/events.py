"""
NRAM Token Event Model — truthful provenance for every emitted token.

Each token event records what happened to it during generation. The `origin`
field MUST be one of the defined provenance types. If the runtime cannot prove
that a specific bias caused the selected token, the origin is `applied_policy`
(not `causal_origin`).

This module defines the schema only. Event emission happens in the telemetry
module and the logit processor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TokenOrigin(str, Enum):
    """Truthful provenance for every emitted token.

    model_generated:      The model sampled this token without any NRAM
                          intervention affecting its probability.
    positive_logit_bias:  A positive logit bias was applied to this token's
                          ID before sampling. The bias increased its
                          probability, but we cannot prove it caused the
                          selection (applied_policy, not causal_origin).
    negative_logit_bias:  A negative logit bias was applied to this token's
                          ID, decreasing its probability. The token was
                          selected despite the penalty.
    hard_mask:            This token was NOT in the forbidden set. Other
                          tokens were masked to -inf, making this one the
                          only viable option (or one of few).
    forced_injection:     This token/fragment was deterministically inserted
                          by NRAM, bypassing model sampling entirely.
    memory_recall:        This token/fragment was injected from a Letta
                          memory block (episodic, canonical, or overlay).
    planner_constraint:   This token was shaped by the rhetorical planner's
                          structured output constraint (llguidance grammar).
    postprocess:          This token was modified after generation (e.g.,
                          reasoning-block stripping, formatting).
    applied_policy:       A steering policy was active for this token, but
                          we cannot prove it causally determined the
                          selection. This is the HONEST default when biases
                          are active but causality is unproven.
    """
    MODEL_GENERATED = "model_generated"
    POSITIVE_LOGIT_BIAS = "positive_logit_bias"
    NEGATIVE_LOGIT_BIAS = "negative_logit_bias"
    HARD_MASK = "hard_mask"
    FORCED_INJECTION = "forced_injection"
    MEMORY_RECALL = "memory_recall"
    PLANNER_CONSTRAINT = "planner_constraint"
    POSTPROCESS = "postprocess"
    APPLIED_POLICY = "applied_policy"


class NRAMStateSnapshot(BaseModel):
    """Point-in-time snapshot of NRAM simulation metrics.

    These are SIMULATION AND CONTROL METRICS, not measurements of
    consciousness. The UI must label them as such.
    """
    influence: float = Field(0.0, ge=0.0, le=1.0)
    entropy: float = Field(0.0, ge=0.0, le=1.0)
    coherence: float = Field(1.0, ge=0.0, le=1.0)
    momentum: float = Field(0.0, ge=0.0, le=1.0)


class TokenEvent(BaseModel):
    """A single emitted token with full provenance.

    If the runtime cannot prove that a specific bias caused the selected
    token, the origin field MUST be `applied_policy`, NOT `causal_origin`.
    """
    request_id: str
    session_id: Optional[str] = None
    position: int
    token_id: int
    text: str

    # Provenance — MUST be one of TokenOrigin values
    origin: TokenOrigin = TokenOrigin.MODEL_GENERATED

    # Phenomenon that triggered this event (if any)
    phenomenon: Optional[str] = None

    # Logit bias delta applied (for positive/negative_logit_bias origins)
    bias_delta: Optional[float] = None

    # Rhetorical phase at time of emission
    rhetorical_phase: str = "idle"

    # NRAM state before and after this token
    state_before: Optional[NRAMStateSnapshot] = None
    state_after: Optional[NRAMStateSnapshot] = None

    # Timestamp
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerationTelemetry(BaseModel):
    """Complete telemetry for a single generation request."""
    request_id: str
    session_id: Optional[str] = None

    # Timing
    first_token_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    planner_latency_ms: Optional[float] = None

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    # Steering configuration
    nram_enabled: bool = False
    profile: Optional[str] = None
    positive_token_count: int = 0
    negative_token_count: int = 0
    forbidden_token_count: int = 0
    positive_bias: float = 0.0
    negative_bias: float = 0.0
    repetition_penalty: float = 0.0

    # Phenomenon counts
    phenomenon_counts: Dict[str, int] = Field(default_factory=dict)

    # Token steering counts by origin
    origin_counts: Dict[str, int] = Field(default_factory=dict)

    # Token events (optional, only when include_telemetry=true)
    events: List[TokenEvent] = Field(default_factory=list)

    # Finish reason
    finish_reason: str = ""

    # Model info
    model: str = ""
    seed: Optional[int] = None


class CompareResult(BaseModel):
    """Result of a baseline vs NRAM comparison."""
    request_id: str
    prompt: str

    # Shared parameters
    model: str
    seed: Optional[int]
    temperature: float
    top_p: float
    max_tokens: int

    # Baseline output
    baseline_output: str
    baseline_latency_ms: float
    baseline_tokens: int
    baseline_finish_reason: str

    # NRAM output
    nram_output: str
    nram_latency_ms: float
    nram_tokens: int
    nram_finish_reason: str
    nram_profile: str

    # Policy differences
    positive_token_count: int = 0
    negative_token_count: int = 0
    forbidden_token_count: int = 0
    positive_bias: float = 0.0
    negative_bias: float = 0.0

    # Metrics comparison
    baseline_metrics: Dict[str, Any] = Field(default_factory=dict)
    nram_metrics: Dict[str, Any] = Field(default_factory=dict)
