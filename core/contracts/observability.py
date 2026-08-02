"""Bounded, opt-in observability events for NRAM streaming clients."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ObservabilityLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    RESEARCH = "research"
    TRACE = "trace"


class StreamEventType(str, Enum):
    REQUEST_STARTED = "request_started"
    RUNTIME_STATE = "runtime_state"
    TOKEN = "token"
    STEERING_UPDATE = "steering_update"
    CANDIDATE_UPDATE = "candidate_update"
    WARNING = "warning"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"


class NRAMStreamEvent(BaseModel):
    """Versioned event with request-local ordering and bounded payloads."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "nram.stream-observability.v1"
    request_id: str
    sequence_number: int = Field(ge=0)
    monotonic_timestamp_ms: float = Field(ge=0)
    event_type: StreamEventType
    method: Optional[str] = None
    observability_level: ObservabilityLevel
    payload: Dict[str, Any] = Field(default_factory=dict)


class TokenEvidence(BaseModel):
    """Reduced scalar evidence; complete logits and hidden states are excluded."""

    model_config = ConfigDict(extra="forbid")
    token_position: int = Field(ge=0)
    token_id: Optional[int] = None
    token_text: str = ""
    emitted_at_ms: Optional[float] = None
    inter_token_latency_ms: Optional[float] = None
    selected_token_base_logp: Optional[float] = None
    selected_token_steered_logp: Optional[float] = None
    selected_token_delta_logp: Optional[float] = None
    base_argmax_token_id: Optional[int] = None
    steered_argmax_token_id: Optional[int] = None
    argmax_changed: Optional[bool] = None
    selected_token_base_rank: Optional[int] = None
    selected_token_steered_rank: Optional[int] = None
    steering_strength: Optional[float] = None
    classification: str = "STOCHASTIC_OR_UNCLASSIFIED"
    availability_reason: Optional[str] = None
    method_contributions: Dict[str, Optional[float]] = Field(default_factory=dict)
    protected_support_retained: Optional[bool] = None
    candidate_id: Optional[str] = None
