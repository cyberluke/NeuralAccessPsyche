"""
NRAM Telemetry — collects token events during generation.

This module hooks into the SGLang streaming response to emit TokenEvents
with truthful provenance. It does NOT claim causal attribution unless
proven by counterfactual testing.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from core.nram.events import (
    GenerationTelemetry,
    NRAMStateSnapshot,
    TokenEvent,
    TokenOrigin,
)


class TelemetryCollector:
    """Collects token events during a single generation request."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_events: bool = False,
    ) -> None:
        self.request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        self.session_id = session_id
        self.include_events = include_events

        self.events: List[TokenEvent] = []
        self.origin_counts: Dict[str, int] = {o.value: 0 for o in TokenOrigin}
        self.phenomenon_counts: Dict[str, int] = {}

        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.reasoning_tokens: int = 0

        # NRAM config snapshot
        self.nram_enabled: bool = False
        self.profile: Optional[str] = None
        self.positive_token_count: int = 0
        self.negative_token_count: int = 0
        self.forbidden_token_count: int = 0
        self.positive_bias: float = 0.0
        self.negative_bias: float = 0.0
        self.repetition_penalty: float = 0.0
        self.seed: Optional[int] = None
        self.model: str = ""
        self.finish_reason: str = ""

        # State snapshots
        self._state_before: Optional[NRAMStateSnapshot] = None
        self._position: int = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def record_first_token(self) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def finish(self, finish_reason: str = "stop") -> None:
        self.end_time = time.perf_counter()
        self.finish_reason = finish_reason

    def record_token(
        self,
        token_id: int,
        text: str,
        origin: TokenOrigin = TokenOrigin.MODEL_GENERATED,
        phenomenon: Optional[str] = None,
        bias_delta: Optional[float] = None,
        rhetorical_phase: str = "idle",
    ) -> None:
        """Record a single token event."""
        self.completion_tokens += 1
        self.origin_counts[origin.value] = self.origin_counts.get(origin.value, 0) + 1

        if phenomenon:
            self.phenomenon_counts[phenomenon] = (
                self.phenomenon_counts.get(phenomenon, 0) + 1
            )

        if self.include_events:
            event = TokenEvent(
                request_id=self.request_id,
                session_id=self.session_id,
                position=self._position,
                token_id=token_id,
                text=text,
                origin=origin,
                phenomenon=phenomenon,
                bias_delta=bias_delta,
                rhetorical_phase=rhetorical_phase,
                state_before=self._state_before,
            )
            self.events.append(event)

        self._position += 1

    def to_telemetry(self) -> GenerationTelemetry:
        """Convert collected data to a GenerationTelemetry object."""
        first_token_ms = None
        total_ms = None
        if self.start_time is not None:
            if self.first_token_time is not None:
                first_token_ms = (self.first_token_time - self.start_time) * 1000
            if self.end_time is not None:
                total_ms = (self.end_time - self.start_time) * 1000

        return GenerationTelemetry(
            request_id=self.request_id,
            session_id=self.session_id,
            first_token_latency_ms=round(first_token_ms, 2) if first_token_ms else None,
            total_latency_ms=round(total_ms, 2) if total_ms else None,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            nram_enabled=self.nram_enabled,
            profile=self.profile,
            positive_token_count=self.positive_token_count,
            negative_token_count=self.negative_token_count,
            forbidden_token_count=self.forbidden_token_count,
            positive_bias=self.positive_bias,
            negative_bias=self.negative_bias,
            repetition_penalty=self.repetition_penalty,
            phenomenon_counts=self.phenomenon_counts,
            origin_counts=self.origin_counts,
            events=self.events if self.include_events else [],
            finish_reason=self.finish_reason,
            model=self.model,
            seed=self.seed,
        )


def classify_token_origin(
    token_id: int,
    positive_ids: List[int],
    negative_ids: List[int],
    forbidden_ids: List[int],
    nram_enabled: bool,
) -> TokenOrigin:
    """Classify the provenance of a token based on active steering.

    This is APPLIED_POLICY classification, not causal proof.
    We report what biases were active, not what caused the selection.
    """
    if not nram_enabled:
        return TokenOrigin.MODEL_GENERATED

    if token_id in forbidden_ids:
        # This token should have been masked — if it appears, something is wrong
        return TokenOrigin.HARD_MASK

    if token_id in positive_ids:
        return TokenOrigin.POSITIVE_LOGIT_BIAS

    if token_id in negative_ids:
        return TokenOrigin.NEGATIVE_LOGIT_BIAS

    # Biases were active but this specific token wasn't directly affected
    return TokenOrigin.APPLIED_POLICY
