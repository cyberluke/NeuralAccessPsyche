"""NRAM metrics and telemetry."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Structured metrics for a single NRAM request."""

    request_id: str = ""
    public_model: str = ""
    upstream_model: str = ""
    nram_enabled: bool = False
    persona_profile: str = ""
    planner_latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    positive_token_count: int = 0
    negative_token_count: int = 0
    forbidden_token_count: int = 0
    positive_bias: float = 0.0
    negative_bias: float = 0.0
    seed: Optional[int] = None
    finish_reason: str = ""
    upstream_error: Optional[str] = None

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for structured logging. Never logs secrets."""
        return {
            "request_id": self.request_id,
            "public_model": self.public_model,
            "upstream_model": self.upstream_model,
            "nram_enabled": self.nram_enabled,
            "persona_profile": self.persona_profile,
            "planner_latency_ms": round(self.planner_latency_ms, 2),
            "first_token_latency_ms": round(self.first_token_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "positive_token_count": self.positive_token_count,
            "negative_token_count": self.negative_token_count,
            "forbidden_token_count": self.forbidden_token_count,
            "positive_bias": self.positive_bias,
            "negative_bias": self.negative_bias,
            "seed": self.seed,
            "finish_reason": self.finish_reason,
            "upstream_error": self.upstream_error,
        }


class MetricsCollector:
    """Collects and logs request metrics."""

    def __init__(self) -> None:
        self._requests: list[RequestMetrics] = []

    def record(self, metrics: RequestMetrics) -> None:
        self._requests.append(metrics)
        logger.info(f"NRAM_METRICS: {metrics.to_log_dict()}")

    @property
    def total_requests(self) -> int:
        return len(self._requests)
