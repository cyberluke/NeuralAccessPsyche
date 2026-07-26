"""
Feature 3: NRAM Telemetry Aggregation — provenance analytics per session.

Tracks token-level provenance (baseline, biased, masked, fragmented, etc.)
and aggregates into per-session statistics. Exposed via REST for dashboards.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionTelemetry:
    """Aggregated telemetry for one session."""
    session_id: str
    total_requests: int = 0
    total_tokens: int = 0
    token_origin_counts: Dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    profiles_used: Dict[str, int] = field(default_factory=dict)
    phenomena_triggered: Dict[str, int] = field(default_factory=dict)
    last_request_at: float = 0.0

    def record_request(
        self,
        profile: str,
        token_origins: Dict[str, int],
        latency_ms: float,
        phenomena: Optional[Dict[str, int]] = None,
    ) -> None:
        self.total_requests += 1
        tokens_this = sum(token_origins.values())
        self.total_tokens += tokens_this
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_requests
        self.profiles_used[profile] = self.profiles_used.get(profile, 0) + 1
        self.last_request_at = time.time()

        for origin, count in token_origins.items():
            self.token_origin_counts[origin] = (
                self.token_origin_counts.get(origin, 0) + count
            )

        if phenomena:
            for name, count in phenomena.items():
                self.phenomena_triggered[name] = (
                    self.phenomena_triggered.get(name, 0) + count
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "token_origin_counts": dict(self.token_origin_counts),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "profiles_used": dict(self.profiles_used),
            "phenomena_triggered": dict(self.phenomena_triggered),
            "last_request_at": self.last_request_at,
            "dominant_origin": max(
                self.token_origin_counts, key=self.token_origin_counts.get
            ) if self.token_origin_counts else None,
        }


class TelemetryAggregator:
    """In-memory telemetry aggregator. Swap for ClickHouse/Postgres in prod."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionTelemetry] = {}

    def get_or_create(self, session_id: str) -> SessionTelemetry:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionTelemetry(session_id=session_id)
        return self._sessions[session_id]

    def record(
        self,
        session_id: str,
        profile: str,
        token_origins: Dict[str, int],
        latency_ms: float,
        phenomena: Optional[Dict[str, int]] = None,
    ) -> None:
        self.get_or_create(session_id).record_request(
            profile, token_origins, latency_ms, phenomena
        )

    def summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self._sessions.get(session_id)
        return s.to_dict() if s else None

    def all_summaries(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def global_stats(self) -> Dict[str, Any]:
        total_tokens = sum(s.total_tokens for s in self._sessions.values())
        total_requests = sum(s.total_requests for s in self._sessions.values())
        all_origins: Counter = Counter()
        for s in self._sessions.values():
            all_origins.update(s.token_origin_counts)
        return {
            "total_sessions": len(self._sessions),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "token_origin_distribution": dict(all_origins),
        }


# Singleton
telemetry_aggregator = TelemetryAggregator()
