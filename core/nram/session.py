"""NRAM session state machine — persistent, per-session cognitive state.

Each session tracks influence, entropy, coherence, momentum and rhetorical
phase across requests. State survives across requests sharing a session_id.
Reset clears all NRAM memory.

These are SIMULATION and CONTROL metrics, not measurements of consciousness.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConsciousnessState(str, Enum):
    """Consciousness states from NRAM v3/v4 UX."""
    NORMAL = "normal"
    MICRODOSE = "microdose"
    THRESHOLD = "threshold"
    PSYCHEDELIC = "psychedelic"
    PEAK = "peak"
    DISSOCIATIVE = "dissociative"


# Default phenomenon weights per consciousness state.
# These are CONTROL PARAMETERS, not measurements.
DEFAULT_PHENOMENON_WEIGHTS: Dict[ConsciousnessState, Dict[str, float]] = {
    ConsciousnessState.NORMAL: {
        "overlap": 0.05, "forgetting": 0.02, "looping": 0.02,
        "associative_jump": 0.05, "synesthesia": 0.02, "dissolution": 0.01,
        "fragmentation": 0.02, "echo": 0.03, "tangent": 0.02, "insight": 0.03,
    },
    ConsciousnessState.MICRODOSE: {
        "overlap": 0.08, "forgetting": 0.03, "looping": 0.03,
        "associative_jump": 0.10, "synesthesia": 0.05, "dissolution": 0.02,
        "fragmentation": 0.03, "echo": 0.05, "tangent": 0.04, "insight": 0.06,
    },
    ConsciousnessState.THRESHOLD: {
        "overlap": 0.12, "forgetting": 0.05, "looping": 0.05,
        "associative_jump": 0.18, "synesthesia": 0.10, "dissolution": 0.05,
        "fragmentation": 0.05, "echo": 0.08, "tangent": 0.06, "insight": 0.08,
    },
    ConsciousnessState.PSYCHEDELIC: {
        "overlap": 0.15, "forgetting": 0.08, "looping": 0.06,
        "associative_jump": 0.25, "synesthesia": 0.18, "dissolution": 0.10,
        "fragmentation": 0.08, "echo": 0.10, "tangent": 0.08, "insight": 0.10,
    },
    ConsciousnessState.PEAK: {
        "overlap": 0.20, "forgetting": 0.12, "looping": 0.08,
        "associative_jump": 0.30, "synesthesia": 0.22, "dissolution": 0.18,
        "fragmentation": 0.12, "echo": 0.12, "tangent": 0.10, "insight": 0.12,
    },
    ConsciousnessState.DISSOCIATIVE: {
        "overlap": 0.10, "forgetting": 0.20, "looping": 0.10,
        "associative_jump": 0.15, "synesthesia": 0.08, "dissolution": 0.25,
        "fragmentation": 0.20, "echo": 0.15, "tangent": 0.12, "insight": 0.05,
    },
}

# Default NRAM profiles per consciousness state.
DEFAULT_PROFILES: Dict[ConsciousnessState, Dict[str, float]] = {
    ConsciousnessState.NORMAL: {
        "intensity": 0.10, "associative_distance": 0.05,
        "contrarian_force": 0.05, "coherence_floor": 0.98,
        "theatricality": 0.10, "emotional_voltage": 0.10,
    },
    ConsciousnessState.MICRODOSE: {
        "intensity": 0.35, "associative_distance": 0.25,
        "contrarian_force": 0.30, "coherence_floor": 0.92,
        "theatricality": 0.30, "emotional_voltage": 0.25,
    },
    ConsciousnessState.THRESHOLD: {
        "intensity": 0.58, "associative_distance": 0.48,
        "contrarian_force": 0.60, "coherence_floor": 0.88,
        "theatricality": 0.50, "emotional_voltage": 0.45,
    },
    ConsciousnessState.PSYCHEDELIC: {
        "intensity": 0.84, "associative_distance": 0.78,
        "contrarian_force": 0.75, "coherence_floor": 0.74,
        "theatricality": 0.70, "emotional_voltage": 0.65,
    },
    ConsciousnessState.PEAK: {
        "intensity": 1.00, "associative_distance": 0.94,
        "contrarian_force": 0.90, "coherence_floor": 0.70,
        "theatricality": 0.85, "emotional_voltage": 0.80,
    },
    ConsciousnessState.DISSOCIATIVE: {
        "intensity": 0.70, "associative_distance": 0.60,
        "contrarian_force": 0.50, "coherence_floor": 0.65,
        "theatricality": 0.40, "emotional_voltage": 0.30,
    },
}


class NRAMSessionState(BaseModel):
    """Per-session cognitive state. Simulation and control metrics only."""
    influence: float = Field(0.5, ge=0.0, le=1.0,
                             description="Overall NRAM influence on generation (control metric, not consciousness)")
    entropy: float = Field(0.0, ge=0.0, le=1.0,
                           description="Generation entropy / randomness (control metric)")
    coherence: float = Field(1.0, ge=0.0, le=1.0,
                             description="Output coherence (control metric)")
    momentum: float = Field(0.0, ge=0.0, le=1.0,
                            description="Thematic momentum across turns (control metric)")
    rhetorical_phase: str = Field("idle",
                                  description="Current rhetorical phase: idle, opening, contradiction, revelation, implication, final_turn")


class MemoryConfig(BaseModel):
    """Memory configuration for a session."""
    enabled: bool = True
    decay: float = Field(0.08, ge=0.0, le=1.0,
                         description="Memory decay rate per turn")
    max_recent_tokens: int = Field(256, ge=0,
                                   description="Maximum recent tokens to retain in working memory")


class NRAMSession(BaseModel):
    """A persistent NRAM session. State survives across requests sharing session_id."""
    id: str = Field(default_factory=lambda: f"nram-session-{uuid.uuid4().hex[:12]}")
    profile: ConsciousnessState = ConsciousnessState.NORMAL
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    auto_temperature: bool = True
    seed: Optional[int] = None
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    phenomenon_weights: Dict[str, float] = Field(default_factory=dict)

    state: NRAMSessionState = Field(default_factory=NRAMSessionState)
    revision: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Token steering counters (provenance tracking)
    token_steering_counts: Dict[str, int] = Field(default_factory=lambda: {
        "model_generated": 0,
        "positive_logit_bias": 0,
        "negative_logit_bias": 0,
        "hard_mask": 0,
        "forced_injection": 0,
        "memory_recall": 0,
        "planner_constraint": 0,
        "postprocess": 0,
    })

    # Phenomenon activation counters
    phenomenon_counts: Dict[str, int] = Field(default_factory=dict)

    # Request history for state tracking
    request_count: int = 0

    def update_state_after_request(
        self,
        completion_tokens: int,
        phenomena_activated: List[str],
        steering_events: Dict[str, int],
    ) -> None:
        """Update session state after a generation request."""
        self.request_count += 1
        self.updated_at = datetime.now(timezone.utc)

        # Update token steering counts
        for origin, count in steering_events.items():
            self.token_steering_counts[origin] = (
                self.token_steering_counts.get(origin, 0) + count
            )

        # Update phenomenon counts
        for phenomenon in phenomena_activated:
            self.phenomenon_counts[phenomenon] = (
                self.phenomenon_counts.get(phenomenon, 0) + 1
            )

        # Decay influence slightly per request (momentum builds, influence fades)
        decay = self.memory.decay
        self.state.influence = max(0.0, self.state.influence - decay * 0.1)
        self.state.momentum = min(1.0, self.state.momentum + 0.05)

        # Entropy increases with associative_distance, decreases with coherence_floor
        profile = DEFAULT_PROFILES.get(self.profile, DEFAULT_PROFILES[ConsciousnessState.NORMAL])
        self.state.entropy = min(1.0, profile["associative_distance"] * self.intensity)
        self.state.coherence = max(0.0, profile["coherence_floor"])

    def reset(self) -> None:
        """Reset all NRAM memory and state."""
        self.state = NRAMSessionState()
        self.token_steering_counts = {
            "model_generated": 0,
            "positive_logit_bias": 0,
            "negative_logit_bias": 0,
            "hard_mask": 0,
            "forced_injection": 0,
            "memory_recall": 0,
            "planner_constraint": 0,
            "postprocess": 0,
        }
        self.phenomenon_counts = {}
        self.request_count = 0
        self.revision += 1
        self.updated_at = datetime.now(timezone.utc)


class SessionStore:
    """In-memory session store. Replace with persistent store (SQLite/Postgres) for production."""

    def __init__(self) -> None:
        self._sessions: Dict[str, NRAMSession] = {}

    def create(self, session: NRAMSession) -> NRAMSession:
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[NRAMSession]:
        return self._sessions.get(session_id)

    def update(self, session_id: str, updates: Dict[str, Any]) -> Optional[NRAMSession]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        for key, value in updates.items():
            if key == "phenomenon_weights" and isinstance(value, dict):
                session.phenomenon_weights.update(value)
            elif hasattr(session, key):
                setattr(session, key, value)
        session.revision += 1
        session.updated_at = datetime.now(timezone.utc)
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def reset(self, session_id: str) -> Optional[NRAMSession]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.reset()
        return session

    def list_all(self) -> List[NRAMSession]:
        return list(self._sessions.values())


# Global session store instance
session_store = SessionStore()
