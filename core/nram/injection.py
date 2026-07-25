"""
NRAM Token Injection Engine — 4 levels of controlled intervention.

Level 1: Soft injection (logit bias) — probabilistic, model still decides
Level 2: Hard injection (forced token) — deterministic, only one token possible
Level 3: Fragment injection — deterministic text insertion (forced_injection)
Level 4: Hidden control injection — side-channel metadata (planner_constraint)

Each level has distinct telemetry provenance:
- Level 1 → applied_policy / positive_logit_bias / negative_logit_bias
- Level 2 → hard_mask (forced token proof)
- Level 3 → forced_injection
- Level 4 → planner_constraint
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Fragment templates for forced injection (Level 3)
# ---------------------------------------------------------------------------

FRAGMENT_TEMPLATES: Dict[str, List[str]] = {
    "associative_jump": [
        "But what if the entire premise is inverted?",
        "This reminds me of a completely different domain:",
        "Wait — there's a parallel in {domain}:",
        "Shift perspective: what would a {role} see here?",
    ],
    "insight": [
        "Now it clicks:",
        "The connection is:",
        "Aha — the real pattern is:",
        "This is where it converges:",
    ],
    "tangent": [
        "Brief detour:",
        "Side observation:",
        "Unrelated but relevant:",
        "A thought from the periphery:",
    ],
    "echo": [
        "As noted before,",
        "Returning to the earlier point,",
        "This echoes what was said:",
    ],
    "forgetting": [
        "Wait, I lost the thread...",
        "Where was I...",
        "The point was... something about",
    ],
}

# Domain pool for associative jumps
DOMAIN_POOL = [
    "control theory", "compiler design", "music composition",
    "operating systems", "scientific instrumentation", "game AI",
    "cognitive architecture", "thermodynamics", "ecology",
    "cryptography", "urban planning", "chess strategy",
    "quantum mechanics", "mythology", "cooking",
]

ROLE_POOL = [
    "child", "skeptic", "poet", "engineer", "historian",
    "musician", "surgeon", "astronaut", "farmer", "monk",
]


class InjectionEvent(BaseModel):
    """Records a single injection event for telemetry."""
    level: int  # 1-4
    phenomenon: str
    origin: str  # TokenOrigin value
    text: Optional[str] = None
    token_ids: List[int] = Field(default_factory=list)
    bias_delta: Optional[float] = None
    position: int = 0
    rhetorical_phase: str = "idle"


class InjectionEngine:
    """Manages token injection across 4 levels.

    This engine is called by the NRAM logit processor (levels 1-2) and
    by the streaming layer (level 3) and the policy compiler (level 4).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._events: List[InjectionEvent] = []
        self._cooldowns: Dict[str, int] = {}  # phenomenon -> last_position
        self._position: int = 0

    def tick(self) -> None:
        """Advance position counter."""
        self._position += 1

    @property
    def events(self) -> List[InjectionEvent]:
        return self._events

    def record_event(self, event: InjectionEvent) -> None:
        event.position = self._position
        self._events.append(event)

    # -----------------------------------------------------------------------
    # Level 1: Soft injection (logit bias)
    # -----------------------------------------------------------------------

    def compute_soft_biases(
        self,
        positive_ids: List[int],
        negative_ids: List[int],
        positive_bias: float,
        negative_bias: float,
        phenomenon_weights: Dict[str, float],
        rhetorical_phase: str,
    ) -> Tuple[Dict[int, float], List[InjectionEvent]]:
        """Compute per-token bias deltas for Level 1 soft injection.

        Returns a dict mapping token_id -> bias_delta (positive or negative).
        """
        biases: Dict[int, float] = {}
        events: List[InjectionEvent] = []

        # Base positive bias
        if positive_bias > 0:
            for tid in positive_ids:
                biases[tid] = biases.get(tid, 0.0) + positive_bias

        # Base negative bias
        if negative_bias > 0:
            for tid in negative_ids:
                biases[tid] = biases.get(tid, 0.0) - negative_bias

        # Phenomenon-modulated biases
        # Overlap: boost recent token IDs (handled by processor via __req__)
        # Echo: boost specific recent tokens
        echo_weight = phenomenon_weights.get("echo", 0.0)
        if echo_weight > 0.05:
            # Echo is handled by the processor's repetition mechanism
            pass

        # Associative jump: flatten distribution (reduce all biases slightly)
        aj_weight = phenomenon_weights.get("associative_jump", 0.0)
        if aj_weight > 0.1:
            # Scale down all biases to flatten the distribution
            flatten_factor = 1.0 - (aj_weight * 0.3)
            for tid in biases:
                biases[tid] *= flatten_factor

        # Record events
        if positive_bias > 0 and positive_ids:
            events.append(InjectionEvent(
                level=1,
                phenomenon="soft_bias",
                origin="positive_logit_bias",
                token_ids=positive_ids[:10],
                bias_delta=positive_bias,
                rhetorical_phase=rhetorical_phase,
            ))

        if negative_bias > 0 and negative_ids:
            events.append(InjectionEvent(
                level=1,
                phenomenon="soft_bias",
                origin="negative_logit_bias",
                token_ids=negative_ids[:10],
                bias_delta=-negative_bias,
                rhetorical_phase=rhetorical_phase,
            ))

        for e in events:
            self.record_event(e)

        return biases, events

    # -----------------------------------------------------------------------
    # Level 2: Hard injection (forced token)
    # -----------------------------------------------------------------------

    def compute_forced_token(
        self,
        forced_token_id: int,
        vocab_size: int,
        phenomenon: str,
        rhetorical_phase: str,
    ) -> Tuple[int, InjectionEvent]:
        """Force a specific token (Level 2). Returns the forced token ID.

        The processor will mask all other tokens to -inf.
        """
        event = InjectionEvent(
            level=2,
            phenomenon=phenomenon,
            origin="hard_mask",
            token_ids=[forced_token_id],
            rhetorical_phase=rhetorical_phase,
        )
        self.record_event(event)
        return forced_token_id, event

    # -----------------------------------------------------------------------
    # Level 3: Fragment injection (forced_injection)
    # -----------------------------------------------------------------------

    def should_inject_fragment(
        self,
        phenomenon: str,
        weight: float,
        threshold: float,
        cooldown_tokens: int,
        rhetorical_phase: str,
        allowed_phases: List[str],
    ) -> bool:
        """Determine if a fragment should be injected at this position."""
        if weight < threshold:
            return False

        # Check rhetorical phase
        if allowed_phases and rhetorical_phase not in allowed_phases:
            return False

        # Check cooldown
        last_pos = self._cooldowns.get(phenomenon, -999)
        if self._position - last_pos < cooldown_tokens:
            return False

        # Probabilistic activation based on weight
        return self._rng.random() < weight

    def generate_fragment(
        self,
        phenomenon: str,
        max_length: int,
        rhetorical_phase: str,
    ) -> Tuple[str, InjectionEvent]:
        """Generate a forced-injection fragment (Level 3).

        This text is NOT model-generated. It is deterministically inserted.
        Telemetry MUST label it as forced_injection.
        """
        templates = FRAGMENT_TEMPLATES.get(phenomenon, ["..."])
        template = self._rng.choice(templates)

        # Fill template variables
        if "{domain}" in template:
            template = template.replace("{domain}", self._rng.choice(DOMAIN_POOL))
        if "{role}" in template:
            template = template.replace("{role}", self._rng.choice(ROLE_POOL))

        # Truncate to max_length tokens (approximate: 1 token ≈ 4 chars)
        max_chars = max_length * 4
        if len(template) > max_chars:
            template = template[:max_chars]

        event = InjectionEvent(
            level=3,
            phenomenon=phenomenon,
            origin="forced_injection",
            text=template,
            rhetorical_phase=rhetorical_phase,
        )
        self.record_event(event)

        # Update cooldown
        self._cooldowns[phenomenon] = self._position

        return template, event

    # -----------------------------------------------------------------------
    # Level 4: Hidden control injection (side-channel metadata)
    # -----------------------------------------------------------------------

    def compile_hidden_control(
        self,
        rhetorical_phase: str,
        nram_profile: str,
        memory_overlay: Optional[str] = None,
        grammar_schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compile hidden control metadata (Level 4).

        This is sent as side-channel metadata, NOT as text tokens.
        The model never sees these as input tokens.
        """
        control = {
            "phase": rhetorical_phase,
            "nram_profile": nram_profile,
            "memory_overlay": memory_overlay or "disabled",
            "grammar": grammar_schema,
        }

        event = InjectionEvent(
            level=4,
            phenomenon="hidden_control",
            origin="planner_constraint",
            rhetorical_phase=rhetorical_phase,
        )
        self.record_event(event)

        return control

    # -----------------------------------------------------------------------
    # Repetition modulation (phenomenon-driven)
    # -----------------------------------------------------------------------

    def compute_repetition_modulation(
        self,
        base_penalty: float,
        phenomenon_weights: Dict[str, float],
    ) -> float:
        """Modulate repetition penalty based on phenomena.

        - forgetting: INCREASE penalty (forget recent tokens)
        - looping: DECREASE penalty (encourage repeats)
        """
        forgetting = phenomenon_weights.get("forgetting", 0.0)
        looping = phenomenon_weights.get("looping", 0.0)

        # Forgetting increases penalty
        penalty = base_penalty * (1.0 + forgetting * 2.0)

        # Looping decreases penalty
        penalty = penalty * (1.0 - looping * 0.8)

        return max(0.0, min(2.0, penalty))
