"""
NRAM Phenomenon Modules — configurable cognitive effects.

Each phenomenon defines:
- stable identifier
- Czech and English label
- description
- default weight per consciousness state
- activation threshold
- steering mechanism (which NRAM layer implements it)
- allowed rhetorical phases
- safety limits
- telemetry event type

Phenomena are NOT decorative text. Each one maps to a real steering mechanism:
- Logit-native: implemented in the SGLang custom logit processor (pre-sampling)
- Prompt-level: implemented via developer instruction or rhetorical plan
- Post-process: implemented after generation (MUST be labeled as such)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SteeringLayer(str, Enum):
    """Which layer implements this phenomenon."""
    LOGIT_BIAS = "logit_bias"           # Pre-sampling logit modification
    LOGIT_MASK = "logit_mask"           # Pre-sampling hard mask
    REPETITION = "repetition"           # Pre-sampling repetition penalty
    PROMPT = "prompt"                   # Prompt-level instruction
    PLANNER = "planner"                 # Rhetorical planner constraint
    FRAGMENT_INJECTION = "fragment_injection"  # Deterministic text insertion (forced_injection)
    POSTPROCESS = "postprocess"         # After generation


class PhenomenonDefinition(BaseModel):
    """A single configurable NRAM phenomenon."""
    id: str
    label_en: str
    label_cs: str
    description: str

    steering_layer: SteeringLayer
    telemetry_event_type: str

    # Default weight per consciousness state (0.0 = disabled)
    default_weights: Dict[str, float] = Field(default_factory=dict)

    # Activation threshold — phenomenon only fires when weight > threshold
    activation_threshold: float = 0.05

    # Allowed rhetorical phases (empty = all phases)
    allowed_phases: List[str] = Field(default_factory=list)

    # Safety limits
    max_bias_delta: float = 1.2         # Max logit bias this phenomenon can add
    max_mask_tokens: int = 50           # Max tokens this phenomenon can mask
    max_fragment_length: int = 30       # Max tokens in a forced-injected fragment
    cooldown_tokens: int = 10           # Min tokens between activations

    # Provenance label for telemetry
    provenance_origin: str = "applied_policy"


# ---------------------------------------------------------------------------
# The 10 NRAM phenomena
# ---------------------------------------------------------------------------

PHENOMENA: List[PhenomenonDefinition] = [
    PhenomenonDefinition(
        id="overlap",
        label_en="Overlap",
        label_cs="Překryv",
        description="Fragments from previous responses bleed into the current generation, "
                    "creating thematic echoes across turns.",
        steering_layer=SteeringLayer.LOGIT_BIAS,
        telemetry_event_type="phenomenon_overlap",
        default_weights={
            "normal": 0.05, "microdose": 0.08, "threshold": 0.12,
            "psychedelic": 0.15, "peak": 0.20, "dissociative": 0.10,
        },
        provenance_origin="positive_logit_bias",
    ),
    PhenomenonDefinition(
        id="forgetting",
        label_en="Forgetting",
        label_cs="Zapomínání",
        description="Recent tokens receive increased repetition penalty, simulating "
                    "working memory decay.",
        steering_layer=SteeringLayer.REPETITION,
        telemetry_event_type="phenomenon_forgetting",
        default_weights={
            "normal": 0.02, "microdose": 0.03, "threshold": 0.05,
            "psychedelic": 0.08, "peak": 0.12, "dissociative": 0.20,
        },
        max_bias_delta=2.0,
        provenance_origin="negative_logit_bias",
    ),
    PhenomenonDefinition(
        id="looping",
        label_en="Thought Loop",
        label_cs="Myšlenková smyčka",
        description="Repetition penalty is reduced, allowing the model to revisit "
                    "recent tokens and create deliberate loops.",
        steering_layer=SteeringLayer.REPETITION,
        telemetry_event_type="phenomenon_looping",
        default_weights={
            "normal": 0.02, "microdose": 0.03, "threshold": 0.05,
            "psychedelic": 0.06, "peak": 0.08, "dissociative": 0.10,
        },
        max_bias_delta=1.0,
        provenance_origin="applied_policy",
    ),
    PhenomenonDefinition(
        id="associative_jump",
        label_en="Associative Jump",
        label_cs="Asociativní skok",
        description="Sampling distribution is flattened (effective temperature increase), "
                    "allowing the model to jump to semantically distant tokens.",
        steering_layer=SteeringLayer.LOGIT_BIAS,
        telemetry_event_type="phenomenon_associative_jump",
        default_weights={
            "normal": 0.05, "microdose": 0.10, "threshold": 0.18,
            "psychedelic": 0.25, "peak": 0.30, "dissociative": 0.15,
        },
        max_bias_delta=1.2,
        provenance_origin="applied_policy",
    ),
    PhenomenonDefinition(
        id="synesthesia",
        label_en="Synesthesia",
        label_cs="Synestézie",
        description="Cross-domain sensory lexemes receive positive bias, creating "
                    "metaphorical bridges between concepts and senses.",
        steering_layer=SteeringLayer.LOGIT_BIAS,
        telemetry_event_type="phenomenon_synesthesia",
        default_weights={
            "normal": 0.02, "microdose": 0.05, "threshold": 0.10,
            "psychedelic": 0.18, "peak": 0.22, "dissociative": 0.08,
        },
        allowed_phases=["revelation", "implication"],
        max_bias_delta=1.0,
        provenance_origin="positive_logit_bias",
    ),
    PhenomenonDefinition(
        id="dissolution",
        label_en="Dissolution",
        label_cs="Rozpuštění",
        description="Structural tokens (punctuation, connectors) receive negative bias, "
                    "creating fragmented, boundary-dissolving output.",
        steering_layer=SteeringLayer.LOGIT_MASK,
        telemetry_event_type="phenomenon_dissolution",
        default_weights={
            "normal": 0.01, "microdose": 0.02, "threshold": 0.05,
            "psychedelic": 0.10, "peak": 0.18, "dissociative": 0.25,
        },
        allowed_phases=["implication", "final_turn"],
        max_mask_tokens=20,
        provenance_origin="negative_logit_bias",
    ),
    PhenomenonDefinition(
        id="fragmentation",
        label_en="Fragmentation",
        label_cs="Fragmentace",
        description="Output is post-processed into shorter, disconnected fragments. "
                    "This is a POST-PROCESS effect, not pre-sampling steering.",
        steering_layer=SteeringLayer.POSTPROCESS,
        telemetry_event_type="phenomenon_fragmentation",
        default_weights={
            "normal": 0.02, "microdose": 0.03, "threshold": 0.05,
            "psychedelic": 0.08, "peak": 0.12, "dissociative": 0.20,
        },
        provenance_origin="postprocess",
    ),
    PhenomenonDefinition(
        id="echo",
        label_en="Echo",
        label_cs="Ozvěna",
        description="Recent token IDs receive positive bias, causing the model to "
                    "echo its own recent output.",
        steering_layer=SteeringLayer.LOGIT_BIAS,
        telemetry_event_type="phenomenon_echo",
        default_weights={
            "normal": 0.03, "microdose": 0.05, "threshold": 0.08,
            "psychedelic": 0.10, "peak": 0.12, "dissociative": 0.15,
        },
        max_bias_delta=0.8,
        provenance_origin="positive_logit_bias",
    ),
    PhenomenonDefinition(
        id="tangent",
        label_en="Tangent",
        label_cs="Tangenta",
        description="A forced-injection fragment redirects the topic. "
                    "This is FORCED_INJECTION, not model-generated.",
        steering_layer=SteeringLayer.FRAGMENT_INJECTION,
        telemetry_event_type="phenomenon_tangent",
        default_weights={
            "normal": 0.02, "microdose": 0.04, "threshold": 0.06,
            "psychedelic": 0.08, "peak": 0.10, "dissociative": 0.12,
        },
        allowed_phases=["contradiction", "implication"],
        max_fragment_length=20,
        cooldown_tokens=50,
        provenance_origin="forced_injection",
    ),
    PhenomenonDefinition(
        id="insight",
        label_en="Insight",
        label_cs="Vhled",
        description="A forced-injection fragment marks a moment of sudden connection. "
                    "This is FORCED_INJECTION, not model-generated.",
        steering_layer=SteeringLayer.FRAGMENT_INJECTION,
        telemetry_event_type="phenomenon_insight",
        default_weights={
            "normal": 0.03, "microdose": 0.06, "threshold": 0.08,
            "psychedelic": 0.10, "peak": 0.12, "dissociative": 0.05,
        },
        allowed_phases=["revelation"],
        max_fragment_length=15,
        cooldown_tokens=80,
        provenance_origin="forced_injection",
    ),
]

# Index by ID for fast lookup
PHENOMENA_BY_ID: Dict[str, PhenomenonDefinition] = {p.id: p for p in PHENOMENA}


def get_phenomena_for_state(state: str) -> List[PhenomenonDefinition]:
    """Get phenomena with non-zero default weight for a consciousness state."""
    return [p for p in PHENOMENA if p.default_weights.get(state, 0.0) > 0]


def compile_phenomenon_config(
    state: str,
    intensity: float,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compile phenomenon configuration for a session.

    Returns a dict mapping phenomenon ID to its effective configuration
    (weight scaled by intensity, thresholds, safety limits).
    """
    config = {}
    weights = custom_weights or {}

    for p in PHENOMENA:
        base_weight = weights.get(p.id, p.default_weights.get(state, 0.0))
        effective_weight = base_weight * intensity

        if effective_weight < p.activation_threshold:
            continue

        config[p.id] = {
            "id": p.id,
            "label_en": p.label_en,
            "label_cs": p.label_cs,
            "steering_layer": p.steering_layer.value,
            "telemetry_event_type": p.telemetry_event_type,
            "effective_weight": round(effective_weight, 4),
            "activation_threshold": p.activation_threshold,
            "allowed_phases": p.allowed_phases,
            "max_bias_delta": p.max_bias_delta,
            "max_mask_tokens": p.max_mask_tokens,
            "max_fragment_length": p.max_fragment_length,
            "cooldown_tokens": p.cooldown_tokens,
            "provenance_origin": p.provenance_origin,
        }

    return config
