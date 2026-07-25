"""Persona policy compiler — converts NRAMState into a SteeringPolicy deterministically."""
from __future__ import annotations

import hashlib
import json

from core.contracts.nram import NRAMState, SteeringPolicy

# Positive lexical concepts — restrained concept families
POSITIVE_CONCEPTS = [
    "human", "experience", "purpose", "simple", "create", "imagine",
    "change", "meaning", "tool", "interface", "design", "product",
    "possibility", "intention", "clarity", "future", "belief", "craft",
]

# Negative lexical concepts — corporate filler suppression
NEGATIVE_CONCEPTS = [
    "synergy", "stakeholder", "alignment", "best-in-class", "leveraging",
    "solutioning", "value-added", "paradigm", "robust", "ecosystem",
    "holistic", "framework", "digital", "transformation", "journey",
]

# Forbidden lexemes — proven-safe corporate jargon to hard-mask
FORBIDDEN_CONCEPTS = [
    "synergy", "solutioning", "leveraging",
]

DEVELOPER_INSTRUCTION_TEMPLATE = """You are producing an original visionary product keynote.

Do not impersonate or identify as a real person.
Do not quote or imitate famous keynote phrases.

Begin with one uncomfortable but understandable truth.
Challenge one accepted assumption.
Connect the technology to a concrete human need.
Alternate precise product details with larger conceptual implications.
Use one unexpected sensory metaphor that serves the argument.
Prefer short declarative sentences at moments of revelation.
Avoid corporate jargon, filler, random mysticism, and empty hype.
Build toward a clear product revelation.
End with a restrained final turn.

The output must answer the user's actual request.
Conceptual novelty must not reduce factual coherence.

NRAM internal dimensions (do not repeat these numbers):
{dimensions}"""


def _scale(value: float, low: float, high: float) -> float:
    """Map [0,1] to [low,high] linearly."""
    return low + value * (high - low)


def compile_policy(state: NRAMState, max_tokens: int = 512) -> SteeringPolicy:
    """Deterministically compile NRAMState into a SteeringPolicy.

    Given identical (input, seed, persona options, model, sampling parameters),
    this produces identical output parameters.
    """
    # Positive bias: 0.0 to 1.2, driven by visionary_intensity and human_focus
    positive_bias = _scale(
        (state.visionary_intensity + state.human_focus) / 2.0, 0.0, 1.2
    )

    # Negative bias: 0.0 to 2.5, driven by corporate_jargon_penalty
    negative_bias = _scale(state.corporate_jargon_penalty, 0.0, 2.5)

    # Repetition penalty: 0.0 to 2.0
    repetition_penalty = _scale(state.repetition_penalty, 0.0, 2.0)

    # Build dimensions block for the developer instruction
    dimensions = {
        "visionary_intensity": state.visionary_intensity,
        "contrarian_force": state.contrarian_force,
        "product_obsession": state.product_obsession,
        "human_focus": state.human_focus,
        "rhetorical_compression": state.rhetorical_compression,
        "associative_distance": state.associative_distance,
        "theatricality": state.theatricality,
        "emotional_voltage": state.emotional_voltage,
        "coherence_floor": state.coherence_floor,
        "novelty_target": state.novelty_target,
    }
    dimensions_str = json.dumps(dimensions, indent=2)

    developer_instruction = DEVELOPER_INSTRUCTION_TEMPLATE.format(
        dimensions=dimensions_str
    )

    return SteeringPolicy(
        positive_lexemes=list(POSITIVE_CONCEPTS),
        negative_lexemes=list(NEGATIVE_CONCEPTS),
        forbidden_lexemes=list(FORBIDDEN_CONCEPTS),
        positive_bias=round(positive_bias, 4),
        negative_bias=round(negative_bias, 4),
        repetition_penalty=round(repetition_penalty, 4),
        min_coherence=state.coherence_floor,
        rhetorical_phase=state.rhetorical_phase,
        max_tokens=max_tokens,
        developer_instruction=developer_instruction,
    )


def policy_hash(policy: SteeringPolicy) -> str:
    """Deterministic hash of a policy for reproducibility verification."""
    blob = policy.model_dump_json()
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
