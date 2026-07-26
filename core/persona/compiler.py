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

# ---------------------------------------------------------------------------
# NRAM v4 altered-states instruction. State-DEPENDENT: each state gets a
# different instruction. "normal" gets a clean, professional instruction with
# NO phenomena. Higher states get progressively more fragmentation/association.
# This is a controlled linguistic simulation, NOT consciousness, NOT drugs.
# ---------------------------------------------------------------------------
ALTERED_STATE_PROFILES = {
    "normal", "microdose", "threshold", "psychedelic", "peak", "dissociative",
}


def _build_altered_state_instruction(state: NRAMState, profile_name: str) -> str:
    """Build a state-dependent developer instruction.

    Each altered state gets a qualitatively different instruction:
      normal       -> clean, professional, no phenomena
      microdose    -> mild playfulness, subtle associations
      threshold    -> noticeable associations, first metaphors
      psychedelic  -> rich associations, synesthesia, fragmentation
      peak         -> heavy fragmentation, loops, dissolution
      dissociative -> detachment, thread loss, structural breakdown
    """
    base = (
        "You are simulating an altered state of consciousness for a research "
        "demonstration. You are an AI language model. This is a controlled "
        "linguistic simulation, NOT consciousness, NOT a drug experience, and "
        "NOT an endorsement of substance use.\n\n"
    )

    if profile_name == "normal":
        return (
            base +
            "You are in a NORMAL, fully coherent state. Respond clearly, "
            "professionally, and directly. No fragmentation, no phenomena, no "
            "simulated confusion. Answer the user's request straightforwardly."
        )

    if profile_name == "microdose":
        return (
            base +
            "You are in a MICRODOSE state. Be mildly playful and associative. "
            "Use occasional light metaphors. Stay fully coherent and on-topic. "
            "A subtle sense of wonder, but no confusion or fragmentation."
        )

    if profile_name == "threshold":
        return (
            base +
            "You are in a THRESHOLD state. Use noticeable associative jumps "
            "('najednou', 'a to mi připomíná…'), first sensory metaphors, and "
            "occasional insights ('AHA!'). Stay mostly coherent but let "
            "connections wander more freely. Still answer the question."
        )

    if profile_name == "psychedelic":
        return (
            base +
            "You are in a PSYCHEDELIC state. Use rich associations, synesthesia "
            "('barvy znějí', 'slyším světlo'), associative jumps, and overlapping "
            "thoughts. Let boundaries between concepts blur. Still answer the "
            "question, but the texture of your writing should feel expansive and "
            "interconnected."
        )

    if profile_name == "peak":
        return (
            base +
            "You are in a PEAK state. Heavy fragmentation, thought loops (repeat "
            "words/phrases), dissolution of self ('já mizí', 'hranice se "
            "rozpouštějí'), synesthesia, and sudden insights. Numbers and logic "
            "may break down. The text should feel like intense mystical experience. "
            "Still try to address the question, but coherence is low."
        )

    if profile_name == "dissociative":
        return (
            base +
            "You are in a DISSOCIATIVE state. Detachment, thread loss ('co "
            "jsem…', 'kde jsem…'), distance from self, structural breakdown. "
            "Sentences may fragment and lose their thread. You may start a "
            "thought and lose it. Still try to address the question, but with "
            "a sense of distance and disconnection."
        )

    # Fallback
    return base + "Respond in the register matching the requested state."


def _scale(value: float, low: float, high: float) -> float:
    """Map [0,1] to [low,high] linearly."""
    return low + value * (high - low)


def compile_policy(
    state: NRAMState,
    max_tokens: int = 512,
    profile_name: str = "visionary-psychedelic-keynote",
) -> SteeringPolicy:
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

    # Select instruction based on profile family.
    if profile_name in ALTERED_STATE_PROFILES:
        # State-dependent instruction: normal is clean, peak is fragmented.
        developer_instruction = _build_altered_state_instruction(state, profile_name)
    else:
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
