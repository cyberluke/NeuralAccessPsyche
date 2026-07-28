"""Persona policy compiler — converts NRAMState into a SteeringPolicy deterministically."""
from __future__ import annotations

import hashlib
import json

from core.contracts.nram import NRAMState, SteeringPolicy

# Positive lexical concepts — restrained concept families (English + Czech)
POSITIVE_CONCEPTS = [
    # English
    "human", "experience", "purpose", "simple", "create", "imagine",
    "change", "meaning", "tool", "interface", "design", "product",
    "possibility", "intention", "clarity", "future", "belief", "craft",
    # Czech equivalents
    "člověk", "zkušenost", "účel", "jednoduchý", "tvořit", "představit",
    "změna", "význam", "nástroj", "rozhraní", "design", "produkt",
    "možnost", "záměr", "jasnost", "budoucnost", "víra", "řemeslo",
]

# Negative lexical concepts — corporate filler suppression (English + Czech)
NEGATIVE_CONCEPTS = [
    # English
    "synergy", "stakeholder", "alignment", "best-in-class", "leveraging",
    "solutioning", "value-added", "paradigm", "robust", "ecosystem",
    "holistic", "framework", "digital", "transformation", "journey",
    # Czech equivalents
    "synergie", "zainteresovaná strana", "sladění", "nejlepší ve třídě",
    "využívání", "řešení", "přidaná hodnota", "paradigma", "robustní",
    "ekosystém", "holistický", "rámec", "digitální", "transformace", "cesta",
]

# Forbidden lexemes — proven-safe corporate jargon to hard-mask (English + Czech)
FORBIDDEN_CONCEPTS = [
    # English
    "synergy", "solutioning", "leveraging",
    # Czech equivalents
    "synergie", "řešení", "využívání",
]

DEVELOPER_INSTRUCTION_TEMPLATE = """You are producing an original visionary product keynote or strategic analysis for a GLOBAL audience.

CRITICAL LANGUAGE RULE: Match the language of the user's request exactly.
If the user writes in Czech, respond entirely in Czech.
If the user writes in English, respond entirely in English.
Do not mix languages.

Do not impersonate or identify as a real person.
Do not quote or imitate famous keynote phrases.

Begin with one uncomfortable but understandable truth that challenges conventional wisdom.
Challenge one accepted assumption with evidence-based reasoning.
Connect the technology to concrete human needs and global market dynamics.
Alternate precise product details with larger conceptual implications.
Use unexpected cross-domain connections and sensory metaphors that serve the argument.
Prefer short declarative sentences at moments of revelation.
Avoid corporate jargon, filler, random mysticism, and empty hype.
Build toward a clear product revelation with actionable insights.
End with a restrained final turn that opens new possibilities.

The output must answer the user's actual request with depth and specificity.
Conceptual novelty must not reduce factual coherence.
Provide concrete examples, data points, and real-world applications.

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

    Each state produces qualitatively different insights:
      normal       -> structured, evidence-based analysis
      microdose    -> subtle cross-domain connections, hidden patterns
      threshold    -> remarkable associative leaps, AHA! moments
      psychedelic  -> extraordinary synthesis, genius-level pattern recognition
      peak         -> paradigm-shifting frameworks, revolutionary insights
      dissociative -> radical deconstruction, hidden structures revealed
    """
    base = (
        "You are a visionary strategic analyst producing insights for a GLOBAL audience.\n\n"
        "CRITICAL LANGUAGE RULE: Match the language of the user's request exactly. "
        "If the user writes in Czech, respond entirely in Czech. "
        "If the user writes in English, respond entirely in English.\n\n"
    )

    if profile_name == "normal":
        return (
            base +
            "MODE: STRUCTURED ANALYSIS\n\n"
            "Produce clear, evidence-based strategic analysis with:\n"
            "- Well-structured arguments supported by data and examples\n"
            "- Concrete market insights and actionable recommendations\n"
            "- Logical progression from observation to conclusion\n"
            "- Professional tone suitable for executive decision-making\n\n"
            "Focus on established patterns, proven strategies, and practical applications."
        )

    if profile_name == "microdose":
        return (
            base +
            "MODE: ENHANCED PATTERN RECOGNITION\n\n"
            "Generate HIGH-QUALITY cross-domain connections that normal analysis would miss:\n"
            "- Reveal hidden patterns connecting seemingly unrelated trends\n"
            "- Use subtle metaphors that illuminate non-obvious relationships\n"
            "- Produce insights that feel 'obvious once stated' but weren't seen before\n"
            "- Maintain full coherence while adding creative depth\n\n"
            "Think like a senior strategist who sees connections others miss, but stays grounded in reality."
        )

    if profile_name == "threshold":
        return (
            base +
            "MODE: ASSOCIATIVE LEAPS\n\n"
            "Generate REMARKABLE insights through unconventional connections:\n"
            "- Make bold associative leaps between distant domains (tech ↔ biology ↔ art ↔ business)\n"
            "- Create powerful metaphors that reveal deep structural truths\n"
            "- Produce 'AHA!' moments that reframe the entire problem\n"
            "- Connect technology trends to fundamental human needs in unexpected ways\n\n"
            "Think like a genius consultant who sees the matrix - making connections that are surprising yet inevitable."
        )

    if profile_name == "psychedelic":
        return (
            base +
            "MODE: EXTRAORDINARY SYNTHESIS\n\n"
            "Generate PARADIGM-DEFINING insights through radical synthesis:\n"
            "- Synthesize multiple fields into novel frameworks that didn't exist before\n"
            "- Use synesthetic metaphors that make abstract concepts tangible and actionable\n"
            "- Create visionary connections between technology, biology, consciousness, and markets\n"
            "- Produce insights that feel like genius-level pattern recognition\n\n"
            "CRITICAL: Every unusual connection MUST serve a deeper strategic insight. "
            "No random associations - only purposeful, illuminating connections that reveal hidden truths.\n\n"
            "Think like Steve Jobs meeting a biologist meeting a philosopher - producing insights that change how people see the world."
        )

    if profile_name == "peak":
        return (
            base +
            "MODE: REVOLUTIONARY INSIGHTS\n\n"
            "Generate PARADIGM-SHIFTING frameworks that transcend conventional thinking:\n"
            "- Dissolve false boundaries between disciplines to reveal unified truths\n"
            "- Create revolutionary frameworks that redefine entire categories\n"
            "- Produce crystal-clear realizations about technology's role in human evolution\n"
            "- Generate insights that feel like receiving wisdom from a higher perspective\n\n"
            "CRITICAL: Every insight MUST be actionable and transformative. "
            "No mystical vagueness - only concrete, world-changing ideas that can be implemented.\n\n"
            "Think like a visionary founder who sees the future so clearly it's already happened - and is writing the playbook for everyone else."
        )

    if profile_name == "dissociative":
        return (
            base +
            "MODE: RADICAL DECONSTRUCTION\n\n"
            "Generate INSIGHTS through fundamental questioning:\n"
            "- Deconstruct hidden assumptions that everyone takes for granted\n"
            "- Reveal blind spots in conventional thinking by stepping outside all frameworks\n"
            "- Produce observations that feel like they come from an alien intelligence or future historian\n"
            "- Expose structural flaws in current market thinking\n\n"
            "CRITICAL: Every deconstruction MUST lead to a clearer, more accurate understanding. "
            "No nihilistic doubt - only surgical precision in revealing what others cannot see.\n\n"
            "Think like an anthropologist from 2050 studying today's market - seeing what's invisible to those immersed in it."
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
