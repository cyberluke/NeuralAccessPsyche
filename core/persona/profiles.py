"""Immutable persona profiles for NRAM steering."""
from __future__ import annotations

from core.contracts.nram import NRAMState

# The first persona profile: visionary-psychedelic-keynote
# Reproduces cognitive and rhetorical properties, not a public figure cosplay.
VISIONARY_PSYCHEDELIC_KEYNOTE = NRAMState(
    visionary_intensity=0.95,
    contrarian_force=0.86,
    product_obsession=0.96,
    human_focus=0.94,
    rhetorical_compression=0.82,
    associative_distance=0.72,
    theatricality=0.84,
    emotional_voltage=0.78,
    coherence_floor=0.80,
    novelty_target=0.76,
    repetition_penalty=0.55,
    corporate_jargon_penalty=0.92,
)

PROFILES: dict[str, NRAMState] = {
    "visionary-psychedelic-keynote": VISIONARY_PSYCHEDELIC_KEYNOTE,
}

DEFAULT_PROFILE = "visionary-psychedelic-keynote"
