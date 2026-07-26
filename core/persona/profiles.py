"""Immutable persona profiles for NRAM steering.

Two families of profiles:
  1. The original "visionary-psychedelic-keynote" persona (Steve-Jobs-style
     visionary rhetoric WITHOUT impersonation).
  2. The NRAM v4 "altered states of consciousness" simulator profiles
     (normal -> dissociative). These are SIMULATIONS of cognitive/phenomenological
     patterns, not measurements of consciousness and not drug endorsement.
"""
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

# ---------------------------------------------------------------------------
# NRAM v4 altered-states profiles (Normální -> Disociativní)
# These SIMULATE phenomenological/cognitive patterns. They are control
# parameters, not consciousness measurements and not drug instructions.
# ---------------------------------------------------------------------------

NORMAL = NRAMState(
    visionary_intensity=0.12, contrarian_force=0.10, product_obsession=0.30,
    human_focus=0.50, rhetorical_compression=0.40, associative_distance=0.05,
    theatricality=0.08, emotional_voltage=0.15, coherence_floor=0.96,
    novelty_target=0.10, repetition_penalty=0.30, corporate_jargon_penalty=0.30,
)

MICRODOSE = NRAMState(
    visionary_intensity=0.35, contrarian_force=0.25, product_obsession=0.40,
    human_focus=0.60, rhetorical_compression=0.45, associative_distance=0.28,
    theatricality=0.25, emotional_voltage=0.35, coherence_floor=0.90,
    novelty_target=0.35, repetition_penalty=0.35, corporate_jargon_penalty=0.35,
)

THRESHOLD = NRAMState(
    visionary_intensity=0.55, contrarian_force=0.45, product_obsession=0.50,
    human_focus=0.70, rhetorical_compression=0.50, associative_distance=0.48,
    theatricality=0.45, emotional_voltage=0.50, coherence_floor=0.84,
    novelty_target=0.55, repetition_penalty=0.45, corporate_jargon_penalty=0.40,
)

PSYCHEDELIC = NRAMState(
    visionary_intensity=0.82, contrarian_force=0.70, product_obsession=0.60,
    human_focus=0.80, rhetorical_compression=0.55, associative_distance=0.74,
    theatricality=0.70, emotional_voltage=0.72, coherence_floor=0.72,
    novelty_target=0.78, repetition_penalty=0.55, corporate_jargon_penalty=0.45,
)

PEAK = NRAMState(
    visionary_intensity=0.96, contrarian_force=0.85, product_obsession=0.65,
    human_focus=0.85, rhetorical_compression=0.60, associative_distance=0.92,
    theatricality=0.88, emotional_voltage=0.90, coherence_floor=0.58,
    novelty_target=0.92, repetition_penalty=0.85, corporate_jargon_penalty=0.50,
)

DISSOCIATIVE = NRAMState(
    visionary_intensity=0.70, contrarian_force=0.55, product_obsession=0.45,
    human_focus=0.55, rhetorical_compression=0.50, associative_distance=0.62,
    theatricality=0.55, emotional_voltage=0.45, coherence_floor=0.52,
    novelty_target=0.65, repetition_penalty=0.95, corporate_jargon_penalty=0.40,
)

PROFILES: dict[str, NRAMState] = {
    "visionary-psychedelic-keynote": VISIONARY_PSYCHEDELIC_KEYNOTE,
    # NRAM v4 altered-states profiles
    "normal": NORMAL,
    "microdose": MICRODOSE,
    "threshold": THRESHOLD,
    "psychedelic": PSYCHEDELIC,
    "peak": PEAK,
    "dissociative": DISSOCIATIVE,
}

DEFAULT_PROFILE = "visionary-psychedelic-keynote"
