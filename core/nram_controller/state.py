"""NRAM state management — per-request isolated state."""
from __future__ import annotations

from core.contracts.nram import NRAMState


def create_request_state(profile: NRAMState) -> NRAMState:
    """Create an isolated copy of the profile state for a single request.

    All request-specific state must be isolated. Never mutate global state.
    """
    return profile.model_copy(deep=True)


def phase_for_progress(generated: int, max_tokens: int) -> str:
    """Determine rhetorical phase based on generation progress."""
    if max_tokens <= 0:
        return "opening"

    progress = generated / max_tokens

    if progress < 0.15:
        return "opening"
    elif progress < 0.40:
        return "contradiction"
    elif progress < 0.70:
        return "revelation"
    elif progress < 0.90:
        return "implication"
    else:
        return "final_turn"
