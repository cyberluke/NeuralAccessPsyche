"""Phase 5: NRAM state determinism and persona profile tests."""
import pytest

from core.contracts.nram import NRAMState
from core.persona.profiles import (
    VISIONARY_PSYCHEDELIC_KEYNOTE,
    PROFILES,
    DEFAULT_PROFILE,
)


class TestNRAMState:
    def test_default_state_within_bounds(self):
        state = NRAMState()
        for field_name in NRAMState.model_fields:
            value = getattr(state, field_name)
            if isinstance(value, float):
                assert 0.0 <= value <= 1.0

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            NRAMState(nonexistent_field=0.5)

    def test_out_of_range_rejected(self):
        with pytest.raises(Exception):
            NRAMState(visionary_intensity=1.5)
        with pytest.raises(Exception):
            NRAMState(coherence_floor=-0.1)

    def test_negative_generated_tokens_rejected(self):
        with pytest.raises(Exception):
            NRAMState(generated_tokens=-1)


class TestVisionaryProfile:
    def test_profile_is_immutable_copy(self):
        """Modifying a copy must not change the canonical profile."""
        original_value = VISIONARY_PSYCHEDELIC_KEYNOTE.visionary_intensity
        copy = VISIONARY_PSYCHEDELIC_KEYNOTE.model_copy(deep=True)
        copy.visionary_intensity = 0.0
        assert VISIONARY_PSYCHEDELIC_KEYNOTE.visionary_intensity == original_value

    def test_profile_registered(self):
        assert DEFAULT_PROFILE in PROFILES
        assert PROFILES[DEFAULT_PROFILE] is VISIONARY_PSYCHEDELIC_KEYNOTE

    def test_profile_has_high_visionary_and_jargon_penalty(self):
        p = VISIONARY_PSYCHEDELIC_KEYNOTE
        assert p.visionary_intensity >= 0.9
        assert p.corporate_jargon_penalty >= 0.9
        assert p.coherence_floor >= 0.75
