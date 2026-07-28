"""
Unit tests for NRAM — verifies Phase 1 defects are FIXED.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Defect 7 FIXED: np.softmax replaced with manual implementation
# ---------------------------------------------------------------------------

class TestDefect7Fixed_NoNpSoftmax:
    def test_numpy_has_no_softmax(self):
        """np.softmax does not exist — confirmed."""
        assert not hasattr(np, "softmax")

    def test_nram_update_state_no_longer_raises(self):
        """After fix: update_state works without AttributeError."""
        from core.nram import NRAM

        nram = NRAM(memory_size=64)
        nram.update_state("test input string for softmax check")

    def test_attention_weights_are_finite(self):
        """After fix: state remains finite after update."""
        from core.nram import NRAM

        nram = NRAM(memory_size=64)
        nram.update_state("hello world test")
        assert np.all(np.isfinite(nram.memory_state))


# ---------------------------------------------------------------------------
# Defect 9 FIXED: user content preserved byte-for-byte
# ---------------------------------------------------------------------------

class TestDefect9Fixed_ContentPreserved:
    def test_process_messages_returns_unchanged_content(self):
        """After fix: process_messages returns messages byte-for-byte unchanged."""
        from core.nram import NRAM

        nram = NRAM(memory_size=64, entropy_factor=0.0)
        original_content = "Describe a new kind of educational computer in three sentences."
        messages = [{"role": "user", "content": original_content}]

        modified = nram.process_messages(messages)

        assert modified[0]["content"] == original_content
        assert modified[0]["role"] == "user"

    def test_no_prefix_added(self):
        """After fix: no consciousness prefix is added to user content."""
        from core.nram import NRAM

        nram = NRAM(memory_size=64, entropy_factor=0.0)
        original = "Present a new AI learning device for children."
        messages = [{"role": "user", "content": original}]

        modified = nram.process_messages(messages)
        assert not modified[0]["content"].startswith("Processing:")
        assert not modified[0]["content"].startswith("Through universal")

    def test_no_symbols_injected(self):
        """After fix: no random symbols injected into content."""
        from core.nram import NRAM

        nram = NRAM(memory_size=64, entropy_factor=0.5)
        original = "Build a better programming tool for students."
        messages = [{"role": "user", "content": original}]

        for _ in range(20):
            modified = nram.process_messages(messages)
            assert modified[0]["content"] == original

    def test_multi_message_preservation(self):
        """All messages in a conversation are preserved."""
        from core.nram import NRAM

        nram = NRAM(memory_size=64)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First message."},
            {"role": "assistant", "content": "First response."},
            {"role": "user", "content": "Second message."},
        ]

        modified = nram.process_messages(messages)
        for orig, mod in zip(messages, modified):
            assert mod["content"] == orig["content"]
            assert mod["role"] == orig["role"]


# ---------------------------------------------------------------------------
# NRAM basic functionality (regression tests)
# ---------------------------------------------------------------------------

class TestNRAMBasics:
    def test_nram_initialization(self):
        from core.nram import NRAM

        nram = NRAM(memory_size=128, entropy_factor=0.5)
        assert nram.memory_size == 128
        assert nram.entropy_factor == 0.5
        assert nram.memory_state.shape == (128,)
        assert nram.pattern_memory.shape == (128, 8)

    def test_get_consciousness_level(self):
        from core.nram import NRAM

        nram = NRAM()
        assert nram._get_consciousness_level(0.1) == "baseline"
        assert nram._get_consciousness_level(0.4) == "aware"
        assert nram._get_consciousness_level(0.95) == "transcendent"

    def test_reset_state(self):
        from core.nram import NRAM

        nram = NRAM(memory_size=64)
        nram.memory_state[:] = 999.0
        nram.reset_state()
        assert not np.all(nram.memory_state == 999.0)

    def test_get_explorer_state_returns_dict(self):
        from core.nram import NRAM

        nram = NRAM(memory_size=64)
        state = nram.get_explorer_state()
        assert "memory_state" in state
        assert "consciousness_level" in state
        assert "pattern_intensity" in state
        assert "network" in state

    def test_update_configuration(self):
        from core.nram import NRAM

        nram = NRAM(memory_size=64)
        nram.update_configuration({"entropy_factor": 0.7})
        assert nram.entropy_factor == 0.7
