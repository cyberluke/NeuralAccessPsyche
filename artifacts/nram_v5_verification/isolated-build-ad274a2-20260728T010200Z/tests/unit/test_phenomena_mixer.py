"""Unit tests for the phenomenon mixer in NRAMLogitProcessor.

Tests all 7 phenomena at logit level:
- overlap: boost recent output tokens
- forgetting: extra repetition penalty
- looping: reward recent tokens
- associative_jump: flatten distribution
- synesthesia: cross-activation noise
- dissolution: scale logits toward zero
- insight: periodic positive_bias spikes
"""
import math
import pytest

torch = pytest.importorskip("torch")

from core.steering.nram_logit_processor import NRAMLogitProcessor


class FakeRequest:
    def __init__(self, output_ids):
        self.output_ids = output_ids


class TestPhenomenonOverlap:
    """overlap: boost recent output tokens (self-reinforcing bleed)."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_overlap_boosts_recent_tokens(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"overlap": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Recent tokens should be boosted
        assert out[0, 5].item() > 0.0
        assert out[0, 10].item() > 0.0
        assert out[0, 15].item() > 0.0
        # Non-recent tokens should not be boosted
        assert out[0, 50].item() == pytest.approx(0.0)

    def test_overlap_weight_zero_no_effect(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"overlap": 0.0},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(0.0)

    def test_overlap_below_threshold_no_effect(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"overlap": 0.03},  # Below 0.05 threshold
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(0.0)

    def test_overlap_with_empty_history(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[])
        params = [{
            "phenomenon_weights": {"overlap": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # No crash, no effect
        assert torch.allclose(out, logits)


class TestPhenomenonForgetting:
    """forgetting: extra repetition penalty on recent tokens."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_forgetting_penalizes_recent_tokens(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"forgetting": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Recent tokens should be penalized (negative)
        assert out[0, 5].item() < 0.0
        assert out[0, 10].item() < 0.0
        assert out[0, 15].item() < 0.0

    def test_forgetting_weight_zero_no_effect(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"forgetting": 0.0},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(0.0)


class TestPhenomenonLooping:
    """looping: REWARD recent tokens (perseveration / thought loops)."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_looping_rewards_recent_tokens(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"looping": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Recent tokens should be rewarded (positive)
        assert out[0, 5].item() > 0.0
        assert out[0, 10].item() > 0.0
        assert out[0, 15].item() > 0.0

    def test_looping_window_is_16_tokens(self):
        logits = torch.zeros((1, 100))
        # Create 20 tokens, only last 16 should be rewarded
        req = FakeRequest(output_ids=list(range(20)))
        params = [{
            "phenomenon_weights": {"looping": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Tokens 0-3 should NOT be rewarded (outside 16-token window)
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[0, 3].item() == pytest.approx(0.0)
        # Tokens 4-19 should be rewarded
        assert out[0, 4].item() > 0.0
        assert out[0, 19].item() > 0.0


class TestPhenomenonAssociativeJump:
    """associative_jump: flatten distribution toward uniform."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_associative_jump_flattens_logits(self):
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.0, -2.0]])
        req = FakeRequest(output_ids=[])
        params = [{
            "phenomenon_weights": {"associative_jump": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Logits should be scaled toward zero (flattened)
        # Original range: 5.0 - (-2.0) = 7.0
        # After scaling, range should be smaller
        original_range = 5.0 - (-2.0)
        new_range = out[0].max().item() - out[0].min().item()
        assert new_range < original_range

    def test_associative_jump_weight_zero_no_effect(self):
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.0, -2.0]])
        req = FakeRequest(output_ids=[])
        params = [{
            "phenomenon_weights": {"associative_jump": 0.0},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        assert torch.allclose(out, logits)


class TestPhenomenonSynesthesia:
    """synesthesia: deterministic cross-activation noise."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_synesthesia_adds_noise_to_scattered_tokens(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[])
        params = [{
            "phenomenon_weights": {"synesthesia": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Some tokens should have positive bias (cross-activation)
        assert (out[0] > 0).any()
        # Not all tokens should be affected (scattered pattern)
        assert (out[0] == 0).any()

    def test_synesthesia_pattern_shifts_with_generation(self):
        logits = torch.zeros((1, 100))
        # Step 0
        req0 = FakeRequest(output_ids=[])
        params0 = [{
            "phenomenon_weights": {"synesthesia": 0.8},
            "__req__": req0,
        }]
        out0 = self.proc(logits.clone(), params0)
        
        # Step 10
        req10 = FakeRequest(output_ids=list(range(10)))
        params10 = [{
            "phenomenon_weights": {"synesthesia": 0.8},
            "__req__": req10,
        }]
        out10 = self.proc(logits.clone(), params10)
        
        # Different generation steps should produce different patterns
        assert not torch.allclose(out0, out10)


class TestPhenomenonDissolution:
    """dissolution: scale logits toward zero (dissolve structure)."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_dissolution_scales_logits_toward_zero(self):
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.0, -2.0]])
        req = FakeRequest(output_ids=[])
        params = [{
            "phenomenon_weights": {"dissolution": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # All logits should be closer to zero
        assert out[0].abs().max().item() < logits.abs().max().item()

    def test_dissolution_is_strongest_effect(self):
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.0, -2.0]])
        req = FakeRequest(output_ids=[])
        # Apply both associative_jump and dissolution
        params = [{
            "phenomenon_weights": {"associative_jump": 0.8, "dissolution": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Should be even more flattened than either alone
        assert out[0].abs().max().item() < 2.0


class TestPhenomenonInsight:
    """insight: periodic positive_bias spikes at 25/50/75% of generation."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_insight_spikes_at_25_percent(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=list(range(25)))
        params = [{
            "positive_token_ids": [5, 10, 15],
            "positive_bias": 1.0,
            "max_tokens": 100,
            "phenomenon_weights": {"insight": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # At 25% progress, positive tokens should be spiked
        assert out[0, 5].item() > 1.0  # More than base positive_bias

    def test_insight_spikes_at_50_percent(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=list(range(50)))
        params = [{
            "positive_token_ids": [5, 10, 15],
            "positive_bias": 1.0,
            "max_tokens": 100,
            "phenomenon_weights": {"insight": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # At 50% progress, positive tokens should be spiked
        assert out[0, 5].item() > 1.0

    def test_insight_spikes_at_75_percent(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=list(range(75)))
        params = [{
            "positive_token_ids": [5, 10, 15],
            "positive_bias": 1.0,
            "max_tokens": 100,
            "phenomenon_weights": {"insight": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # At 75% progress, positive tokens should be spiked
        assert out[0, 5].item() > 1.0

    def test_insight_no_spike_outside_windows(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=list(range(10)))  # 10% progress
        params = [{
            "positive_token_ids": [5, 10, 15],
            "positive_bias": 1.0,
            "max_tokens": 100,
            "phenomenon_weights": {"insight": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # At 10% progress, no spike (only base positive_bias)
        assert out[0, 5].item() == pytest.approx(1.0)


class TestPhenomenonCombinations:
    """Test multiple phenomena working together."""

    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_overlap_and_forgetting_cancel_partially(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[5, 10, 15])
        params = [{
            "phenomenon_weights": {"overlap": 0.8, "forgetting": 0.8},
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Both operate on recent tokens but in opposite directions
        # forgetting is stronger (1.8) than overlap (0.35)
        assert out[0, 5].item() < 0.0  # Net negative

    def test_all_phenomena_no_crash(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=list(range(50)))
        params = [{
            "positive_token_ids": [5, 10, 15],
            "positive_bias": 1.0,
            "max_tokens": 100,
            "phenomenon_weights": {
                "overlap": 0.5,
                "forgetting": 0.5,
                "looping": 0.5,
                "associative_jump": 0.5,
                "synesthesia": 0.5,
                "dissolution": 0.5,
                "insight": 0.5,
            },
            "__req__": req,
        }]
        out = self.proc(logits.clone(), params)
        # Should not crash, output should be valid
        assert out.shape == (1, 100)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_phenomena_require_request(self):
        """Phenomena should not apply without a request object."""
        logits = torch.zeros((1, 100))
        params = [{
            "phenomenon_weights": {"overlap": 0.8, "forgetting": 0.8},
            # No __req__
        }]
        out = self.proc(logits.clone(), params)
        # Should not crash, should return logits unchanged
        assert torch.allclose(out, logits)
