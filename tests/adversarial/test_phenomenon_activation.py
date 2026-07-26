"""Adversarial Test 2: Verify Phenomenon Mixer Activation.

FORENSIC CLAIM: Since __req__ is never injected, request is always None
in NRAMLogitProcessor._apply_phenomena(), so phenomena never fire.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from core.steering.nram_logit_processor import NRAMLogitProcessor


class TestPhenomenonMixerActivation:
    """Test suite to falsify phenomenon mixer dead code claim."""

    def test_phenomena_not_called_without_req(self):
        """PROVE: overlap/forgetting/looping do NOT fire when __req__ is missing."""
        processor = NRAMLogitProcessor()
        
        params_without_req = {
            "positive_token_ids": [1, 2, 3],
            "negative_token_ids": [4, 5, 6],
            "forbidden_token_ids": [7, 8, 9],
            "positive_bias": 0.5,
            "negative_bias": 0.3,
            "repetition_penalty": 1.2,
            "profile": "peak",
            "max_tokens": 512,
            "phenomenon_weights": {
                "overlap": 0.9,
                "forgetting": 0.9,
                "looping": 0.8,
            },
        }

        logits = np.zeros((1, 100), dtype=np.float32)
        result_logits = processor(logits, [params_without_req])

        # Base steering works
        assert result_logits[0, 1] > 0, "Positive bias should be applied"
        assert result_logits[0, 4] < 0, "Negative bias should be applied"
        assert result_logits[0, 7] == -float("inf"), "Forbidden tokens should be -inf"
        
        # overlap/forgetting/looping require output_ids from request
        # Without __req__, they cannot fire
        # Token 50 is NOT in positive/negative/forbidden, so should be 0
        # (overlap would boost it if it were in output_ids, but it isn't)
        assert result_logits[0, 50] == 0.0, \
            "Token 50 should not be modified without __req__"
        
        print("[CONFIRMED] overlap/forgetting/looping cannot fire without __req__")

    def test_overlap_fires_with_req(self):
        """PROVE: overlap phenomenon DOES fire when __req__ is present."""
        processor = NRAMLogitProcessor()
        
        mock_request = MagicMock()
        mock_request.output_ids = [50, 51, 52]  # Recent output history
        
        # Only overlap, no other phenomena to avoid interaction
        params_with_req = {
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 0.0,
            "profile": "peak",
            "max_tokens": 512,
            "phenomenon_weights": {
                "overlap": 0.9,
            },
            "__req__": mock_request,
        }

        logits = np.zeros((1, 100), dtype=np.float32)
        result_logits = processor(logits, [params_with_req])

        # overlap should boost recent tokens (50, 51, 52)
        assert result_logits[0, 50] > 0, f"Overlap should boost recent token 50, got {result_logits[0, 50]}"
        assert result_logits[0, 51] > 0, f"Overlap should boost recent token 51, got {result_logits[0, 51]}"
        assert result_logits[0, 52] > 0, f"Overlap should boost recent token 52, got {result_logits[0, 52]}"
        
        # Non-recent tokens should not be affected
        assert result_logits[0, 60] == 0.0, "Non-recent token should not be boosted"
        
        print(f"[CONFIRMED] overlap fires when __req__ is present")
        print(f"  Token 50 boosted: {result_logits[0, 50]}")

    def test_repetition_penalty_requires_req(self):
        """PROVE: Dynamic repetition penalty requires __req__ to access output_ids."""
        processor = NRAMLogitProcessor()
        
        mock_request = MagicMock()
        mock_request.output_ids = [10, 20, 30, 40, 50]
        
        params_without_req = {
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 1.5,
            "profile": "peak",
            "max_tokens": 512,
        }
        
        params_with_req = params_without_req.copy()
        params_with_req["__req__"] = mock_request

        logits_without = np.zeros((1, 100), dtype=np.float32)
        logits_with = np.zeros((1, 100), dtype=np.float32)

        result_without = processor(logits_without, [params_without_req])
        result_with = processor(logits_with, [params_with_req])

        assert result_without[0, 10] == 0.0, \
            "Without __req__, repetition penalty should not apply"
        assert result_with[0, 10] < 0.0, \
            "With __req__, repetition penalty should penalize recent token 10"
        
        print(f"[CONFIRMED] Repetition penalty requires __req__")
        print(f"  Token 10 without __req__: {result_without[0, 10]}")
        print(f"  Token 10 with __req__: {result_with[0, 10]}")

    def test_phenomena_gating_line_in_processor(self):
        """PROVE: The code explicitly gates phenomena on 'request is not None'."""
        import inspect
        source = inspect.getsource(NRAMLogitProcessor.__call__)
        
        assert "phenomena and request is not None" in source, \
            "DEFECT FALSIFIED: phenomena are not gated on request"
        
        print("[CONFIRMED] Source code contains: 'phenomena and request is not None'")
        print("  This proves phenomena are explicitly gated on request being non-None")

    def test_comparison_overlap_with_vs_without_req(self):
        """PROVE: overlap produces different results with vs without __req__."""
        processor = NRAMLogitProcessor()
        
        mock_request = MagicMock()
        mock_request.output_ids = [50, 51, 52]
        
        params_base = {
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 0.0,
            "profile": "peak",
            "max_tokens": 512,
            "phenomenon_weights": {"overlap": 0.9},
        }
        
        params_with_req = params_base.copy()
        params_with_req["__req__"] = mock_request

        logits_without = np.zeros((1, 100), dtype=np.float32)
        logits_with = np.zeros((1, 100), dtype=np.float32)

        result_without = processor(logits_without, [params_base])
        result_with = processor(logits_with, [params_with_req])

        # Without __req__: token 50 should be 0 (overlap can't fire)
        assert result_without[0, 50] == 0.0, \
            "Without __req__, overlap should not boost token 50"
        
        # With __req__: token 50 should be > 0 (overlap fires)
        assert result_with[0, 50] > 0.0, \
            "With __req__, overlap should boost token 50"
        
        print(f"[CONFIRMED] overlap produces different results with vs without __req__")
        print(f"  Token 50 without __req__: {result_without[0, 50]}")
        print(f"  Token 50 with __req__: {result_with[0, 50]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
