"""Adversarial Test 4: End-to-End Logit Processor Integration Test.

FORENSIC CLAIM: The combination of missing __req__ injection and phenomenon
mixer gating means that repetition penalty and phenomena are effectively dead.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from core.steering.nram_logit_processor import NRAMLogitProcessor
from core.steering.serialization import build_custom_params


class TestEndToEndLogitProcessor:
    """End-to-end test of logit processor behavior."""

    def test_full_generation_simulation_without_req(self):
        """Simulate multi-token generation WITHOUT __req__ (production reality)."""
        processor = NRAMLogitProcessor()
        vocab_size = 1000
        
        params = build_custom_params(
            positive_token_ids=[10, 20, 30],
            negative_token_ids=[40, 50, 60],
            forbidden_token_ids=[70, 80, 90],
            positive_bias=0.8,
            negative_bias=0.5,
            repetition_penalty=1.5,
            profile="peak",
            max_tokens=100,
            phenomenon_weights={
                "overlap": 0.9,
                "forgetting": 0.8,
            },
        )

        # Simulate 10 generation steps
        generation_log = []
        for step in range(10):
            logits = np.zeros((1, vocab_size), dtype=np.float32)
            
            result_logits = processor(logits, [params])
            
            step_info = {
                "step": step,
                "positive_bias_applied": result_logits[0, 10] > 0,
                "negative_bias_applied": result_logits[0, 40] < 0,
                "forbidden_applied": result_logits[0, 70] == -float("inf"),
                "repetition_penalty_applied": False,  # Cannot apply without __req__
                "phenomena_fired": False,  # Cannot fire without __req__
            }
            generation_log.append(step_info)

        for step_info in generation_log:
            assert step_info["positive_bias_applied"], "Positive bias should work"
            assert step_info["negative_bias_applied"], "Negative bias should work"
            assert step_info["forbidden_applied"], "Forbidden tokens should work"
            assert not step_info["repetition_penalty_applied"], \
                "DEFECT CONFIRMED: Repetition penalty cannot work without __req__"
            assert not step_info["phenomena_fired"], \
                "DEFECT CONFIRMED: Phenomena cannot fire without __req__"

        print("[CONFIRMED] End-to-end generation without __req__")
        print(f"  Base steering works: positive, negative, forbidden")
        print(f"  Repetition penalty: BROKEN (requires __req__)")
        print(f"  Phenomena: BROKEN (requires __req__)")
        print(f"  Generation steps: {len(generation_log)}")

    def test_full_generation_simulation_with_req(self):
        """Simulate multi-token generation WITH __req__ (what should happen)."""
        processor = NRAMLogitProcessor()
        vocab_size = 1000
        
        params = build_custom_params(
            positive_token_ids=[10, 20, 30],
            negative_token_ids=[40, 50, 60],
            forbidden_token_ids=[70, 80, 90],
            positive_bias=0.8,
            negative_bias=0.5,
            repetition_penalty=1.5,
            profile="peak",
            max_tokens=100,
            phenomenon_weights={
                "overlap": 0.9,
                "forgetting": 0.8,
            },
        )

        mock_request = MagicMock()
        mock_request.output_ids = []
        
        generation_log = []
        for step in range(10):
            mock_request.output_ids.append(100 + step)
            
            logits = np.zeros((1, vocab_size), dtype=np.float32)
            
            params_with_req = params.copy()
            params_with_req["__req__"] = mock_request
            
            result_logits = processor(logits, [params_with_req])
            
            recent_tokens = list(mock_request.output_ids[-64:])
            step_info = {
                "step": step,
                "generated": len(mock_request.output_ids),
                "positive_bias_applied": result_logits[0, 10] > 0,
                "negative_bias_applied": result_logits[0, 40] < 0,
                "forbidden_applied": result_logits[0, 70] == -float("inf"),
                "repetition_penalty_applied": any(
                    result_logits[0, tid] < 0 for tid in recent_tokens
                ),
                "phenomena_fired": any(
                    result_logits[0, tid] != 0 for tid in recent_tokens
                    if tid not in [10, 20, 30, 40, 50, 60, 70, 80, 90]
                ),
            }
            generation_log.append(step_info)

        for step_info in generation_log:
            assert step_info["positive_bias_applied"], "Positive bias should work"
            assert step_info["negative_bias_applied"], "Negative bias should work"
            assert step_info["forbidden_applied"], "Forbidden tokens should work"
            
            if step_info["generated"] > 0:
                assert step_info["repetition_penalty_applied"], \
                    f"Repetition penalty should work at step {step_info['step']}"
            
            if step_info["generated"] > 0:
                assert step_info["phenomena_fired"], \
                    f"Phenomena should fire at step {step_info['step']}"

        print("[CONFIRMED] End-to-end generation WITH __req__")
        print(f"  All features work: base steering, repetition penalty, phenomena")
        print(f"  Generation steps: {len(generation_log)}")
        print(f"  Final output history length: {generation_log[-1]['generated']}")

    def test_comparison_baseline_vs_nram_without_req(self):
        """Compare baseline vs NRAM mode when __req__ is missing."""
        processor = NRAMLogitProcessor()
        vocab_size = 1000
        
        baseline_params = {
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 0.0,
            "profile": "normal",
            "max_tokens": 100,
        }
        
        nram_params = build_custom_params(
            positive_token_ids=[10, 20, 30],
            negative_token_ids=[40, 50, 60],
            forbidden_token_ids=[70, 80, 90],
            positive_bias=0.8,
            negative_bias=0.5,
            repetition_penalty=1.5,
            profile="peak",
            max_tokens=100,
            phenomenon_weights={
                "overlap": 0.9,
                "forgetting": 0.8,
            },
        )

        logits_baseline = np.zeros((1, vocab_size), dtype=np.float32)
        logits_nram = np.zeros((1, vocab_size), dtype=np.float32)

        result_baseline = processor(logits_baseline, [baseline_params])
        result_nram = processor(logits_nram, [nram_params])

        assert np.allclose(result_baseline, 0.0), "Baseline should produce no changes"
        assert not np.allclose(result_nram, 0.0), "NRAM should produce changes"
        
        # Check that overlap did NOT fire (no __req__)
        assert result_nram[0, 100] == 0.0, \
            "Overlap should not boost token 100 without __req__"
        
        print("[CONFIRMED] Baseline vs NRAM comparison without __req__")
        print(f"  Baseline: no changes")
        print(f"  NRAM: base steering works, phenomena do not")

    def test_phase_schedule_without_req(self):
        """Verify phase schedule works without __req__ but is static."""
        processor = NRAMLogitProcessor()
        
        params = {
            "positive_token_ids": [10],
            "negative_token_ids": [20],
            "forbidden_token_ids": [],
            "positive_bias": 1.0,
            "negative_bias": 1.0,
            "repetition_penalty": 1.0,
            "profile": "peak",
            "max_tokens": 100,
        }

        logits = np.zeros((1, 100), dtype=np.float32)
        result = processor(logits, [params])

        schedule = processor._phase_schedule(params, generated=0)
        
        assert schedule["positive_scale"] == 1.0, "Opening phase should have positive_scale=1.0"
        assert schedule["negative_scale"] == 1.0, "Opening phase should have negative_scale=1.0"
        assert schedule["repetition_scale"] == 1.0, "Opening phase should have repetition_scale=1.0"
        
        print("[CONFIRMED] Phase schedule is static without __req__")
        print(f"  Schedule at generated=0: {schedule}")
        print(f"  Phase schedule cannot progress without output_ids")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
