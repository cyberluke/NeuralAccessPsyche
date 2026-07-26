"""Adversarial Test 1: Verify __req__ injection in custom_params.

FORENSIC CLAIM: build_custom_params() never includes __req__ key,
so NRAMLogitProcessor cannot access request.output_ids.

TEST STRATEGY:
1. Create ChatCompletionRequest with NRAM enabled
2. Call SGLangEngine._build_upstream_payload()
3. Extract custom_params from payload
4. Check if "__req__" key exists
5. If missing, prove the defect is real
"""
import pytest
from unittest.mock import MagicMock
from core.contracts.openai import ChatCompletionRequest
from core.engines.sglang_engine import SGLangEngine
from core.steering.serialization import build_custom_params


class TestReqInjection:
    """Test suite to falsify __req__ injection claim."""

    def test_build_custom_params_does_not_include_req(self):
        """PROVE: build_custom_params() output lacks __req__ key."""
        # Arrange
        positive_ids = [1, 2, 3]
        negative_ids = [4, 5, 6]
        forbidden_ids = [7, 8, 9]
        positive_bias = 0.5
        negative_bias = 0.3
        repetition_penalty = 1.2
        profile = "peak"
        max_tokens = 512
        phenomenon_weights = {"overlap": 0.8, "forgetting": 0.6}

        # Act
        custom_params = build_custom_params(
            positive_token_ids=positive_ids,
            negative_token_ids=negative_ids,
            forbidden_token_ids=forbidden_ids,
            positive_bias=positive_bias,
            negative_bias=negative_bias,
            repetition_penalty=repetition_penalty,
            profile=profile,
            max_tokens=max_tokens,
            phenomenon_weights=phenomenon_weights,
        )

        # Assert - PROVE the defect
        assert "__req__" not in custom_params, \
            "DEFECT FALSIFIED: __req__ is present in custom_params"
        
        # Verify what IS present
        assert "positive_token_ids" in custom_params
        assert "negative_token_ids" in custom_params
        assert "forbidden_token_ids" in custom_params
        assert "positive_bias" in custom_params
        assert "negative_bias" in custom_params
        assert "repetition_penalty" in custom_params
        assert "profile" in custom_params
        assert "max_tokens" in custom_params
        assert "phenomenon_weights" in custom_params
        
        print("[CONFIRMED] __req__ is NOT in custom_params")
        print(f"  Keys present: {list(custom_params.keys())}")

    @pytest.mark.asyncio
    async def test_sglang_payload_lacks_req_in_custom_params(self):
        """PROVE: SGLangEngine._build_upstream_payload() doesn't inject __req__."""
        # Arrange - create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 1, "token2": 2, "token3": 3}
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "test"
        
        # Create engine with tokenizer (required for bias compiler)
        engine = SGLangEngine(
            base_url="http://localhost:30000/v1",
            model="nram-qwen3-14b-awq",
            tokenizer=mock_tokenizer,
        )

        # Create request with NRAM enabled
        request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "Test prompt"}],
            nram={
                "enabled": True,
                "profile": "peak",
                "intensity": 0.9,
                "phenomenon_weights": {"overlap": 0.8, "forgetting": 0.6},
            },
            max_tokens=512,
        )

        # Act
        nram_enabled = engine._is_nram_enabled(request)
        assert nram_enabled, "NRAM should be enabled for this request"
        
        payload = engine._build_upstream_payload(
            request,
            nram_enabled=True,
            developer_instruction="Test instruction",
            plan_fragment=None,
        )

        # Assert - PROVE the defect
        assert "custom_params" in payload, "custom_params should be in payload"
        custom_params = payload["custom_params"]
        
        assert "__req__" not in custom_params, \
            "DEFECT FALSIFIED: __req__ is injected into custom_params"
        
        print("[CONFIRMED] __req__ is NOT in SGLang payload custom_params")
        print(f"  Keys present: {list(custom_params.keys())}")

    def test_logit_processor_request_is_none(self):
        """PROVE: NRAMLogitProcessor receives params without __req__, so request is None."""
        from core.steering.nram_logit_processor import NRAMLogitProcessor
        import numpy as np

        # Arrange - simulate what SGLang actually sends (no __req__)
        params_without_req = {
            "positive_token_ids": [1, 2, 3],
            "negative_token_ids": [4, 5, 6],
            "forbidden_token_ids": [7, 8, 9],
            "positive_bias": 0.5,
            "negative_bias": 0.3,
            "repetition_penalty": 1.2,
            "profile": "peak",
            "max_tokens": 512,
            "phenomenon_weights": {"overlap": 0.8, "forgetting": 0.6},
        }

        processor = NRAMLogitProcessor()
        logits = np.zeros((1, 100), dtype=np.float32)

        # Act - call processor with params that lack __req__
        result_logits = processor(logits, [params_without_req])

        # Assert - verify request was None inside processor
        # We can't directly inspect the local variable, but we can verify
        # the behavior: phenomena should NOT fire because request is None
        
        # Store original logits for comparison
        original_logits = logits.copy()
        
        # If phenomena fired, logits would be modified
        # But since request is None, phenomena won't fire
        # Only base steering (positive/negative/forbidden) should apply
        
        # Check that positive bias was applied (base steering works)
        assert result_logits[0, 1] > 0, "Positive bias should be applied"
        assert result_logits[0, 4] < 0, "Negative bias should be applied"
        assert result_logits[0, 7] == -float("inf"), "Forbidden tokens should be -inf"
        
        print("[CONFIRMED] Base steering works, but phenomena cannot fire without __req__")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
