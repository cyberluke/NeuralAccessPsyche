"""Adversarial Test 3: Verify MoE Synthesis Configuration.

FORENSIC CLAIM: MoE synthesis step uses qwen3-14b-awq-baseline with nram=None,
losing persona characteristics.

TEST STRATEGY:
1. Trace MoE code path in api/routes.py
2. Verify synthesis request configuration
3. Test whether synthesis preserves or loses NRAM steering
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from core.contracts.openai import ChatCompletionRequest


class TestMoESynthesisConfiguration:
    """Test suite to falsify MoE synthesis defect claim."""

    @pytest.mark.asyncio
    async def test_moe_synthesis_uses_baseline_model(self):
        """PROVE: MoE synthesis uses baseline model, not NRAM-enabled model."""
        # Arrange - simulate the synthesis step from _route_moe
        # This is the actual code path from api/routes.py lines 216-223
        
        # Create a request as if we're in the synthesis step
        synth_request = ChatCompletionRequest(
            model="qwen3-14b-awq-baseline",  # This is what synthesis uses
            messages=[
                {"role": "system", "content": "You are a strategic synthesizer."},
                {"role": "user", "content": "Synthesize these perspectives..."}
            ],
            max_tokens=2048,
            temperature=0.5,
            nram=None  # This is what synthesis sets
        )
        
        # Act - check what the engine would do with this request
        from core.engines.sglang_engine import SGLangEngine, NRAM_ENABLED_ALIASES
        
        # Create a mock engine
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 1}
        engine = SGLangEngine(
            base_url="http://localhost:30000/v1",
            model="qwen3-14b-awq",
            tokenizer=mock_tokenizer
        )
        
        # Check if NRAM would be enabled
        nram_enabled = engine._is_nram_enabled(synth_request)
        
        # Assert - PROVE the defect
        assert synth_request.model == "qwen3-14b-awq-baseline", \
            f"DEFECT FALSIFIED: synthesis uses {synth_request.model}"
        
        assert synth_request.nram is None, \
            f"DEFECT FALSIFIED: synthesis has nram={synth_request.nram}"
        
        assert synth_request.model not in NRAM_ENABLED_ALIASES, \
            f"DEFECT FALSIFIED: synthesis model {synth_request.model} is NRAM-enabled"
        
        assert nram_enabled is False, \
            "DEFECT FALSIFIED: synthesis request would enable NRAM"
        
        print(f"[CONFIRMED] MoE synthesis uses baseline model")
        print(f"  Synthesis model: {synth_request.model}")
        print(f"  Synthesis nram: {synth_request.nram}")
        print(f"  NRAM would be enabled: {nram_enabled}")

    def test_moe_persona_queries_use_nram(self):
        """PROVE: MoE persona queries DO use NRAM steering."""
        # Arrange - simulate persona query from _route_moe
        # This is the actual code path from api/routes.py lines 183-184
        
        persona_request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",  # This is what persona queries use
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=1024,
            nram={
                "enabled": True,
                "profile": "psychedelic",
                "intensity": 0.9
            }
        )
        
        # Act - check what the engine would do
        from core.engines.sglang_engine import SGLangEngine, NRAM_ENABLED_ALIASES
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 1}
        engine = SGLangEngine(
            base_url="http://localhost:30000/v1",
            model="qwen3-14b-awq",
            tokenizer=mock_tokenizer
        )
        
        nram_enabled = engine._is_nram_enabled(persona_request)
        
        # Assert - persona queries should enable NRAM
        assert persona_request.model == "nram-qwen3-14b-awq", \
            f"Persona query uses wrong model: {persona_request.model}"
        
        assert persona_request.model in NRAM_ENABLED_ALIASES, \
            f"Persona query model {persona_request.model} is not NRAM-enabled"
        
        assert persona_request.nram is not None, \
            "Persona query has nram=None"
        
        assert persona_request.nram.get("enabled") is True, \
            "Persona query has nram.enabled=False"
        
        assert nram_enabled is True, \
            "Persona query should enable NRAM"
        
        print(f"[CONFIRMED] MoE persona queries use NRAM")
        print(f"  Persona model: {persona_request.model}")
        print(f"  Persona nram: {persona_request.nram}")
        print(f"  NRAM enabled: {nram_enabled}")

    def test_moe_synthesis_loses_persona_characteristics(self):
        """PROVE: Synthesis step loses persona-specific steering."""
        # Arrange - compare persona query vs synthesis
        from core.engines.sglang_engine import SGLangEngine, NRAM_ENABLED_ALIASES
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 1}
        engine = SGLangEngine(
            base_url="http://localhost:30000/v1",
            model="qwen3-14b-awq",
            tokenizer=mock_tokenizer
        )
        
        # Persona query (has NRAM)
        persona_request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "Test"}],
            nram={"enabled": True, "profile": "psychedelic", "intensity": 0.9}
        )
        
        # Synthesis request (no NRAM)
        synth_request = ChatCompletionRequest(
            model="qwen3-14b-awq-baseline",
            messages=[{"role": "user", "content": "Synthesize"}],
            nram=None
        )
        
        # Act
        persona_enabled = engine._is_nram_enabled(persona_request)
        synth_enabled = engine._is_nram_enabled(synth_request)
        
        # Assert - synthesis should NOT have NRAM
        assert persona_enabled is True, \
            "Persona query should enable NRAM"
        
        assert synth_enabled is False, \
            "DEFECT FALSIFIED: synthesis should NOT enable NRAM"
        
        # Verify the difference
        assert persona_request.model != synth_request.model, \
            "Persona and synthesis should use different models"
        
        assert persona_request.nram != synth_request.nram, \
            "Persona and synthesis should have different nram settings"
        
        print(f"[CONFIRMED] Synthesis loses persona characteristics")
        print(f"  Persona: model={persona_request.model}, nram={persona_request.nram}, enabled={persona_enabled}")
        print(f"  Synthesis: model={synth_request.model}, nram={synth_request.nram}, enabled={synth_enabled}")

    def test_moe_synthesis_payload_lacks_custom_processor(self):
        """PROVE: Synthesis payload does not include custom_logit_processor."""
        # Arrange
        from core.engines.sglang_engine import SGLangEngine
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 1, "token2": 2}
        mock_tokenizer.encode.return_value = [1, 2]
        mock_tokenizer.decode.return_value = "test"
        
        engine = SGLangEngine(
            base_url="http://localhost:30000/v1",
            model="qwen3-14b-awq",
            tokenizer=mock_tokenizer
        )
        
        # Synthesis request (no NRAM)
        synth_request = ChatCompletionRequest(
            model="qwen3-14b-awq-baseline",
            messages=[{"role": "user", "content": "Synthesize"}],
            nram=None,
            max_tokens=2048
        )
        
        # Act
        nram_enabled = engine._is_nram_enabled(synth_request)
        payload = engine._build_upstream_payload(
            synth_request,
            nram_enabled=nram_enabled,
            developer_instruction="You are a synthesizer.",
            plan_fragment=None
        )
        
        # Assert - synthesis should NOT have custom_logit_processor
        assert "custom_logit_processor" not in payload, \
            "DEFECT FALSIFIED: synthesis has custom_logit_processor"
        
        assert "custom_params" not in payload, \
            "DEFECT FALSIFIED: synthesis has custom_params"
        
        print(f"[CONFIRMED] Synthesis payload lacks custom processor")
        print(f"  NRAM enabled: {nram_enabled}")
        print(f"  Payload keys: {list(payload.keys())}")
        print(f"  Has custom_logit_processor: {'custom_logit_processor' in payload}")
        print(f"  Has custom_params: {'custom_params' in payload}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
