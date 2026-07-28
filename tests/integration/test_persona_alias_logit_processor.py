"""
Integration test: Verify that persona aliases activate NRAM logit processor.

This test proves that when a request uses a persona model alias (e.g., persona-peak),
the engine internally switches to nram-qwen3-14b-awq and injects custom_logit_processor
into the upstream payload.

This is the P0-5 fix from docs/nram-fix-spec-v1.md.
"""
from __future__ import annotations

import os
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Stub openai module if not installed (needed by core.llm_handler)
if "openai" not in sys.modules:
    try:
        import openai  # noqa: F401
    except ImportError:
        _fake_openai = types.ModuleType("openai")
        _fake_openai.OpenAI = type("OpenAI", (), {"__init__": lambda s, *a, **kw: None})
        _fake_openai.AsyncOpenAI = type("AsyncOpenAI", (), {"__init__": lambda s, *a, **kw: None})
        _fake_openai.APIError = type("APIError", (Exception,), {})
        sys.modules["openai"] = _fake_openai

# Stub guidance if not installed
try:
    import guidance  # noqa: F401
except ImportError:
    _fake_guidance = types.ModuleType("guidance")
    _fake_models = types.ModuleType("guidance.models")

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    _fake_models.OpenAI = _FakeOpenAI
    _fake_guidance.models = _fake_models
    _fake_guidance.gen = lambda *a, **kw: None
    _fake_guidance.select = lambda *a, **kw: None
    _fake_guidance.system = lambda: type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
    _fake_guidance.user = lambda: type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
    _fake_guidance.assistant = lambda: type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

    sys.modules["guidance"] = _fake_guidance
    sys.modules["guidance.models"] = _fake_models


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Set required env vars for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")
    monkeypatch.setenv("AI_INTEGRATIONS_OPENROUTER_API_KEY", "test-dummy-key")
    monkeypatch.setenv("NRAM_API_KEY", "dev-nram-key")
    monkeypatch.setenv("NRAM_ENGINE", "sglang")
    monkeypatch.setenv("NRAM_TOKENIZER_PATH", "/fake/tokenizer/path")


class TestPersonaAliasActivatesLogitProcessor:
    """Test that persona aliases correctly activate NRAM logit processor."""

    @pytest.mark.asyncio
    async def test_route_persona_uses_internal_model_without_mutating_input(self):
        """Persona routing uses Qwen internally and preserves caller state."""
        from api.routes import _route_persona, ChatCompletionRequest
        from core.engines.registry import reset_engine_cache

        # Reset engine cache to ensure clean state
        reset_engine_cache()

        # Create a persona request
        request = ChatCompletionRequest(
            model="persona-peak",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=100,
        )

        # Mock the engine
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "nram-qwen3-14b-awq",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_engine.complete = AsyncMock(return_value=mock_response)

        with patch("api.routes.get_sglang_engine", return_value=mock_engine):
            result = await _route_persona(request, "persona-peak")

        assert request.model == "persona-peak"
        engine_request = mock_engine.complete.await_args.args[0]
        assert engine_request.model == "nram-qwen3-14b-awq"
        assert engine_request.public_model == "persona-peak"

        # Verify response returns public model name
        assert result["model"] == "persona-peak", (
            "Response should return public model name 'persona-peak'"
        )

        # Verify NRAM options were injected only into the immutable internal copy.
        assert request.nram is None
        assert engine_request.nram.get("enabled") is True
        assert engine_request.nram.get("profile") == "peak"

    @pytest.mark.asyncio
    async def test_moe_subrequest_sets_internal_model(self):
        """Verify MoE query_persona() sets sub_request.model to nram-qwen3-14b-awq."""
        from api.routes import _route_moe, ChatCompletionRequest
        from core.engines.registry import reset_engine_cache

        reset_engine_cache()

        request = ChatCompletionRequest(
            model="nram-moe-orchestrator",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=2048,
        )

        # Mock the engine
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "qwen3-14b-awq-baseline",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthesized response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        }
        mock_engine.complete = AsyncMock(return_value=mock_response)

        with patch("api.routes.get_sglang_engine", return_value=mock_engine):
            result = await _route_moe(request)

        # Verify engine.complete was called multiple times (personas + synthesis)
        assert mock_engine.complete.call_count >= 5, (
            f"Expected at least 5 calls (4 personas + 1 synthesis), got {mock_engine.complete.call_count}"
        )

        # Verify all persona subrequests used nram-qwen3-14b-awq
        calls = mock_engine.complete.call_args_list
        for i, call in enumerate(calls[:4]):  # First 4 are persona queries
            sub_request = call[0][0]
            assert sub_request.model == "nram-qwen3-14b-awq", (
                f"Persona subrequest {i} should use nram-qwen3-14b-awq, got {sub_request.model}"
            )
            assert sub_request.nram is not None, f"Persona subrequest {i} should have NRAM options"
            assert sub_request.nram.get("enabled") is True, f"Persona subrequest {i} should have NRAM enabled"
            assert sub_request.max_tokens >= 800, (
                f"Persona subrequest {i} should have max_tokens >= 800, got {sub_request.max_tokens}"
            )

    def test_engine_injects_custom_logit_processor(self):
        """Verify SGLangEngine injects custom_logit_processor when NRAM is enabled."""
        from core.engines.sglang_engine import SGLangEngine
        from core.contracts.openai import ChatCompletionRequest

        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1234]
        mock_tokenizer.decode.return_value = "test"
        mock_tokenizer.vocab_size = 152064

        # Create engine with tokenizer
        engine = SGLangEngine(
            base_url="http://test:30000/v1",
            model="nram-qwen3-14b-awq",
            tokenizer=mock_tokenizer,
        )

        # Verify bias compiler was created
        assert engine._bias_compiler is not None, (
            "TokenBiasCompiler should be created when tokenizer is provided"
        )

        # Create an NRAM-enabled request
        request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "Test"}],
            nram={"enabled": True, "profile": "peak"},
        )

        # Verify NRAM is detected as enabled
        assert engine._is_nram_enabled(request) is True, (
            "NRAM should be enabled for nram-qwen3-14b-awq model"
        )

        # Build payload and verify custom_logit_processor is injected
        payload = engine._build_upstream_payload(request, nram_enabled=True)

        assert "custom_logit_processor" in payload, (
            "custom_logit_processor should be injected into payload when NRAM is enabled"
        )
        assert "custom_params" in payload, (
            "custom_params should be injected into payload when NRAM is enabled"
        )

    def test_persona_model_not_in_nram_enabled_aliases(self):
        """Verify persona models are NOT in NRAM_ENABLED_ALIASES (they route through _route_persona)."""
        from core.engines.sglang_engine import NRAM_ENABLED_ALIASES

        persona_models = [
            "persona-normal",
            "persona-microdose",
            "persona-threshold",
            "persona-psychedelic",
            "persona-peak",
            "persona-dissociative",
        ]

        for model in persona_models:
            assert model not in NRAM_ENABLED_ALIASES, (
                f"{model} should NOT be in NRAM_ENABLED_ALIASES. "
                "Persona models route through _route_persona() which sets model to nram-qwen3-14b-awq."
            )

    def test_nram_enabled_aliases_contains_actual_nram_models(self):
        """Only the checkpoint actually loaded by SGLang may be enabled."""
        from core.engines.sglang_engine import NRAM_ENABLED_ALIASES

        assert NRAM_ENABLED_ALIASES == {"nram-qwen3-14b-awq"}


class TestTokenizerFailFast:
    """Test that missing tokenizer causes fail-fast when NRAM is enabled."""

    def test_missing_tokenizer_raises_error(self, monkeypatch):
        """Verify RuntimeError when NRAM_TOKENIZER_PATH is not set."""
        from core.engines.registry import get_sglang_engine, reset_engine_cache

        reset_engine_cache()
        monkeypatch.setenv("NRAM_ENGINE", "sglang")
        monkeypatch.delenv("NRAM_TOKENIZER_PATH", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            get_sglang_engine()

        assert "NRAM_TOKENIZER_PATH" in str(exc_info.value), (
            "Error message should mention NRAM_TOKENIZER_PATH"
        )
        assert "silent" in str(exc_info.value).lower(), (
            "Error message should mention silent degradation"
        )


class TestVisionaryPeakProfile:
    """Test the VISIONARY_PEAK profile (P1-4)."""

    def test_visionary_peak_exists(self):
        """Verify VISIONARY_PEAK profile exists in PROFILES."""
        from core.persona.profiles import PROFILES, VISIONARY_PEAK

        assert "visionary-peak" in PROFILES, "visionary-peak should be in PROFILES"
        assert PROFILES["visionary-peak"] is VISIONARY_PEAK

    def test_visionary_peak_has_high_coherence(self):
        """Verify VISIONARY_PEAK has high coherence (0.88) unlike altered-state PEAK."""
        from core.persona.profiles import VISIONARY_PEAK, PEAK

        assert VISIONARY_PEAK.coherence_floor == 0.88, (
            "VISIONARY_PEAK should have coherence_floor=0.88"
        )
        assert PEAK.coherence_floor == 0.58, (
            "PEAK (altered-state) should have coherence_floor=0.58"
        )

    def test_visionary_peak_has_high_novelty(self):
        """Verify VISIONARY_PEAK has high novelty and associative distance."""
        from core.persona.profiles import VISIONARY_PEAK

        assert VISIONARY_PEAK.novelty_target >= 0.90, (
            "VISIONARY_PEAK should have novelty_target >= 0.90"
        )
        assert VISIONARY_PEAK.associative_distance >= 0.85, (
            "VISIONARY_PEAK should have associative_distance >= 0.85"
        )

    def test_visionary_peak_has_high_product_obsession(self):
        """Verify VISIONARY_PEAK has high product obsession."""
        from core.persona.profiles import VISIONARY_PEAK

        assert VISIONARY_PEAK.product_obsession >= 0.90, (
            "VISIONARY_PEAK should have product_obsession >= 0.90"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
