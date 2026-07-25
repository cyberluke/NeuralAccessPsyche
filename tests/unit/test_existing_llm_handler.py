"""
Unit tests for LLMHandler — verifies all Phase 1 defects are FIXED.
"""
import asyncio
import inspect
import ast
import textwrap
import pytest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Defect 1 FIXED: content extracted from dict as string
# ---------------------------------------------------------------------------

class TestDefect1Fixed_ContentIsString:
    def test_generate_response_returns_string_content(self):
        """After fix: content is extracted from guidance dict as a string."""
        from core.llm_handler import LLMHandler
        import core.llm_handler as mod

        fake_guidance_result = {
            "main_response": "Hello world",
            "consciousness_level": "baseline",
            "neural_insight": None,
            "token_phenomena": ["coherent"],
            "coherence_score": 0.7,
            "raw_tokens": [],
            "guidance_used": True,
        }

        with patch.object(mod, "GuidanceHandler") as MockGH:
            mock_instance = MockGH.return_value
            mock_instance.process_with_guidance = AsyncMock(return_value=fake_guidance_result)
            mock_instance.get_token_metrics = AsyncMock(
                return_value={"token_count": 5, "character_count": 20, "estimated_tokens": 5}
            )

            handler = LLMHandler.__new__(LLMHandler)
            handler.guidance_handler = mock_instance
            handler.model = "test-model"
            handler.consciousness_levels = {"baseline": 0.3}

            result = asyncio.get_event_loop().run_until_complete(
                handler.generate_response(
                    messages=[{"role": "user", "content": "Hi"}],
                    temperature=0.5,
                    max_tokens=50,
                    model="nram-gpt-oss-20b",
                )
            )

            content = result["choices"][0]["message"]["content"]
            assert isinstance(content, str)
            assert content == "Hello world"


# ---------------------------------------------------------------------------
# Defect 2 FIXED: get_token_metrics is awaited
# ---------------------------------------------------------------------------

class TestDefect2Fixed_Awaited:
    def test_get_token_metrics_is_awaited_in_source(self):
        """After fix: generate_response awaits get_token_metrics."""
        from core.llm_handler import LLMHandler

        src = textwrap.dedent(inspect.getsource(LLMHandler.generate_response))
        tree = ast.parse(src)

        found_awaited = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Await):
                if isinstance(node.value, ast.Call):
                    if hasattr(node.value.func, "attr") and node.value.func.attr == "get_token_metrics":
                        found_awaited = True

        assert found_awaited, "get_token_metrics must be awaited after fix"


# ---------------------------------------------------------------------------
# Defect 4 FIXED: sync OpenAI client no longer awaited
# ---------------------------------------------------------------------------

class TestDefect4Fixed_NoAwaitOnSyncClient:
    def test_config_suggester_no_longer_awaits_sync_client(self):
        """After fix: _get_ai_suggestions does NOT await the sync OpenAI client."""
        from core.config_suggester import NRAMConfigSuggester

        src = inspect.getsource(NRAMConfigSuggester._get_ai_suggestions)
        assert "await self.client.chat.completions.create" not in src, (
            "Sync OpenAI client must not be awaited after fix"
        )


# ---------------------------------------------------------------------------
# Defect 5 FIXED: model field reflected in response
# ---------------------------------------------------------------------------

class TestDefect5Fixed_ModelReflected:
    def test_generate_response_accepts_model_param(self):
        """After fix: generate_response accepts and uses a model parameter."""
        from core.llm_handler import LLMHandler

        sig = inspect.signature(LLMHandler.generate_response)
        assert "model" in sig.parameters

    def test_model_field_in_response(self):
        """After fix: response model field reflects the requested alias."""
        from core.llm_handler import LLMHandler
        import core.llm_handler as mod

        fake_guidance_result = {"main_response": "test", "coherence_score": 0.7}

        with patch.object(mod, "GuidanceHandler") as MockGH:
            mock_instance = MockGH.return_value
            mock_instance.process_with_guidance = AsyncMock(return_value=fake_guidance_result)
            mock_instance.get_token_metrics = AsyncMock(
                return_value={"token_count": 1, "character_count": 4, "estimated_tokens": 1}
            )

            handler = LLMHandler.__new__(LLMHandler)
            handler.guidance_handler = mock_instance
            handler.model = "default-model"
            handler.consciousness_levels = {"baseline": 0.3}

            result = asyncio.get_event_loop().run_until_complete(
                handler.generate_response(
                    messages=[{"role": "user", "content": "Hi"}],
                    temperature=0.5,
                    max_tokens=50,
                    model="nram-gpt-oss-20b",
                )
            )

            assert result["model"] == "nram-gpt-oss-20b"


# ---------------------------------------------------------------------------
# Defect 8 FIXED: errors raise InferenceError, not fake 200
# ---------------------------------------------------------------------------

class TestDefect8Fixed_InferenceErrorRaised:
    def test_generate_response_raises_on_failure(self):
        """After fix: generate_response raises InferenceError instead of returning fake 200."""
        from core.llm_handler import LLMHandler, InferenceError
        import core.llm_handler as mod

        with patch.object(mod, "GuidanceHandler") as MockGH:
            mock_instance = MockGH.return_value
            mock_instance.process_with_guidance = AsyncMock(
                side_effect=RuntimeError("upstream failed")
            )

            handler = LLMHandler.__new__(LLMHandler)
            handler.guidance_handler = mock_instance
            handler.model = "test"
            handler.consciousness_levels = {"baseline": 0.3}

            with pytest.raises(InferenceError):
                asyncio.get_event_loop().run_until_complete(
                    handler.generate_response(
                        messages=[{"role": "user", "content": "Hi"}],
                        temperature=0.5,
                        max_tokens=50,
                    )
                )

    def test_inference_error_has_correct_fields(self):
        """InferenceError carries message and code."""
        from core.llm_handler import InferenceError

        err = InferenceError("test failure", code="upstream_inference_failed")
        assert err.message == "test failure"
        assert err.code == "upstream_inference_failed"
