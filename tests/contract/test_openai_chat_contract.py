"""
Contract tests for the OpenAI-compatible /v1/chat/completions endpoint.

These tests define the CORRECT behavior the API must exhibit after Phase 1 fixes.
All tests must pass after the fixes are applied.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# A valid Bearer token: starts with "Bearer " and len > 10
AUTH_HEADER = {"Authorization": "Bearer dev-nram-key-12345678"}
# An invalid token: too short (len <= 10)
BAD_AUTH_HEADER = {"Authorization": "Bearer x"}


@pytest.fixture
def client():
    """Create a test client with the llm_handler instance patched at module level."""
    # Patch the singleton instance, not the class — the instance is created at import time
    with patch("api.routes.llm_handler") as mock_llm, \
         patch("api.routes.nram") as mock_nram, \
         patch("api.routes.config_suggester"):

        # NRAM must NOT modify messages (defect 9 fix)
        mock_nram.process_messages = MagicMock(side_effect=lambda msgs: list(msgs))
        mock_nram.broadcast_state = AsyncMock()

        # LLM returns a proper OpenAI-shaped response
        mock_llm.generate_response = AsyncMock(return_value={
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        })

        from main import app
        yield TestClient(app), mock_llm, mock_nram


# ---------------------------------------------------------------------------
# Auth contract
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_auth_returns_401(self, client):
        """Requests without Authorization header must return 401."""
        c, _, _ = client
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client):
        """Requests with a too-short/invalid token must return 401."""
        c, _, _ = client
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=BAD_AUTH_HEADER,
        )
        assert resp.status_code == 401

    def test_valid_token_returns_200(self, client):
        """Requests with a valid Bearer token must return 200."""
        c, _, _ = client
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Request validation contract
# ---------------------------------------------------------------------------

class TestRequestValidation:
    def test_missing_messages_returns_400(self, client):
        """Empty messages list must return 400."""
        c, _, _ = client
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "nram-gpt-oss-20b", "messages": []},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 400

    def test_invalid_message_format_returns_400(self, client):
        """Messages without role/content must return 400."""
        c, _, _ = client
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"bad_field": "no role or content"}],
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Error handling contract — no fake 200 on upstream failure
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_upstream_failure_returns_openai_error_not_fake_200(self, client):
        """When the inference engine fails, the API must return an OpenAI-shaped error, not fake 200."""
        from core.llm_handler import InferenceError

        c, mock_llm, _ = client
        mock_llm.generate_response = AsyncMock(
            side_effect=InferenceError("SGLang upstream unreachable")
        )

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=AUTH_HEADER,
        )

        assert resp.status_code != 200, (
            "API must not return HTTP 200 when inference engine fails"
        )
        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        assert body["error"]["type"] == "inference_error"
        assert body["error"]["code"] == "upstream_inference_failed"


# ---------------------------------------------------------------------------
# User content preservation contract
# ---------------------------------------------------------------------------

class TestContentPreservation:
    def test_user_content_not_modified_by_nram(self, client):
        """NRAM must not alter the user's message content before forwarding to the model."""
        c, _, mock_nram = client

        original_content = "Present a new AI learning device for children that removes traditional menus."

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": original_content}],
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200

        # NRAM.process_messages must return messages unchanged
        call_args = mock_nram.process_messages.call_args[0][0]
        assert call_args[0]["content"] == original_content, (
            "NRAM must not modify user message content"
        )


# ---------------------------------------------------------------------------
# Model alias contract
# ---------------------------------------------------------------------------

class TestModelAlias:
    def test_requested_model_passed_to_llm_handler(self, client):
        """The requested model alias must be forwarded to the LLM handler."""
        c, mock_llm, _ = client

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200

        call_kwargs = mock_llm.generate_response.call_args[1]
        assert call_kwargs.get("model") == "nram-gpt-oss-20b"

    def test_baseline_model_alias_works(self, client):
        """The baseline alias must also work."""
        c, mock_llm, _ = client

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-r1-qwen-7b-baseline",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Response shape contract
# ---------------------------------------------------------------------------

class TestResponseShape:
    def test_response_has_required_openai_fields(self, client):
        """Response must have id, object, created, model, choices, usage."""
        c, _, _ = client

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert body["object"] == "chat.completion"
        assert "created" in body
        assert "model" in body
        assert "choices" in body
        assert "usage" in body
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)

    def test_content_is_string_not_dict(self, client):
        """Response content must be a string, not a raw dict from GuidanceHandler."""
        c, _, _ = client

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "nram-gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=AUTH_HEADER,
        )
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str), (
            f"Content must be a string, got {type(content)}"
        )
