"""
Contract tests for the OpenAI-compatible /v1/models endpoint.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


AUTH_HEADER = {"Authorization": "Bearer dev-nram-key-12345678"}


@pytest.fixture
def client():
    with patch("api.routes.LLMHandler"), \
         patch("api.routes.NRAM"), \
         patch("api.routes.NRAMConfigSuggester"):
        from main import app
        yield TestClient(app)


class TestModelsEndpoint:
    def test_models_requires_auth(self, client):
        """GET /v1/models without auth must return 401."""
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_models_returns_both_aliases(self, client):
        """GET /v1/models must list both baseline and NRAM aliases."""
        resp = client.get("/v1/models", headers=AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body

        model_ids = {m["id"] for m in body["data"]}
        assert "nram-gpt-oss-20b" in model_ids, (
            "nram-gpt-oss-20b alias must be listed"
        )
        assert "deepseek-r1-qwen-7b-baseline" in model_ids or "gpt-oss-20b-baseline" in model_ids, (
            "baseline alias must be listed"
        )

    def test_models_have_required_fields(self, client):
        """Each model entry must have id, object, owned_by."""
        resp = client.get("/v1/models", headers=AUTH_HEADER)
        body = resp.json()
        for model in body["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "owned_by" in model
