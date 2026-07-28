"""
Contract tests for the OpenAI-compatible /v1/models endpoint.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


AUTH_HEADER = {"Authorization": "Bearer dev-nram-key"}


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

    def test_models_returns_only_loaded_immutable_identity(self, client):
        """GET /v1/models must not advertise virtual or unloaded checkpoints."""
        resp = client.get("/v1/models", headers=AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body

        model_ids = {m["id"] for m in body["data"]}
        assert model_ids == {"nram-qwen3-14b-awq"}
        assert "nram-gpt-oss-20b" not in model_ids
        assert "nram-deepseek-r1-qwen-7b" not in model_ids
        assert "persona-peak" not in model_ids

    def test_models_have_required_fields(self, client):
        """Each model entry must have id, object, owned_by."""
        resp = client.get("/v1/models", headers=AUTH_HEADER)
        body = resp.json()
        for model in body["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "owned_by" in model
