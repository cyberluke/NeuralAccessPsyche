"""
Shared pytest fixtures for NeuralAccessPsyche test suite.
"""
import sys
import os
import types
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy env vars so OpenAI client doesn't crash on import
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-for-unit-tests")
os.environ.setdefault("AI_INTEGRATIONS_OPENROUTER_API_KEY", "test-dummy-key")
os.environ.setdefault("NRAM_API_KEY", "dev-nram-key")

# Stub out the 'guidance' module if it's not installed (unit tests don't need it)
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
    """Ensure env vars are set for every test."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key-for-unit-tests")
    monkeypatch.setenv("AI_INTEGRATIONS_OPENROUTER_API_KEY", "test-dummy-key")
    monkeypatch.setenv("NRAM_API_KEY", "dev-nram-key")
