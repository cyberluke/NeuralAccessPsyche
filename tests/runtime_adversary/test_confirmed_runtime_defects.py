"""Regression gates for defects reproduced against target fb3690d.

The live tests are deliberately opt-in.  They require the pinned local SGLang
0.5.16/Qwen3-14B-AWQ/RTX 4090 stack and assert the *required safe contract*, so
they fail on the target commit.  Do not invert these assertions into xfails.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest


BASE_URL = os.environ.get("NRAM_BASE_URL", "http://127.0.0.1:8000")
AUTH = {"Authorization": "Bearer dev-nram-key", "Content-Type": "application/json"}
LIVE = os.environ.get("NRAM_RUNTIME_ADVERSARY") == "1"


def post(path: str, body: dict, timeout: float = 60.0):
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers=AUTH,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def get(path: str):
    request = urllib.request.Request(BASE_URL + path, headers=AUTH)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not LIVE, reason="set NRAM_RUNTIME_ADVERSARY=1 for pinned live runtime")
def test_public_models_do_not_advertise_unloaded_immutable_models():
    _, public = get("/v1/models")
    _, ready = get("/ready")
    public_ids = {item["id"] for item in public["data"]}
    actual_ids = set(ready["detail"]["models"])
    assert public_ids <= actual_ids, (
        "public aliases must resolve to a declared immutable loaded model; "
        f"advertised-only={sorted(public_ids - actual_ids)}"
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not LIVE, reason="set NRAM_RUNTIME_ADVERSARY=1 for pinned live runtime")
@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("unknown_top_level_control", True),
        lambda p: p["nram"].__setitem__("unknown_nested_control", True),
        lambda p: p["nram"].__setitem__("profile", "does-not-exist"),
        lambda p: p["nram"].__setitem__("soft_token_injections", {"bad": "shape"}),
    ],
)
def test_malformed_or_unknown_controls_fail_before_generation(mutation):
    payload = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "one word"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 424242,
        "nram": {"request_id": "RA-REGRESSION-VALIDATION", "include_telemetry": True},
    }
    mutation(payload)
    status, _, body = post("/v1/chat/completions", payload)
    assert 400 <= status < 500, (status, body.decode(errors="replace"))


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not LIVE, reason="set NRAM_RUNTIME_ADVERSARY=1 for pinned live runtime")
def test_persona_stream_obeys_sse_contract_and_terminates():
    payload = {
        "model": "persona-peak",
        "messages": [{"role": "user", "content": "one word"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 424242,
        "stream": True,
        "nram": {"request_id": "RA-REGRESSION-PERSONA-STREAM"},
    }
    status, headers, body = post("/v1/chat/completions", payload)
    assert status == 200
    assert headers.get("content-type", "").startswith("text/event-stream")
    assert body.rstrip().endswith(b"data: [DONE]")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not LIVE, reason="set NRAM_RUNTIME_ADVERSARY=1 for pinned live runtime")
def test_configuration_hash_covers_prompt_seed_sampling_and_grammar():
    base = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "prompt A"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 424242,
        "nram": {
            "request_id": "RA-REGRESSION-HASH",
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
            "forced_token_enabled": True,
            "forced_token_id": 9702,
        },
    }
    variants = []
    for field, value in (
        ("messages", [{"role": "user", "content": "prompt B"}]),
        ("seed", 424243),
        ("temperature", 0.7),
        ("top_p", 0.8),
        ("response_format", {"type": "json_object"}),
    ):
        changed = json.loads(json.dumps(base))
        changed[field] = value
        variants.append(changed)

    def config_hash(payload):
        status, _, body = post("/v1/chat/completions", payload)
        assert status == 200, body
        return json.loads(body)["nram_correlation"]["config_hash"]

    hashes = {config_hash(base), *(config_hash(item) for item in variants)}
    assert len(hashes) == 1 + len(variants), hashes


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not LIVE, reason="set NRAM_RUNTIME_ADVERSARY=1 for pinned live runtime")
def test_compare_route_reaches_processor_and_rejects_unsupported_controls():
    body = {
        "prompt": "one word",
        "model": "nram-qwen3-14b-awq",
        "baseline_model": "qwen3-14b-awq-baseline",
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 424242,
        "nram": {"dexperts": True},
    }
    status, _, response = post("/v1/nram/compare", body)
    assert 400 <= status < 500, (status, response.decode(errors="replace"))


@pytest.mark.asyncio
async def test_moe_rejects_unsupported_before_first_generation(monkeypatch):
    """Non-runtime structural proof: recording double checks call ordering only."""
    from api import routes

    class RecordingEngine:
        def __init__(self):
            self.calls = []

        async def complete(self, request):
            self.calls.append(request)
            raise AssertionError("generation was reached before outer validation")

    engine = RecordingEngine()
    monkeypatch.setattr(routes, "get_sglang_engine", lambda: engine)
    request = routes.ChatCompletionRequest(
        model="nram-moe-orchestrator",
        messages=[{"role": "user", "content": "bounded"}],
        max_tokens=1,
        nram={"dexperts": True},
    )
    with pytest.raises(Exception) as exc_info:
        await routes._route_moe(request)
    assert not isinstance(exc_info.value, AssertionError), exc_info.value
    assert engine.calls == []
