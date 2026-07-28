"""Regression coverage for the consolidated NRAM v5 runtime remediation."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from api.routes import (
    ChatCompletionRequest,
    _apply_session_snapshot,
    _complete_with_lifecycle,
    _record_session_result,
    _route_moe,
    _session_runtime_events,
)
from api.nram_routes import CompareRequest, compare
from core.contracts.nram_runtime import normalize_nram_options, validate_tokenizer_ids
from core.contracts.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest as EngineRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from core.engines.sglang_engine import SGLangEngine, SGLangEngineError
from core.nram.session import ConsciousnessState, NRAMSession, session_store
from utils.auth import verify_token
from utils.validators import validate_request


class FakeTokenizer:
    name_or_path = "fake-qwen-tokenizer"
    vocab_size = 100
    unk_token = "<unk>"
    unk_token_id = 0

    def __len__(self):
        return 110  # includes undefined padding slots that public IDs must reject

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "thank you": [1, 2],
            " thank you": [3, 2],
            "source text": [4, 5],
            " source text": [6, 5],
        }
        return mapping.get(text, [7])

    def convert_ids_to_tokens(self, token_id):
        return f"tok-{token_id}" if 0 <= token_id < self.vocab_size else None

    def decode(self, ids):
        return "".join(f"tok-{item}" for item in ids)


def make_engine() -> SGLangEngine:
    return SGLangEngine(
        base_url="http://sglang:30000/v1",
        model="nram-qwen3-14b-awq",
        tokenizer=FakeTokenizer(),
    )


def make_response(model="nram-qwen3-14b-awq", correlation=None) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="rid",
        created=1,
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        nram_correlation=correlation,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"unknown_nested": True},
        {"profile": "not-a-profile"},
        {"forced_token_enabled": True, "forced_token_id": 1.5},
        {"concept_strength": float("nan")},
        {"soft_token_injections": {"token_ids": [1]}},
        {"dexperts": True},
        {"forced_token_enabled": True, "forced_token_id": 3, "forbidden_token_ids": [3]},
    ],
)
def test_strict_nram_schema_rejects_confirmed_malformed_controls(options):
    with pytest.raises(ValidationError):
        normalize_nram_options(options)


def test_top_level_schema_and_sampling_are_rejected_before_dispatch():
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "model": "nram-qwen3-14b-awq",
                "messages": [{"role": "user", "content": "x"}],
                "unknown_top_level": True,
            }
        )
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "x"}],
            temperature=float("nan"),
        )


def test_tokenizer_undefined_padding_ids_rejected_for_every_sparse_control():
    tokenizer = FakeTokenizer()
    for options in (
        {"forced_token_id": 105},
        {"hard_token_schedule": [105]},
        {"soft_token_injections": [{"token_ids": [105]}]},
        {"vocabulary_logit_vectors": [{"entries": [{"token_id": 105}]}]},
    ):
        with pytest.raises(ValueError, match="not_defined_by_active_tokenizer"):
            validate_tokenizer_ids(options, tokenizer)


def test_local_auth_accepts_only_configured_token(monkeypatch):
    monkeypatch.setenv("NRAM_API_KEY", "configured-secret")
    assert verify_token("Bearer configured-secret")
    assert not verify_token("Bearer arbitrary-long-but-wrong-token")
    assert not verify_token("configured-secret")


def _finalized_hash(engine: SGLangEngine, **changes):
    values = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "prompt A\nline"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "normal",
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
            "forced_token_enabled": True,
            "forced_token_id": 9,
        },
        "runtime_request_id": "rid",
    }
    values.update(changes)
    request = EngineRequest(**values)
    payload = engine._build_upstream_payload(request, nram_enabled=True)
    engine._finalize_payload(request, payload, stream=False)
    return payload["custom_params"]["config_hash"]


def test_applied_state_hash_changes_for_behavior_and_normalizes_newlines():
    engine = make_engine()
    base = _finalized_hash(engine)
    variants = {
        _finalized_hash(engine, messages=[{"role": "user", "content": "prompt B\nline"}]),
        _finalized_hash(engine, seed=43),
        _finalized_hash(engine, temperature=0.7),
        _finalized_hash(engine, top_p=0.8),
        _finalized_hash(engine, response_format={"type": "json_object"}),
    }
    assert base not in variants
    assert len(variants) == 5
    assert base == _finalized_hash(
        engine,
        messages=[{"content": "prompt A\r\nline", "role": "user"}],
    )


def test_phrase_builder_adds_leading_and_nonleading_tokenizer_variants():
    engine = make_engine()
    request = EngineRequest(
        model="nram-qwen3-14b-awq",
        messages=[{"role": "user", "content": "x"}],
    )
    config = engine._build_phrase_constraint_config(
        request,
        {"forbidden_phrases": ["thank you"]},
    )
    assert [1, 2] in config["forbidden_phrase_ids"]
    assert [3, 2] in config["forbidden_phrase_ids"]


@pytest.mark.asyncio
async def test_nonstream_disconnect_aborts_real_scheduler_id_before_cancel():
    class Engine:
        def __init__(self):
            self.aborted = []

        async def complete(self, request):
            await asyncio.sleep(10)

        async def abort(self, rid, reason):
            self.aborted.append((rid, reason))

    class Disconnected:
        async def is_disconnected(self):
            return True

    engine = Engine()
    request = EngineRequest(
        model="nram-qwen3-14b-awq",
        messages=[{"role": "user", "content": "x"}],
        runtime_request_id="scheduler-rid",
    )
    with pytest.raises(Exception):
        await _complete_with_lifecycle(engine, request, Disconnected())
    assert engine.aborted == [("scheduler-rid", "client_disconnected")]


@pytest.mark.asyncio
async def test_moe_preserves_caller_bound_and_propagates_subrequest_error(monkeypatch):
    from api import routes

    class RecordingEngine:
        def __init__(self):
            self.calls = []

        async def complete(self, request):
            self.calls.append(request)
            if len(self.calls) == 2:
                raise SGLangEngineError("expert failed", status_code=503)
            return make_response()

    engine = RecordingEngine()
    monkeypatch.setattr(routes, "get_sglang_engine", lambda: engine)
    request = ChatCompletionRequest(
        model="nram-moe-orchestrator",
        messages=[{"role": "user", "content": "bounded"}],
        max_tokens=3,
        nram={"profile_logit_steering_enabled": False},
    )
    with pytest.raises(SGLangEngineError, match="threshold"):
        await _route_moe(request)
    assert [call.max_tokens for call in engine.calls] == [3, 3]
    assert all(call.nram["profile_logit_steering_enabled"] is False for call in engine.calls)


def test_session_snapshot_affects_request_and_records_real_response_accounting():
    session = NRAMSession(
        profile=ConsciousnessState.PEAK,
        intensity=1.0,
        phenomenon_weights={"insight": 0.8},
    )
    session_store.create(session)
    _session_runtime_events.pop(session.id, None)
    request = ChatCompletionRequest(
        model="nram-qwen3-14b-awq",
        messages=[{"role": "user", "content": "x"}],
        nram={"session_id": session.id},
    )
    validate_request(request)
    snapshot = _apply_session_snapshot(request)
    assert request.nram["profile"] == "peak"
    assert request.nram["intensity"] == 1.0
    assert request.nram["phenomenon_weights"]["insight"] == 0.8

    _record_session_result(
        snapshot,
        make_response(
            correlation={
                "scheduler_request_id": "rid-session",
                "config_hash": "hash-session",
                "sampled_token_ids": [9],
            }
        ),
    )
    current = session_store.get(session.id)
    assert current.request_count == 1
    assert current.token_steering_counts["model_generated"] == 1
    assert _session_runtime_events[session.id][0]["scheduler_request_id"] == "rid-session"
    assert _session_runtime_events[session.id][0]["causal_processor_event"] is False
    session_store.delete(session.id)


@pytest.mark.asyncio
async def test_comparison_preflights_both_arms_and_preserves_sampling(monkeypatch):
    from core.engines import registry

    class RecordingEngine:
        def __init__(self):
            self.events = []
            self.requests = []

        def validate(self, request):
            self.events.append(("validate", request.route_kind))

        async def complete(self, request):
            self.events.append(("complete", request.route_kind))
            self.requests.append(request)
            correlation = None
            if request.nram is not None:
                correlation = {
                    "scheduler_request_id": request.runtime_request_id,
                    "config_hash": "controlled-hash",
                    "sampled_token_ids": [9],
                }
            return make_response(model=request.public_model, correlation=correlation)

    engine = RecordingEngine()
    monkeypatch.setattr(registry, "get_sglang_engine", lambda: engine)
    result = await compare(
        CompareRequest(
            prompt="paired",
            max_tokens=2,
            temperature=0.2,
            top_p=0.9,
            seed=123,
            nram={
                "profile": "normal",
                "forced_token_enabled": True,
                "forced_token_id": 9,
            },
        ),
        current_user={"id": "local"},
    )
    assert engine.events[:2] == [
        ("validate", "nram.compare.baseline"),
        ("validate", "nram.compare.controlled"),
    ]
    baseline, controlled = engine.requests
    assert baseline.nram is None
    assert controlled.nram["forced_token_id"] == 9
    assert (baseline.max_tokens, baseline.temperature, baseline.top_p, baseline.seed) == (
        controlled.max_tokens,
        controlled.temperature,
        controlled.top_p,
        controlled.seed,
    )
    assert result["baseline"]["processor_intervened"] is False
    assert result["nram"]["processor_intervened"] is True
    assert result["nram"]["correlation"]["sampled_token_ids"] == [9]


@pytest.mark.asyncio
async def test_engine_stream_emits_public_identity_correlation_and_done(monkeypatch):
    from core.engines import sglang_engine as module

    class FakeResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield "data: " + json.dumps(
                {
                    "id": "upstream",
                    "model": "nram-qwen3-14b-awq",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

    class StreamContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def stream(self, *args, **kwargs):
            return StreamContext()

        async def post(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None)

    engine = make_engine()
    engine._client = FakeClient()
    monkeypatch.setattr(module, "build_rhetorical_plan", AsyncMock(return_value=None))
    request = EngineRequest(
        model="nram-qwen3-14b-awq",
        public_model="persona-peak",
        route_kind="persona.chat.completions",
        runtime_request_id="scheduler-stream",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=1,
        nram={
            "enabled": True,
            "profile": "peak",
            "request_id": "application-id",
            "include_telemetry": True,
        },
    )
    output = [chunk async for chunk in engine.stream(request)]
    assert output[-1] == b"data: [DONE]\n\n"
    assert output.count(b"data: [DONE]\n\n") == 1
    first = json.loads(output[0].decode().removeprefix("data: "))
    assert first["model"] == "persona-peak"
    assert first["nram_correlation"]["scheduler_request_id"] == "scheduler-stream"
    assert first["nram_correlation"]["actual_base_model"] == "nram-qwen3-14b-awq"
    assert first["nram_correlation"]["sample_join"] == "unavailable_for_streaming"
