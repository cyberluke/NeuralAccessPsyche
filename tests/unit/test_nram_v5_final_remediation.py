"""Focused regressions for final RVR-002/003/004 remediation."""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from api import routes
from api.routes import (
    ChatCompletionRequest,
    _complete_with_lifecycle,
    _stream_with_lifecycle,
    _to_engine_request,
)
from core.contracts.nram_runtime import (
    EntropyPhaseTargets,
    NRAMOptions,
    PhenomenonWeights,
    SoftTokenInjection,
    VocabularyLogitVector,
    VocabularyVectorEntry,
    normalize_nram_options,
)
from core.contracts.openai import ChatCompletionRequest as EngineRequest
from core.engines.sglang_engine import SGLangEngine
from scripts.collect_nram_v5_artifacts import (
    CollectionError,
    _parse_source_mapping,
    collect_byte_domains,
    collect_git_identity,
    validate_manifest_schema,
)


PUBLIC_BASE = {
    "model": "nram-qwen3-14b-awq",
    "messages": [{"role": "user", "content": "x"}],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("temperature", "top_p", "frequency_penalty", "presence_penalty")
        for value in (float("nan"), float("inf"), float("-inf"))
    ],
)
def test_public_nonfinite_sampling_is_stable_422_and_never_calls_engine(
    monkeypatch, field, value
):
    calls = []

    class RecordingEngine:
        async def complete(self, request):
            calls.append(request)
            raise AssertionError("validation must reject before dispatch")

    monkeypatch.setattr(routes, "get_sglang_engine", lambda: RecordingEngine())
    body = {**PUBLIC_BASE, field: value}
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            content=json.dumps(body, allow_nan=True),
            headers={"Authorization": "Bearer dev-nram-key", "Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request_schema"
    assert calls == []
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    ("field", "accepted", "rejected"),
    [
        ("temperature", (0.0, 2.0), (-0.0001, 2.0001)),
        ("top_p", (0.0, 1.0), (-0.0001, 1.0001)),
        ("frequency_penalty", (-2.0, 2.0), (-2.0001, 2.0001)),
        ("presence_penalty", (-2.0, 2.0), (-2.0001, 2.0001)),
    ],
)
def test_public_sampling_boundaries(field, accepted, rejected):
    for value in accepted:
        ChatCompletionRequest.model_validate({**PUBLIC_BASE, field: value})
    for value in rejected:
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate({**PUBLIC_BASE, field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nested_nonfinite_is_machine_readable_400_with_zero_engine_calls(monkeypatch, value):
    calls = []

    class RecordingEngine:
        async def complete(self, request):
            calls.append(request)

    monkeypatch.setattr(routes, "get_sglang_engine", lambda: RecordingEngine())
    body = {**PUBLIC_BASE, "nram": {"concept_strength": value}}
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            content=json.dumps(body, allow_nan=True),
            headers={"Authorization": "Bearer dev-nram-key", "Content-Type": "application/json"},
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request_schema"
    assert error["errors"][0]["code"] == "finite_number"
    assert calls == []


@pytest.mark.parametrize(
    "field",
    [
        "intensity", "vocabulary_logit_vector_clip", "entropy_kp", "entropy_ki",
        "entropy_kd", "entropy_integral_limit", "entropy_scale_min", "entropy_scale_max",
        "concept_strength", "visionary_intensity", "contrarian_force", "product_obsession",
        "human_focus", "rhetorical_compression", "associative_distance", "theatricality",
        "emotional_voltage", "coherence_floor", "novelty_target", "repetition_penalty",
        "corporate_jargon_penalty",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_every_top_level_nram_float_rejects_nonfinite(field, value):
    with pytest.raises(ValidationError):
        normalize_nram_options({field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nested_nram_float_models_reject_nonfinite(value):
    for factory in (
        lambda: EntropyPhaseTargets(extraction=value),
        lambda: PhenomenonWeights(insight=value),
        lambda: SoftTokenInjection(token_ids=[1], bias=value),
        lambda: VocabularyVectorEntry(token_id=1, weight=value),
        lambda: VocabularyLogitVector(
            vector_id="v", coefficient=value, entries=[{"token_id": 1, "weight": 1.0}]
        ),
    ):
        with pytest.raises(ValidationError):
            factory()


class ReceiveDisconnect:
    def __init__(self):
        self.messages: asyncio.Queue[dict[str, str]] = asyncio.Queue()

    async def receive(self):
        return await self.messages.get()


@pytest.mark.asyncio
async def test_nonstream_asgi_disconnect_aborts_once_and_cancels_local_work():
    class Engine:
        def __init__(self):
            self.aborts = []
            self.cancelled = False

        async def complete(self, request):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def abort(self, rid, reason):
            self.aborts.append((rid, reason))

    engine = Engine()
    request = EngineRequest(
        model="nram-qwen3-14b-awq",
        messages=[{"role": "user", "content": "x"}],
        runtime_request_id="scheduler-nonstream",
    )
    receive = ReceiveDisconnect()
    task = asyncio.create_task(_complete_with_lifecycle(engine, request, receive))
    await receive.messages.put({"type": "http.disconnect"})
    with pytest.raises(Exception) as exc_info:
        await task
    assert getattr(exc_info.value, "status_code", None) == 499
    assert engine.aborts == [("scheduler-nonstream", "client_disconnected")]
    assert engine.cancelled is True


@pytest.mark.asyncio
async def test_nonstream_normal_completion_has_no_abort_or_lingering_receive():
    class Engine:
        def __init__(self):
            self.aborts = []

        async def complete(self, request):
            return "done"

        async def abort(self, rid, reason):
            self.aborts.append((rid, reason))

    engine = Engine()
    receive = ReceiveDisconnect()
    request = EngineRequest(model="m", messages=[], runtime_request_id="scheduler-normal")
    assert await _complete_with_lifecycle(engine, request, receive) == "done"
    assert engine.aborts == []


@pytest.mark.asyncio
async def test_stream_disconnect_and_generator_close_abort_exactly_once():
    class Engine:
        def __init__(self):
            self.aborts = []
            self.closed = 0

        async def stream(self, request):
            try:
                yield b"data: first\n\n"
                await asyncio.Event().wait()
            finally:
                self.closed += 1

        async def abort(self, rid, reason):
            self.aborts.append((rid, reason))

    engine = Engine()
    receive = ReceiveDisconnect()
    request = EngineRequest(model="m", messages=[], runtime_request_id="scheduler-stream")
    relay = _stream_with_lifecycle(engine, request, receive)
    assert await anext(relay) == b"data: first\n\n"
    await receive.messages.put({"type": "http.disconnect"})
    with pytest.raises(StopAsyncIteration):
        await anext(relay)
    await relay.aclose()
    assert engine.aborts == [("scheduler-stream", "client_disconnected")]
    assert engine.closed == 1


@pytest.mark.asyncio
async def test_stream_normal_done_has_no_abort():
    class Engine:
        def __init__(self):
            self.aborts = []

        async def stream(self, request):
            yield b"data: first\n\n"
            yield b"data: [DONE]\n\n"

        async def abort(self, rid, reason):
            self.aborts.append((rid, reason))

    engine = Engine()
    request = EngineRequest(model="m", messages=[], runtime_request_id="scheduler-done")
    chunks = [item async for item in _stream_with_lifecycle(engine, request, None)]
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert engine.aborts == []


@pytest.mark.asyncio
async def test_official_abort_transport_is_idempotent_per_scheduler_id():
    class Client:
        def __init__(self):
            self.calls = []

        async def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return SimpleNamespace(raise_for_status=lambda: None)

    engine = object.__new__(SGLangEngine)
    engine._client = Client()
    engine._abort_url = "http://sglang:30000/abort_request"
    engine._abort_lock = asyncio.Lock()
    engine._aborted_request_ids = {}
    await asyncio.gather(
        engine.abort("scheduler-id", "first"),
        engine.abort("scheduler-id", "second"),
    )
    assert len(engine._client.calls) == 1
    assert engine._client.calls[0][1]["json"]["rid"] == "scheduler-id"


def test_immediate_retry_gets_fresh_scheduler_identity():
    public = ChatCompletionRequest.model_validate(PUBLIC_BASE)
    first = _to_engine_request(public, public.messages)
    retry = _to_engine_request(public, public.messages)
    assert first.runtime_request_id != retry.runtime_request_id


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Collector Test")
    (repo / "spec.md").write_bytes(b"line one\nline two\n")
    _git(repo, "add", "spec.md")
    _git(repo, "commit", "-m", "spec")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_collector_git_clean_dirty_branch_and_detached_fixtures(tmp_path):
    repo, commit = _repository(tmp_path)
    clean = collect_git_identity(repo)
    assert clean["implementation_commit"] == commit
    assert clean["branch"] == "main" and clean["detached"] is False
    assert clean["dirty"] is False

    (repo / "spec.md").write_bytes(b"dirty\n")
    (repo / "excluded.txt").write_text("generated", encoding="utf-8")
    dirty = collect_git_identity(repo, excluded_paths=["excluded.txt"])
    assert dirty["dirty"] is True
    assert any("spec.md" in line for line in dirty["included_dirty_status"])
    assert any("excluded.txt" in line for line in dirty["excluded_dirty_status"])

    _git(repo, "checkout", "--", "spec.md")
    (repo / "excluded.txt").unlink()
    _git(repo, "checkout", "--detach", commit)
    detached = collect_git_identity(repo)
    assert detached["branch"] is None and detached["detached"] is True


def test_collector_distinguishes_git_lf_checkout_crlf_and_normalization(tmp_path):
    repo, commit = _repository(tmp_path)
    (repo / "spec.md").write_bytes(b"line one\r\nline two\r\n")
    identity = collect_byte_domains(repo, commit, "spec.md")
    assert identity["git_blob_sha256"] != identity["checkout_bytes_sha256"]
    assert identity["git_blob_sha256"] == identity["lf_normalized_checkout_sha256"]
    assert identity["git_blob_bytes"] == 18
    assert identity["checkout_bytes"] == 20


def test_collector_has_no_hard_coded_prior_implementation_sha():
    source = Path("scripts/collect_nram_v5_artifacts.py").read_text(encoding="utf-8")
    assert "7aa12854800327ea905e608b9ececc3e3f5fd758" not in source
    assert "ad274a2c4b0575cf796659073fb69a962dbb66d0" not in source


def test_collector_fails_loudly_for_missing_git_prerequisite(tmp_path):
    with pytest.raises(CollectionError, match="command failed"):
        collect_git_identity(tmp_path)


def test_collector_requires_explicit_container_to_repository_source_mapping():
    assert _parse_source_mapping("nram-api=/app/api/routes.py=api/routes.py") == (
        "nram-api", "/app/api/routes.py", "api/routes.py"
    )
    with pytest.raises(CollectionError, match="SERVICE=CONTAINER_FILE=REPOSITORY_FILE"):
        _parse_source_mapping("nram-api=/app/api/routes.py")


def test_manifest_schema_supports_distinct_optional_evidence_commit():
    manifest = {
        "schema_version": "nram.v5.evidence-manifest.v2",
        "run_id": "run",
        "implementation_commit": "a" * 40,
        "implementation_tree_oid": "b" * 40,
        "evidence_commit": "c" * 40,
        "git": {},
        "specification": {},
        "environment": {},
        "containers": [],
        "source_hashes": [],
        "model_artifact": {},
        "tokenizer_artifact": {},
        "runtime": {},
        "launch_command": "python -m sglang.launch_server",
        "configuration_sha256": "d" * 64,
        "run_inputs": {},
        "commands": [],
        "tests": {},
        "artifacts": {},
    }
    validate_manifest_schema(manifest)
    manifest["implementation_commit"] = "placeholder"
    with pytest.raises(CollectionError, match="implementation_commit"):
        validate_manifest_schema(manifest)
