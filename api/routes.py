from fastapi import APIRouter, HTTPException, Depends, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any
from core.llm_handler import LLMHandler, InferenceError
from core.nram import NRAM
try:
    from core.config_suggester import NRAMConfigSuggester
except ImportError as _config_suggester_import_error:  # optional diagnostic dependency
    _CONFIG_SUGGESTER_IMPORT_ERROR = str(_config_suggester_import_error)

    class NRAMConfigSuggester:  # type: ignore[no-redef]
        def __init__(self):
            self._error = _CONFIG_SUGGESTER_IMPORT_ERROR

        async def analyze_performance(self, state):
            raise RuntimeError(f"Config suggester unavailable: {self._error}")

        def update_config(self, config):
            raise RuntimeError(f"Config suggester unavailable: {self._error}")
from core.engines.registry import (
    get_sglang_engine,
    should_route_to_sglang,
    sglang_enabled,
)
from core.engines.sglang_engine import SGLangEngineError
from core.contracts.openai import ChatCompletionRequest as EngineRequest
from core.tools.searxng_client import SearXNGClient
from utils.validators import validate_request
from utils.auth import get_current_user
from utils.api_logger import api_metrics
from core.steering.dexperts_feature import dexperts_enabled
from core.contracts.observability import NRAMStreamEvent, ObservabilityLevel, StreamEventType
import logging
import json
import asyncio
import time
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")
llm_handler = LLMHandler()
nram = NRAM()
templates = Jinja2Templates(directory="templates")
config_suggester = NRAMConfigSuggester()
searxng_client = SearXNGClient()

# Public enumeration contains immutable loaded model identities only. Virtual
# routes remain accepted but are disclosed separately by capabilities.
MODEL_ALIASES = {
    "nram-qwen3-14b-awq": "nram-qwen3-14b-awq",
}

_BASELINE_ALIASES = {"qwen3-14b-awq-baseline"}

# Persona model -> NRAM profile mapping
_PERSONA_MODEL_MAP = {
    "persona-normal": "normal",
    "persona-microdose": "microdose",
    "persona-threshold": "threshold",
    "persona-psychedelic": "psychedelic",
    "persona-peak": "peak",
    "persona-dissociative": "dissociative",
    "persona-keynote": "visionary-psychedelic-keynote",
}

# MoE orchestrator personas (which personas to query)
_MOE_DEFAULT_PERSONAS = [
    "normal", "threshold", "psychedelic", "visionary-psychedelic-keynote",
]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(100, ge=1, le=8192)
    stream: Optional[bool] = False
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    # NRAM extension — optional, validated server-side only
    nram: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def _openai_error(status_code: int, message: str, err_type: str, code: str) -> JSONResponse:
    """Return an OpenAI-shaped error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


def _is_persona_model(model: str) -> bool:
    """Check if model is a persona model that routes to NRAM profiles."""
    return model in _PERSONA_MODEL_MAP


def _is_moe_model(model: str) -> bool:
    """Check if model is the MoE orchestrator."""
    return model == "nram-moe-orchestrator"


def _validate_model_identity(model: str) -> None:
    accepted = set(MODEL_ALIASES) | _BASELINE_ALIASES | set(_PERSONA_MODEL_MAP) | {"nram-moe-orchestrator"}
    if sglang_enabled() and model not in accepted:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "model_not_loaded",
                "param": "model",
                "message": f"Model {model!r} is not backed by the loaded checkpoint.",
            },
        )


def _apply_session_snapshot(request: ChatCompletionRequest) -> Optional[Any]:
    """Apply one immutable session snapshot to this request, if requested."""
    options = request.nram or {}
    session_id = options.get("session_id")
    if not session_id:
        return None
    from core.nram.session import session_store

    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "session_id": session_id})
    snapshot = session.model_copy(deep=True)
    options["profile"] = snapshot.profile.value
    options["intensity"] = snapshot.intensity
    from core.contracts.nram_runtime import PHENOMENA

    options["phenomenon_weights"] = {
        key: value
        for key, value in snapshot.phenomenon_weights.items()
        if key in PHENOMENA
    }
    omitted = sorted(set(snapshot.phenomenon_weights) - set(PHENOMENA))
    if omitted:
        _session_runtime_events.setdefault(snapshot.id, []).append(
            {
                "type": "unsupported_session_controls_omitted",
                "controls": omitted,
                "causal_processor_event": False,
            }
        )
    request.nram = options
    if request.seed is None and snapshot.seed is not None:
        request.seed = snapshot.seed
    return snapshot


_session_runtime_events: Dict[str, List[Dict[str, Any]]] = {}


def _record_session_result(snapshot: Optional[Any], response: Any) -> None:
    """Record actual response accounting; do not synthesize processor origins."""
    if snapshot is None:
        return
    usage = response.usage if hasattr(response, "usage") else None
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    snapshot_live = None
    from core.nram.session import session_store

    snapshot_live = session_store.get(snapshot.id)
    if snapshot_live is None:
        return
    snapshot_live.update_state_after_request(
        completion_tokens=completion_tokens,
        phenomena_activated=[],
        steering_events={"model_generated": completion_tokens},
    )
    correlation = getattr(response, "nram_correlation", None) or {}
    _session_runtime_events.setdefault(snapshot.id, []).append(
        {
            "type": "response_correlation",
            "causal_processor_event": False,
            "scheduler_request_id": correlation.get("scheduler_request_id"),
            "config_hash": correlation.get("config_hash"),
            "sampled_token_ids": correlation.get("sampled_token_ids", []),
            "completion_tokens": completion_tokens,
        }
    )


async def _wait_for_disconnect(raw_request: Any) -> bool:
    """Wait for the ASGI disconnect event without depending on response cleanup."""
    receive = getattr(raw_request, "receive", None)
    if callable(receive):
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return True

    # Small test doubles and older Request implementations may expose only
    # is_disconnected(). Production Starlette requests use the receive path.
    while True:
        if await raw_request.is_disconnected():
            return True
        await asyncio.sleep(0.01)


async def _complete_with_lifecycle(
    engine: Any,
    engine_request: EngineRequest,
    raw_request: Optional[Request] = None,
):
    """Race non-stream completion against a real ASGI disconnect event."""
    task = asyncio.create_task(engine.complete(engine_request))
    if raw_request is None:
        return await task

    watcher = asyncio.create_task(_wait_for_disconnect(raw_request))
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        # A response that completes in the same event-loop turn wins; completed
        # requests must never receive a spurious abort.
        if task in done:
            return await task
        if watcher.result():
            await engine.abort(engine_request.runtime_request_id, reason="client_disconnected")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise HTTPException(status_code=499, detail={"code": "client_disconnected"})
        return await task
    except asyncio.CancelledError:
        if not task.done():
            await engine.abort(engine_request.runtime_request_id, reason="downstream_cancelled")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def _stream_with_lifecycle(
    engine: Any,
    engine_request: EngineRequest,
    raw_request: Optional[Request],
):
    """Relay SSE while independently monitoring disconnect and closing upstream."""
    upstream = engine.stream(engine_request).__aiter__()
    disconnect = (
        asyncio.create_task(_wait_for_disconnect(raw_request))
        if raw_request is not None
        else None
    )
    next_chunk: Optional[asyncio.Task[Any]] = None
    completed = False
    aborted = False

    async def abort_once(reason: str) -> None:
        nonlocal aborted
        if aborted:
            return
        aborted = True
        await engine.abort(engine_request.runtime_request_id, reason=reason)

    try:
        while True:
            next_chunk = asyncio.create_task(anext(upstream))
            if disconnect is None:
                done, _ = await asyncio.wait({next_chunk})
            else:
                done, _ = await asyncio.wait(
                    {next_chunk, disconnect},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            if next_chunk in done:
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    completed = True
                    break
                if chunk.strip() == b"data: [DONE]":
                    completed = True
                yield chunk
                if completed:
                    break
                continue

            await abort_once("client_disconnected")
            next_chunk.cancel()
            await asyncio.gather(next_chunk, return_exceptions=True)
            break
    except asyncio.CancelledError:
        if not completed:
            await abort_once("stream_cancelled")
        raise
    finally:
        if disconnect is not None:
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)
        if next_chunk is not None and not next_chunk.done():
            next_chunk.cancel()
            await asyncio.gather(next_chunk, return_exceptions=True)
        if not completed:
            await abort_once("stream_closed")
        await upstream.aclose()


class _LifecycleStreamingResponse(StreamingResponse):
    """Streaming response whose body iterator exclusively owns ASGI receive.

    Starlette's standard response starts a second disconnect listener on older
    ASGI specs. That listener can consume ``http.disconnect`` before the route
    lifecycle wrapper can target and abort the scheduler request. This response
    delegates disconnect handling to ``_stream_with_lifecycle`` and always
    closes the iterator when socket send fails.
    """

    async def __call__(self, scope, receive, send) -> None:
        try:
            await self.stream_response(send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if callable(close):
                await close()
            if self.background is not None:
                await self.background()


async def _route_persona(
    request: ChatCompletionRequest,
    model: str,
    raw_request: Optional[Request] = None,
    session_snapshot: Optional[Any] = None,
):
    """Route a persona-model request to SGLang with NRAM opts injected."""
    validate_request(request)
    engine = get_sglang_engine()
    if engine is None:
        raise InferenceError("SGLang engine not available", code="engine_unavailable")

    profile = _PERSONA_MODEL_MAP[model]

    # State-dependent intensity: each state gets a different intensity level
    # This creates qualitatively different outputs, not just quantitative differences
    intensity_map = {
        "normal": 0.1,           # Minimal steering, structured analysis
        "microdose": 0.35,       # Subtle pattern recognition
        "threshold": 0.58,       # Bold associative leaps
        "psychedelic": 0.84,     # Extraordinary synthesis
        "peak": 1.0,             # Maximum revolutionary insights
        "dissociative": 0.70,    # Radical deconstruction
    }
    intensity = intensity_map.get(profile, 0.5)

    # Inject NRAM options from the persona profile
    # Preserve include_telemetry from original request
    include_telemetry = request.nram.get("include_telemetry", False) if request.nram else False

    nram_opts = {
        "enabled": True,
        "profile": profile,
        "intensity": intensity,
        "include_telemetry": include_telemetry,
    }
    if request.nram:
        nram_opts.update(request.nram)
    internal = request.model_copy(deep=True)
    internal.nram = nram_opts
    internal.model = "nram-qwen3-14b-awq"
    engine_request = _to_engine_request(
        internal,
        internal.messages,
        public_model=model,
        route_kind="persona.chat.completions",
    )
    if request.stream:
        return _LifecycleStreamingResponse(
            _sse_observability(_stream_with_lifecycle(engine, engine_request, raw_request), request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    response = await _complete_with_lifecycle(engine, engine_request, raw_request)
    _record_session_result(session_snapshot, response)
    result = response.model_dump()
    result["model"] = model
    return result


async def _route_moe(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Bounded sequential persona orchestration; this is not model-level MoE."""
    validate_request(request)
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail={"code": "moe_streaming_not_supported", "message": "Orchestration is non-streaming."},
        )
    engine = get_sglang_engine()
    if engine is None:
        raise InferenceError("SGLang engine not available", code="engine_unavailable")

    user_content = ""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    async def query_persona(profile: str):
        sub_request = _to_engine_request(request, request.messages)
        sub_request.model = "nram-qwen3-14b-awq"
        sub_request.public_model = "nram-moe-orchestrator"
        sub_request.route_kind = f"persona-orchestration.expert.{profile}"
        sub_request.nram = {
            **(request.nram or {}),
            "enabled": True,
            "profile": profile,
        }
        sub_request.max_tokens = request.max_tokens
        try:
            result = await engine.complete(sub_request)
            return (profile, result.choices[0].message.content)
        except Exception as exc:
            raise SGLangEngineError(
                f"Persona orchestration failed for {profile}: {exc}",
                status_code=getattr(exc, "status_code", 502),
            ) from exc

    # Deterministic single-request runtime: issue persona calls sequentially.
    # This route is an orchestration workflow, not DExperts or model-level MoE.
    persona_results = []
    for persona in _MOE_DEFAULT_PERSONAS:
        persona_results.append(await query_persona(persona))

    # Build synthesis prompt — respect user's max_tokens for strategic analysis
    persona_summaries = []
    for profile, output in persona_results:
        # Keep FULL content for rich synthesis — no truncation
        persona_summaries.append(f"[{profile} perspective]:\n{output}")

    synthesis_max_tokens = request.max_tokens

    synthesis_input = (
        f"Question: {user_content[:500]}\n\n"
        "Perspectives from multiple analytical lenses:\n" + "\n".join(persona_summaries) +
        f"\n\nSynthesize these perspectives into a comprehensive, strategic response. "
        f"Integrate the best insights into a coherent analysis. Be thorough, substantive, and detailed. "
        f"Provide concrete examples, data points, and actionable recommendations."
    )

    # CRITICAL FIX: Synthesis uses NRAM-enabled model to preserve persona characteristics
    # Using "visionary-peak" profile for balanced synthesis with steering
    synth_messages = [
        {"role": "system", "content": "You are a strategic synthesizer. Integrate multiple analytical perspectives into a comprehensive, well-structured response. Be thorough, substantive, insightful, and detailed. Provide concrete examples and actionable recommendations."},
        {"role": "user", "content": synthesis_input},
    ]
    synth_request = _to_engine_request(request, synth_messages)
    synth_request.model = "nram-qwen3-14b-awq"
    synth_request.public_model = "nram-moe-orchestrator"
    synth_request.route_kind = "persona-orchestration.synthesis"
    synth_request.nram = {
        **(request.nram or {}),
        "enabled": True,
        "profile": "visionary-peak",
    }
    synth_request.max_tokens = synthesis_max_tokens
    synth_request.temperature = 0.5

    synth_response = await engine.complete(synth_request)
    result = synth_response.model_dump()
    result["model"] = "nram-moe-orchestrator"
    return result

@router.get("/visualize", response_class=HTMLResponse)
async def visualize_nram(request: Request):
    """Serve the NRAM visualization page"""
    return templates.TemplateResponse(
        "visualization.html",
        {"request": request}
    )

@router.websocket("/ws/nram-state")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time NRAM state updates"""
    await nram.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        nram.disconnect(websocket)

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Handle chat completion requests — OpenAI-compatible, no fake 200 on failure."""
    try:
        validate_request(request)
        _validate_model_identity(request.model)
        session_snapshot = _apply_session_snapshot(request)

        # Observability is an explicit per-request opt-in. Minimal mode is
        # intentionally represented by absence of telemetry fields upstream.
        if request.nram and request.nram.get("observability_level"):
            try:
                level = ObservabilityLevel(str(request.nram["observability_level"]).lower())
            except ValueError:
                raise HTTPException(status_code=400, detail={"code": "invalid_observability_level"})
            if level != ObservabilityLevel.MINIMAL:
                request.nram["include_telemetry"] = True
                request.nram.setdefault("request_id", f"nram-{uuid.uuid4().hex}")
                request.nram["telemetry_top_k"] = min(20 if level == ObservabilityLevel.TRACE else 5,
                                                       int(request.nram.get("telemetry_top_k", 5)))

        # Reject client-supplied processor injection attempts
        nram_opts = request.nram or {}
        _FORBIDDEN_FIELDS = {
            "custom_logit_processor", "serialized_processor",
            "processor_class", "python_code",
        }
        if _FORBIDDEN_FIELDS & set(nram_opts.keys()):
            return _openai_error(
                400,
                "Forbidden field in nram options: processor injection is not allowed.",
                "invalid_request_error",
                "forbidden_field",
            )

        # Route persona models (e.g., persona-peak -> NRAM peak profile)
        if _is_persona_model(request.model):
            return await _route_persona(
                request,
                request.model,
                raw_request=raw_request,
                session_snapshot=session_snapshot,
            )

        # Route MoE orchestrator model
        if _is_moe_model(request.model):
            return await _route_moe(request)

        # Process through NRAM (state tracking only — messages returned unchanged)
        messages = nram.process_messages(request.messages)

        # Broadcast updated state for visualization
        await nram.broadcast_state()

        # Route to SGLang engine when feature flag is active for this model
        if should_route_to_sglang(request.model):
            return await _handle_sglang(
                request,
                messages,
                raw_request=raw_request,
                session_snapshot=session_snapshot,
            )

        # Legacy path below
        # FIX defect 6: handle streaming requests
        if request.stream:
            return await _stream_completion(request, messages)

        # FIX defect 5: pass requested model alias
        response = await llm_handler.generate_response(
            messages=messages,
            temperature=request.temperature if request.temperature is not None else 1.0,
            max_tokens=request.max_tokens if request.max_tokens is not None else 256,
            model=request.model,
        )

        return response

    except SGLangEngineError as e:
        logger.error(f"SGLang engine error: {e.message}")
        return _openai_error(e.status_code, e.message, "inference_error", "upstream_inference_failed")

    except InferenceError as e:
        logger.error(f"Inference error: {e.message}")
        # FIX defect 8: return OpenAI-shaped error, not a fake 200
        return _openai_error(502, e.message, "inference_error", e.code)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return _openai_error(500, str(e), "inference_error", "upstream_inference_failed")


def _to_engine_request(
    request: ChatCompletionRequest,
    messages: list,
    *,
    public_model: Optional[str] = None,
    route_kind: str = "chat.completions",
) -> EngineRequest:
    """Convert the route request model into the engine contract request."""
    return EngineRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        response_format=request.response_format,
        tools=request.tools,
        tool_choice=request.tool_choice,
        nram=request.nram,
        runtime_request_id=f"nram-scheduler-{uuid.uuid4().hex}",
        public_model=public_model or request.model,
        route_kind=route_kind,
    )


def _sse_observability(source: Any, request: ChatCompletionRequest):
    """Add opt-in NRAM events alongside, never instead of, OpenAI SSE chunks."""
    raw_level = (request.nram or {}).get("observability_level")
    if not raw_level:
        return source
    try:
        level = ObservabilityLevel(str(raw_level).lower())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_observability_level"}) from exc
    if level == ObservabilityLevel.MINIMAL:
        return source

    async def generate():
        started = time.perf_counter()
        request_id = (request.nram or {}).get("request_id") or f"nram-{uuid.uuid4().hex}"
        method = (request.nram or {}).get("method") or (request.nram or {}).get("profile")
        sequence = 0

        def make_event(event_type: StreamEventType, payload: Dict[str, Any]) -> bytes:
            nonlocal sequence
            item = NRAMStreamEvent(
                request_id=request_id,
                sequence_number=sequence,
                monotonic_timestamp_ms=(time.perf_counter() - started) * 1000,
                event_type=event_type,
                method=method,
                observability_level=level,
                payload=payload,
            )
            sequence += 1
            return ("event: nram\ndata: " + item.model_dump_json() + "\n\n").encode()

        yield make_event(StreamEventType.REQUEST_STARTED, {"availability": "enabled"})
        position = 0
        last_ms = 0.0
        try:
            async for raw in source:
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                for line in text.splitlines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:].strip())
                    except json.JSONDecodeError:
                        yield make_event(StreamEventType.WARNING, {"message": "malformed_upstream_event"})
                        continue
                    for choice in chunk.get("choices", []):
                        fragment = (choice.get("delta") or {}).get("content")
                        if fragment:
                            now_ms = (time.perf_counter() - started) * 1000
                            yield make_event(StreamEventType.TOKEN, {
                                "token_position": position,
                                "token_id": None,
                                "token_text": fragment,
                                "emitted_at_ms": now_ms,
                                "inter_token_latency_ms": None if position == 0 else now_ms - last_ms,
                                "classification": "STOCHASTIC_OR_UNCLASSIFIED",
                                "availability_reason": "streaming_upstream_does_not_expose_sampled_token_ids",
                            })
                            position += 1
                            last_ms = now_ms
                yield raw
            yield make_event(StreamEventType.REQUEST_COMPLETED, {"token_count": position})
        except Exception as exc:
            yield make_event(StreamEventType.REQUEST_FAILED, {"message": str(exc)[:500]})
            raise

    return generate()


async def _stream_completion(request: ChatCompletionRequest, messages: list):
    """Adapt the legacy non-SGLang handler to a valid one-chunk SSE stream."""
    async def generate():
        response = await llm_handler.generate_response(
            messages=messages,
            temperature=request.temperature if request.temperature is not None else 1.0,
            max_tokens=request.max_tokens if request.max_tokens is not None else 256,
            model=request.model,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        chunk = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4()}"),
            "object": "chat.completion.chunk",
            "created": response.get("created", int(time.time())),
            "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _handle_sglang(
    request: ChatCompletionRequest,
    messages: list,
    *,
    raw_request: Optional[Request] = None,
    session_snapshot: Optional[Any] = None,
):
    """Route a request through the SGLang engine with tool call support."""
    engine = get_sglang_engine()
    if engine is None:
        return _openai_error(
            503,
            "SGLang engine is enabled but not available.",
            "inference_error",
            "engine_unavailable",
        )

    tools = request.tools

    engine_request = _to_engine_request(request, messages)
    engine_request.tools = tools

    if request.stream:
        return _LifecycleStreamingResponse(
            _sse_observability(_stream_with_lifecycle(engine, engine_request, raw_request), request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await _complete_with_lifecycle(engine, engine_request, raw_request)
    _record_session_result(session_snapshot, response)
    response_dict = response.model_dump()

    # Check if model wants to call a tool
    if response_dict.get("choices") and response_dict["choices"][0].get("message", {}).get("tool_calls"):
        tool_calls = response_dict["choices"][0]["message"]["tool_calls"]

        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            if tool_call["function"]["name"] == "searxng_search":
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args.get("query", "")
                    search_type = args.get("search_type", "general")

                    if search_type == "market_research":
                        result = await searxng_client.market_research(query)
                    elif search_type == "company_analysis":
                        result = await searxng_client.company_analysis(query)
                    elif search_type == "technology_trends":
                        result = await searxng_client.technology_trends(query)
                    else:
                        result = await searxng_client.search(query)

                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                except Exception as e:
                    logger.error(f"Tool call error: {e}")
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps({"error": str(e)})
                    })

        # Add tool results to messages and get final response
        messages_with_tools = messages + [response_dict["choices"][0]["message"]] + tool_results
        engine_request.messages = messages_with_tools
        engine_request.tools = None  # Don't allow nested tool calls

        final_response = await engine.complete(engine_request)
        return final_response.model_dump()

    return response_dict

@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models — OpenAI-compatible /v1/models."""
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "neuralaccesspsyche",
                "permission": [],
            }
            for alias in MODEL_ALIASES
        ],
    }


@router.get("/nram/capabilities")
async def nram_capabilities(current_user: dict = Depends(get_current_user)):
    """Report the active engine and which steering controls are verified with evidence states."""
    engine_active = sglang_enabled()
    
    # Evidence states: CAUSALLY_PROVEN, EXECUTING, CONFIGURED, NOT_VERIFIED
    # Based on actual test results from tests/unit/test_nram_v5_phase2_runtime.py
    capabilities = {
        "engine": "sglang" if engine_active else "legacy",
        "sglang_enabled": engine_active,
        "loaded_model_ids": list(MODEL_ALIASES.keys()),
        "virtual_routes": {
            "baseline_aliases": sorted(_BASELINE_ALIASES),
            "persona_profiles": dict(_PERSONA_MODEL_MAP),
            "persona_orchestration": "nram-moe-orchestrator",
        },
        "actual_base_model": "nram-qwen3-14b-awq",
        "controls": {
            "prompt_steering": True,
            "hard_token_masking": True,
            "soft_logit_biasing": True,
            "dynamic_repetition_penalty": True,
            "phrase_and_source_masks": True,
            "entropy_pid": True,
            "vocabulary_logit_vectors": True,
            # NRAM v5 representation control - evidence-based reporting
            "activation_addition": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "hidden_state_vector_addition",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestActivationAddition",
                "proof_tests": [
                    "test_zero_strength_zero_delta",
                    "test_nonzero_strength_measured_delta",
                    "test_sign_reversal_reverses_delta_direction",
                    "test_increasing_strength_dose_response",
                ],
            },
            "multi_vector_representation": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "multi_mode_vector_combination",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestMultiVector",
                "proof_tests": [
                    "test_sum_mode",
                    "test_normalized_mode_differs_from_sum",
                    "test_orthogonalized_mode_differs",
                    "test_norm_budgeted_mode_clamps",
                ],
            },
            "conceptor_steering": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "low_rank_subspace_projection",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestConceptor",
                "proof_tests": [
                    "test_aperture_zero_is_identity",
                    "test_changing_aperture_changes_projection",
                    "test_conceptor_weights_formula",
                ],
            },
            "hidden_state_probes": {
                "state": "CONFIGURED",
                "runtime_wired": True,
                "mechanism": "hidden_state_linear_classification",
                "note": "Probe infrastructure implemented, awaiting trained probe artifacts",
            },
            "latent_closed_loop": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "pid_controller_probe_to_intervention",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestLatentClosedLoop",
                "proof_tests": [
                    "test_probe_score_changes_over_time",
                    "test_controller_action_changes_after_score",
                    "test_disabled_feedback_fixed_strength",
                    "test_state_does_not_leak_between_requests",
                ],
            },
            "semantic_closed_loop": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "chunk_level_semantic_evaluation",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestSemanticClosedLoop",
                "proof_tests": [
                    "test_iteration_recorded",
                    "test_max_iterations_terminates",
                ],
            },
            "branch_tournament": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "multi_branch_generation_and_scoring",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestBranchTournament",
                "proof_tests": [
                    "test_multiple_branches_generated",
                    "test_all_branches_scored",
                    "test_winner_has_highest_score",
                    "test_tournament_result_includes_evidence",
                ],
            },
            "batch_context": {
                "state": "CAUSALLY_PROVEN",
                "runtime_wired": True,
                "mechanism": "request_scoped_batch_aware_context",
                "last_proof_artifact": "tests/unit/test_nram_v5_phase2_runtime.py::TestBatchContext",
                "proof_tests": [
                    "test_register_and_lookup",
                    "test_request_isolation",
                    "test_complete_request_cleanup",
                    "test_mask_building",
                ],
            },
        },
        "verified": {
            "pre_sampling_logit_modification": True,
            "forced_token_proof": True,
            "streaming": True,
            "representation_control": True,
            "closed_loop_steering": True,
            "batch_aware_context": True,
            "request_isolation": True,
        },
        "blocked_not_implemented": [
            "full_causal_statistical_study",
        ],
        "telemetry": {
            "processor_events": "causal_log_events",
            "nonstream_sample_join": "sglang_meta_info",
            "stream_sample_token_ids": "not_available_without_supported_post_sample_hook",
        },
    }
    if dexperts_enabled():
        capabilities["controls"]["dexperts"] = {
            "state": "NOT_IMPLEMENTED",
            "runtime_wired": False,
            "mechanism": "expert_anti_expert_logit_combination",
            "note": "DExperts is opt-in via NRAM_DEXPERTS_ENABLED and remains subject to its existing runtime readiness gates.",
        }
    capabilities["methods"] = [
        name for name, value in capabilities["controls"].items()
        if isinstance(value, bool) and value
        or isinstance(value, dict) and value.get("runtime_wired") is True
    ]
    capabilities["observability"] = {
        "schema_version": "nram.stream-observability.v1",
        "levels": [level.value for level in ObservabilityLevel],
        "default": "minimal",
        "telemetry_opt_in": True,
        "stream_token_ids": False,
        "stream_token_ids_reason": "upstream_sglang_stream_does_not_expose_sampled_token_ids",
        "top_k_max": {"standard": 5, "research": 5, "trace": 20},
        "event_types": [event.value for event in StreamEventType],
    }
    return capabilities


@router.get("/nram/token-policy/{profile}")
async def nram_token_policy(
    profile: str,
    current_user: dict = Depends(get_current_user),
):
    """Diagnostics: show token IDs, decoded text, skipped reasons, and bias values.

    Requires authentication. Returns the compiled policy for a persona profile.
    """
    from core.persona.profiles import PROFILES
    from core.persona.compiler import compile_policy
    from core.steering.tokenizer_bias import TokenBiasCompiler

    state = PROFILES.get(profile)
    if state is None:
        return _openai_error(404, f"Unknown profile: {profile}", "invalid_request_error", "unknown_profile")

    policy = compile_policy(state)

    engine = get_sglang_engine()
    tokenizer = getattr(engine, "_tokenizer", None) if engine else None

    if tokenizer is None:
        return {
            "profile": profile,
            "policy": policy.model_dump(),
            "tokens": None,
            "note": "No tokenizer loaded (set NRAM_TOKENIZER_PATH). Showing lexemes only.",
        }

    compiler = TokenBiasCompiler(tokenizer)
    compiled, diagnostics = compiler.compile_with_diagnostics(policy)
    return {
        "profile": profile,
        "policy": policy.model_dump(),
        "compiled": compiled.model_dump(),
        "diagnostics": diagnostics,
    }

@router.get("/visualize/api", response_class=HTMLResponse)
async def visualize_api(request: Request):
    """Serve the API visualization dashboard"""
    return templates.TemplateResponse(
        "api_visualization.html",
        {"request": request}
    )

@router.websocket("/ws/api-viz")
async def api_visualization_websocket(websocket: WebSocket):
    """WebSocket endpoint for API visualization"""
    await api_metrics.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        api_metrics.disconnect(websocket)

@router.get("/nram/explorer", response_class=HTMLResponse)
async def nram_explorer(request: Request):
    """Serve the NRAM architecture explorer page"""
    try:
        return templates.TemplateResponse(
            "nram_explorer.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Error serving NRAM explorer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/nram-explorer")
async def nram_explorer_websocket(websocket: WebSocket):
    """WebSocket endpoint for NRAM architecture exploration"""
    await websocket.accept()
    try:
        while True:
            # Check for client messages (e.g., reset command)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("action") == "reset":
                    nram.reset_state()
            except json.JSONDecodeError:
                pass

            # Get current NRAM state
            state = nram.get_explorer_state()

            # Send state update
            await websocket.send_json(state)

            # Brief delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket error in NRAM explorer: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@router.get("/nram/config/analyze")
async def analyze_nram_config(current_user: dict = Depends(get_current_user)):
    """Get NRAM configuration analysis and suggestions"""
    try:
        # Get current state from NRAM
        state = nram.get_explorer_state()

        # Analyze performance and get suggestions
        analysis = await config_suggester.analyze_performance(state)

        return analysis
    except Exception as e:
        logger.error(f"Error analyzing NRAM configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nram/config/update")
async def update_nram_config(
    config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Update NRAM configuration based on suggestions"""
    try:
        # Update config suggester
        new_config = config_suggester.update_config(config)

        # Update NRAM with new configuration
        nram.update_configuration(new_config)

        return {"message": "Configuration updated successfully", "config": new_config}
    except Exception as e:
        logger.error(f"Error updating NRAM configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
