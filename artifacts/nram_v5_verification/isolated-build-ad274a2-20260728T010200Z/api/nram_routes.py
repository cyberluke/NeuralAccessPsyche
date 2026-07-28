"""
NRAM REST API — dedicated endpoints outside the OpenAI specification.

Endpoints:
  GET  /v1/nram/capabilities
  GET  /v1/nram/profiles
  GET  /v1/nram/phenomena

  POST /v1/nram/sessions
  GET  /v1/nram/sessions/{session_id}
  PATCH /v1/nram/sessions/{session_id}
  POST /v1/nram/sessions/{session_id}/reset
  DELETE /v1/nram/sessions/{session_id}

  GET  /v1/nram/sessions/{session_id}/state
  GET  /v1/nram/sessions/{session_id}/events
  GET  /v1/nram/sessions/{session_id}/events/stream

  POST /v1/nram/compare
  POST /v1/nram/analyze
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.nram.phenomena import (
    PHENOMENA,
    PHENOMENA_BY_ID,
    compile_phenomenon_config,
    get_phenomena_for_state,
)
from core.nram.session import (
    ConsciousnessState,
    DEFAULT_PHENOMENON_WEIGHTS,
    DEFAULT_PROFILES,
    MemoryConfig,
    NRAMSession,
    NRAMSessionState,
    SessionStore,
    session_store,
)
from utils.auth import get_current_user

nram_router = APIRouter(prefix="/v1/nram")


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    profile: ConsciousnessState = ConsciousnessState.NORMAL
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    auto_temperature: bool = True
    seed: Optional[int] = None
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    phenomenon_weights: Dict[str, float] = Field(default_factory=dict)


class SessionUpdateRequest(BaseModel):
    profile: Optional[ConsciousnessState] = None
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0)
    auto_temperature: Optional[bool] = None
    seed: Optional[int] = None
    memory: Optional[MemoryConfig] = None
    phenomenon_weights: Optional[Dict[str, float]] = None


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prompt: str
    model: str = "nram-qwen3-14b-awq"
    baseline_model: str = "qwen3-14b-awq-baseline"
    seed: Optional[int] = 271
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(500, ge=1, le=8192)
    nram: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    session_id: Optional[str] = None
    text: Optional[str] = None
    include_token_analysis: bool = False


# ---------------------------------------------------------------------------
# Capabilities, Profiles, Phenomena
# ---------------------------------------------------------------------------

@nram_router.get("/capabilities")
async def nram_capabilities(current_user: dict = Depends(get_current_user)):
    """Report NRAM capabilities, active engine, and verified controls."""
    from core.engines.registry import sglang_enabled

    return {
        "engine": "sglang" if sglang_enabled() else "legacy",
        "sglang_enabled": sglang_enabled(),
        "controls": {
            "token_biasing": True,
            "token_masking": True,
            "repetition_penalty": True,
            "dynamic_logits": True,
            "forced_token": True,
            "session_profile_snapshot": True,
            "session_memory_inference": False,
            "phenomena": True,
            "compare": True,
            "causal_event_streaming": False,
        },
        "verified": {
            "pre_sampling_logit_modification": True,  # Forced-token proof passed
            "streaming": True,
            "structured_output": True,
            "forced_token_proof": True,
        },
        "provenance": {
            "processor_log_events": "causal",
            "response_counters": "non_causal_accounting",
            "memory_recall": "not_implemented",
        },
        "note": "NRAM metrics are simulation and control metrics, "
                "NOT measurements of consciousness.",
    }


@nram_router.get("/profiles")
async def nram_profiles(current_user: dict = Depends(get_current_user)):
    """List all consciousness profiles with their default parameters."""
    profiles = {}
    for state in ConsciousnessState:
        profiles[state.value] = {
            "profile": state.value,
            "default_parameters": DEFAULT_PROFILES.get(state, {}),
            "default_phenomenon_weights": DEFAULT_PHENOMENON_WEIGHTS.get(state, {}),
        }
    return {"profiles": profiles}


@nram_router.get("/phenomena")
async def nram_phenomena(current_user: dict = Depends(get_current_user)):
    """List all NRAM phenomena with their definitions."""
    return {
        "phenomena": [
            {
                "id": p.id,
                "label_en": p.label_en,
                "label_cs": p.label_cs,
                "description": p.description,
                "steering_layer": p.steering_layer.value,
                "telemetry_event_type": p.telemetry_event_type,
                "default_weights": p.default_weights,
                "activation_threshold": p.activation_threshold,
                "allowed_phases": p.allowed_phases,
                "safety_limits": {
                    "max_bias_delta": p.max_bias_delta,
                    "max_mask_tokens": p.max_mask_tokens,
                    "max_fragment_length": p.max_fragment_length,
                    "cooldown_tokens": p.cooldown_tokens,
                },
                "provenance_origin": p.provenance_origin,
            }
            for p in PHENOMENA
        ],
        "note": "Phenomena are configurable cognitive effects, NOT decorative text. "
                "Each maps to a real steering mechanism.",
    }


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

@nram_router.post("/sessions")
async def create_session(
    request: SessionCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new NRAM session."""
    # Merge custom phenomenon weights with defaults for the profile
    default_weights = DEFAULT_PHENOMENON_WEIGHTS.get(request.profile, {})
    merged_weights = {**default_weights, **request.phenomenon_weights}

    session = NRAMSession(
        profile=request.profile,
        intensity=request.intensity,
        auto_temperature=request.auto_temperature,
        seed=request.seed,
        memory=request.memory,
        phenomenon_weights=merged_weights,
    )

    # Initialize state from profile defaults
    profile_params = DEFAULT_PROFILES.get(request.profile, {})
    session.state.influence = request.intensity
    session.state.coherence = profile_params.get("coherence_floor", 0.98)
    session.state.entropy = profile_params.get("associative_distance", 0.05) * request.intensity

    session_store.create(session)

    return {
        "id": session.id,
        "profile": session.profile.value,
        "revision": session.revision,
        "state": session.state.model_dump(),
    }


@nram_router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get a session by ID."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


@nram_router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update a session. Partial changes without resetting memory."""
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    session = session_store.update(session_id, updates)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": session.id,
        "profile": session.profile.value,
        "revision": session.revision,
        "state": session.state.model_dump(),
        "updated_fields": list(updates.keys()),
    }


@nram_router.post("/sessions/{session_id}/reset")
async def reset_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Reset all NRAM memory and state for a session."""
    session = session_store.reset(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": session.id,
        "profile": session.profile.value,
        "revision": session.revision,
        "state": session.state.model_dump(),
        "message": "Session state and all NRAM memory cleared.",
    }


@nram_router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a session."""
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "id": session_id}


# ---------------------------------------------------------------------------
# Session state and events
# ---------------------------------------------------------------------------

@nram_router.get("/sessions/{session_id}/state")
async def get_session_state(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get the current state of a session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": session.id,
        "state": session.state.model_dump(),
        "profile": session.profile.value,
        "intensity": session.intensity,
        "token_steering_counts": session.token_steering_counts,
        "phenomenon_counts": session.phenomenon_counts,
        "request_count": session.request_count,
        "note": "These are simulation and control metrics, NOT measurements of consciousness.",
    }


@nram_router.get("/sessions/{session_id}/events")
async def get_session_events(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get durable response correlations; processor events remain causal logs."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from api.routes import _session_runtime_events

    return {
        "session_id": session_id,
        "request_count": session.request_count,
        "token_steering_counts": session.token_steering_counts,
        "phenomenon_counts": session.phenomenon_counts,
        "events": list(_session_runtime_events.get(session_id, [])),
        "event_semantics": "response correlation/accounting; not synthetic causal attribution",
    }


@nram_router.get("/sessions/{session_id}/events/stream")
async def stream_session_events(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return a bounded SSE snapshot; real-time causal event streaming is unavailable."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        # Send initial state
        yield f"data: {json.dumps({'type': 'session_state', 'data': session.state.model_dump()})}\n\n"

        from api.routes import _session_runtime_events
        yield f"data: {json.dumps({'type': 'response_correlations', 'data': _session_runtime_events.get(session_id, [])})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Compare and Analyze
# ---------------------------------------------------------------------------

@nram_router.post("/compare")
async def compare(
    request: CompareRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run a paired baseline/NRAM comparison through the production engine."""
    # Import metrics before preflight/generation so a bad production image
    # fails without consuming either arm.
    from evaluation.metrics import compute_all
    from api.routes import ChatCompletionRequest, _to_engine_request
    from core.engines.registry import get_sglang_engine
    from utils.validators import validate_request

    engine = get_sglang_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail={"code": "engine_unavailable"})

    shared = {
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
        "stream": False,
        "tools": [],
    }
    baseline_public = ChatCompletionRequest(model=request.baseline_model, nram=None, **shared)
    controlled_options = {
        **request.nram,
        "enabled": True,
        "include_telemetry": True,
        "request_id": request.nram.get("request_id") or f"compare-{uuid.uuid4().hex}",
    }
    controlled_public = ChatCompletionRequest(model=request.model, nram=controlled_options, **shared)
    validate_request(baseline_public)
    validate_request(controlled_public)

    baseline_engine = _to_engine_request(
        baseline_public,
        baseline_public.messages,
        route_kind="nram.compare.baseline",
    )
    controlled_engine = _to_engine_request(
        controlled_public,
        controlled_public.messages,
        route_kind="nram.compare.controlled",
    )
    engine.validate(baseline_engine)
    engine.validate(controlled_engine)

    t0 = time.perf_counter()
    baseline_response = await engine.complete(baseline_engine)
    baseline_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    controlled_response = await engine.complete(controlled_engine)
    nram_ms = (time.perf_counter() - t0) * 1000

    baseline_content = baseline_response.choices[0].message.content
    nram_content = controlled_response.choices[0].message.content
    baseline_metrics = compute_all(baseline_content)
    nram_metrics = compute_all(nram_content)

    return {
        "request_id": f"compare-{uuid.uuid4().hex[:8]}",
        "prompt": request.prompt,
        "model": request.model,
        "baseline_model": request.baseline_model,
        "seed": request.seed,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "baseline": {
            "output": baseline_content,
            "latency_ms": round(baseline_ms, 1),
            "tokens": baseline_response.usage.completion_tokens,
            "finish_reason": baseline_response.choices[0].finish_reason,
            "metrics": baseline_metrics,
            "processor_intervened": False,
            "correlation": baseline_response.nram_correlation,
        },
        "nram": {
            "output": nram_content,
            "latency_ms": round(nram_ms, 1),
            "tokens": controlled_response.usage.completion_tokens,
            "finish_reason": controlled_response.choices[0].finish_reason,
            "metrics": nram_metrics,
            "profile": (controlled_public.nram or {}).get("profile", "normal"),
            "processor_intervened": True,
            "correlation": controlled_response.nram_correlation,
        },
    }


@nram_router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Analyze text or session for NRAM steering patterns."""
    if request.text:
        from evaluation.metrics import compute_all
        metrics = compute_all(request.text)
        return {
            "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "metrics": metrics,
            "note": "Heuristic analysis only. Not a measurement of consciousness.",
        }

    if request.session_id:
        session = session_store.get(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.id,
            "profile": session.profile.value,
            "state": session.state.model_dump(),
            "token_steering_counts": session.token_steering_counts,
            "phenomenon_counts": session.phenomenon_counts,
            "request_count": session.request_count,
        }

    raise HTTPException(status_code=400, detail="Provide either 'text' or 'session_id'")
