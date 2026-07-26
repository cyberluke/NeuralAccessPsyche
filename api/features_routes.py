"""
Features 6/7/8 REST API — batch completions, telemetry, session<->memory linkage.

Endpoints:
  POST /v1/features/batch                      (Feature 6)
  GET  /v1/features/telemetry/{session_id}     (Feature 7)
  GET  /v1/features/telemetry                  (Feature 7 - global)
  GET  /v1/features/provenance-map             (Feature 3 surface)
  POST /v1/features/memory                     (Feature 1/8 - write shared memory)
  GET  /v1/features/memory                     (Feature 8 - list memory blocks)
  GET  /v1/features/memory/context/{agent}     (Feature 8 - per-agent compiled context)
  POST /v1/features/sessions/{session_id}/memory  (Feature 8 - link session<->memory)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from core.agentic.memory import (
    InMemoryLettaBackend,
    MemoryBlock,
    MemoryPolicy,
    LettaContextProvider,
)
from core.nram.batch import BatchItem, BatchRequest, BatchResponse
from core.nram.provenance_map import full_map as provenance_full_map
from core.nram.telemetry_agg import telemetry_aggregator
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

features_router = APIRouter(prefix="/v1/features", tags=["features"])

# Shared gateway base URL (route batch through NRAM gateway so steering +
# telemetry aggregation fire for every completion).
_GATEWAY_URL = "http://localhost:8000/v1/chat/completions"

# Shared in-process memory plane + context provider (persisted via SqliteLettaBackend
# when a path is supplied; defaults to in-memory for zero-config runs).
_memory_backend: Optional[InMemoryLettaBackend] = None
_context_provider: Optional[LettaContextProvider] = None
# session_id -> list of memory block ids (session<->memory linkage)
_session_memory_links: Dict[str, List[str]] = {}


def _get_memory() -> InMemoryLettaBackend:
    global _memory_backend, _context_provider
    if _memory_backend is None:
        try:
            from core.agentic.memory import SqliteLettaBackend
            _memory_backend = SqliteLettaBackend()
        except Exception:  # fall back to in-memory (no /data volume)
            _memory_backend = InMemoryLettaBackend()
        _context_provider = LettaContextProvider(_memory_backend)
    return _memory_backend


def _get_provider() -> LettaContextProvider:
    _get_memory()
    assert _context_provider is not None
    return _context_provider


# ---------------------------------------------------------------------------
# Feature 6: Batch completions
# ---------------------------------------------------------------------------

@features_router.post("/batch", response_model=BatchResponse)
async def batch_completions(
    request: BatchRequest,
    authorization: str = Header(None),
    current_user: dict = Depends(get_current_user),
):
    """Process multiple prompts concurrently through the NRAM gateway."""
    headers = {
        "Authorization": authorization or "Bearer dev-nram-key",
        "Content-Type": "application/json",
    }

    async def _one(index: int, prompt: str, client: httpx.AsyncClient) -> BatchItem:
        body = {
            "model": request.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
            "nram": {"enabled": True, "profile": request.profile, "intensity": request.intensity},
        }
        t0 = time.perf_counter()
        try:
            resp = await client.post(_GATEWAY_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return BatchItem(
                index=index, prompt=prompt, output=content,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                model=data.get("model", request.model),
            )
        except Exception as e:
            logger.warning("batch item %s failed: %s", index, e)
            return BatchItem(
                index=index, prompt=prompt, output="",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                model=request.model, error=str(e),
            )

    t_total = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        items = await asyncio.gather(
            *[_one(i, p, client) for i, p in enumerate(request.prompts)]
        )

    succeeded = sum(1 for it in items if it.error is None)
    return BatchResponse(
        total=len(items),
        succeeded=succeeded,
        failed=len(items) - succeeded,
        total_latency_ms=round((time.perf_counter() - t_total) * 1000, 1),
        items=list(items),
    )


# ---------------------------------------------------------------------------
# Feature 7: Telemetry
# ---------------------------------------------------------------------------

@features_router.get("/telemetry")
async def global_telemetry(current_user: dict = Depends(get_current_user)):
    """Global telemetry stats across all sessions."""
    return {
        "global": telemetry_aggregator.global_stats(),
        "sessions": telemetry_aggregator.all_summaries(),
    }


@features_router.get("/telemetry/{session_id}")
async def session_telemetry(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Per-session telemetry aggregation."""
    summary = telemetry_aggregator.summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="No telemetry for session")
    return summary


# ---------------------------------------------------------------------------
# Feature 3 surface: provenance map
# ---------------------------------------------------------------------------

@features_router.get("/provenance-map")
async def get_provenance_map(current_user: dict = Depends(get_current_user)):
    """Phenomenon -> steering-layer -> provenance-origin mapping."""
    return provenance_full_map()


# ---------------------------------------------------------------------------
# Features 1/8: Shared memory management + session linkage
# ---------------------------------------------------------------------------

class MemoryWriteRequest(BaseModel):
    name: str
    content: str
    memory_class: str = "working"  # canonical | episodic | working | overlay | audit
    shared_with: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None


@features_router.post("/memory")
async def write_memory(
    request: MemoryWriteRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a shared memory block (enforces shared_with on read)."""
    backend = _get_memory()
    block = MemoryBlock(
        name=request.name,
        memory_class=request.memory_class,
        content=request.content,
        shared_with=request.shared_with,
        read_only=(request.memory_class == "canonical"),
    )
    block = backend.create_block(block)

    # Feature 8: link block to session if provided
    if request.session_id:
        _session_memory_links.setdefault(request.session_id, []).append(block.id)

    return {"block": block.model_dump(mode="json"), "session_linked": request.session_id}


@features_router.get("/memory")
async def list_memory(current_user: dict = Depends(get_current_user)):
    """List all memory blocks with visibility metadata."""
    backend = _get_memory()
    blocks = backend.list_all()
    return {
        "count": len(blocks),
        "blocks": [b.model_dump(mode="json") for b in blocks],
    }


@features_router.get("/memory/context/{agent_name}")
async def compiled_context(
    agent_name: str,
    memory_class: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Compile the memory context visible to a specific agent (shared_with enforced)."""
    provider = _get_provider()
    backend = _get_memory()

    # Build policy from all block names (visibility filter handles shared_with)
    names = {b.name for b in backend.list_all()}
    policy = MemoryPolicy(
        canonical_blocks=list(names),
        working_blocks=list(names),
        episodic_blocks=list(names),
    )
    compiled = await provider.provide_context(policy, agent_name=agent_name)
    return {
        "agent_name": agent_name,
        "messages": compiled.messages,
        "block_ids_used": compiled.block_ids_used,
        "total_tokens": compiled.total_tokens,
        "synthetic_memories": compiled.synthetic_memories,
    }


@features_router.post("/sessions/{session_id}/memory")
async def link_session_memory(
    session_id: str,
    request: MemoryWriteRequest,
    current_user: dict = Depends(get_current_user),
):
    """Feature 8: create a memory block tied to a session's lifecycle."""
    request.session_id = session_id
    return await write_memory(request, current_user)


@features_router.get("/sessions/{session_id}/memory")
async def get_session_memory(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Feature 8: list memory blocks linked to a session."""
    backend = _get_memory()
    block_ids = _session_memory_links.get(session_id, [])
    blocks = [backend.get_block(bid) for bid in block_ids]
    blocks = [b for b in blocks if b is not None]
    return {
        "session_id": session_id,
        "count": len(blocks),
        "blocks": [b.model_dump(mode="json") for b in blocks],
    }
