"""
MAF Workflow REST API — multi-agent orchestration endpoints.

Exposes Microsoft Agent Framework workflows through the NRAM API:
  POST /v1/maf/concurrent   — parallel personas + synthesis
  POST /v1/maf/sequential   — chained personas
  GET  /v1/maf/personas     — list available personas
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.agentic.maf_client import MAF_AVAILABLE
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/maf", tags=["maf"])

# Lazy singleton orchestrator
_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        if not MAF_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="agent-framework is not installed in this environment.",
            )
        from core.agentic.maf_workflow import MAFWorkflowOrchestrator
        from core.agentic.persona_agents import PersonaAgentFactory

        factory = PersonaAgentFactory(
            nram_api_base="http://localhost:8000/v1",
            api_key="dev-nram-key",
        )
        _orchestrator = MAFWorkflowOrchestrator(factory)
    return _orchestrator


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ConcurrentRequest(BaseModel):
    question: str
    personas: Optional[List[str]] = None
    max_tokens: int = Field(200, ge=50, le=1000)
    synthesize: bool = True


class SequentialRequest(BaseModel):
    question: str
    personas: Optional[List[str]] = None
    max_tokens: int = Field(200, ge=50, le=1000)


class WorkflowResponse(BaseModel):
    workflow_id: str
    pattern: str
    final_output: str
    agent_outputs: Dict[str, str]
    latencies_ms: Dict[str, float]
    total_latency_ms: float
    memory_blocks_used: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/personas")
async def list_personas(current_user: dict = Depends(get_current_user)):
    """List available personas and their NRAM profiles."""
    from core.agentic.persona_agents import PERSONA_INSTRUCTIONS, PERSONA_PROFILES

    return {
        "personas": [
            {
                "name": name,
                "nram_profile": PERSONA_PROFILES[name],
                "description": PERSONA_INSTRUCTIONS[name][:120],
            }
            for name in PERSONA_PROFILES
        ]
    }


@router.post("/concurrent", response_model=WorkflowResponse)
async def run_concurrent(
    request: ConcurrentRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run personas in parallel, synthesize final answer."""
    orch = _get_orchestrator()
    try:
        result = await orch.run_concurrent(
            question=request.question,
            personas=request.personas,
            max_tokens=request.max_tokens,
            synthesize=request.synthesize,
        )
    except Exception as e:
        logger.exception("MAF concurrent workflow failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return WorkflowResponse(
        workflow_id=result.workflow_id,
        pattern=result.pattern,
        final_output=result.final_output,
        agent_outputs=result.agent_outputs,
        latencies_ms=result.latencies_ms,
        total_latency_ms=result.total_latency_ms,
        memory_blocks_used=result.memory_blocks_used,
    )


@router.post("/sequential", response_model=WorkflowResponse)
async def run_sequential(
    request: SequentialRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run personas in a sequential chain."""
    orch = _get_orchestrator()
    try:
        result = await orch.run_sequential(
            question=request.question,
            personas=request.personas,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        logger.exception("MAF sequential workflow failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return WorkflowResponse(
        workflow_id=result.workflow_id,
        pattern=result.pattern,
        final_output=result.final_output,
        agent_outputs=result.agent_outputs,
        latencies_ms=result.latencies_ms,
        total_latency_ms=result.total_latency_ms,
        memory_blocks_used=result.memory_blocks_used,
    )
