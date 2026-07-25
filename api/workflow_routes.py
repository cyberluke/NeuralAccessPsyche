"""
Agentic Workflow REST API.

Endpoints:
  POST   /v1/nram/workflows
  GET    /v1/nram/workflows/{workflow_id}

  POST   /v1/nram/workflows/{workflow_id}/run
  POST   /v1/nram/workflows/{workflow_id}/pause
  POST   /v1/nram/workflows/{workflow_id}/resume
  POST   /v1/nram/workflows/{workflow_id}/cancel

  GET    /v1/nram/workflows/{workflow_id}/artifacts
  GET    /v1/nram/workflows/{workflow_id}/events
  GET    /v1/nram/workflows/{workflow_id}/events/stream
  GET    /v1/nram/workflows/{workflow_id}/report

  POST   /v1/nram/workflows/{workflow_id}/approve
  POST   /v1/nram/workflows/{workflow_id}/reject
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agentic.orchestrator import InnovationOrchestrator
from core.agentic.workflow_store import workflow_store
from utils.auth import get_current_user

workflow_router = APIRouter(prefix="/v1/nram/workflows")

# Global orchestrator instance
_orchestrator = InnovationOrchestrator()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class WorkflowCreateRequest(BaseModel):
    workflow_type: str = "codebase_innovation"
    repository: Dict[str, str] = Field(
        default_factory=lambda: {"path": "/workspace/NeuralAccessPsyche", "revision": "HEAD"}
    )
    goal: str
    constraints: List[str] = Field(default_factory=list)
    excluded_ideas: List[str] = Field(default_factory=list)
    workflow_seed: int = 271
    max_iterations: int = 2
    require_human_approval_before_final_selection: bool = True


# ---------------------------------------------------------------------------
# Workflow CRUD
# ---------------------------------------------------------------------------

@workflow_router.post("")
async def create_workflow(
    request: WorkflowCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new innovation workflow."""
    state = await _orchestrator.create_workflow(
        repository_path=request.repository.get("path", "/workspace"),
        repository_revision=request.repository.get("revision", "HEAD"),
        user_goal=request.goal,
        constraints=request.constraints,
        excluded_ideas=request.excluded_ideas,
        workflow_seed=request.workflow_seed,
        max_iterations=request.max_iterations,
        require_human_approval=request.require_human_approval_before_final_selection,
    )
    return {
        "workflow_id": state.workflow_id,
        "status": state.status.value,
        "phase": state.phase,
        "created_at": state.created_at.isoformat(),
    }


@workflow_router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get workflow state."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return state.model_dump()


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------

@workflow_router.post("/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Run (or resume) a workflow. Returns immediately; check status via GET."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Run asynchronously in background
    asyncio.create_task(_orchestrator.run_workflow(workflow_id))

    return {
        "workflow_id": workflow_id,
        "status": "running",
        "message": "Workflow started. Poll GET /v1/nram/workflows/{id} for status.",
    }


@workflow_router.post("/{workflow_id}/pause")
async def pause_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Pause a running workflow."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    from core.agentic.contracts import WorkflowStatus
    state.status = WorkflowStatus.PAUSED
    workflow_store.update_workflow(state)
    return {"workflow_id": workflow_id, "status": "paused"}


@workflow_router.post("/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Resume a paused workflow."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    from core.agentic.contracts import WorkflowStatus
    if state.status != WorkflowStatus.PAUSED:
        raise HTTPException(status_code=400, detail=f"Workflow is not paused (status={state.status.value})")

    asyncio.create_task(_orchestrator.run_workflow(workflow_id))
    return {"workflow_id": workflow_id, "status": "resuming"}


@workflow_router.post("/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel a workflow."""
    await _orchestrator.cancel_workflow(workflow_id)
    return {"workflow_id": workflow_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

@workflow_router.post("/{workflow_id}/approve")
async def approve_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Approve and resume a workflow waiting for human approval."""
    state = await _orchestrator.approve_workflow(workflow_id)
    return {
        "workflow_id": workflow_id,
        "status": state.status.value,
        "phase": state.phase,
    }


@workflow_router.post("/{workflow_id}/reject")
async def reject_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Reject a workflow waiting for approval (cancels it)."""
    await _orchestrator.cancel_workflow(workflow_id)
    return {"workflow_id": workflow_id, "status": "cancelled", "reason": "rejected_by_user"}


# ---------------------------------------------------------------------------
# Artifacts, Events, Report
# ---------------------------------------------------------------------------

@workflow_router.get("/{workflow_id}/artifacts")
async def get_artifacts(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get all artifacts for a workflow."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    artifacts = {
        "repository_map": workflow_store.get_artifacts_by_type(workflow_id, "repository_map"),
        "assumption_challenges": workflow_store.get_artifacts_by_type(workflow_id, "assumption_challenge"),
        "hypotheses": workflow_store.get_artifacts_by_type(workflow_id, "hypothesis"),
        "reviews": workflow_store.get_artifacts_by_type(workflow_id, "review"),
        "selected_direction": workflow_store.get_artifacts_by_type(workflow_id, "selected_direction"),
        "roadmap": workflow_store.get_artifacts_by_type(workflow_id, "roadmap"),
    }
    return {"workflow_id": workflow_id, "artifacts": artifacts}


@workflow_router.get("/{workflow_id}/events")
async def get_events(
    workflow_id: str,
    after_sequence: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """Get workflow events (optionally after a sequence number)."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    events = workflow_store.get_events(workflow_id, after_sequence)
    return {
        "workflow_id": workflow_id,
        "events": [e.model_dump() for e in events],
        "count": len(events),
    }


@workflow_router.get("/{workflow_id}/events/stream")
async def stream_events(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream workflow events in real-time (SSE)."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async def event_generator():
        last_seq = 0
        while True:
            events = workflow_store.get_events(workflow_id, last_seq)
            for event in events:
                yield f"data: {event.model_dump_json()}\n\n"
                last_seq = event.sequence

            # Check if workflow is done
            current_state = workflow_store.get_workflow(workflow_id)
            if current_state and current_state.status.value in ("completed", "failed", "cancelled"):
                yield f"data: {json.dumps({'type': 'workflow_done', 'status': current_state.status.value})}\n\n"
                break

            await asyncio.sleep(1)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@workflow_router.get("/{workflow_id}/report")
async def get_report(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Generate the final report for a completed workflow."""
    state = workflow_store.get_workflow(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if state.status.value != "completed":
        raise HTTPException(status_code=400, detail=f"Workflow not completed (status={state.status.value})")

    # Gather artifacts
    direction_data = workflow_store.get_artifacts_by_type(workflow_id, "selected_direction")
    roadmap_data = workflow_store.get_artifacts_by_type(workflow_id, "roadmap")
    hypotheses = workflow_store.get_artifacts_by_type(workflow_id, "hypothesis")
    reviews = workflow_store.get_artifacts_by_type(workflow_id, "review")
    repo_map = workflow_store.get_artifacts_by_type(workflow_id, "repository_map")
    challenges = workflow_store.get_artifacts_by_type(workflow_id, "assumption_challenge")

    report = {
        "workflow_id": workflow_id,
        "repository_path": state.repository_path,
        "repository_revision": state.repository_revision,
        "user_goal": state.user_goal,
        "constraints": state.constraints,
        "status": state.status.value,
        "created_at": state.created_at.isoformat(),
        "completed_at": state.updated_at.isoformat(),
        "workflow_seed": state.workflow_seed,
        "sections": {
            "repository_reality": repo_map[0] if repo_map else None,
            "assumptions_challenged": challenges,
            "innovation_hypotheses": hypotheses,
            "cto_review": reviews,
            "selected_direction": direction_data[0] if direction_data else None,
            "roadmap": roadmap_data[0] if roadmap_data else None,
        },
    }
    return report
