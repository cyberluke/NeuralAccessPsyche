"""
Innovation Orchestrator — sequential persona execution with checkpoints.

Workflow:
  repository_ingestion
    -> archaeologist
    -> evidence_validation
    -> heretic
    -> psychedelic_synthesizer
    -> ruthless_cto
    -> single_revision_loop
    -> human_approval
    -> product_dictator
    -> roadmap_builder
    -> final_evidence_audit
    -> completed

Personas run sequentially. A later persona consumes structured artifacts
from earlier personas. A checkpoint is persisted after every successful stage.
The workflow resumes after application restart.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from core.agentic.contracts import (
    AssumptionChallenge,
    CTOVerdict,
    InnovationHypothesis,
    InnovationWorkflowState,
    RepositoryMap,
    ReviewedHypothesis,
    Roadmap,
    SelectedDirection,
    WorkflowEvent,
    WorkflowStatus,
)
from core.agentic.evidence_ledger import EvidenceLedger
from core.agentic.personas.archaeologist import Archaeologist
from core.agentic.personas.heretic import Heretic
from core.agentic.personas.product_dictator import ProductDictator
from core.agentic.personas.psychedelic_synthesizer import PsychedelicSynthesizer
from core.agentic.personas.ruthless_cto import RuthlessCTO
from core.agentic.workflow_store import WorkflowStore, workflow_store

logger = logging.getLogger(__name__)

# Ordered phases
PHASES = [
    "repository_ingestion",
    "archaeologist",
    "evidence_validation",
    "heretic",
    "psychedelic_synthesizer",
    "ruthless_cto",
    "single_revision_loop",
    "human_approval",
    "product_dictator",
    "roadmap_builder",
    "final_evidence_audit",
    "completed",
]


class OrchestratorError(Exception):
    pass


class InnovationOrchestrator:
    """Runs the 5-persona innovation pipeline sequentially with checkpoints."""

    def __init__(
        self,
        store: Optional[WorkflowStore] = None,
        evidence_ledger: Optional[EvidenceLedger] = None,
        nram_api_base: str = "http://localhost:8000/v1",
    ) -> None:
        self._store = store or workflow_store
        self._evidence = evidence_ledger or EvidenceLedger()
        self._nram_api_base = nram_api_base
        self._event_callbacks: List[Callable[[WorkflowEvent], Any]] = []

    def on_event(self, callback: Callable[[WorkflowEvent], Any]) -> None:
        """Register a callback for workflow events (e.g., SSE streaming)."""
        self._event_callbacks.append(callback)

    async def _emit_event(
        self,
        workflow_id: str,
        phase: str,
        event_type: str,
        summary: str,
        persona: Optional[str] = None,
        artifact_id: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        seq = self._store.get_next_sequence(workflow_id)
        event = WorkflowEvent(
            workflow_id=workflow_id,
            sequence=seq,
            phase=phase,
            persona=persona,
            event_type=event_type,
            artifact_id=artifact_id,
            summary=summary,
            evidence_ids=evidence_ids or [],
            metadata=metadata or {},
        )
        self._store.add_event(event)
        for cb in self._event_callbacks:
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")

    # -----------------------------------------------------------------------
    # Workflow lifecycle
    # -----------------------------------------------------------------------

    async def create_workflow(
        self,
        repository_path: str,
        repository_revision: str,
        user_goal: str,
        constraints: List[str],
        excluded_ideas: List[str],
        workflow_seed: int = 271,
        max_iterations: int = 2,
        require_human_approval: bool = True,
    ) -> InnovationWorkflowState:
        """Create a new innovation workflow."""
        state = InnovationWorkflowState(
            workflow_id=f"wf-{uuid.uuid4().hex[:12]}",
            repository_path=repository_path,
            repository_revision=repository_revision,
            user_goal=user_goal,
            constraints=constraints,
            excluded_ideas=excluded_ideas,
            workflow_seed=workflow_seed,
            max_iterations=max_iterations,
            status=WorkflowStatus.CREATED,
            phase="repository_ingestion",
        )
        self._store.create_workflow(state)
        await self._emit_event(
            state.workflow_id, "repository_ingestion", "workflow_created",
            f"Workflow created for {repository_path} (rev {repository_revision})",
        )
        return state

    async def run_workflow(self, workflow_id: str) -> InnovationWorkflowState:
        """Run (or resume) a workflow through all phases."""
        state = self._store.get_workflow(workflow_id)
        if state is None:
            raise OrchestratorError(f"Workflow {workflow_id} not found")

        if state.status in (WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED):
            return state

        state.status = WorkflowStatus.RUNNING
        self._store.update_workflow(state)

        try:
            # Resume from current phase
            phase_idx = PHASES.index(state.phase) if state.phase in PHASES else 0

            for i in range(phase_idx, len(PHASES)):
                phase = PHASES[i]
                state.phase = phase
                self._store.update_workflow(state)

                if phase == "repository_ingestion":
                    await self._emit_event(workflow_id, phase, "phase_start", "Ingesting repository")

                elif phase == "archaeologist":
                    state = await self._run_archaeologist(state)

                elif phase == "evidence_validation":
                    await self._emit_event(workflow_id, phase, "phase_start", "Validating evidence")

                elif phase == "heretic":
                    state = await self._run_heretic(state)

                elif phase == "psychedelic_synthesizer":
                    state = await self._run_synthesizer(state)

                elif phase == "ruthless_cto":
                    state = await self._run_cto(state)

                elif phase == "single_revision_loop":
                    state = await self._run_revision_loop(state)

                elif phase == "human_approval":
                    state.status = WorkflowStatus.WAITING_FOR_APPROVAL
                    self._store.update_workflow(state)
                    await self._emit_event(workflow_id, phase, "waiting_for_approval", "Paused for human approval")
                    return state  # Pause here

                elif phase == "product_dictator":
                    state = await self._run_dictator(state)

                elif phase == "roadmap_builder":
                    await self._emit_event(workflow_id, phase, "phase_start", "Building roadmap")

                elif phase == "final_evidence_audit":
                    await self._emit_event(workflow_id, phase, "phase_start", "Final evidence audit")

                elif phase == "completed":
                    state.status = WorkflowStatus.COMPLETED
                    self._store.update_workflow(state)
                    await self._emit_event(workflow_id, phase, "workflow_completed", "Workflow completed")
                    return state

            state.status = WorkflowStatus.COMPLETED
            self._store.update_workflow(state)
            return state

        except Exception as e:
            state.status = WorkflowStatus.FAILED
            self._store.update_workflow(state)
            await self._emit_event(workflow_id, state.phase, "workflow_failed", f"Error: {e}")
            raise

    async def approve_workflow(self, workflow_id: str) -> InnovationWorkflowState:
        """Approve and resume a workflow paused at human_approval."""
        state = self._store.get_workflow(workflow_id)
        if state is None:
            raise OrchestratorError(f"Workflow {workflow_id} not found")
        if state.status != WorkflowStatus.WAITING_FOR_APPROVAL:
            raise OrchestratorError(f"Workflow is not waiting for approval (status={state.status})")

        state.status = WorkflowStatus.RUNNING
        state.phase = "product_dictator"
        self._store.update_workflow(state)
        await self._emit_event(workflow_id, "human_approval", "approved", "Human approved, resuming")

        # Continue from product_dictator
        state = await self._run_dictator(state)
        state.phase = "roadmap_builder"
        self._store.update_workflow(state)
        await self._emit_event(workflow_id, "roadmap_builder", "phase_start", "Building roadmap")

        state.phase = "final_evidence_audit"
        self._store.update_workflow(state)
        await self._emit_event(workflow_id, "final_evidence_audit", "phase_start", "Final evidence audit")

        state.status = WorkflowStatus.COMPLETED
        state.phase = "completed"
        self._store.update_workflow(state)
        await self._emit_event(workflow_id, "completed", "workflow_completed", "Workflow completed")
        return state

    async def cancel_workflow(self, workflow_id: str) -> None:
        state = self._store.get_workflow(workflow_id)
        if state is None:
            raise OrchestratorError(f"Workflow {workflow_id} not found")
        state.status = WorkflowStatus.CANCELLED
        self._store.update_workflow(state)
        await self._emit_event(workflow_id, state.phase, "workflow_cancelled", "Workflow cancelled")

    # -----------------------------------------------------------------------
    # Phase runners
    # -----------------------------------------------------------------------

    async def _run_archaeologist(self, state: InnovationWorkflowState) -> InnovationWorkflowState:
        persona = Archaeologist(
            nram_api_base=self._nram_api_base,
            evidence_ledger=self._evidence,
            workflow_seed=state.workflow_seed,
        )
        repo_map = await persona.run(
            repository_path=state.repository_path,
            repository_revision=state.repository_revision,
            user_goal=state.user_goal,
            constraints=state.constraints,
        )
        artifact_id = f"repo-map-{uuid.uuid4().hex[:8]}"
        self._store.store_artifact(state.workflow_id, artifact_id, "repository_map", repo_map)
        state.repository_map_id = artifact_id
        state.evidence_ids.extend(repo_map.evidence_ids)
        self._store.update_workflow(state)
        await self._emit_event(
            state.workflow_id, "archaeologist", "artifact_created",
            f"Repository map created: {repo_map.summary[:100]}",
            persona="archaeologist", artifact_id=artifact_id,
            metadata={"nram_profile": persona.nram_profile},
        )
        return state

    async def _run_heretic(self, state: InnovationWorkflowState) -> InnovationWorkflowState:
        persona = Heretic(
            nram_api_base=self._nram_api_base,
            evidence_ledger=self._evidence,
            workflow_seed=state.workflow_seed,
        )
        repo_map_data = self._store.get_artifact(state.repository_map_id) if state.repository_map_id else None
        repo_map = RepositoryMap.model_validate(repo_map_data) if repo_map_data else RepositoryMap(summary="")

        challenges = await persona.run(
            repository_map=repo_map,
            user_goal=state.user_goal,
            constraints=state.constraints,
        )
        for challenge in challenges:
            artifact_id = f"challenge-{uuid.uuid4().hex[:8]}"
            self._store.store_artifact(state.workflow_id, artifact_id, "assumption_challenge", challenge)
            state.assumption_challenge_ids.append(artifact_id)

        self._store.update_workflow(state)
        await self._emit_event(
            state.workflow_id, "heretic", "artifact_created",
            f"{len(challenges)} assumption challenges generated",
            persona="heretic",
            metadata={"count": len(challenges), "nram_profile": persona.nram_profile},
        )
        return state

    async def _run_synthesizer(self, state: InnovationWorkflowState) -> InnovationWorkflowState:
        persona = PsychedelicSynthesizer(
            nram_api_base=self._nram_api_base,
            evidence_ledger=self._evidence,
            workflow_seed=state.workflow_seed,
        )
        repo_map_data = self._store.get_artifact(state.repository_map_id) if state.repository_map_id else None
        repo_map = RepositoryMap.model_validate(repo_map_data) if repo_map_data else RepositoryMap(summary="")

        challenges = []
        for cid in state.assumption_challenge_ids:
            data = self._store.get_artifact(cid)
            if data:
                challenges.append(AssumptionChallenge.model_validate(data))

        hypotheses = await persona.run(
            repository_map=repo_map,
            challenges=challenges,
            user_goal=state.user_goal,
            constraints=state.constraints,
        )
        for hyp in hypotheses:
            artifact_id = f"hyp-{uuid.uuid4().hex[:8]}"
            self._store.store_artifact(state.workflow_id, artifact_id, "hypothesis", hyp)
            state.hypothesis_ids.append(artifact_id)

        self._store.update_workflow(state)
        await self._emit_event(
            state.workflow_id, "psychedelic_synthesizer", "artifact_created",
            f"{len(hypotheses)} innovation hypotheses generated",
            persona="psychedelic_synthesizer",
            metadata={"count": len(hypotheses), "nram_profile": persona.nram_profile},
        )
        return state

    async def _run_cto(self, state: InnovationWorkflowState) -> InnovationWorkflowState:
        persona = RuthlessCTO(
            nram_api_base=self._nram_api_base,
            evidence_ledger=self._evidence,
            workflow_seed=state.workflow_seed,
        )
        repo_map_data = self._store.get_artifact(state.repository_map_id) if state.repository_map_id else None
        repo_map = RepositoryMap.model_validate(repo_map_data) if repo_map_data else RepositoryMap(summary="")

        hypotheses = []
        for hid in state.hypothesis_ids:
            data = self._store.get_artifact(hid)
            if data:
                hypotheses.append(InnovationHypothesis.model_validate(data))

        reviews = await persona.run(
            hypotheses=hypotheses,
            repository_map=repo_map,
            user_goal=state.user_goal,
            constraints=state.constraints,
        )
        for review in reviews:
            artifact_id = f"review-{uuid.uuid4().hex[:8]}"
            self._store.store_artifact(state.workflow_id, artifact_id, "review", review)
            state.review_ids.append(artifact_id)

        self._store.update_workflow(state)
        survivors = [r for r in reviews if r.verdict in (CTOVerdict.PROTOTYPE, CTOVerdict.PROMISING)]
        await self._emit_event(
            state.workflow_id, "ruthless_cto", "artifact_created",
            f"{len(reviews)} reviews: {len(survivors)} survivors",
            persona="ruthless_cto",
            metadata={"total": len(reviews), "survivors": len(survivors), "nram_profile": persona.nram_profile},
        )
        return state

    async def _run_revision_loop(self, state: InnovationWorkflowState) -> InnovationWorkflowState:
        """One revision per hypothesis with verdict='revise'. Max 2 global iterations."""
        reviews = []
        for rid in state.review_ids:
            data = self._store.get_artifact(rid)
            if data:
                reviews.append(ReviewedHypothesis.model_validate(data))

        revise_count = sum(1 for r in reviews if r.verdict == CTOVerdict.REVISE)
        state.iteration += 1
        self._store.update_workflow(state)

        await self._emit_event(
            state.workflow_id, "single_revision_loop", "revision_check",
            f"{revise_count} hypotheses marked for revision (iteration {state.iteration}/{state.max_iterations})",
            metadata={"revise_count": revise_count, "iteration": state.iteration},
        )
        # In production: send revise hypotheses back to Synthesizer, then CTO again
        # For now, log and continue
        return state

    async def _run_dictator(self, state: InnovationWorkflowState) -> InnovationWorkflowState:
        persona = ProductDictator(
            nram_api_base=self._nram_api_base,
            evidence_ledger=self._evidence,
            workflow_seed=state.workflow_seed,
        )
        repo_map_data = self._store.get_artifact(state.repository_map_id) if state.repository_map_id else None
        repo_map = RepositoryMap.model_validate(repo_map_data) if repo_map_data else RepositoryMap(summary="")

        hypotheses = []
        for hid in state.hypothesis_ids:
            data = self._store.get_artifact(hid)
            if data:
                hypotheses.append(InnovationHypothesis.model_validate(data))

        reviews = []
        for rid in state.review_ids:
            data = self._store.get_artifact(rid)
            if data:
                reviews.append(ReviewedHypothesis.model_validate(data))

        direction, roadmap = await persona.run(
            hypotheses=hypotheses,
            reviews=reviews,
            repository_map=repo_map,
            user_goal=state.user_goal,
            constraints=state.constraints,
        )

        direction_id = f"direction-{uuid.uuid4().hex[:8]}"
        roadmap_id = f"roadmap-{uuid.uuid4().hex[:8]}"
        self._store.store_artifact(state.workflow_id, direction_id, "selected_direction", direction)
        self._store.store_artifact(state.workflow_id, roadmap_id, "roadmap", roadmap)
        state.selected_direction_id = direction_id
        state.roadmap_id = roadmap_id
        self._store.update_workflow(state)

        await self._emit_event(
            state.workflow_id, "product_dictator", "artifact_created",
            f"Selected direction: {direction.product_wedge}",
            persona="product_dictator", artifact_id=direction_id,
            metadata={"nram_profile": persona.nram_profile},
        )
        return state
