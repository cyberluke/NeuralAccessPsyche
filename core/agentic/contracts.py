"""
Agentic Innovation Pipeline — contracts and Pydantic schemas.

All persona inputs/outputs are typed. No raw free-form conversations
pass between stages. Every artifact is validated before storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------

class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InnovationWorkflowState(BaseModel):
    workflow_id: str
    workflow_type: Literal["codebase_innovation"] = "codebase_innovation"

    repository_path: str
    repository_revision: str
    user_goal: str
    constraints: List[str] = Field(default_factory=list)
    excluded_ideas: List[str] = Field(default_factory=list)

    status: WorkflowStatus = WorkflowStatus.CREATED
    phase: str = "created"
    iteration: int = 0
    max_iterations: int = 2
    workflow_seed: int = 271

    # Artifact references (large artifacts stored separately)
    repository_map_id: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    assumption_challenge_ids: List[str] = Field(default_factory=list)
    hypothesis_ids: List[str] = Field(default_factory=list)
    review_ids: List[str] = Field(default_factory=list)
    selected_direction_id: Optional[str] = None
    roadmap_id: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Evidence boundary
# ---------------------------------------------------------------------------

class EvidenceSourceType(str, Enum):
    SOURCE_FILE = "source_file"
    SYMBOL = "symbol"
    CONFIGURATION = "configuration"
    TEST_RESULT = "test_result"
    RUNTIME_RESULT = "runtime_result"
    GIT_HISTORY = "git_history"
    DOCUMENTATION = "documentation"
    USER_CONSTRAINT = "user_constraint"


class EvidenceItem(BaseModel):
    """Append-only evidence record. Persona output is NEVER evidence."""
    id: str
    claim: str
    source_type: EvidenceSourceType
    repository_revision: str
    path: Optional[str] = None
    symbol: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    command: Optional[str] = None
    observed_value: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    verified: bool = False


# ---------------------------------------------------------------------------
# Repository map (Archaeologist output)
# ---------------------------------------------------------------------------

class CodeReference(BaseModel):
    path: str
    symbol: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class TechnologyFact(BaseModel):
    name: str
    version: Optional[str] = None
    role: str
    evidence_ids: List[str] = Field(default_factory=list)


class CapabilityClassification(str, Enum):
    VERIFIED_IMPLEMENTED = "verified_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    SIMULATED_OR_MOCKED = "simulated_or_mocked"
    DOCUMENTED_ONLY = "documented_only"
    UNKNOWN = "unknown"


class CapabilityFact(BaseModel):
    name: str
    classification: CapabilityClassification
    description: str
    evidence_ids: List[str] = Field(default_factory=list)


class DebtFact(BaseModel):
    description: str
    severity: Literal["low", "medium", "high"]
    evidence_ids: List[str] = Field(default_factory=list)


class RepositoryMap(BaseModel):
    summary: str
    technologies: List[TechnologyFact] = Field(default_factory=list)
    entry_points: List[CodeReference] = Field(default_factory=list)
    verified_capabilities: List[CapabilityFact] = Field(default_factory=list)
    partial_capabilities: List[CapabilityFact] = Field(default_factory=list)
    simulated_capabilities: List[CapabilityFact] = Field(default_factory=list)
    documented_only_capabilities: List[CapabilityFact] = Field(default_factory=list)
    technical_debt: List[DebtFact] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Assumption challenges (Heretic output)
# ---------------------------------------------------------------------------

class AssumptionChallenge(BaseModel):
    id: str
    assumption: str
    why_it_may_be_wrong: str
    proposed_reframe: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    falsification_test: str
    expected_effort: str
    expected_learning: str


# ---------------------------------------------------------------------------
# Innovation hypotheses (Psychedelic Synthesizer output)
# ---------------------------------------------------------------------------

class ExperimentDefinition(BaseModel):
    description: str
    success_metric: str
    kill_criterion: str
    estimated_effort: str


class InnovationHypothesis(BaseModel):
    id: str
    title: str
    thesis: str
    target_user: str
    user_problem: str
    source_components: List[CodeReference] = Field(default_factory=list)
    external_concept: str
    connection_explanation: str
    proposed_capability: str
    technical_mechanism: str
    differentiation: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    first_experiment: ExperimentDefinition
    novelty_score: float = Field(0.5, ge=0.0, le=1.0)
    feasibility_score: float = Field(0.5, ge=0.0, le=1.0)
    strategic_value_score: float = Field(0.5, ge=0.0, le=1.0)
    evidence_strength: float = Field(0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# CTO review (Ruthless CTO output)
# ---------------------------------------------------------------------------

class CTOVerdict(str, Enum):
    REJECT = "reject"
    REVISE = "revise"
    PROTOTYPE = "prototype"
    PROMISING = "promising"


class ReviewedHypothesis(BaseModel):
    hypothesis_id: str
    verdict: CTOVerdict
    strongest_argument_for: str
    strongest_argument_against: str
    technical_blockers: List[str] = Field(default_factory=list)
    hidden_costs: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    simpler_alternatives: List[str] = Field(default_factory=list)
    prompt_only_baseline_test: str
    required_proof: List[str] = Field(default_factory=list)
    kill_criteria: List[str] = Field(default_factory=list)
    revised_scope: Optional[str] = None
    estimated_effort: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Selected direction (Product Dictator output)
# ---------------------------------------------------------------------------

class SelectedDirection(BaseModel):
    product_wedge: str
    target_user: str
    urgent_problem: str
    unique_mechanism: str
    why_now: str
    chosen_hypothesis_ids: List[str] = Field(default_factory=list)
    rejected_hypothesis_ids: List[str] = Field(default_factory=list)
    immediate_demo: str
    stopped_work: List[str] = Field(default_factory=list)
    decisive_experiments: List[str] = Field(default_factory=list)
    north_star_metric: str
    monetization_hypothesis: str
    defensibility: str
    principal_failure_mode: str
    evidence_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Roadmap
# ---------------------------------------------------------------------------

class RoadmapItem(BaseModel):
    id: str
    horizon: Literal["72h", "2w", "6w", "3m", "6m"]
    title: str
    objective: str
    repository_components: List[CodeReference] = Field(default_factory=list)
    implementation_steps: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    acceptance_tests: List[str] = Field(default_factory=list)
    evidence_required: List[str] = Field(default_factory=list)
    kill_criteria: List[str] = Field(default_factory=list)
    estimated_effort: str
    risk: Literal["low", "medium", "high"]
    owner_role: str
    source_hypothesis_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class Roadmap(BaseModel):
    items: List[RoadmapItem] = Field(default_factory=list)
    horizons: List[str] = Field(default_factory=lambda: ["72h", "2w", "6w", "3m", "6m"])


# ---------------------------------------------------------------------------
# Workflow events
# ---------------------------------------------------------------------------

class WorkflowEvent(BaseModel):
    workflow_id: str
    sequence: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: str
    persona: Optional[str] = None
    event_type: str
    artifact_id: Optional[str] = None
    summary: str
    evidence_ids: List[str] = Field(default_factory=list)
    # Metadata (scores, verdict, profile) — no hidden reasoning
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# NRAM profile per persona
# ---------------------------------------------------------------------------

PERSONA_NRAM_PROFILES: Dict[str, Dict[str, Any]] = {
    "archaeologist": {
        "profile": "normal",
        "intensity": 0.10,
        "temperature": 0.20,
        "associative_distance": 0.05,
        "contrarian_force": 0.05,
        "coherence_floor": 0.98,
    },
    "heretic": {
        "profile": "threshold",
        "intensity": 0.58,
        "temperature": 0.70,
        "associative_distance": 0.48,
        "contrarian_force": 0.95,
        "coherence_floor": 0.88,
    },
    "psychedelic_synthesizer": {
        "profile": "psychedelic",
        "intensity": 0.84,
        "temperature": 1.05,
        "associative_distance": 0.94,
        "contrarian_force": 0.75,
        "coherence_floor": 0.74,
    },
    "ruthless_cto": {
        "profile": "normal",
        "intensity": 0.22,
        "temperature": 0.25,
        "associative_distance": 0.08,
        "contrarian_force": 0.90,
        "coherence_floor": 0.98,
    },
    "product_dictator": {
        "profile": "threshold",
        "intensity": 0.48,
        "temperature": 0.35,
        "associative_distance": 0.20,
        "contrarian_force": 0.72,
        "coherence_floor": 0.97,
    },
}
