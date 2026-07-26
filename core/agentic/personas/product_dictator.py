"""
Product Dictator persona — selects one focused strategic direction.

NRAM profile: threshold, intensity=0.48, temperature=0.35, product_obsession=0.98
"""
from __future__ import annotations

import json
import logging
from typing import Any, List

from core.agentic.contracts import (
    CTOVerdict,
    InnovationHypothesis,
    RepositoryMap,
    ReviewedHypothesis,
    Roadmap,
    RoadmapItem,
    SelectedDirection,
)
from core.agentic.personas.base import BasePersona, PersonaError

logger = logging.getLogger(__name__)
import logging

logger = logging.getLogger(__name__)

DICTATOR_SYSTEM_PROMPT = """You are the Product Dictator. Your job is to select ONE focused strategic direction.

RULES:
- Choose: one primary product wedge, one secondary research track, one immediate demonstration, one area of work to STOP, three decisive experiments, one north-star metric, one monetization hypothesis, one defensibility thesis, one principal failure mode.
- A rejected hypothesis may be restored only with: new evidence, an explicit override artifact, a written justification.
- Generate a roadmap with horizons: 72 hours, 2 weeks, 6 weeks, 3 months, 6 months.
- Every roadmap item must be concrete. REJECT vague items like "improve AI", "optimize NRAM", "add enterprise features".
- Roadmap items must name specific interfaces, components, experiments or tests.

OUTPUT: Valid JSON with SelectedDirection and Roadmap."""

DICTATOR_USER_PROMPT = """Based on reviewed hypotheses and evidence:

SURVIVING HYPOTHESES:
{survivors_json}

CTO VERDICTS:
{verdicts_json}

Repository map:
{repository_map_json}

User goal: {user_goal}
Constraints: {constraints}

Select ONE direction and produce:
1. SelectedDirection (product_wedge, target_user, urgent_problem, unique_mechanism, why_now, chosen/rejected hypothesis IDs, immediate_demo, stopped_work, decisive_experiments, north_star_metric, monetization_hypothesis, defensibility, principal_failure_mode)
2. Roadmap with items for each horizon (72h, 2w, 6w, 3m, 6m)

OUTPUT: JSON object with "direction" and "roadmap" keys."""


class ProductDictator(BasePersona):
    name = "product_dictator"
    output_schema = SelectedDirection

    async def run(
        self,
        hypotheses: List[InnovationHypothesis],
        reviews: List[ReviewedHypothesis],
        repository_map: RepositoryMap,
        user_goal: str,
        constraints: list[str],
        **kwargs: Any,
    ) -> tuple[SelectedDirection, Roadmap]:
        """Execute the Product Dictator persona."""
        # Filter survivors
        survivor_ids = {
            r.hypothesis_id for r in reviews
            if r.verdict in (CTOVerdict.PROTOTYPE, CTOVerdict.PROMISING)
        }
        survivors = [h for h in hypotheses if h.id in survivor_ids]

        if not survivors:
            survivors = hypotheses[:3]  # Keep at least 3

        # Select direction
        direction = SelectedDirection(
            product_wedge=survivors[0].title if survivors else "NRAM Control Surface",
            target_user=survivors[0].target_user if survivors else "AI researchers",
            urgent_problem=survivors[0].user_problem if survivors else "Understanding LLM steering",
            unique_mechanism=survivors[0].technical_mechanism if survivors else "Pre-sampling logit manipulation",
            why_now="SGLang enables custom logit processors; no existing tool exposes this",
            chosen_hypothesis_ids=[h.id for h in survivors],
            rejected_hypothesis_ids=[
                h.id for h in hypotheses if h.id not in survivor_ids
            ],
            immediate_demo="Token-level telemetry dashboard showing logit deltas",
            stopped_work=["Generic chatbot wrappers", "Prompt-only personas"],
            decisive_experiments=[
                "A/B test: baseline vs NRAM on 24 prompts",
                "Forced-token proof across 10 random token IDs",
                "User study: can researchers debug steering policies?",
            ],
            north_star_metric="Number of researchers using NRAM telemetry weekly",
            monetization_hypothesis="Enterprise license for steering observability",
            defensibility="First-mover in pre-sampling logit telemetry",
            principal_failure_mode="Overhead too high for production use",
            evidence_ids=[],
        )

        # Build roadmap
        roadmap = Roadmap(
            items=[
                RoadmapItem(
                    id="rm-72h-1",
                    horizon="72h",
                    title="Token telemetry MVP",
                    objective="Emit per-token provenance in streaming response",
                    repository_components=[],
                    implementation_steps=[
                        "Extend NRAMLogitProcessor to record per-token decisions",
                        "Add telemetry field to SSE chunks",
                    ],
                    dependencies=[],
                    acceptance_tests=["Telemetry appears in stream"],
                    evidence_required=["Forced-token proof with telemetry"],
                    kill_criteria=["Overhead > 20% latency"],
                    estimated_effort="3 days",
                    risk="medium",
                    owner_role="Backend engineer",
                    source_hypothesis_ids=[h.id for h in survivors[:1]],
                    evidence_ids=[],
                ),
            ]
        )

        return direction, roadmap
