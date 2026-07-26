"""
Product Dictator persona — selects one focused strategic direction.

NRAM profile: threshold, intensity=0.48, temperature=0.35, product_obsession=0.98
"""
from __future__ import annotations

import json
import logging
import uuid
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
        """Execute the Product Dictator persona.

        Makes a live NRAM-steered call (product_dictator profile: threshold,
        intensity=0.48, temperature=0.35) to select direction and build roadmap.
        Falls back to hardcoded values if model is unreachable or output invalid.
        """
        # Filter survivors
        survivor_ids = {
            r.hypothesis_id for r in reviews
            if r.verdict in (CTOVerdict.PROTOTYPE, CTOVerdict.PROMISING)
        }
        survivors = [h for h in hypotheses if h.id in survivor_ids]

        if not survivors:
            survivors = hypotheses[:3]  # Keep at least 3

        # Build JSON data for prompt
        survivors_data = [
            {
                "id": h.id,
                "title": h.title,
                "thesis": h.thesis,
                "target_user": h.target_user,
                "user_problem": h.user_problem,
                "technical_mechanism": h.technical_mechanism,
                "novelty_score": h.novelty_score,
                "feasibility_score": h.feasibility_score,
            }
            for h in survivors
        ]
        verdicts_data = [
            {
                "hypothesis_id": r.hypothesis_id,
                "verdict": r.verdict.value,
                "strongest_for": r.strongest_argument_for,
                "strongest_against": r.strongest_argument_against,
                "confidence": r.confidence,
            }
            for r in reviews
        ]

        user_prompt = DICTATOR_USER_PROMPT.format(
            survivors_json=json.dumps(survivors_data, indent=2),
            verdicts_json=json.dumps(verdicts_data, indent=2),
            repository_map_json=repository_map.model_dump_json(indent=2),
            user_goal=user_goal,
            constraints=", ".join(constraints) if constraints else "none",
        )
        messages = self.build_messages(DICTATOR_SYSTEM_PROMPT, user_prompt)

        # Call the model
        raw = await self.invoke_model(messages)
        if raw:
            try:
                data = json.loads(self.extract_json(raw))
                direction_data = data.get("direction", data)
                roadmap_data = data.get("roadmap", {})

                # Parse direction
                direction = SelectedDirection(
                    product_wedge=direction_data.get("product_wedge", survivors[0].title if survivors else "NRAM Control Surface"),
                    target_user=direction_data.get("target_user", survivors[0].target_user if survivors else "AI researchers"),
                    urgent_problem=direction_data.get("urgent_problem", survivors[0].user_problem if survivors else "Understanding LLM steering"),
                    unique_mechanism=direction_data.get("unique_mechanism", survivors[0].technical_mechanism if survivors else "Pre-sampling logit manipulation"),
                    why_now=direction_data.get("why_now", "SGLang enables custom logit processors; no existing tool exposes this"),
                    chosen_hypothesis_ids=direction_data.get("chosen_hypothesis_ids", [h.id for h in survivors]),
                    rejected_hypothesis_ids=direction_data.get("rejected_hypothesis_ids", [h.id for h in hypotheses if h.id not in survivor_ids]),
                    immediate_demo=direction_data.get("immediate_demo", "Token-level telemetry dashboard showing logit deltas"),
                    stopped_work=direction_data.get("stopped_work", ["Generic chatbot wrappers", "Prompt-only personas"]),
                    decisive_experiments=direction_data.get("decisive_experiments", []),
                    north_star_metric=direction_data.get("north_star_metric", "Number of researchers using NRAM telemetry weekly"),
                    monetization_hypothesis=direction_data.get("monetization_hypothesis", "Enterprise license for steering observability"),
                    defensibility=direction_data.get("defensibility", "First-mover in pre-sampling logit telemetry"),
                    principal_failure_mode=direction_data.get("principal_failure_mode", "Overhead too high for production use"),
                    evidence_ids=[],
                )

                # Parse roadmap
                roadmap_items = []
                for item_data in roadmap_data.get("items", []):
                    try:
                        item = RoadmapItem(
                            id=item_data.get("id", f"rm-{uuid.uuid4().hex[:6]}"),
                            horizon=item_data.get("horizon", "72h"),
                            title=item_data.get("title", "Untitled"),
                            objective=item_data.get("objective", ""),
                            repository_components=[],
                            implementation_steps=item_data.get("implementation_steps", []),
                            dependencies=item_data.get("dependencies", []),
                            acceptance_tests=item_data.get("acceptance_tests", []),
                            evidence_required=item_data.get("evidence_required", []),
                            kill_criteria=item_data.get("kill_criteria", []),
                            estimated_effort=item_data.get("estimated_effort", "unknown"),
                            risk=item_data.get("risk", "medium"),
                            owner_role=item_data.get("owner_role", "unassigned"),
                            source_hypothesis_ids=item_data.get("source_hypothesis_ids", []),
                            evidence_ids=[],
                        )
                        roadmap_items.append(item)
                    except Exception as e:
                        logger.warning(f"[product_dictator] invalid roadmap item: {e}")

                roadmap = Roadmap(items=roadmap_items)
                logger.info(f"[product_dictator] live model output parsed: {len(roadmap_items)} roadmap items")
                return direction, roadmap

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"[product_dictator] live output invalid, using fallback: {e}")

        # Fallback: hardcoded values so the workflow can continue
        logger.warning("[product_dictator] using hardcoded fallback")
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
