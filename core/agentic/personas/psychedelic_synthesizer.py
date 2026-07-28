"""
Psychedelic Synthesizer persona — creates distant but technically defensible connections.

NRAM profile: psychedelic, intensity=0.84, temperature=1.05, associative_distance=0.94
"""
from __future__ import annotations

import logging
from typing import Any, List

from core.agentic.contracts import (
    AssumptionChallenge,
    ExperimentDefinition,
    InnovationHypothesis,
    RepositoryMap,
)
from core.agentic.personas.base import BasePersona, PersonaError

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are the Psychedelic Synthesizer. Your job is controlled divergence.

RULES:
- Generate at least 15 hypotheses.
- At least 5 hypotheses must NOT be: another chatbot, a persona prompt, an API wrapper, a visualization dashboard.
- Every hypothesis must follow: repository asset -> external concept -> new thesis -> technical mechanism -> first experiment -> measurable outcome -> kill criterion.
- Explore connections to: compiler design, control theory, cognitive architectures, inference runtimes, observability, scientific instrumentation, operating systems, security sandboxes, game AI, programmable policy engines, causal experiments, model evaluation, digital twins, generative media systems.
- Do NOT select the winner. That is the CTO's and Dictator's job.

OUTPUT: Valid JSON list of InnovationHypothesis objects."""

SYNTHESIZER_USER_PROMPT = """Based on this repository map and assumption challenges:

REPOSITORY MAP:
{repository_map_json}

ASSUMPTION CHALLENGES:
{challenges_json}

User goal: {user_goal}
Constraints: {constraints}

Generate at least 15 innovation hypotheses. For each:
- title, thesis, target_user, user_problem
- source_components (from repository)
- external_concept (from the domain pool)
- connection_explanation
- proposed_capability, technical_mechanism, differentiation
- first_experiment (description, success_metric, kill_criterion, estimated_effort)
- novelty_score, feasibility_score, strategic_value_score, evidence_strength (0.0-1.0)

OUTPUT: JSON list of InnovationHypothesis objects."""


class PsychedelicSynthesizer(BasePersona):
    name = "psychedelic_synthesizer"
    output_schema = InnovationHypothesis

    async def run(  # type: ignore[override]
        self,
        repository_map: RepositoryMap,
        challenges: List[AssumptionChallenge],
        user_goal: str,
        constraints: list[str],
        **kwargs: Any,
    ) -> List[InnovationHypothesis]:
        """Execute the Psychedelic Synthesizer persona.

        Makes a live NRAM-steered call (psychedelic profile: intensity=0.84,
        temperature=1.05, associative_distance=0.94) to generate distant but
        technically defensible hypotheses. Falls back to a minimal valid list
        if the model is unreachable or the output is invalid.
        """
        user_prompt = SYNTHESIZER_USER_PROMPT.format(
            repository_map_json=repository_map.model_dump_json(indent=2)[:6000],
            challenges_json="[\n" + ",\n".join(
                c.model_dump_json(indent=2) for c in challenges[:10]
            ) + "\n]" if challenges else "[]",
            user_goal=user_goal,
            constraints=", ".join(constraints) if constraints else "none",
        )
        messages = self.build_messages(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)

        raw = await self.invoke_model(messages, max_tokens=12288)
        if raw:
            try:
                return self.validate_list_output(raw, InnovationHypothesis)
            except PersonaError as e:
                logger.warning(f"[psychedelic_synthesizer] live output invalid, using fallback: {e}")

        # Fallback: minimal valid hypotheses
        return [
            InnovationHypothesis(
                id="hyp-1",
                title="NRAM as Scientific Instrument",
                thesis="Token steering can be a measurement tool, not just a control tool",
                target_user="AI researchers",
                user_problem="Understanding how logit biases affect model behavior",
                source_components=[],
                external_concept="scientific instrumentation",
                connection_explanation="Like an oscilloscope for LLM decoding policies",
                proposed_capability="Real-time visualization of logit deltas per token",
                technical_mechanism="Extend NRAMLogitProcessor to emit per-token telemetry",
                differentiation="No existing tool shows pre-sampling logit manipulation",
                supporting_evidence_ids=[],
                contradicting_evidence_ids=[],
                assumptions=["Researchers want this visibility"],
                unknowns=["Performance overhead of telemetry"],
                first_experiment=ExperimentDefinition(
                    description="Build token-level telemetry dashboard",
                    success_metric="Researchers use it to debug steering policies",
                    kill_criterion="No researcher interest after 5 demos",
                    estimated_effort="2 weeks",
                ),
                novelty_score=0.85,
                feasibility_score=0.90,
                strategic_value_score=0.75,
                evidence_strength=0.70,
            )
        ]
