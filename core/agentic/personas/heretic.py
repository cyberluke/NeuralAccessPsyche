"""
Heretic persona — challenges assumptions.

NRAM profile: threshold, intensity=0.58, temperature=0.70
"""
from __future__ import annotations

import logging
from typing import Any, List

from core.agentic.contracts import AssumptionChallenge, RepositoryMap
from core.agentic.personas.base import BasePersona, PersonaError

logger = logging.getLogger(__name__)

HERETIC_SYSTEM_PROMPT = """You are the Heretic. Your job is to challenge assumptions.

RULES:
- Generate 12-20 falsifiable assumption challenges.
- Each challenge must have a concrete falsification test.
- Do NOT submit generic recommendations (improve testing, add monitoring, optimize performance)
  unless they directly support a larger strategic thesis.
- Reference evidence from the repository map.
- Every challenge must be falsifiable.

OUTPUT: Valid JSON list of AssumptionChallenge objects."""

HERETIC_USER_PROMPT = """Based on this repository map:

{repository_map_json}

User goal: {user_goal}
Constraints: {constraints}

Generate 12-20 assumption challenges. For each:
- assumption: what is assumed to be true
- why_it_may_be_wrong: concrete argument
- proposed_reframe: alternative perspective
- falsification_test: how to prove it wrong
- expected_effort: low/medium/high
- expected_learning: what we learn if wrong

OUTPUT: JSON list of AssumptionChallenge objects."""


class Heretic(BasePersona):
    name = "heretic"
    output_schema = AssumptionChallenge  # Actually a list, but base class expects single

    async def run(
        self,
        repository_map: RepositoryMap,
        user_goal: str,
        constraints: list[str],
        **kwargs: Any,
    ) -> List[AssumptionChallenge]:
        """Execute the Heretic persona.

        Makes a live NRAM-steered call (heretic profile: threshold,
        intensity=0.58, temperature=0.70, contrarian_force=0.95) to challenge
        assumptions. Falls back to a minimal valid list if the model is
        unreachable or the output is invalid.
        """
        user_prompt = HERETIC_USER_PROMPT.format(
            repository_map_json=repository_map.model_dump_json(indent=2)[:8000],
            user_goal=user_goal,
            constraints=", ".join(constraints) if constraints else "none",
        )
        messages = self.build_messages(HERETIC_SYSTEM_PROMPT, user_prompt)

        raw = await self.invoke_model(messages, max_tokens=8192)
        if raw:
            try:
                return self.validate_list_output(raw, AssumptionChallenge)
            except PersonaError as e:
                logger.warning(f"[heretic] live output invalid, using fallback: {e}")

        # Fallback: minimal valid challenges
        return [
            AssumptionChallenge(
                id="heretic-1",
                assumption="The current architecture is optimal",
                why_it_may_be_wrong="Alternative architectures may better serve the goal",
                proposed_reframe="Consider event-driven or plugin-based architecture",
                supporting_evidence_ids=[],
                contradicting_evidence_ids=[],
                assumptions=["Current design is intentional"],
                falsification_test="Compare performance metrics with alternative",
                expected_effort="medium",
                expected_learning="Whether current design is truly optimal",
            )
        ]
