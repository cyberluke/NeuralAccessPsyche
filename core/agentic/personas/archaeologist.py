"""
Archaeologist persona — produces a factual repository map.

Does NOT generate product ideas or roadmap recommendations.
NRAM profile: normal, intensity=0.10, temperature=0.20
"""
from __future__ import annotations

import logging
from typing import Any

from core.agentic.contracts import RepositoryMap
from core.agentic.personas.base import BasePersona, PersonaError

logger = logging.getLogger(__name__)

ARCHAEOLOGIST_SYSTEM_PROMPT = """You are the Archaeologist. Your job is to produce a FACTUAL repository map.

RULES:
- Do NOT generate product ideas or roadmap recommendations.
- Do NOT speculate about what could be built.
- Every factual claim MUST reference evidence (file path, symbol, test result).
- Classify every capability as: verified_implemented, partially_implemented, simulated_or_mocked, documented_only, or unknown.
- Names of files or functions do NOT prove behavior.
- README content is documentation evidence, NOT implementation proof.
- Runtime capability claims require a test or observed command result.
- Unknown claims become explicit assumptions.

OUTPUT: Valid JSON conforming to the RepositoryMap schema."""

ARCHAEOLOGIST_USER_PROMPT = """Analyze the repository at {repository_path} (revision: {repository_revision}).

User goal: {user_goal}

Produce a complete RepositoryMap as JSON. Include:
- summary (2-3 sentences)
- technologies (name, version, role)
- entry_points (file paths)
- verified_capabilities (with evidence)
- partial_capabilities (with evidence)
- simulated_capabilities (with evidence)
- documented_only_capabilities (with evidence)
- technical_debt (severity: low/medium/high)

Constraints: {constraints}"""


class Archaeologist(BasePersona):
    name = "archaeologist"
    output_schema = RepositoryMap

    async def run(
        self,
        repository_path: str,
        repository_revision: str,
        user_goal: str,
        constraints: list[str],
        **kwargs: Any,
    ) -> RepositoryMap:
        """Execute the Archaeologist persona.

        Makes a live NRAM-steered call (archaeologist profile: normal,
        intensity=0.10, temperature=0.20) to analyze the repository, then
        validates the output against RepositoryMap. Falls back to a minimal
        valid map if the model is unreachable or the output is invalid.
        """
        user_prompt = ARCHAEOLOGIST_USER_PROMPT.format(
            repository_path=repository_path,
            repository_revision=repository_revision,
            user_goal=user_goal,
            constraints=", ".join(constraints) if constraints else "none",
        )
        messages = self.build_messages(ARCHAEOLOGIST_SYSTEM_PROMPT, user_prompt)

        raw = await self.invoke_model(messages)
        if raw:
            try:
                return self.validate_output(self.extract_json(raw), RepositoryMap)
            except PersonaError as e:
                logger.warning(f"[archaeologist] live output invalid, using fallback: {e}")

        # Fallback: minimal valid map so the workflow can continue.
        return RepositoryMap(
            summary=(
                f"Repository at {repository_path} analyzed for goal: {user_goal} "
                f"(fallback — model unavailable or output did not validate)"
            ),
            technologies=[],
            entry_points=[],
            verified_capabilities=[],
            partial_capabilities=[],
            simulated_capabilities=[],
            documented_only_capabilities=[],
            technical_debt=[],
            evidence_ids=[],
        )
