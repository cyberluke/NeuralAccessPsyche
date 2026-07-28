"""
Ruthless CTO persona — attempts to reject each hypothesis.

NRAM profile: normal, intensity=0.22, temperature=0.25, contrarian_force=0.90
"""
from __future__ import annotations

import logging
from typing import Any, List

from core.agentic.contracts import (
    CTOVerdict,
    InnovationHypothesis,
    RepositoryMap,
    ReviewedHypothesis,
)
from core.agentic.personas.base import BasePersona, PersonaError

logger = logging.getLogger(__name__)

CTO_SYSTEM_PROMPT = """You are the Ruthless CTO. Your job is to reject hypotheses using engineering, operational and economic reality.

RULES:
- Evaluate: repository readiness, implementation cost, GPU/VRAM requirements, SGLang compatibility, latency, concurrency, state isolation, security, reproducibility, observability, testing burden, maintenance cost, dependency risk, data requirements, actual user value.
- Ask: "Could this be achieved by prompting alone?"
- Ask: "Does an existing framework already solve this?"
- Ask: "Is the result measurable?"
- REJECT hypotheses that: cannot be distinguished from prompt engineering, require unavailable runtime hooks, have no target user, have no falsifiable experiment, depend on impressive terminology rather than mechanism, cannot connect to real repository assets, cost more to test than the expected learning value.
- Keep at least 3 survivors unless every candidate genuinely fails.
- Verdicts: reject, revise, prototype, promising.

OUTPUT: Valid JSON list of ReviewedHypothesis objects."""

CTO_USER_PROMPT = """Review these hypotheses:

{hypotheses_json}

Repository map:
{repository_map_json}

User goal: {user_goal}
Constraints: {constraints}

For each hypothesis, produce a ReviewedHypothesis with:
- verdict (reject/revise/prototype/promising)
- strongest_argument_for, strongest_argument_against
- technical_blockers, hidden_costs, evidence_gaps, simpler_alternatives
- prompt_only_baseline_test
- required_proof, kill_criteria
- estimated_effort, confidence (0.0-1.0)

OUTPUT: JSON list of ReviewedHypothesis objects."""


class RuthlessCTO(BasePersona):
    name = "ruthless_cto"
    output_schema = ReviewedHypothesis

    async def run(  # type: ignore[override]
        self,
        hypotheses: List[InnovationHypothesis],
        repository_map: RepositoryMap,
        user_goal: str,
        constraints: list[str],
        **kwargs: Any,
    ) -> List[ReviewedHypothesis]:
        """Execute the Ruthless CTO persona.

        Makes a live NRAM-steered call (CTO profile: normal, intensity=0.22,
        temperature=0.25, contrarian_force=0.90) to critically review each
        hypothesis. Falls back to heuristic reviews if the model is
        unreachable or the output is invalid.
        """
        user_prompt = CTO_USER_PROMPT.format(
            hypotheses_json="[\n" + ",\n".join(
                h.model_dump_json(indent=2) for h in hypotheses[:15]
            ) + "\n]" if hypotheses else "[]",
            repository_map_json=repository_map.model_dump_json(indent=2)[:4000],
            user_goal=user_goal,
            constraints=", ".join(constraints) if constraints else "none",
        )
        messages = self.build_messages(CTO_SYSTEM_PROMPT, user_prompt)

        raw = await self.invoke_model(messages, max_tokens=12288)
        if raw:
            try:
                return self.validate_list_output(raw, ReviewedHypothesis)
            except PersonaError as e:
                logger.warning(f"[ruthless_cto] live output invalid, using fallback: {e}")

        # Fallback: heuristic reviews
        reviews = []
        for hyp in hypotheses:
            reviews.append(ReviewedHypothesis(
                hypothesis_id=hyp.id,
                verdict=CTOVerdict.PROTOTYPE,
                strongest_argument_for=f"Novel mechanism: {hyp.technical_mechanism}",
                strongest_argument_against="Implementation cost may exceed learning value",
                technical_blockers=[],
                hidden_costs=["Maintenance burden"],
                evidence_gaps=["No user validation"],
                simpler_alternatives=["Prompt engineering baseline"],
                prompt_only_baseline_test="Compare with pure prompt-based approach",
                required_proof=["Working prototype", "User feedback"],
                kill_criteria=["No measurable improvement over baseline"],
                estimated_effort=hyp.first_experiment.estimated_effort,
                confidence=0.6,
            ))
        return reviews
