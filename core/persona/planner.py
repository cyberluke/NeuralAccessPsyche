"""Structured rhetorical planner using llguidance through SGLang."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VisionaryPlan(BaseModel):
    """Structured rhetorical plan generated before final generation."""

    uncomfortable_truth: str
    rejected_assumption: str
    human_need: str
    product_revelation: str
    unexpected_connection: str
    sensory_metaphor: str
    practical_consequence: str
    final_turn: str


PLANNER_SYSTEM_PROMPT = """You are a rhetorical planner for a visionary product keynote.
Given the user's request, produce a structured plan as JSON.
Do NOT write the keynote itself. Only produce the plan.
Each field must be a single concise sentence."""

PLANNER_SCHEMA = VisionaryPlan.model_json_schema()


async def build_rhetorical_plan(
    engine: Any,
    user_request: str,
    temperature: float = 0.4,
    max_tokens: int = 512,
) -> Optional[VisionaryPlan]:
    """Build a structured rhetorical plan using the engine's structured output.

    The planner call:
    - does NOT use the custom NRAM processor
    - uses temperature between 0.3 and 0.6
    - uses a limited token budget
    - returns valid schema-conforming JSON
    - remains hidden from the end user
    """
    from core.contracts.openai import ChatCompletionRequest

    plan_request = ChatCompletionRequest(
        model="__planner__",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ],
        temperature=max(0.3, min(0.6, temperature)),
        max_tokens=max_tokens,
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "visionary_plan",
                "strict": True,
                "schema": PLANNER_SCHEMA,
            },
        },
    )

    try:
        response = await engine.complete(plan_request)
        content = response.choices[0].message.content
        data = json.loads(content)
        return VisionaryPlan(**data)
    except Exception as e:
        logger.warning(f"Planner failed, proceeding without plan: {e}")
        return None


def plan_to_prompt_fragment(plan: VisionaryPlan) -> str:
    """Convert a VisionaryPlan into a prompt fragment for the final generation."""
    return f"""[Internal rhetorical plan — follow this structure, do not mention it]
1. Uncomfortable truth: {plan.uncomfortable_truth}
2. Rejected assumption: {plan.rejected_assumption}
3. Human need: {plan.human_need}
4. Product revelation: {plan.product_revelation}
5. Unexpected connection: {plan.unexpected_connection}
6. Sensory metaphor: {plan.sensory_metaphor}
7. Practical consequence: {plan.practical_consequence}
8. Final turn: {plan.final_turn}"""
