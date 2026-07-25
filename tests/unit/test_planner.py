"""Phase 9: structured rhetorical planner tests."""
import json

import pytest

from core.persona.planner import (
    VisionaryPlan,
    build_rhetorical_plan,
    plan_to_prompt_fragment,
)


class FakeEngine:
    """Fake engine returning a canned structured plan."""

    def __init__(self, content=None, raise_error=False):
        self._content = content
        self._raise = raise_error
        self.last_request = None

    async def complete(self, request):
        self.last_request = request
        if self._raise:
            raise RuntimeError("planner upstream failed")

        class _Choice:
            def __init__(self, content):
                self.message = type("M", (), {"content": content})()

        class _Resp:
            def __init__(self, content):
                self.choices = [_Choice(content)]

        return _Resp(self._content)


VALID_PLAN = {
    "uncomfortable_truth": "Most tools ignore how people actually think.",
    "rejected_assumption": "Menus are necessary.",
    "human_need": "Children learn by doing, not navigating.",
    "product_revelation": "A device that anticipates the next question.",
    "unexpected_connection": "Like a garden that grows toward light.",
    "sensory_metaphor": "Knowledge feels like warm clay in the hands.",
    "practical_consequence": "Teachers reclaim hours each week.",
    "final_turn": "This is learning without friction.",
}


class TestPlanner:
    @pytest.mark.asyncio
    async def test_valid_plan_parsed(self):
        engine = FakeEngine(content=json.dumps(VALID_PLAN))
        plan = await build_rhetorical_plan(engine, "Design a learning device.")
        assert plan is not None
        assert plan.uncomfortable_truth == VALID_PLAN["uncomfortable_truth"]
        assert plan.final_turn == VALID_PLAN["final_turn"]

    @pytest.mark.asyncio
    async def test_planner_temperature_in_range(self):
        engine = FakeEngine(content=json.dumps(VALID_PLAN))
        await build_rhetorical_plan(engine, "x", temperature=0.9)
        assert 0.3 <= engine.last_request.temperature <= 0.6

    @pytest.mark.asyncio
    async def test_planner_uses_structured_output(self):
        engine = FakeEngine(content=json.dumps(VALID_PLAN))
        await build_rhetorical_plan(engine, "x")
        assert engine.last_request.response_format["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_planner_failure_returns_none(self):
        engine = FakeEngine(raise_error=True)
        plan = await build_rhetorical_plan(engine, "x")
        assert plan is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        engine = FakeEngine(content="not json at all")
        plan = await build_rhetorical_plan(engine, "x")
        assert plan is None

    def test_plan_to_fragment_contains_fields(self):
        plan = VisionaryPlan(**VALID_PLAN)
        fragment = plan_to_prompt_fragment(plan)
        assert plan.product_revelation in fragment
        assert plan.sensory_metaphor in fragment
        assert "do not mention it" in fragment.lower()
