"""Phase 13: stateful per-token schedule tests (no GPU required)."""
import pytest

torch = pytest.importorskip("torch")

from core.steering.nram_logit_processor import NRAMLogitProcessor


class FakeRequest:
    def __init__(self, output_ids):
        self.output_ids = output_ids


class TestPhaseSchedule:
    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_opening_phase_no_scaling(self):
        scale = NRAMLogitProcessor._phase_schedule({"max_tokens": 100}, generated=5)
        assert scale == {"positive_scale": 1.0, "negative_scale": 1.0, "repetition_scale": 1.0}

    def test_contradiction_phase_boosts_negative(self):
        scale = NRAMLogitProcessor._phase_schedule({"max_tokens": 100}, generated=25)
        assert scale["negative_scale"] > 1.0
        assert scale["positive_scale"] == 1.0

    def test_revelation_phase_boosts_positive(self):
        scale = NRAMLogitProcessor._phase_schedule({"max_tokens": 100}, generated=55)
        assert scale["positive_scale"] > 1.0

    def test_final_turn_boosts_repetition(self):
        scale = NRAMLogitProcessor._phase_schedule({"max_tokens": 100}, generated=95)
        assert scale["repetition_scale"] > 1.0
        assert scale["positive_scale"] < 1.0

    def test_no_max_tokens_disables_schedule(self):
        scale = NRAMLogitProcessor._phase_schedule({}, generated=50)
        assert scale["positive_scale"] == 1.0
        assert scale["negative_scale"] == 1.0

    def test_schedule_actually_changes_logits(self):
        """In final-turn phase, repetition penalty is stronger than opening."""
        base_params = {
            "repetition_penalty": 1.0,
            "max_tokens": 100,
            "__req__": FakeRequest(output_ids=[7] * 95),
        }
        logits_opening = torch.zeros((1, 100))
        # generated=5 => opening
        out_opening = self.proc(
            logits_opening.clone(),
            [{**base_params, "__req__": FakeRequest(output_ids=[7] * 5)}],
        )
        logits_final = torch.zeros((1, 100))
        out_final = self.proc(
            logits_final.clone(),
            [{**base_params, "__req__": FakeRequest(output_ids=[7] * 95)}],
        )
        # final turn applies a stronger repetition penalty to token 7
        assert out_final[0, 7].item() < out_opening[0, 7].item()

    def test_request_state_isolated(self):
        """Two batch rows with different progress get independent schedules."""
        logits = torch.zeros((2, 100))
        params = [
            {"repetition_penalty": 1.0, "max_tokens": 100,
             "__req__": FakeRequest(output_ids=[5] * 5)},
            {"repetition_penalty": 1.0, "max_tokens": 100,
             "__req__": FakeRequest(output_ids=[5] * 95)},
        ]
        out = self.proc(logits.clone(), params)
        # row 1 (final turn) penalized more than row 0 (opening)
        assert out[1, 5].item() < out[0, 5].item()
