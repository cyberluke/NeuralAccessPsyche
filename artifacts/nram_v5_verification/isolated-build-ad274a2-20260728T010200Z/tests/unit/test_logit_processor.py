"""Phase 8: NRAM logit processor unit tests (no GPU required).

Proves the processor modifies logits BEFORE sampling:
- positive token logits increase
- negative token logits decrease
- forbidden tokens become -inf
- recent tokens receive repetition penalty
- out-of-range IDs ignored
- invalid param types don't crash
- batch isolation (one row cannot alter another)
- serialization/deserialization preserves behavior
"""
import math

import pytest

torch = pytest.importorskip("torch")

from core.steering.nram_logit_processor import NRAMLogitProcessor


class FakeRequest:
    def __init__(self, output_ids):
        self.output_ids = output_ids


class TestLogitProcessor:
    def setup_method(self):
        self.proc = NRAMLogitProcessor()

    def test_positive_bias_increases_logits(self):
        logits = torch.zeros((1, 100))
        params = [{"positive_token_ids": [5, 6], "positive_bias": 1.0}]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(1.0)
        assert out[0, 6].item() == pytest.approx(1.0)
        assert out[0, 7].item() == pytest.approx(0.0)

    def test_negative_bias_decreases_logits(self):
        logits = torch.zeros((1, 100))
        params = [{"negative_token_ids": [10], "negative_bias": 2.0}]
        out = self.proc(logits.clone(), params)
        assert out[0, 10].item() == pytest.approx(-2.0)

    def test_forbidden_tokens_negative_infinity(self):
        logits = torch.zeros((1, 100))
        params = [{"forbidden_token_ids": [42]}]
        out = self.proc(logits.clone(), params)
        assert out[0, 42].item() == -math.inf

    def test_repetition_penalty_on_recent_tokens(self):
        logits = torch.zeros((1, 100))
        req = FakeRequest(output_ids=[3, 4, 5])
        params = [{"repetition_penalty": 1.5, "__req__": req}]
        out = self.proc(logits.clone(), params)
        assert out[0, 3].item() == pytest.approx(-1.5)
        assert out[0, 4].item() == pytest.approx(-1.5)
        assert out[0, 99].item() == pytest.approx(0.0)

    def test_out_of_range_ids_ignored(self):
        logits = torch.zeros((1, 100))
        params = [{"positive_token_ids": [-1, 100, 5000, 5], "positive_bias": 1.0}]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(1.0)
        # No crash, only in-range id applied.

    def test_invalid_param_types_do_not_crash(self):
        logits = torch.zeros((1, 100))
        params = [{
            "positive_token_ids": "not-a-list",
            "positive_bias": "not-a-number",
            "negative_token_ids": None,
        }]
        out = self.proc(logits.clone(), params)
        assert out.shape == (1, 100)

    def test_batch_isolation(self):
        logits = torch.zeros((2, 100))
        params = [
            {"positive_token_ids": [5], "positive_bias": 1.0},
            {},  # second row untouched
        ]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(1.0)
        assert out[1, 5].item() == pytest.approx(0.0)

    def test_empty_params_returns_logits(self):
        logits = torch.zeros((1, 100))
        out = self.proc(logits.clone(), None)
        assert torch.equal(out, logits)

    def test_bias_clamped_to_bounds(self):
        logits = torch.zeros((1, 100))
        # Excessive bias must be clamped to 1.2 max for positive.
        params = [{"positive_token_ids": [5], "positive_bias": 999.0}]
        out = self.proc(logits.clone(), params)
        assert out[0, 5].item() == pytest.approx(1.2)


class TestProcessorSerialization:
    def test_dill_roundtrip_preserves_behavior(self):
        dill = pytest.importorskip("dill")
        proc = NRAMLogitProcessor()
        blob = dill.dumps(proc)
        restored = dill.loads(blob)

        logits = torch.zeros((1, 100))
        params = [{"positive_token_ids": [5], "positive_bias": 1.0,
                   "forbidden_token_ids": [9]}]
        original = proc(logits.clone(), params)
        roundtripped = restored(logits.clone(), params)
        assert torch.equal(original, roundtripped)

    def test_serialize_processor_helper_returns_hex(self):
        from core.steering.serialization import serialize_processor
        import json
        result = serialize_processor(NRAMLogitProcessor)
        parsed = json.loads(result)
        assert "callable" in parsed
        # hex string must be decodable
        bytes.fromhex(parsed["callable"])


class TestForcedTokenProof:
    """The hard evidence: mask every token except one, prove that one is selected."""

    def test_forced_token_is_argmax(self):
        logits = torch.zeros((1, 100))
        forced_id = 7
        all_but_forced = [i for i in range(100) if i != forced_id]
        params = [{"forbidden_token_ids": all_but_forced}]
        out = NRAMLogitProcessor()(logits.clone(), params)
        # After masking, the forced token must be the unique argmax.
        assert torch.argmax(out[0]).item() == forced_id
        assert out[0, forced_id].item() == pytest.approx(0.0)
        for i in all_but_forced:
            assert out[0, i].item() == -math.inf
