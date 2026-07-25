"""Phase 6: persona policy compiler determinism and bias bounds."""
import pytest

from core.contracts.nram import NRAMState
from core.persona.compiler import compile_policy, policy_hash
from core.persona.profiles import VISIONARY_PSYCHEDELIC_KEYNOTE


class TestPolicyCompilerDeterminism:
    def test_identical_inputs_produce_identical_policy(self):
        p1 = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE, max_tokens=512)
        p2 = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE, max_tokens=512)
        assert p1 == p2
        assert policy_hash(p1) == policy_hash(p2)

    def test_different_inputs_produce_different_hash(self):
        p1 = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE, max_tokens=512)
        other = VISIONARY_PSYCHEDELIC_KEYNOTE.model_copy(deep=True)
        other.visionary_intensity = 0.1
        p2 = compile_policy(other, max_tokens=512)
        assert policy_hash(p1) != policy_hash(p2)


class TestBiasBounds:
    def test_positive_bias_within_conservative_range(self):
        policy = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE)
        assert 0.0 <= policy.positive_bias <= 1.2

    def test_negative_bias_within_conservative_range(self):
        policy = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE)
        assert 0.0 <= policy.negative_bias <= 2.5

    def test_repetition_penalty_bounded(self):
        policy = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE)
        assert 0.0 <= policy.repetition_penalty <= 2.0

    def test_extreme_state_still_bounds_positive_bias(self):
        extreme = NRAMState(visionary_intensity=1.0, human_focus=1.0)
        policy = compile_policy(extreme)
        assert policy.positive_bias <= 1.2


class TestLexemeFamilies:
    def test_positive_lexemes_present(self):
        policy = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE)
        assert len(policy.positive_lexemes) > 0
        assert "human" in policy.positive_lexemes

    def test_negative_lexemes_are_corporate_filler(self):
        policy = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE)
        assert "synergy" in policy.negative_lexemes

    def test_developer_instruction_does_not_impersonate(self):
        policy = compile_policy(VISIONARY_PSYCHEDELIC_KEYNOTE)
        instr = policy.developer_instruction.lower()
        assert "do not impersonate" in instr
        assert "steve jobs" not in instr
        assert "apple" not in instr
