"""Phase 7: tokenizer-aware bias compiler tests using a fake tokenizer."""
import pytest

from core.contracts.nram import SteeringPolicy
from core.steering.tokenizer_bias import TokenBiasCompiler


class FakeTokenizer:
    """Minimal tokenizer that maps whole words to single token IDs.

    Vocabulary:
      "human"->10, " human"->11, "design"->20, " design"->21,
      "synergy"->30, " synergy"->31, "the"->40, " the"->41
    Punctuation/whitespace decode to themselves. Multi-token words return 2 ids.
    """

    vocab_size = 100

    _encode_map = {
        "human": [10],
        " human": [11],
        "design": [20],
        " design": [21],
        "synergy": [30],
        " synergy": [31],
        "the": [40],
        " the": [41],
        # multi-token word (should be skipped)
        "blockchain": [50, 51],
        " blockchain": [50, 51],
        # punctuation-only token
        ".": [60],
        " .": [60],
    }
    _decode_map = {
        10: "human", 11: " human", 20: "design", 21: " design",
        30: "synergy", 31: " synergy", 40: "the", 41: " the",
        50: "block", 51: "chain", 60: ".",
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._encode_map.get(text, []))

    def decode(self, ids):
        return "".join(self._decode_map.get(i, "?") for i in ids)

    def __len__(self):
        return self.vocab_size


def make_policy(pos=None, neg=None, forb=None):
    return SteeringPolicy(
        positive_lexemes=pos or [],
        negative_lexemes=neg or [],
        forbidden_lexemes=forb or [],
        positive_bias=0.5,
        negative_bias=1.0,
        repetition_penalty=0.3,
    )


class TestTokenBiasCompiler:
    def setup_method(self):
        self.compiler = TokenBiasCompiler(FakeTokenizer())

    def test_single_token_lexeme_selected(self):
        policy = make_policy(pos=["human"])
        compiled = self.compiler.compile(policy)
        # both "human"(10) and " human"(11) are valid single tokens
        assert set(compiled.positive_token_ids) == {10, 11}

    def test_multi_token_lexeme_skipped(self):
        policy = make_policy(pos=["blockchain"])
        compiled = self.compiler.compile(policy)
        assert compiled.positive_token_ids == []

    def test_punctuation_never_biased(self):
        policy = make_policy(pos=["."])
        compiled = self.compiler.compile(policy)
        assert 60 not in compiled.positive_token_ids

    def test_ids_deduplicated(self):
        policy = make_policy(pos=["human", "human", "design"])
        compiled = self.compiler.compile(policy)
        assert len(compiled.positive_token_ids) == len(set(compiled.positive_token_ids))

    def test_ids_validated_against_vocab(self):
        policy = make_policy(pos=["human"])
        compiled = self.compiler.compile(policy)
        for tid in compiled.positive_token_ids:
            assert 0 <= tid < FakeTokenizer.vocab_size

    def test_bias_values_propagated(self):
        policy = make_policy(pos=["human"], neg=["synergy"])
        compiled = self.compiler.compile(policy)
        assert compiled.positive_bias == 0.5
        assert compiled.negative_bias == 1.0
        assert compiled.repetition_penalty == 0.3

    def test_diagnostics_include_skipped_reason(self):
        policy = make_policy(pos=["blockchain", "human"])
        _, diagnostics = self.compiler.compile_with_diagnostics(policy)
        entries = diagnostics["positive"]
        blockchain_entry = next(e for e in entries if e["lexeme"] == "blockchain")
        assert blockchain_entry["skipped_reason"]
        human_entry = next(e for e in entries if e["lexeme"] == "human")
        assert human_entry["token_ids"]
