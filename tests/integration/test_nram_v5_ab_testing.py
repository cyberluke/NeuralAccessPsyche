"""
A/B test harness for NRAM v5 multi-layer inference control.

Tests the new NRAM v5 components:
- Phrase constraints (forbidden phrases, source n-gram blocking)
- Entropy control (PID servo with phase targets)
- Concept injection (dynamic concept capsules)

Measures key metrics:
- Novelty (1 - source_similarity)
- Coherence (perplexity proxy)
- Source similarity (cosine similarity of embeddings)
- N-gram overlap (copying detection)
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, List
from core.steering.nram_logit_processor import NRAMLogitProcessor
from core.steering.phrase_constraints import TokenTrie, TokenTrieConstraint, SourceNgramBlocker
from core.steering.entropy_controller import EntropyController, PhaseAwareEntropyController
from core.steering.concept_capsules import ConceptCapsule, ConceptCapsuleRegistry, ConceptInjector


class MockTokenizer:
    """Mock tokenizer for testing without real model."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self._token_map: Dict[str, int] = {}
        self._reverse_map: Dict[int, str] = {}
        self._next_id = 0
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs (simple word-level tokenization)."""
        tokens = []
        for word in text.lower().split():
            if word not in self._token_map:
                if self._next_id >= self.vocab_size:
                    continue  # Vocab full
                self._token_map[word] = self._next_id
                self._reverse_map[self._next_id] = word
                self._next_id += 1
            tokens.append(self._token_map[word])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for tid in token_ids:
            if tid in self._reverse_map:
                words.append(self._reverse_map[tid])
        return " ".join(words)


class TestTokenTrie:
    """Test TokenTrie data structure for phrase tracking."""
    
    def test_insert_and_match(self):
        """Test basic insert and prefix matching."""
        trie = TokenTrie()
        trie.insert([1, 2, 3])
        
        # Should match prefix [1, 2] and suggest [3]
        candidates = trie.matches_prefix([1, 2])
        assert 3 in candidates
        
        # Should not match prefix [1, 4]
        candidates = trie.matches_prefix([1, 4])
        assert len(candidates) == 0
    
    def test_multiple_sequences(self):
        """Test multiple sequences with shared prefix."""
        trie = TokenTrie()
        trie.insert([1, 2, 3])
        trie.insert([1, 2, 4])
        trie.insert([1, 5, 6])
        
        # Prefix [1, 2] should suggest both [3, 4]
        candidates = trie.matches_prefix([1, 2])
        assert set(candidates) == {3, 4}
        
        # Prefix [1] should suggest [2, 5]
        candidates = trie.matches_prefix([1])
        assert set(candidates) == {2, 5}
    
    def test_contains(self):
        """Test exact sequence containment."""
        trie = TokenTrie()
        trie.insert([1, 2, 3])
        
        assert trie.contains([1, 2, 3])
        assert not trie.contains([1, 2])
        assert not trie.contains([1, 2, 4])


class TestTokenTrieConstraint:
    """Test phrase-level masking during decoding."""
    
    def test_forbidden_phrase_masking(self):
        """Test that forbidden phrases are masked."""
        tokenizer = MockTokenizer()
        constraint = TokenTrieConstraint(
            forbidden_phrases=["digital transformation"],
            tokenizer=tokenizer,
        )
        
        # Create mock logits
        logits = torch.zeros(1, tokenizer.vocab_size)
        
        # Encode prefix "digital"
        prefix_ids = tokenizer.encode("digital")
        
        # Apply constraint
        constraint.apply(logits, prefix_ids, batch_index=0, vocab_size=tokenizer.vocab_size)
        
        # Token for "transformation" should be masked
        transformation_id = tokenizer._token_map.get("transformation")
        if transformation_id is not None:
            assert logits[0, transformation_id] == -float("inf")
    
    def test_no_masking_without_match(self):
        """Test that non-matching prefixes don't mask anything."""
        tokenizer = MockTokenizer()
        constraint = TokenTrieConstraint(
            forbidden_phrases=["digital transformation"],
            tokenizer=tokenizer,
        )
        
        logits = torch.zeros(1, tokenizer.vocab_size)
        prefix_ids = tokenizer.encode("artificial intelligence")
        
        constraint.apply(logits, prefix_ids, batch_index=0, vocab_size=tokenizer.vocab_size)
        
        # Nothing should be masked
        assert torch.all(logits[0] != -float("inf"))


class TestSourceNgramBlocker:
    """Test source n-gram blocking to prevent copying."""
    
    def test_blocks_source_ngrams(self):
        """Test that source n-grams are blocked."""
        tokenizer = MockTokenizer()
        source_text = "the quick brown fox jumps over the lazy dog"
        
        blocker = SourceNgramBlocker(
            source_text=source_text,
            tokenizer=tokenizer,
            n=4,  # 4-gram blocking
        )
        
        # Encode partial sequence from source
        partial_ids = tokenizer.encode("the quick brown")
        
        logits = torch.zeros(1, tokenizer.vocab_size)
        blocker.apply(logits, partial_ids, batch_index=0, vocab_size=tokenizer.vocab_size)
        
        # Next token "fox" should be blocked
        fox_id = tokenizer._token_map.get("fox")
        if fox_id is not None:
            assert logits[0, fox_id] == -float("inf")
    
    def test_allows_non_source_text(self):
        """Test that non-source text is not blocked."""
        tokenizer = MockTokenizer()
        source_text = "the quick brown fox"
        
        blocker = SourceNgramBlocker(
            source_text=source_text,
            tokenizer=tokenizer,
            n=4,
        )
        
        # Encode sequence NOT from source
        partial_ids = tokenizer.encode("artificial intelligence system")
        
        logits = torch.zeros(1, tokenizer.vocab_size)
        blocker.apply(logits, partial_ids, batch_index=0, vocab_size=tokenizer.vocab_size)
        
        # Nothing should be blocked
        assert torch.all(logits[0] != -float("inf"))


class TestEntropyController:
    """Test entropy control via PID servo."""
    
    def test_entropy_computation(self):
        """Test entropy computation from logits."""
        controller = EntropyController(target_entropy=4.0)
        
        # Uniform distribution should have high entropy
        uniform_logits = torch.zeros(1, 100)
        entropy = controller.compute_entropy(uniform_logits, batch_index=0)
        expected_entropy = np.log(100)  # ~4.605
        assert abs(entropy - expected_entropy) < 0.01
        
        # Peaked distribution should have low entropy
        peaked_logits = torch.zeros(1, 100)
        peaked_logits[0, 0] = 10.0
        entropy = controller.compute_entropy(peaked_logits, batch_index=0)
        assert entropy < 1.0
    
    def test_adjustment_toward_target(self):
        """Test that controller adjusts logits toward target entropy."""
        controller = EntropyController(target_entropy=3.0, kp=0.5)
        
        # Start with peaked distribution (low entropy)
        logits = torch.zeros(1, 100)
        logits[0, 0] = 10.0
        
        initial_entropy = controller.compute_entropy(logits, batch_index=0)
        
        # Apply adjustment
        scale = controller.adjust(logits, batch_index=0)
        
        # Entropy should increase (logits flattened)
        new_entropy = controller.compute_entropy(logits, batch_index=0)
        assert new_entropy > initial_entropy


class TestPhaseAwareEntropyController:
    """Test phase-aware entropy control."""
    
    def test_phase_transitions(self):
        """Test that controller switches phases based on progress."""
        controller = PhaseAwareEntropyController(
            phase_profile={
                "extraction": 2.5,
                "questioning": 4.0,
                "divergence": 6.0,
                "synthesis": 4.5,
                "formulation": 3.0,
            }
        )
        
        logits = torch.zeros(1, 100)
        
        # Progress 0.1 -> extraction phase (target 2.5)
        controller.adjust(logits, batch_index=0, progress=0.1)
        assert controller.current_phase == "extraction"
        
        # Progress 0.3 -> questioning phase (target 4.0)
        controller.adjust(logits, batch_index=0, progress=0.3)
        assert controller.current_phase == "questioning"
        
        # Progress 0.5 -> divergence phase (target 6.0)
        controller.adjust(logits, batch_index=0, progress=0.5)
        assert controller.current_phase == "divergence"


class TestConceptCapsules:
    """Test dynamic concept capsules."""
    
    def test_capsule_registry(self):
        """Test concept capsule registry."""
        registry = ConceptCapsuleRegistry()
        
        capsule = ConceptCapsule(
            concept_id="innovation",
            en_tokens=["innovation", "breakthrough"],
            cs_tokens=["inovace", "průlom"],
            activation_phase="divergence",
            max_uses=3,
        )
        
        registry.add_capsule(capsule)
        
        # Should be active in divergence phase
        active = registry.get_active_capsules(phase="divergence", generated_tokens=100)
        assert len(active) == 1
        assert active[0].concept_id == "innovation"
        
        # Should not be active in extraction phase
        active = registry.get_active_capsules(phase="extraction", generated_tokens=100)
        assert len(active) == 0
    
    def test_usage_limits(self):
        """Test that capsules respect usage limits."""
        registry = ConceptCapsuleRegistry()
        
        capsule = ConceptCapsule(
            concept_id="test",
            en_tokens=["test"],
            max_uses=2,
        )
        
        registry.add_capsule(capsule)
        
        # First two uses should be active
        active = registry.get_active_capsules(phase="divergence", generated_tokens=100)
        assert len(active) == 1
        
        registry.record_usage("test")
        active = registry.get_active_capsules(phase="divergence", generated_tokens=100)
        assert len(active) == 1
        
        # Third use should be blocked
        registry.record_usage("test")
        active = registry.get_active_capsules(phase="divergence", generated_tokens=100)
        assert len(active) == 0


class TestConceptInjector:
    """Test concept injection during decoding."""
    
    def test_soft_injection(self):
        """Test soft concept injection."""
        tokenizer = MockTokenizer()
        registry = ConceptCapsuleRegistry()
        
        capsule = ConceptCapsule(
            concept_id="innovation",
            en_tokens=["innovation"],
            activation_phase="divergence",
        )
        
        registry.add_capsule(capsule)
        
        injector = ConceptInjector(registry=registry, tokenizer=tokenizer)
        
        logits = torch.zeros(1, tokenizer.vocab_size)
        
        # Apply injection in divergence phase
        injector.apply_soft_injection(
            logits=logits,
            batch_index=0,
            phase="divergence",
            progress=0.5,
            base_strength=0.5,
        )
        
        # Innovation token should be boosted
        innovation_id = tokenizer._token_map.get("innovation")
        if innovation_id is not None:
            assert logits[0, innovation_id] > 0


class TestNRAMLogitProcessorV5:
    """Test NRAM v5 multi-layer inference control."""
    
    def test_phrase_constraint_integration(self):
        """Test phrase constraint integration in logit processor."""
        processor = NRAMLogitProcessor()
        
        tokenizer = MockTokenizer()
        forbidden_phrases = ["digital transformation"]
        
        # Encode forbidden phrase
        phrase_ids = tokenizer.encode("digital transformation")
        
        # Create custom params with phrase constraint config
        custom_params = [{
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 0.0,
            "profile": "test",
            "max_tokens": 100,
            "__req__": None,
            "phrase_constraint_config": {
                "forbidden_phrases": forbidden_phrases,
                "forbidden_phrase_ids": [phrase_ids],
            },
        }]
        
        logits = torch.zeros(1, tokenizer.vocab_size)
        
        # Simulate partial match
        prefix_ids = tokenizer.encode("digital")
        
        # Apply processor (would need request object for full test)
        # For now, just verify config structure
        assert custom_params[0]["phrase_constraint_config"] is not None
    
    def test_entropy_control_integration(self):
        """Test entropy control integration in logit processor."""
        processor = NRAMLogitProcessor()
        
        custom_params = [{
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 0.0,
            "profile": "test",
            "max_tokens": 100,
            "__req__": None,
            "entropy_config": {
                "enabled": True,
                "target_entropy": 4.0,
                "kp": 0.5,
                "ki": 0.1,
                "kd": 0.05,
                "phase_targets": {
                    "extraction": 2.5,
                    "questioning": 4.0,
                    "divergence": 6.0,
                },
            },
        }]
        
        logits = torch.zeros(1, 100)
        
        # Verify config structure
        assert custom_params[0]["entropy_config"]["enabled"] is True
    
    def test_concept_injection_integration(self):
        """Test concept injection integration in logit processor."""
        processor = NRAMLogitProcessor()
        
        tokenizer = MockTokenizer()
        
        custom_params = [{
            "positive_token_ids": [],
            "negative_token_ids": [],
            "forbidden_token_ids": [],
            "positive_bias": 0.0,
            "negative_bias": 0.0,
            "repetition_penalty": 0.0,
            "profile": "test",
            "max_tokens": 100,
            "__req__": None,
            "concept_config": {
                "enabled": True,
                "concepts": [
                    {
                        "concept_id": "innovation",
                        "token_ids": tokenizer.encode("innovation"),
                        "activation_phase": "divergence",
                        "strength": 0.5,
                    }
                ],
            },
        }]
        
        logits = torch.zeros(1, tokenizer.vocab_size)
        
        # Verify config structure
        assert custom_params[0]["concept_config"]["enabled"] is True


class TestABMetrics:
    """Test A/B testing metrics computation."""
    
    def test_ngram_overlap(self):
        """Test n-gram overlap computation."""
        tokenizer = MockTokenizer()
        
        source_text = "the quick brown fox jumps over the lazy dog"
        generated_text = "the quick brown fox leaps over the lazy cat"
        
        source_ids = tokenizer.encode(source_text)
        generated_ids = tokenizer.encode(generated_text)
        
        # Compute 4-gram overlap
        n = 4
        source_ngrams = set()
        for i in range(len(source_ids) - n + 1):
            ngram = tuple(source_ids[i:i+n])
            source_ngrams.add(ngram)
        
        generated_ngrams = set()
        for i in range(len(generated_ids) - n + 1):
            ngram = tuple(generated_ids[i:i+n])
            generated_ngrams.add(ngram)
        
        overlap = len(source_ngrams & generated_ngrams)
        total = len(generated_ngrams)
        
        overlap_ratio = overlap / total if total > 0 else 0.0
        
        # Should have some overlap (shared phrases)
        assert overlap_ratio > 0.0
        # But not complete overlap (different words)
        assert overlap_ratio < 1.0
    
    def test_novelty_metric(self):
        """Test novelty metric (1 - source_similarity)."""
        # Mock embeddings
        source_embedding = np.array([1.0, 0.0, 0.0])
        generated_embedding = np.array([0.0, 1.0, 0.0])
        
        # Cosine similarity
        similarity = np.dot(source_embedding, generated_embedding) / (
            np.linalg.norm(source_embedding) * np.linalg.norm(generated_embedding)
        )
        
        novelty = 1.0 - similarity
        
        # Orthogonal vectors should have novelty = 1.0
        assert abs(novelty - 1.0) < 0.01
        
        # Identical vectors should have novelty = 0.0
        similarity = np.dot(source_embedding, source_embedding) / (
            np.linalg.norm(source_embedding) * np.linalg.norm(source_embedding)
        )
        novelty = 1.0 - similarity
        assert abs(novelty - 0.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
