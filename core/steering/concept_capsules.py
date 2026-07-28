"""Dynamic concept capsules for NRAM v5.

This module implements semantic steering through pre-computed concept units:
1. ConceptCapsule: Semantic unit with multiple lexical forms and activation rules
2. ConceptCapsuleRegistry: Manages capsules for a generation request
3. ConceptInjector: Applies concept injection during decoding

Concept capsules enable:
- Phase-aware concept activation (only inject at appropriate generation stages)
- Multi-lingual support (Czech/English lexical forms)
- Usage limits (prevent over-repetition of concepts)
- Forbidden frame prevention (block unwanted semantic frames)

This replaces static token bias lists with dynamic, context-aware semantic steering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ConceptCapsule:
    """A semantic unit with multiple lexical forms and activation rules.
    
    Concept capsules represent abstract concepts (e.g., "ambient computing",
    "biological memory") with multiple surface forms across languages and
    synonyms. They can be activated at specific generation phases and have
    usage limits to prevent over-repetition.
    
    Attributes:
        concept_id: Unique identifier (e.g., "ambient_computing")
        en_tokens: English token forms (e.g., ["ambient", "computing"])
        cs_tokens: Czech token forms (e.g., ["ambientní", "výpočetní"])
        synonyms: Additional synonyms (e.g., ["ubiquitous", "everywhere"])
        embedding: Optional embedding centroid (for semantic similarity)
        activation_phase: Phase when concept should be injected (None=always)
        max_uses: Maximum times this concept can appear in generation
        min_distance: Minimum semantic distance from source (0.0-1.0)
        forbidden_frames: Semantic frames to prevent (e.g., ["better chatbot"])
        revelation_tokens: Key tokens that signal concept revelation
    """
    
    concept_id: str
    en_tokens: List[str] = field(default_factory=list)
    cs_tokens: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Embedding centroid
    activation_phase: Optional[str] = None  # "revelation", "synthesis", None=always
    max_uses: int = 3
    min_distance: float = 0.3
    forbidden_frames: List[str] = field(default_factory=list)
    revelation_tokens: List[str] = field(default_factory=list)
    
    def get_all_token_forms(self) -> List[str]:
        """Get all lexical forms (English + Czech + synonyms).
        
        Returns:
            List of all token forms for this concept
        """
        return self.en_tokens + self.cs_tokens + self.synonyms
    
    def matches_phase(self, phase: str) -> bool:
        """Check if concept should be active in given phase.
        
        Args:
            phase: Current generation phase
        
        Returns:
            True if concept should be active, False otherwise
        """
        if self.activation_phase is None:
            return True  # Always active
        return self.activation_phase == phase


class ConceptCapsuleRegistry:
    """Manages concept capsules for a generation request.
    
    Tracks capsule usage, enforces activation rules, and provides
    active capsules for injection at each generation step.
    
    Example:
        registry = ConceptCapsuleRegistry()
        registry.load_from_config({
            "required_concepts": [
                {
                    "concept_id": "ambient_computing",
                    "en_tokens": ["ambient", "computing"],
                    "cs_tokens": ["ambientní", "výpočetní"],
                    "activation_phase": "divergence",
                    "max_uses": 2,
                }
            ]
        })
        
        # During generation
        active = registry.get_active_capsules(phase="divergence", generated_tokens=100)
        for capsule in active:
            # Inject capsule tokens
            registry.record_usage(capsule.concept_id)
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self.capsules: Dict[str, ConceptCapsule] = {}
        self.usage_counts: Dict[str, int] = {}
    
    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load capsules from request config.
        
        Args:
            config: Request config with "required_concepts" list
        """
        for capsule_data in config.get("required_concepts", []):
            capsule = ConceptCapsule(**capsule_data)
            self.capsules[capsule.concept_id] = capsule
            self.usage_counts[capsule.concept_id] = 0
    
    def add_capsule(self, capsule: ConceptCapsule) -> None:
        """Add a single capsule to registry.
        
        Args:
            capsule: Concept capsule to add
        """
        self.capsules[capsule.concept_id] = capsule
        self.usage_counts[capsule.concept_id] = 0
    
    def get_active_capsules(
        self,
        phase: str,
        generated_tokens: int,
    ) -> List[ConceptCapsule]:
        """Return capsules eligible for injection at current phase.
        
        Filters capsules by:
        - Phase activation rules
        - Usage limits (max_uses not exceeded)
        
        Args:
            phase: Current generation phase
            generated_tokens: Number of tokens generated so far
        
        Returns:
            List of active concept capsules
        """
        active = []
        for capsule in self.capsules.values():
            # Check usage limit
            if self.usage_counts[capsule.concept_id] >= capsule.max_uses:
                continue
            
            # Check phase activation
            if not capsule.matches_phase(phase):
                continue
            
            active.append(capsule)
        
        return active
    
    def record_usage(self, concept_id: str) -> None:
        """Record that a concept was used in generation.
        
        Args:
            concept_id: ID of concept that was used
        """
        if concept_id in self.usage_counts:
            self.usage_counts[concept_id] += 1
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for all capsules.
        
        Returns:
            Dict mapping concept_id to usage count
        """
        return self.usage_counts.copy()
    
    def reset(self) -> None:
        """Reset all usage counts."""
        for concept_id in self.usage_counts:
            self.usage_counts[concept_id] = 0


class ConceptInjector:
    """Applies concept injection during decoding.
    
    Three injection types:
    1. Hard injection: Force specific tokens (structure, section markers)
    2. Soft injection: Time-windowed bias on concept tokens
    3. Latent injection: Inject concept as hidden-state vector (Sprint 2)
    
    Example:
        injector = ConceptInjector(
            registry=registry,
            tokenizer=tokenizer,
        )
        
        # During generation
        progress = generated_tokens / max_tokens
        injector.apply_soft_injection(
            logits=logits,
            batch_index=0,
            phase="divergence",
            progress=progress,
        )
    """
    
    def __init__(
        self,
        registry: ConceptCapsuleRegistry,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize concept injector.
        
        Args:
            registry: Concept capsule registry
            tokenizer: Tokenizer for encoding concept tokens
        """
        self.registry = registry
        self.tokenizer = tokenizer
        
        # Pre-compute token IDs for each capsule
        self.capsule_token_ids: Dict[str, List[int]] = {}
        if tokenizer:
            self._precompute_token_ids()
    
    def _precompute_token_ids(self) -> None:
        """Pre-compute token IDs for all capsules."""
        for concept_id, capsule in self.registry.capsules.items():
            token_ids = []
            for token_form in capsule.get_all_token_forms():
                try:
                    ids = self.tokenizer.encode(token_form, add_special_tokens=False)
                    token_ids.extend(ids)
                except Exception:
                    pass
            self.capsule_token_ids[concept_id] = list(set(token_ids))  # Deduplicate
    
    def _phase_strength_curve(self, progress: float) -> float:
        """Compute injection strength based on generation progress.
        
        Ramps up in first 20%, holds steady, ramps down in last 20%.
        
        Args:
            progress: Generation progress (0.0 to 1.0)
        
        Returns:
            Injection strength (0.0 to 0.9)
        """
        if progress < 0.2:
            return progress / 0.2 * 0.9  # 0.0 → 0.9
        elif progress < 0.8:
            return 0.9  # Hold steady
        else:
            return (1.0 - progress) / 0.2 * 0.9  # 0.9 → 0.0
    
    def apply_soft_injection(
        self,
        logits: Any,
        batch_index: int,
        phase: str,
        progress: float,
        base_strength: float = 0.5,
    ) -> None:
        """Apply soft concept injection (time-windowed bias).
        
        Adds bias to concept token logits based on phase and progress.
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            batch_index: Index in batch to apply injection
            phase: Current generation phase
            progress: Generation progress (0.0 to 1.0)
            base_strength: Base injection strength (0.0 to 1.0)
        """
        # Get active capsules for current phase
        active_capsules = self.registry.get_active_capsules(
            phase=phase,
            generated_tokens=int(progress * 1000),  # Approximate
        )
        
        if not active_capsules:
            return
        
        # Compute phase-dependent strength
        phase_strength = self._phase_strength_curve(progress)
        injection_strength = base_strength * phase_strength
        
        # Apply bias to concept tokens
        for capsule in active_capsules:
            token_ids = self.capsule_token_ids.get(capsule.concept_id, [])
            for token_id in token_ids:
                if 0 <= token_id < logits.shape[-1]:
                    logits[batch_index, token_id] += injection_strength
            
            # Record usage (approximate — actual usage tracked by token detection)
            # In production, would detect actual token generation
    
    def apply_hard_injection(
        self,
        logits: Any,
        batch_index: int,
        concept_id: str,
        strength: float = 10.0,
    ) -> None:
        """Apply hard concept injection (force specific tokens).
        
        Use sparingly — overuse destroys grammar. Best for:
        - Structure enforcement (JSON, section markers)
        - Phase transitions
        - Critical concept revelation
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            batch_index: Index in batch to apply injection
            concept_id: ID of concept to force
            strength: Injection strength (default: 10.0 for near-certain)
        """
        token_ids = self.capsule_token_ids.get(concept_id, [])
        for token_id in token_ids:
            if 0 <= token_id < logits.shape[-1]:
                logits[batch_index, token_id] = strength
    
    def get_injection_telemetry(self) -> Dict[str, Any]:
        """Get telemetry for concept injection.
        
        Returns:
            Dict with usage stats and active capsules
        """
        return {
            "usage_counts": self.registry.get_usage_stats(),
            "total_capsules": len(self.registry.capsules),
        }


# Pre-defined concept capsules for common use cases
DEFAULT_CONCEPT_CAPSULES = [
    ConceptCapsule(
        concept_id="human_experience",
        en_tokens=["human", "experience", "person", "user"],
        cs_tokens=["člověk", "zkušenost", "osoba", "uživatel"],
        synonyms=["individual", "people", "person"],
        activation_phase=None,  # Always active
        max_uses=5,
    ),
    ConceptCapsule(
        concept_id="future_vision",
        en_tokens=["future", "tomorrow", "next", "ahead"],
        cs_tokens=["budoucnost", "zítra", "další", "kupředu"],
        synonyms=["horizon", "prospect", "vision"],
        activation_phase="divergence",
        max_uses=3,
    ),
    ConceptCapsule(
        concept_id="product_obsession",
        en_tokens=["product", "craft", "build", "create"],
        cs_tokens=["produkt", "řemeslo", "vytvořit", "tvořit"],
        synonyms=["artifact", "solution", "tool"],
        activation_phase="synthesis",
        max_uses=4,
    ),
]
