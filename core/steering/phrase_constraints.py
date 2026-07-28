"""Phrase-aware masking and anti-copy constraints for NRAM v5.

This module implements:
1. TokenTrie: Trie data structure for tracking multi-token sequences
2. TokenTrieConstraint: Phrase-level masking during decoding
3. SourceNgramBlocker: Prevents copying of n-grams from source document

These constraints operate at the logit level, masking tokens that would
complete forbidden phrases or copy source n-grams BEFORE sampling.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class TokenTrie:
    """Trie for tracking multi-token sequences.
    
    Used for:
    - Forbidden phrase detection (mask tokens that would complete clichés)
    - Source n-gram blocking (prevent copying from source document)
    - Allowed phrase enforcement (force specific phrases)
    
    Example:
        trie = TokenTrie()
        trie.insert([1, 2, 3])  # Insert sequence [1, 2, 3]
        trie.matches_prefix([1, 2])  # Returns [3] (would complete forbidden phrase)
    """
    
    def __init__(self):
        self.root: Dict[int, Dict] = {}  # token_id -> children
        self._size = 0
    
    def insert(self, token_ids: List[int]) -> None:
        """Insert a sequence into the trie.
        
        Args:
            token_ids: List of token IDs representing a phrase
        """
        if not token_ids:
            return
        
        node = self.root
        for token_id in token_ids:
            if token_id not in node:
                node[token_id] = {}
            node = node[token_id]
        node["$"] = True  # End marker
        self._size += 1
    
    def matches_prefix(self, recent_tokens: List[int]) -> List[int]:
        """Return token IDs that would complete a forbidden phrase.
        
        Args:
            recent_tokens: Recent output token IDs (typically last n-1 tokens)
        
        Returns:
            List of token IDs that would complete a forbidden phrase.
            Empty list if no match.
        
        Example:
            If trie contains [1, 2, 3] and [1, 2, 4]:
            matches_prefix([1, 2]) returns [3, 4]
        """
        if not recent_tokens:
            return []
        
        node = self.root
        for token_id in recent_tokens:
            if token_id not in node:
                return []  # No match
            node = node[token_id]
        
        # Collect all possible next tokens (excluding end marker)
        candidates = []
        for next_token, children in node.items():
            if next_token != "$":
                candidates.append(next_token)
        
        return candidates
    
    def contains(self, token_ids: List[int]) -> bool:
        """Check if exact sequence exists in trie.
        
        Args:
            token_ids: List of token IDs to check
        
        Returns:
            True if sequence exists, False otherwise
        """
        if not token_ids:
            return False
        
        node = self.root
        for token_id in token_ids:
            if token_id not in node:
                return False
            node = node[token_id]
        
        return "$" in node
    
    def __len__(self) -> int:
        return self._size


class TokenTrieConstraint:
    """Applies phrase-level masking during decoding.
    
    Prevents tokens that would complete forbidden phrases from being sampled.
    Can also enforce specific phrases (allowed_trie).
    
    Example:
        constraint = TokenTrieConstraint(
            forbidden_phrases=["digitální transformace", "klíčovým prvkem"],
            tokenizer=tokenizer,
        )
        constraint.apply(logits, recent_tokens=[1, 2], batch_index=0)
        # Tokens that would complete forbidden phrases are masked to -inf
    """
    
    def __init__(
        self,
        forbidden_phrases: Optional[List[str]] = None,
        allowed_phrases: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize phrase constraint.
        
        Args:
            forbidden_phrases: List of phrases to prevent (as strings)
            allowed_phrases: List of phrases to enforce (as strings)
            tokenizer: Tokenizer for encoding phrases
        """
        self.forbidden_trie = TokenTrie()
        self.allowed_trie = TokenTrie()
        self.tokenizer = tokenizer
        
        if tokenizer and forbidden_phrases:
            for phrase in forbidden_phrases:
                token_ids = self._encode_phrase(phrase)
                if token_ids:
                    self.forbidden_trie.insert(token_ids)
        
        if tokenizer and allowed_phrases:
            for phrase in allowed_phrases:
                token_ids = self._encode_phrase(phrase)
                if token_ids:
                    self.allowed_trie.insert(token_ids)
    
    def _encode_phrase(self, phrase: str) -> List[int]:
        """Encode phrase to token IDs.
        
        Args:
            phrase: Text phrase to encode
        
        Returns:
            List of token IDs, or empty list if encoding fails
        """
        if not self.tokenizer:
            return []
        
        try:
            # Use encode() without special tokens for phrase matching
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            return token_ids if token_ids else []
        except Exception:
            return []
    
    def apply(
        self,
        logits: Any,
        recent_tokens: List[int],
        batch_index: int,
        vocab_size: int,
    ) -> None:
        """Apply phrase-level masking to logits.
        
        Masks tokens that would complete forbidden phrases to -inf.
        If allowed_trie is non-empty, only allows tokens from allowed_trie.
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            recent_tokens: Recent output token IDs
            batch_index: Index in batch to apply masking
            vocab_size: Vocabulary size for bounds checking
        """
        # Forbidden phrase masking
        forbidden_next = self.forbidden_trie.matches_prefix(recent_tokens)
        for token_id in forbidden_next:
            if 0 <= token_id < vocab_size:
                logits[batch_index, token_id] = -float("inf")
        
        # Allowed phrase enforcement (if configured)
        if len(self.allowed_trie) > 0:
            allowed_next = self.allowed_trie.matches_prefix(recent_tokens)
            if allowed_next:
                # Mask all tokens except allowed ones
                allowed_set = set(allowed_next)
                for token_id in range(vocab_size):
                    if token_id not in allowed_set:
                        logits[batch_index, token_id] = -float("inf")


class SourceNgramBlocker:
    """Blocks tokens that would complete source n-grams.
    
    Prevents copying of n-grams from source document during decoding.
    This is a proactive measure: the model never sees the copied token
    as a valid option.
    
    Example:
        blocker = SourceNgramBlocker(
            source_text="Original document text...",
            tokenizer=tokenizer,
            n=8,  # Block 8-gram copying
        )
        blocker.apply(logits, output_ids=[1, 2, 3], batch_index=0)
    """
    
    def __init__(
        self,
        source_text: str,
        tokenizer: Any,
        n: int = 8,
    ):
        """Initialize source n-gram blocker.
        
        Args:
            source_text: Source document text to prevent copying from
            tokenizer: Tokenizer for encoding source text
            n: N-gram size to block (default: 8)
        """
        self.n = n
        self.source_trie = TokenTrie()
        self.tokenizer = tokenizer
        
        # Extract all n-grams from source
        source_tokens = self._encode_source(source_text)
        for i in range(len(source_tokens) - n + 1):
            ngram = source_tokens[i:i+n]
            self.source_trie.insert(ngram)
    
    def _encode_source(self, text: str) -> List[int]:
        """Encode source text to token IDs.
        
        Args:
            text: Source text to encode
        
        Returns:
            List of token IDs
        """
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return []
    
    def apply(
        self,
        logits: Any,
        output_ids: List[int],
        batch_index: int,
        vocab_size: int,
    ) -> None:
        """Apply source n-gram blocking to logits.
        
        Masks tokens that would complete a source n-gram to -inf.
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            output_ids: Generated token IDs so far
            batch_index: Index in batch to apply masking
            vocab_size: Vocabulary size for bounds checking
        """
        if len(output_ids) < self.n - 1:
            return  # Not enough tokens to match
        
        # Get recent tokens (last n-1 tokens)
        recent = output_ids[-(self.n-1):]
        forbidden_next = self.source_trie.matches_prefix(recent)
        
        for token_id in forbidden_next:
            if 0 <= token_id < vocab_size:
                logits[batch_index, token_id] = -float("inf")


class PhraseConstraintManager:
    """Manages multiple phrase constraints for a generation request.
    
    Combines:
    - Forbidden phrase masking (clichés, corporate jargon)
    - Source n-gram blocking (prevent copying)
    - Allowed phrase enforcement (optional)
    
    Example:
        manager = PhraseConstraintManager(
            forbidden_phrases=["digitální transformace", "klíčovým prvkem"],
            source_text="Original document...",
            tokenizer=tokenizer,
        )
        manager.apply(logits, output_ids=[1, 2, 3], batch_index=0)
    """
    
    def __init__(
        self,
        forbidden_phrases: Optional[List[str]] = None,
        allowed_phrases: Optional[List[str]] = None,
        source_text: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        source_ngram_size: int = 8,
    ):
        """Initialize phrase constraint manager.
        
        Args:
            forbidden_phrases: List of phrases to prevent
            allowed_phrases: List of phrases to enforce (optional)
            source_text: Source document to prevent copying from (optional)
            tokenizer: Tokenizer for encoding phrases
            source_ngram_size: N-gram size for source blocking (default: 8)
        """
        self.token_trie_constraint = TokenTrieConstraint(
            forbidden_phrases=forbidden_phrases,
            allowed_phrases=allowed_phrases,
            tokenizer=tokenizer,
        )
        
        self.source_blocker = None
        if source_text and tokenizer:
            self.source_blocker = SourceNgramBlocker(
                source_text=source_text,
                tokenizer=tokenizer,
                n=source_ngram_size,
            )
    
    def apply(
        self,
        logits: Any,
        output_ids: List[int],
        batch_index: int,
        vocab_size: int,
    ) -> None:
        """Apply all phrase constraints to logits.
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            output_ids: Generated token IDs so far
            batch_index: Index in batch to apply masking
            vocab_size: Vocabulary size for bounds checking
        """
        # Apply token trie constraint (forbidden/allowed phrases)
        self.token_trie_constraint.apply(
            logits=logits,
            recent_tokens=output_ids,
            batch_index=batch_index,
            vocab_size=vocab_size,
        )
        
        # Apply source n-gram blocker (if configured)
        if self.source_blocker:
            self.source_blocker.apply(
                logits=logits,
                output_ids=output_ids,
                batch_index=batch_index,
                vocab_size=vocab_size,
            )
