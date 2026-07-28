"""Tokenizer-aware bias compiler.

Uses the exact tokenizer for the target model to compile lexemes into
concrete token IDs. Skips ambiguous multi-token fragments. Never biases
punctuation, whitespace, or special tokens.
"""
from __future__ import annotations

import logging
import string
from typing import Any, Dict, List, Optional, Tuple

from core.contracts.nram import CompiledTokenPolicy, SteeringPolicy

logger = logging.getLogger(__name__)

# Characters that must never be biased
_UNSAFE_CHARS = set(string.punctuation + string.whitespace)


def _is_punctuation_or_whitespace(text: str) -> bool:
    """Check if a decoded token is only punctuation/whitespace."""
    return all(c in _UNSAFE_CHARS for c in text)


def _encode_lexeme(tokenizer: Any, lexeme: str) -> Tuple[List[int], List[str]]:
    """Encode a lexeme and return (token_ids, skip_reasons).

    For each lexeme:
    1. Encode the word as written.
    2. Encode it with a leading space.
    3. Prefer exact single-token representations.
    4. Skip ambiguous common subword fragments.
    5. Record skipped multi-token phrases.
    """
    ids: List[int] = []
    reasons: List[str] = []

    for variant in [lexeme, f" {lexeme}"]:
        try:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
        except Exception:
            reasons.append(f"encode_failed:{variant!r}")
            continue

        if not encoded:
            reasons.append(f"empty_encoding:{variant!r}")
            continue

        if len(encoded) == 1:
            tid = encoded[0]
            try:
                decoded = tokenizer.decode([tid])
            except Exception:
                decoded = variant

            # Never bias punctuation, whitespace, or special tokens
            if _is_punctuation_or_whitespace(decoded):
                reasons.append(f"punctuation_or_whitespace:{decoded!r}")
                continue

            if decoded.strip() == "":
                reasons.append(f"whitespace_only:{decoded!r}")
                continue

            ids.append(tid)
        else:
            reasons.append(
                f"multi_token_skipped:{variant!r}->tokens={encoded}"
            )

    return ids, reasons


class TokenBiasCompiler:
    """Compiles a SteeringPolicy into concrete token IDs using a tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = len(tokenizer)
            except Exception:
                vocab_size = 152064  # Qwen-family default
        self._vocab_size: int = int(vocab_size)

    def compile(self, policy: SteeringPolicy) -> CompiledTokenPolicy:
        """Compile lexemes to token IDs with diagnostics."""
        pos_ids, pos_skipped = self._compile_lexemes(policy.positive_lexemes)
        neg_ids, neg_skipped = self._compile_lexemes(policy.negative_lexemes)
        forb_ids, forb_skipped = self._compile_lexemes(policy.forbidden_lexemes)

        logger.info(
            f"TokenBiasCompiler: +{len(pos_ids)} pos, -{len(neg_ids)} neg, "
            f"x{len(forb_ids)} forbidden. "
            f"Skipped: pos={pos_skipped}, neg={neg_skipped}, forb={forb_skipped}"
        )

        return CompiledTokenPolicy(
            positive_token_ids=pos_ids,
            negative_token_ids=neg_ids,
            forbidden_token_ids=forb_ids,
            positive_bias=policy.positive_bias,
            negative_bias=policy.negative_bias,
            repetition_penalty=policy.repetition_penalty,
        )

    def compile_with_diagnostics(
        self, policy: SteeringPolicy
    ) -> Tuple[CompiledTokenPolicy, Dict[str, Any]]:
        """Compile and return diagnostics for the /v1/nram/token-policy endpoint."""
        diagnostics: Dict[str, List[Dict[str, Any]]] = {
            "positive": [],
            "negative": [],
            "forbidden": [],
        }

        pos_ids = self._compile_with_diag(policy.positive_lexemes, diagnostics["positive"])
        neg_ids = self._compile_with_diag(policy.negative_lexemes, diagnostics["negative"])
        forb_ids = self._compile_with_diag(policy.forbidden_lexemes, diagnostics["forbidden"])

        compiled = CompiledTokenPolicy(
            positive_token_ids=pos_ids,
            negative_token_ids=neg_ids,
            forbidden_token_ids=forb_ids,
            positive_bias=policy.positive_bias,
            negative_bias=policy.negative_bias,
            repetition_penalty=policy.repetition_penalty,
        )
        return compiled, diagnostics

    def _compile_lexemes(self, lexemes: List[str]) -> Tuple[List[int], List[str]]:
        all_ids: List[int] = []
        all_reasons: List[str] = []

        for lexeme in lexemes:
            ids, reasons = _encode_lexeme(self._tokenizer, lexeme)
            all_ids.extend(ids)
            all_reasons.extend(reasons)

        # Deduplicate and validate
        valid_ids = sorted(set(
            tid for tid in all_ids
            if 0 <= tid < self._vocab_size
        ))
        return valid_ids, all_reasons

    def _compile_with_diag(
        self, lexemes: List[str], diag_list: List[Dict[str, Any]]
    ) -> List[int]:
        all_ids: List[int] = []

        for lexeme in lexemes:
            ids, reasons = _encode_lexeme(self._tokenizer, lexeme)
            decoded_texts = []
            for tid in ids:
                try:
                    decoded_texts.append(self._tokenizer.decode([tid]))
                except Exception:
                    decoded_texts.append("?")

            diag_list.append({
                "lexeme": lexeme,
                "token_ids": ids,
                "decoded": decoded_texts,
                "skipped_reason": reasons if reasons else None,
            })
            all_ids.extend(ids)

        return sorted(set(
            tid for tid in all_ids
            if 0 <= tid < self._vocab_size
        ))
