"""Steering policy — conservative bias ranges and safety constraints."""
from __future__ import annotations

# Conservative bias ranges
MAX_POSITIVE_BIAS = 1.2
MAX_NEGATIVE_BIAS = 2.5
MAX_REPETITION_PENALTY = 2.0

# Forbidden token mask: negative infinity only for proven-safe tokens.
# Never bias punctuation, whitespace, or special tokens.
FORBIDDEN_MASK_VALUE = float("-inf")
