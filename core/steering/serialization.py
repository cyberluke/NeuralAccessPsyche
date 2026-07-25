"""Serialization utilities for the trusted custom logit processor."""
from __future__ import annotations

import json
from typing import Any, Dict


def serialize_processor(processor_class: type) -> str:
    """Serialize a logit processor class with dill for SGLang.

    Only the server itself may call this. Never accept client-supplied
    serialized processors.
    """
    import dill

    return json.dumps({"callable": dill.dumps(processor_class).hex()})


def build_custom_params(
    positive_token_ids: list[int],
    negative_token_ids: list[int],
    forbidden_token_ids: list[int],
    positive_bias: float,
    negative_bias: float,
    repetition_penalty: float,
    profile: str,
    max_tokens: int = 0,
) -> Dict[str, Any]:
    """Build the trusted custom_params dict for SGLang.

    max_tokens enables the Phase 13 per-token phase schedule inside the
    processor. 0 disables scheduling (constant request-level biases).
    """
    return {
        "positive_token_ids": positive_token_ids,
        "negative_token_ids": negative_token_ids,
        "forbidden_token_ids": forbidden_token_ids,
        "positive_bias": positive_bias,
        "negative_bias": negative_bias,
        "repetition_penalty": repetition_penalty,
        "profile": profile,
        "max_tokens": max_tokens,
    }
