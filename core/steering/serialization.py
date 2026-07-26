"""Serialization utilities for the trusted custom logit processor."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def serialize_processor(processor_class: type) -> str:
    """Serialize a logit processor class with dill for SGLang.

    Only the server itself may call this. Never accept client-supplied
    serialized processors.
    
    Uses dill recurse mode to serialize the class by value (not by reference),
    so SGLang can deserialize it without needing the original module.
    """
    import dill

    # Serialize by value - class is embedded in the pickle, not referenced by module path
    # This prevents "ModuleNotFoundError: No module named 'core'" in SGLang
    # dill.settings is a dict, not a context manager
    old_recurse = dill.settings.get('recurse', False)
    dill.settings['recurse'] = True
    try:
        serialized = dill.dumps(processor_class)
    finally:
        dill.settings['recurse'] = old_recurse
    
    return json.dumps({"callable": serialized.hex()})


def build_custom_params(
    positive_token_ids: list[int],
    negative_token_ids: list[int],
    forbidden_token_ids: list[int],
    positive_bias: float,
    negative_bias: float,
    repetition_penalty: float,
    profile: str,
    max_tokens: int = 0,
    phenomenon_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build the trusted custom_params dict for SGLang.

    max_tokens enables the Phase 13 per-token phase schedule inside the
    processor. 0 disables scheduling (constant request-level biases).

    phenomenon_weights maps phenomenon id → weight (0.0–1.0). The logit
    processor uses these to apply concrete logit-level operations:
      overlap          → boost recent output token IDs
      forgetting       → extra repetition penalty on recent tokens
      looping          → reward recent tokens (perseveration)
      associative_jump → flatten distribution toward uniform
      synesthesia      → deterministic cross-activation noise
      dissolution      → scale logits toward zero
      insight          → periodic positive_bias spikes
    """
    result = {
        "positive_token_ids": positive_token_ids,
        "negative_token_ids": negative_token_ids,
        "forbidden_token_ids": forbidden_token_ids,
        "positive_bias": positive_bias,
        "negative_bias": negative_bias,
        "repetition_penalty": repetition_penalty,
        "profile": profile,
        "max_tokens": max_tokens,
    }
    if phenomenon_weights:
        result["phenomenon_weights"] = phenomenon_weights
    return result
