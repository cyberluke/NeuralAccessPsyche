"""Serialization utilities for the trusted custom logit processor."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def serialize_processor(processor_class: type) -> str:
    """Serialize the importable processor class using SGLang's dill contract."""
    import dill

    serialized = dill.dumps(processor_class)

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
    request: Optional[Any] = None,
    phrase_constraint_config: Optional[Dict[str, Any]] = None,
    entropy_config: Optional[Dict[str, Any]] = None,
    concept_config: Optional[Dict[str, Any]] = None,
    soft_injection_config: Optional[Dict[str, Any]] = None,
    logit_vector_config: Optional[Dict[str, Any]] = None,
    hard_injection_config: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    telemetry_enabled: bool = False,
    telemetry_max_steps: int = 16,
    telemetry_top_k: int = 5,
    forced_token_id: Optional[int] = None,
    forced_token_enabled: bool = False,
    applied_state_hash: Optional[str] = None,
    applied_state_schema: str = "nram.applied-state.v1",
    # NRAM v5 representation control configs
    activation_addition_config: Optional[Dict[str, Any]] = None,
    conceptor_config: Optional[Dict[str, Any]] = None,
    probes_config: Optional[Dict[str, Any]] = None,
    latent_loop_config: Optional[Dict[str, Any]] = None,
    semantic_loop_config: Optional[Dict[str, Any]] = None,
    branch_tournament_config: Optional[Dict[str, Any]] = None,
    dexperts_config: Optional[Dict[str, Any]] = None,
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

    request is retained for API compatibility but is deliberately not put on
    the wire. SGLang 0.5.16 injects the live server-side request as ``__req__``
    in ``Req.__init__``. Sending an application request here would fail JSON
    serialization and violate the scheduler trust boundary.

    NRAM v5 additions:
    phrase_constraint_config: Phrase-level masking and source n-gram blocking
    entropy_config: PID-based entropy control with phase targets
    concept_config: Dynamic concept capsule injection
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

    # NRAM v5 multi-layer inference control
    if phrase_constraint_config:
        result["phrase_constraint_config"] = phrase_constraint_config
    if entropy_config:
        result["entropy_config"] = entropy_config
    if concept_config:
        result["concept_config"] = concept_config
    if soft_injection_config:
        result["soft_injection_config"] = soft_injection_config
    if logit_vector_config:
        result["logit_vector_config"] = logit_vector_config
    if hard_injection_config:
        result["hard_injection_config"] = hard_injection_config
    
    # NRAM v5 representation control configs
    if activation_addition_config:
        result["activation_addition_config"] = activation_addition_config
    if conceptor_config:
        result["conceptor_config"] = conceptor_config
    if probes_config:
        result["probes_config"] = probes_config
    if latent_loop_config:
        result["latent_loop_config"] = latent_loop_config
    if semantic_loop_config:
        result["semantic_loop_config"] = semantic_loop_config
    if branch_tournament_config:
        result["branch_tournament_config"] = branch_tournament_config
    if dexperts_config:
        result["dexperts_config"] = dexperts_config

    if forced_token_id is not None:
        result["forced_token_id"] = int(forced_token_id)
    result["forced_token_enabled"] = bool(forced_token_enabled)
    result["telemetry_enabled"] = bool(telemetry_enabled)
    if telemetry_enabled:
        result["telemetry_max_steps"] = max(0, min(128, int(telemetry_max_steps)))
        result["telemetry_top_k"] = max(1, min(20, int(telemetry_top_k)))

    # The engine normally supplies a hash of the complete resolved applied
    # state (prompt, sampling, grammar, artifacts, route, and processor
    # parameters). Keep the local fallback for direct unit callers and older
    # integrations, but mark both forms with an explicit schema version.
    result["applied_state_schema"] = applied_state_schema
    if applied_state_hash is None:
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
        applied_state_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    result["config_hash"] = applied_state_hash
    if request_id:
        result["request_id"] = str(request_id)[:128]

    return result
