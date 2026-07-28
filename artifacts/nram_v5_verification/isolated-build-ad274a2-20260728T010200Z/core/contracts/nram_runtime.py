"""Strict, versioned public validation for the bounded NRAM runtime.

This module validates only mechanisms that are connected to the live SGLang
logit-processor path.  Advanced research mechanisms remain explicit rejected
keys; accepting a key here is a runtime contract, not a documentation claim.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


NRAM_SCHEMA_VERSION = "nram.request.v1"

SUPPORTED_PROFILES = {
    "normal",
    "microdose",
    "threshold",
    "psychedelic",
    "peak",
    "dissociative",
    "visionary-peak",
    "visionary-psychedelic-keynote",
}

UNSUPPORTED_NRAM_FEATURES = {
    "dexperts",
    "activation_addition",
    "actadd",
    "conceptor_steering",
    "hidden_state_probes",
    "latent_closed_loop",
    "semantic_novelty_controller",
    "evidence_guard",
    "branch_tournament",
    "reft",
    "soft_prompts",
    "attention_head_gating",
    "kv_cache_firewall",
    "gpu_native_semantic_control",
}

PHASES = ("extraction", "questioning", "divergence", "synthesis", "formulation")
PHENOMENA = (
    "overlap",
    "forgetting",
    "looping",
    "associative_jump",
    "synesthesia",
    "dissolution",
    "echo",
    "tangent",
    "insight",
)


class StrictRuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class EntropyPhaseTargets(StrictRuntimeModel):
    extraction: StrictFloat = Field(2.5, ge=0.0, le=20.0)
    questioning: StrictFloat = Field(4.0, ge=0.0, le=20.0)
    divergence: StrictFloat = Field(6.0, ge=0.0, le=20.0)
    synthesis: StrictFloat = Field(4.5, ge=0.0, le=20.0)
    formulation: StrictFloat = Field(3.0, ge=0.0, le=20.0)


class ConceptControl(StrictRuntimeModel):
    concept_id: StrictStr = Field(min_length=1, max_length=64)
    en_tokens: List[StrictStr] = Field(default_factory=list, max_length=64)
    cs_tokens: List[StrictStr] = Field(default_factory=list, max_length=64)
    synonyms: List[StrictStr] = Field(default_factory=list, max_length=64)
    activation_phase: Optional[
        Literal["extraction", "questioning", "divergence", "synthesis", "formulation"]
    ] = None
    max_uses: StrictInt = Field(0, ge=0, le=1024)

    @model_validator(mode="after")
    def require_forms(self) -> "ConceptControl":
        if not (self.en_tokens or self.cs_tokens or self.synonyms):
            raise ValueError("concept_requires_at_least_one_token_form")
        return self


class SoftTokenInjection(StrictRuntimeModel):
    token_ids: List[StrictInt] = Field(min_length=1, max_length=256)
    bias: StrictFloat = Field(ge=-5.0, le=5.0)
    start_step: StrictInt = Field(0, ge=0, le=1_000_000)
    end_step: StrictInt = Field(2**31 - 1, ge=1, le=2**31 - 1)

    @model_validator(mode="after")
    def validate_window(self) -> "SoftTokenInjection":
        if self.end_step <= self.start_step:
            raise ValueError("soft_injection_end_must_exceed_start")
        return self


class VocabularyVectorEntry(StrictRuntimeModel):
    token_id: StrictInt = Field(ge=0)
    weight: StrictFloat = Field(ge=-10.0, le=10.0)


class VocabularyLogitVector(StrictRuntimeModel):
    vector_id: StrictStr = Field(min_length=1, max_length=64)
    coefficient: StrictFloat = Field(ge=-4.0, le=4.0)
    normalize: StrictBool = False
    entries: List[VocabularyVectorEntry] = Field(min_length=1, max_length=256)


class PhenomenonWeights(StrictRuntimeModel):
    overlap: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    forgetting: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    looping: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    associative_jump: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    synesthesia: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    dissolution: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    echo: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    tangent: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    insight: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)


class NRAMOptions(StrictRuntimeModel):
    schema_version: Literal["nram.request.v1"] = "nram.request.v1"
    enabled: StrictBool = True
    profile: StrictStr = "normal"
    intensity: StrictFloat = Field(1.0, ge=0.0, le=1.0)
    request_id: Optional[StrictStr] = Field(None, min_length=1, max_length=128)
    session_id: Optional[StrictStr] = Field(None, min_length=1, max_length=128)

    include_telemetry: StrictBool = False
    telemetry_max_steps: StrictInt = Field(16, ge=0, le=128)
    telemetry_top_k: StrictInt = Field(5, ge=1, le=20)
    prompt_steering_enabled: StrictBool = True
    profile_logit_steering_enabled: StrictBool = True

    forced_token_enabled: StrictBool = False
    forced_token_id: Optional[StrictInt] = Field(None, ge=0)
    forbidden_token_ids: List[StrictInt] = Field(default_factory=list, max_length=65536)
    forbidden_phrases: List[StrictStr] = Field(default_factory=list, max_length=256)
    source_text: Optional[StrictStr] = Field(None, max_length=2_000_000)
    source_ngram_size: StrictInt = Field(8, ge=2, le=128)

    hard_token_schedule: List[StrictInt] = Field(default_factory=list, max_length=4096)
    hard_token_schedule_start: StrictInt = Field(0, ge=0, le=1_000_000)
    soft_token_injections: List[SoftTokenInjection] = Field(default_factory=list, max_length=32)
    vocabulary_logit_vectors: List[VocabularyLogitVector] = Field(default_factory=list, max_length=16)
    vocabulary_logit_vector_clip: StrictFloat = Field(5.0, ge=0.0, le=10.0)

    entropy_control: StrictBool = False
    entropy_phase_targets: Optional[EntropyPhaseTargets] = None
    entropy_kp: StrictFloat = Field(0.5, ge=0.0, le=2.0)
    entropy_ki: StrictFloat = Field(0.02, ge=0.0, le=0.5)
    entropy_kd: StrictFloat = Field(0.05, ge=0.0, le=1.0)
    entropy_integral_limit: StrictFloat = Field(20.0, ge=0.0, le=100.0)
    entropy_scale_min: StrictFloat = Field(0.6, ge=0.05, le=1.0)
    entropy_scale_max: StrictFloat = Field(1.8, ge=1.0, le=5.0)

    concepts: List[ConceptControl] = Field(default_factory=list, max_length=64)
    concept_strength: StrictFloat = Field(0.5, ge=-5.0, le=5.0)
    phenomenon_weights: Optional[PhenomenonWeights] = None

    # Explicit profile-state overrides supported by resolve_request_state().
    visionary_intensity: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    contrarian_force: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    product_obsession: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    human_focus: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    rhetorical_compression: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    associative_distance: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    theatricality: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    emotional_voltage: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    coherence_floor: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    novelty_target: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[StrictFloat] = Field(None, ge=0.0, le=2.0)
    corporate_jargon_penalty: Optional[StrictFloat] = Field(None, ge=0.0, le=2.5)

    # Known unsupported keys are represented so they produce a stable
    # unsupported_feature error rather than being confused with unknown keys.
    dexperts: Optional[StrictBool] = None
    activation_addition: Optional[StrictBool] = None
    actadd: Optional[StrictBool] = None
    conceptor_steering: Optional[StrictBool] = None
    hidden_state_probes: Optional[StrictBool] = None
    latent_closed_loop: Optional[StrictBool] = None
    semantic_novelty_controller: Optional[StrictBool] = None
    evidence_guard: Optional[StrictBool] = None
    branch_tournament: Optional[StrictBool] = None
    reft: Optional[StrictBool] = None
    soft_prompts: Optional[StrictBool] = None
    attention_head_gating: Optional[StrictBool] = None
    kv_cache_firewall: Optional[StrictBool] = None
    gpu_native_semantic_control: Optional[StrictBool] = None

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        if value not in SUPPORTED_PROFILES:
            raise ValueError("unknown_nram_profile")
        return value

    @field_validator("forbidden_phrases")
    @classmethod
    def validate_phrases(cls, values: List[str]) -> List[str]:
        if any(not value or len(value) > 4096 for value in values):
            raise ValueError("forbidden_phrase_must_be_nonempty_and_bounded")
        return values

    @model_validator(mode="after")
    def validate_conflicts_and_unsupported(self) -> "NRAMOptions":
        enabled_unsupported = [
            name for name in sorted(UNSUPPORTED_NRAM_FEATURES)
            if getattr(self, name, None) is True
        ]
        if enabled_unsupported:
            raise ValueError("unsupported_nram_features:" + ",".join(enabled_unsupported))
        if self.forced_token_enabled and self.forced_token_id is None:
            raise ValueError("forced_token_id_required_when_enabled")
        if (
            self.forced_token_enabled
            and self.forced_token_id is not None
            and self.forced_token_id in self.forbidden_token_ids
        ):
            raise ValueError("forced_token_conflicts_with_forbidden_token")
        if self.entropy_scale_min > self.entropy_scale_max:
            raise ValueError("entropy_scale_min_exceeds_max")
        return self


def normalize_nram_options(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate and return JSON-compatible options while preserving dict callers."""
    if value is None:
        return None
    validated = NRAMOptions.model_validate(value)
    return validated.model_dump(mode="json", exclude_none=True)


def validate_tokenizer_ids(options: Optional[Dict[str, Any]], tokenizer: Any) -> None:
    """Reject IDs not defined by the active tokenizer before upstream generation.

    `len(tokenizer)` can include added/padding slots that are not decodable model
    tokens.  A public token ID is accepted only when it is in the model vocabulary
    and conversion does not resolve to the tokenizer's undefined token.
    """
    if not options:
        return
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    unk_id = getattr(tokenizer, "unk_token_id", None)

    def check(token_id: int, path: str) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"{path}:token_id_must_be_strict_integer")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{path}:token_id_not_defined_by_active_tokenizer")
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
        except Exception as exc:
            raise ValueError(f"{path}:token_id_not_defined_by_active_tokenizer") from exc
        if token is None or (unk_id is not None and token_id != unk_id and token == tokenizer.unk_token):
            raise ValueError(f"{path}:token_id_not_defined_by_active_tokenizer")

    for index, token_id in enumerate(options.get("forbidden_token_ids", [])):
        check(token_id, f"forbidden_token_ids[{index}]")
    if options.get("forced_token_id") is not None:
        check(options["forced_token_id"], "forced_token_id")
    for index, token_id in enumerate(options.get("hard_token_schedule", [])):
        check(token_id, f"hard_token_schedule[{index}]")
    for item_index, item in enumerate(options.get("soft_token_injections", [])):
        for token_index, token_id in enumerate(item["token_ids"]):
            check(token_id, f"soft_token_injections[{item_index}].token_ids[{token_index}]")
    for vector_index, vector in enumerate(options.get("vocabulary_logit_vectors", [])):
        for entry_index, entry in enumerate(vector["entries"]):
            check(entry["token_id"], f"vocabulary_logit_vectors[{vector_index}].entries[{entry_index}]")


def stable_validation_error(exc: Exception) -> Dict[str, Any]:
    """Return a stable machine-readable validation detail object."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        detail = []
        for item in errors(include_url=False):
            detail.append(
                {
                    "path": ".".join(str(part) for part in item.get("loc", ())),
                    "code": item.get("type", "invalid_value"),
                    "message": item.get("msg", "invalid value"),
                }
            )
        return {"code": "invalid_request_schema", "schema": NRAM_SCHEMA_VERSION, "errors": detail}
    return {
        "code": "invalid_nram_control",
        "schema": NRAM_SCHEMA_VERSION,
        "errors": [{"path": "nram", "code": "invalid_value", "message": str(exc)}],
    }
