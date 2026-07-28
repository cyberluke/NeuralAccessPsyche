"""SGLang engine — OpenAI-compatible client with NRAM logit processor injection."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import httpx

from core.contracts.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    EngineCapabilities,
    OpenAIError,
    Usage,
)
from core.engines.base import SGLANG_CAPABILITIES
from core.contracts.nram_runtime import (
    UNSUPPORTED_NRAM_FEATURES,
    validate_tokenizer_ids,
)
from core.persona.compiler import compile_policy
from core.persona.planner import build_rhetorical_plan, plan_to_prompt_fragment
from core.persona.profiles import DEFAULT_PROFILE, PROFILES
from nram_sglang.processor import NRAMLogitProcessor
from core.steering.serialization import build_custom_params, serialize_processor
from core.steering.tokenizer_bias import TokenBiasCompiler

if TYPE_CHECKING:
    from core.contracts.nram import NRAMState

logger = logging.getLogger(__name__)

# Model alias → upstream model path mapping
MODEL_ALIAS_MAP: Dict[str, str] = {
    "nram-qwen3-14b-awq": "nram-qwen3-14b-awq",
    "qwen3-14b-awq-baseline": "nram-qwen3-14b-awq",
}

# NRAM-enabled aliases
NRAM_ENABLED_ALIASES = {"nram-qwen3-14b-awq"}

# Explicit allowlist of fields forwarded to SGLang
_FORWARDABLE_FIELDS = {
    "model", "messages", "temperature", "top_p", "max_tokens",
    "stream", "stop", "seed", "frequency_penalty", "presence_penalty",
    "response_format",
}

# Fields that clients must NEVER supply
_FORBIDDEN_CLIENT_FIELDS = {
    "custom_logit_processor", "custom_params", "serialized_processor",
    "processor_class", "python_code", "__req__",
}

# These names describe mechanisms that require unavailable models, datasets,
# hidden-state hooks, or branch runtimes. Silently ignoring them would create a
# false scientific claim, so public requests fail explicitly.
_UNSUPPORTED_NRAM_FEATURES = UNSUPPORTED_NRAM_FEATURES

# Display names for the phenomenon mixer (English primary, Czech secondary for UI).
_PHENOMENON_LABELS = {
    "overlap": "Overlap (Překryv)",
    "forgetting": "Forgetting (Zapomenutí)",
    "looping": "Looping (Zacyklení)",
    "associative_jump": "Associative Jump (Skok)",
    "synesthesia": "Synesthesia (Synestézie)",
    "dissolution": "Dissolution (Rozpuštění)",
    "echo": "Echo (Ozvěna)",
    "tangent": "Tangent (Tangenta)",
    "insight": "Insight (Vhled)",
}


def _apply_phenomenon_mix(
    instruction: Optional[str],
    phenomenon_weights: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Append an active-phenomena block to the developer instruction.

    The phenomenon mixer in the UI sends per-phenomenon weights. We render the
    active ones (weight > 0) into a concise, honest block so the model knows
    which simulated phenomena to emphasize. This is prompt-level steering and
    is labeled as a simulation, not a measured state.
    """
    if not instruction:
        return instruction
    if not phenomenon_weights:
        return instruction

    active = []
    for key, label in _PHENOMENON_LABELS.items():
        weight = phenomenon_weights.get(key)
        if weight is None:
            continue
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w > 0:
            active.append(f"- {label}: {w:.2f}")

    if not active:
        return instruction

    block = (
        "\n\nActive simulated phenomena (linguistic simulation only, "
        "intensities 0.0-1.0):\n" + "\n".join(active)
    )
    return instruction + block


class SGLangEngineError(Exception):
    """Raised when the SGLang upstream fails."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _apply_global_intensity(base: "NRAMState", intensity: float) -> "NRAMState":
    """Mix neutral profile with base profile according to intensity.

    intensity=0.0 -> pure neutral (minimal steering)
    intensity=1.0 -> full base profile
    """
    from core.contracts.nram import NRAMState
    neutral = PROFILES.get("normal", PROFILES[DEFAULT_PROFILE])

    def mix(a: float, b: float) -> float:
        return a + (b - a) * intensity

    return NRAMState(
        visionary_intensity=mix(neutral.visionary_intensity, base.visionary_intensity),
        contrarian_force=mix(neutral.contrarian_force, base.contrarian_force),
        product_obsession=mix(neutral.product_obsession, base.product_obsession),
        human_focus=mix(neutral.human_focus, base.human_focus),
        rhetorical_compression=mix(neutral.rhetorical_compression, base.rhetorical_compression),
        associative_distance=mix(neutral.associative_distance, base.associative_distance),
        theatricality=mix(neutral.theatricality, base.theatricality),
        emotional_voltage=mix(neutral.emotional_voltage, base.emotional_voltage),
        coherence_floor=mix(neutral.coherence_floor, base.coherence_floor),
        novelty_target=mix(neutral.novelty_target, base.novelty_target),
        repetition_penalty=mix(neutral.repetition_penalty, base.repetition_penalty),
        corporate_jargon_penalty=mix(neutral.corporate_jargon_penalty, base.corporate_jargon_penalty),
    )


def resolve_request_state(nram_opts: Optional[Dict[str, Any]], profile_name: str = DEFAULT_PROFILE) -> "NRAMState":
    """Resolve the final NRAMState by applying intensity and per-request overrides.

    This is the SINGLE source of truth for computing the state used by:
    - developer instruction
    - token bias
    - telemetry
    - audit
    - UI display
    """
    from core.contracts.nram import NRAMState

    profile = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])

    # Step 1: Apply intensity mixing
    intensity = nram_opts.get("intensity") if nram_opts else None
    if intensity is not None:
        try:
            intensity = float(intensity)
            state = _apply_global_intensity(profile, intensity)
        except (TypeError, ValueError):
            state = profile
    else:
        state = profile

    # Step 2: Apply per-request overrides (explicit dimension values)
    override_fields = {
        "visionary_intensity", "contrarian_force", "product_obsession",
        "human_focus", "rhetorical_compression", "associative_distance",
        "theatricality", "emotional_voltage", "coherence_floor",
        "novelty_target", "repetition_penalty", "corporate_jargon_penalty",
    }
    state_dict = state.model_dump()
    if nram_opts:
        for field in override_fields:
            if field in nram_opts and nram_opts[field] is not None:
                state_dict[field] = nram_opts[field]

    return NRAMState(**state_dict)


# ---------------------------------------------------------------------------
# Reasoning content stripping — never forward hidden chain-of-thought
# ---------------------------------------------------------------------------

_REASONING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Unclosed  tag: strip everything from an unmatched  onward.
_UNCLOSED_THINK_RE = re.compile(r"<think>(?:(?!</think>).)*$", re.DOTALL)

# Heuristic patterns that indicate raw chain-of-thought (not a final answer).
# These catch reasoning that the model emits WITHOUT  tags.
_REASONING_HEURISTIC_RE = re.compile(
    r"^(?:Okay,?|Hmm,?|Let me|Let's|I need to|I'm trying|I should|So,?|"
    r"First,?|Alright,?|Well,?|The user|I think|I wonder|I'll|Now,?|"
    # Czech reasoning preambles (strong NRAM steering → Czech CoT)
    r"Potřebuji|Musím|Uživatel|Podívám|Zkusím|Pojďme|Nejprve|"
    r"Zamyslím|Přemýšlím|Dobře,?|Takže,?|Hmm|Aha,?|No,?|"
    r"Tak,?|Podívejme|Pojď|Uvažuj|Analyzuj|Zvaž)",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Remove  tags AND heuristic chain-of-thought from final text.

    DeepSeek-R1 models emit reasoning as plain text (no tags), so we also
    detect and strip common reasoning preambles. Strong NRAM steering can
    cause the model to emit unclosed  tags or Czech-language CoT.
    """
    # First strip explicit  blocks
    text = _REASONING_RE.sub("", text).strip()

    # Strip unclosed trailing  (strong NRAM can prevent the close tag)
    text = _UNCLOSED_THINK_RE.sub("", text).strip()

    # Iteratively strip leading reasoning preambles (max 3 sentences to preserve content)
    for _ in range(3):
        if not _REASONING_HEURISTIC_RE.match(text):
            break
        parts = re.split(r"(?<=[.!?…])\s+", text, maxsplit=1)
        if len(parts) > 1 and len(parts[1].strip()) > 20:
            text = parts[1].strip()
        else:
            break

    return text


class _StreamState:
    """Per-stream state machine that strips <think></think> blocks from SSE chunks.

    Rewrites the model field to the public alias and drops unparseable lines.
    Handles partial tags split across chunk boundaries by holding back any
    trailing prefix of the close tag while an open tag is pending.
    """

    def __init__(self, public_model: str, correlation: Optional[Dict[str, Any]] = None) -> None:
        self._public_model = public_model
        self._correlation = correlation
        self._buffer = ""

    def feed(self, data_line: str) -> List[str]:
        """Consume one raw ``data: {...}`` line; yield sanitized ``data: ...`` lines."""
        payload = data_line[len("data: "):].strip()
        if not payload or payload == "[DONE]":
            return []

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # Unparseable — drop silently rather than leak raw reasoning.
            return []

        chunk["model"] = self._public_model
        if self._correlation is not None:
            chunk["nram_correlation"] = self._correlation
        out: List[str] = []

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            piece = delta.get("content")
            if piece is None:
                continue

            self._buffer += piece
            visible = self._extract_visible()
            if visible:
                delta["content"] = visible
                out.append("data: " + json.dumps(chunk))

        # Preserve finish_reason chunks even without content.
        if not out and any(c.get("finish_reason") for c in chunk.get("choices", [])):
            for choice in chunk.get("choices", []):
                choice.get("delta", {}).pop("content", None)
            out.append("data: " + json.dumps(chunk))

        return out

    def _extract_visible(self) -> str:
        """Return content outside <think></think> blocks, buffering partial tags."""
        # Drop completed blocks.
        self._buffer = _REASONING_RE.sub("", self._buffer)

        # If an open tag is pending, hold everything from it onward.
        open_idx = self._buffer.rfind("<think>")
        if open_idx != -1:
            close_idx = self._buffer.find("</think>", open_idx)
            if close_idx == -1:
                visible = self._buffer[:open_idx]
                self._buffer = self._buffer[open_idx:]
                return visible

        # Hold back a trailing partial close-tag to avoid emitting it piecewise.
        hold = 0
        close_tag = "</think>"
        max_hold = min(len(close_tag) - 1, len(self._buffer))
        for n in range(max_hold, 0, -1):
            if close_tag.startswith(self._buffer[-n:]):
                hold = n
                break
        if hold:
            visible = self._buffer[:-hold]
            self._buffer = self._buffer[-hold:]
            return visible

        visible = self._buffer
        self._buffer = ""
        return visible


class SGLangEngine:
    """Production engine that talks to SGLang's OpenAI-compatible API."""

    capabilities = SGLANG_CAPABILITIES

    def __init__(
        self,
        base_url: str = "http://sglang:30000/v1",
        model: str = "nram-qwen3-14b-awq",
        tokenizer: Any = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._abort_url = self._base_url.removesuffix("/v1") + "/abort_request"
        self._model = model
        self._tokenizer = tokenizer
        self._bias_compiler: Optional[TokenBiasCompiler] = None
        if tokenizer is not None:
            self._bias_compiler = TokenBiasCompiler(tokenizer)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=30.0) if timeout else None,
        )
        self._abort_lock = asyncio.Lock()
        self._aborted_request_ids: Dict[str, None] = {}
        self._serialized_processor = serialize_processor(NRAMLogitProcessor)
        logger.info(f"SGLangEngine initialized: base_url={base_url}, model={model}")

    async def close(self) -> None:
        await self._client.aclose()

    async def abort(self, scheduler_request_id: str, reason: str = "client_cancelled") -> None:
        """Abort one real SGLang request through the official 0.5.16 endpoint."""
        if not scheduler_request_id:
            return
        async with self._abort_lock:
            if scheduler_request_id in self._aborted_request_ids:
                return
            self._aborted_request_ids[scheduler_request_id] = None
            # Request IDs are unique. Keep bounded history so route cleanup and
            # engine cancellation cannot issue duplicate official abort calls.
            if len(self._aborted_request_ids) > 4096:
                self._aborted_request_ids.pop(next(iter(self._aborted_request_ids)))
        try:
            response = await self._client.post(
                self._abort_url,
                json={"rid": scheduler_request_id, "abort_message": reason},
                timeout=2.0,
            )
            response.raise_for_status()
        except Exception:
            logger.error(
                "Failed to abort upstream SGLang request rid=%s reason=%s",
                scheduler_request_id,
                reason,
                exc_info=True,
            )

    @staticmethod
    def _normalize_hash_value(value: Any) -> Any:
        """Normalize mapping order and text newlines for cross-platform hashes."""
        if isinstance(value, dict):
            return {
                str(key): SGLangEngine._normalize_hash_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [SGLangEngine._normalize_hash_value(item) for item in value]
        if isinstance(value, str):
            return value.replace("\r\n", "\n").replace("\r", "\n")
        return value

    def _applied_state_hash(
        self,
        request: ChatCompletionRequest,
        payload: Dict[str, Any],
    ) -> str:
        """Hash the complete behaviorally relevant, resolved request state."""
        processor_params = dict(payload.get("custom_params") or {})
        processor_params.pop("request_id", None)
        processor_params.pop("config_hash", None)
        processor_params.pop("applied_state_schema", None)
        applied = {
            "schema": "nram.applied-state.v1",
            "route": request.route_kind,
            "public_model": request.public_model or request.model,
            "actual_base_model": payload.get("model"),
            "model_artifact": {
                "configured_model": self._model,
                "snapshot": os.environ.get("NRAM_MODEL_SNAPSHOT", "unknown"),
            },
            "tokenizer_artifact": {
                "identity": os.environ.get(
                    "NRAM_TOKENIZER_ID",
                    str(getattr(self._tokenizer, "name_or_path", "unknown")),
                ),
                "hash": os.environ.get("NRAM_TOKENIZER_HASH", "unknown"),
                "vocab_size": int(
                    getattr(self._tokenizer, "vocab_size", 0) or len(self._tokenizer)
                ),
            },
            "messages": payload.get("messages", []),
            "sampling": {
                key: payload.get(key)
                for key in (
                    "max_tokens", "temperature", "top_p", "seed", "stop",
                    "frequency_penalty", "presence_penalty", "stream",
                )
            },
            "grammar_or_response_format": payload.get("response_format"),
            "processor": processor_params,
            "defaults_version": "nram.sglang.defaults.v1",
        }
        normalized = self._normalize_hash_value(applied)
        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _finalize_payload(
        self,
        request: ChatCompletionRequest,
        payload: Dict[str, Any],
        *,
        stream: bool,
    ) -> Dict[str, Any]:
        """Apply effective defaults, trusted rid, observability, and final hash."""
        payload["stream"] = stream
        payload["rid"] = request.runtime_request_id or f"nram-scheduler-{uuid.uuid4().hex}"
        if "presence_penalty" not in payload:
            payload["presence_penalty"] = 0.8
        if request.temperature is None:
            payload["temperature"] = 0.7
        if request.top_p is None:
            payload["top_p"] = 0.8

        params = payload.get("custom_params")
        if params is not None:
            # The supported non-stream OpenAI extension returns raw sampled IDs
            # in meta_info.output_token_logprobs. Streaming rejects meta_info.
            if params.get("telemetry_enabled") and not stream:
                payload["logprobs"] = True
                payload["top_logprobs"] = 0
                payload["return_meta_info"] = True
            params["applied_state_schema"] = "nram.applied-state.v1"
            params["config_hash"] = self._applied_state_hash(request, payload)
        return payload

    @staticmethod
    def _correlation(
        request: ChatCompletionRequest,
        payload: Dict[str, Any],
        *,
        sampled: Optional[List[List[Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        params = payload.get("custom_params")
        if params is None:
            return {
                "scheduler_request_id": payload["rid"],
                "public_model": request.public_model or request.model,
                "actual_base_model": payload["model"],
                "processor_intervened": False,
                "sampled_token_ids": [],
                "sampled_tokens": [],
                "sample_join": "not_requested",
            }
        sampled = sampled or []
        return {
            "request_id": params["request_id"],
            "scheduler_request_id": payload["rid"],
            "config_hash": params["config_hash"],
            "applied_state_schema": params["applied_state_schema"],
            "public_model": request.public_model or request.model,
            "actual_base_model": payload["model"],
            "processor_intervened": True,
            "sampled_token_ids": [int(item[1]) for item in sampled if len(item) >= 2],
            "sampled_tokens": [str(item[2]) for item in sampled if len(item) >= 3],
            "sample_join": "sglang_meta_info" if sampled else (
                "unavailable_for_streaming" if payload.get("stream") else "not_requested"
            ),
        }

    def _is_nram_enabled(self, request: ChatCompletionRequest) -> bool:
        """Check if NRAM steering is enabled for this request."""
        if request.model in NRAM_ENABLED_ALIASES:
            nram_opts = request.nram or {}
            return nram_opts.get("enabled", True)
        return False

    def _resolve_upstream_model(self, request: ChatCompletionRequest) -> str:
        """Map public alias to upstream model name."""
        try:
            return MODEL_ALIAS_MAP[request.model]
        except KeyError as exc:
            raise SGLangEngineError(
                f"Model {request.model!r} is not backed by the loaded checkpoint.",
                status_code=400,
            ) from exc

    def validate(self, request: ChatCompletionRequest) -> None:
        """Preflight one request without planning, network I/O, or generation."""
        self._resolve_upstream_model(request)
        options = request.nram or {}
        requested_unsupported = sorted(
            key for key in _UNSUPPORTED_NRAM_FEATURES if options.get(key)
        )
        if requested_unsupported:
            raise SGLangEngineError(
                "Unsupported NRAM runtime feature(s): " + ", ".join(requested_unsupported),
                status_code=400,
            )
        if self._tokenizer is not None:
            try:
                validate_tokenizer_ids(options, self._tokenizer)
            except ValueError as exc:
                raise SGLangEngineError(str(exc), status_code=400) from exc

    # ------------------------------------------------------------------
    # NRAM v5 multi-layer config builders
    # ------------------------------------------------------------------
    def _build_phrase_constraint_config(
        self,
        request: ChatCompletionRequest,
        nram_opts: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build phrase constraint configuration for the logit processor.

        Extracts forbidden phrases and source n-gram blocking config from
        request nram options. Returns None if no phrase constraints configured.
        """
        forbidden_phrases = nram_opts.get("forbidden_phrases", [])
        source_text = nram_opts.get("source_text")

        if not forbidden_phrases and not source_text:
            return None

        config: Dict[str, Any] = {}

        # Encode tokenizer-aware variants. A configured phrase applies across
        # leading-space/no-leading-space and Unicode normalization variants so
        # callers do not need tokenizer-specific spellings.
        if forbidden_phrases and self._tokenizer:
            forbidden_phrase_ids = set()
            for phrase in forbidden_phrases:
                normalized = unicodedata.normalize("NFC", phrase)
                stripped = normalized.strip()
                variants = {normalized, stripped}
                if stripped:
                    variants.add(" " + stripped)
                for variant in variants:
                    try:
                        token_ids = self._tokenizer.encode(variant, add_special_tokens=False)
                        if token_ids:
                            forbidden_phrase_ids.add(tuple(token_ids))
                    except Exception:
                        logger.debug("Tokenizer rejected forbidden phrase variant", exc_info=True)
            if forbidden_phrase_ids:
                config["forbidden_phrase_ids"] = [list(ids) for ids in sorted(forbidden_phrase_ids)]

        # Source n-gram blocking
        if source_text and self._tokenizer:
            source_ngram_size = nram_opts.get("source_ngram_size", 8)
            try:
                normalized = unicodedata.normalize("NFC", source_text)
                stripped = normalized.strip()
                source_variants = {normalized, stripped}
                if stripped:
                    source_variants.add(" " + stripped)
                source_ngrams_set = set()
                for variant in source_variants:
                    source_tokens = self._tokenizer.encode(variant, add_special_tokens=False)
                    for i in range(len(source_tokens) - source_ngram_size + 1):
                        source_ngrams_set.add(tuple(source_tokens[i:i + source_ngram_size]))
                source_ngrams = [list(ids) for ids in sorted(source_ngrams_set)]
                if source_ngrams:
                    config["source_ngram_ids"] = source_ngrams
                    config["source_ngram_size"] = source_ngram_size
            except Exception:
                pass

        return config if config else None

    def _build_entropy_config(
        self,
        request: ChatCompletionRequest,
        nram_opts: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build entropy control configuration for the logit processor.

        Returns None if entropy control is not enabled.
        """
        entropy_enabled = nram_opts.get("entropy_control", False)
        if not entropy_enabled:
            return None

        return {
            "phase_targets": nram_opts.get("entropy_phase_targets", {
                "extraction": 2.5,
                "questioning": 4.0,
                "divergence": 6.0,
                "synthesis": 4.5,
                "formulation": 3.0,
            }),
            "kp": nram_opts.get("entropy_kp", 0.5),
            "ki": nram_opts.get("entropy_ki", 0.02),
            "kd": nram_opts.get("entropy_kd", 0.05),
            "integral_limit": nram_opts.get("entropy_integral_limit", 20.0),
            "scale_min": nram_opts.get("entropy_scale_min", 0.6),
            "scale_max": nram_opts.get("entropy_scale_max", 1.8),
        }

    def _build_concept_config(
        self,
        request: ChatCompletionRequest,
        nram_opts: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build concept injection configuration for the logit processor.

        Returns None if no concepts configured.
        """
        concepts = nram_opts.get("concepts", [])
        if not concepts or not self._tokenizer:
            return None

        # Encode concept tokens to IDs
        encoded_concepts = []
        for concept in concepts:
            token_forms = concept.get("en_tokens", []) + concept.get("cs_tokens", []) + concept.get("synonyms", [])
            token_ids = []
            for form in token_forms:
                try:
                    ids = self._tokenizer.encode(form, add_special_tokens=False)
                    token_ids.extend(ids)
                except Exception:
                    pass
            if token_ids:
                encoded_concepts.append({
                    "concept_id": concept.get("concept_id", ""),
                    "token_ids": list(set(token_ids)),
                    "activation_phase": concept.get("activation_phase"),
                    "max_uses": concept.get("max_uses", 0),
                })

        if not encoded_concepts:
            return None

        return {
            "concepts": encoded_concepts,
            "base_strength": nram_opts.get("concept_strength", 0.5),
        }

    def _build_upstream_payload(
        self,
        request: ChatCompletionRequest,
        nram_enabled: bool,
        developer_instruction: Optional[str] = None,
        plan_fragment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the upstream payload with explicit allowlist filtering."""
        # Reject forbidden client fields
        nram_opts = request.nram or {}
        if self._tokenizer is not None:
            try:
                validate_tokenizer_ids(nram_opts, self._tokenizer)
            except ValueError as exc:
                raise SGLangEngineError(str(exc), status_code=400) from exc
        if _FORBIDDEN_CLIENT_FIELDS & set(nram_opts.keys()):
            raise SGLangEngineError(
                "Forbidden field in nram options: processor injection is not allowed.",
                status_code=400,
            )
        requested_unsupported = sorted(
            key for key in _UNSUPPORTED_NRAM_FEATURES if nram_opts.get(key)
        )
        if requested_unsupported:
            raise SGLangEngineError(
                "Unsupported NRAM runtime feature(s): " + ", ".join(requested_unsupported),
                status_code=400,
            )

        payload: Dict[str, Any] = {}

        # Allowlisted fields only
        for field in _FORWARDABLE_FIELDS:
            value = getattr(request, field, None)
            if value is not None:
                payload[field] = value

        # Feature 4: normalize response_format into SGLang grammar constraints.
        if payload.get("response_format") is not None:
            from core.steering.grammar import compile_response_format
            compiled_fmt = compile_response_format(payload["response_format"])
            if compiled_fmt is not None:
                payload["response_format"] = compiled_fmt
            else:
                payload.pop("response_format", None)

        # Map model alias
        payload["model"] = self._resolve_upstream_model(request)

        # Inject developer instruction and plan into messages
        messages = list(request.messages)
        if nram_enabled and (developer_instruction or plan_fragment):
            system_parts = []
            if developer_instruction:
                system_parts.append(developer_instruction)
            if plan_fragment:
                system_parts.append(plan_fragment)

            system_content = "\n\n".join(system_parts)

            # Prepend as system message — do NOT modify user content
            messages = [{"role": "system", "content": system_content}] + messages

        payload["messages"] = messages

        # Add trusted NRAM processor for NRAM-enabled requests
        if nram_enabled and self._bias_compiler is not None:
            profile_name = (request.nram or {}).get("profile", DEFAULT_PROFILE)
            # UNIFIED: use resolve_request_state (same as in complete/stream)
            state = resolve_request_state(request.nram, profile_name)

            policy = compile_policy(state, max_tokens=request.max_tokens or 512, profile_name=profile_name)
            compiled = self._bias_compiler.compile(policy)
            direct_forbidden_ids = nram_opts.get("forbidden_token_ids", [])
            compiled.forbidden_token_ids = sorted(
                set(compiled.forbidden_token_ids) | set(direct_forbidden_ids)
            )
            if not nram_opts.get("profile_logit_steering_enabled", True):
                compiled.positive_token_ids = []
                compiled.negative_token_ids = []
                compiled.forbidden_token_ids = []
                compiled.positive_bias = 0.0
                compiled.negative_bias = 0.0
                compiled.repetition_penalty = 0.0

            # Build NRAM v5 multi-layer configs
            phrase_constraint_config = self._build_phrase_constraint_config(request, nram_opts)
            entropy_config = self._build_entropy_config(request, nram_opts)
            concept_config = self._build_concept_config(request, nram_opts)
            soft_injection_config = None
            if isinstance(nram_opts.get("soft_token_injections"), list):
                soft_injection_config = {
                    "injections": nram_opts["soft_token_injections"][:32]
                }
            logit_vector_config = None
            if isinstance(nram_opts.get("vocabulary_logit_vectors"), list):
                logit_vector_config = {
                    "vectors": nram_opts["vocabulary_logit_vectors"][:16],
                    "clip": nram_opts.get("vocabulary_logit_vector_clip", 5.0),
                }
            hard_injection_config = None
            if isinstance(nram_opts.get("hard_token_schedule"), list):
                hard_injection_config = {
                    "token_ids": nram_opts["hard_token_schedule"][:256],
                    "start_step": nram_opts.get("hard_token_schedule_start", 0),
                }

            # CRITICAL: Enable custom logit processor for token-level steering
            # NRAM steering now works via both developer instruction AND logit processor
            payload["custom_logit_processor"] = self._serialized_processor
            request_id = nram_opts.get("request_id") or f"nram-{uuid.uuid4().hex}"
            forced_token_id = nram_opts.get("forced_token_id")
            if forced_token_id is not None:
                if isinstance(forced_token_id, bool) or not isinstance(forced_token_id, int):
                    raise SGLangEngineError("forced_token_id must be a strict integer", status_code=400)
                tokenizer_size = int(getattr(self._tokenizer, "vocab_size", 0) or len(self._tokenizer))
                if forced_token_id < 0 or forced_token_id >= tokenizer_size:
                    raise SGLangEngineError(
                        f"forced_token_id must be in [0, {tokenizer_size})",
                        status_code=400,
                    )

            payload["custom_params"] = build_custom_params(
                positive_token_ids=compiled.positive_token_ids,
                negative_token_ids=compiled.negative_token_ids,
                forbidden_token_ids=compiled.forbidden_token_ids,
                positive_bias=compiled.positive_bias,
                negative_bias=compiled.negative_bias,
                repetition_penalty=compiled.repetition_penalty,
                profile=profile_name,
                max_tokens=request.max_tokens or 0,
                phenomenon_weights=nram_opts.get("phenomenon_weights"),
                request=request,  # deliberately omitted from wire parameters
                phrase_constraint_config=phrase_constraint_config,
                entropy_config=entropy_config,
                concept_config=concept_config,
                soft_injection_config=soft_injection_config,
                logit_vector_config=logit_vector_config,
                hard_injection_config=hard_injection_config,
                request_id=request_id,
                telemetry_enabled=bool(nram_opts.get("include_telemetry", False)),
                telemetry_max_steps=nram_opts.get("telemetry_max_steps", 16),
                telemetry_top_k=nram_opts.get("telemetry_top_k", 5),
                forced_token_id=forced_token_id,
                forced_token_enabled=bool(nram_opts.get("forced_token_enabled", False)),
            )

        return payload

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Non-streaming completion through SGLang."""
        nram_enabled = self._is_nram_enabled(request)

        # Build rhetorical plan for NRAM requests
        developer_instruction = None
        plan_fragment = None
        if nram_enabled:
            profile_name = (request.nram or {}).get("profile", DEFAULT_PROFILE)
            # UNIFIED: use resolve_request_state for both instruction and logit policy
            state = resolve_request_state(request.nram, profile_name)
            policy = compile_policy(state, max_tokens=request.max_tokens or 512, profile_name=profile_name)
            developer_instruction = policy.developer_instruction
            developer_instruction = _apply_phenomenon_mix(
                developer_instruction, (request.nram or {}).get("phenomenon_weights")
            )

            user_content = ""
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break

            plan = await build_rhetorical_plan(self, user_content)
            if plan is not None:
                plan_fragment = plan_to_prompt_fragment(plan)
            if not (request.nram or {}).get("prompt_steering_enabled", True):
                developer_instruction = None
                plan_fragment = None

        payload = self._build_upstream_payload(
            request, nram_enabled, developer_instruction, plan_fragment
        )
        payload = self._finalize_payload(request, payload, stream=False)

        t_start = time.perf_counter()
        try:
            deadline = request.deadline_seconds
            if deadline is None:
                deadline = float(os.environ.get("NRAM_REQUEST_TIMEOUT_SECONDS", "300"))
            async with asyncio.timeout(deadline):
                resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

            usage = Usage(**data.get("usage", {}))
            latency_ms = (time.perf_counter() - t_start) * 1000

            sampled = []
            for choice in data.get("choices", []):
                sampled.extend((choice.get("meta_info") or {}).get("output_token_logprobs", []))

            return ChatCompletionResponse(
                id=data.get("id", f"chatcmpl-{uuid.uuid4()}"),
                created=data.get("created", int(time.time())),
                model=request.public_model or request.model,
                choices=[
                    ChatCompletionChoice(
                        index=c.get("index", 0),
                        message=ChatMessage(
                            role=c["message"]["role"],
                            content=strip_reasoning(c["message"].get("content", "")),
                        ),
                        finish_reason=c.get("finish_reason", "stop"),
                        # Qwen3/DeepSeek-R1 reasoning parser separates thinking into
                        # reasoning_content. We pass it through for clients that want
                        # it, but the main content field contains only the final answer.
                        reasoning_content=c["message"].get("reasoning_content"),
                    )
                    for c in data.get("choices", [])
                ],
                usage=usage,
                nram_correlation=self._correlation(request, payload, sampled=sampled),
            )
        except TimeoutError as exc:
            await self.abort(payload["rid"], reason="deadline_exceeded")
            raise SGLangEngineError("SGLang request deadline exceeded", status_code=504) from exc
        except asyncio.CancelledError:
            await self.abort(payload["rid"], reason="downstream_cancelled")
            raise
        except httpx.HTTPStatusError as e:
            raise SGLangEngineError(
                f"SGLang returned {e.response.status_code}: {e.response.text[:500]}",
                status_code=502,
            ) from e
        except httpx.RequestError as e:
            raise SGLangEngineError(
                f"SGLang unreachable: {e}",
                status_code=503,
            ) from e

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Streaming completion through SGLang — real SSE passthrough."""
        nram_enabled = self._is_nram_enabled(request)

        developer_instruction = None
        plan_fragment = None
        if nram_enabled:
            profile_name = (request.nram or {}).get("profile", DEFAULT_PROFILE)
            # UNIFIED: use resolve_request_state for both instruction and logit policy
            state = resolve_request_state(request.nram, profile_name)
            policy = compile_policy(state, max_tokens=request.max_tokens or 512, profile_name=profile_name)
            developer_instruction = policy.developer_instruction
            developer_instruction = _apply_phenomenon_mix(
                developer_instruction, (request.nram or {}).get("phenomenon_weights")
            )

            user_content = ""
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break

            plan = await build_rhetorical_plan(self, user_content)
            if plan is not None:
                plan_fragment = plan_to_prompt_fragment(plan)
            if not (request.nram or {}).get("prompt_steering_enabled", True):
                developer_instruction = None
                plan_fragment = None

        payload = self._build_upstream_payload(
            request, nram_enabled, developer_instruction, plan_fragment
        )
        payload = self._finalize_payload(request, payload, stream=True)

        completed = False
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                correlation = self._correlation(request, payload)
                stream_state = _StreamState(request.public_model or request.model, correlation)
                async for line in resp.aiter_lines():
                    if line.strip() == "data: [DONE]":
                        completed = True
                        yield b"data: [DONE]\n\n"
                    elif line.startswith("data: "):
                        for out_line in stream_state.feed(line):
                            yield (out_line + "\n\n").encode()
                if not completed:
                    completed = True
                    yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            await self.abort(payload["rid"], reason="stream_cancelled")
            raise
        except httpx.HTTPStatusError as e:
            error_chunk = {
                "error": {
                    "message": f"SGLang returned {e.response.status_code}",
                    "type": "inference_error",
                    "param": None,
                    "code": "upstream_inference_failed",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            completed = True
        except httpx.RequestError as e:
            error_chunk = {
                "error": {
                    "message": f"SGLang unreachable: {e}",
                    "type": "inference_error",
                    "param": None,
                    "code": "upstream_inference_failed",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            completed = True
        finally:
            if not completed:
                await self.abort(payload["rid"], reason="stream_closed")

    async def health(self) -> Dict[str, Any]:
        """Check SGLang health."""
        try:
            resp = await self._client.get("/models", timeout=5.0)
            resp.raise_for_status()
            models = resp.json()
            return {
                "engine": "sglang",
                "healthy": True,
                "models": [m.get("id") for m in models.get("data", [])],
            }
        except Exception as e:
            return {"engine": "sglang", "healthy": False, "error": str(e)}
