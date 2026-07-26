"""SGLang engine — OpenAI-compatible client with NRAM logit processor injection."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

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
from core.persona.compiler import compile_policy
from core.persona.planner import build_rhetorical_plan, plan_to_prompt_fragment
from core.persona.profiles import DEFAULT_PROFILE, PROFILES
from core.steering.nram_logit_processor import NRAMLogitProcessor
from core.steering.serialization import build_custom_params, serialize_processor
from core.steering.tokenizer_bias import TokenBiasCompiler

logger = logging.getLogger(__name__)

# Model alias → upstream model path mapping
MODEL_ALIAS_MAP: Dict[str, str] = {
    "nram-gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-20b-baseline": "openai/gpt-oss-20b",
    "nram-deepseek-r1-qwen-7b": "nram-deepseek-r1-qwen-7b",
    "deepseek-r1-qwen-7b-baseline": "nram-deepseek-r1-qwen-7b",
    "nram-qwen3-14b-awq": "nram-qwen3-14b-awq",
    "qwen3-14b-awq-baseline": "nram-qwen3-14b-awq",
}

# NRAM-enabled aliases
NRAM_ENABLED_ALIASES = {"nram-gpt-oss-20b", "nram-deepseek-r1-qwen-7b", "nram-qwen3-14b-awq"}

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

    def __init__(self, public_model: str) -> None:
        self._public_model = public_model
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
        model: str = "openai/gpt-oss-20b",
        tokenizer: Any = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._tokenizer = tokenizer
        self._bias_compiler: Optional[TokenBiasCompiler] = None
        if tokenizer is not None:
            self._bias_compiler = TokenBiasCompiler(tokenizer)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=30.0) if timeout else None,
        )
        self._serialized_processor = serialize_processor(NRAMLogitProcessor)
        logger.info(f"SGLangEngine initialized: base_url={base_url}, model={model}")

    async def close(self) -> None:
        await self._client.aclose()

    def _is_nram_enabled(self, request: ChatCompletionRequest) -> bool:
        """Check if NRAM steering is enabled for this request."""
        if request.model in NRAM_ENABLED_ALIASES:
            nram_opts = request.nram or {}
            return nram_opts.get("enabled", True)
        return False

    def _resolve_upstream_model(self, request: ChatCompletionRequest) -> str:
        """Map public alias to upstream model name."""
        return MODEL_ALIAS_MAP.get(request.model, self._model)

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
        if _FORBIDDEN_CLIENT_FIELDS & set(nram_opts.keys()):
            raise SGLangEngineError(
                "Forbidden field in nram options: processor injection is not allowed.",
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

            payload["custom_logit_processor"] = self._serialized_processor
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
            )
            
            # CRITICAL FIX: Inject request object for dynamic features
            # This enables: repetition penalty, phenomenon mixer, phase scheduling
            payload["custom_params"]["__req__"] = request

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

        payload = self._build_upstream_payload(
            request, nram_enabled, developer_instruction, plan_fragment
        )
        payload["stream"] = False

        # Qwen3 best practice for quantized models: presence_penalty reduces
        # endless repetitions. Use moderate value (0.8) to allow thematic repetition
        # while preventing loops. Only apply if the client didn't explicitly set it.
        if "presence_penalty" not in payload:
            payload["presence_penalty"] = 0.8
        # Qwen3 non-thinking mode recommends temperature=0.7, top_p=0.8.
        if request.temperature is None:
            payload["temperature"] = 0.7
        if request.top_p is None:
            payload["top_p"] = 0.8

        t_start = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

            usage = Usage(**data.get("usage", {}))
            latency_ms = (time.perf_counter() - t_start) * 1000

            # Feature 2: record provenance telemetry into the aggregator.
            # Token origins are derived from the active phenomenon weights
            # (applied_policy classification, not causal attribution).
            try:
                from core.nram.provenance_map import derive_origin_counts
                from core.nram.telemetry_agg import telemetry_aggregator

                nram_opts = request.nram or {}
                phen_weights = nram_opts.get("phenomenon_weights") or {}
                # Treat each active phenomenon (weight > threshold) as firing
                # roughly proportional to its weight * completion token count.
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                phen_counts = {
                    pid: max(1, int(w * completion_tokens))
                    for pid, w in phen_weights.items()
                    if isinstance(w, (int, float)) and w > 0.05
                }
                model_tokens = max(0, completion_tokens - sum(phen_counts.values()))
                origin_counts = derive_origin_counts(phen_counts, model_tokens)

                session_id = nram_opts.get("session_id") or request.model
                telemetry_aggregator.record(
                    session_id=session_id,
                    profile=nram_opts.get("profile", "baseline"),
                    token_origins=origin_counts,
                    latency_ms=latency_ms,
                    phenomena=phen_counts,
                )
            except Exception:  # telemetry must never break inference
                logger.debug("telemetry aggregation skipped", exc_info=True)

            return ChatCompletionResponse(
                id=data.get("id", f"chatcmpl-{uuid.uuid4()}"),
                created=data.get("created", int(time.time())),
                model=request.model,
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
            )
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

        payload = self._build_upstream_payload(
            request, nram_enabled, developer_instruction, plan_fragment
        )
        payload["stream"] = True

        # Qwen3 best practice for quantized models: presence_penalty reduces
        # endless repetitions. Use moderate value (0.8) to allow thematic repetition
        # while preventing loops. Only apply if the client didn't explicitly set it.
        if "presence_penalty" not in payload:
            payload["presence_penalty"] = 0.8
        # Qwen3 non-thinking mode recommends temperature=0.7, top_p=0.8.
        if request.temperature is None:
            payload["temperature"] = 0.7
        if request.top_p is None:
            payload["top_p"] = 0.8

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                state = _StreamState(request.model)
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        for out_line in state.feed(line):
                            yield (out_line + "\n\n").encode()
                    elif line.strip() == "data: [DONE]":
                        yield b"data: [DONE]\n\n"
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
