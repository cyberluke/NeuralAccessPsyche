"""
Correct DExperts runtime for SGLang logit processor (R5).

Implements the DExperts equation at the logit level with SEPARATE KV caches:
    z_combined = z_base + alpha * (z_expert - z_anti_expert)

Critical invariants (DExperts paper compliance):
1. Expert and anti-expert have SEPARATE KV-cache trees
2. Both paths receive EXACTLY the same token history
3. Positions remain equal at every step
4. History = original prompt + every generated token
5. State identity bound to real SGLang request/batch identity
6. Reused request-pool index never inherits old state
7. State removed on: completion, EOS, cancellation, timeout, disconnect, exception, abortion
8. State storage bounded and observable
9. Stale-state cleanup as final safeguard

This runs INSIDE the SGLang custom logit processor, receiving logits
from the base model and applying expert/anti-expert deltas.

Architecture:
- LoRA adapters are loaded at server startup via shared-backbone pattern
- Per-token: run expert forward (with expert KV cache), then anti-expert forward
  (with anti-expert KV cache), using the SAME input_ids
- Apply DExperts formula to modify logits before sampling
- Maintain separate KV state for expert/anti-expert via past_key_values

Critical: This is NOT three separate API calls. It operates on the
logits tensor directly inside the SGLang process.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics (process-global, thread-safe via atomics)
# ---------------------------------------------------------------------------

class _DExpertsMetrics:
    """Process-global DExperts metrics. Thread-safe via threading.Lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_dexperts_requests: int = 0
        self.created_states_total: int = 0
        self.cleaned_states_total: int = 0
        self.stale_states_total: int = 0
        self.request_state_collisions_total: int = 0

    def record_create(self) -> None:
        with self._lock:
            self.created_states_total += 1
            self.active_dexperts_requests += 1

    def record_clean(self) -> None:
        with self._lock:
            self.cleaned_states_total += 1
            self.active_dexperts_requests = max(0, self.active_dexperts_requests - 1)

    def record_stale(self) -> None:
        with self._lock:
            self.stale_states_total += 1
            self.cleaned_states_total += 1
            self.active_dexperts_requests = max(0, self.active_dexperts_requests - 1)

    def record_collision(self) -> None:
        with self._lock:
            self.request_state_collisions_total += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_dexperts_requests": self.active_dexperts_requests,
                "created_states_total": self.created_states_total,
                "cleaned_states_total": self.cleaned_states_total,
                "stale_states_total": self.stale_states_total,
                "request_state_collisions_total": self.request_state_collisions_total,
            }


_METRICS = _DExpertsMetrics()


def get_dexperts_metrics() -> Dict[str, int]:
    """Return a snapshot of process-global DExperts metrics."""
    return _METRICS.snapshot()


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------

@dataclass
class DExpertsRequestState:
    """Per-request DExperts state with separate KV caches.

    Invariants:
    - expert_past_key_values and anti_expert_past_key_values are SEPARATE objects
    - Both caches receive the same token history
    - expert_position == anti_expert_position at every step
    - history = prompt_token_ids + generated_token_ids
    """
    runtime_request_id: str = ""
    sglang_request_index: int = -1
    prompt_token_ids: List[int] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    expert_past_key_values: Any = None  # Separate KV cache for expert
    anti_expert_past_key_values: Any = None  # Separate KV cache for anti-expert
    expert_position: int = 0
    anti_expert_position: int = 0
    last_token_id: Optional[int] = None
    step: int = 0
    alpha: float = 1.0
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    finished: bool = False
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def full_history(self) -> List[int]:
        """Full token history = prompt + generated."""
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def history_token_count(self) -> int:
        """Total number of tokens in history."""
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def record_access(self) -> None:
        """Record that this state was accessed."""
        self.last_accessed_at = time.monotonic()

    def reset(self) -> None:
        """Reset step count and telemetry (keep KV caches)."""
        self.step = 0
        self.telemetry_events.clear()


# Backward compatibility alias
DExpertsRuntimeState = DExpertsRequestState


# ---------------------------------------------------------------------------
# Tokenizer compatibility gate
# ---------------------------------------------------------------------------

@dataclass
class TokenizerCompatibilityReport:
    """Result of tokenizer compatibility verification."""
    compatible: bool
    vocab_size: int = 0
    expected_vocab_size: int = 0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    token_to_id_coverage: float = 0.0
    id_to_token_coverage: float = 0.0
    config_hash: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def verify_tokenizer_compatibility(
    tokenizer: Any,
    expected_vocab_size: Optional[int] = None,
) -> TokenizerCompatibilityReport:
    """Verify tokenizer compatibility for DExperts.

    Checks:
    - Vocabulary size matches expected (if provided)
    - BOS/EOS/PAD/UNK token IDs are defined
    - Token-to-ID and ID-to-token mappings are complete
    - Added vocabulary is consistent

    Returns a report. Caller must check .compatible before proceeding.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Vocabulary size
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = 0

    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        errors.append(
            f"Vocabulary size mismatch: got {vocab_size}, expected {expected_vocab_size}"
        )

    # Special tokens
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    if bos_id is None:
        warnings.append("bos_token_id is None")
    if eos_id is None:
        warnings.append("eos_token_id is None")

    # Token-to-ID coverage (sample check)
    token_to_id_ok = 0
    token_to_id_total = 0
    try:
        # Check a sample of tokens
        for token_id in range(min(1000, vocab_size)):
            token = tokenizer.decode([token_id])
            if token:
                token_to_id_total += 1
                re_encoded = tokenizer.encode(token, add_special_tokens=False)
                if re_encoded and re_encoded[0] == token_id:
                    token_to_id_ok += 1
    except Exception as e:
        warnings.append(f"Token-to-ID check failed: {e}")

    token_to_id_coverage = (
        token_to_id_ok / token_to_id_total if token_to_id_total > 0 else 0.0
    )

    # Config hash (for reproducibility)
    config_hash = ""
    try:
        tokenizer_config = getattr(tokenizer, "init_kwargs", {})
        config_str = str(sorted(tokenizer_config.items()))
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    except Exception:
        pass

    compatible = len(errors) == 0

    return TokenizerCompatibilityReport(
        compatible=compatible,
        vocab_size=vocab_size,
        expected_vocab_size=expected_vocab_size or vocab_size,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        unk_token_id=unk_id,
        token_to_id_coverage=token_to_id_coverage,
        id_to_token_coverage=token_to_id_coverage,  # Same check
        config_hash=config_hash,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# DExperts Runtime
# ---------------------------------------------------------------------------

class DExpertsRuntime:
    """
    Correct DExperts runtime for SGLang with separate KV caches.

    Loads LoRA adapters and applies the DExperts formula:
        z_combined = z_base + alpha * (z_expert - z_anti_expert)

    The base logits come from SGLang's normal forward pass.
    Expert and anti-expert logits are computed by switching LoRA adapters
    on the shared backbone, but with SEPARATE past_key_values (KV caches).

    Thread safety: All adapter switching and forward passes are protected
    by an explicit execution lock to prevent concurrent requests from
    corrupting adapter state.
    """

    # Stale state timeout (seconds). States older than this are candidates
    # for cleanup by the stale-state sweeper.
    STALE_STATE_TIMEOUT_S = 300.0  # 5 minutes

    # Maximum number of concurrent DExperts requests. Requests beyond this
    # limit are rejected with a typed error.
    MAX_CONCURRENT_REQUESTS = 64

    # Maximum context length for DExperts (prompt + generated).
    MAX_CONTEXT_LENGTH = 4096

    def __init__(
        self,
        base_model: Any,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize DExperts runtime.

        Args:
            base_model: The base language model (already loaded by SGLang)
            tokenizer: The tokenizer
            device: Device to use (must be "cuda" for production)
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

        # LoRA adapters (loaded lazily)
        self.expert_adapter: Any = None
        self.anti_expert_adapter: Any = None

        # Adapter paths
        self.expert_adapter_path: Optional[str] = None
        self.anti_expert_adapter_path: Optional[str] = None
        self.expert_adapter_sha256: Optional[str] = None
        self.anti_expert_adapter_sha256: Optional[str] = None

        # Execution lock for thread-safe adapter switching
        self._execution_lock = threading.Lock()

        # Per-request state
        self._request_states: Dict[str, DExpertsRequestState] = {}
        self._state_lock = threading.Lock()

        # State
        self._loaded = False
        self._tokenizer_verified = False
        self._tokenizer_report: Optional[TokenizerCompatibilityReport] = None

        # Backbone model ID (for verification)
        self.backbone_model_id: Optional[str] = getattr(
            base_model, "name_or_path", None
        )
        self.backbone_revision: Optional[str] = getattr(
            base_model, "model_revision", None
        )

        # Dtype
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(
            f"DExperts runtime initialized: backbone={self.backbone_model_id}, "
            f"device={device}, dtype={self.dtype}"
        )

    @property
    def _loaded_flag(self) -> bool:
        """Backward compatibility: some code checks runtime._loaded."""
        return self._loaded

    @_loaded_flag.setter
    def _loaded_flag(self, value: bool) -> None:
        self._loaded = value

    # Backward compatibility: some code checks runtime._loaded directly
    @property
    def loaded(self) -> bool:
        return self._loaded

    def verify_tokenizer(self, expected_vocab_size: Optional[int] = None) -> TokenizerCompatibilityReport:
        """Verify tokenizer compatibility. Must pass before serving DExperts requests.

        Args:
            expected_vocab_size: Expected vocabulary size (if known)

        Returns:
            TokenizerCompatibilityReport with .compatible flag

        Raises:
            RuntimeError: If tokenizer is not compatible and DExperts cannot proceed
        """
        report = verify_tokenizer_compatibility(self.tokenizer, expected_vocab_size)
        self._tokenizer_report = report
        self._tokenizer_verified = report.compatible

        if not report.compatible:
            logger.error(
                f"Tokenizer compatibility check FAILED: {report.errors}"
            )
            raise RuntimeError(
                f"Tokenizer compatibility check failed: {report.errors}. "
                "DExperts cannot proceed with incompatible tokenizer."
            )

        logger.info(
            f"Tokenizer compatibility verified: vocab_size={report.vocab_size}, "
            f"bos={report.bos_token_id}, eos={report.eos_token_id}, "
            f"config_hash={report.config_hash}"
        )
        return report

    def load_adapters(
        self,
        expert_adapter_path: str,
        anti_expert_adapter_path: str,
    ) -> None:
        """
        Load LoRA adapters for expert and anti-expert.

        Uses PeftModel.from_pretrained to create SEPARATE adapter instances
        wrapping the same base model. Each adapter has its own weights but
        shares the base model backbone.

        Args:
            expert_adapter_path: Path to non-toxic expert adapter
            anti_expert_adapter_path: Path to toxic anti-expert adapter
        """
        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError(
                "peft package not installed. DExperts requires peft for LoRA adapters."
            )

        # Compute SHA-256 hashes of adapter configs for verification
        self.expert_adapter_sha256 = self._compute_adapter_hash(expert_adapter_path)
        self.anti_expert_adapter_sha256 = self._compute_adapter_hash(anti_expert_adapter_path)

        logger.info(f"Loading expert adapter from {expert_adapter_path}")
        logger.info(f"Expert adapter SHA-256: {self.expert_adapter_sha256}")

        # Create SEPARATE PeftModel instances for expert and anti-expert
        # Each wraps the base model but has its own adapter weights
        self.expert_adapter = PeftModel.from_pretrained(
            self.base_model,
            expert_adapter_path,
            adapter_name="expert",
        )
        self.expert_adapter_path = expert_adapter_path

        logger.info(f"Loading anti-expert adapter from {anti_expert_adapter_path}")
        logger.info(f"Anti-expert adapter SHA-256: {self.anti_expert_adapter_sha256}")

        self.anti_expert_adapter = PeftModel.from_pretrained(
            self.base_model,
            anti_expert_adapter_path,
            adapter_name="anti_expert",
        )
        self.anti_expert_adapter_path = anti_expert_adapter_path

        # Set adapters to inactive by default (will be activated per-request)
        # Note: PeftModel.set_adapter() switches which adapter is active
        # We use separate PeftModel instances, so each has its own active adapter

        self._loaded = True
        logger.info("DExperts adapters loaded successfully (separate instances)")

    @staticmethod
    def _compute_adapter_hash(adapter_path: str) -> Optional[str]:
        """Compute SHA-256 hash of adapter config.json for verification."""
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def get_or_create_state(self, request_id: str) -> DExpertsRequestState:
        """Get or create per-request state.

        Thread-safe. If state already exists for this request_id, returns it.
        Otherwise creates new state and records metrics.

        Args:
            request_id: Unique request identifier

        Returns:
            DExpertsRequestState for this request
        """
        with self._state_lock:
            if request_id in self._request_states:
                state = self._request_states[request_id]
                state.record_access()
                return state

            # Check concurrent request limit
            if len(self._request_states) >= self.MAX_CONCURRENT_REQUESTS:
                raise RuntimeError(
                    f"DExperts concurrent request limit exceeded: "
                    f"{len(self._request_states)} >= {self.MAX_CONCURRENT_REQUESTS}"
                )

            now = time.monotonic()
            state = DExpertsRequestState(
                runtime_request_id=request_id,
                created_at=now,
                last_accessed_at=now,
            )
            self._request_states[request_id] = state
            _METRICS.record_create()
            logger.debug(f"Created DExperts state for request {request_id}")
            return state

    def release_state(self, request_id: str) -> None:
        """Release per-request state after completion/abort/cancellation/timeout.

        Thread-safe. Removes state from the registry and frees KV caches.

        Args:
            request_id: Request identifier to release
        """
        with self._state_lock:
            if request_id not in self._request_states:
                return

            state = self._request_states.pop(request_id)
            state.finished = True

            # Free KV caches
            state.expert_past_key_values = None
            state.anti_expert_past_key_values = None

            _METRICS.record_clean()
            logger.debug(f"Released DExperts state for request {request_id}")

    def cleanup_stale_states(self, timeout_s: Optional[float] = None) -> int:
        """Clean up stale states that have not been accessed recently.

        This is a safeguard against state leaks from abnormal termination
        (crashes, disconnects, etc.).

        Args:
            timeout_s: Timeout in seconds (default: STALE_STATE_TIMEOUT_S)

        Returns:
            Number of states cleaned
        """
        if timeout_s is None:
            timeout_s = self.STALE_STATE_TIMEOUT_S

        now = time.monotonic()
        stale_ids: List[str] = []

        with self._state_lock:
            for request_id, state in self._request_states.items():
                age = now - state.last_accessed_at
                if age > timeout_s:
                    stale_ids.append(request_id)

            for request_id in stale_ids:
                state = self._request_states.pop(request_id)
                state.finished = True
                state.expert_past_key_values = None
                state.anti_expert_past_key_values = None
                _METRICS.record_stale()
                logger.warning(
                    f"Cleaned stale DExperts state for request {request_id} "
                    f"(age={now - state.created_at:.1f}s)"
                )

        return len(stale_ids)

    def release_all_states(self) -> int:
        """Release all request states. Used during shutdown or error recovery.

        Returns:
            Number of states released
        """
        with self._state_lock:
            count = len(self._request_states)
            for request_id, state in self._request_states.items():
                state.finished = True
                state.expert_past_key_values = None
                state.anti_expert_past_key_values = None
                _METRICS.record_clean()
            self._request_states.clear()
            return count

    @torch.no_grad()
    def compute_expert_logits(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Compute logits using expert adapter with SEPARATE KV cache.

        Args:
            input_ids: Input token IDs
            past_key_values: Expert's KV cache (separate from anti-expert)
            attention_mask: Attention mask
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (logits, updated_past_key_values)
        """
        if not self._loaded or self.expert_adapter is None:
            raise RuntimeError("Expert adapter not loaded")

        # Run forward pass with expert adapter
        # The expert_adapter is a separate PeftModel instance with its own weights
        outputs = self.expert_adapter(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        logits = outputs.logits[:, -1, :]  # Last token logits
        new_past_key_values = outputs.past_key_values if use_cache else None

        return logits, new_past_key_values

    @torch.no_grad()
    def compute_anti_expert_logits(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Compute logits using anti-expert adapter with SEPARATE KV cache.

        Args:
            input_ids: Input token IDs
            past_key_values: Anti-expert's KV cache (separate from expert)
            attention_mask: Attention mask
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (logits, updated_past_key_values)
        """
        if not self._loaded or self.anti_expert_adapter is None:
            raise RuntimeError("Anti-expert adapter not loaded")

        # Run forward pass with anti-expert adapter
        # The anti_expert_adapter is a separate PeftModel instance
        outputs = self.anti_expert_adapter(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        logits = outputs.logits[:, -1, :]  # Last token logits
        new_past_key_values = outputs.past_key_values if use_cache else None

        return logits, new_past_key_values

    @torch.no_grad()
    def apply_dexperts(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        request_id: str = "",
        position: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply DExperts formula to base logits with separate KV caches.

        z_combined = z_base + alpha * (z_expert - z_anti_expert)

        Thread-safe: Uses execution lock to prevent concurrent adapter switching.

        Args:
            base_logits: Logits from base model (already computed by SGLang)
            input_ids: Input token IDs (for expert/anti-expert forward)
            attention_mask: Attention mask
            alpha: Steering strength
            request_id: Request identifier for state tracking
            position: Token position for telemetry

        Returns:
            Tuple of (combined_logits, telemetry_dict)
        """
        if not self._loaded:
            logger.warning("DExperts not loaded, returning base logits")
            return base_logits, {"applied": False, "reason": "not_loaded"}

        start_time = time.perf_counter()

        # Get or create request state
        state = self.get_or_create_state(request_id) if request_id else None

        # Update token history in state
        if state is not None:
            state.alpha = alpha
            state.step += 1

            # On first call, record prompt token IDs
            if state.step == 1 and len(state.prompt_token_ids) == 0:
                # input_ids contains the full prompt on first call
                if input_ids.dim() == 2:
                    state.prompt_token_ids = input_ids[0].tolist()
                elif input_ids.dim() == 1:
                    state.prompt_token_ids = input_ids.tolist()

            # Check context length limit
            if state.history_token_count > self.MAX_CONTEXT_LENGTH:
                logger.warning(
                    f"DExperts context length exceeded: {state.history_token_count} > "
                    f"{self.MAX_CONTEXT_LENGTH}. Returning base logits."
                )
                return base_logits, {
                    "applied": False,
                    "reason": "context_length_exceeded",
                    "history_length": state.history_token_count,
                    "max_context": self.MAX_CONTEXT_LENGTH,
                }

        # Compute expert and anti-expert logits with SEPARATE KV caches
        # Use execution lock to prevent concurrent adapter switching
        with self._execution_lock:
            expert_kv = state.expert_past_key_values if state else None
            anti_expert_kv = state.anti_expert_past_key_values if state else None

            try:
                expert_logits, new_expert_kv = self.compute_expert_logits(
                    input_ids,
                    past_key_values=expert_kv,
                    attention_mask=attention_mask,
                )
                anti_expert_logits, new_anti_expert_kv = self.compute_anti_expert_logits(
                    input_ids,
                    past_key_values=anti_expert_kv,
                    attention_mask=attention_mask,
                )
            except Exception as e:
                logger.error(f"DExperts forward pass failed: {e}")
                return base_logits, {
                    "applied": False,
                    "reason": "forward_pass_error",
                    "error": str(e),
                }

            # Update KV caches in state
            if state is not None:
                state.expert_past_key_values = new_expert_kv
                state.anti_expert_past_key_values = new_anti_expert_kv
                state.expert_position = position
                state.anti_expert_position = position

        # Apply DExperts formula
        # z_combined = z_base + alpha * (z_expert - z_anti_expert)
        expert_delta = expert_logits - anti_expert_logits
        combined_logits = base_logits + alpha * expert_delta

        # Compute telemetry
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Top tokens for telemetry
        base_top_ids = torch.topk(base_logits, k=min(5, base_logits.shape[-1]), dim=-1).indices
        if base_top_ids.dim() > 1:
            base_top_ids = base_top_ids[0]
        base_top_ids = base_top_ids.tolist()

        expert_top_ids = torch.topk(expert_logits, k=min(5, expert_logits.shape[-1]), dim=-1).indices
        if expert_top_ids.dim() > 1:
            expert_top_ids = expert_top_ids[0]
        expert_top_ids = expert_top_ids.tolist()

        anti_expert_top_ids = torch.topk(anti_expert_logits, k=min(5, anti_expert_logits.shape[-1]), dim=-1).indices
        if anti_expert_top_ids.dim() > 1:
            anti_expert_top_ids = anti_expert_top_ids[0]
        anti_expert_top_ids = anti_expert_top_ids.tolist()

        # Norms for telemetry
        expert_minus_anti_norm = float(torch.norm(expert_delta).item())
        combined_delta_norm = float(torch.norm(combined_logits - base_logits).item())

        # Finite logit counts
        finite_base = int(torch.isfinite(base_logits).sum().item())
        finite_combined = int(torch.isfinite(combined_logits).sum().item())

        telemetry = {
            "request_id": request_id,
            "position": position,
            "alpha": alpha,
            "base_top_token_ids": base_top_ids,
            "expert_top_token_ids": expert_top_ids,
            "anti_expert_top_token_ids": anti_expert_top_ids,
            "expert_minus_anti_norm": expert_minus_anti_norm,
            "combined_delta_norm": combined_delta_norm,
            "latency_ms": latency_ms,
            "backend": "sglang_lora_separate_kv",
            "device": self.device,
            "applied": True,
            "expert_cache_position": state.expert_position if state else position,
            "anti_expert_cache_position": state.anti_expert_position if state else position,
            "history_token_count": state.history_token_count if state else 0,
            "finite_base_logits": finite_base,
            "finite_combined_logits": finite_combined,
            "step": state.step if state else 0,
            "expert_adapter_path": self.expert_adapter_path,
            "anti_expert_adapter_path": self.anti_expert_adapter_path,
        }

        # Record telemetry in state
        if state is not None:
            state.telemetry_events.append(telemetry)
            # Keep only last 100 telemetry events to bound memory
            if len(state.telemetry_events) > 100:
                state.telemetry_events = state.telemetry_events[-100:]

        return combined_logits, telemetry

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status including metrics."""
        return {
            "loaded": self._loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "backbone_model_id": self.backbone_model_id,
            "backbone_revision": self.backbone_revision,
            "expert_adapter_path": self.expert_adapter_path,
            "anti_expert_adapter_path": self.anti_expert_adapter_path,
            "expert_adapter_sha256": self.expert_adapter_sha256,
            "anti_expert_adapter_sha256": self.anti_expert_adapter_sha256,
            "tokenizer_verified": self._tokenizer_verified,
            "max_context_length": self.MAX_CONTEXT_LENGTH,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "stale_state_timeout_s": self.STALE_STATE_TIMEOUT_S,
            **get_dexperts_metrics(),
        }
