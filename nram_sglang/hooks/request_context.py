"""
Request-scoped context for NRAM hooks.

This module provides async-safe storage for request-specific hook configuration
using contextvars (not threading.local). Each request gets its own context that is:
- Installed before generation begins
- Active during all forward pass invocations for that request
- Removed after generation completes (success, failure, or abort)
- Isolated from other concurrent requests

Critical: This is NOT global state. Each request has its own context identified
by request_id. Stale contexts are rejected and cleaned up.

Async Safety: Uses contextvars.ContextVar instead of threading.local() because
SGLang uses asyncio for concurrent request handling. threading.local() causes
cross-contamination between async tasks on the same thread.
"""
from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class InterventionConfig:
    """Configuration for a single intervention (activation vector, conceptor, etc.)."""
    intervention_id: str
    intervention_type: str  # "activation_addition", "conceptor", "probe", etc.
    layer_index: int
    strength: float = 1.0
    phase: Optional[str] = None  # "prefill", "decode", or None for both
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class NRAMHookContext:
    """
    Request-scoped context for NRAM hooks.
    
    Contains all configuration needed by hooks during forward pass:
    - Request ID for isolation
    - Active interventions (vectors, conceptors, probes)
    - Telemetry settings
    - State for closed-loop controllers
    
    Lifecycle:
    - Created when request arrives
    - Installed as thread-local before forward pass
    - Removed after generation completes
    - Never shared between requests
    """
    request_id: str
    interventions: List[InterventionConfig] = field(default_factory=list)
    telemetry_enabled: bool = False
    telemetry_max_steps: int = 16
    
    # Closed-loop state
    latent_loop_state: Optional[Dict[str, Any]] = None
    semantic_loop_state: Optional[Dict[str, Any]] = None
    
    # Runtime tracking
    created_at: float = field(default_factory=time.time)
    hook_invocation_count: int = 0
    last_invocation_at: Optional[float] = None
    
    # Telemetry events (bounded)
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_intervention(self, intervention: InterventionConfig) -> None:
        """Add an intervention to this request context."""
        self.interventions.append(intervention)
        logger.debug(f"Added intervention {intervention.intervention_id} to request {self.request_id}")
    
    def get_interventions_for_layer(self, layer_index: int, phase: Optional[str] = None) -> List[InterventionConfig]:
        """Get all interventions targeting a specific layer and phase."""
        result = []
        for intervention in self.interventions:
            if not intervention.enabled:
                continue
            if intervention.layer_index != layer_index:
                continue
            if phase is not None and intervention.phase is not None and intervention.phase != phase:
                continue
            result.append(intervention)
        return result
    
    def record_hook_invocation(self) -> None:
        """Record that a hook was invoked."""
        self.hook_invocation_count += 1
        self.last_invocation_at = time.time()
    
    def emit_telemetry_event(self, event: Dict[str, Any]) -> None:
        """Emit a bounded telemetry event."""
        if not self.telemetry_enabled:
            return
        if len(self.telemetry_events) >= self.telemetry_max_steps:
            return  # Bounded
        self.telemetry_events.append(event)
    
    def reset(self) -> None:
        """Reset request-scoped state (called at end of request)."""
        self.hook_invocation_count = 0
        self.last_invocation_at = None
        self.telemetry_events.clear()
        self.latent_loop_state = None
        self.semantic_loop_state = None
        logger.debug(f"Reset context for request {self.request_id}")


# Async-safe context variable for current request context.
# Uses contextvars.ContextVar instead of threading.local() because SGLang
# uses asyncio for concurrent request handling. ContextVar provides proper
# isolation across async tasks on the same thread.
_current_context_var: contextvars.ContextVar[Optional[NRAMHookContext]] = contextvars.ContextVar(
    "nram_hook_context", default=None
)


def set_current_context(context: Optional[NRAMHookContext]) -> None:
    """
    Set the current request context for this async task.

    Called before generation begins. The context is active for all
    forward pass invocations in this async task until removed.

    Uses contextvars.ContextVar for async-safe isolation. Each asyncio
    task gets its own context, preventing cross-contamination between
    concurrent requests on the same thread.

    Args:
        context: The request context to install, or None to clear
    """
    _current_context_var.set(context)
    if context is not None:
        logger.debug(f"Installed context for request {context.request_id}")
    else:
        logger.debug("Cleared current context")


def get_current_context() -> Optional[NRAMHookContext]:
    """
    Get the current request context for this async task.

    Called by hooks during forward pass to access request-specific
    configuration and state.

    Uses contextvars.ContextVar for async-safe isolation. Each asyncio
    task gets its own context, preventing cross-contamination between
    concurrent requests on the same thread.

    Returns:
        The current context, or None if no request is active
    """
    return _current_context_var.get()


def get_current_context_for_batch() -> Optional[NRAMHookContext]:
    """
    Get the current request context for batch processing.

    This is a wrapper around get_current_context() that provides
    compatibility with SGLang's batch processing model. Uses
    contextvars.ContextVar for async-safe context lookup.

    Returns:
        The current context, or None if no request is active
    """
    return get_current_context()


def clear_current_context() -> None:
    """Clear the current request context."""
    set_current_context(None)


class ContextManager:
    """
    Manages request contexts with lifecycle tracking.

    Ensures contexts are properly installed and removed, preventing
    state leakage between requests. Uses contextvars.ContextVar for
    async-safe context isolation.
    """

    def __init__(self):
        self._active_contexts: Dict[str, NRAMHookContext] = {}
        self._lock = threading.Lock()

    def create_context(self, request_id: str) -> NRAMHookContext:
        """Create a new context for a request."""
        with self._lock:
            # Reject duplicate request IDs
            if request_id in self._active_contexts:
                raise ValueError(f"Context already exists for request {request_id}")

            context = NRAMHookContext(request_id=request_id)
            self._active_contexts[request_id] = context
            logger.info(f"Created context for request {request_id}")
            return context

    def install_context(self, request_id: str) -> None:
        """Install a context as the current async-task context."""
        with self._lock:
            if request_id not in self._active_contexts:
                raise ValueError(f"No context exists for request {request_id}")
            context = self._active_contexts[request_id]

        set_current_context(context)

    def remove_context(self, request_id: str) -> None:
        """Remove a context and clear it from the current async task."""
        with self._lock:
            if request_id in self._active_contexts:
                context = self._active_contexts[request_id]
                context.reset()
                del self._active_contexts[request_id]
                logger.info(f"Removed context for request {request_id}")

        # Clear contextvar if this was the current context
        current = get_current_context()
        if current is not None and current.request_id == request_id:
            clear_current_context()
    
    def get_context(self, request_id: str) -> Optional[NRAMHookContext]:
        """Get a context by request ID."""
        with self._lock:
            return self._active_contexts.get(request_id)
    
    def cleanup_stale_contexts(self, max_age_seconds: float = 300.0) -> int:
        """Remove contexts older than max_age_seconds. Returns count removed."""
        now = time.time()
        removed = 0
        
        with self._lock:
            stale_ids = [
                request_id
                for request_id, context in self._active_contexts.items()
                if (now - context.created_at) > max_age_seconds
            ]
            
            for request_id in stale_ids:
                context = self._active_contexts[request_id]
                context.reset()
                del self._active_contexts[request_id]
                removed += 1
                logger.warning(f"Cleaned up stale context for request {request_id}")
        
        return removed


# Global context manager instance
context_manager = ContextManager()
