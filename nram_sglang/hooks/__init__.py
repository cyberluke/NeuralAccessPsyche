"""
NRAM SGLang forward hooks for hidden-state interventions.

This package provides hooks that execute INSIDE the SGLang model forward pass
to modify hidden states at specific transformer layers.

Architecture:
- Hooks are registered on Qwen decoder layers via model.named_modules()
- Request-scoped configuration is propagated through NRAMHookContext
- Batch-aware context manager replaces thread-local for SGLang compatibility
- Hooks emit telemetry proving execution inside real forward path
- Interventions are applied during both prefill and decode phases

Critical: These hooks modify actual hidden states, not logits.
A logit modification is NOT a hidden-state intervention.
"""
from nram_sglang.hooks.factory import NRAMHookFactory, HookManager, hook_manager
from nram_sglang.hooks.request_context import (
    NRAMHookContext,
    InterventionConfig,
    get_current_context,
    context_manager,
)
from nram_sglang.hooks.batch_context import (
    BatchContextManager,
    batch_context_manager,
    register_nram_request,
    complete_nram_request,
    get_context_for_batch_index,
)
from nram_sglang.hooks.qwen_hook import QwenDecoderHook
from nram_sglang.hooks.telemetry import HookTelemetry

__all__ = [
    "NRAMHookFactory",
    "HookManager",
    "hook_manager",
    "NRAMHookContext",
    "InterventionConfig",
    "get_current_context",
    "context_manager",
    "BatchContextManager",
    "batch_context_manager",
    "register_nram_request",
    "complete_nram_request",
    "get_context_for_batch_index",
    "QwenDecoderHook",
    "HookTelemetry",
]
