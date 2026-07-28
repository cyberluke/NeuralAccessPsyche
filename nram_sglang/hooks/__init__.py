"""
NRAM SGLang forward hooks for hidden-state interventions.

This package provides hooks that execute INSIDE the SGLang model forward pass
to modify hidden states at specific transformer layers.

Architecture:
- Hooks are registered on Qwen decoder layers via model.named_modules()
- Request-scoped configuration is propagated through NRAMHookContext
- Hooks emit telemetry proving execution inside real forward path
- Interventions are applied during both prefill and decode phases

Critical: These hooks modify actual hidden states, not logits.
A logit modification is NOT a hidden-state intervention.
"""
from nram_sglang.hooks.factory import NRAMHookFactory
from nram_sglang.hooks.request_context import NRAMHookContext, get_current_context
from nram_sglang.hooks.qwen_hook import QwenDecoderHook
from nram_sglang.hooks.telemetry import HookTelemetry

__all__ = [
    "NRAMHookFactory",
    "NRAMHookContext",
    "get_current_context",
    "QwenDecoderHook",
    "HookTelemetry",
]
