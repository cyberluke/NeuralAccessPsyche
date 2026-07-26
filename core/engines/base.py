"""Base engine protocol and shared utilities."""
from __future__ import annotations

from core.contracts.openai import EngineCapabilities


# SGLang capabilities — set to True only after integration tests prove them.
SGLANG_CAPABILITIES = EngineCapabilities(
    streaming=True,
    strict_json=True,
    regex_grammar=True,
    cfg_grammar=True,
    token_masking=True,
    dynamic_logits=True,
    hidden_state_access=False,
    mid_generation_state_updates=True,
)
