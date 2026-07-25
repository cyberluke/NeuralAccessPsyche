"""Legacy engine — wraps the existing LLMHandler for backward compatibility."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict

from core.contracts.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EngineCapabilities,
)


class LegacyEngine:
    """Wraps the original GuidanceHandler-based path."""

    capabilities = EngineCapabilities(
        streaming=False,
        strict_json=False,
        regex_grammar=False,
        cfg_grammar=False,
        token_masking=False,
        dynamic_logits=False,
        hidden_state_access=False,
        mid_generation_state_updates=False,
    )

    def __init__(self, llm_handler: Any) -> None:
        self._llm = llm_handler

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raw = await self._llm.generate_response(
            messages=request.messages,
            temperature=request.temperature or 1.0,
            max_tokens=request.max_tokens or 256,
            model=request.model,
        )
        return ChatCompletionResponse(**raw)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        raise NotImplementedError("LegacyEngine does not support streaming")

    async def health(self) -> Dict[str, Any]:
        return {"engine": "legacy", "healthy": True}
