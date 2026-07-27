"""OpenAI-compatible API contracts for NeuralAccessPsyche."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Public request model. Mirrors the OpenAI Chat Completions API."""

    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    nram: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"
    reasoning_content: Optional[str] = None  # Qwen3/DeepSeek-R1 reasoning (when reasoning_parser is active)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    # Correlation metadata only. Causal telemetry is emitted by the processor
    # in the SGLang runtime and must be matched by these values.
    nram_correlation: Optional[Dict[str, str]] = None


class OpenAIError(BaseModel):
    message: str
    type: str = "inference_error"
    param: Optional[str] = None
    code: str = "upstream_inference_failed"


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


class EngineCapabilities(BaseModel):
    streaming: bool = False
    strict_json: bool = False
    regex_grammar: bool = False
    cfg_grammar: bool = False
    token_masking: bool = False
    dynamic_logits: bool = False
    hidden_state_access: bool = False
    mid_generation_state_updates: bool = False


class GenerationEngine(Protocol):
    capabilities: EngineCapabilities

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ...

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        ...

    async def health(self) -> Dict[str, Any]:
        ...
