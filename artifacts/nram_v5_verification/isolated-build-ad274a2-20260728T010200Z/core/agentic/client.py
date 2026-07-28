"""
NRAMChatClient — routes agent inference through NeuralAccessPsyche.

This is the integration boundary between an agentic framework (Microsoft
Agent Framework / AutoGen-style BaseChatClient) and the NRAM inference
gateway. The framework controls WHO thinks and WHEN; this client controls
HOW the model thinks (NRAM profile, steering, telemetry).

It implements the BaseChatClient contract (normal + streaming) and maps
agent invocation options onto the NRAM API request body.

Design notes:
- Does NOT call SGLang directly. Always routes through the NRAM API so the
  trusted policy compiler, provenance tracking, and session state apply.
- Structured output (llguidance grammar) is requested by the agent; this
  client forwards response_format to the NRAM API, which compiles it into
  the appropriate grammar for SGLang.
- Never exposes hidden chain-of-thought in returned messages.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


class ChatMessage:
    """Minimal framework-agnostic chat message."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class ChatResponse:
    """Minimal framework-agnostic chat response."""

    def __init__(
        self,
        content: str,
        model: str = "",
        finish_reason: str = "stop",
        usage: Optional[Dict[str, int]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.content = content
        self.model = model
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.telemetry = telemetry or {}


class NRAMChatClient:
    """BaseChatClient implementation that routes through the NRAM API.

    Compatible with the Microsoft Agent Framework / AutoGen BaseChatClient
    contract: get_response() and get_streaming_response().
    """

    def __init__(
        self,
        nram_api_base: str = "http://localhost:8000/v1",
        api_key: str = "dev-nram-key",
        default_model: str = "nram-deepseek-r1-qwen-7b",
        timeout: float = 300.0,
    ) -> None:
        self._base_url = nram_api_base.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _build_request_body(
        self,
        messages: List[Dict[str, str]],
        chat_options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Map agent invocation options onto the NRAM API request body."""
        opts = chat_options or {}

        body: Dict[str, Any] = {
            "model": opts.get("model", self._default_model),
            "messages": messages,
            "stream": stream,
        }

        # Forward standard sampling params
        for key in ("temperature", "top_p", "max_tokens", "seed", "stop"):
            if key in opts and opts[key] is not None:
                body[key] = opts[key]

        # NRAM options — the agent passes these under "nram" or
        # additional_chat_options["nram"].
        nram = opts.get("nram") or opts.get("additional_chat_options", {}).get("nram")
        if nram:
            body["nram"] = nram

        # Structured output grammar (llguidance) — forwarded as response_format.
        grammar = opts.get("grammar") or opts.get("response_format")
        if grammar:
            body["response_format"] = grammar

        # Telemetry flag
        if opts.get("include_telemetry"):
            body.setdefault("nram", {})["include_telemetry"] = True

        return body

    async def get_response(
        self,
        messages: List[Dict[str, str]],
        chat_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Non-streaming completion (BaseChatClient._inner_get_response)."""
        body = self._build_request_body(messages, chat_options, stream=False)
        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            telemetry=data.get("nram_telemetry", {}),
        )

    async def get_streaming_response(
        self,
        messages: List[Dict[str, str]],
        chat_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming completion (BaseChatClient._inner_get_streaming_response).

        Yields content deltas. Reasoning blocks are already stripped by the
        NRAM API's stream sanitization layer.
        """
        body = self._build_request_body(messages, chat_options, stream=True)
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    # -----------------------------------------------------------------------
    # Microsoft Agent Framework aliases
    # -----------------------------------------------------------------------

    async def _inner_get_response(self, *, messages, chat_options=None, **kwargs):
        """MS Agent Framework hook."""
        return await self.get_response(messages, chat_options, **kwargs)

    async def _inner_get_streaming_response(self, *, messages, chat_options=None, **kwargs):
        """MS Agent Framework hook."""
        async for chunk in self.get_streaming_response(messages, chat_options, **kwargs):
            yield chunk
