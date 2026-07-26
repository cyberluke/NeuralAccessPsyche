"""
NRAMChatClient for Microsoft Agent Framework (MAF).

Implements BaseChatClient from agent-framework-core, routing all inference
through the NRAM API so persona steering, logit processors, and telemetry
apply transparently to every agent invocation.

This is the ONLY integration boundary between MAF and NRAM. Agents created
with `client.as_agent(...)` or `Agent(client=..., ...)` will automatically
use NRAM steering when the model alias maps to an NRAM-enabled profile.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any, ClassVar

import httpx

try:
    from agent_framework import (
        BaseChatClient,
        ChatResponse,
        ChatResponseUpdate,
        Content,
        Message,
        ResponseStream,
    )
    MAF_AVAILABLE = True
except ImportError:
    MAF_AVAILABLE = False

logger = logging.getLogger(__name__)


class NRAMMAFChatClient(BaseChatClient[Any] if MAF_AVAILABLE else object):
    """MAF BaseChatClient that routes through the NRAM API.

    Every agent created with this client gets NRAM steering, persona
    profiles, and token-level telemetry for free.

    Usage:
        client = NRAMMAFChatClient(
            nram_api_base="http://localhost:8000/v1",
            default_model="nram-qwen3-14b-awq",
        )
        agent = client.as_agent(
            instructions="You are a visionary product designer.",
            name="visionary",
        )
        response = await agent.run("Design a new AI interface.")
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "NRAMMAFChatClient"

    def __init__(
        self,
        nram_api_base: str = "http://localhost:8000/v1",
        api_key: str = "dev-nram-key",
        default_model: str = "nram-qwen3-14b-awq",
        default_nram_profile: str | None = None,
        **kwargs: Any,
    ) -> None:
        if MAF_AVAILABLE:
            super().__init__(**kwargs)
        self._base_url = nram_api_base.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._default_nram_profile = default_nram_profile
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=None,  # No timeout — NRAM generation can be slow
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def close(self) -> None:
        await self._http.aclose()

    def _nram_options_from_maf(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Extract NRAM-specific options from MAF options dict."""
        nram_opts: dict[str, Any] = {}

        # Direct nram key
        if "nram" in options:
            nram_opts.update(options["nram"])

        # Profile from model alias or explicit setting
        if "nram_profile" in options:
            nram_opts["profile"] = options["nram_profile"]
        elif self._default_nram_profile:
            nram_opts["profile"] = self._default_nram_profile

        if nram_opts:
            nram_opts.setdefault("enabled", True)

        return nram_opts

    def _build_nram_body(
        self,
        messages: Sequence[Any],
        options: Mapping[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        """Convert MAF messages + options into an NRAM API request body."""
        # Convert MAF Message objects to OpenAI-compatible dicts
        msg_dicts: list[dict[str, str]] = []
        for msg in messages:
            role = getattr(msg, "role", "user")
            text = getattr(msg, "text", None) or ""
            if not text and hasattr(msg, "contents"):
                # MAF Message with content blocks
                parts = []
                for c in msg.contents:
                    if hasattr(c, "text"):
                        parts.append(c.text)
                    elif isinstance(c, str):
                        parts.append(c)
                text = " ".join(parts)
            msg_dicts.append({"role": str(role), "content": text})

        body: dict[str, Any] = {
            "model": options.get("model", self._default_model),
            "messages": msg_dicts,
            "stream": stream,
        }

        # Forward standard sampling params
        for key in ("temperature", "top_p", "max_tokens", "seed", "stop", "presence_penalty"):
            if key in options and options[key] is not None:
                body[key] = options[key]

        # NRAM steering
        nram_opts = self._nram_options_from_maf(options)
        if nram_opts:
            body["nram"] = nram_opts

        # Structured output grammar
        if "response_format" in options and options["response_format"]:
            body["response_format"] = options["response_format"]

        return body

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Any],
        stream: bool = False,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """MAF BaseChatClient hook — routes to NRAM API."""
        body = self._build_nram_body(messages, options, stream)

        if stream:
            return self._build_response_stream(
                self._stream_impl(body),
                response_format=options.get("response_format"),
            )

        async def _get_response() -> Any:
            resp = await self._http.post("/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            content = choice["message"].get("content", "")

            return ChatResponse(
                messages=[Message(role="assistant", contents=[content])],
                model=data.get("model", self._default_model),
                response_id=data.get("id", ""),
            )

        return _get_response()

    async def _stream_impl(self, body: dict[str, Any]) -> AsyncIterable[Any]:
        """Stream SSE chunks from NRAM API, yielding MAF ChatResponseUpdate."""
        response_id = ""
        async with self._http.stream("POST", "/chat/completions", json=body) as resp:
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
                if not response_id:
                    response_id = chunk.get("id", "")
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield ChatResponseUpdate(
                        contents=[Content.from_text(content)],
                        role="assistant",
                        response_id=response_id,
                        model=chunk.get("model", self._default_model),
                    )
