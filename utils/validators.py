"""Early public-request validation shared by every inference entry path."""
from __future__ import annotations

import math
from typing import Any, Dict

from fastapi import HTTPException

from core.contracts.nram_runtime import normalize_nram_options, stable_validation_error


def _reject(message: str, code: str, param: str) -> None:
    raise HTTPException(
        status_code=400,
        detail={
            "code": code,
            "param": param,
            "message": message,
        },
    )


def _finite_number(value: Any, *, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(f"{name} must be a number", "invalid_number", name)
    if not math.isfinite(float(value)):
        _reject(f"{name} must be finite", "non_finite_number", name)
    if not minimum <= float(value) <= maximum:
        _reject(
            f"{name} must be between {minimum} and {maximum}",
            "number_out_of_range",
            name,
        )


def _validate_response_format(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _reject("response_format must be an object", "invalid_response_format", "response_format")
    response_type = value.get("type")
    if response_type == "json_object" and set(value) == {"type"}:
        return
    if response_type != "json_schema":
        _reject(
            "response_format.type must be json_object or json_schema",
            "unsupported_response_format",
            "response_format.type",
        )
    wrapper = value.get("json_schema")
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("schema"), dict):
        _reject(
            "json_schema response_format requires json_schema.schema object",
            "invalid_json_schema",
            "response_format.json_schema.schema",
        )
    allowed = {"type", "json_schema"}
    if set(value) - allowed:
        _reject(
            "response_format contains unknown keys",
            "unknown_response_format_key",
            "response_format",
        )


def validate_request(request: Any) -> bool:
    """Validate and normalize a request before any upstream engine call."""
    if not hasattr(request, "messages") or not isinstance(request.messages, list) or not request.messages:
        _reject("Messages are required", "messages_required", "messages")

    for index, message in enumerate(request.messages):
        if not isinstance(message, dict):
            _reject("Each message must be an object", "invalid_message", f"messages.{index}")
        if set(message) - {"role", "content", "name", "tool_call_id", "tool_calls"}:
            _reject("Message contains unknown keys", "unknown_message_key", f"messages.{index}")
        if not isinstance(message.get("role"), str) or not message["role"]:
            _reject("Message role is required", "invalid_message_role", f"messages.{index}.role")
        if "content" not in message or not isinstance(message["content"], str):
            _reject("Message content must be text", "invalid_message_content", f"messages.{index}.content")

    _finite_number(request.temperature, name="temperature", minimum=0.0, maximum=2.0)
    # top_p must be strictly greater than 0.0 per SGLang sampling requirements
    if request.top_p is not None:
        _finite_number(request.top_p, name="top_p", minimum=0.0, maximum=1.0)
        if request.top_p <= 0.0:
            _reject("top_p must be greater than 0.0", "number_out_of_range", "top_p")
    _finite_number(
        request.frequency_penalty,
        name="frequency_penalty",
        minimum=-2.0,
        maximum=2.0,
    )
    _finite_number(
        request.presence_penalty,
        name="presence_penalty",
        minimum=-2.0,
        maximum=2.0,
    )
    if isinstance(request.max_tokens, bool) or not isinstance(request.max_tokens, int):
        _reject("max_tokens must be a strict integer", "invalid_integer", "max_tokens")
    if not 1 <= request.max_tokens <= 8192:
        _reject("max_tokens must be between 1 and 8192", "integer_out_of_range", "max_tokens")

    _validate_response_format(request.response_format)
    try:
        request.nram = normalize_nram_options(request.nram)
    except Exception as exc:
        validation_messages = " ".join(
            str(item.get("msg", "")) for item in getattr(exc, "errors", lambda: [])()
        )
        if "DEXPERTS_DISABLED" in str(exc) or "DEXPERTS_DISABLED" in validation_messages:
            _reject(
                "DExperts is disabled by NRAM_DEXPERTS_ENABLED",
                "DEXPERTS_DISABLED",
                "nram.dexperts",
            )
        raise HTTPException(status_code=400, detail=stable_validation_error(exc)) from exc
    return True
