"""
Feature 4: Grammar-constrained decoding for SGLang.

Compiles response_format / Pydantic models into SGLang-compatible
structural constraints (json_schema, regex, ebnf).

SGLang supports:
  - response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}
  - response_format={"type": "json_object"}
  - regex="..."  (raw regex constraint)
  - structural_tag={...}

This module normalizes whatever the client sends into the correct SGLang shape.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def compile_response_format(
    response_format: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize a client-supplied response_format for SGLang.

    Accepts:
      {"type": "json_object"}                              -> pass through
      {"type": "json_schema", "json_schema": {...}}        -> pass through
      {"type": "json_schema", "schema": {...}}             -> normalize to json_schema key
      A bare Pydantic .model_json_schema() dict            -> wrap as json_schema
      None                                                 -> None (unconstrained)
    """
    if response_format is None:
        return None

    rtype = response_format.get("type")

    # Already well-formed
    if rtype == "json_object":
        return {"type": "json_object"}

    if rtype == "json_schema":
        js = response_format.get("json_schema")
        if js and "schema" in js:
            return response_format  # already {"name":..., "schema":...}
        # Client passed {"type": "json_schema", "schema": {...}}
        schema = response_format.get("schema")
        if schema:
            name = schema.get("title", "response")
            return {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        return response_format

    # Bare schema dict (no "type" key) — assume it's a JSON schema
    if "properties" in response_format or "type" not in response_format:
        name = response_format.get("title", "response")
        return {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": response_format, "strict": True},
        }

    return response_format


def pydantic_to_response_format(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model class to an SGLang json_schema response_format."""
    schema = model_cls.model_json_schema()
    name = schema.get("title", model_cls.__name__)
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }
