"""
Agentic persona base class.

Each persona:
- Uses a dedicated NRAM steering profile
- Has explicitly allowed inputs
- Has a strict Pydantic output schema
- Has validation gates
- Stores artifacts
- Uses deterministic seeds
- Has a bounded number of retries
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from core.agentic.contracts import PERSONA_NRAM_PROFILES, EvidenceItem
from core.agentic.evidence_ledger import EvidenceLedger

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PersonaError(Exception):
    """Raised when a persona fails after max retries."""
    pass


class BasePersona:
    """Base class for all agentic personas."""

    name: str = "base"
    output_schema: Type[BaseModel] = BaseModel
    max_retries: int = 2

    def __init__(
        self,
        nram_api_base: str = "http://localhost:8000/v1",
        evidence_ledger: Optional[EvidenceLedger] = None,
        workflow_seed: int = 271,
    ) -> None:
        self._nram_api_base = nram_api_base
        self._evidence = evidence_ledger
        self._workflow_seed = workflow_seed
        self._nram_config = PERSONA_NRAM_PROFILES.get(self.name, {})

    @property
    def nram_profile(self) -> Dict[str, Any]:
        return self._nram_config

    def build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        evidence_items: Optional[List[EvidenceItem]] = None,
    ) -> List[Dict[str, str]]:
        """Build the message list for this persona's invocation."""
        messages = [{"role": "system", "content": system_prompt}]

        # Inject evidence context
        if evidence_items:
            evidence_text = "\n".join(
                f"[{e.source_type.value}] {e.claim}"
                + (f" (path: {e.path})" if e.path else "")
                + (f" (verified: {e.verified})" if e.verified else "")
                for e in evidence_items
            )
            messages.append({
                "role": "user",
                "content": f"VERIFIED EVIDENCE (do not contradict without new evidence):\n{evidence_text}",
            })

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def build_nram_options(self) -> Dict[str, Any]:
        """Build the NRAM options for this persona's invocation.

        Includes temperature and per-dimension overrides from PERSONA_NRAM_PROFILES
        so that each persona runs with its declared cognitive parameters.
        """
        opts: Dict[str, Any] = {
            "enabled": True,
            "profile": self._nram_config.get("profile", "normal"),
            "intensity": self._nram_config.get("intensity", 0.5),
            "seed": self._workflow_seed,
            "include_telemetry": True,
        }
        # Forward per-dimension overrides (associative_distance, contrarian_force, etc.)
        dimension_fields = {
            "visionary_intensity", "contrarian_force", "product_obsession",
            "human_focus", "rhetorical_compression", "associative_distance",
            "theatricality", "emotional_voltage", "coherence_floor",
            "novelty_target", "repetition_penalty", "corporate_jargon_penalty",
        }
        for field in dimension_fields:
            if field in self._nram_config:
                opts[field] = self._nram_config[field]
        return opts

    @property
    def persona_temperature(self) -> float:
        """Return the declared temperature for this persona (default 0.7)."""
        return self._nram_config.get("temperature", 0.7)

    def validate_output(self, raw: str, schema: Type[T]) -> T:
        """Validate and parse persona output against the strict schema."""
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise PersonaError(f"Output validation failed: {e}") from e

    def validate_list_output(self, raw: str, schema: Type[T]) -> List[T]:
        """Validate and parse a JSON array of schema objects from model output."""
        try:
            json_str = self.extract_json(raw)
            data = json.loads(json_str)
            if isinstance(data, dict) and "items" in data:
                data = data["items"]
            if not isinstance(data, list):
                raise PersonaError(f"Expected a JSON array, got {type(data).__name__}")
            return [schema.model_validate(item) for item in data]
        except (json.JSONDecodeError, ValidationError) as e:
            raise PersonaError(f"List output validation failed: {e}") from e

    @staticmethod
    def extract_json(raw: str) -> str:
        """Extract a JSON object/array from raw model output.

        Strips markdown code fences and surrounding prose, then returns the
        first balanced {...} or [...] block. Robust to reasoning preambles
        (already stripped by the NRAM API) and chatty wrappers.
        """
        text = raw.strip()
        # Strip ```json ... ``` (or ```) fences if present.
        if "```" in text:
            parts = text.split("```")
            candidates = []
            for i in range(1, len(parts), 2):
                chunk = parts[i]
                if chunk.startswith("json"):
                    chunk = chunk[4:]
                candidates.append(chunk.strip())
            if candidates:
                text = max(candidates, key=len)
        # Locate the first balanced JSON object or array.
        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [s for s in (start_obj, start_arr) if s != -1]
        if not starts:
            return text
        start = min(starts)
        open_char = text[start]
        close_char = "}" if open_char == "{" else "]"
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return text[start:]

    async def invoke_model(
        self,
        messages: List[Dict[str, str]],
        grammar: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        """Invoke the NRAM API with this persona's steering profile.

        Returns the raw model output string, or None if the call fails
        (callers fall back to structured stub data in that case). Retries
        up to max_retries times. Never exposes hidden chain-of-thought —
        the NRAM API strips reasoning blocks.
        """
        from core.agentic.client import NRAMChatClient

        chat_options: Dict[str, Any] = {
            "nram": self.build_nram_options(),
            "max_tokens": max_tokens,
            "temperature": self.persona_temperature,
            "seed": self._workflow_seed,
            "include_telemetry": True,
        }
        if grammar is not None:
            chat_options["response_format"] = grammar

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            client = NRAMChatClient(
                nram_api_base=self._nram_api_base,
                default_model="nram-qwen3-14b-awq",
            )
            try:
                response = await client.get_response(messages, chat_options)
                return response.content
            except Exception as e:  # noqa: BLE001 - fall back to stub on any failure
                last_error = e
                logger.warning(
                    f"[{self.name}] model invocation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
            finally:
                await client.close()

        logger.error(f"[{self.name}] all {self.max_retries + 1} attempts failed: {last_error}")
        return None

    async def run(self, **kwargs: Any) -> BaseModel:
        """Execute this persona. Override in subclasses."""
        raise NotImplementedError
