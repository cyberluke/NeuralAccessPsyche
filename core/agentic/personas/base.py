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
        """Build the NRAM options for this persona's invocation."""
        return {
            "enabled": True,
            "profile": self._nram_config.get("profile", "normal"),
            "intensity": self._nram_config.get("intensity", 0.5),
            "auto_temperature": False,
            "seed": self._workflow_seed,
            "include_telemetry": True,
        }

    def validate_output(self, raw: str, schema: Type[T]) -> T:
        """Validate and parse persona output against the strict schema."""
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise PersonaError(f"Output validation failed: {e}") from e

    async def run(self, **kwargs: Any) -> BaseModel:
        """Execute this persona. Override in subclasses."""
        raise NotImplementedError
