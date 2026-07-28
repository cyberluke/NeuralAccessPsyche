"""
Feature 5: Structured persona output schemas.

Typed Pydantic artifacts each persona emits, so the orchestrator receives
structured data (not free text) and can validate, diff, and route it.

Each persona has a schema + an SGLang json_schema response_format derived
from it (Feature 4). Persona prompts instruct the model to emit the schema.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from core.steering.grammar import pydantic_to_response_format


class AnalystOutput(BaseModel):
    """Precise, evidence-driven analyst artifact."""
    summary: str = Field(description="One-paragraph evidence-based assessment")
    key_findings: List[str] = Field(default_factory=list, description="Bullet findings")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence 0-1")


class ExplorerOutput(BaseModel):
    """Curious explorer artifact — patterns and questions."""
    observation: str = Field(description="What patterns were noticed")
    connections: List[str] = Field(default_factory=list, description="Cross-domain links")
    open_questions: List[str] = Field(default_factory=list, description="Questions to probe")


class DreamerOutput(BaseModel):
    """Psychedelic synthesizer artifact — novel associations."""
    vision: str = Field(description="The synthesized vision")
    associations: List[str] = Field(default_factory=list, description="Non-obvious connections")
    metaphor: Optional[str] = Field(None, description="Central metaphor")


class VisionaryOutput(BaseModel):
    """Visionary keynote artifact — paradigm framing."""
    thesis: str = Field(description="The bold thesis statement")
    paradigm_shift: str = Field(description="What is being reframed")
    call_to_action: str = Field(description="The rallying close")


class MysticOutput(BaseModel):
    """Peak consciousness artifact — fragmented, transcendent."""
    essence: str = Field(description="The core insight, compressed")
    fragments: List[str] = Field(default_factory=list, description="Poetic fragments")


class VoidOutput(BaseModel):
    """Dissociative artifact — detached, minimal."""
    observation: str = Field(description="Sparse, detached observation")
    stripped_pretense: List[str] = Field(default_factory=list, description="What was removed")


# Persona name -> output schema
PERSONA_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "analyst": AnalystOutput,
    "explorer": ExplorerOutput,
    "dreamer": DreamerOutput,
    "visionary": VisionaryOutput,
    "mystic": MysticOutput,
    "void": VoidOutput,
}


def persona_response_format(persona: str):
    """Return an SGLang json_schema response_format for a persona, or None."""
    schema_cls = PERSONA_SCHEMAS.get(persona)
    if schema_cls is None:
        return None
    return pydantic_to_response_format(schema_cls)


def persona_output_instruction(persona: str) -> str:
    """Instruction fragment telling the persona to emit its JSON schema."""
    schema_cls = PERSONA_SCHEMAS.get(persona)
    if schema_cls is None:
        return ""
    fields = ", ".join(schema_cls.model_fields.keys())
    return (
        f"\n\nRespond ONLY with a JSON object with these fields: {fields}. "
        f"No prose outside the JSON."
    )
