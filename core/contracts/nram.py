"""NRAM state and steering contracts."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class NRAMState(BaseModel):
    """Deterministic NRAM persona state. No unrestricted random numbers."""

    model_config = ConfigDict(extra="forbid")

    visionary_intensity: float = Field(0.85, ge=0.0, le=1.0)
    contrarian_force: float = Field(0.75, ge=0.0, le=1.0)
    product_obsession: float = Field(0.90, ge=0.0, le=1.0)
    human_focus: float = Field(0.90, ge=0.0, le=1.0)
    rhetorical_compression: float = Field(0.75, ge=0.0, le=1.0)
    associative_distance: float = Field(0.65, ge=0.0, le=1.0)
    theatricality: float = Field(0.72, ge=0.0, le=1.0)
    emotional_voltage: float = Field(0.68, ge=0.0, le=1.0)

    coherence_floor: float = Field(0.78, ge=0.0, le=1.0)
    novelty_target: float = Field(0.70, ge=0.0, le=1.0)
    repetition_penalty: float = Field(0.45, ge=0.0, le=1.0)
    corporate_jargon_penalty: float = Field(0.85, ge=0.0, le=1.0)

    generated_tokens: int = Field(0, ge=0)
    rhetorical_phase: str = "opening"


class SteeringPolicy(BaseModel):
    """Compiled policy produced by the persona compiler."""

    positive_lexemes: List[str] = Field(default_factory=list)
    negative_lexemes: List[str] = Field(default_factory=list)
    forbidden_lexemes: List[str] = Field(default_factory=list)

    positive_bias: float = 0.0
    negative_bias: float = 0.0
    repetition_penalty: float = 0.0

    min_coherence: float = 0.75
    rhetorical_phase: str = "opening"
    max_tokens: int = 256

    developer_instruction: str = ""


class CompiledTokenPolicy(BaseModel):
    """Tokenizer-aware compiled policy with concrete token IDs."""

    positive_token_ids: List[int] = Field(default_factory=list)
    negative_token_ids: List[int] = Field(default_factory=list)
    forbidden_token_ids: List[int] = Field(default_factory=list)

    positive_bias: float = 0.0
    negative_bias: float = 0.0
    repetition_penalty: float = 0.0
