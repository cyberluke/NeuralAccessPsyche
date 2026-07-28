"""
Feature 2: NRAM Batch Completions — process multiple prompts in one request.

POST /v1/nram/batch takes a list of prompts and a shared profile, returns
all completions. Useful for evaluation harnesses and A/B sweeps.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BatchRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=20)
    profile: str = "normal"
    model: str = "nram-qwen3-14b-awq"
    max_tokens: int = Field(200, ge=20, le=2000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    intensity: float = Field(0.9, ge=0.0, le=1.0)


class BatchItem(BaseModel):
    index: int
    prompt: str
    output: str
    latency_ms: float
    model: str
    error: Optional[str] = None


class BatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    total_latency_ms: float
    items: List[BatchItem]
