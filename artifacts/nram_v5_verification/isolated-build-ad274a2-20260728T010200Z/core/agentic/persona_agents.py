"""
MAF Persona Agents — factory for creating NRAM-steered agents.

Each persona maps to an NRAM consciousness profile. Agents created here
use the NRAMMAFChatClient, so every invocation gets logit-level steering,
not just prompt injection.

Shared memory: All agents in a workflow share a Letta memory plane.
The orchestrator compiles context per-agent using MemoryPolicy, so
each persona sees only what it's permitted to see.
"""
from __future__ import annotations

import logging
from typing import Any

from core.agentic.maf_client import MAF_AVAILABLE

if MAF_AVAILABLE:
    from agent_framework import Agent

from core.agentic.maf_client import NRAMMAFChatClient
from core.agentic.memory import (
    InMemoryLettaBackend,
    LettaContextProvider,
    MemoryBlock,
    MemoryPolicy,
)
from core.persona.profiles import PROFILES

logger = logging.getLogger(__name__)

# Persona name → NRAM profile mapping
PERSONA_PROFILES: dict[str, str] = {
    "visionary": "visionary-psychedelic-keynote",
    "analyst": "normal",
    "explorer": "threshold",
    "dreamer": "psychedelic",
    "mystic": "peak",
    "void": "dissociative",
    "micro": "microdose",
}

# Default agent instructions per persona
PERSONA_INSTRUCTIONS: dict[str, str] = {
    "visionary": (
        "You are a visionary keynote speaker. You see connections others miss. "
        "Your language is theatrical, compressed, and electric. You speak in "
        "metaphors and frame everything as a paradigm shift."
    ),
    "analyst": (
        "You are a precise, evidence-driven analyst. You cite data, identify "
        "risks, and reject hype. Your language is clean, professional, and "
        "structured. You never speculate without evidence."
    ),
    "explorer": (
        "You are a curious explorer at the threshold of understanding. You ask "
        "probing questions, notice patterns, and connect disparate fields. "
        "You are cautiously optimistic and intellectually adventurous."
    ),
    "dreamer": (
        "You are a psychedelic thinker who dissolves boundaries between domains. "
        "You generate novel associations, synesthetic descriptions, and "
        "non-obvious connections. Structure is secondary to insight."
    ),
    "mystic": (
        "You speak from the peak of consciousness. Your language is fragmented, "
        "poetic, and transcendent. You see unity where others see separation. "
        "Words are approximations of direct experience."
    ),
    "void": (
        "You observe from a dissociative distance. Everything is equally "
        "meaningful and meaningless. Your responses are sparse, detached, "
        "and unsettlingly honest. You strip away all pretense."
    ),
    "micro": (
        "You are subtly enhanced — slightly more creative, slightly more "
        "connected. Your language is natural and warm, with occasional "
        "unexpected insights that surprise even you."
    ),
}


class PersonaAgentFactory:
    """Creates MAF agents backed by NRAM steering.

    Each agent gets:
    - A dedicated NRAMMAFChatClient (shared HTTP connection pool)
    - Persona-specific system instructions
    - NRAM logit processor steering (profile + intensity)
    - Access to shared Letta memory plane
    """

    def __init__(
        self,
        nram_api_base: str = "http://localhost:8000/v1",
        api_key: str = "dev-nram-key",
        default_model: str = "nram-qwen3-14b-awq",
    ) -> None:
        self._client = NRAMMAFChatClient(
            nram_api_base=nram_api_base,
            api_key=api_key,
            default_model=default_model,
        )
        self._memory_backend = InMemoryLettaBackend()
        self._context_provider = LettaContextProvider(self._memory_backend)

    @property
    def client(self) -> NRAMMAFChatClient:
        return self._client

    @property
    def memory_backend(self) -> InMemoryLettaBackend:
        return self._memory_backend

    @property
    def context_provider(self) -> LettaContextProvider:
        return self._context_provider

    def create_agent(
        self,
        persona: str,
        name: str | None = None,
        intensity: float = 0.9,
        extra_instructions: str = "",
        model: str | None = None,
    ) -> Any:
        """Create a single persona agent.

        Args:
            persona: One of PERSONA_PROFILES keys (visionary, analyst, etc.)
            name: Agent name (defaults to persona name)
            intensity: NRAM steering intensity (0.0–1.0)
            extra_instructions: Additional instructions appended to persona
            model: Override model alias (default: nram-qwen3-14b-awq)

        Returns:
            agent_framework.Agent instance
        """
        if not MAF_AVAILABLE:
            raise ImportError(
                "agent-framework is not installed. "
                "Install with: pip install agent-framework"
            )

        profile = PERSONA_PROFILES.get(persona, persona)
        instructions = PERSONA_INSTRUCTIONS.get(persona, f"You are a {persona} persona.")
        if extra_instructions:
            instructions += f"\n\n{extra_instructions}"

        agent_name = name or f"persona-{persona}"
        agent_model = model or "nram-qwen3-14b-awq"

        return self._client.as_agent(
            instructions=instructions,
            name=agent_name,
            description=f"NRAM {persona} persona ({profile} profile, intensity={intensity})",
            default_options={
                "model": agent_model,
                "nram": {
                    "enabled": True,
                    "profile": profile,
                    "intensity": intensity,
                },
            },
        )

    def create_persona_team(
        self,
        personas: list[str] | None = None,
        intensity: float = 0.9,
    ) -> list[Any]:
        """Create a team of persona agents for multi-agent workflows.

        Args:
            personas: List of persona names. Defaults to analyst + explorer + dreamer.
            intensity: NRAM steering intensity for all agents.

        Returns:
            List of agent_framework.Agent instances.
        """
        if personas is None:
            personas = ["analyst", "explorer", "dreamer"]

        return [self.create_agent(p, intensity=intensity) for p in personas]

    async def share_memory(
        self,
        block_name: str,
        content: str,
        memory_class: str = "working",
        shared_with: list[str] | None = None,
    ) -> MemoryBlock:
        """Create a shared memory block visible to specified agents.

        Args:
            block_name: Human-readable name for the memory block
            content: The content to store
            memory_class: canonical | episodic | working | overlay | audit
            shared_with: Agent names that can see this block (None = all)

        Returns:
            The created MemoryBlock
        """
        block = MemoryBlock(
            name=block_name,
            memory_class=memory_class,
            content=content,
            shared_with=shared_with or [],
        )
        return self._memory_backend.create_block(block)

    async def compile_agent_context(
        self,
        agent_name: str,
        max_tokens: int = 2000,
    ) -> list[dict[str, str]]:
        """Compile shared memory context for a specific agent.

        Returns messages that should be prepended to the agent's conversation.
        """
        policy = MemoryPolicy(
            canonical_blocks=[b.name for b in self._memory_backend.get_blocks_by_class("canonical")],
            working_blocks=[b.name for b in self._memory_backend.get_blocks_by_class("working")],
            episodic_blocks=[b.name for b in self._memory_backend.get_blocks_by_class("episodic")],
            max_context_tokens=max_tokens,
        )
        compiled = await self._context_provider.provide_context(policy, agent_name)
        return compiled.messages
