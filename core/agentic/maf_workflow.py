"""
MAF Workflow Orchestrator — multi-agent workflows with shared memory.

Implements two orchestration patterns using Microsoft Agent Framework:
1. Concurrent: All personas answer in parallel, synthesizer merges.
2. Sequential: Personas build on each other's output in a chain.

Both patterns share a Letta memory plane so agents can read/write
shared context blocks during the workflow.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.agentic.maf_client import MAF_AVAILABLE
from core.agentic.persona_agents import PersonaAgentFactory

if MAF_AVAILABLE:
    from agent_framework import Agent

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of a multi-agent workflow execution."""
    workflow_id: str
    pattern: str  # "concurrent" | "sequential"
    final_output: str
    agent_outputs: dict[str, str] = field(default_factory=dict)
    latencies_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    memory_blocks_used: list[str] = field(default_factory=list)


class MAFWorkflowOrchestrator:
    """Orchestrates multi-agent workflows using MAF agents with NRAM steering.

    Usage:
        factory = PersonaAgentFactory()
        orch = MAFWorkflowOrchestrator(factory)

        # Concurrent: all personas answer in parallel
        result = await orch.run_concurrent(
            question="What is consciousness?",
            personas=["analyst", "dreamer", "mystic"],
        )

        # Sequential: chain of personas building on each other
        result = await orch.run_sequential(
            question="Design a new product",
            personas=["explorer", "analyst", "visionary"],
        )
    """

    def __init__(self, factory: PersonaAgentFactory) -> None:
        self._factory = factory

    async def run_concurrent(
        self,
        question: str,
        personas: list[str] | None = None,
        max_tokens: int = 200,
        synthesize: bool = True,
    ) -> WorkflowResult:
        """Run personas in parallel, optionally synthesize.

        Each persona gets the question + shared memory context.
        If synthesize=True, a final analyst agent merges all outputs.
        """
        if personas is None:
            personas = ["analyst", "explorer", "dreamer"]

        workflow_id = f"wf-concurrent-{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()

        # Create agents
        agents: dict[str, Any] = {}
        for p in personas:
            agents[p] = self._factory.create_agent(
                p, name=f"{workflow_id}-{p}", intensity=0.9
            )

        # Share the question as working memory
        await self._factory.share_memory(
            block_name=f"question-{workflow_id}",
            content=question,
            memory_class="working",
            shared_with=list(agents.keys()),
        )

        # Run all agents concurrently
        async def _run_one(persona: str, agent: Any) -> tuple[str, str, float]:
            t_start = time.perf_counter()
            try:
                # Compile shared memory context for this agent
                context_msgs = await self._factory.compile_agent_context(
                    f"{workflow_id}-{persona}", max_tokens=1000
                )
                # Build full prompt with memory context
                full_prompt = question
                if context_msgs:
                    mem_text = "\n".join(m["content"] for m in context_msgs)
                    full_prompt = f"Shared context:\n{mem_text}\n\nQuestion: {question}"

                response = await agent.run(full_prompt)
                text = response.text if hasattr(response, "text") else str(response)
                # Truncate for synthesis
                return persona, text[:300], (time.perf_counter() - t_start) * 1000
            except Exception as e:
                logger.error(f"Agent {persona} failed: {e}")
                return persona, f"[error: {e}]", (time.perf_counter() - t_start) * 1000

        tasks = [_run_one(p, a) for p, a in agents.items()]
        results = await asyncio.gather(*tasks)

        agent_outputs = {p: out for p, out, _ in results}
        latencies = {p: lat for p, _, lat in results}

        # Synthesize
        final_output = ""
        if synthesize and len(agent_outputs) > 1:
            synth_agent = self._factory.create_agent(
                "analyst",
                name=f"{workflow_id}-synthesizer",
                intensity=0.3,  # Low intensity for clean synthesis
                extra_instructions="You synthesize multiple expert perspectives into one concise answer.",
            )
            summaries = "\n".join(
                f"[{p}]: {out[:150]}" for p, out in agent_outputs.items()
            )
            synth_prompt = (
                f"Question: {question}\n\n"
                f"Expert perspectives:\n{summaries}\n\n"
                "Write ONE concise paragraph (max 3 sentences) synthesizing the best insight."
            )
            t_synth = time.perf_counter()
            try:
                synth_resp = await synth_agent.run(synth_prompt)
                final_output = synth_resp.text if hasattr(synth_resp, "text") else str(synth_resp)
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                final_output = f"[synthesis error: {e}]"
            latencies["synthesizer"] = (time.perf_counter() - t_synth) * 1000
        else:
            # No synthesis — just concatenate
            final_output = "\n\n".join(
                f"**{p}**: {out}" for p, out in agent_outputs.items()
            )

        total_ms = (time.perf_counter() - t0) * 1000
        return WorkflowResult(
            workflow_id=workflow_id,
            pattern="concurrent",
            final_output=final_output,
            agent_outputs=agent_outputs,
            latencies_ms=latencies,
            total_latency_ms=total_ms,
            memory_blocks_used=[f"question-{workflow_id}"],
        )

    async def run_sequential(
        self,
        question: str,
        personas: list[str] | None = None,
        max_tokens: int = 200,
    ) -> WorkflowResult:
        """Run personas in sequence, each building on the previous output.

        Pattern: persona[0] answers → persona[1] critiques/builds → ... → final
        """
        if personas is None:
            personas = ["explorer", "analyst", "visionary"]

        workflow_id = f"wf-sequential-{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()

        agent_outputs: dict[str, str] = {}
        latencies: dict[str, float] = {}
        running_context = question

        for i, persona in enumerate(personas):
            agent = self._factory.create_agent(
                persona,
                name=f"{workflow_id}-{persona}",
                intensity=0.8,
            )

            # Share previous outputs as episodic memory
            if i > 0:
                await self._factory.share_memory(
                    block_name=f"chain-{workflow_id}-{i}",
                    content=running_context[-500:],  # Last 500 chars
                    memory_class="episodic",
                    shared_with=[f"{workflow_id}-{persona}"],
                )

            prompt = running_context if i == 0 else (
                f"Previous analysis:\n{running_context[-400:]}\n\n"
                f"Original question: {question}\n\n"
                f"Build on the previous analysis from your {persona} perspective."
            )

            t_start = time.perf_counter()
            try:
                response = await agent.run(prompt)
                text = response.text if hasattr(response, "text") else str(response)
                agent_outputs[persona] = text[:300]
                running_context = text
            except Exception as e:
                logger.error(f"Sequential agent {persona} failed: {e}")
                agent_outputs[persona] = f"[error: {e}]"
            latencies[persona] = (time.perf_counter() - t_start) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        return WorkflowResult(
            workflow_id=workflow_id,
            pattern="sequential",
            final_output=running_context,
            agent_outputs=agent_outputs,
            latencies_ms=latencies,
            total_latency_ms=total_ms,
        )
