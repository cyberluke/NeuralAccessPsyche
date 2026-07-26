"""
Letta Memory Plane — context compilation for agents.

This module implements the memory half of NRAM. It maps the 5 memory classes
onto Letta-style memory blocks and compiles the permitted context for each
agent invocation.

Memory classes:
  1. Canonical evidence memory — read-only facts, code references, test results
  2. Episodic memory — previous interactions, experiments, agent experiences
  3. Working memory — current task state and unresolved hypotheses
  4. Psyche overlay — temporary distortion (overlap, forgetting, association)
  5. Audit ledger — records which memories were exposed/hidden/transformed

Letta manipulates CONTEXTUAL memory (information in the context window), not
hidden activations or logits. It is therefore distinct from NRAM decoder
steering. A generated false memory MUST be labeled synthetic_overlay and never
silently written into factual memory.

This is a self-contained reference implementation. For production, swap
InMemoryLettaBackend for a self-hosted Letta server client. The
LettaContextProvider interface stays the same.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryBlock(BaseModel):
    """A Letta-style memory block."""
    id: str = Field(default_factory=lambda: f"block-{uuid.uuid4().hex[:8]}")
    name: str
    memory_class: str  # canonical | episodic | working | overlay | audit
    content: str
    read_only: bool = False
    shared_with: List[str] = Field(default_factory=list)  # agent names
    synthetic_overlay: bool = False  # True = generated false memory
    expires_after_request: bool = False
    token_estimate: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryPolicy(BaseModel):
    """Controls which memories an agent may see for one invocation."""
    canonical_blocks: List[str] = Field(default_factory=list)
    working_blocks: List[str] = Field(default_factory=list)
    episodic_blocks: List[str] = Field(default_factory=list)
    overlay: Optional[Dict[str, Any]] = None
    allow_canonical_writes: bool = False
    max_context_tokens: int = 4000


class CompiledContext(BaseModel):
    """The compiled context (messages) for one agent invocation."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    block_ids_used: List[str] = Field(default_factory=list)
    total_tokens: int = 0
    overlay_applied: bool = False
    synthetic_memories: List[str] = Field(default_factory=list)


def _block_visible_to(block: MemoryBlock, agent_name: Optional[str]) -> bool:
    """Check whether a block is visible to a given agent.

    Visibility rules:
    - shared_with empty  -> visible to everyone (public block)
    - shared_with set    -> visible only to listed agents
    - agent_name None    -> sees all blocks (system/orchestrator view)
    """
    if agent_name is None:
        return True
    if not block.shared_with:
        return True
    return agent_name in block.shared_with


class InMemoryLettaBackend:
    """Reference in-memory Letta backend. Swap for a real Letta client in prod."""

    def __init__(self) -> None:
        self._blocks: Dict[str, MemoryBlock] = {}

    def create_block(self, block: MemoryBlock) -> MemoryBlock:
        if block.token_estimate == 0:
            block.token_estimate = len(block.content) // 4
        self._blocks[block.id] = block
        return block

    def get_block(self, block_id: str) -> Optional[MemoryBlock]:
        return self._blocks.get(block_id)

    def get_blocks_by_name(self, name: str) -> List[MemoryBlock]:
        return [b for b in self._blocks.values() if b.name == name]

    def get_blocks_by_class(self, memory_class: str) -> List[MemoryBlock]:
        return [b for b in self._blocks.values() if b.memory_class == memory_class]

    def update_block(self, block_id: str, content: str) -> Optional[MemoryBlock]:
        block = self._blocks.get(block_id)
        if block is None:
            return None
        if block.read_only:
            raise ValueError(f"Block {block_id} is read-only (canonical memory)")
        block.content = content
        block.token_estimate = len(content) // 4
        return block

    def delete_block(self, block_id: str) -> bool:
        return self._blocks.pop(block_id, None) is not None

    def list_all(self) -> List[MemoryBlock]:
        return list(self._blocks.values())


class SqliteLettaBackend(InMemoryLettaBackend):
    """SQLite-backed Letta backend. Memory survives container restarts.

    Extends InMemoryLettaBackend with write-through persistence. On init,
    loads all persisted blocks into memory; every mutation is flushed to disk.
    """

    def __init__(self, db_path: str = "/data/letta_memory.db") -> None:
        super().__init__()
        import sqlite3
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_blocks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                memory_class TEXT NOT NULL,
                content TEXT NOT NULL,
                read_only INTEGER NOT NULL DEFAULT 0,
                shared_with TEXT NOT NULL DEFAULT '[]',
                synthetic_overlay INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()
        self._load_all()

    def _load_all(self) -> None:
        import json
        from datetime import datetime, timezone
        rows = self._conn.execute("SELECT * FROM memory_blocks").fetchall()
        cols = [d[0] for d in self._conn.execute("SELECT * FROM memory_blocks LIMIT 0").description]
        for row in rows:
            d = dict(zip(cols, row))
            block = MemoryBlock(
                id=d["id"],
                name=d["name"],
                memory_class=d["memory_class"],
                content=d["content"],
                read_only=bool(d["read_only"]),
                shared_with=json.loads(d["shared_with"]),
                synthetic_overlay=bool(d["synthetic_overlay"]),
                token_estimate=d["token_estimate"],
                created_at=datetime.fromisoformat(d["created_at"]),
            )
            self._blocks[block.id] = block

    def _flush(self, block: MemoryBlock) -> None:
        import json
        self._conn.execute("""
            INSERT INTO memory_blocks (id, name, memory_class, content, read_only,
                                       shared_with, synthetic_overlay, token_estimate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content, token_estimate=excluded.token_estimate,
                shared_with=excluded.shared_with
        """, (
            block.id, block.name, block.memory_class, block.content,
            int(block.read_only), json.dumps(block.shared_with),
            int(block.synthetic_overlay), block.token_estimate,
            block.created_at.isoformat(),
        ))
        self._conn.commit()

    def create_block(self, block: MemoryBlock) -> MemoryBlock:
        result = super().create_block(block)
        self._flush(result)
        return result

    def update_block(self, block_id: str, content: str) -> Optional[MemoryBlock]:
        result = super().update_block(block_id, content)
        if result:
            self._flush(result)
        return result

    def delete_block(self, block_id: str) -> bool:
        result = super().delete_block(block_id)
        if result:
            self._conn.execute("DELETE FROM memory_blocks WHERE id = ?", (block_id,))
            self._conn.commit()
        return result


class LettaContextProvider:
    """Compiles permitted Letta blocks into agent context.

    This is the ContextProvider for the agentic framework. It controls WHAT
    each agent remembers for a given invocation, enforcing the memory policy.
    """

    def __init__(self, backend: Optional[InMemoryLettaBackend] = None) -> None:
        self._backend = backend or InMemoryLettaBackend()

    @property
    def backend(self) -> InMemoryLettaBackend:
        return self._backend

    async def provide_context(
        self,
        policy: MemoryPolicy,
        agent_name: Optional[str] = None,
    ) -> CompiledContext:
        """Compile the context messages for one agent invocation.

        Enforces:
        - Read-only canonical blocks cannot be mutated.
        - Token budget (max_context_tokens).
        - Psyche overlay is applied as a temporary layer and flagged.
        - Synthetic (false) memories are labeled and never canonical.
        """
        messages: List[Dict[str, str]] = []
        block_ids_used: List[str] = []
        total_tokens = 0
        synthetic: List[str] = []

        def _add_block(block: MemoryBlock, role: str = "user") -> bool:
            nonlocal total_tokens
            if total_tokens + block.token_estimate > policy.max_context_tokens:
                return False  # budget exceeded
            prefix = ""
            if block.synthetic_overlay:
                prefix = "[SYNTHETIC_OVERLAY — not factual, expires after request]\n"
                synthetic.append(block.id)
            label = f"[{block.memory_class.upper()} MEMORY: {block.name}]\n"
            messages.append({"role": role, "content": label + prefix + block.content})
            block_ids_used.append(block.id)
            total_tokens += block.token_estimate
            return True

        # Visibility filter: enforce shared_with per-agent access control.
        def _visible(block: MemoryBlock) -> bool:
            return _block_visible_to(block, agent_name)

        # 1. Canonical evidence (read-only facts)
        for name in policy.canonical_blocks:
            for block in self._backend.get_blocks_by_name(name):
                if block.memory_class == "canonical" and _visible(block):
                    _add_block(block)

        # 2. Working memory (current task state)
        for name in policy.working_blocks:
            for block in self._backend.get_blocks_by_name(name):
                if block.memory_class == "working" and _visible(block):
                    _add_block(block)

        # 3. Episodic memory (past experiences)
        for name in policy.episodic_blocks:
            for block in self._backend.get_blocks_by_name(name):
                if block.memory_class == "episodic" and _visible(block):
                    _add_block(block)

        # 4. Psyche overlay (temporary distortion)
        overlay_applied = False
        if policy.overlay:
            overlay_applied = True
            # Overlay is expressed as a temporary block describing the distortion
            overlay_content = (
                f"PSYCHE OVERLAY (temporary, for this invocation only):\n"
                f"profile={policy.overlay.get('profile', 'normal')}\n"
                f"overlap={policy.overlay.get('overlap', 0)}\n"
                f"forgetting={policy.overlay.get('forgetting', 0)}\n"
                f"associative_distance={policy.overlay.get('associative_distance', 0)}\n"
                f"persona_blending={policy.overlay.get('persona_blending', 0)}"
            )
            overlay_block = MemoryBlock(
                name="psyche_overlay",
                memory_class="overlay",
                content=overlay_content,
                synthetic_overlay=True,
                expires_after_request=True,
            )
            overlay_block.token_estimate = len(overlay_content) // 4
            _add_block(overlay_block)

        return CompiledContext(
            messages=messages,
            block_ids_used=block_ids_used,
            total_tokens=total_tokens,
            overlay_applied=overlay_applied,
            synthetic_memories=synthetic,
        )

    def record_audit(
        self,
        agent_name: str,
        block_ids_exposed: List[str],
        overlay_applied: bool,
        synthetic_memories: List[str],
    ) -> MemoryBlock:
        """Append an audit-ledger block recording what was exposed/transformed."""
        content = (
            f"AUDIT: agent={agent_name} exposed_blocks={block_ids_exposed} "
            f"overlay={overlay_applied} synthetic={synthetic_memories} "
            f"at={datetime.now(timezone.utc).isoformat()}"
        )
        block = MemoryBlock(
            name="audit_ledger",
            memory_class="audit",
            content=content,
            read_only=True,
        )
        return self._backend.create_block(block)
