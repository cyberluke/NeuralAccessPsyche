"""
Evidence Ledger — append-only evidence store.

Rules:
- Persona output is NEVER evidence.
- README content is documentation evidence, not implementation proof.
- Names of files or functions do not prove behavior.
- Runtime capability claims require a test or observed command result.
- Unknown claims become explicit assumptions.
- Evidence records are append-only.
- Personas may reference evidence but may not rewrite it.
- Final roadmap decisions must trace to evidence and reviewed hypotheses.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agentic.contracts import EvidenceItem, EvidenceSourceType


class EvidenceLedger:
    """Append-only evidence store backed by SQLite."""

    def __init__(self, db_path: str = "data/evidence.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                source_type TEXT NOT NULL,
                repository_revision TEXT NOT NULL,
                path TEXT,
                symbol TEXT,
                line_start INTEGER,
                line_end INTEGER,
                command TEXT,
                observed_value TEXT,
                confidence REAL DEFAULT 0.5,
                verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    def add(self, evidence: EvidenceItem) -> str:
        """Append an evidence record. Returns the evidence ID."""
        self._conn.execute(
            """INSERT INTO evidence
               (id, claim, source_type, repository_revision, path, symbol,
                line_start, line_end, command, observed_value, confidence, verified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence.id,
                evidence.claim,
                evidence.source_type.value,
                evidence.repository_revision,
                evidence.path,
                evidence.symbol,
                evidence.line_start,
                evidence.line_end,
                evidence.command,
                evidence.observed_value,
                evidence.confidence,
                int(evidence.verified),
            ),
        )
        self._conn.commit()
        return evidence.id

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        row = self._conn.execute(
            "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def get_many(self, evidence_ids: List[str]) -> List[EvidenceItem]:
        if not evidence_ids:
            return []
        placeholders = ",".join("?" * len(evidence_ids))
        rows = self._conn.execute(
            f"SELECT * FROM evidence WHERE id IN ({placeholders})",
            evidence_ids,
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def list_all(self) -> List[EvidenceItem]:
        rows = self._conn.execute(
            "SELECT * FROM evidence ORDER BY created_at"
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def _row_to_item(self, row: sqlite3.Row) -> EvidenceItem:
        return EvidenceItem(
            id=row["id"],
            claim=row["claim"],
            source_type=EvidenceSourceType(row["source_type"]),
            repository_revision=row["repository_revision"],
            path=row["path"],
            symbol=row["symbol"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            command=row["command"],
            observed_value=row["observed_value"],
            confidence=row["confidence"],
            verified=bool(row["verified"]),
        )

    def close(self) -> None:
        self._conn.close()
