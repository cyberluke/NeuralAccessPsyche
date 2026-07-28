"""
Workflow Store — persists workflow state for resume after restart.

Backed by SQLite for durability. Each workflow's state, artifacts, and
events are stored separately. Large artifacts are referenced by ID.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agentic.contracts import (
    InnovationWorkflowState,
    Roadmap,
    SelectedDirection,
    WorkflowEvent,
    WorkflowStatus,
)


class WorkflowStore:
    """SQLite-backed workflow persistence."""

    def __init__(self, db_path: str = "data/workflows.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS workflow_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            );
            CREATE TABLE IF NOT EXISTS workflow_artifacts (
                artifact_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_workflow
                ON workflow_events(workflow_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_artifacts_workflow
                ON workflow_artifacts(workflow_id, artifact_type);
        """)
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Workflow CRUD
    # -----------------------------------------------------------------------

    def create_workflow(self, state: InnovationWorkflowState) -> str:
        self._conn.execute(
            "INSERT INTO workflows (workflow_id, state_json) VALUES (?, ?)",
            (state.workflow_id, state.model_dump_json()),
        )
        self._conn.commit()
        return state.workflow_id

    def get_workflow(self, workflow_id: str) -> Optional[InnovationWorkflowState]:
        row = self._conn.execute(
            "SELECT state_json FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        return InnovationWorkflowState.model_validate_json(row["state_json"])

    def update_workflow(self, state: InnovationWorkflowState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        self._conn.execute(
            "UPDATE workflows SET state_json = ?, updated_at = ? WHERE workflow_id = ?",
            (state.model_dump_json(), state.updated_at.isoformat(), state.workflow_id),
        )
        self._conn.commit()

    def list_workflows(self) -> List[InnovationWorkflowState]:
        rows = self._conn.execute(
            "SELECT state_json FROM workflows ORDER BY created_at DESC"
        ).fetchall()
        return [InnovationWorkflowState.model_validate_json(r["state_json"]) for r in rows]

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def add_event(self, event: WorkflowEvent) -> None:
        self._conn.execute(
            "INSERT INTO workflow_events (workflow_id, sequence, event_json) VALUES (?, ?, ?)",
            (event.workflow_id, event.sequence, event.model_dump_json()),
        )
        self._conn.commit()

    def get_events(self, workflow_id: str, after_sequence: int = 0) -> List[WorkflowEvent]:
        rows = self._conn.execute(
            "SELECT event_json FROM workflow_events WHERE workflow_id = ? AND sequence > ? ORDER BY sequence",
            (workflow_id, after_sequence),
        ).fetchall()
        return [WorkflowEvent.model_validate_json(r["event_json"]) for r in rows]

    def get_next_sequence(self, workflow_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(sequence) as max_seq FROM workflow_events WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        return (row["max_seq"] or 0) + 1

    # -----------------------------------------------------------------------
    # Artifacts
    # -----------------------------------------------------------------------

    def store_artifact(self, workflow_id: str, artifact_id: str, artifact_type: str, data: Any) -> None:
        if hasattr(data, "model_dump_json"):
            artifact_json = data.model_dump_json()
        else:
            artifact_json = json.dumps(data)
        self._conn.execute(
            "INSERT OR REPLACE INTO workflow_artifacts (artifact_id, workflow_id, artifact_type, artifact_json) VALUES (?, ?, ?, ?)",
            (artifact_id, workflow_id, artifact_type, artifact_json),
        )
        self._conn.commit()

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT artifact_json FROM workflow_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["artifact_json"])

    def get_artifacts_by_type(self, workflow_id: str, artifact_type: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT artifact_json FROM workflow_artifacts WHERE workflow_id = ? AND artifact_type = ? ORDER BY created_at",
            (workflow_id, artifact_type),
        ).fetchall()
        return [json.loads(r["artifact_json"]) for r in rows]

    def close(self) -> None:
        self._conn.close()


# Global workflow store instance
workflow_store = WorkflowStore()
