"""
Feature 1: NRAM Session Persistence — save/load sessions to SQLite.

Sessions survive container restarts. Each session's full state (profile,
intensity, phenomenon weights, memory config, event history) is persisted.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("/data/nram_sessions.db")


class SessionPersistence:
    """SQLite-backed session persistence layer."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL DEFAULT 'normal',
                intensity REAL NOT NULL DEFAULT 0.5,
                phenomenon_weights TEXT NOT NULL DEFAULT '{}',
                memory_config TEXT NOT NULL DEFAULT '{}',
                state_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def save(self, session_id: str, data: Dict[str, Any]) -> None:
        """Persist a session's state."""
        now = time.time()
        self._conn.execute("""
            INSERT INTO sessions (session_id, profile, intensity, phenomenon_weights,
                                  memory_config, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                profile=excluded.profile,
                intensity=excluded.intensity,
                phenomenon_weights=excluded.phenomenon_weights,
                memory_config=excluded.memory_config,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
        """, (
            session_id,
            data.get("profile", "normal"),
            data.get("intensity", 0.5),
            json.dumps(data.get("phenomenon_weights", {})),
            json.dumps(data.get("memory_config", {})),
            json.dumps(data.get("state", {})),
            data.get("created_at", now),
            now,
        ))
        self._conn.commit()

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session's state from disk."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.execute("SELECT * FROM sessions LIMIT 0").description]
        data = dict(zip(cols, row))
        data["phenomenon_weights"] = json.loads(data["phenomenon_weights"])
        data["memory_config"] = json.loads(data["memory_config"])
        data["state"] = json.loads(data["state_json"])
        return data

    def list_sessions(self, limit: int = 50) -> list[Dict[str, Any]]:
        """List recent sessions."""
        rows = self._conn.execute(
            "SELECT session_id, profile, intensity, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [{"session_id": r[0], "profile": r[1], "intensity": r[2], "updated_at": r[3]} for r in rows]

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        cur = self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


# Singleton
persistence = SessionPersistence()
