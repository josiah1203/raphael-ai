"""Persist planner IR for replay and eval."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raphael_ai.intelligence.planner import PlannerIR


class PlanStore:
    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or Path(os.environ.get("RAPHAEL_AI_DB", "/tmp/raphael-ai-memory.db"))
        self._db = path
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS intelligence_plans (
                    plan_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT,
                    ir JSON NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        plan_id: str,
        question: str,
        workspace_id: str,
        ir: PlannerIR,
        project_id: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO intelligence_plans
                (plan_id, question, workspace_id, project_id, ir, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (plan_id, question, workspace_id, project_id or "", ir.model_dump_json(by_alias=True), now),
            )

    def get(self, plan_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT question, workspace_id, project_id, ir, created_at FROM intelligence_plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "plan_id": plan_id,
            "question": row[0],
            "workspace_id": row[1],
            "project_id": row[2] or None,
            "plan": json.loads(row[3]),
            "created_at": row[4],
        }
