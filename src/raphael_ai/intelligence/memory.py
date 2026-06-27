"""Workspace memory — auto-regenerated summaries."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from raphael_contracts import db as rdb


def _audit_url() -> str:
    return os.environ.get("RAPHAEL_AUDIT_URL", "http://127.0.0.1:8093").rstrip("/")


def fetch_audit_cursor(project_id: str | None = None) -> str | None:
    """Return latest audit next_cursor for memory watermark."""
    try:
        params: dict[str, Any] = {"limit": 1}
        if project_id:
            params["project_id"] = project_id
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{_audit_url()}/v1/audit/timeline", params=params)
            if res.status_code != 200:
                return None
            body = res.json()
            events = body.get("events") or []
            if events:
                return events[0].get("timestamp_utc") or body.get("next_cursor")
            return body.get("next_cursor")
    except httpx.HTTPError:
        return None


def _row_to_blob(workspace_id: str, project_id: str | None, row: Any) -> dict[str, Any]:
    risks = row["risks"]
    recent = row["recent_accomplishments"]
    convs = row["conventions_observed"]
    if isinstance(risks, str):
        risks = json.loads(risks or "[]")
    if isinstance(recent, str):
        recent = json.loads(recent or "[]")
    if isinstance(convs, str):
        convs = json.loads(convs or "[]")
    return {
        "workspace_id": workspace_id,
        "project_id": project_id,
        "goal": row["goal"],
        "milestone": row["milestone"],
        "risks": risks,
        "recent_accomplishments": recent,
        "conventions_observed": convs,
        "generated_at": row["generated_at"],
        "source_event_cursor": row["source_event_cursor"],
    }


class WorkspaceMemoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._postgres = rdb.is_postgres()
        if self._postgres:
            rdb.ensure_migrations()
            self._db: Path | None = None
        else:
            path = db_path or Path(os.environ.get("RAPHAEL_AI_DB", "/tmp/raphael-ai-memory.db"))
            self._db = path
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        assert self._db is not None
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_memory (
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL DEFAULT '',
                    goal TEXT,
                    milestone TEXT,
                    risks JSON,
                    recent_accomplishments JSON,
                    conventions_observed JSON,
                    generated_at TEXT,
                    source_event_cursor TEXT,
                    PRIMARY KEY (workspace_id, project_id)
                )
                """
            )

    def get(self, workspace_id: str, project_id: str | None = None) -> dict[str, Any]:
        pid = project_id or ""
        row = self._fetch_row(workspace_id, pid)
        if not row:
            return self.regenerate(workspace_id, project_id)
        return _row_to_blob(workspace_id, project_id, row)

    def _fetch_row(self, workspace_id: str, pid: str) -> Any | None:
        if self._postgres:
            return rdb.pg_fetchone(
                """
                SELECT goal, milestone, risks, recent_accomplishments, conventions_observed,
                       generated_at, source_event_cursor
                FROM workspace_memory
                WHERE workspace_id = %s AND project_id = %s
                """,
                (workspace_id, pid),
            )
        assert self._db is not None
        with sqlite3.connect(self._db) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                """
                SELECT goal, milestone, risks, recent_accomplishments, conventions_observed,
                       generated_at, source_event_cursor
                FROM workspace_memory
                WHERE workspace_id = ? AND project_id = ?
                """,
                (workspace_id, pid),
            ).fetchone()

    def regenerate(
        self,
        workspace_id: str,
        project_id: str | None = None,
        model_blob: dict[str, Any] | None = None,
        *,
        event_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Deterministic baseline; Gemma enriches when model_blob provided."""
        pid = project_id or ""
        now = datetime.now(UTC).isoformat()
        cursor = event_cursor if event_cursor is not None else fetch_audit_cursor(project_id)
        blob = {
            "workspace_id": workspace_id,
            "project_id": project_id,
            "goal": f"Active development in {project_id or workspace_id}",
            "milestone": "In progress",
            "risks": ["thermals", "EMI"],
            "recent_accomplishments": ["VCS foundation", "Module lineage"],
            "conventions_observed": ["PCB changes trigger DRC", "Prototype releases need QA"],
            "generated_at": now,
            "source_event_cursor": cursor,
        }
        if model_blob:
            for key in ("goal", "milestone", "risks", "recent_accomplishments", "conventions_observed"):
                if key in model_blob and model_blob[key]:
                    blob[key] = model_blob[key]
        self._upsert(workspace_id, pid, blob, now)
        return blob

    def _upsert(self, workspace_id: str, pid: str, blob: dict[str, Any], now: str) -> None:
        if self._postgres:
            sql = rdb.adapt_insert_or_replace(
                """
                INSERT OR REPLACE INTO workspace_memory
                (workspace_id, project_id, goal, milestone, risks, recent_accomplishments,
                 conventions_observed, generated_at, source_event_cursor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                "workspace_id, project_id",
                "goal = EXCLUDED.goal, milestone = EXCLUDED.milestone, "
                "risks = EXCLUDED.risks, recent_accomplishments = EXCLUDED.recent_accomplishments, "
                "conventions_observed = EXCLUDED.conventions_observed, generated_at = EXCLUDED.generated_at, "
                "source_event_cursor = EXCLUDED.source_event_cursor",
            )
            rdb.pg_execute(
                sql,
                (
                    workspace_id,
                    pid,
                    blob["goal"],
                    blob["milestone"],
                    json.dumps(blob["risks"]),
                    json.dumps(blob["recent_accomplishments"]),
                    json.dumps(blob["conventions_observed"]),
                    now,
                    blob["source_event_cursor"],
                ),
            )
            return
        assert self._db is not None
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workspace_memory
                (workspace_id, project_id, goal, milestone, risks, recent_accomplishments,
                 conventions_observed, generated_at, source_event_cursor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    pid,
                    blob["goal"],
                    blob["milestone"],
                    json.dumps(blob["risks"]),
                    json.dumps(blob["recent_accomplishments"]),
                    json.dumps(blob["conventions_observed"]),
                    now,
                    blob["source_event_cursor"],
                ),
            )

    def add_convention(self, workspace_id: str, convention: str, project_id: str | None = None) -> dict[str, Any]:
        mem = self.get(workspace_id, project_id)
        convs = list(mem.get("conventions_observed") or [])
        if convention not in convs:
            convs.append(convention)
        pid = project_id or ""
        if self._postgres:
            rdb.pg_execute(
                """
                UPDATE workspace_memory SET conventions_observed = %s::jsonb
                WHERE workspace_id = %s AND project_id = %s
                """,
                (json.dumps(convs), workspace_id, pid),
            )
        else:
            assert self._db is not None
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    "UPDATE workspace_memory SET conventions_observed = ? WHERE workspace_id = ? AND project_id = ?",
                    (json.dumps(convs), workspace_id, pid),
                )
        mem["conventions_observed"] = convs
        return mem

    def update_cursor(self, workspace_id: str, cursor: str, project_id: str | None = None) -> None:
        pid = project_id or ""
        if self._postgres:
            rdb.pg_execute(
                "UPDATE workspace_memory SET source_event_cursor = %s WHERE workspace_id = %s AND project_id = %s",
                (cursor, workspace_id, pid),
            )
            return
        assert self._db is not None
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "UPDATE workspace_memory SET source_event_cursor = ? WHERE workspace_id = ? AND project_id = ?",
                (cursor, workspace_id, pid),
            )
