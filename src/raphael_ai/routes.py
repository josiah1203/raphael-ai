"""AI jobs and suggestions API."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["ai"])
_db = Path(os.environ.get("RAPHAEL_AI_DB", "/tmp/raphael-ai.db"))
_conn = sqlite3.connect(_db, check_same_thread=False)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        module_id TEXT,
        created_at TEXT NOT NULL
    )"""
)
_conn.commit()


@router.get("/jobs")
def list_jobs() -> dict[str, list]:
    rows = _conn.execute("SELECT id, job_type, status, module_id, created_at FROM jobs ORDER BY created_at DESC").fetchall()
    return {"jobs": [{"id": r[0], "job_type": r[1], "status": r[2], "module_id": r[3], "created_at": r[4]} for r in rows]}


@router.post("/jobs")
def create_job(body: dict[str, Any]) -> dict[str, Any]:
    jid = f"job-{int(datetime.now(timezone.utc).timestamp())}"
    now = datetime.now(timezone.utc).isoformat()
    _conn.execute(
        "INSERT INTO jobs (id, job_type, status, module_id, created_at) VALUES (?, ?, 'queued', ?, ?)",
        (jid, body.get("job_type", "copilot"), body.get("module_id"), now),
    )
    _conn.commit()
    return {"id": jid, "status": "queued", "created_at": now}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    row = _conn.execute("SELECT id, job_type, status, module_id, created_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, detail="not_found")
    return {"id": row[0], "job_type": row[1], "status": row[2], "module_id": row[3], "created_at": row[4]}


@router.get("/suggestions")
def suggestions(module_id: str | None = None) -> dict[str, list]:
    return {
        "suggestions": [
            {"id": "s1", "text": "Review USB-PD input stage net lengths", "module_id": module_id or "power-board-v2"},
        ]
    }
