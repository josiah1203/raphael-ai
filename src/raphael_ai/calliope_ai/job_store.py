"""Persisted federated AI job store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raphael_audit.core.paths import calliope_home
from raphael_audit.core.uuid7 import uuid7_str


class AIJobStore:
    """SQLite store for federated AI training jobs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or calliope_home() / "ai_jobs.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_jobs (
                job_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                body TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_jobs_tenant ON ai_jobs(tenant_id)"
        )
        self._conn.commit()

    def create_job(
        self,
        *,
        tenant_id: str,
        model_type: str = "lora",
        status: str = "pending",
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = f"job_{uuid7_str()}"
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        body = {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "model_type": model_type,
            "status": status,
            "metrics": metrics or {},
            "created_at": created_at,
            **(extra or {}),
        }
        self._conn.execute(
            """
            INSERT INTO ai_jobs (job_id, tenant_id, model_type, status, metrics_json, created_at, body)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                tenant_id,
                model_type,
                status,
                json.dumps(body["metrics"]),
                created_at,
                json.dumps(body, separators=(",", ":")),
            ),
        )
        self._conn.commit()
        return body

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT body FROM ai_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["body"])

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if not job:
            return None
        job.update(updates)
        self._conn.execute(
            """
            UPDATE ai_jobs
            SET status = ?, metrics_json = ?, body = ?
            WHERE job_id = ?
            """,
            (
                job["status"],
                json.dumps(job.get("metrics", {})),
                json.dumps(job, separators=(",", ":")),
                job_id,
            ),
        )
        self._conn.commit()
        return job

    def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        if tenant_id:
            rows = self._conn.execute(
                "SELECT body FROM ai_jobs WHERE tenant_id = ? ORDER BY created_at",
                (tenant_id,),
            )
        else:
            rows = self._conn.execute("SELECT body FROM ai_jobs ORDER BY created_at")
        return [json.loads(r["body"]) for r in rows]

    def close(self) -> None:
        self._conn.close()
