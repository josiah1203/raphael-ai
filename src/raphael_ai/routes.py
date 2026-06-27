"""AI jobs API (federated training) + legacy suggestions alias."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from raphael_ai.calliope_ai.job_store import AIJobStore

router = APIRouter(tags=["ai"])
_raw = os.environ.get("RAPHAEL_AI_DB", "").strip()
_db = Path(_raw) if _raw else None
_store = AIJobStore(db_path=_db)


@router.get("/jobs")
def list_jobs(tenant_id: str | None = None) -> dict[str, list]:
    return {"jobs": _store.list_jobs(tenant_id=tenant_id)}


@router.post("/jobs")
def create_job(body: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if body.get("module_id"):
        extra["module_id"] = body["module_id"]
    return _store.create_job(
        tenant_id=body.get("tenant_id", "default"),
        model_type=body.get("job_type", body.get("model_type", "lora")),
        status=body.get("status", "pending"),
        metrics=body.get("metrics"),
        extra=extra or None,
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = _store.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="not_found")
    return job


@router.get("/suggestions")
def suggestions(module_id: str | None = None) -> dict[str, list]:
    """Legacy alias — prefer /v1/intelligence/suggestions."""
    return {
        "suggestions": [
            {
                "id": "s1",
                "text": "Review USB-PD input stage net lengths",
                "module_id": module_id or "power-board-v2",
            }
        ]
    }
