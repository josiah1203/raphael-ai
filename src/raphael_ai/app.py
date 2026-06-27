"""Raphael service: raphael-ai."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from raphael_contracts.errors import ErrorResponse
from raphael_ai.intelligence.routes import router as intelligence_router
from raphael_ai.intelligence.memory import WorkspaceMemoryStore, fetch_audit_cursor
from raphael_ai.intelligence.service import IntelligenceService
from raphael_ai.intelligence.summaries import SummaryStore
from raphael_ai.intelligence.workflows import PatternObserver
from raphael_ai.routes import router

_patterns = PatternObserver()
_memory = WorkspaceMemoryStore()
_summaries = SummaryStore()
_intel = IntelligenceService()
_debounce_lock = threading.Lock()
_debounce_at = 0.0
_DEBOUNCE_SEC = float(__import__("os").environ.get("RAPHAEL_MEMORY_DEBOUNCE_SEC", "5"))


def _schedule_memory_refresh(workspace_id: str = "default", project_id: str | None = None) -> None:
    global _debounce_at
    with _debounce_lock:
        now = time.monotonic()
        if now - _debounce_at < _DEBOUNCE_SEC:
            return
        _debounce_at = now

    def _run() -> None:
        cursor = fetch_audit_cursor(project_id)
        facts: dict[str, Any] = {"trigger": "kafka"}
        blob = _intel.regenerate_memory(workspace_id, facts)
        _memory.regenerate(workspace_id, project_id, blob, event_cursor=cursor)

    threading.Thread(target=_run, daemon=True, name="memory-refresh").start()


def _on_bus_event(envelope: dict) -> None:
    event_type = envelope.get("type") or ""
    data = envelope.get("data") or {}
    if not (
        event_type.startswith("raphael.audit.")
        or event_type.startswith("raphael.workspaces.")
        or event_type == "raphael.workspaces.commit"
    ):
        return

    ev = {"event_type": data.get("event_type") or event_type, "project_id": data.get("project_id"), **data}
    _patterns.observe_timeline([ev])

    workspace_id = data.get("workspace_id") or "default"
    project_id = data.get("project_id")
    _schedule_memory_refresh(workspace_id, project_id)

    events = data.get("events") or [ev]
    if events:
        threading.Thread(
            target=lambda: _summaries.summarize_events(workspace_id, events[:30]),
            daemon=True,
            name="summary-refresh",
        ).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime = _intel._runtime  # noqa: SLF001
    if runtime.backend != "stub":
        # Warm Ollama in the background so /health and intelligence routes are
        # available immediately (rule_stub tier until the model responds).
        threading.Thread(
            target=lambda: runtime.startup_probe(block=True),
            daemon=True,
            name="ollama-warmup",
        ).start()
    try:
        from raphael_contracts.kafka import start_consumer

        start_consumer(_on_bus_event, group_id="raphael-ai-patterns")
    except Exception:
        pass
    yield


app = FastAPI(
    title="raphael-ai",
    description="Workspace Intelligence + federated AI jobs",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/v1/ai")
app.include_router(intelligence_router, prefix="/v1/intelligence")


@app.get("/health")
def health() -> dict[str, str | dict]:
    status = _intel.model_status()
    return {
        "status": "ok",
        "service": "raphael-ai",
        "intelligence": {
            "model_version": _intel.model_version,
            "model_available": status["available"],
            "backend": status["backend"],
            "tier": status.get("tier", "rule_stub"),
        },
    }


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:
    err = ErrorResponse(code="internal_error", message=str(exc))
    return JSONResponse(status_code=500, content=err.model_dump())
