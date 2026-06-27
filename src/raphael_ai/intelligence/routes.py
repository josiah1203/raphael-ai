"""Workspace Intelligence API — unified semantic layer."""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Header

from raphael_ai.intelligence.context import ContextAssembler
from raphael_ai.intelligence.executor import QueryExecutor
from raphael_ai.intelligence.memory import WorkspaceMemoryStore, fetch_audit_cursor
from raphael_ai.intelligence.service import IntelligenceService
from raphael_ai.intelligence.summaries import SummaryStore
from raphael_ai.intelligence.workflows import PatternObserver, WorkflowDrafter

router = APIRouter(tags=["intelligence"])

_intel = IntelligenceService()
_memory = WorkspaceMemoryStore()
_executor = QueryExecutor()
_context = ContextAssembler(_memory)
_summaries = SummaryStore()
_drafter = WorkflowDrafter()
_patterns = PatternObserver()


def _service_url(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default).rstrip("/")


def _downstream_health() -> dict[str, dict[str, Any]]:
    checks = {
        "audit": (_service_url("RAPHAEL_AUDIT_URL", "http://127.0.0.1:8093"), "/health"),
        "graph": (_service_url("RAPHAEL_GRAPH_URL", "http://127.0.0.1:8100"), "/health"),
        "workspaces": (_service_url("RAPHAEL_WORKSPACES_URL", "http://127.0.0.1:8083"), "/health"),
    }
    out: dict[str, dict[str, Any]] = {}
    try:
        with httpx.Client(timeout=3.0) as client:
            for name, (base, path) in checks.items():
                try:
                    res = client.get(f"{base}{path}")
                    out[name] = {"reachable": res.status_code < 500, "status_code": res.status_code}
                except httpx.HTTPError as exc:
                    out[name] = {"reachable": False, "error": str(exc)}
    except Exception:
        pass
    return out


@router.get("/status")
def intelligence_status() -> dict[str, Any]:
    status = _intel.model_status()
    return {
        "service": "raphael-ai-intelligence",
        "model_version": _intel.model_version,
        "downstream": _downstream_health(),
        **status,
    }


def _emit_intelligence_rwu(org_id: str | None) -> None:
    try:
        from raphael_contracts.rwu import emit_rwu

        emit_rwu(org_id or "org_default", 1.0, "intelligence.ask")
    except Exception:
        pass


@router.post("/ask")
def intelligence_ask(body: dict[str, Any], x_raphael_org_id: str | None = Header(default=None)) -> dict[str, Any]:
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(400, detail="question_required")
    workspace_id = body.get("workspace_id", "default")
    project_id = body.get("project_id")
    plan, plan_id = _intel.plan(question, workspace_id, project_id)
    retrieved, errors = _executor.execute(plan, workspace_id, project_id)
    pack = _context.assemble(workspace_id, project_id, retrieved, body.get("conversation"))
    citations = [
        {"source": r.get("source"), "id": r.get("id"), "summary": r.get("summary")} for r in retrieved[:20]
    ]
    answer = _intel.summarize_answer(question, citations, pack, plan_id=plan_id)
    _emit_intelligence_rwu(x_raphael_org_id or body.get("org_id"))
    return {
        "plan_id": plan_id,
        "plan": plan.model_dump(by_alias=True),
        "answer": answer,
        "citations": citations,
        "errors": errors,
        "context_pack": pack,
        "sources": plan.sources,
        "model_version": _intel.model_version,
        "model_tier": _intel.last_tier,
    }


@router.post("/plan")
def intelligence_plan(body: dict[str, Any]) -> dict[str, Any]:
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(400, detail="question_required")
    workspace_id = body.get("workspace_id", "default")
    project_id = body.get("project_id")
    plan, plan_id = _intel.plan(question, workspace_id, project_id)
    return {
        "plan_id": plan_id,
        "plan": plan.model_dump(by_alias=True),
        "model_version": _intel.model_version,
        "model_tier": _intel.last_tier,
    }


@router.get("/memory")
def intelligence_memory(workspace_id: str = "default", project_id: str | None = None) -> dict[str, Any]:
    return _memory.get(workspace_id, project_id)


@router.post("/memory/regenerate")
def regenerate_memory(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    workspace_id = body.get("workspace_id", "default")
    project_id = body.get("project_id")
    facts: dict[str, Any] = {"modules": [], "recent_events": []}
    cursor = fetch_audit_cursor(project_id)
    try:
        ws = _service_url("RAPHAEL_WORKSPACES_URL", "http://127.0.0.1:8083")
        audit = _service_url("RAPHAEL_AUDIT_URL", "http://127.0.0.1:8093")
        with httpx.Client(timeout=10.0) as client:
            mres = client.get(f"{ws}/v1/workspaces/{workspace_id}/modules")
            if mres.status_code == 200:
                facts["modules"] = mres.json().get("modules", [])[:20]
            params: dict[str, Any] = {"limit": 50}
            if project_id:
                params["project_id"] = project_id
            eres = client.get(f"{audit}/v1/audit/timeline", params=params)
            if eres.status_code == 200:
                facts["recent_events"] = eres.json().get("events", [])[:30]
                cursor = eres.json().get("next_cursor") or cursor
    except Exception:
        pass
    model_blob = _intel.regenerate_memory(workspace_id, facts)
    return _memory.regenerate(workspace_id, project_id, model_blob, event_cursor=cursor)


@router.post("/memory/conventions")
def add_memory_convention(body: dict[str, Any]) -> dict[str, Any]:
    convention = (body.get("convention") or body.get("label") or "").strip()
    if not convention:
        raise HTTPException(400, detail="convention_required")
    workspace_id = body.get("workspace_id", "default")
    project_id = body.get("project_id")
    return _memory.add_convention(workspace_id, convention, project_id)


@router.get("/suggestions")
def intelligence_suggestions() -> dict[str, list]:
    return {"suggestions": _patterns.list_suggestions()}


@router.post("/workflows/draft")
def workflow_draft(body: dict[str, Any]) -> dict[str, Any]:
    desc = body.get("description", "").strip()
    if not desc:
        raise HTTPException(400, detail="description_required")
    draft = _intel.draft_workflow(desc) or _drafter.draft(desc)
    return {"draft": draft, "model_version": _intel.model_version}


@router.post("/squash/label")
def squash_label(body: dict[str, Any]) -> dict[str, str]:
    ops = body.get("ops") or []
    if not ops and body.get("events"):
        ops = [
            {"type": ev.get("event_type", "event"), "name": str(ev.get("payload", {}))[:40]}
            for ev in body["events"]
        ]
    label = _intel.squash_label(ops if isinstance(ops, list) else [], body.get("message"))
    return {"intent_summary": label, "model_version": _intel.model_version, "model_tier": _intel.last_tier}


@router.get("/summaries")
def list_summaries(workspace_id: str = "default") -> dict[str, list]:
    return {"summaries": _summaries.list_summaries(workspace_id)}


@router.post("/summaries/generate")
def generate_summary(body: dict[str, Any]) -> dict[str, Any]:
    workspace_id = body.get("workspace_id", "default")
    events = body.get("events", [])
    model_summary = _intel.summarize_events(events)
    if model_summary:
        return _summaries.store_summary(workspace_id, events, model_summary)
    return _summaries.summarize_events(workspace_id, events)
