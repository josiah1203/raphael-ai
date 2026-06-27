"""Intelligence API tests."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RAPHAEL_MODEL_BACKEND", "stub")

from raphael_ai.app import app
from raphael_ai.intelligence.model_runtime import GemmaRuntime
from raphael_ai.intelligence.service import IntelligenceService
from fake_ollama import install_fake_ollama

client = TestClient(app)


def test_intelligence_status() -> None:
    res = client.get("/v1/intelligence/status")
    assert res.status_code == 200
    body = res.json()
    assert body["service"] == "raphael-ai-intelligence"
    assert "model_version" in body
    assert "available" in body
    assert "tier" in body
    assert "downstream" in body


def test_intelligence_ask() -> None:
    res = client.post("/v1/intelligence/ask", json={"question": "Where did we increase pressure?"})
    assert res.status_code == 200
    body = res.json()
    assert body.get("plan_id")
    assert "answer" in body
    assert "citations" in body
    assert "errors" in body
    assert "plan" in body
    assert body.get("model_version")
    assert body.get("model_tier") in ("live", "cached", "rule_stub")


def test_intelligence_memory() -> None:
    res = client.get("/v1/intelligence/memory?workspace_id=default")
    assert res.status_code == 200
    assert res.json().get("workspace_id") == "default"


def test_memory_conventions() -> None:
    res = client.post(
        "/v1/intelligence/memory/conventions",
        json={"workspace_id": "default", "convention": "Run thermal sim on enclosure edits"},
    )
    assert res.status_code == 200
    assert "Run thermal sim on enclosure edits" in res.json().get("conventions_observed", [])


def test_intelligence_plan() -> None:
    res = client.post("/v1/intelligence/plan", json={"question": "list modules"})
    assert res.status_code == 200
    body = res.json()
    assert body.get("plan_id")
    assert body.get("plan", {}).get("sources")
    assert body.get("model_tier") == "rule_stub"


def test_intelligence_plan_live_tier_with_mocked_ollama(
    monkeypatch: pytest.MonkeyPatch,
    golden_chat_responder,
) -> None:
    install_fake_ollama(monkeypatch, chat_response=golden_chat_responder)
    svc = IntelligenceService(runtime=GemmaRuntime())
    ir, plan_id = svc.plan("List project modules", "ws-smoke")
    assert plan_id.startswith("plan_")
    assert ir.intent == "find_modules"
    assert svc.last_tier == "live"


def test_workflow_draft() -> None:
    res = client.post(
        "/v1/intelligence/workflows/draft",
        json={"description": "Whenever PCB changes, rerun simulations"},
    )
    assert res.status_code == 200
    assert res.json()["draft"]["trigger_type"]


def test_squash_label_stub() -> None:
    res = client.post(
        "/v1/intelligence/squash/label",
        json={"ops": [{"type": "parameter.update", "name": "pressure_max"}]},
    )
    assert res.status_code == 200
    assert res.json()["intent_summary"]


def test_health_reports_intelligence() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["intelligence"]["model_version"]
