"""Planner golden-set — rule fallback and model planner IR expectations."""

from __future__ import annotations

import pytest

from raphael_ai.intelligence.model_runtime import GemmaRuntime
from raphael_ai.intelligence.planner import PlannerIR, model_planner, rule_planner
from raphael_ai.intelligence.service import IntelligenceService
from fake_ollama import install_fake_ollama

GOLDEN = [
    ("Where did we increase pressure?", "find_events", ["events"], True),
    ("Show connector BOM items", "find_artifacts", ["graph", "artifacts"], True),
    ("Open reviews waiting for approval", "find_reviews", ["reviews"], False),
    ("List project modules", "find_modules", ["modules"], False),
    ("What is the impact of changing conn-a?", "graph_impact", ["graph"], False),
    ("Summarize recent activity", "general", ["events", "modules"], False),
    ("parameter requirement changes", "find_events", ["events"], True),
    ("bom connector pinout", "find_artifacts", ["graph", "artifacts"], True),
    ("approval review status", "find_reviews", ["reviews"], False),
    ("module lineage project", "find_modules", ["modules"], False),
    ("affect downstream impact", "graph_impact", ["graph"], False),
    ("latest workspace updates", "general", ["events", "modules"], False),
]


@pytest.mark.parametrize("question,expected_intent,expected_sources,has_filters", GOLDEN)
def test_rule_planner_golden(
    question: str, expected_intent: str, expected_sources: list[str], has_filters: bool
) -> None:
    ir, plan_id = rule_planner(question, "default")
    assert plan_id.startswith("plan_")
    assert isinstance(ir, PlannerIR)
    assert ir.intent == expected_intent
    assert ir.sources == expected_sources
    if has_filters:
        assert ir.filters or ir.graph_hops


@pytest.mark.parametrize("question,expected_intent,expected_sources,has_filters", GOLDEN)
def test_model_planner_golden_live_tier(
    monkeypatch: pytest.MonkeyPatch,
    golden_chat_responder,
    question: str,
    expected_intent: str,
    expected_sources: list[str],
    has_filters: bool,
) -> None:
    install_fake_ollama(monkeypatch, chat_response=golden_chat_responder)
    runtime = GemmaRuntime()
    ir = model_planner(question, "default", runtime)
    assert ir is not None
    assert runtime.last_tier == "live"
    assert ir.intent == expected_intent
    assert ir.sources == expected_sources
    if has_filters:
        assert ir.filters or ir.graph_hops


def test_model_planner_returns_none_on_stub_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "stub")
    runtime = GemmaRuntime()
    assert model_planner("list modules", "default", runtime) is None
    assert runtime.last_tier == "rule_stub"


def test_service_plan_uses_model_when_available(
    monkeypatch: pytest.MonkeyPatch,
    golden_chat_responder,
) -> None:
    install_fake_ollama(monkeypatch, chat_response=golden_chat_responder)
    svc = IntelligenceService(runtime=GemmaRuntime())
    ir, plan_id = svc.plan("List project modules", "default")
    assert plan_id.startswith("plan_")
    assert ir.intent == "find_modules"
    assert svc.last_tier == "live"


def test_service_plan_falls_back_to_rules_when_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "stub")
    svc = IntelligenceService(runtime=GemmaRuntime())
    ir, plan_id = svc.plan("List project modules", "default")
    assert plan_id.startswith("plan_")
    assert ir.intent == "find_modules"
    assert svc.last_tier == "rule_stub"
