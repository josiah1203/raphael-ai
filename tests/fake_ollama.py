"""Fake Ollama client for unit tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest


def install_fake_ollama(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name: str = "gemma2:2b",
    chat_response: str | Callable[[dict[str, Any]], str] | None = None,
) -> None:
    """Patch httpx.Client so GemmaRuntime sees a reachable Ollama with chat/tags."""

    def default_chat(_payload: dict[str, Any]) -> str:
        return '{"intent":"general","sources":["events"],"limit":50}'

    responder = chat_response if callable(chat_response) else (lambda _p: chat_response or default_chat(_p))

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str) -> Any:
            class TagsRes:
                status_code = 200

                def json(self) -> dict[str, Any]:
                    return {"models": [{"name": model_name}]}

            return TagsRes()

        def post(self, url: str, json: dict[str, Any]) -> Any:
            class ChatRes:
                status_code = 200

                def json(self) -> dict[str, Any]:
                    return {"message": {"content": responder(json)}}

            return ChatRes()

    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("RAPHAEL_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr("raphael_ai.intelligence.model_runtime.httpx.Client", FakeClient)


def golden_model_responses() -> dict[str, dict[str, Any]]:
    """Expected PlannerIR payloads for golden question patterns (model path)."""
    return {
        "Where did we increase pressure?": {
            "intent": "find_events",
            "filters": [{"field": "parameter.name", "op": "contains", "value": "pressure"}],
            "sources": ["events"],
            "limit": 50,
        },
        "Show connector BOM items": {
            "intent": "find_artifacts",
            "graph_hops": [{"from": "connector", "edge": "bom_contains", "depth": 2}],
            "sources": ["graph", "artifacts"],
            "limit": 50,
        },
        "Open reviews waiting for approval": {
            "intent": "find_reviews",
            "sources": ["reviews"],
            "limit": 50,
        },
        "List project modules": {
            "intent": "find_modules",
            "sources": ["modules"],
            "limit": 50,
        },
        "What is the impact of changing conn-a?": {
            "intent": "graph_impact",
            "sources": ["graph"],
            "limit": 50,
        },
        "Summarize recent activity": {
            "intent": "general",
            "sources": ["events", "modules"],
            "limit": 50,
        },
        "parameter requirement changes": {
            "intent": "find_events",
            "filters": [{"field": "parameter.name", "op": "contains", "value": "pressure"}],
            "sources": ["events"],
            "limit": 50,
        },
        "bom connector pinout": {
            "intent": "find_artifacts",
            "graph_hops": [{"from": "connector", "edge": "bom_contains", "depth": 2}],
            "sources": ["graph", "artifacts"],
            "limit": 50,
        },
        "approval review status": {
            "intent": "find_reviews",
            "sources": ["reviews"],
            "limit": 50,
        },
        "module lineage project": {
            "intent": "find_modules",
            "sources": ["modules"],
            "limit": 50,
        },
        "affect downstream impact": {
            "intent": "graph_impact",
            "sources": ["graph"],
            "limit": 50,
        },
        "latest workspace updates": {
            "intent": "general",
            "sources": ["events", "modules"],
            "limit": 50,
        },
    }


def golden_chat_responder(responses: dict[str, dict[str, Any]] | None = None) -> Callable[[dict[str, Any]], str]:
    mapping = responses or golden_model_responses()

    def responder(payload: dict[str, Any]) -> str:
        user = ""
        for msg in payload.get("messages", []):
            if msg.get("role") == "user":
                user = msg.get("content", "")
        question = user.split("Question:", 1)[-1].strip() if "Question:" in user else user
        for key, ir in mapping.items():
            if key in question or question in key:
                return json.dumps(ir)
        return json.dumps({"intent": "general", "sources": ["events"], "limit": 50})

    return responder
