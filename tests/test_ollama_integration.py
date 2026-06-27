"""Mocked Ollama integration tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from raphael_ai.intelligence.model_runtime import GemmaRuntime
from raphael_ai.intelligence.service import IntelligenceService


def test_ollama_plan_live_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("RAPHAEL_OLLAMA_URL", "http://ollama.test")

    tags = MagicMock()
    tags.status_code = 200
    tags.json.return_value = {"models": [{"name": "gemma2:2b"}]}

    chat = MagicMock()
    chat.status_code = 200
    chat.json.return_value = {
        "message": {
            "content": '{"intent":"find_modules","sources":["modules"],"limit":10,"filters":[],"graph_hops":[]}'
        }
    }

    with patch.object(httpx.Client, "get", return_value=tags), patch.object(httpx.Client, "post", return_value=chat):
        svc = IntelligenceService(GemmaRuntime())
        ir, plan_id = svc.plan("list modules", "default")
    assert plan_id.startswith("plan_")
    assert ir.intent == "find_modules"
    assert svc.last_tier == "live"


def test_ollama_cached_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("RAPHAEL_OLLAMA_URL", "http://ollama.test")

    tags = MagicMock()
    tags.status_code = 200
    tags.json.return_value = {"models": [{"name": "gemma2:2b"}]}

    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"message": {"content": "cached answer"}}

    fail = MagicMock(side_effect=httpx.ConnectError("down"))

    with patch.object(httpx.Client, "get", return_value=tags), patch.object(
        httpx.Client, "post", side_effect=[ok, fail]
    ), patch.object(GemmaRuntime, "startup_probe", return_value=tags):
        rt = GemmaRuntime()
        first = rt.complete("sys", "user", task="summarize")
        second = rt.complete("sys", "user", task="summarize")
    assert first == "cached answer"
    assert second == "cached answer"
    assert rt.last_tier == "cached"
