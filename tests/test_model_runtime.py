"""Model runtime unit tests."""

import pytest

from raphael_ai.intelligence.model_runtime import GemmaRuntime


def test_parse_json_embedded() -> None:
    assert GemmaRuntime.parse_json('Here: {"intent":"general","sources":["events"]}') == {
        "intent": "general",
        "sources": ["events"],
    }


def test_stub_backend_skips_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "stub")
    rt = GemmaRuntime()
    text = rt.complete("sys", "user")
    assert text is None
    assert rt.last_tier == "rule_stub"
    assert rt.model_version == "stub-v1"


def test_status_when_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("RAPHAEL_OLLAMA_URL", "http://127.0.0.1:1")
    rt = GemmaRuntime()
    s = rt.status()
    assert s.backend == "ollama"
    assert s.available is False
    assert s.tier == "rule_stub"


def test_mocked_ollama_live_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPHAEL_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("RAPHAEL_OLLAMA_URL", "http://127.0.0.1:11434")

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def get(self, url: str):
            class TagsRes:
                status_code = 200

                def json(self):
                    return {"models": [{"name": "gemma2:2b"}]}

            return TagsRes()

        def post(self, url: str, json: dict):
            class ChatRes:
                status_code = 200

                def json(self):
                    return {"message": {"content": '{"intent":"general","sources":["events"],"limit":10}'}}

            return ChatRes()

    monkeypatch.setattr("raphael_ai.intelligence.model_runtime.httpx.Client", FakeClient)
    rt = GemmaRuntime()
    text = rt.complete("sys", "user", json_mode=True, task="plan")
    assert rt.last_tier == "live"
    assert text is not None
    assert "intent" in text
