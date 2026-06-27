"""Shared test fixtures for raphael-ai."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from fake_ollama import golden_chat_responder as _golden_chat_responder
from fake_ollama import golden_model_responses as _golden_model_responses


@pytest.fixture
def golden_model_responses() -> dict[str, dict[str, Any]]:
    return _golden_model_responses()


@pytest.fixture
def golden_chat_responder(golden_model_responses: dict[str, dict[str, Any]]) -> Callable[[dict[str, Any]], str]:
    return _golden_chat_responder(golden_model_responses)
