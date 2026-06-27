"""Gemma model runtime — Ollama with tiered degradation (live/cached/rule_stub)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_GEMMA_MODEL = "gemma2:2b"
ModelTier = Literal["live", "cached", "rule_stub"]


@dataclass(frozen=True)
class ModelStatus:
    backend: str
    model: str
    available: bool
    detail: str
    tier: ModelTier


class GemmaRuntime:
    """Open-weights Gemma via Ollama. Inference-only; no customer data retention."""

    def __init__(self) -> None:
        self.backend = os.environ.get("RAPHAEL_MODEL_BACKEND", "ollama").lower()
        self.ollama_url = os.environ.get("RAPHAEL_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
        self.model = os.environ.get("RAPHAEL_GEMMA_MODEL", DEFAULT_GEMMA_MODEL)
        self.timeout_plan = float(os.environ.get("RAPHAEL_MODEL_TIMEOUT_PLAN_SEC", "45"))
        self.timeout_summarize = float(os.environ.get("RAPHAEL_MODEL_TIMEOUT_SUMMARIZE_SEC", "60"))
        self.timeout_squash = float(os.environ.get("RAPHAEL_MODEL_TIMEOUT_SQUASH_SEC", "30"))
        self.timeout = float(os.environ.get("RAPHAEL_MODEL_TIMEOUT_SEC", "60"))
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._last_tier: ModelTier = "rule_stub"
        self._probe_detail = ""
        self._warmed = False
        if self.backend != "stub":
            self.startup_probe(block=False)

    @property
    def model_version(self) -> str:
        if self.backend == "stub":
            return "stub-v1"
        return f"gemma/{self.model}@{self.backend}"

    @property
    def last_tier(self) -> ModelTier:
        return self._last_tier

    def startup_probe(self, *, block: bool = True, max_wait_sec: float | None = None) -> ModelStatus:
        """Pull/warmup check against Ollama on startup; optionally blocks until model responds."""
        if not block:
            status = self.status()
            self._probe_detail = status.detail
            return ModelStatus(status.backend, status.model, status.available, self._probe_detail, status.tier)

        wait_budget = max_wait_sec
        if wait_budget is None:
            wait_budget = float(os.environ.get("RAPHAEL_MODEL_STARTUP_WAIT_SEC", "300"))
        deadline = time.monotonic() + wait_budget
        last: ModelStatus | None = None
        while time.monotonic() < deadline:
            status = self.status()
            last = status
            if status.available and self.backend == "ollama":
                try:
                    with httpx.Client(timeout=10.0) as client:
                        client.post(
                            f"{self.ollama_url}/api/generate",
                            json={"model": self.model, "prompt": "ping", "stream": False, "options": {"num_predict": 1}},
                        )
                    self._warmed = True
                    self._probe_detail = "ready (warmed)"
                    return ModelStatus(status.backend, status.model, True, self._probe_detail, "live")
                except httpx.HTTPError as exc:
                    logger.warning("Ollama warmup failed: %s", exc)
                    self._probe_detail = f"warmup failed: {exc}"
            elif status.available:
                self._probe_detail = status.detail
                return ModelStatus(status.backend, status.model, True, self._probe_detail, self._resolve_tier())
            logger.info("Waiting for model %s (%s)", self.model, status.detail)
            time.sleep(5)
        if last is None:
            last = self.status()
        self._probe_detail = last.detail
        return ModelStatus(last.backend, last.model, last.available, self._probe_detail, last.tier)

    def status(self) -> ModelStatus:
        if self.backend == "stub":
            return ModelStatus("stub", self.model, False, "RAPHAEL_MODEL_BACKEND=stub", "rule_stub")
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.ollama_url}/api/tags")
                if res.status_code != 200:
                    return ModelStatus("ollama", self.model, False, f"tags HTTP {res.status_code}", "rule_stub")
                names = {m.get("name", "") for m in res.json().get("models", [])}
                base = self.model.split(":")[0]
                ok = any(n == self.model or n.startswith(f"{base}:") for n in names)
                detail = self._probe_detail or ("ready" if ok else f"model {self.model} not pulled")
                tier: ModelTier = "live" if ok else "rule_stub"
                return ModelStatus("ollama", self.model, ok, detail, tier)
        except httpx.HTTPError as exc:
            return ModelStatus("ollama", self.model, False, str(exc), "rule_stub")

    def _resolve_tier(self) -> ModelTier:
        if self.backend == "stub":
            return "rule_stub"
        s = self.status()
        if s.available:
            return "live"
        if self._cache:
            return "cached"
        return "rule_stub"

    @staticmethod
    def _cache_key(system: str, user: str, json_mode: bool) -> str:
        raw = f"{system}\0{user}\0{json_mode}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        task: Literal["plan", "summarize", "squash", "default"] = "default",
    ) -> str | None:
        if self.backend == "stub":
            self._last_tier = "rule_stub"
            return None

        timeout = {
            "plan": self.timeout_plan,
            "summarize": self.timeout_summarize,
            "squash": self.timeout_squash,
        }.get(task, self.timeout)

        key = self._cache_key(system, user, json_mode)
        live = self._complete_live(system, user, json_mode=json_mode, timeout=timeout)
        if live is not None:
            self._last_tier = "live"
            with self._cache_lock:
                self._cache[key] = live
            return live

        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            self._last_tier = "cached"
            logger.info("model tier=cached key=%s task=%s", key, task)
            return cached

        self._last_tier = "rule_stub"
        return None

    def _complete_live(self, system: str, user: str, *, json_mode: bool, timeout: float) -> str | None:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }
        if json_mode:
            payload["format"] = "json"
        started = time.monotonic()
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.post(f"{self.ollama_url}/api/chat", json=payload)
                if res.status_code != 200:
                    logger.warning("Ollama chat failed: %s %s", res.status_code, res.text[:200])
                    return None
                text = res.json().get("message", {}).get("content", "").strip()
                logger.info("model tier=live latency_ms=%.0f", (time.monotonic() - started) * 1000)
                return text
        except httpx.HTTPError as exc:
            logger.warning("Ollama unreachable: %s", exc)
            return None

    @staticmethod
    def parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None
