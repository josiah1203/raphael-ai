"""Intelligence orchestration — Gemma when available, deterministic fallback."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from raphael_ai.intelligence.model_runtime import GemmaRuntime, ModelTier
from raphael_ai.intelligence.plan_store import PlanStore
from raphael_ai.intelligence.planner import PlannerIR, model_planner, rule_planner

logger = logging.getLogger(__name__)


class IntelligenceService:
    def __init__(self, runtime: GemmaRuntime | None = None, plan_store: PlanStore | None = None) -> None:
        self._runtime = runtime or GemmaRuntime()
        self._plans = plan_store or PlanStore()

    @property
    def model_version(self) -> str:
        return self._runtime.model_version

    @property
    def last_tier(self) -> ModelTier:
        return self._runtime.last_tier

    def model_status(self) -> dict[str, Any]:
        s = self._runtime.status()
        return {
            "backend": s.backend,
            "model": s.model,
            "available": s.available,
            "detail": s.detail,
            "tier": s.tier,
        }

    def plan(self, question: str, workspace_id: str = "default", project_id: str | None = None) -> tuple[PlannerIR, str]:
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        ir = model_planner(question, workspace_id, self._runtime)
        if ir is not None:
            self._plans.save(plan_id, question, workspace_id, ir, project_id)
            logger.info(
                "plan_id=%s stage=plan tier=%s latency_ms=%.0f",
                plan_id,
                self._runtime.last_tier,
                (time.monotonic() - started) * 1000,
            )
            return ir, plan_id
        ir, _ = rule_planner(question, workspace_id)
        self._plans.save(plan_id, question, workspace_id, ir, project_id)
        logger.info("plan_id=%s stage=plan tier=rule_stub latency_ms=%.0f", plan_id, (time.monotonic() - started) * 1000)
        return ir, plan_id

    def summarize_answer(
        self,
        question: str,
        citations: list[dict[str, Any]],
        pack: dict[str, Any],
        *,
        plan_id: str | None = None,
    ) -> str:
        if not citations:
            return f"No results found for: {question}"
        started = time.monotonic()
        raw = self._runtime.complete(
            "Summarize search results for a hardware engineering workspace. Be concise. Cite sources by id.",
            json.dumps({"question": question, "citations": citations[:15], "memory": pack.get("workspace_summary")}, default=str)[:6000],
            task="summarize",
        )
        logger.info("plan_id=%s stage=summarize tier=%s latency_ms=%.0f", plan_id or "-", self._runtime.last_tier, (time.monotonic() - started) * 1000)
        if raw:
            return raw.strip()
        lines = [f"Found {len(citations)} result(s):"]
        for c in citations[:10]:
            lines.append(f"- [{c.get('source')}] {c.get('id')}: {c.get('summary')}")
        return "\n".join(lines)

    def squash_label(self, ops: list[dict[str, Any]], message: str | None = None) -> str:
        from raphael_ai.intelligence.summaries import semantic_squash_label

        preview = json.dumps(ops[:40], default=str)[:4000]
        started = time.monotonic()
        raw = self._runtime.complete(
            "Describe the engineer's intent in one sentence (max 120 chars). No markdown. "
            "Example: Finalize pump operating pressure and update manufacturer information.",
            f"Commit message: {message or ''}\nOperations:\n{preview}",
            task="squash",
        )
        logger.info("stage=squash tier=%s latency_ms=%.0f", self._runtime.last_tier, (time.monotonic() - started) * 1000)
        if raw:
            return raw.strip().strip('"')[:200]
        return semantic_squash_label(ops)

    def regenerate_memory(self, workspace_id: str, facts: dict[str, Any]) -> dict[str, Any] | None:
        raw = self._runtime.complete(
            "Generate workspace memory JSON only. Keys: goal, milestone, risks (array), "
            "recent_accomplishments (array), conventions_observed (array).",
            json.dumps({"workspace_id": workspace_id, "facts": facts}, default=str)[:5000],
            json_mode=True,
            task="summarize",
        )
        if raw:
            data = self._runtime.parse_json(raw)
            if isinstance(data, dict):
                return data
        return None

    def draft_workflow(self, description: str) -> dict[str, Any] | None:
        raw = self._runtime.complete(
            'Output JSON only: {"name":"...","trigger_type":"on_commit|pcb_updated|bom_updated|on_merge|scheduled",'
            '"action":"step1 → step2","conditions":["..."]}',
            description,
            json_mode=True,
            task="plan",
        )
        if raw:
            data = self._runtime.parse_json(raw)
            if isinstance(data, dict) and data.get("name"):
                data["draft"] = True
                return data
        return None

    def summarize_events(self, events: list[dict[str, Any]]) -> dict[str, str] | None:
        if not events:
            return None
        raw = self._runtime.complete(
            "Summarize design activity. Output JSON: {\"title\":\"...\",\"bullets\":[\"...\"]}",
            json.dumps(events[:30], default=str)[:5000],
            json_mode=True,
            task="summarize",
        )
        if raw:
            data = self._runtime.parse_json(raw)
            if isinstance(data, dict) and data.get("title"):
                return {"title": str(data["title"]), "bullets": [str(b) for b in data.get("bullets", [])[:8]]}
        return None
