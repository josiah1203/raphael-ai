"""Planner IR schema, model planner, and rule-based fallback."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from raphael_ai.intelligence.model_runtime import GemmaRuntime

PLAN_SYSTEM = (
    "You are Raphael Workspace Intelligence. Output ONLY valid JSON matching this schema: "
    '{"intent":"find_events|find_artifacts|find_modules|find_reviews|graph_impact|general",'
    '"filters":[{"field":"string","op":"contains|eq","value":"string"}],'
    '"graph_hops":[{"from":"node_id","edge":"used_by","depth":2}],'
    '"sources":["events","graph","artifacts","modules","reviews"],"limit":50}'
)


class FilterClause(BaseModel):
    field: str
    op: Literal["eq", "contains", "gt", "lt"] = "contains"
    value: str


class GraphHop(BaseModel):
    from_id: str = Field(alias="from")
    edge: str = "used_by"
    depth: int = 2

    model_config = {"populate_by_name": True}


class PlannerIR(BaseModel):
    intent: Literal[
        "find_artifacts",
        "find_events",
        "find_modules",
        "find_reviews",
        "graph_impact",
        "general",
    ] = "general"
    filters: list[FilterClause] = Field(default_factory=list)
    graph_hops: list[GraphHop] = Field(default_factory=list)
    time_range: dict[str, str] | None = None
    sources: list[str] = Field(default_factory=lambda: ["events", "graph", "artifacts"])
    limit: int = 50


def model_planner(question: str, workspace_id: str, runtime: GemmaRuntime) -> PlannerIR | None:
    """Plan via Gemma when available; returns None to trigger rule fallback."""
    raw = runtime.complete(
        PLAN_SYSTEM,
        f"Workspace: {workspace_id}\nQuestion: {question}",
        json_mode=True,
        task="plan",
    )
    if not raw:
        return None
    data = runtime.parse_json(raw)
    if not data:
        return None
    try:
        return PlannerIR.model_validate(data)
    except Exception:
        return None


def rule_planner(question: str, workspace_id: str = "default") -> tuple[PlannerIR, str]:
    """Rule-based fallback when the model is unavailable or returns invalid JSON."""
    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    q = question.lower()
    ir = PlannerIR(limit=50)

    if "pressure" in q or "parameter" in q or "requirement" in q:
        ir.intent = "find_events"
        ir.filters = [FilterClause(field="parameter.name", op="contains", value="pressure")]
        ir.sources = ["events"]
    elif "connector" in q or "bom" in q:
        ir.intent = "find_artifacts"
        m = re.search(r"connector[-\s]?(\w+)", q)
        node = m.group(1) if m else "connector"
        ir.graph_hops = [GraphHop(**{"from": node, "edge": "bom_contains", "depth": 2})]
        ir.sources = ["graph", "artifacts"]
    elif "approv" in q or "review" in q:
        ir.intent = "find_reviews"
        ir.sources = ["reviews"]
    elif "impact" in q or "affect" in q:
        ir.intent = "graph_impact"
        ir.sources = ["graph"]
    elif "project" in q or "module" in q:
        ir.intent = "find_modules"
        ir.sources = ["modules"]
    else:
        ir.intent = "general"
        ir.sources = ["events", "modules"]

    _ = workspace_id
    return ir, plan_id
