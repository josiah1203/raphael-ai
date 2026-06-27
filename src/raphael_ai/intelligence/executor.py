"""Deterministic query executor — runs Planner IR against Raphael services."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from raphael_ai.intelligence.planner import FilterClause, PlannerIR


def _url(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default).rstrip("/")


def _nested_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _match_filter(value: Any, clause: FilterClause) -> bool:
    text = "" if value is None else str(value)
    if clause.op == "contains":
        return clause.value.lower() in text.lower()
    if clause.op == "eq":
        return text.lower() == clause.value.lower()
    if clause.op == "gt":
        try:
            return float(text) > float(clause.value)
        except ValueError:
            return False
    if clause.op == "lt":
        try:
            return float(text) < float(clause.value)
        except ValueError:
            return False
    return False


def _event_matches(ev: dict[str, Any], plan: PlannerIR) -> bool:
    if not plan.filters:
        return True
    payload = ev.get("payload") or {}
    merged = {**ev, **payload, "payload": payload}
    for clause in plan.filters:
        val = _nested_get(merged, clause.field)
        if val is None and "." in clause.field:
            val = _nested_get(payload, clause.field)
        if _match_filter(val, clause):
            return True
    return False


def _in_time_range(ev: dict[str, Any], time_range: dict[str, str] | None) -> bool:
    if not time_range:
        return True
    ts = ev.get("timestamp_utc") or ev.get("created_at") or ""
    if not ts:
        return True
    since = time_range.get("since") or time_range.get("start")
    until = time_range.get("until") or time_range.get("end")
    if since and ts < since:
        return False
    if until and ts > until:
        return False
    return True


class QueryExecutor:
    def __init__(self) -> None:
        self.audit = _url("RAPHAEL_AUDIT_URL", "http://127.0.0.1:8093")
        self.graph = _url("RAPHAEL_GRAPH_URL", "http://127.0.0.1:8100")
        self.workspaces = _url("RAPHAEL_WORKSPACES_URL", "http://127.0.0.1:8083")
        self.reviews = _url("RAPHAEL_REVIEWS_URL", "http://127.0.0.1:8087")
        self.artifacts = _url("RAPHAEL_ARTIFACTS_URL", "http://127.0.0.1:8107")

    def execute(
        self,
        plan: PlannerIR,
        workspace_id: str = "default",
        project_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        errors: list[str] = []
        results: list[dict[str, Any]] = []
        tasks: dict[str, Any] = {}

        with httpx.Client(timeout=15.0) as client:
            if "events" in plan.sources:
                tasks["events"] = lambda c=client: self._events(c, plan, project_id)
            if "modules" in plan.sources:
                tasks["modules"] = lambda c=client: self._modules(c, workspace_id, project_id)
            if "graph" in plan.sources:
                tasks["graph"] = lambda c=client: self._graph(c, plan)
            if "reviews" in plan.sources:
                tasks["reviews"] = lambda c=client: self._reviews(c, project_id)
            if "artifacts" in plan.sources:
                tasks["artifacts"] = lambda c=client: self._artifacts(c, project_id)

            with ThreadPoolExecutor(max_workers=min(5, len(tasks) or 1)) as pool:
                futures = {pool.submit(fn): name for name, fn in tasks.items()}
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        chunk, err = fut.result()
                        results.extend(chunk)
                        if err:
                            errors.append(err)
                    except Exception as exc:
                        errors.append(f"{name}: {exc}")

        return results[: plan.limit], errors

    def _events(
        self, client: httpx.Client, plan: PlannerIR, project_id: str | None
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            params: dict[str, Any] = {"limit": plan.limit}
            if project_id:
                params["project_id"] = project_id
            res = client.get(f"{self.audit}/v1/audit/timeline", params=params)
            if res.status_code != 200:
                return [], f"events: HTTP {res.status_code}"
            events = res.json().get("events", [])
            out = []
            for ev in events:
                if not _in_time_range(ev, plan.time_range):
                    continue
                if _event_matches(ev, plan):
                    out.append(
                        {
                            "source": "event",
                            "id": ev.get("event_id"),
                            "summary": ev.get("event_type"),
                            "data": ev,
                        }
                    )
            return out, None
        except httpx.HTTPError as exc:
            return [], f"events: {exc}"

    def _modules(
        self, client: httpx.Client, workspace_id: str, project_id: str | None
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            res = client.get(f"{self.workspaces}/v1/workspaces/{workspace_id}/modules")
            if res.status_code != 200:
                return [], f"modules: HTTP {res.status_code}"
            modules = res.json().get("modules", [])
            if project_id:
                modules = [m for m in modules if m.get("id") == project_id or m.get("project_id") == project_id]
            return [
                {"source": "module", "id": m["id"], "summary": m.get("name"), "data": m} for m in modules
            ], None
        except httpx.HTTPError as exc:
            return [], f"modules: {exc}"

    def _graph(self, client: httpx.Client, plan: PlannerIR) -> tuple[list[dict[str, Any]], str | None]:
        out: list[dict[str, Any]] = []
        try:
            if plan.graph_hops:
                for hop in plan.graph_hops:
                    params = {"depth": hop.depth}
                    res = client.get(f"{self.graph}/v1/graph/impact/{hop.from_id}", params=params)
                    if res.status_code != 200:
                        return out, f"graph impact {hop.from_id}: HTTP {res.status_code}"
                    body = res.json()
                    if hop.edge:
                        edges_res = client.get(f"{self.graph}/v1/graph/edges", params={"from_id": hop.from_id})
                        if edges_res.status_code == 200:
                            filtered = [
                                e
                                for e in edges_res.json().get("edges", [])
                                if e.get("type") == hop.edge or e.get("edge_type") == hop.edge
                            ]
                            body = {**body, "edges": filtered}
                    out.append({"source": "graph", "id": hop.from_id, "summary": "impact_set", "data": body})
            else:
                res = client.get(f"{self.graph}/v1/graph/edges")
                if res.status_code != 200:
                    return [], f"graph: HTTP {res.status_code}"
                for edge in res.json().get("edges", [])[: plan.limit]:
                    out.append({"source": "graph", "id": edge.get("id"), "summary": edge.get("type"), "data": edge})
            return out, None
        except httpx.HTTPError as exc:
            return out, f"graph: {exc}"

    def _reviews(self, client: httpx.Client, project_id: str | None) -> tuple[list[dict[str, Any]], str | None]:
        try:
            res = client.get(f"{self.reviews}/v1/reviews")
            if res.status_code != 200:
                return [], f"reviews: HTTP {res.status_code}"
            reviews = res.json().get("reviews", [])
            if project_id:
                reviews = [
                    r
                    for r in reviews
                    if r.get("module_id") == project_id or r.get("repo_id") == project_id
                ]
            return [
                {"source": "review", "id": r["id"], "summary": r.get("title"), "data": r} for r in reviews
            ], None
        except httpx.HTTPError as exc:
            return [], f"reviews: {exc}"

    def _artifacts(self, client: httpx.Client, project_id: str | None) -> tuple[list[dict[str, Any]], str | None]:
        try:
            res = client.get(f"{self.artifacts}/v1/artifacts")
            if res.status_code != 200:
                return [], f"artifacts: HTTP {res.status_code}"
            artifacts = res.json().get("artifacts", [])
            if project_id:
                artifacts = [a for a in artifacts if a.get("module_id") == project_id or a.get("project_id") == project_id]
            return [
                {"source": "artifact", "id": a["id"], "summary": a.get("kind"), "data": a} for a in artifacts
            ], None
        except httpx.HTTPError as exc:
            return [], f"artifacts: {exc}"
