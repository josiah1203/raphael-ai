"""Context pack assembler with token budget."""

from __future__ import annotations

from typing import Any

from raphael_ai.intelligence.memory import WorkspaceMemoryStore


class ContextAssembler:
    MAX_EVENTS = 20
    MAX_ARTIFACTS = 10
    MAX_GRAPH = 15

    def __init__(self, memory: WorkspaceMemoryStore) -> None:
        self._memory = memory

    def assemble(
        self,
        workspace_id: str,
        project_id: str | None,
        retrieved: list[dict[str, Any]],
        conversation: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        memory = self._memory.get(workspace_id, project_id)
        events = [r for r in retrieved if r.get("source") == "event"][: self.MAX_EVENTS]
        artifacts = [r for r in retrieved if r.get("source") == "artifact"][: self.MAX_ARTIFACTS]
        graph = [r for r in retrieved if r.get("source") == "graph"][: self.MAX_GRAPH]
        reviews = [r for r in retrieved if r.get("source") == "review"][:5]
        modules = [r for r in retrieved if r.get("source") == "module"][:10]
        return {
            "workspace_summary": memory,
            "retrieved_events": events,
            "artifacts": artifacts,
            "graph_context": graph,
            "reviews": reviews,
            "modules": modules,
            "conversation": (conversation or [])[-6:],
        }
