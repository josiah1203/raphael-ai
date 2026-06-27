"""Semantic summaries and squash labels."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SummaryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or Path(os.environ.get("RAPHAEL_AI_DB", "/tmp/raphael-ai-memory.db"))
        self._db = path
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_summaries (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT,
                    window_start TEXT,
                    window_end TEXT,
                    title TEXT,
                    bullets JSON,
                    event_count INTEGER,
                    created_at TEXT
                )
                """
            )

    def summarize_events(self, workspace_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Produce semantic summary from event window."""
        if not events:
            return {"title": "No activity", "bullets": [], "event_count": 0}
        types: dict[str, int] = {}
        for ev in events:
            t = ev.get("event_type") or ev.get("summary") or "unknown"
            types[t] = types.get(t, 0) + 1
        top = sorted(types.items(), key=lambda x: -x[1])[:5]
        bullets = [f"{t} ({n})" for t, n in top]
        title = f"{len(events)} events — {top[0][0]}" if top else f"{len(events)} events"
        sid = f"sum_{datetime.now(UTC).timestamp():.0f}"
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO event_summaries (id, workspace_id, title, bullets, event_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, workspace_id, title, json.dumps(bullets), len(events), datetime.now(UTC).isoformat()),
            )
        return {"id": sid, "title": title, "bullets": bullets, "event_count": len(events)}

    def store_summary(
        self, workspace_id: str, events: list[dict[str, Any]], model_summary: dict[str, str | list]
    ) -> dict[str, Any]:
        sid = f"sum_{datetime.now(UTC).timestamp():.0f}"
        title = str(model_summary.get("title", "Activity summary"))
        bullets = [str(b) for b in model_summary.get("bullets", [])[:8]]
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO event_summaries (id, workspace_id, title, bullets, event_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, workspace_id, title, json.dumps(bullets), len(events), datetime.now(UTC).isoformat()),
            )
        return {"id": sid, "title": title, "bullets": bullets, "event_count": len(events)}

    def list_summaries(self, workspace_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT id, title, bullets, event_count, created_at FROM event_summaries WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
        return [
            {"id": r[0], "title": r[1], "bullets": json.loads(r[2] or "[]"), "event_count": r[3], "created_at": r[4]}
            for r in rows
        ]


def semantic_squash_label(ops: list[dict[str, Any]]) -> str:
    """Rule-based intent label until model wired."""
    if not ops:
        return "Empty commit"
    types = {op.get("type") for op in ops if op.get("type")}
    if "parameter.update" in types:
        names = [op.get("name", "?") for op in ops if op.get("type") == "parameter.update"]
        return f"Update parameters: {', '.join(names[:3])}"
    if "component.move" in types:
        return "Reposition components"
    return f"Apply {len(ops)} design change(s)"
