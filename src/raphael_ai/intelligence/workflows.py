"""Workflow drafter and pattern observer."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WorkflowDrafter:
    """NL → automation definition draft (no execution)."""

    def draft(self, description: str) -> dict[str, Any]:
        desc = description.lower()
        trigger = "commit"
        if "pcb" in desc:
            trigger = "pcb_updated"
        elif "bom" in desc:
            trigger = "bom_updated"
        actions = []
        if "simulation" in desc or "sim" in desc:
            actions.append("run_simulation")
        if "notify" in desc or "manufacturing" in desc:
            actions.append("notify_manufacturing")
        if "review" in desc:
            actions.append("create_review_task")
        if "drc" in desc:
            actions.append("run_drc")
        if not actions:
            actions = ["notify_team"]
        return {
            "name": description[:80],
            "trigger_type": trigger,
            "action": " → ".join(actions),
            "conditions": ["critical_component_modified"] if "critical" in desc else [],
            "draft": True,
        }


class PatternObserver:
    """Detect repeated manual sequences and emit suggestions."""

    THRESHOLD = 3

    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or Path(os.environ.get("RAPHAEL_AI_DB", "/tmp/raphael-ai-memory.db"))
        self._db = path
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_counts (
                    pattern_key TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    description TEXT,
                    last_seen TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS suggestions (
                    id TEXT PRIMARY KEY,
                    kind TEXT,
                    message TEXT,
                    draft JSON,
                    created_at TEXT,
                    dismissed INTEGER DEFAULT 0
                )
                """
            )

    def record_sequence(self, actions: list[str], module_id: str) -> None:
        key = f"{module_id}:{'+'.join(sorted(actions))}"
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db) as conn:
            row = conn.execute("SELECT count FROM pattern_counts WHERE pattern_key = ?", (key,)).fetchone()
            count = (row[0] if row else 0) + 1
            desc = f"{' → '.join(actions)} on {module_id}"
            conn.execute(
                "INSERT OR REPLACE INTO pattern_counts (pattern_key, count, description, last_seen) VALUES (?, ?, ?, ?)",
                (key, count, desc, now),
            )
            if count >= self.THRESHOLD:
                sid = f"sug_{key[:32]}"
                conn.execute(
                    "INSERT OR IGNORE INTO suggestions (id, kind, message, draft, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        sid,
                        "automation",
                        f"You've manually {desc} {count} times. Would you like to automate this?",
                        json.dumps({"trigger_type": "commit", "action": " → ".join(actions)}),
                        now,
                    ),
                )

    def list_suggestions(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT id, kind, message, draft, created_at FROM suggestions WHERE dismissed = 0 ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        return [
            {"id": r[0], "kind": r[1], "message": r[2], "draft": json.loads(r[3] or "{}"), "created_at": r[4]}
            for r in rows
        ]

    def observe_timeline(self, events: list[dict[str, Any]]) -> None:
        """Group events by module and record action types."""
        by_module: dict[str, list[str]] = {}
        for ev in events:
            mid = ev.get("project_id") or ev.get("module_id") or "default"
            by_module.setdefault(mid, []).append(ev.get("event_type") or "edit")
        for mid, types in by_module.items():
            counts = Counter(types)
            top = [t for t, _ in counts.most_common(4)]
            if top:
                self.record_sequence(top, mid)
