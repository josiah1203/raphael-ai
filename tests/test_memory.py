"""Memory store tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from raphael_ai.intelligence.memory import WorkspaceMemoryStore, fetch_audit_cursor


def test_memory_uses_audit_cursor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "mem.db"
        store = WorkspaceMemoryStore(db_path=db)
        with patch("raphael_ai.intelligence.memory.fetch_audit_cursor", return_value="2026-06-01T12:00:00Z"):
            mem = store.regenerate("default", None)
        assert mem["source_event_cursor"] == "2026-06-01T12:00:00Z"
        assert mem["source_event_cursor"] != "evt_latest"


def test_fetch_audit_cursor_from_timeline() -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def get(self, url: str, params: dict | None = None):
            res = MagicMock()
            res.status_code = 200
            res.json.return_value = {
                "events": [{"event_id": "e1", "timestamp_utc": "2026-06-02T00:00:00Z"}],
                "next_cursor": "2026-06-02T00:00:00Z",
            }
            return res

    with patch("raphael_ai.intelligence.memory.httpx.Client", FakeClient):
        cursor = fetch_audit_cursor("proj-1")
    assert cursor == "2026-06-02T00:00:00Z"


def test_add_convention_endpoint_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "mem.db"
        store = WorkspaceMemoryStore(db_path=db)
        with patch("raphael_ai.intelligence.memory.fetch_audit_cursor", return_value="c1"):
            store.regenerate("default")
        mem = store.add_convention("default", "Always run DRC before merge")
        assert "Always run DRC before merge" in mem["conventions_observed"]
