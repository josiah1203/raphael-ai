"""Health check tests."""

from fastapi.testclient import TestClient

from raphael_ai.app import app


def test_health() -> None:
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["service"] == "raphael-ai"
