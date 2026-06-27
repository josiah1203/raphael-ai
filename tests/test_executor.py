"""QueryExecutor unit tests with mocked httpx."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from raphael_ai.intelligence.executor import QueryExecutor, _event_matches, _nested_get
from raphael_ai.intelligence.planner import FilterClause, GraphHop, PlannerIR


def _mock_response(status: int, json_body: dict) -> MagicMock:
    res = MagicMock()
    res.status_code = status
    res.json.return_value = json_body
    return res


def test_nested_get() -> None:
    obj = {"payload": {"parameter": {"name": "pressure_max"}}}
    assert _nested_get(obj, "payload.parameter.name") == "pressure_max"


def test_event_matches_nested_filter() -> None:
    ev = {"event_type": "parameter.update", "payload": {"parameter": {"name": "operating_pressure"}}}
    plan = PlannerIR(
        filters=[FilterClause(field="payload.parameter.name", op="contains", value="pressure")],
        sources=["events"],
    )
    assert _event_matches(ev, plan)


@patch.object(httpx.Client, "get")
def test_executor_graph_depth_and_edge(mock_get: MagicMock) -> None:
    mock_get.side_effect = [
        _mock_response(200, {"node_id": "conn-a", "impact_set": ["mod-1"]}),
        _mock_response(200, {"edges": [{"id": "e1", "type": "bom_contains", "from_id": "conn-a"}]}),
    ]
    plan = PlannerIR(
        graph_hops=[GraphHop(**{"from": "conn-a", "edge": "bom_contains", "depth": 3})],
        sources=["graph"],
    )
    ex = QueryExecutor()
    with httpx.Client() as client:
        results, errors = ex._graph(client, plan)
    assert not errors
    assert results[0]["id"] == "conn-a"
    assert mock_get.call_args_list[0][0][0].endswith("/impact/conn-a")
    assert mock_get.call_args_list[0][1]["params"] == {"depth": 3}


@patch.object(httpx.Client, "get")
def test_executor_time_range_and_project(mock_get: MagicMock) -> None:
    mock_get.return_value = _mock_response(
        200,
        {
            "events": [
                {"event_id": "e1", "event_type": "edit", "timestamp_utc": "2026-06-01T00:00:00Z", "payload": {}},
                {"event_id": "e2", "event_type": "edit", "timestamp_utc": "2026-06-15T00:00:00Z", "payload": {}},
            ]
        },
    )
    plan = PlannerIR(
        sources=["events"],
        time_range={"since": "2026-06-10T00:00:00Z"},
    )
    ex = QueryExecutor()
    with httpx.Client() as client:
        results, err = ex._events(client, plan, project_id="pump-v3")
    assert err is None
    assert len(results) == 1
    assert results[0]["id"] == "e2"
    assert mock_get.call_args[1]["params"]["project_id"] == "pump-v3"


@patch.object(httpx.Client, "get")
def test_executor_returns_errors_on_failure(mock_get: MagicMock) -> None:
    mock_get.side_effect = httpx.ConnectError("refused")
    plan = PlannerIR(sources=["events", "modules"])
    ex = QueryExecutor()
    results, errors = ex.execute(plan, "default")
    assert results == []
    assert len(errors) == 2
    assert any("events" in e for e in errors)


@patch.object(httpx.Client, "get")
def test_executor_concurrent_sources(mock_get: MagicMock) -> None:
    def route(url: str, **kwargs):
        if "modules" in url:
            return _mock_response(200, {"modules": [{"id": "m1", "name": "Pump"}]})
        if "reviews" in url:
            return _mock_response(200, {"reviews": [{"id": "r1", "title": "Review"}]})
        return _mock_response(404, {})

    mock_get.side_effect = route
    plan = PlannerIR(sources=["modules", "reviews"], limit=10)
    ex = QueryExecutor()
    results, errors = ex.execute(plan, "default")
    assert len(results) == 2
    assert not errors
