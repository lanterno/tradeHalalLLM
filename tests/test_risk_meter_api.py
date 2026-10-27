"""GET/PUT /api/system/risk-meter: the dashboard's view of the risk meter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from halal_trader.core import risk_meter
from halal_trader.web import app as web_app

CONFIRM = {"X-Trader-Confirm": "true"}


@pytest.fixture
def announced(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    sent: list[Any] = []

    async def record(settings: Any, status: Any) -> bool:
        sent.append(status)
        return True

    monkeypatch.setattr(risk_meter, "announce", record)
    return sent


@pytest.fixture
def client(database_url, tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "meter-test-token")
    with TestClient(web_app.create_app()) as c:
        c.headers["X-Trader-Token"] = "meter-test-token"
        yield c


def test_an_unset_meter_shows_level_3_and_what_each_level_sets(client) -> None:
    body = client.get("/api/system/risk-meter").json()

    assert (body["level"], body["name"], body["effective_from"]) == (3, "Standard", None)
    assert body["limits"]["max_position_pct"] == 0.20
    levels = {row["level"]: row for row in body["levels"]}
    assert sorted(levels) == [1, 2, 3, 4, 5]
    assert levels[3]["refusal"] is None
    assert "unlock test" in levels[5]["refusal"]


def test_setting_a_level_needs_the_confirmation_header(client, announced) -> None:
    r = client.put("/api/system/risk-meter", json={"level": 3})

    assert r.status_code == 412
    assert announced == []


def test_a_confirmed_change_is_stored_announced_and_audited(client, announced) -> None:
    r = client.put(
        "/api/system/risk-meter", json={"level": 3, "reason": "hold for HAL-1"}, headers=CONFIRM
    )

    assert r.status_code == 200
    assert (r.json()["set_by"], r.json()["reason"]) == ("dashboard", "hold for HAL-1")
    assert [s.level for s in announced] == [3]
    assert client.get("/api/system/risk-meter").json()["reason"] == "hold for HAL-1"
    audited = client.get("/api/activity?limit=5").json()
    assert any(a["method"] == "PUT" and a["path"] == "/api/system/risk-meter" for a in audited)


def test_a_level_that_is_not_open_is_refused_with_the_reason(client, announced) -> None:
    r = client.put("/api/system/risk-meter", json={"level": 4}, headers=CONFIRM)

    assert r.status_code == 409
    assert "unlock test" in r.json()["detail"]
    assert announced == []
    assert client.get("/api/system/risk-meter").json()["set_by"] is None


def test_a_level_that_is_not_a_number_is_refused(client, announced) -> None:
    r = client.put("/api/system/risk-meter", json={"level": "high"}, headers=CONFIRM)

    assert r.status_code == 422
