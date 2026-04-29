"""Tests for the versioned ml_artefacts blob store."""

from __future__ import annotations

import pytest

from halal_trader.db.ml_artefacts import (
    list_versions,
    load_artefact,
    pickle_dumps,
    save_artefact,
)
from halal_trader.db.repository import Repository


async def test_save_and_load_json_artefact(engine) -> None:
    repo = Repository(engine)
    art_id = await save_artefact(
        repo=repo,
        name="slippage_v1",
        payload_json={"intercept": 0.0005, "coefs": {"size_usd": 1e-7}},
    )
    assert art_id > 0
    loaded = await load_artefact(repo=repo, name="slippage_v1")
    assert loaded is not None
    assert loaded["intercept"] == 0.0005


async def test_save_pickle_artefact_round_trip(engine) -> None:
    repo = Repository(engine)
    payload = {"any": "object", "nested": [1, 2, 3]}
    blob = pickle_dumps(payload)
    await save_artefact(repo=repo, name="anomaly_detector", payload_bytes=blob)
    loaded = await load_artefact(repo=repo, name="anomaly_detector")
    assert loaded is not None
    assert loaded["_pickle"] == payload


async def test_load_returns_latest_version(engine) -> None:
    repo = Repository(engine)
    await save_artefact(repo=repo, name="cal", payload_json={"v": 1})
    await save_artefact(repo=repo, name="cal", payload_json={"v": 2})
    await save_artefact(repo=repo, name="cal", payload_json={"v": 3})
    loaded = await load_artefact(repo=repo, name="cal")
    assert loaded == {"v": 3}


async def test_list_versions_orders_newest_first(engine) -> None:
    repo = Repository(engine)
    await save_artefact(repo=repo, name="x", payload_json={"a": 1})
    await save_artefact(repo=repo, name="x", payload_json={"a": 2})
    rows = await list_versions(repo=repo, name="x")
    assert len(rows) == 2
    assert rows[0]["version"] > rows[1]["version"]


async def test_save_rejects_dual_payloads(engine) -> None:
    repo = Repository(engine)
    with pytest.raises(ValueError):
        await save_artefact(
            repo=repo,
            name="bad",
            payload_json={"x": 1},
            payload_bytes=b"\x00",
        )


async def test_slippage_model_persists_via_db(engine) -> None:
    """Round-trip the SlippageModel through save_to_db / load_from_db."""
    from halal_trader.ml.slippage import SlippageModel, load_from_db, save_to_db

    repo = Repository(engine)
    model = SlippageModel(
        coefs={"size_usd": 1e-6, "spread_bps": 0.0001},
        intercept=0.0007,
        n_samples=100,
        feature_means={"size_usd": 500.0, "spread_bps": 5.0},
    )
    await save_to_db(model, repo)
    back = await load_from_db(repo)
    assert back.intercept == 0.0007
    assert back.n_samples == 100
