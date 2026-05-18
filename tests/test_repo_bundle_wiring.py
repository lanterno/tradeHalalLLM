"""Wave D wiring tests — Repository.bundle + RepoBundle entrypoint.

The per-table mini-repos and ``RepoBundle.from_engine`` were shipped
in round-4/5. This commit added ``Repository.bundle`` so legacy callers
can migrate one site at a time without restructuring downstream
consumers. These tests pin:

* The ``Repository.bundle`` accessor returns the *same* mini-repo
  instances the legacy delegators forward to.
* The bundle's per-table fields satisfy their Protocol shapes.
* ``RepoBundle.from_engine`` constructs an independent bundle (no
  Repository required).
"""

from __future__ import annotations

import pytest

from halal_trader.db.repos import RepoBundle


class _FakeEngine:
    """Stand-in for AsyncEngine — the Repo Impls just stash it."""


# ── Repository.bundle accessor ───────────────────────────────────


def test_repository_bundle_returns_same_impls_as_delegators() -> None:
    """The legacy ``Repository`` already constructs the mini-repos in
    ``__init__``. The new ``.bundle`` property must expose the same
    instances, not fresh copies — otherwise mutable state (caches,
    counters) would diverge between paths."""
    from halal_trader.db.repository import Repository

    repo = Repository(_FakeEngine())  # type: ignore[arg-type]
    bundle = repo.bundle
    assert bundle.trades is repo._trades
    assert bundle.crypto_trades is repo._crypto_trades
    assert bundle.pnl is repo._pnl
    assert bundle.stock_pnl is repo._stock_pnl
    assert bundle.halal_cache is repo._halal_cache
    assert bundle.stock_halal_cache is repo._stock_halal_cache
    assert bundle.halal_screening is repo._halal_screening
    assert bundle.runtime_config is repo._runtime_config
    assert bundle.research_jobs is repo._research_jobs
    assert bundle.web_audit is repo._web_audit
    assert bundle.indicator_snapshots is repo._indicator_snapshots
    assert bundle.llm_decisions is repo._llm_decisions
    assert bundle.purification is repo._purification
    assert bundle.pair_pauses is repo._pair_pause
    assert bundle.strategy_adjustments is repo._strategy_adjustments


def test_repository_bundle_is_frozen() -> None:
    """RepoBundle is frozen so callers can't accidentally swap a
    repo out from under a long-lived consumer."""
    from halal_trader.db.repository import Repository

    repo = Repository(_FakeEngine())  # type: ignore[arg-type]
    bundle = repo.bundle
    with pytest.raises(Exception):  # dataclass frozen → FrozenInstanceError
        bundle.trades = None  # type: ignore[misc]


# ── RepoBundle.from_engine independent path ─────────────────────


def test_from_engine_builds_full_bundle_without_repository() -> None:
    """``RepoBundle.from_engine(engine)`` is the canonical entrypoint
    for new code that wants no dependency on the legacy ``Repository``
    facade. Every field must resolve to a non-None impl."""
    bundle = RepoBundle.from_engine(_FakeEngine())  # type: ignore[arg-type]
    for field_name in (
        "trades",
        "crypto_trades",
        "pnl",
        "stock_pnl",
        "halal_cache",
        "stock_halal_cache",
        "halal_screening",
        "runtime_config",
        "research_jobs",
        "web_audit",
        "indicator_snapshots",
        "llm_decisions",
        "purification",
        "pair_pauses",
        "strategy_adjustments",
    ):
        assert getattr(bundle, field_name) is not None, (
            f"RepoBundle.from_engine produced None for {field_name!r}"
        )


def test_from_engine_uses_distinct_instances_per_call() -> None:
    """Two calls to from_engine produce independent bundles — confirms
    callers that want isolated state get it, and the bundle isn't
    accidentally a cached singleton."""
    a = RepoBundle.from_engine(_FakeEngine())  # type: ignore[arg-type]
    b = RepoBundle.from_engine(_FakeEngine())  # type: ignore[arg-type]
    assert a.trades is not b.trades
    assert a.crypto_trades is not b.crypto_trades


# ── Protocol shape sanity ───────────────────────────────────────


def test_crypto_trade_repo_satisfies_protocol_shape() -> None:
    """One of the broader Protocol surfaces — ensures the mini-repo
    implements every method downstream code expects to call. Pure
    structural check; Protocols aren't ``@runtime_checkable`` here
    (mypy enforces the contract at type-check time)."""
    bundle = RepoBundle.from_engine(_FakeEngine())  # type: ignore[arg-type]
    repo = bundle.crypto_trades
    for method in (
        "record_crypto_trade",
        "update_crypto_trade_stop_loss",
        "close_crypto_trade",
        "get_today_crypto_trades",
        "get_open_crypto_trades",
        "get_open_crypto_trades_for_pair",
        "close_open_crypto_trades_for_pair",
        "get_recent_crypto_trades",
        "get_filled_trades",
    ):
        assert hasattr(repo, method), f"CryptoTradeRepo impl missing {method!r}"


# ── Legacy Repository delegation still works ───────────────────


@pytest.mark.asyncio
async def test_legacy_repository_delegators_route_through_bundle(monkeypatch) -> None:
    """Existing callers that still use ``Repository.X`` must keep
    working after Wave D — the delegators must still hit the same
    mini-repo the bundle exposes."""
    from halal_trader.db.repository import Repository

    repo = Repository(_FakeEngine())  # type: ignore[arg-type]
    bundle = repo.bundle

    called: dict[str, list] = {"args": []}

    async def fake_get_recent_trades(self, limit: int = 50) -> list:
        called["args"].append(limit)
        return [{"sentinel": True}]

    monkeypatch.setattr(type(bundle.trades), "get_recent_trades", fake_get_recent_trades)
    out = await repo.get_recent_trades(limit=7)
    assert out == [{"sentinel": True}]
    assert called["args"] == [7]
