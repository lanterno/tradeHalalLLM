"""Tests for halal/audit_nft.py — Round-5 Wave 22.F."""

from __future__ import annotations

import hashlib
from datetime import date

import pytest

from halal_trader.halal.audit_nft import (
    AnchorStatus,
    AuditAnchor,
    AuditEvent,
    ChainId,
    compute_anchor_hash,
    hash_event,
    mark_confirmed,
    mark_failed,
    mark_submitted,
    merkle_root,
    prepare_anchor,
    render_anchor,
    verify_anchor,
    verify_event_inclusion,
)


def _event(
    event_id: str = "E1",
    event_kind: str = "trade_filled",
    timestamp_iso: str = "2026-05-11T10:00:00Z",
    payload_hash: str | None = None,
) -> AuditEvent:
    if payload_hash is None:
        payload_hash = hashlib.sha256(event_id.encode()).hexdigest()
    return AuditEvent(
        event_id=event_id,
        event_kind=event_kind,
        timestamp_iso=timestamp_iso,
        payload_hash=payload_hash,
    )


# --- AuditEvent validation ----------------------


def test_event_valid():
    e = _event()
    assert e.event_kind == "trade_filled"


def test_event_empty_id_rejected():
    with pytest.raises(ValueError):
        _event(event_id="")


def test_event_empty_kind_rejected():
    with pytest.raises(ValueError):
        _event(event_kind=" ")


def test_event_empty_timestamp_rejected():
    with pytest.raises(ValueError):
        _event(timestamp_iso=" ")


def test_event_wrong_hash_length_rejected():
    with pytest.raises(ValueError):
        _event(payload_hash="short")


def test_event_immutable():
    e = _event()
    with pytest.raises(AttributeError):
        e.event_kind = "x"  # type: ignore[misc]


# --- hash_event + merkle_root ----------------------


def test_hash_event_stable():
    e1 = _event()
    e2 = _event()
    assert hash_event(e1) == hash_event(e2)


def test_hash_event_changes_with_fields():
    e1 = _event(event_id="E1")
    e2 = _event(event_id="E2")
    assert hash_event(e1) != hash_event(e2)


def test_merkle_root_single_event():
    e = _event()
    root = merkle_root([e])
    assert len(root) == 64
    assert root == hash_event(e)


def test_merkle_root_two_events():
    e1 = _event(event_id="E1")
    e2 = _event(event_id="E2")
    root = merkle_root([e1, e2])
    expected = hashlib.sha256((hash_event(e1) + hash_event(e2)).encode()).hexdigest()
    assert root == expected


def test_merkle_root_odd_count_duplicates_last():
    e1 = _event(event_id="E1")
    e2 = _event(event_id="E2")
    e3 = _event(event_id="E3")
    root = merkle_root([e1, e2, e3])
    # Layer 1: h(E1+E2), h(E3+E3)
    # Layer 2: h(L1[0] + L1[1])
    layer1_a = hashlib.sha256((hash_event(e1) + hash_event(e2)).encode()).hexdigest()
    layer1_b = hashlib.sha256((hash_event(e3) + hash_event(e3)).encode()).hexdigest()
    expected = hashlib.sha256((layer1_a + layer1_b).encode()).hexdigest()
    assert root == expected


def test_merkle_root_stable_across_calls():
    events = [_event(event_id=f"E{i}") for i in range(5)]
    r1 = merkle_root(events)
    r2 = merkle_root(events)
    assert r1 == r2


def test_merkle_root_changes_with_one_event_tampered():
    events_a = [_event(event_id=f"E{i}") for i in range(5)]
    events_b = list(events_a)
    events_b[2] = _event(event_id="E2", event_kind="tampered")
    assert merkle_root(events_a) != merkle_root(events_b)


def test_merkle_root_empty_rejected():
    with pytest.raises(ValueError):
        merkle_root([])


# --- compute_anchor_hash --------------------------


def test_compute_anchor_hash_deterministic():
    root = "a" * 64
    h1 = compute_anchor_hash(root, "2026-Q2", "alice", ChainId.POLYGON)
    h2 = compute_anchor_hash(root, "2026-Q2", "alice", ChainId.POLYGON)
    assert h1 == h2


def test_compute_anchor_hash_changes_with_chain():
    root = "a" * 64
    h_poly = compute_anchor_hash(root, "2026-Q2", "alice", ChainId.POLYGON)
    h_arb = compute_anchor_hash(root, "2026-Q2", "alice", ChainId.ARBITRUM)
    assert h_poly != h_arb


def test_compute_anchor_hash_changes_with_owner():
    root = "a" * 64
    h_a = compute_anchor_hash(root, "2026-Q2", "alice", ChainId.POLYGON)
    h_b = compute_anchor_hash(root, "2026-Q2", "bob", ChainId.POLYGON)
    assert h_a != h_b


def test_compute_anchor_hash_changes_with_epoch():
    root = "a" * 64
    h_q2 = compute_anchor_hash(root, "2026-Q2", "alice", ChainId.POLYGON)
    h_q3 = compute_anchor_hash(root, "2026-Q3", "alice", ChainId.POLYGON)
    assert h_q2 != h_q3


# --- prepare_anchor + AuditAnchor validation ---------


def test_prepare_anchor_basic():
    events = [_event(event_id=f"E{i}") for i in range(5)]
    a = prepare_anchor(
        anchor_id="A1",
        owner_id="alice",
        epoch_id="2026-Q2",
        chain_id=ChainId.POLYGON,
        events=events,
        prepared_on=date(2026, 7, 1),
    )
    assert a.status is AnchorStatus.PREPARED
    assert a.n_events == 5
    assert verify_anchor(a)


def test_prepare_anchor_empty_events_rejected():
    with pytest.raises(ValueError):
        prepare_anchor(
            anchor_id="A1",
            owner_id="alice",
            epoch_id="2026-Q2",
            chain_id=ChainId.POLYGON,
            events=[],
            prepared_on=date(2026, 7, 1),
        )


def test_anchor_invalid_merkle_length_rejected():
    with pytest.raises(ValueError):
        AuditAnchor(
            anchor_id="A1",
            owner_id="alice",
            epoch_id="2026-Q2",
            chain_id=ChainId.POLYGON,
            merkle_root_hex="short",
            n_events=5,
            prepared_on=date(2026, 7, 1),
            anchor_hash="a" * 64,
        )


def test_anchor_zero_events_rejected():
    with pytest.raises(ValueError):
        AuditAnchor(
            anchor_id="A1",
            owner_id="alice",
            epoch_id="2026-Q2",
            chain_id=ChainId.POLYGON,
            merkle_root_hex="a" * 64,
            n_events=0,
            prepared_on=date(2026, 7, 1),
            anchor_hash="a" * 64,
        )


def test_anchor_submitted_without_tx_rejected():
    with pytest.raises(ValueError):
        AuditAnchor(
            anchor_id="A1",
            owner_id="alice",
            epoch_id="2026-Q2",
            chain_id=ChainId.POLYGON,
            merkle_root_hex="a" * 64,
            n_events=5,
            prepared_on=date(2026, 7, 1),
            anchor_hash="a" * 64,
            status=AnchorStatus.SUBMITTED,
        )


def test_anchor_confirmed_without_block_rejected():
    with pytest.raises(ValueError):
        AuditAnchor(
            anchor_id="A1",
            owner_id="alice",
            epoch_id="2026-Q2",
            chain_id=ChainId.POLYGON,
            merkle_root_hex="a" * 64,
            n_events=5,
            prepared_on=date(2026, 7, 1),
            anchor_hash="a" * 64,
            status=AnchorStatus.CONFIRMED,
            tx_hash="0xabc",
        )


def test_anchor_failed_without_reason_rejected():
    with pytest.raises(ValueError):
        AuditAnchor(
            anchor_id="A1",
            owner_id="alice",
            epoch_id="2026-Q2",
            chain_id=ChainId.POLYGON,
            merkle_root_hex="a" * 64,
            n_events=5,
            prepared_on=date(2026, 7, 1),
            anchor_hash="a" * 64,
            status=AnchorStatus.FAILED,
        )


# --- FSM transitions ------------------------------


def _prepared() -> AuditAnchor:
    events = [_event(event_id=f"E{i}") for i in range(5)]
    return prepare_anchor(
        anchor_id="A1",
        owner_id="alice",
        epoch_id="2026-Q2",
        chain_id=ChainId.POLYGON,
        events=events,
        prepared_on=date(2026, 7, 1),
    )


def test_mark_submitted_basic():
    a = _prepared()
    a2 = mark_submitted(a, tx_hash="0xabc1234")
    assert a2.status is AnchorStatus.SUBMITTED
    assert a2.tx_hash == "0xabc1234"


def test_mark_submitted_empty_tx_rejected():
    a = _prepared()
    with pytest.raises(ValueError):
        mark_submitted(a, tx_hash=" ")


def test_mark_submitted_long_tx_rejected():
    a = _prepared()
    with pytest.raises(ValueError):
        mark_submitted(a, tx_hash="0x" + "a" * 100)


def test_mark_submitted_non_prepared_rejected():
    a = mark_submitted(_prepared(), tx_hash="0xabc")
    with pytest.raises(ValueError):
        mark_submitted(a, tx_hash="0xabc")


def test_mark_confirmed_basic():
    a = mark_submitted(_prepared(), tx_hash="0xabc")
    a2 = mark_confirmed(a, block_number=12345)
    assert a2.status is AnchorStatus.CONFIRMED
    assert a2.confirmed_block == 12345


def test_mark_confirmed_zero_block_rejected():
    a = mark_submitted(_prepared(), tx_hash="0xabc")
    with pytest.raises(ValueError):
        mark_confirmed(a, block_number=0)


def test_mark_confirmed_non_submitted_rejected():
    a = _prepared()
    with pytest.raises(ValueError):
        mark_confirmed(a, block_number=12345)


def test_mark_failed_from_prepared():
    a = _prepared()
    a2 = mark_failed(a, reason="chain timeout")
    assert a2.status is AnchorStatus.FAILED


def test_mark_failed_from_submitted():
    a = mark_submitted(_prepared(), tx_hash="0xabc")
    a2 = mark_failed(a, reason="reorg")
    assert a2.status is AnchorStatus.FAILED


def test_mark_failed_from_confirmed_rejected():
    a = mark_confirmed(mark_submitted(_prepared(), tx_hash="0xabc"), block_number=1)
    with pytest.raises(ValueError):
        mark_failed(a, reason="too late")


def test_mark_failed_empty_reason_rejected():
    a = _prepared()
    with pytest.raises(ValueError):
        mark_failed(a, reason=" ")


def test_confirmed_is_terminal():
    a = mark_confirmed(mark_submitted(_prepared(), tx_hash="0xabc"), block_number=1)
    with pytest.raises(ValueError):
        mark_submitted(a, tx_hash="0xdef")


# --- verify_anchor ---------------------------


def test_verify_clean_anchor():
    a = _prepared()
    assert verify_anchor(a)


def test_verify_detects_tampered_root():
    a = _prepared()
    bad = AuditAnchor(
        anchor_id=a.anchor_id,
        owner_id=a.owner_id,
        epoch_id=a.epoch_id,
        chain_id=a.chain_id,
        merkle_root_hex="b" * 64,  # tampered
        n_events=a.n_events,
        prepared_on=a.prepared_on,
        anchor_hash=a.anchor_hash,
    )
    assert not verify_anchor(bad)


def test_verify_detects_tampered_owner():
    a = _prepared()
    bad = AuditAnchor(
        anchor_id=a.anchor_id,
        owner_id="mallory",  # tampered
        epoch_id=a.epoch_id,
        chain_id=a.chain_id,
        merkle_root_hex=a.merkle_root_hex,
        n_events=a.n_events,
        prepared_on=a.prepared_on,
        anchor_hash=a.anchor_hash,
    )
    assert not verify_anchor(bad)


# --- verify_event_inclusion ----------------


def test_verify_inclusion_clean():
    events = [_event(event_id=f"E{i}") for i in range(5)]
    anchor = prepare_anchor(
        anchor_id="A1",
        owner_id="alice",
        epoch_id="2026-Q2",
        chain_id=ChainId.POLYGON,
        events=events,
        prepared_on=date(2026, 7, 1),
    )
    assert verify_event_inclusion(events[2], events, anchor)


def test_verify_inclusion_event_not_in_set():
    events = [_event(event_id=f"E{i}") for i in range(5)]
    anchor = prepare_anchor(
        anchor_id="A1",
        owner_id="alice",
        epoch_id="2026-Q2",
        chain_id=ChainId.POLYGON,
        events=events,
        prepared_on=date(2026, 7, 1),
    )
    stranger = _event(event_id="OUTSIDE")
    assert not verify_event_inclusion(stranger, events, anchor)


def test_verify_inclusion_set_tampered():
    events = [_event(event_id=f"E{i}") for i in range(5)]
    anchor = prepare_anchor(
        anchor_id="A1",
        owner_id="alice",
        epoch_id="2026-Q2",
        chain_id=ChainId.POLYGON,
        events=events,
        prepared_on=date(2026, 7, 1),
    )
    tampered = list(events)
    tampered[2] = _event(event_id="E2", event_kind="injected")
    # event_id still present, but root won't match.
    assert not verify_event_inclusion(events[0], tampered, anchor)


# --- Render -------------------------------


def test_render_no_secret_leak():
    events = [_event(event_id=f"E{i}") for i in range(5)]
    anchor = prepare_anchor(
        anchor_id="A1",
        owner_id="alice-secret@example.com",
        epoch_id="2026-Q2",
        chain_id=ChainId.POLYGON,
        events=events,
        prepared_on=date(2026, 7, 1),
    )
    out = render_anchor(anchor)
    assert "alice-secret@example.com" not in out


def test_render_status_emoji():
    a = _prepared()
    out = render_anchor(a)
    assert "📝" in out


def test_render_confirmed_block():
    a = mark_confirmed(
        mark_submitted(_prepared(), tx_hash="0xabcdef0123456789"),
        block_number=12345,
    )
    out = render_anchor(a)
    assert "block 12345" in out


def test_render_failed_reason():
    a = mark_failed(_prepared(), reason="chain timeout")
    out = render_anchor(a)
    assert "Failure" in out
    assert "chain timeout" in out


def test_render_masks_long_hashes():
    a = _prepared()
    out = render_anchor(a)
    assert a.merkle_root_hex not in out  # full hash never leaked


def test_render_includes_chain_id():
    a = _prepared()
    out = render_anchor(a)
    assert "polygon" in out
