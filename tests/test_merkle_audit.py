"""Tests for core/merkle_audit.py — Round-5 Wave 19.H."""

from __future__ import annotations

import pytest

from halal_trader.core.merkle_audit import (
    EMPTY_ROOT,
    ChainEntry,
    MerkleTree,
    append_chain,
    chain_root,
    leaf_hash,
    node_hash,
    render_chain_summary,
    render_tree_summary,
    verify_chain,
    verify_inclusion,
)

# --- Validation -------------------------------------------------------------


def test_empty_root_is_zero():
    assert EMPTY_ROOT == "0" * 64


def test_leaf_hash_64_hex():
    h = leaf_hash(b"x")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_node_hash_distinct_from_leaf_hash():
    """Domain separation: node_hash(L, L) ≠ leaf_hash(L+L)."""
    L = leaf_hash(b"a")
    assert node_hash(L, L) != leaf_hash(bytes.fromhex(L) + bytes.fromhex(L))


def test_tree_invalid_leaf_hash_rejected():
    with pytest.raises(ValueError):
        MerkleTree(leaves=("abc",))  # too short


# --- Tree growth + roots ----------------------------------------------------


def test_empty_tree_root():
    assert MerkleTree().root() == EMPTY_ROOT


def test_single_leaf_root_equals_leaf_hash():
    tree = MerkleTree().add_leaf(b"hello")
    assert tree.root() == leaf_hash(b"hello")


def test_two_leaves_root_is_node_hash():
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b")
    expected = node_hash(leaf_hash(b"a"), leaf_hash(b"b"))
    assert tree.root() == expected


def test_three_leaves_pads_with_self():
    """RFC 6962: odd trailing leaf duplicates itself for the parent."""
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b").add_leaf(b"c")
    a = leaf_hash(b"a")
    b = leaf_hash(b"b")
    c = leaf_hash(b"c")
    ab = node_hash(a, b)
    cc = node_hash(c, c)
    expected = node_hash(ab, cc)
    assert tree.root() == expected


def test_add_leaf_returns_new_tree():
    a = MerkleTree()
    b = a.add_leaf(b"x")
    assert len(a) == 0
    assert len(b) == 1


def test_tree_immutable():
    t = MerkleTree(leaves=(leaf_hash(b"a"),))
    with pytest.raises(AttributeError):
        t.leaves = ()  # type: ignore[misc]


# --- Inclusion proofs -------------------------------------------------------


def test_proof_single_leaf_is_empty_path():
    tree = MerkleTree().add_leaf(b"a")
    proof = tree.proof(0)
    assert proof == ()


def test_proof_two_leaves_returns_one_step():
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b")
    proof = tree.proof(0)
    assert len(proof) == 1
    assert proof[0][1] == "R"  # sibling is on the right


def test_proof_two_leaves_index_one_left_sibling():
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b")
    proof = tree.proof(1)
    assert proof[0][1] == "L"


def test_proof_out_of_range_rejected():
    tree = MerkleTree().add_leaf(b"a")
    with pytest.raises(IndexError):
        tree.proof(1)


def test_proof_negative_rejected():
    tree = MerkleTree().add_leaf(b"a")
    with pytest.raises(IndexError):
        tree.proof(-1)


def test_proof_empty_tree_rejected():
    with pytest.raises(IndexError):
        MerkleTree().proof(0)


def test_verify_inclusion_valid_proof():
    tree = MerkleTree()
    payloads = [b"a", b"b", b"c", b"d", b"e"]
    for p in payloads:
        tree = tree.add_leaf(p)
    root = tree.root()
    for i, p in enumerate(payloads):
        proof = tree.proof(i)
        assert verify_inclusion(p, proof, root) is True


def test_verify_inclusion_wrong_payload_fails():
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b").add_leaf(b"c").add_leaf(b"d")
    proof = tree.proof(1)
    assert not verify_inclusion(b"X", proof, tree.root())


def test_verify_inclusion_wrong_root_fails():
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b")
    proof = tree.proof(0)
    assert not verify_inclusion(b"a", proof, "f" * 64)


def test_verify_inclusion_invalid_position_rejected():
    with pytest.raises(ValueError):
        verify_inclusion(b"a", [("0" * 64, "X")], "0" * 64)


# --- Hash chain -------------------------------------------------------------


def test_chain_empty_root():
    assert chain_root(()) == EMPTY_ROOT


def test_chain_append_advances_root():
    chain = append_chain((), b"event-1")
    assert chain_root(chain) != EMPTY_ROOT


def test_chain_two_appends_distinct_roots():
    chain = append_chain((), b"event-1")
    r1 = chain_root(chain)
    chain = append_chain(chain, b"event-2")
    r2 = chain_root(chain)
    assert r1 != r2


def test_chain_verify_clean_passes():
    chain = ()
    for i in range(5):
        chain = append_chain(chain, f"event-{i}".encode())
    assert verify_chain(chain)


def test_chain_verify_tampered_fails():
    chain = append_chain((), b"original")
    bad_entry = ChainEntry(
        payload=b"tampered",
        prior_root=chain[0].prior_root,
        current_root=chain[0].current_root,
    )
    assert not verify_chain((bad_entry,))


def test_chain_verify_broken_link_fails():
    chain = append_chain((), b"a")
    chain = append_chain(chain, b"b")
    bad_entry = ChainEntry(
        payload=chain[1].payload, prior_root="f" * 64, current_root=chain[1].current_root
    )
    assert not verify_chain((chain[0], bad_entry))


def test_chain_entry_short_root_rejected():
    with pytest.raises(ValueError):
        ChainEntry(payload=b"x", prior_root="abc", current_root="0" * 64)


# --- Render ----------------------------------------------------------------


def test_render_tree_empty():
    out = render_tree_summary(MerkleTree())
    assert "empty" in out


def test_render_tree_with_leaves():
    tree = MerkleTree().add_leaf(b"a").add_leaf(b"b")
    out = render_tree_summary(tree)
    assert "2 leaves" in out
    # Root is truncated
    full_root = tree.root()
    assert full_root not in out
    assert full_root[:16] in out


def test_render_chain_empty():
    out = render_chain_summary(())
    assert "empty" in out


def test_render_chain_with_entries():
    chain = append_chain((), b"event-1")
    chain = append_chain(chain, b"event-2")
    out = render_chain_summary(chain)
    assert "2 entries" in out


def test_render_no_payload_leak():
    """Render output never contains raw event bytes."""
    chain = append_chain((), b"SECRET-PAYLOAD-XYZ")
    out = render_chain_summary(chain)
    assert "SECRET" not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_audit_log_for_compliance():
    """Build a 100-event audit log + verify chain + extract one inclusion proof."""
    chain = ()
    tree = MerkleTree()
    for i in range(100):
        payload = f"compliance-event-{i:03d}".encode()
        chain = append_chain(chain, payload)
        tree = tree.add_leaf(payload)
    assert verify_chain(chain)
    # Pick entry 42 and prove inclusion
    payload = b"compliance-event-042"
    proof = tree.proof(42)
    assert verify_inclusion(payload, proof, tree.root())


def test_replay_consistency():
    a = MerkleTree().add_leaf(b"x")
    b = MerkleTree().add_leaf(b"x")
    assert a.root() == b.root()
