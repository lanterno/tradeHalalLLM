"""Merkle-tree-anchored audit log — Round-5 Wave 19.H.

For SOC 2 / ISO 27001 / regulator-audit packages, the bot needs a
**tamper-evident** audit log of operationally-significant events. The
Round-5 Wave 1.K ``halal/charity_receipts.py`` shipped a hash-chain
primitive for charity receipts; this module is the **general-purpose
Merkle log** that any subsystem can anchor evidence into.

Two variants:

1. **Hash chain** (linear, append-only). Cheap, simple, what
   ``charity_receipts`` already uses. Suitable for streams of
   strictly-ordered events.
2. **Merkle tree** (logarithmic inclusion proofs). Used when an
   auditor wants a *proof of inclusion* for a specific entry without
   replaying the entire chain. Standard binary Merkle tree over
   SHA-256.

Pinned semantics:

- **Closed-set anchoring algorithm** — SHA-256 only.
- **Inclusion proofs are deterministic.**
- **Empty-tree root is 32-zero-bytes hex** (pinned).
- **Append-only**: ``add_leaf`` returns a new tree; the original is
  unchanged.
- **No-secret-leak pin** on render output — entries' raw bytes are
  hashed; only digests + indices appear.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

EMPTY_ROOT = "0" * 64


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_hash(payload: bytes) -> str:
    """Hash of a leaf — domain-separated with a 0x00 prefix per RFC 6962."""
    return _h(b"\x00" + payload)


def node_hash(left: str, right: str) -> str:
    """Hash of an internal node — domain-separated with a 0x01 prefix per RFC 6962."""
    return _h(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


@dataclass(frozen=True)
class MerkleTree:
    """Append-only binary Merkle tree."""

    leaves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for h in self.leaves:
            if len(h) != 64:
                raise ValueError("leaf hashes must be 64-hex-char SHA-256 digests")

    def root(self) -> str:
        if not self.leaves:
            return EMPTY_ROOT
        layer = list(self.leaves)
        while len(layer) > 1:
            next_layer: list[str] = []
            for i in range(0, len(layer), 2):
                if i + 1 < len(layer):
                    next_layer.append(node_hash(layer[i], layer[i + 1]))
                else:
                    # Odd-trailing leaf duplicates itself (RFC 6962 — pad)
                    next_layer.append(node_hash(layer[i], layer[i]))
            layer = next_layer
        return layer[0]

    def add_leaf(self, payload: bytes) -> "MerkleTree":
        return MerkleTree(leaves=self.leaves + (leaf_hash(payload),))

    def __len__(self) -> int:
        return len(self.leaves)

    def proof(self, index: int) -> tuple[tuple[str, str], ...]:
        """Inclusion proof for the leaf at ``index``.

        Returns a tuple of ``(sibling_hash, position)`` pairs from leaf to
        root, where ``position`` is ``"L"`` if the sibling is on the left,
        ``"R"`` if on the right.
        """
        if not self.leaves:
            raise IndexError("empty tree")
        if not 0 <= index < len(self.leaves):
            raise IndexError(f"index {index} out of range [0, {len(self.leaves)})")
        layer = list(self.leaves)
        idx = index
        path: list[tuple[str, str]] = []
        while len(layer) > 1:
            next_layer: list[str] = []
            sibling: str
            position: str
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else layer[i]
                next_layer.append(node_hash(left, right))
                if i == idx or i + 1 == idx:
                    if idx == i:
                        sibling = right
                        position = "R"
                    else:
                        sibling = left
                        position = "L"
                    path.append((sibling, position))
            idx //= 2
            layer = next_layer
        return tuple(path)


def verify_inclusion(
    leaf_payload: bytes,
    proof: Sequence[tuple[str, str]],
    expected_root: str,
) -> bool:
    """Verify that ``leaf_payload`` is included in a tree with ``expected_root``."""
    digest = leaf_hash(leaf_payload)
    for sibling, position in proof:
        if position == "L":
            digest = node_hash(sibling, digest)
        elif position == "R":
            digest = node_hash(digest, sibling)
        else:
            raise ValueError(f"invalid position {position!r}")
    return digest == expected_root


# --- Hash-chain primitive (re-export of the audit log shape) -----------------


@dataclass(frozen=True)
class ChainEntry:
    """One entry of a hash-chain audit log."""

    payload: bytes
    prior_root: str
    current_root: str

    def __post_init__(self) -> None:
        if len(self.prior_root) != 64 or len(self.current_root) != 64:
            raise ValueError("roots must be 64-hex-char SHA-256 digests")


def chain_root(chain: tuple[ChainEntry, ...]) -> str:
    return chain[-1].current_root if chain else EMPTY_ROOT


def append_chain(chain: tuple[ChainEntry, ...], payload: bytes) -> tuple[ChainEntry, ...]:
    prior = chain_root(chain)
    digest = _h(bytes.fromhex(prior) + b"\n" + payload)
    return chain + (
        ChainEntry(payload=payload, prior_root=prior, current_root=digest),
    )


def verify_chain(chain: tuple[ChainEntry, ...]) -> bool:
    expected_prior = EMPTY_ROOT
    for entry in chain:
        if entry.prior_root != expected_prior:
            return False
        expected = _h(bytes.fromhex(entry.prior_root) + b"\n" + entry.payload)
        if expected != entry.current_root:
            return False
        expected_prior = entry.current_root
    return True


# --- Render -----------------------------------------------------------------


def render_tree_summary(tree: MerkleTree) -> str:
    """Short summary — never includes raw payloads."""
    return (
        f"Merkle tree: {len(tree)} leaves, "
        f"root={tree.root()[:16]}…"
        if tree.leaves
        else "Merkle tree: empty (root=0…0)"
    )


def render_chain_summary(chain: tuple[ChainEntry, ...]) -> str:
    if not chain:
        return "Hash chain: empty (root=0…0)"
    return f"Hash chain: {len(chain)} entries, root={chain_root(chain)[:16]}…"
